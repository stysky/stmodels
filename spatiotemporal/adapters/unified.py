from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..core import SplitConfig, TrafficData, TrainingResult
from ..metrics import masked_mae, masked_mape, masked_rmse
from ..preprocessing import (
    FeatureScaler,
    StandardScaler,
    build_feature_tensor,
    make_forecasting_windows,
    save_adj_npz,
    split_indices,
)
from .base import BaseModelAdapter, coerce_history_frame, resolve_device


@dataclass
class ForecastingConfig:
    history_steps: int = 12
    horizon_steps: int = 12
    y_start: int = 1
    split: SplitConfig = SplitConfig(0.7, 0.1, 0.2)
    batch_size: int = 64
    device: str = "cuda"
    epochs: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    patience: int = 10
    min_delta: float = 0.0
    grad_clip: float = 5.0
    include_time_of_day: bool = True
    include_day_of_week: bool = True
    checkpoint_path: str | None = None


@dataclass
class ForecastingBundle:
    data: TrafficData
    feature_names: list[str]
    raw_splits: dict[str, tuple[np.ndarray, np.ndarray]]
    scaled_splits: dict[str, tuple[np.ndarray, np.ndarray]]
    loaders: dict[str, DataLoader]
    input_scaler: FeatureScaler
    target_scaler: StandardScaler
    adjacency: np.ndarray
    x_offsets: np.ndarray
    y_offsets: np.ndarray

    @property
    def input_dim(self) -> int:
        train_x, _ = self.scaled_splits["train"]
        return int(train_x.shape[-1])

    @property
    def target_dim(self) -> int:
        _, train_y = self.scaled_splits["train"]
        return int(train_y.shape[-1])


class UnifiedForecastAdapter(BaseModelAdapter):
    config_cls = ForecastingConfig

    def __init__(self, project_root: str | Path, config: ForecastingConfig | None = None) -> None:
        super().__init__(project_root)
        self.config = config or self.config_cls()

    def adjacency_tensor(self, bundle: ForecastingBundle, device: torch.device) -> torch.Tensor:
        return torch.tensor(bundle.adjacency, dtype=torch.float32, device=device)

    def _target_channels(self, data: TrafficData) -> tuple[int, ...]:
        return (data.target_index,)

    def _bundle_feature_tensor(self, data: TrafficData) -> tuple[np.ndarray, list[str]]:
        return build_feature_tensor(
            data.values,
            data.timestamps,
            include_time_of_day=self.config.include_time_of_day,
            include_day_of_week=self.config.include_day_of_week,
            base_feature_names=data.feature_names,
        )

    def _make_loader(self, split: tuple[np.ndarray, np.ndarray], shuffle: bool) -> DataLoader:
        x, y = split
        dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)

    def _resolve_adjacency(self, data: TrafficData) -> np.ndarray:
        if data.adjacency is None:
            return np.eye(data.num_nodes, dtype=np.float32)
        if hasattr(data.adjacency, "toarray"):
            return np.asarray(data.adjacency.toarray(), dtype=np.float32)
        return np.asarray(data.adjacency, dtype=np.float32)

    def prepare_training_bundle(self, data: TrafficData) -> ForecastingBundle:
        feature_tensor, feature_names = self._bundle_feature_tensor(data)
        train_slice, val_slice, test_slice = split_indices(data.num_timesteps, self.config.split)
        series_splits = {
            "train": feature_tensor[train_slice],
            "val": feature_tensor[val_slice],
            "test": feature_tensor[test_slice],
        }

        raw_splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        x_offsets = y_offsets = None
        for name, values in series_splits.items():
            try:
                x, y, x_offsets, y_offsets = make_forecasting_windows(
                    values,
                    history_steps=self.config.history_steps,
                    horizon_steps=self.config.horizon_steps,
                    target_indices=self._target_channels(data),
                    y_start=self.config.y_start,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Failed to create {name} windows for {data.name}: "
                    f"history_steps={self.config.history_steps}, horizon_steps={self.config.horizon_steps}"
                ) from exc
            raw_splits[name] = (x, y)

        target_channel = data.target_index
        train_target = series_splits["train"][..., target_channel]
        train_mean = float(train_target.mean())
        train_std = float(train_target.std())
        input_scaler = FeatureScaler(
            channels=(target_channel,),
            mean=train_mean,
            std=train_std,
        )
        target_scaler = StandardScaler(
            mean=train_mean,
            std=train_std,
        )

        scaled_splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        loaders: dict[str, DataLoader] = {}
        for name, (split_x, split_y) in raw_splits.items():
            scaled_x = input_scaler.transform(split_x).astype(np.float32)
            scaled_y = split_y.copy().astype(np.float32)
            scaled_y[..., 0] = target_scaler.transform(scaled_y[..., 0]).astype(np.float32)
            scaled_splits[name] = (scaled_x, scaled_y)
            loaders[name] = self._make_loader(scaled_splits[name], shuffle=(name == "train"))

        bundle = ForecastingBundle(
            data=data,
            feature_names=feature_names,
            raw_splits=raw_splits,
            scaled_splits=scaled_splits,
            loaders=loaders,
            input_scaler=input_scaler,
            target_scaler=target_scaler,
            adjacency=self._resolve_adjacency(data),
            x_offsets=x_offsets if x_offsets is not None else np.array([], dtype=np.int64),
            y_offsets=y_offsets if y_offsets is not None else np.array([], dtype=np.int64),
        )
        self.last_bundle = bundle
        return bundle

    def export_native_artifacts(self, bundle: ForecastingBundle, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for split_name, (x, y) in bundle.raw_splits.items():
            np.savez_compressed(
                output_dir / f"{split_name}.npz",
                x=x,
                y=y,
                x_offsets=bundle.x_offsets.reshape(-1, 1),
                y_offsets=bundle.y_offsets.reshape(-1, 1),
                feature_names=np.asarray(bundle.feature_names),
                sensor_ids=np.asarray(bundle.data.sensor_ids),
            )
        adj_path = save_adj_npz(output_dir / "adjacency.npz", bundle.adjacency)
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "dataset": bundle.data.name,
                    "model": self.model_name,
                    "feature_names": bundle.feature_names,
                    "target_feature": bundle.data.target_feature,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return {"data_dir": output_dir, "adjacency_path": adj_path, "metadata_path": metadata_path}

    def _build_model(self, bundle: ForecastingBundle, device: torch.device):
        raise NotImplementedError

    def _build_optimizer(self, model: torch.nn.Module):
        return torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _build_scheduler(self, optimizer):
        return None

    def _compute_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(prediction, target)

    def train(self, bundle: ForecastingBundle) -> TrainingResult:
        device = resolve_device(self.config.device)
        model = self._build_model(bundle, device)
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)

        checkpoint_path = Path(
            self.config.checkpoint_path
            or (self.project_root / "checkpoints" / f"{bundle.data.name}_{self.model_name}.pt")
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        history = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        best_epoch = 0
        patience_steps = 0

        for epoch in range(1, self.config.epochs + 1):
            model.train()
            train_losses = []
            for batch_x, batch_y in bundle.loaders["train"]:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                prediction = model(batch_x)
                loss = self._compute_loss(prediction, batch_y)
                loss.backward()
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                optimizer.step()
                train_losses.append(float(loss.item()))

            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_y in bundle.loaders["val"]:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    prediction = model(batch_x)
                    val_losses.append(float(self._compute_loss(prediction, batch_y).item()))

            if scheduler is not None:
                scheduler.step()

            mean_train = float(np.mean(train_losses))
            mean_val = float(np.mean(val_losses))
            history["train_loss"].append(mean_train)
            history["val_loss"].append(mean_val)
            print(
                f"[{self.model_name}][Epoch {epoch:03d}] train={mean_train:.4f} val={mean_val:.4f}",
                flush=True,
            )

            if mean_val < best_val - self.config.min_delta:
                best_val = mean_val
                best_epoch = epoch
                patience_steps = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                patience_steps += 1
                if self.config.patience > 0 and patience_steps >= self.config.patience:
                    break

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        predictions = []
        reals = []
        with torch.no_grad():
            for batch_x, batch_y in bundle.loaders["test"]:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                predictions.append(model(batch_x).cpu())
                reals.append(batch_y.cpu())

        pred_tensor = torch.cat(predictions, dim=0)
        real_tensor = torch.cat(reals, dim=0)
        pred_tensor[..., 0] = bundle.target_scaler.inverse_transform(pred_tensor[..., 0])
        real_tensor[..., 0] = bundle.target_scaler.inverse_transform(real_tensor[..., 0])

        mae = float(masked_mae(pred_tensor, real_tensor, 0.0).item())
        mape = float(masked_mape(pred_tensor, real_tensor, 0.0).item())
        rmse = float(masked_rmse(pred_tensor, real_tensor, 0.0).item())

        self.model = model
        result = TrainingResult(
            model_name=self.model_name,
            dataset_name=bundle.data.name,
            checkpoint_path=checkpoint_path,
            best_epoch=best_epoch,
            metrics={"mae": mae, "mape": mape, "rmse": rmse, "best_val_loss": best_val},
            history=history,
            extra={"history_steps": self.config.history_steps, "horizon_steps": self.config.horizon_steps},
        )
        self.last_result = result
        return result

    def load_checkpoint(self, bundle: ForecastingBundle, checkpoint_path: str | Path):
        device = resolve_device(self.config.device)
        model = self._build_model(bundle, device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model
        self.last_bundle = bundle
        return model

    def predict(self, history, bundle: ForecastingBundle | None = None):
        bundle = bundle or self.last_bundle
        if bundle is None or self.model is None:
            raise RuntimeError(f"{self.model_name} model has not been prepared and trained")

        frame = coerce_history_frame(
            history,
            sensor_ids=bundle.data.sensor_ids,
            freq=bundle.data.freq,
            fallback_timestamps=bundle.data.timestamps,
        )
        if len(frame) < self.config.history_steps:
            raise ValueError(f"Need at least {self.config.history_steps} history steps for prediction")

        frame = frame.iloc[-self.config.history_steps :]
        feature_tensor, _ = build_feature_tensor(
            frame.values[..., None],
            frame.index,
            include_time_of_day=self.config.include_time_of_day,
            include_day_of_week=self.config.include_day_of_week,
            base_feature_names=[bundle.data.target_feature],
        )
        feature_tensor = bundle.input_scaler.transform(feature_tensor[None, ...]).astype(np.float32)
        tensor = torch.from_numpy(feature_tensor).float().to(next(self.model.parameters()).device)
        with torch.no_grad():
            prediction = self.model(tensor).cpu()
        prediction[..., 0] = bundle.target_scaler.inverse_transform(prediction[..., 0])
        return prediction.squeeze(0).numpy()
