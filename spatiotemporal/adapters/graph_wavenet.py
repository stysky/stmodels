from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..core import SplitConfig, TrafficData, TrainingResult
from ..metrics import masked_mae, masked_mape, masked_rmse
from ..preprocessing import (
    StandardScaler,
    add_graph_wavenet_features,
    graph_wavenet_supports,
    make_graph_wavenet_windows,
    save_adj_pkl,
    split_indices,
)
from .base import BaseModelAdapter, coerce_history_frame, resolve_device


@dataclass
class GraphWaveNetConfig:
    history_steps: int = 12
    horizon_steps: int = 12
    y_start: int = 1
    add_time_in_day: bool = True
    add_day_in_week: bool = False
    split: SplitConfig = SplitConfig(0.7, 0.1, 0.2)
    batch_size: int = 64
    device: str = "cuda"
    learning_rate: float = 0.001
    dropout: float = 0.3
    weight_decay: float = 0.0001
    epochs: int = 10
    nhid: int = 32
    adjtype: str = "doubletransition"
    gcn_bool: bool = True
    addaptadj: bool = True
    aptonly: bool = False
    randomadj: bool = False
    print_every: int = 50
    patience: int = 10
    min_delta: float = 0.0
    checkpoint_path: str | None = None


@dataclass
class GraphWaveNetBundle:
    data: TrafficData
    raw_splits: dict[str, tuple[np.ndarray, np.ndarray]]
    scaled_splits: dict[str, tuple[np.ndarray, np.ndarray]]
    scaler: StandardScaler
    adjacency: np.ndarray
    x_offsets: np.ndarray
    y_offsets: np.ndarray


class GraphWaveNetAdapter(BaseModelAdapter):
    model_name = "graph-wavenet"

    def __init__(self, project_root: str | Path, config: GraphWaveNetConfig | None = None) -> None:
        super().__init__(project_root)
        self.config = config or GraphWaveNetConfig()
        self._gwnet_class = self._load_gwnet_class()

    def _load_gwnet_class(self):
        model_path = self.project_root / "models" / "Graph-WaveNet-master" / "model.py"
        module_name = "_spatiotemporal_graph_wavenet_model"
        spec = importlib.util.spec_from_file_location(module_name, model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load Graph WaveNet from {model_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.gwnet

    def prepare_training_bundle(self, data: TrafficData) -> GraphWaveNetBundle:
        frame = data.to_dataframe()
        train_slice, val_slice, test_slice = split_indices(len(frame), self.config.split)

        frame_splits = {
            "train": frame.iloc[train_slice],
            "val": frame.iloc[val_slice],
            "test": frame.iloc[test_slice],
        }

        raw_splits = {
            name: make_graph_wavenet_windows(
                split_frame,
                history_steps=self.config.history_steps,
                horizon_steps=self.config.horizon_steps,
                y_start=self.config.y_start,
                add_time_in_day=self.config.add_time_in_day,
                add_day_in_week=self.config.add_day_in_week,
            )[:2]
            for name, split_frame in frame_splits.items()
        }
        x_offsets, y_offsets = make_graph_wavenet_windows(
            frame_splits["train"],
            history_steps=self.config.history_steps,
            horizon_steps=self.config.horizon_steps,
            y_start=self.config.y_start,
            add_time_in_day=self.config.add_time_in_day,
            add_day_in_week=self.config.add_day_in_week,
        )[2:]
        scaler = StandardScaler(
            mean=raw_splits["train"][0][..., 0].mean(),
            std=raw_splits["train"][0][..., 0].std(),
        )

        scaled_splits = {}
        for name, (split_x, split_y) in raw_splits.items():
            scaled_x = split_x.copy()
            scaled_x[..., 0] = scaler.transform(scaled_x[..., 0])
            scaled_splits[name] = (scaled_x.astype(np.float32), split_y.astype(np.float32))

        if data.adjacency is None:
            adjacency = np.eye(data.num_nodes, dtype=np.float32)
        elif hasattr(data.adjacency, "toarray"):
            adjacency = np.asarray(data.adjacency.toarray(), dtype=np.float32)
        else:
            adjacency = np.asarray(data.adjacency, dtype=np.float32)

        bundle = GraphWaveNetBundle(
            data=data,
            raw_splits=raw_splits,
            scaled_splits=scaled_splits,
            scaler=scaler,
            adjacency=adjacency,
            x_offsets=x_offsets,
            y_offsets=y_offsets,
        )
        self.last_bundle = bundle
        return bundle

    def export_native_artifacts(self, bundle: GraphWaveNetBundle, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for split_name, (x, y) in bundle.raw_splits.items():
            np.savez_compressed(
                output_dir / f"{split_name}.npz",
                x=x,
                y=y,
                x_offsets=bundle.x_offsets.reshape(-1, 1),
                y_offsets=bundle.y_offsets.reshape(-1, 1),
            )
        adj_path = save_adj_pkl(output_dir / "adj.pkl", bundle.data.sensor_ids, bundle.adjacency)
        return {"data_dir": output_dir, "adj_path": adj_path}

    def _make_loader(self, split: tuple[np.ndarray, np.ndarray], shuffle: bool) -> DataLoader:
        x, y = split
        dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)

    def _build_model(self, bundle: GraphWaveNetBundle, device: torch.device):
        supports = [
            torch.tensor(arr, dtype=torch.float32, device=device)
            for arr in graph_wavenet_supports(bundle.adjacency, self.config.adjtype)
        ]
        adjinit = None if self.config.randomadj or not supports else supports[0]
        if self.config.aptonly:
            supports = None

        model = self._gwnet_class(
            device=device,
            num_nodes=bundle.data.num_nodes,
            dropout=self.config.dropout,
            supports=supports,
            gcn_bool=self.config.gcn_bool,
            addaptadj=self.config.addaptadj,
            aptinit=adjinit,
            in_dim=bundle.scaled_splits["train"][0].shape[-1],
            out_dim=self.config.horizon_steps,
            residual_channels=self.config.nhid,
            dilation_channels=self.config.nhid,
            skip_channels=self.config.nhid * 8,
            end_channels=self.config.nhid * 16,
        ).to(device)
        return model

    def train(self, bundle: GraphWaveNetBundle) -> TrainingResult:
        device = resolve_device(self.config.device)
        model = self._build_model(bundle, device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        train_loader = self._make_loader(bundle.scaled_splits["train"], shuffle=True)
        val_loader = self._make_loader(bundle.scaled_splits["val"], shuffle=False)
        test_loader = self._make_loader(bundle.scaled_splits["test"], shuffle=False)

        history = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        best_epoch = 0
        patience_steps = 0
        checkpoint_path = Path(
            self.config.checkpoint_path
            or (self.project_root / "checkpoints" / f"{bundle.data.name}_graph_wavenet.pt")
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.config.epochs + 1):
            model.train()
            train_losses = []
            for step, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                model_input = F.pad(batch_x.transpose(1, 3), (1, 0, 0, 0))
                output = model(model_input).transpose(1, 3)
                real = batch_y[..., 0].permute(0, 2, 1).unsqueeze(1)
                prediction = bundle.scaler.inverse_transform(output)
                loss = masked_mae(prediction, real, 0.0)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                train_losses.append(float(loss.item()))

                if step % self.config.print_every == 0:
                    print(
                        f"[GraphWaveNet][Epoch {epoch:03d}] Step {step:03d} Train MAE {train_losses[-1]:.4f}",
                        flush=True,
                    )

            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    model_input = F.pad(batch_x.transpose(1, 3), (1, 0, 0, 0))
                    output = model(model_input).transpose(1, 3)
                    real = batch_y[..., 0].permute(0, 2, 1).unsqueeze(1)
                    prediction = bundle.scaler.inverse_transform(output)
                    val_losses.append(float(masked_mae(prediction, real, 0.0).item()))

            mean_train = float(np.mean(train_losses))
            mean_val = float(np.mean(val_losses))
            history["train_loss"].append(mean_train)
            history["val_loss"].append(mean_val)
            print(
                f"[GraphWaveNet][Epoch {epoch:03d}] train={mean_train:.4f} val={mean_val:.4f}",
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
        outputs = []
        reals = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                preds = model(F.pad(batch_x.transpose(1, 3), (1, 0, 0, 0))).transpose(1, 3)
                outputs.append(preds.cpu())
                reals.append(batch_y[..., 0].permute(0, 2, 1).cpu())

        pred_tensor = torch.cat(outputs, dim=0)
        real_tensor = torch.cat(reals, dim=0)
        pred_tensor = bundle.scaler.inverse_transform(pred_tensor).squeeze(1)

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
            extra={"horizon_steps": self.config.horizon_steps},
        )
        self.last_result = result
        return result

    def load_checkpoint(self, bundle: GraphWaveNetBundle, checkpoint_path: str | Path):
        device = resolve_device(self.config.device)
        model = self._build_model(bundle, device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model
        self.last_bundle = bundle
        return model

    def predict(self, history, bundle: GraphWaveNetBundle | None = None):
        bundle = bundle or self.last_bundle
        if bundle is None or self.model is None:
            raise RuntimeError("Graph WaveNet model has not been prepared and trained")

        frame = coerce_history_frame(
            history,
            sensor_ids=bundle.data.sensor_ids,
            freq=bundle.data.freq,
            fallback_timestamps=bundle.data.timestamps,
        )
        if len(frame) < self.config.history_steps:
            raise ValueError(f"Need at least {self.config.history_steps} history steps for prediction")

        frame = frame.iloc[-self.config.history_steps :]
        features = add_graph_wavenet_features(
            frame,
            add_time_in_day=self.config.add_time_in_day,
            add_day_in_week=self.config.add_day_in_week,
        )[None, ...]
        features[..., 0] = bundle.scaler.transform(features[..., 0])

        device = next(self.model.parameters()).device
        tensor = torch.from_numpy(features).float().to(device)
        with torch.no_grad():
            output = self.model(F.pad(tensor.transpose(1, 3), (1, 0, 0, 0))).transpose(1, 3)
        output = bundle.scaler.inverse_transform(output).squeeze(0).squeeze(0).transpose(0, 1)
        return output.cpu().numpy()
