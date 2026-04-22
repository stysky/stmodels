from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..core import SplitConfig, TrafficData, TrainingResult
from ..preprocessing import StandardScaler, calc_chebynet_gso, calc_gso, make_stgcn_windows, split_indices
from .base import BaseModelAdapter, coerce_history_frame, resolve_device


@dataclass
class STGCNConfig:
    history_steps: int = 12
    prediction_step: int = 3
    split: SplitConfig = SplitConfig(0.7, 0.15, 0.15)
    batch_size: int = 32
    device: str = "cuda"
    epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    step_size: int = 10
    gamma: float = 0.95
    patience: int = 10
    Kt: int = 3
    Ks: int = 3
    stblock_num: int = 2
    act_func: str = "glu"
    graph_conv_type: str = "cheb_graph_conv"
    gso_type: str = "sym_norm_lap"
    enable_bias: bool = True
    droprate: float = 0.5
    checkpoint_path: str | None = None


@dataclass
class STGCNBundle:
    data: TrafficData
    scaled_splits: dict[str, np.ndarray]
    loaders: dict[str, DataLoader]
    scaler: StandardScaler
    gso: torch.Tensor


class STGCNAdapter(BaseModelAdapter):
    model_name = "stgcn"

    def __init__(self, project_root: str | Path, config: STGCNConfig | None = None) -> None:
        super().__init__(project_root)
        self.config = config or STGCNConfig()
        self._models_module = self._load_models_module()

    def _load_models_module(self):
        stgcn_root = self.project_root / "models" / "stgcn-main"
        if str(stgcn_root) not in sys.path:
            sys.path.insert(0, str(stgcn_root))
        from model import models as stgcn_models

        return stgcn_models

    def _compute_blocks(self):
        ko = self.config.history_steps - (self.config.Kt - 1) * 2 * self.config.stblock_num
        blocks = [[1]]
        for _ in range(self.config.stblock_num):
            blocks.append([64, 16, 64])
        if ko == 0:
            blocks.append([128])
        elif ko > 0:
            blocks.append([128, 128])
        blocks.append([1])
        return ko, blocks

    def _build_model(self, bundle: STGCNBundle, device: torch.device):
        ko, blocks = self._compute_blocks()
        args = SimpleNamespace(
            n_his=self.config.history_steps,
            Kt=self.config.Kt,
            Ks=self.config.Ks,
            act_func=self.config.act_func,
            graph_conv_type=self.config.graph_conv_type,
            gso=bundle.gso.to(device),
            enable_bias=self.config.enable_bias,
            droprate=self.config.droprate,
        )
        if self.config.graph_conv_type == "cheb_graph_conv":
            model = self._models_module.STGCNChebGraphConv(args, blocks, bundle.data.num_nodes).to(device)
        else:
            model = self._models_module.STGCNGraphConv(args, blocks, bundle.data.num_nodes).to(device)
        return model, ko

    def prepare_training_bundle(self, data: TrafficData) -> STGCNBundle:
        train_slice, val_slice, test_slice = split_indices(data.num_timesteps, self.config.split)
        raw_splits = {
            "train": data.values[train_slice],
            "val": data.values[val_slice],
            "test": data.values[test_slice],
        }
        scaler = StandardScaler(
            mean=raw_splits["train"].mean(axis=0, keepdims=True),
            std=raw_splits["train"].std(axis=0, keepdims=True),
        )
        scaled_splits = {name: scaler.transform(values).astype(np.float32) for name, values in raw_splits.items()}

        loaders = {}
        for name, values in scaled_splits.items():
            x, y = make_stgcn_windows(values, self.config.history_steps, self.config.prediction_step)
            dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
            loaders[name] = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=(name == "train"))

        adjacency = data.adjacency
        if adjacency is None:
            adjacency = sp.eye(data.num_nodes, format="csc", dtype=np.float32)
        elif not sp.issparse(adjacency):
            adjacency = sp.csc_matrix(adjacency)
        gso = calc_gso(adjacency, self.config.gso_type)
        if self.config.graph_conv_type == "cheb_graph_conv":
            gso = calc_chebynet_gso(gso)
        gso_tensor = torch.from_numpy(gso.toarray().astype(np.float32))

        bundle = STGCNBundle(
            data=data,
            scaled_splits=scaled_splits,
            loaders=loaders,
            scaler=scaler,
            gso=gso_tensor,
        )
        self.last_bundle = bundle
        return bundle

    def export_native_artifacts(self, bundle: STGCNBundle, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_dir / "vel.csv", bundle.data.values, delimiter=",", fmt="%.6f")
        adjacency = bundle.data.adjacency
        if adjacency is None:
            adjacency = sp.eye(bundle.data.num_nodes, format="csc", dtype=np.float32)
        elif not sp.issparse(adjacency):
            adjacency = sp.csc_matrix(adjacency)
        sp.save_npz(output_dir / "adj.npz", adjacency)
        return {"data_dir": output_dir}

    def train(self, bundle: STGCNBundle) -> TrainingResult:
        device = resolve_device(self.config.device)
        model, ko = self._build_model(bundle, device)

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.step_size,
            gamma=self.config.gamma,
        )
        checkpoint_path = Path(
            self.config.checkpoint_path
            or (self.project_root / "checkpoints" / f"{bundle.data.name}_stgcn.pt")
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
                prediction = model(batch_x).view(len(batch_x), -1)
                loss = loss_fn(prediction, batch_y)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))

            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_y in bundle.loaders["val"]:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    prediction = model(batch_x).view(len(batch_x), -1)
                    val_losses.append(float(loss_fn(prediction, batch_y).item()))

            scheduler.step()
            mean_train = float(np.mean(train_losses))
            mean_val = float(np.mean(val_losses))
            history["train_loss"].append(mean_train)
            history["val_loss"].append(mean_val)
            print(f"[STGCN][Epoch {epoch:03d}] train={mean_train:.4f} val={mean_val:.4f}", flush=True)

            if mean_val < best_val:
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
        preds = []
        reals = []
        with torch.no_grad():
            for batch_x, batch_y in bundle.loaders["test"]:
                batch_x = batch_x.to(device)
                preds.append(model(batch_x).view(len(batch_x), -1).cpu().numpy())
                reals.append(batch_y.numpy())
        pred = bundle.scaler.inverse_transform(np.concatenate(preds, axis=0))
        real = bundle.scaler.inverse_transform(np.concatenate(reals, axis=0))

        mae = float(np.mean(np.abs(pred - real)))
        rmse = float(np.sqrt(np.mean((pred - real) ** 2)))
        wmape = float(np.sum(np.abs(pred - real)) / np.sum(real))

        self.model = model
        result = TrainingResult(
            model_name=self.model_name,
            dataset_name=bundle.data.name,
            checkpoint_path=checkpoint_path,
            best_epoch=best_epoch,
            metrics={"mae": mae, "rmse": rmse, "wmape": wmape, "best_val_loss": best_val},
            history=history,
            extra={"prediction_step": self.config.prediction_step, "ko": ko},
        )
        self.last_result = result
        return result

    def load_checkpoint(self, bundle: STGCNBundle, checkpoint_path: str | Path):
        device = resolve_device(self.config.device)
        model, _ = self._build_model(bundle, device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model
        self.last_bundle = bundle
        return model

    def predict(self, history, bundle: STGCNBundle | None = None):
        bundle = bundle or self.last_bundle
        if bundle is None or self.model is None:
            raise RuntimeError("STGCN model has not been prepared and trained")

        frame = coerce_history_frame(
            history,
            sensor_ids=bundle.data.sensor_ids,
            freq=bundle.data.freq,
            fallback_timestamps=bundle.data.timestamps,
        )
        if len(frame) < self.config.history_steps:
            raise ValueError(f"Need at least {self.config.history_steps} history steps for prediction")

        values = frame.iloc[-self.config.history_steps :].values.astype(np.float32)
        values = bundle.scaler.transform(values)
        model_input = torch.from_numpy(values.reshape(1, 1, self.config.history_steps, bundle.data.num_nodes)).float()

        device = next(self.model.parameters()).device
        with torch.no_grad():
            prediction = self.model(model_input.to(device)).view(1, -1).cpu().numpy()
        return bundle.scaler.inverse_transform(prediction)[0]
