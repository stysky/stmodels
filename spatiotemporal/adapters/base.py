from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..core import TrafficData, TrainingResult


def resolve_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def coerce_history_frame(
    history,
    sensor_ids: list[str],
    freq: str,
    fallback_timestamps: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    def build_default_index(length: int) -> pd.DatetimeIndex:
        if fallback_timestamps is not None and len(fallback_timestamps) >= length:
            return fallback_timestamps[-length:]
        return pd.date_range(start="2000-01-01", periods=length, freq=freq)

    if isinstance(history, pd.DataFrame):
        frame = history.copy()
        if set(sensor_ids).issubset(frame.columns):
            frame = frame.loc[:, sensor_ids]
        elif list(frame.columns) != list(sensor_ids):
            raise ValueError("History DataFrame columns must exactly match sensor_ids")

        if not isinstance(frame.index, pd.DatetimeIndex):
            frame.index = build_default_index(len(frame))
        return frame

    values = np.asarray(history, dtype=np.float32)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2 or values.shape[1] != len(sensor_ids):
        raise ValueError("History must have shape [T, N]")

    frame = pd.DataFrame(values, columns=sensor_ids)
    frame.index = build_default_index(len(frame))
    return frame


class BaseModelAdapter(ABC):
    model_name: str = ""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root)
        self.model = None
        self.last_bundle = None
        self.last_result: TrainingResult | None = None

    @abstractmethod
    def prepare_training_bundle(self, data: TrafficData):
        raise NotImplementedError

    @abstractmethod
    def export_native_artifacts(self, bundle, output_dir: str | Path):
        raise NotImplementedError

    @abstractmethod
    def train(self, bundle) -> TrainingResult:
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, bundle, checkpoint_path: str | Path):
        raise NotImplementedError

    @abstractmethod
    def predict(self, history, bundle=None):
        raise NotImplementedError
