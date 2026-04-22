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
    if isinstance(history, pd.DataFrame):
        return history.copy()

    values = np.asarray(history, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(sensor_ids):
        raise ValueError("History must have shape [T, N]")

    frame = pd.DataFrame(values, columns=sensor_ids)
    if fallback_timestamps is not None and len(fallback_timestamps) >= len(frame):
        frame.index = fallback_timestamps[-len(frame):]
    else:
        frame.index = pd.date_range(start="2000-01-01", periods=len(frame), freq=freq)
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
