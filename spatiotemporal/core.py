from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float
    val_ratio: float
    test_ratio: float

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


@dataclass
class TrafficData:
    name: str
    values: np.ndarray
    sensor_ids: list[str]
    timestamps: pd.DatetimeIndex
    adjacency: sp.spmatrix | np.ndarray | None = None
    freq: str = "5min"
    feature_names: list[str] = field(default_factory=lambda: ["speed"])
    target_feature: str = "speed"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        if values.ndim == 2:
            values = values[..., None]
        if values.ndim != 3:
            raise ValueError(f"Traffic values must have shape [T, N, F], got {values.shape}")

        self.values = values
        if self.values.shape[1] != len(self.sensor_ids):
            raise ValueError("Sensor id count must match data width")
        if len(self.timestamps) != self.values.shape[0]:
            raise ValueError("Timestamp count must match data length")
        if len(self.feature_names) != self.values.shape[2]:
            raise ValueError("Feature name count must match feature dimension")
        if self.target_feature not in self.feature_names:
            raise ValueError(f"Unknown target_feature: {self.target_feature}")

    @property
    def num_timesteps(self) -> int:
        return int(self.values.shape[0])

    @property
    def num_nodes(self) -> int:
        return int(self.values.shape[1])

    @property
    def num_features(self) -> int:
        return int(self.values.shape[2])

    @property
    def target_index(self) -> int:
        return self.feature_names.index(self.target_feature)

    @property
    def target_values(self) -> np.ndarray:
        return self.values[..., self.target_index]

    def feature_dataframe(self, feature: int | str = 0) -> pd.DataFrame:
        if isinstance(feature, str):
            feature_index = self.feature_names.index(feature)
        else:
            feature_index = int(feature)
        return pd.DataFrame(
            self.values[..., feature_index],
            index=self.timestamps,
            columns=self.sensor_ids,
        )

    def to_dataframe(self) -> pd.DataFrame:
        return self.feature_dataframe(self.target_feature)


@dataclass
class TrainingResult:
    model_name: str
    dataset_name: str
    checkpoint_path: Path | None
    best_epoch: int | None
    metrics: dict[str, float]
    history: dict[str, list[float]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    run_dir: Path | None = None
