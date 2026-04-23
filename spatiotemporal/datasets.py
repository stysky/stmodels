from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .core import TrafficData
from .preprocessing import (
    build_correlation_adjacency,
    clean_speed_dataframe,
    convert_adj_pkl_to_npz,
    load_adjacency_npz,
    load_pickle_adj,
    read_traffic_h5,
)


class BaseTrafficDataset:
    dataset_name: str = ""
    raw_filename: str = ""
    default_start_date: str = ""
    default_freq: str = "5min"
    graph_path: str | None = None
    graph_source_pkl_path: str | None = None
    graph_source_npz_path: str | None = None
    target_feature_name: str = "speed"

    def __init__(
        self,
        project_root: str | Path,
        zero_as_missing: bool = True,
        fill_method: str = "interpolate",
        adjacency_strategy: str = "auto",
        correlation_top_k: int = 10,
        correlation_min_weight: float = 0.1,
    ) -> None:
        self.project_root = Path(project_root)
        self.zero_as_missing = zero_as_missing
        self.fill_method = fill_method
        self.adjacency_strategy = adjacency_strategy
        self.correlation_top_k = correlation_top_k
        self.correlation_min_weight = correlation_min_weight

    @property
    def raw_path(self) -> Path:
        return self.project_root / "data" / "raw" / self.raw_filename

    def load_raw_frame(self):
        df, sensor_ids = read_traffic_h5(
            self.raw_path,
            start_date=self.default_start_date,
            freq=self.default_freq,
        )
        df = clean_speed_dataframe(
            df,
            treat_zero_as_missing=self.zero_as_missing,
            fill_method=self.fill_method,
        )
        return df, sensor_ids

    def load(self) -> TrafficData:
        df, sensor_ids = self.load_raw_frame()
        adjacency, source = self._resolve_adjacency(sensor_ids=sensor_ids, values=df.values)
        return TrafficData(
            name=self.dataset_name,
            values=df.values[..., None],
            sensor_ids=sensor_ids,
            timestamps=df.index,
            adjacency=adjacency,
            freq=self.default_freq,
            feature_names=[self.target_feature_name],
            target_feature=self.target_feature_name,
            metadata={"adjacency_source": source},
        )

    def _canonical_graph_path(self) -> Path | None:
        if self.graph_path is None:
            return None
        return self.project_root / self.graph_path

    def _resolve_adjacency(self, sensor_ids: list[str], values: np.ndarray):
        strategy = self.adjacency_strategy
        if strategy == "none":
            return None, "none"

        canonical_path = self._canonical_graph_path()
        if self.graph_source_pkl_path and strategy in {"auto", "official", "official_pkl"}:
            source_path = self.project_root / self.graph_source_pkl_path
            if source_path.exists():
                official_sensor_ids, _, adj = load_pickle_adj(source_path)
                if official_sensor_ids and list(official_sensor_ids) != list(sensor_ids):
                    raise ValueError(
                        f"Sensor order mismatch between raw data and official graph for {self.dataset_name}"
                    )
                if canonical_path is not None:
                    convert_adj_pkl_to_npz(source_path, canonical_path)
                if canonical_path is not None and canonical_path.exists():
                    return load_adjacency_npz(canonical_path), str(canonical_path)
                return sp.csc_matrix(adj), str(source_path)

        if canonical_path is not None and strategy in {"auto", "canonical", "graph_npz"} and canonical_path.exists():
            return load_adjacency_npz(canonical_path), str(canonical_path)

        if self.graph_source_npz_path and strategy in {"auto", "fallback_npz", "graph_npz"}:
            path = self.project_root / self.graph_source_npz_path
            if path.exists():
                if canonical_path is not None and not canonical_path.exists():
                    canonical_path.parent.mkdir(parents=True, exist_ok=True)
                    canonical_path.write_bytes(path.read_bytes())
                    return load_adjacency_npz(canonical_path), str(canonical_path)
                return load_adjacency_npz(path), str(path)

        if strategy in {"auto", "correlation"}:
            adj = build_correlation_adjacency(
                values,
                top_k=self.correlation_top_k,
                min_weight=self.correlation_min_weight,
            )
            return sp.csc_matrix(adj), "correlation"
        raise FileNotFoundError(f"Unable to resolve adjacency for dataset {self.dataset_name}")


class MetrLADataset(BaseTrafficDataset):
    dataset_name = "metr-la"
    raw_filename = "metr-la.h5"
    default_start_date = "2012-03-01"
    graph_path = "data/graphs/metr-la_adj.npz"
    graph_source_pkl_path = "data/raw/adj_METR-LA.pkl"


class PemsBayDataset(BaseTrafficDataset):
    dataset_name = "pems-bay"
    raw_filename = "pems-bay.h5"
    default_start_date = "2017-01-01"
    graph_path = "data/graphs/pems-bay_adj.npz"
    graph_source_pkl_path = "data/raw/adj_mx_bay.pkl"


DATASET_REGISTRY = {
    "metr-la": MetrLADataset,
    "pems-bay": PemsBayDataset,
}


def list_datasets() -> tuple[str, ...]:
    return tuple(sorted(DATASET_REGISTRY))


def normalize_dataset_name(name: str) -> str:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Dataset name must not be empty")
    return normalized


def get_dataset_class(name: str) -> type[BaseTrafficDataset]:
    normalized = normalize_dataset_name(name)
    dataset_cls = DATASET_REGISTRY.get(normalized)
    if dataset_cls is None:
        supported = ", ".join(list_datasets())
        raise KeyError(f"Unsupported dataset: {name}. Supported datasets: {supported}")
    return dataset_cls


def create_dataset(name: str, project_root: str | Path, **kwargs) -> BaseTrafficDataset:
    dataset_cls = get_dataset_class(name)
    return dataset_cls(project_root=project_root, **kwargs)
