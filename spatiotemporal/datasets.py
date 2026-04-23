from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp

from .core import TrafficData
from .preprocessing import (
    build_correlation_adjacency,
    build_datetime_index,
    build_edge_adjacency,
    clean_feature_array,
    convert_adj_pkl_to_npz,
    load_adjacency_npz,
    load_pickle_adj,
    read_sensor_ids,
    read_traffic_h5,
    read_traffic_npz,
    save_adj_npz,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    data_format: str
    data_path: str
    data_key: str = "data"
    default_start_date: str = "2000-01-01"
    default_freq: str = "5min"
    graph_path: str | None = None
    graph_source_pkl_path: str | None = None
    graph_source_npz_path: str | None = None
    graph_source_csv_path: str | None = None
    graph_source_col: str = "from"
    graph_target_col: str = "to"
    graph_weight_col: str | None = None
    graph_weight_mode: str = "gaussian"
    graph_threshold: float = 0.1
    graph_include_reverse: bool = False
    graph_self_loop_weight: float | None = None
    sensor_ids_path: str | None = None
    feature_names: tuple[str, ...] = field(default_factory=tuple)
    target_feature_name: str = ""
    clean_feature_indices: tuple[int, ...] | None = None
    zero_as_missing: bool = True
    fill_method: str = "interpolate"


class BaseTrafficDataset:
    spec: DatasetSpec = DatasetSpec(
        name="",
        data_format="h5",
        data_path="",
    )

    def __init__(
        self,
        project_root: str | Path,
        *,
        spec: DatasetSpec | None = None,
        spec_base_dir: str | Path | None = None,
        start_date: str | None = None,
        freq: str | None = None,
        feature_names: Sequence[str] | None = None,
        target_feature: str | None = None,
        clean_feature_indices: Sequence[int] | None = None,
        zero_as_missing: bool | None = None,
        fill_method: str | None = None,
        adjacency_strategy: str = "auto",
        correlation_top_k: int = 10,
        correlation_min_weight: float = 0.1,
        data_path: str | None = None,
        data_key: str | None = None,
        sensor_ids_path: str | None = None,
        graph_path: str | None = None,
        graph_source_pkl_path: str | None = None,
        graph_source_npz_path: str | None = None,
        graph_source_csv_path: str | None = None,
        graph_weight_col: str | None = None,
        graph_weight_mode: str | None = None,
        graph_threshold: float | None = None,
        graph_include_reverse: bool | None = None,
        graph_self_loop_weight: float | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.spec_base_dir = Path(spec_base_dir) if spec_base_dir is not None else self.project_root
        base_spec = spec or self.spec
        self.spec = replace(
            base_spec,
            data_path=data_path or base_spec.data_path,
            data_key=data_key or base_spec.data_key,
            sensor_ids_path=sensor_ids_path if sensor_ids_path is not None else base_spec.sensor_ids_path,
            graph_path=graph_path if graph_path is not None else base_spec.graph_path,
            graph_source_pkl_path=graph_source_pkl_path if graph_source_pkl_path is not None else base_spec.graph_source_pkl_path,
            graph_source_npz_path=graph_source_npz_path if graph_source_npz_path is not None else base_spec.graph_source_npz_path,
            graph_source_csv_path=graph_source_csv_path if graph_source_csv_path is not None else base_spec.graph_source_csv_path,
            graph_weight_col=graph_weight_col if graph_weight_col is not None else base_spec.graph_weight_col,
            graph_weight_mode=graph_weight_mode or base_spec.graph_weight_mode,
            graph_threshold=graph_threshold if graph_threshold is not None else base_spec.graph_threshold,
            graph_include_reverse=graph_include_reverse if graph_include_reverse is not None else base_spec.graph_include_reverse,
            graph_self_loop_weight=graph_self_loop_weight if graph_self_loop_weight is not None else base_spec.graph_self_loop_weight,
            feature_names=tuple(feature_names) if feature_names is not None else base_spec.feature_names,
            target_feature_name=target_feature or base_spec.target_feature_name,
            clean_feature_indices=tuple(int(index) for index in clean_feature_indices)
            if clean_feature_indices is not None
            else base_spec.clean_feature_indices,
            default_start_date=start_date or base_spec.default_start_date,
            default_freq=freq or base_spec.default_freq,
            zero_as_missing=base_spec.zero_as_missing if zero_as_missing is None else bool(zero_as_missing),
            fill_method=fill_method or base_spec.fill_method,
        )
        self.adjacency_strategy = adjacency_strategy
        self.correlation_top_k = correlation_top_k
        self.correlation_min_weight = correlation_min_weight

    @property
    def dataset_name(self) -> str:
        return self.spec.name

    @property
    def raw_path(self) -> Path:
        return self._resolve_path(self.spec.data_path, require_exists=True)

    def _resolve_path(self, path_value: str | None, require_exists: bool) -> Path | None:
        if not path_value:
            return None
        raw_path = Path(path_value)
        if raw_path.is_absolute():
            return raw_path

        candidates: list[Path] = []
        if self.spec_base_dir is not None:
            candidates.append(self.spec_base_dir / raw_path)
        candidates.append(self.project_root / raw_path)

        deduped: list[Path] = []
        seen: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.resolve(strict=False)
            if resolved not in seen:
                seen.add(resolved)
                deduped.append(resolved)

        if require_exists:
            for candidate in deduped:
                if candidate.exists():
                    return candidate
            joined = ", ".join(str(candidate) for candidate in deduped)
            raise FileNotFoundError(f"Unable to resolve path '{path_value}' for dataset {self.dataset_name}. Tried: {joined}")
        return deduped[0] if deduped else None

    def _default_sensor_ids(self, num_nodes: int) -> list[str]:
        return [str(index) for index in range(num_nodes)]

    def _resolve_sensor_ids(self, values: np.ndarray) -> list[str]:
        sensor_ids_path = self._resolve_path(self.spec.sensor_ids_path, require_exists=True) if self.spec.sensor_ids_path else None
        if sensor_ids_path is None:
            return self._default_sensor_ids(values.shape[1])
        sensor_ids = read_sensor_ids(sensor_ids_path)
        if len(sensor_ids) != values.shape[1]:
            raise ValueError(
                f"Sensor id count mismatch for {self.dataset_name}: "
                f"expected {values.shape[1]}, got {len(sensor_ids)} from {sensor_ids_path}"
            )
        return sensor_ids

    def _resolve_feature_names(self, num_features: int) -> list[str]:
        if self.spec.feature_names:
            if len(self.spec.feature_names) != num_features:
                raise ValueError(
                    f"Feature name count mismatch for {self.dataset_name}: "
                    f"expected {num_features}, got {len(self.spec.feature_names)}"
                )
            return list(self.spec.feature_names)
        if num_features == 1:
            return [self.spec.target_feature_name or "value"]
        return [f"feature_{index}" for index in range(num_features)]

    def _resolve_target_feature(self, feature_names: Sequence[str]) -> str:
        if self.spec.target_feature_name:
            return self.spec.target_feature_name
        return feature_names[0]

    def load_raw(self) -> tuple[np.ndarray, list[str], list[str], str, Any]:
        data_format = self.spec.data_format.strip().lower()
        if data_format == "h5":
            frame, sensor_ids = read_traffic_h5(
                self.raw_path,
                start_date=self.spec.default_start_date,
                freq=self.spec.default_freq,
            )
            values = frame.to_numpy(dtype=np.float32)[..., None]
            timestamps = frame.index
        elif data_format == "npz":
            values = read_traffic_npz(self.raw_path, data_key=self.spec.data_key)
            sensor_ids = self._resolve_sensor_ids(values)
            timestamps = build_datetime_index(
                values.shape[0],
                start_date=self.spec.default_start_date,
                freq=self.spec.default_freq,
            )
        else:
            raise ValueError(f"Unsupported data_format for {self.dataset_name}: {self.spec.data_format}")

        values = clean_feature_array(
            values,
            treat_zero_as_missing=self.spec.zero_as_missing,
            fill_method=self.spec.fill_method,
            feature_indices=self.spec.clean_feature_indices,
        )
        feature_names = self._resolve_feature_names(values.shape[-1])
        target_feature = self._resolve_target_feature(feature_names)
        return values, sensor_ids, feature_names, target_feature, timestamps

    def load(self) -> TrafficData:
        values, sensor_ids, feature_names, target_feature, timestamps = self.load_raw()
        adjacency, source = self._resolve_adjacency(sensor_ids=sensor_ids, values=values[..., 0])
        return TrafficData(
            name=self.dataset_name,
            values=values,
            sensor_ids=sensor_ids,
            timestamps=timestamps,
            adjacency=adjacency,
            freq=self.spec.default_freq,
            feature_names=feature_names,
            target_feature=target_feature,
            metadata={
                "adjacency_source": source,
                "data_path": str(self.raw_path),
                "data_format": self.spec.data_format,
            },
        )

    def _canonical_graph_path(self) -> Path | None:
        return self._resolve_path(self.spec.graph_path, require_exists=False) if self.spec.graph_path else None

    def _resolve_adjacency(self, sensor_ids: list[str], values: np.ndarray):
        strategy = self.adjacency_strategy
        if strategy == "none":
            return None, "none"

        canonical_path = self._canonical_graph_path()
        if self.spec.graph_source_pkl_path and strategy in {"auto", "official", "official_pkl"}:
            source_path = self._resolve_path(self.spec.graph_source_pkl_path, require_exists=True)
            if source_path is not None and source_path.exists():
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

        if self.spec.graph_source_npz_path and strategy in {"auto", "fallback_npz", "graph_npz"}:
            npz_path = self._resolve_path(self.spec.graph_source_npz_path, require_exists=True)
            if npz_path is not None and npz_path.exists():
                if canonical_path is not None and not canonical_path.exists():
                    canonical_path.parent.mkdir(parents=True, exist_ok=True)
                    canonical_path.write_bytes(npz_path.read_bytes())
                    return load_adjacency_npz(canonical_path), str(canonical_path)
                return load_adjacency_npz(npz_path), str(npz_path)

        if self.spec.graph_source_csv_path and strategy in {"auto", "official", "official_csv", "edge_csv"}:
            csv_path = self._resolve_path(self.spec.graph_source_csv_path, require_exists=True)
            if csv_path is not None and csv_path.exists():
                adjacency = build_edge_adjacency(
                    csv_path,
                    sensor_ids=sensor_ids,
                    source_col=self.spec.graph_source_col,
                    target_col=self.spec.graph_target_col,
                    weight_col=self.spec.graph_weight_col,
                    weight_mode=self.spec.graph_weight_mode,
                    threshold=self.spec.graph_threshold,
                    include_reverse=self.spec.graph_include_reverse,
                    self_loop_weight=self.spec.graph_self_loop_weight,
                )
                if canonical_path is not None:
                    save_adj_npz(canonical_path, adjacency)
                    return load_adjacency_npz(canonical_path), str(canonical_path)
                return sp.csc_matrix(adjacency), str(csv_path)

        if strategy in {"auto", "correlation"}:
            adj = build_correlation_adjacency(
                values,
                top_k=self.correlation_top_k,
                min_weight=self.correlation_min_weight,
            )
            return sp.csc_matrix(adj), "correlation"
        raise FileNotFoundError(f"Unable to resolve adjacency for dataset {self.dataset_name}")


class ConfigurableTrafficDataset(BaseTrafficDataset):
    def __init__(self, project_root: str | Path, spec: DatasetSpec, spec_base_dir: str | Path, **kwargs) -> None:
        super().__init__(project_root, spec=spec, spec_base_dir=spec_base_dir, **kwargs)


class MetrLADataset(BaseTrafficDataset):
    spec = DatasetSpec(
        name="metr-la",
        data_format="h5",
        data_path="data/raw/METR-LA/metr-la.h5",
        default_start_date="2012-03-01",
        graph_path="data/graphs/metr-la_adj.npz",
        graph_source_pkl_path="data/raw/METR-LA/adj_METR-LA.pkl",
        feature_names=("speed",),
        target_feature_name="speed",
        clean_feature_indices=(0,),
    )


class PemsBayDataset(BaseTrafficDataset):
    spec = DatasetSpec(
        name="pems-bay",
        data_format="h5",
        data_path="data/raw/PEMS-BAY/pems-bay.h5",
        default_start_date="2017-01-01",
        graph_path="data/graphs/pems-bay_adj.npz",
        graph_source_pkl_path="data/raw/PEMS-BAY/adj_mx_bay.pkl",
        feature_names=("speed",),
        target_feature_name="speed",
        clean_feature_indices=(0,),
    )


class Pems03Dataset(BaseTrafficDataset):
    spec = DatasetSpec(
        name="pems03",
        data_format="npz",
        data_path="data/raw/PEMS03/PEMS03.npz",
        sensor_ids_path="data/raw/PEMS03/PEMS03.txt",
        default_start_date="2018-09-01",
        graph_path="data/graphs/pems03_adj.npz",
        graph_source_csv_path="data/raw/PEMS03/PEMS03.csv",
        graph_weight_col="distance",
        graph_weight_mode="gaussian",
        feature_names=("flow",),
        target_feature_name="flow",
        clean_feature_indices=(0,),
    )


class Pems04Dataset(BaseTrafficDataset):
    spec = DatasetSpec(
        name="pems04",
        data_format="npz",
        data_path="data/raw/PEMS04/PEMS04.npz",
        default_start_date="2018-01-01",
        graph_path="data/graphs/pems04_adj.npz",
        graph_source_csv_path="data/raw/PEMS04/PEMS04.csv",
        graph_weight_col="cost",
        graph_weight_mode="gaussian",
        feature_names=("flow", "occupancy", "speed"),
        target_feature_name="flow",
        clean_feature_indices=(0,),
    )


class Pems07Dataset(BaseTrafficDataset):
    spec = DatasetSpec(
        name="pems07",
        data_format="npz",
        data_path="data/raw/PEMS07/PEMS07.npz",
        default_start_date="2017-05-01",
        graph_path="data/graphs/pems07_adj.npz",
        graph_source_csv_path="data/raw/PEMS07/PEMS07.csv",
        graph_weight_col="cost",
        graph_weight_mode="gaussian",
        feature_names=("flow",),
        target_feature_name="flow",
        clean_feature_indices=(0,),
    )


class Pems08Dataset(BaseTrafficDataset):
    spec = DatasetSpec(
        name="pems08",
        data_format="npz",
        data_path="data/raw/PEMS08/PEMS08.npz",
        default_start_date="2016-07-01",
        graph_path="data/graphs/pems08_adj.npz",
        graph_source_csv_path="data/raw/PEMS08/PEMS08.csv",
        graph_weight_col="cost",
        graph_weight_mode="gaussian",
        feature_names=("flow", "occupancy", "speed"),
        target_feature_name="flow",
        clean_feature_indices=(0,),
    )


DATASET_REGISTRY = {
    "metr-la": MetrLADataset,
    "pems-bay": PemsBayDataset,
    "pems03": Pems03Dataset,
    "pems04": Pems04Dataset,
    "pems07": Pems07Dataset,
    "pems08": Pems08Dataset,
}


def normalize_dataset_name(name: str) -> str:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Dataset name must not be empty")
    return normalized


def load_dataset_spec(spec_path: str | Path) -> DatasetSpec:
    spec_path = Path(spec_path)
    with spec_path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)

    dataset_name = normalize_dataset_name(payload.get("name", spec_path.parent.name))
    feature_names = tuple(payload.get("feature_names", ()))
    clean_feature_indices = payload.get("clean_feature_indices")
    return DatasetSpec(
        name=dataset_name,
        data_format=payload["data_format"],
        data_path=payload["data_path"],
        data_key=payload.get("data_key", "data"),
        default_start_date=payload.get("default_start_date", "2000-01-01"),
        default_freq=payload.get("default_freq", "5min"),
        graph_path=payload.get("graph_path"),
        graph_source_pkl_path=payload.get("graph_source_pkl_path"),
        graph_source_npz_path=payload.get("graph_source_npz_path"),
        graph_source_csv_path=payload.get("graph_source_csv_path"),
        graph_source_col=payload.get("graph_source_col", "from"),
        graph_target_col=payload.get("graph_target_col", "to"),
        graph_weight_col=payload.get("graph_weight_col"),
        graph_weight_mode=payload.get("graph_weight_mode", "gaussian"),
        graph_threshold=float(payload.get("graph_threshold", 0.1)),
        graph_include_reverse=bool(payload.get("graph_include_reverse", False)),
        graph_self_loop_weight=payload.get("graph_self_loop_weight"),
        sensor_ids_path=payload.get("sensor_ids_path"),
        feature_names=feature_names,
        target_feature_name=payload.get("target_feature_name", ""),
        clean_feature_indices=tuple(int(index) for index in clean_feature_indices) if clean_feature_indices is not None else None,
        zero_as_missing=bool(payload.get("zero_as_missing", True)),
        fill_method=payload.get("fill_method", "interpolate"),
    )


def _discover_dataset_specs(project_root: str | Path | None = None) -> dict[str, Path]:
    root = Path(project_root) if project_root is not None else _project_root()
    spec_paths: dict[str, Path] = {}
    raw_root = root / "data" / "raw"
    if not raw_root.exists():
        return spec_paths

    for spec_path in raw_root.glob("**/dataset.json"):
        try:
            spec = load_dataset_spec(spec_path)
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
        spec_paths[spec.name] = spec_path
    return spec_paths


def list_datasets(project_root: str | Path | None = None) -> tuple[str, ...]:
    names = set(DATASET_REGISTRY)
    names.update(_discover_dataset_specs(project_root))
    return tuple(sorted(names))


def get_dataset_class(name: str, project_root: str | Path | None = None) -> type[BaseTrafficDataset]:
    normalized = normalize_dataset_name(name)
    dataset_cls = DATASET_REGISTRY.get(normalized)
    if dataset_cls is not None:
        return dataset_cls
    if normalized in _discover_dataset_specs(project_root):
        return ConfigurableTrafficDataset
    supported = ", ".join(list_datasets(project_root))
    raise KeyError(f"Unsupported dataset: {name}. Supported datasets: {supported}")


def create_dataset(name: str, project_root: str | Path, **kwargs) -> BaseTrafficDataset:
    normalized = normalize_dataset_name(name)
    dataset_cls = DATASET_REGISTRY.get(normalized)
    if dataset_cls is not None:
        return dataset_cls(project_root=project_root, **kwargs)

    spec_path = kwargs.pop("dataset_spec_path", None) or kwargs.pop("spec_path", None)
    discovered_specs = _discover_dataset_specs(project_root)
    if spec_path is None:
        spec_path = discovered_specs.get(normalized)
    if spec_path is None:
        supported = ", ".join(list_datasets(project_root))
        raise KeyError(f"Unsupported dataset: {name}. Supported datasets: {supported}")

    spec = load_dataset_spec(spec_path)
    if spec.name != normalized:
        spec = replace(spec, name=normalized)
    return ConfigurableTrafficDataset(project_root=project_root, spec=spec, spec_base_dir=Path(spec_path).parent, **kwargs)
