from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from ..core import SplitConfig
from .model_zoo import (
    AGCRNAdapter,
    AGCRNConfig,
    ASTGCNAdapter,
    ASTGCNConfig,
    DCRNNAdapter,
    DCRNNConfig,
    DGCRNAdapter,
    DGCRNConfig,
    GMANAdapter,
    GMANConfig,
    CONFIG_REGISTRY,
    GraphWaveNetAdapter,
    GraphWaveNetConfig,
    MODEL_REGISTRY,
    MODEL_SPECS,
    ModelSpec,
    MTGNNAdapter,
    MTGNNConfig,
    RegisteredForecastAdapter,
    STGCNAdapter,
    STGCNConfig,
    STIDAdapter,
    STIDConfig,
)


def list_models() -> tuple[str, ...]:
    return tuple(sorted(MODEL_SPECS))


def normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip().lower()
    if not normalized:
        raise ValueError("Model name must not be empty")
    return normalized


def get_model_spec(model_name: str) -> ModelSpec:
    normalized = normalize_model_name(model_name)
    spec = MODEL_SPECS.get(normalized)
    if spec is None:
        supported = ", ".join(list_models())
        raise KeyError(f"Unsupported model: {model_name}. Supported models: {supported}")
    return spec


def create_model_adapter(model_name: str, project_root, **model_kwargs):
    spec = get_model_spec(model_name)
    adapter_cls = MODEL_REGISTRY[spec.name]
    return adapter_cls(project_root=project_root, **model_kwargs)


def coerce_model_config(model_name: str, config: Any = None):
    spec = get_model_spec(model_name)
    config_cls = spec.config_cls
    if config is None:
        return config_cls()
    if isinstance(config, config_cls):
        return config
    if isinstance(config, dict):
        config_data = config.copy()
        split = config_data.get("split")
        if isinstance(split, dict):
            config_data["split"] = SplitConfig(**split)
        return config_cls(**config_data)
    if is_dataclass(config):
        return config_cls(**asdict(config))
    raise TypeError(f"Unsupported config for model {normalize_model_name(model_name)}: {type(config)!r}")

__all__ = [
    "AGCRNAdapter",
    "AGCRNConfig",
    "ASTGCNAdapter",
    "ASTGCNConfig",
    "CONFIG_REGISTRY",
    "DCRNNAdapter",
    "DCRNNConfig",
    "DGCRNAdapter",
    "DGCRNConfig",
    "GMANAdapter",
    "GMANConfig",
    "GraphWaveNetAdapter",
    "GraphWaveNetConfig",
    "ModelSpec",
    "MODEL_SPECS",
    "MODEL_REGISTRY",
    "MTGNNAdapter",
    "MTGNNConfig",
    "RegisteredForecastAdapter",
    "STGCNAdapter",
    "STGCNConfig",
    "STIDAdapter",
    "STIDConfig",
    "coerce_model_config",
    "create_model_adapter",
    "get_model_spec",
    "list_models",
    "normalize_model_name",
]
