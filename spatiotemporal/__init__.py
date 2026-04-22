from .core import SplitConfig, TrafficData, TrainingResult
from .datasets import DATASET_REGISTRY, BaseTrafficDataset, create_dataset

__all__ = [
    "BaseTrafficDataset",
    "DATASET_REGISTRY",
    "ExperimentRunner",
    "RunManager",
    "SplitConfig",
    "TrafficData",
    "TrainingResult",
    "create_dataset",
]


def __getattr__(name):
    if name == "ExperimentRunner":
        from .api import ExperimentRunner

        return ExperimentRunner
    if name == "RunManager":
        from .run_manager import RunManager

        return RunManager
    raise AttributeError(name)
