from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

from .adapters import (
    coerce_model_config as coerce_adapter_config,
    create_model_adapter,
    normalize_model_name,
)
from .core import TrainingResult
from .datasets import create_dataset, normalize_dataset_name
from .run_manager import RunManager, RunPaths


class ExperimentRunner:
    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root)
        self.run_manager = RunManager(self.project_root / "runs")

    def resolve_run_root(self, run_root: str | Path | None = None) -> Path:
        if run_root is None:
            return self.run_manager.run_root
        resolved = Path(run_root)
        if not resolved.is_absolute():
            resolved = self.project_root / resolved
        return resolved

    def coerce_model_config(self, model_name: str, config=None):
        return coerce_adapter_config(model_name, config)

    def _coerce_model_config(self, model_name: str, config=None):
        return self.coerce_model_config(model_name, config)

    def create_dataset(self, dataset_name: str, **dataset_kwargs):
        return create_dataset(dataset_name, project_root=self.project_root, **dataset_kwargs)

    def create_model(self, model_name: str, **model_kwargs):
        return create_model_adapter(model_name, project_root=self.project_root, **model_kwargs)

    def train(
        self,
        model_name: str,
        dataset_name: str,
        config=None,
        dataset_kwargs=None,
        tag: str = "",
        run_root: str | Path | None = None,
        save_run: bool = True,
    ) -> TrainingResult:
        model_name = normalize_model_name(model_name)
        dataset_name = normalize_dataset_name(dataset_name)
        model_config = self.coerce_model_config(model_name, config)
        run_paths = None
        if save_run:
            run_paths = self.create_run(model_name, dataset_name, tag=tag, run_root=run_root)
            model_config = replace(model_config, checkpoint_path=str(run_paths.checkpoint_dir / "best.pt"))

        dataset = self.create_dataset(dataset_name, **(dataset_kwargs or {}))
        data = dataset.load()
        adapter = self.create_model(model_name, config=model_config)
        bundle = adapter.prepare_training_bundle(data)
        result = adapter.train(bundle)

        if run_paths is not None:
            result.run_dir = run_paths.run_dir
            self.run_manager.save_config(
                run_paths.config_path,
                {
                    "model": model_name,
                    "dataset": dataset_name,
                    "dataset_kwargs": dataset_kwargs or {},
                    "model_config": asdict(model_config),
                    "run_dir": str(run_paths.run_dir),
                },
            )
            self.run_manager.save_result(result)
        return result

    def export(
        self,
        model_name: str,
        dataset_name: str,
        config=None,
        output_dir: str | Path | None = None,
        dataset_kwargs=None,
    ):
        model_name = normalize_model_name(model_name)
        dataset_name = normalize_dataset_name(dataset_name)
        model_config = self.coerce_model_config(model_name, config)
        dataset = self.create_dataset(dataset_name, **(dataset_kwargs or {}))
        data = dataset.load()
        adapter = self.create_model(model_name, config=model_config)
        bundle = adapter.prepare_training_bundle(data)
        if output_dir is None:
            output_dir = self.project_root / "artifacts" / dataset_name / model_name
        return adapter.export_native_artifacts(bundle, output_dir)

    def predict(
        self,
        model_name: str,
        dataset_name: str,
        checkpoint_path: str | Path,
        history,
        config=None,
        dataset_kwargs=None,
    ):
        model_name = normalize_model_name(model_name)
        dataset_name = normalize_dataset_name(dataset_name)
        model_config = self.coerce_model_config(model_name, config)
        dataset = self.create_dataset(dataset_name, **(dataset_kwargs or {}))
        data = dataset.load()
        adapter = self.create_model(model_name, config=model_config)
        bundle = adapter.prepare_training_bundle(data)
        adapter.load_checkpoint(bundle, checkpoint_path)
        return adapter.predict(history, bundle=bundle)

    def create_run(self, model_name: str, dataset_name: str, tag: str = "", run_root: str | Path | None = None) -> RunPaths:
        resolved_run_root = self.resolve_run_root(run_root)
        manager = self.run_manager if resolved_run_root == self.run_manager.run_root else RunManager(resolved_run_root)
        return manager.create_paths(model_name, dataset_name, tag=tag)
