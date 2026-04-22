from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .core import TrainingResult


@dataclass
class RunPaths:
    run_dir: Path
    checkpoint_dir: Path
    export_dir: Path
    config_path: Path
    result_path: Path
    summary_path: Path


class RunManager:
    def __init__(self, run_root: str | Path) -> None:
        self.run_root = Path(run_root)

    def create_paths(self, model_name: str, dataset_name: str, tag: str = "") -> RunPaths:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{timestamp}"
        if tag:
            run_name = f"{run_name}-{tag}"

        run_dir = self.run_root / model_name / dataset_name / run_name
        checkpoint_dir = run_dir / "checkpoints"
        export_dir = run_dir / "exports"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        export_dir.mkdir(parents=True, exist_ok=True)

        return RunPaths(
            run_dir=run_dir,
            checkpoint_dir=checkpoint_dir,
            export_dir=export_dir,
            config_path=run_dir / "resolved_config.json",
            result_path=run_dir / "result.json",
            summary_path=run_dir / "summary.txt",
        )

    def save_config(self, config_path: str | Path, payload: dict[str, Any]) -> None:
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=self._json_default)

    def save_result(self, result: TrainingResult) -> None:
        if result.run_dir is None:
            raise ValueError("TrainingResult.run_dir must be set before saving")

        payload = asdict(result)
        payload["checkpoint_path"] = str(result.checkpoint_path) if result.checkpoint_path else None
        payload["run_dir"] = str(result.run_dir)
        with open(result.run_dir / "result.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=self._json_default)

        summary_lines = [
            f"model: {result.model_name}",
            f"dataset: {result.dataset_name}",
            f"run_dir: {result.run_dir}",
            f"checkpoint: {result.checkpoint_path}",
            f"best_epoch: {result.best_epoch}",
            f"metrics: {result.metrics}",
        ]
        with open(result.run_dir / "summary.txt", "w", encoding="utf-8") as handle:
            handle.write("\n".join(summary_lines) + "\n")

    @staticmethod
    def _json_default(value):
        if isinstance(value, Path):
            return str(value)
        return value
