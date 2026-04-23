from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import asdict, replace

from spatiotemporal.adapters import list_models, normalize_model_name
from spatiotemporal.api import ExperimentRunner
from spatiotemporal.config_loader import load_json_config
from spatiotemporal.datasets import list_datasets, normalize_dataset_name
from spatiotemporal.run_manager import RunManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified runner for spatial-temporal forecasting models")
    parser.add_argument("--model", choices=list_models())
    parser.add_argument("--dataset", choices=list_datasets())
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override")
    parser.add_argument("--export_dir", type=str, default="", help="Optional directory for exporting native artifacts")
    parser.add_argument("--export_only", action="store_true", help="Only export model-native artifacts and skip training")
    parser.add_argument("--config", type=str, default="", help="Optional JSON config file")
    parser.add_argument("--run_root", type=str, default="runs", help="Root directory for managed experiment outputs")
    parser.add_argument("--tag", type=str, default="", help="Optional run tag")
    args = parser.parse_args()

    config_data = load_json_config(args.config) if args.config else {}
    raw_model_name = args.model or config_data.get("model", "")
    raw_dataset_name = args.dataset or config_data.get("dataset", "")
    if not raw_model_name or not raw_dataset_name:
        raise ValueError("Either pass --model/--dataset or provide them in --config")
    model_name = normalize_model_name(raw_model_name)
    dataset_name = normalize_dataset_name(raw_dataset_name)

    runner = ExperimentRunner(project_root=PROJECT_ROOT)
    run_root = runner.resolve_run_root(args.run_root)
    run_manager = RunManager(run_root)
    run_paths = runner.create_run(model_name, dataset_name, tag=args.tag, run_root=run_root)

    dataset_kwargs = config_data.get("dataset_kwargs", {})
    config_kwargs = config_data.get("model_config", {}).copy()
    if args.epochs is not None:
        config_kwargs["epochs"] = args.epochs
    base_config = runner.coerce_model_config(model_name, config_kwargs or None)

    model_config = replace(
        base_config,
        checkpoint_path=str(run_paths.checkpoint_dir / "best.pt"),
    )

    resolved_payload = {
        "model": model_name,
        "dataset": dataset_name,
        "dataset_kwargs": dataset_kwargs,
        "model_config": asdict(model_config),
        "run_dir": str(run_paths.run_dir),
    }
    run_manager.save_config(run_paths.config_path, resolved_payload)

    dataset = runner.create_dataset(dataset_name, **dataset_kwargs)
    data = dataset.load()
    model = runner.create_model(model_name, config=model_config)
    bundle = model.prepare_training_bundle(data)

    export_dir = args.export_dir or str(run_paths.export_dir / "native")
    if args.export_dir or args.export_only:
        export_result = model.export_native_artifacts(bundle, export_dir)
        print(f"Exported native artifacts: {export_result}")
        if args.export_only:
            return

    result = model.train(bundle)
    result.run_dir = run_paths.run_dir
    run_manager.save_result(result)
    print(
        f"Finished {result.model_name} on {result.dataset_name}. "
        f"Best epoch={result.best_epoch}, metrics={result.metrics}, run_dir={result.run_dir}"
    )


if __name__ == "__main__":
    main()
