from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_MODELS = (
    "graph-wavenet",
    "agcrn",
    "mtgnn",
    "stid",
    "dgcrn",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick architecture selection on METR-LA with temporary runs cleanup."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Models to compare. Defaults to five newer representative architectures.",
    )
    parser.add_argument("--dataset", default="metr-la", help="Dataset name. Defaults to metr-la.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=6,
        help="Epoch budget for each model. Keep this small for quick architecture screening.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience for quick screening runs.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Training device hint. Falls back to CPU automatically if CUDA is unavailable.",
    )
    parser.add_argument(
        "--report_dir",
        default="reports/architecture_selection",
        help="Directory used to save comparison summaries.",
    )
    parser.add_argument(
        "--temp_run_root",
        default=".tmp_architecture_selection_runs",
        help="Temporary run root used during training before cleanup.",
    )
    parser.add_argument(
        "--keep_runs",
        action="store_true",
        help="Keep temporary run artifacts instead of deleting them after the report is written.",
    )
    return parser.parse_args()


def load_model_setup(model_name: str, dataset_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from spatiotemporal.config_loader import load_json_config

    config_path = PROJECT_ROOT / "configs" / f"{model_name}-{dataset_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    config_data = load_json_config(config_path)
    return config_data.get("dataset_kwargs", {}), config_data.get("model_config", {})


def format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def build_markdown_report(
    dataset_name: str,
    epochs: int,
    patience: int,
    device: str,
    rows: list[dict[str, Any]],
) -> str:
    lines = [
        f"# Architecture Selection Report ({dataset_name})",
        "",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- epochs_per_model: {epochs}",
        f"- patience: {patience}",
        f"- device_hint: {device}",
        "- ranking_rule: sort by best_val_loss, then mae, then rmse",
        "",
        "| rank | model | best_val_loss | mae | rmse | mape | best_epoch | status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    row["model"],
                    format_metric(row.get("best_val_loss", "NA")),
                    format_metric(row.get("mae", "NA")),
                    format_metric(row.get("rmse", "NA")),
                    format_metric(row.get("mape", "NA")),
                    format_metric(row.get("best_epoch", "NA")),
                    row["status"],
                ]
            )
            + " |"
        )
        if row.get("error"):
            lines.append(f"| note | {row['model']} | error | {row['error']} |  |  |  | failed |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    try:
        from spatiotemporal.api import ExperimentRunner
    except ModuleNotFoundError as exc:
        missing_name = exc.name or "dependency"
        raise SystemExit(
            f"Missing required dependency: {missing_name}. "
            "Install training dependencies before running architecture selection."
        ) from exc

    runner = ExperimentRunner(project_root=PROJECT_ROOT)
    temp_run_root = runner.resolve_run_root(args.temp_run_root)
    report_root = runner.resolve_run_root(args.report_dir)
    report_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_base = report_root / f"{args.dataset}-{timestamp}"

    rows: list[dict[str, Any]] = []

    try:
        for model_name in args.models:
            print(
                f"=== [{model_name}] quick screening on {args.dataset} "
                f"(epochs={args.epochs}, patience={args.patience}) ===",
                flush=True,
            )
            row: dict[str, Any] = {"model": model_name, "status": "ok"}
            try:
                dataset_kwargs, model_kwargs = load_model_setup(model_name, args.dataset)
                model_config = runner.coerce_model_config(model_name, model_kwargs)
                model_config = replace(
                    model_config,
                    epochs=args.epochs,
                    patience=args.patience,
                    device=args.device,
                )
                result = runner.train(
                    model_name=model_name,
                    dataset_name=args.dataset,
                    config=model_config,
                    dataset_kwargs=dataset_kwargs,
                    tag="screening",
                    run_root=temp_run_root,
                    save_run=True,
                )
                row.update(
                    {
                        "best_val_loss": result.metrics["best_val_loss"],
                        "mae": result.metrics["mae"],
                        "rmse": result.metrics["rmse"],
                        "mape": result.metrics["mape"],
                        "best_epoch": result.best_epoch,
                        "run_dir": str(result.run_dir) if result.run_dir else None,
                        "model_config": asdict(model_config),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                row["status"] = "failed"
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"[warning] {model_name} failed: {row['error']}", flush=True)
            rows.append(row)
    finally:
        successful_rows = [row for row in rows if row["status"] == "ok"]
        failed_rows = [row for row in rows if row["status"] != "ok"]
        successful_rows.sort(
            key=lambda row: (
                row["best_val_loss"],
                row["mae"],
                row["rmse"],
            )
        )
        ranked_rows = successful_rows + failed_rows

        payload = {
            "dataset": args.dataset,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "epochs": args.epochs,
            "patience": args.patience,
            "device": args.device,
            "cleanup_runs": not args.keep_runs,
            "models": list(args.models),
            "results": ranked_rows,
            "winner": ranked_rows[0]["model"] if ranked_rows and ranked_rows[0]["status"] == "ok" else None,
        }
        json_path = report_base.with_suffix(".json")
        md_path = report_base.with_suffix(".md")
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(
            build_markdown_report(args.dataset, args.epochs, args.patience, args.device, ranked_rows),
            encoding="utf-8",
        )

        print("\n=== Ranking Summary ===", flush=True)
        for index, row in enumerate(ranked_rows, start=1):
            summary = (
                f"{index:02d}. {row['model']}: status={row['status']}"
                f", best_val_loss={format_metric(row.get('best_val_loss', 'NA'))}"
                f", mae={format_metric(row.get('mae', 'NA'))}"
                f", rmse={format_metric(row.get('rmse', 'NA'))}"
            )
            if row.get("error"):
                summary += f", error={row['error']}"
            print(summary, flush=True)
        print(f"\nSaved report: {json_path}", flush=True)
        print(f"Saved report: {md_path}", flush=True)

        if not args.keep_runs and temp_run_root.exists():
            shutil.rmtree(temp_run_root)
            print(f"Removed temporary runs: {temp_run_root}", flush=True)


if __name__ == "__main__":
    main()
