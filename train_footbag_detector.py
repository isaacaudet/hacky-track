#!/usr/bin/env python3
"""Train a custom footbag detector from exported reviewed labels."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def portable(path: Path, base: Path = ROOT) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except (OSError, ValueError):
        return path.name if path.is_absolute() else str(path)


def resolve_data_yaml(dataset: Path) -> Path:
    if dataset.is_dir():
        return dataset / "data.yaml"
    return dataset


def dataset_manifest_for(data_yaml: Path) -> Path | None:
    candidate = data_yaml.parent / "manifest.json"
    return candidate if candidate.exists() else None


def latest_best_weight(project_dir: Path, name: str) -> Path | None:
    candidates = [
        project_dir / name / "weights" / "best.pt",
        project_dir / name / "weights" / "last.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    weights = sorted(project_dir.glob("**/weights/best.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    return weights[0] if weights else None


def normalized_output_dir(out_dir: Path) -> Path:
    return out_dir.expanduser().resolve()


def write_runtime_data_yaml(data_yaml: Path, out_dir: Path) -> Path:
    runtime = out_dir / "runtime_data.yaml"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(
        "\n".join(
            [
                f"path: {data_yaml.parent.resolve()}",
                "train: images/train",
                "val: images/validation",
                "test: images/test",
                "names:",
                "  0: footbag",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return runtime


def parse_results_csv(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return None
    last = {str(key).strip(): value for key, value in rows[-1].items()}
    metrics: dict[str, Any] = {}
    for key, value in last.items():
        clean_key = key.strip()
        try:
            metrics[clean_key] = float(value)
        except (TypeError, ValueError):
            metrics[clean_key] = value
    return metrics


def parse_best_results_csv(path: Path, primary_metric: str = "metrics/mAP50(B)") -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [{str(key).strip(): value for key, value in row.items()} for row in reader]
    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        parsed: dict[str, Any] = {}
        for key, value in row.items():
            clean_key = key.strip()
            try:
                parsed[clean_key] = float(value)
            except (TypeError, ValueError):
                parsed[clean_key] = value
        parsed_rows.append(parsed)
    scored = [row for row in parsed_rows if isinstance(row.get(primary_metric), float)]
    if not scored:
        return parsed_rows[-1] if parsed_rows else None
    return max(scored, key=lambda row: float(row[primary_metric]))


def training_report_for(results_dir: Path, note: str) -> dict[str, Any]:
    results_csv = results_dir / "results.csv"
    return {
        "results_dir": portable(results_dir),
        "results_csv": portable(results_csv) if results_csv.exists() else None,
        "final_metrics": parse_results_csv(results_csv),
        "best_metrics": parse_best_results_csv(results_csv),
        "best_metric": "metrics/mAP50(B)",
        "note": note,
    }


def build_summary(
    *,
    data_yaml: Path,
    out_dir: Path,
    base_model: str,
    epochs: int,
    imgsz: int,
    batch: str,
    device: str | None,
    run_name: str,
    dry_run: bool,
) -> dict[str, Any]:
    manifest_path = dataset_manifest_for(data_yaml)
    dataset_manifest = read_json(manifest_path) if manifest_path else None
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trainer": "ultralytics-yolo",
        "dry_run": dry_run,
        "data_yaml": portable(data_yaml),
        "runtime_data_yaml": None,
        "dataset_manifest": portable(manifest_path) if manifest_path else None,
        "dataset_counts": None
        if not dataset_manifest
        else {
            "positive_labels": dataset_manifest.get("positive_labels"),
            "hard_negative_points": dataset_manifest.get("hard_negative_points"),
            "written_images": dataset_manifest.get("written_images"),
            "written_hard_negative_yolo_images": dataset_manifest.get("written_hard_negative_yolo_images"),
            "splits": dataset_manifest.get("splits"),
        },
        "out_dir": portable(out_dir),
        "base_model": base_model,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "run_name": run_name,
        "model_path": None,
        "training_report": None,
    }


def finalize_existing_training_run(
    *,
    data_yaml: Path,
    out_dir: Path,
    base_model: str,
    epochs: int,
    imgsz: int,
    batch: str,
    device: str | None,
    run_name: str,
    recovered_existing_run: bool = False,
) -> dict[str, Any]:
    summary = build_summary(
        data_yaml=data_yaml,
        out_dir=out_dir,
        base_model=base_model,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        run_name=run_name,
        dry_run=False,
    )
    runtime_data_yaml = write_runtime_data_yaml(data_yaml, out_dir)
    summary["runtime_data_yaml"] = portable(runtime_data_yaml)
    weight = latest_best_weight(out_dir, run_name)
    if weight is None:
        raise FileNotFoundError(f"No Ultralytics weights found under {out_dir / run_name}")
    final_weight = out_dir / "footbag_detector_best.pt"
    shutil.copy2(weight, final_weight)
    results_dir = weight.parents[1]
    summary["model_path"] = portable(final_weight)
    summary["source_weight"] = portable(weight)
    summary["recovered_existing_run"] = recovered_existing_run
    summary["training_report"] = training_report_for(
        results_dir,
        "Recovered from an existing Ultralytics run." if recovered_existing_run else "Use Ultralytics results.csv and plots in the run directory for detailed training metrics.",
    )
    write_json(out_dir / "training_manifest.json", summary)
    return summary


def train_detector(
    dataset: Path,
    out_dir: Path,
    *,
    base_model: str = "yolo11n.pt",
    epochs: int = 80,
    imgsz: int = 640,
    batch: str = "-1",
    device: str | None = None,
    run_name: str = "footbag-detector",
    dry_run: bool = False,
    recover_existing: bool = False,
) -> dict[str, Any]:
    out_dir = normalized_output_dir(out_dir)
    data_yaml = resolve_data_yaml(dataset).expanduser().resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"Detector data YAML not found: {data_yaml}")
    summary = build_summary(
        data_yaml=data_yaml,
        out_dir=out_dir,
        base_model=base_model,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        run_name=run_name,
        dry_run=dry_run,
    )
    if dry_run:
        return summary
    if recover_existing:
        return finalize_existing_training_run(
            data_yaml=data_yaml,
            out_dir=out_dir,
            base_model=base_model,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            run_name=run_name,
            recovered_existing_run=True,
        )

    try:
        from ultralytics import YOLO  # type: ignore
        import ultralytics  # type: ignore
    except ModuleNotFoundError as exc:
        summary["error"] = {
            "type": "missing_dependency",
            "message": "Install detector dependencies with `pip install -r requirements-detector.txt`.",
            "missing_module": exc.name,
        }
        write_json(out_dir / "training_manifest.json", summary)
        raise RuntimeError(summary["error"]["message"]) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    summary["ultralytics_version"] = getattr(ultralytics, "__version__", None)
    runtime_data_yaml = write_runtime_data_yaml(data_yaml, out_dir)
    summary["runtime_data_yaml"] = portable(runtime_data_yaml)
    model = YOLO(base_model)
    train_kwargs: dict[str, Any] = {
        "data": str(runtime_data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": int(batch) if str(batch).lstrip("-").isdigit() else batch,
        "project": str(out_dir),
        "name": run_name,
        "exist_ok": True,
    }
    if device:
        train_kwargs["device"] = device
    results = model.train(**train_kwargs)
    results_dir = Path(getattr(results, "save_dir", out_dir / run_name))
    weight = latest_best_weight(out_dir, run_name) or latest_best_weight(results_dir.parent, results_dir.name)
    if weight is not None:
        final_weight = out_dir / "footbag_detector_best.pt"
        shutil.copy2(weight, final_weight)
        summary["model_path"] = portable(final_weight)
        summary["source_weight"] = portable(weight)
    summary["training_report"] = training_report_for(
        results_dir,
        "Use Ultralytics results.csv and plots in the run directory for detailed training metrics.",
    )
    write_json(out_dir / "training_manifest.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO footbag detector from exported Hacky Track labels")
    parser.add_argument("--dataset", type=Path, required=True, help="Detector dataset directory or data.yaml")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", default="-1")
    parser.add_argument("--device")
    parser.add_argument("--run-name", default="footbag-detector")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--recover-existing", action="store_true", help="Write the public manifest/model copy from an existing Ultralytics run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        summary = train_detector(
            args.dataset,
            args.out_dir,
            base_model=args.base_model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            run_name=args.run_name,
            dry_run=args.dry_run,
            recover_existing=args.recover_existing,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if args.dry_run:
        print(json.dumps(summary, indent=2))
    else:
        print(f"training manifest: {args.out_dir / 'training_manifest.json'}")
        if summary.get("model_path"):
            print(f"model: {summary['model_path']}")


if __name__ == "__main__":
    main()
