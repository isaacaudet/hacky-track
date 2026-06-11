#!/usr/bin/env python3
"""Summarize detector batch inference coverage and failure modes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parent


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def portable(path: Path | None, base: Path = ROOT) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except (OSError, ValueError):
        return path.name if path.is_absolute() else str(path)


def numeric(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if math.isfinite(value_f) else None


def resolve_path(raw: Any, bases: list[Path]) -> Path:
    path = Path(str(raw or "")).expanduser()
    if path.exists() or path.is_absolute():
        return path
    for base in bases:
        candidate = base / path
        if candidate.exists():
            return candidate
    return ROOT / path


def manifest_paths(*, tracks_root: Path | None, batch_manifest: Path | None) -> list[Path]:
    if batch_manifest is not None:
        doc = read_json(batch_manifest)
        bases = [batch_manifest.parent, batch_manifest.parent.parent, ROOT]
        paths = []
        for run in doc.get("runs", []):
            if str(run.get("status") or "") not in {"completed", "dry_run", ""}:
                continue
            paths.append(resolve_path(run.get("manifest"), bases))
        return [path for path in paths if path.exists()]
    if tracks_root is None:
        return []
    return sorted(tracks_root.glob("*/detector_inference_manifest.json"))


def detector_row(manifest_path: Path) -> dict[str, Any]:
    doc = read_json(manifest_path)
    counts = doc.get("counts") or {}
    params = doc.get("parameters") or {}
    video_info = doc.get("video_info") or {}
    scanned_frames = numeric(video_info.get("scanned_frames")) or numeric(params.get("max_frames")) or 0.0
    raw = int(counts.get("raw_detections") or 0)
    track = int(counts.get("track_points") or 0)
    model_points = int(counts.get("model_detection_points") or 0)
    predicted = int(counts.get("predicted_track_points") or 0)
    return {
        "video": doc.get("video") or manifest_path.parent.name,
        "manifest": portable(manifest_path),
        "source": doc.get("source"),
        "detector": doc.get("detector"),
        "confidence_threshold": params.get("confidence_threshold"),
        "confidence_threshold_source": params.get("confidence_threshold_source"),
        "tracker_mode": params.get("tracker_mode"),
        "scanned_frames": int(scanned_frames),
        "raw_detections": raw,
        "track_points": track,
        "model_detection_points": model_points,
        "predicted_track_points": predicted,
        "raw_per_scanned_frame": None if not scanned_frames else round(raw / scanned_frames, 6),
        "track_coverage": None if not scanned_frames else round(track / scanned_frames, 6),
        "model_coverage": None if not scanned_frames else round(model_points / scanned_frames, 6),
        "prediction_share": None if not track else round(predicted / track, 6),
        "filtered_detections": video_info.get("filtered_detections") or {},
        "status": "ok",
    }


def summarize_numeric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [numeric(row.get(key)) for row in rows]
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": round(min(clean), 6),
        "median": round(float(median(clean)), 6),
        "mean": round(float(mean(clean)), 6),
        "max": round(max(clean), 6),
    }


def coverage_flags(
    rows: list[dict[str, Any]],
    *,
    high_prediction_share: float,
    low_model_coverage: float,
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for row in rows:
        reasons: list[str] = []
        if int(row.get("raw_detections") or 0) == 0:
            reasons.append("no_raw_detections")
        if int(row.get("track_points") or 0) == 0:
            reasons.append("no_track_points")
        prediction_share = numeric(row.get("prediction_share"))
        model_coverage = numeric(row.get("model_coverage"))
        if prediction_share is not None and prediction_share > high_prediction_share:
            reasons.append("high_interpolation_share")
        if model_coverage is not None and model_coverage < low_model_coverage:
            reasons.append("low_model_detection_coverage")
        if reasons:
            flags.append(
                {
                    "video": row["video"],
                    "manifest": row["manifest"],
                    "reasons": reasons,
                    "raw_detections": row["raw_detections"],
                    "track_points": row["track_points"],
                    "model_detection_points": row["model_detection_points"],
                    "predicted_track_points": row["predicted_track_points"],
                    "model_coverage": row["model_coverage"],
                    "prediction_share": row["prediction_share"],
                }
            )
    return flags


def summarize_detector_batch(
    *,
    out_dir: Path,
    tracks_root: Path | None = None,
    batch_manifest: Path | None = None,
    high_prediction_share: float = 0.35,
    low_model_coverage: float = 0.20,
    dry_run: bool = False,
) -> dict[str, Any]:
    paths = manifest_paths(tracks_root=tracks_root, batch_manifest=batch_manifest)
    rows = [detector_row(path) for path in paths]
    flags = coverage_flags(rows, high_prediction_share=high_prediction_share, low_model_coverage=low_model_coverage)
    total_scanned = sum(int(row.get("scanned_frames") or 0) for row in rows)
    total_raw = sum(int(row.get("raw_detections") or 0) for row in rows)
    total_track = sum(int(row.get("track_points") or 0) for row in rows)
    total_model = sum(int(row.get("model_detection_points") or 0) for row in rows)
    total_predicted = sum(int(row.get("predicted_track_points") or 0) for row in rows)
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tracks_root": portable(tracks_root),
        "batch_manifest": portable(batch_manifest),
        "out_dir": portable(out_dir),
        "thresholds": {
            "high_prediction_share": high_prediction_share,
            "low_model_coverage": low_model_coverage,
        },
        "counts": {
            "videos": len(rows),
            "total_scanned_frames": total_scanned,
            "total_raw_detections": total_raw,
            "total_track_points": total_track,
            "total_model_detection_points": total_model,
            "total_predicted_track_points": total_predicted,
            "videos_with_no_raw_detections": sum(1 for row in rows if int(row.get("raw_detections") or 0) == 0),
            "videos_with_no_track_points": sum(1 for row in rows if int(row.get("track_points") or 0) == 0),
            "flagged_videos": len(flags),
        },
        "rates": {
            "raw_per_scanned_frame": None if not total_scanned else round(total_raw / total_scanned, 6),
            "track_coverage": None if not total_scanned else round(total_track / total_scanned, 6),
            "model_coverage": None if not total_scanned else round(total_model / total_scanned, 6),
            "prediction_share": None if not total_track else round(total_predicted / total_track, 6),
        },
        "distributions": {
            "raw_detections": summarize_numeric(rows, "raw_detections"),
            "track_points": summarize_numeric(rows, "track_points"),
            "model_coverage": summarize_numeric(rows, "model_coverage"),
            "prediction_share": summarize_numeric(rows, "prediction_share"),
        },
        "flags": flags,
        "rows": rows,
        "outputs": {
            "summary_json": portable(out_dir / "detector_batch_summary.json"),
            "summary_csv": portable(out_dir / "detector_batch_summary.csv"),
            "summary_md": portable(out_dir / "detector_batch_summary.md"),
        },
    }
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "detector_batch_summary.json", summary)
        write_csv(out_dir / "detector_batch_summary.csv", rows)
        write_markdown(out_dir / "detector_batch_summary.md", summary)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "video",
        "manifest",
        "source",
        "detector",
        "confidence_threshold",
        "confidence_threshold_source",
        "tracker_mode",
        "scanned_frames",
        "raw_detections",
        "track_points",
        "model_detection_points",
        "predicted_track_points",
        "raw_per_scanned_frame",
        "track_coverage",
        "model_coverage",
        "prediction_share",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    counts = summary["counts"]
    rates = summary["rates"]
    lines = [
        "# Detector Batch Summary",
        "",
        f"- Videos: `{counts['videos']}`",
        f"- Total scanned frames: `{counts['total_scanned_frames']}`",
        f"- Total raw detections: `{counts['total_raw_detections']}`",
        f"- Total track points: `{counts['total_track_points']}`",
        f"- Model coverage: `{rates['model_coverage']}`",
        f"- Prediction/interpolation share: `{rates['prediction_share']}`",
        f"- Flagged videos: `{counts['flagged_videos']}`",
        "",
        "## Flagged Videos",
        "",
        "| Video | Reasons | Raw | Track | Model Coverage | Prediction Share |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for flag in summary["flags"]:
        lines.append(
            f"| {flag['video']} | {', '.join(flag['reasons'])} | {flag['raw_detections']} | "
            f"{flag['track_points']} | {flag['model_coverage']} | {flag['prediction_share']} |"
        )
    lines.extend(
        [
            "",
            "## Per-Video Coverage",
            "",
            "| Video | Raw | Track | Model Points | Predicted Points | Model Coverage | Prediction Share |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(summary["rows"], key=lambda item: (numeric(item.get("model_coverage")) or 0.0, str(item.get("video")))):
        lines.append(
            f"| {row['video']} | {row['raw_detections']} | {row['track_points']} | "
            f"{row['model_detection_points']} | {row['predicted_track_points']} | "
            f"{row['model_coverage']} | {row['prediction_share']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize detector batch inference manifests")
    parser.add_argument("--tracks-root", type=Path)
    parser.add_argument("--batch-manifest", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--high-prediction-share", type=float, default=0.35)
    parser.add_argument("--low-model-coverage", type=float, default=0.20)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tracks_root is None and args.batch_manifest is None:
        raise SystemExit("Provide --tracks-root or --batch-manifest.")
    summary = summarize_detector_batch(
        tracks_root=args.tracks_root,
        batch_manifest=args.batch_manifest,
        out_dir=args.out_dir,
        high_prediction_share=args.high_prediction_share,
        low_model_coverage=args.low_model_coverage,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(f"summary: {args.out_dir / 'detector_batch_summary.json'}")
    print(json.dumps({"counts": summary["counts"], "rates": summary["rates"]}, indent=2))


if __name__ == "__main__":
    main()
