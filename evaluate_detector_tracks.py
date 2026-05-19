#!/usr/bin/env python3
"""Evaluate detector ball tracks against reviewed detector labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
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


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def resolve_dataset_file(dataset: Path | None, explicit: Path | None, name: str) -> Path | None:
    if explicit is not None:
        return explicit
    if dataset is None:
        return None
    candidate = dataset / name
    return candidate if candidate.exists() else None


def video_stem(label: dict[str, Any]) -> str:
    return Path(str(label.get("source_video") or label.get("video") or "")).stem


def track_path_for_label(label: dict[str, Any], tracks_root: Path) -> Path:
    return tracks_root / video_stem(label) / "detector_track.json"


def track_time(point: dict[str, Any]) -> float | None:
    value = point.get("time_sec")
    return None if value is None else float(value)


def nearest_track_point(track: list[dict[str, Any]], time_sec: float, max_time_delta_sec: float) -> tuple[dict[str, Any] | None, float | None]:
    timed = [(point, track_time(point)) for point in track]
    timed = [(point, time_value) for point, time_value in timed if time_value is not None]
    if not timed:
        return None, None
    point, nearest_time = min(timed, key=lambda item: abs(float(item[1]) - time_sec))
    delta = abs(float(nearest_time) - time_sec)
    if delta > max_time_delta_sec:
        return None, delta
    return point, delta


def center_error(label: dict[str, Any], point: dict[str, Any]) -> float:
    center = point.get("center") or [None, None]
    return math.hypot(float(label["x"]) - float(center[0]), float(label["y"]) - float(center[1]))


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return ordered[idx]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [row for row in rows if row["kind"] == "positive"]
    hard_negatives = [row for row in rows if row["kind"] == "hard_negative"]
    errors = [float(row["center_error_px"]) for row in labels if row.get("center_error_px") is not None]
    false_positive_hard_negatives = sum(1 for row in hard_negatives if row.get("hard_negative_result") == "false_positive_near_bad_point")
    return {
        "positive_labels": len(labels),
        "matched_positive_labels": sum(1 for row in labels if row["result"] in {"pass", "fail"}),
        "passed_positive_labels": sum(1 for row in labels if row["result"] == "pass"),
        "failed_positive_labels": sum(1 for row in labels if row["result"] == "fail"),
        "missing_track_files": sum(1 for row in labels if row["result"] == "missing_track_file"),
        "missing_track_points": sum(1 for row in labels if row["result"] == "missing_track_point"),
        "pass_rate": None if not labels else round(sum(1 for row in labels if row["result"] == "pass") / len(labels), 6),
        "mean_center_error_px": None if not errors else round(mean(errors), 3),
        "p95_center_error_px": None if not errors else round(float(percentile(errors, 0.95)), 3),
        "hard_negative_points": len(hard_negatives),
        "false_positive_hard_negatives": false_positive_hard_negatives,
        "hard_negative_false_positive_rate": None if not hard_negatives else round(false_positive_hard_negatives / len(hard_negatives), 6),
    }


def evaluate_label(
    label: dict[str, Any],
    track_cache: dict[Path, dict[str, Any] | None],
    tracks_root: Path,
    *,
    max_time_delta_sec: float,
    tolerance_px: float,
    radius_multiplier: float,
    hard_negative: bool = False,
) -> dict[str, Any]:
    track_path = track_path_for_label(label, tracks_root)
    if track_path not in track_cache:
        track_cache[track_path] = read_json(track_path) if track_path.exists() else None
    track_doc = track_cache[track_path]
    row = {
        "kind": "hard_negative" if hard_negative else "positive",
        "item_id": label.get("item_id"),
        "split": label.get("split", "unknown"),
        "video": label.get("source_video") or label.get("video"),
        "time_sec": label.get("time_sec"),
        "expected_x": label.get("x"),
        "expected_y": label.get("y"),
        "radius": label.get("radius"),
        "track_json": portable(track_path),
        "track_frame_index": None,
        "track_time_sec": None,
        "time_delta_sec": None,
        "track_source": None,
        "confidence": None,
        "track_x": None,
        "track_y": None,
        "center_error_px": None,
        "threshold_px": None,
        "result": None,
        "hard_negative_result": None,
        "uncertainty_reasons": [],
    }
    if track_doc is None:
        row["result"] = "missing_track_file"
        if hard_negative:
            row["hard_negative_result"] = "not_evaluated_missing_track_file"
        return row
    time_sec = label.get("time_sec")
    if time_sec is None:
        row["result"] = "missing_label_time"
        if hard_negative:
            row["hard_negative_result"] = "not_evaluated_missing_label_time"
        return row
    point, delta = nearest_track_point(list(track_doc.get("track", [])), float(time_sec), max_time_delta_sec)
    if point is None:
        row["result"] = "missing_track_point"
        row["time_delta_sec"] = None if delta is None else round(float(delta), 6)
        if hard_negative:
            row["hard_negative_result"] = "pass_no_prediction_near_bad_point"
        return row

    error = center_error(label, point)
    threshold = max(tolerance_px, float(label.get("radius") or 0.0) * radius_multiplier)
    center = point.get("center") or [None, None]
    row.update(
        {
            "track_frame_index": point.get("frame_index"),
            "track_time_sec": point.get("time_sec"),
            "time_delta_sec": None if delta is None else round(float(delta), 6),
            "track_source": point.get("source"),
            "confidence": point.get("confidence"),
            "track_x": center[0],
            "track_y": center[1],
            "center_error_px": round(error, 3),
            "threshold_px": round(threshold, 3),
            "uncertainty_reasons": point.get("uncertainty_reasons", []),
        }
    )
    if hard_negative:
        row["result"] = "not_applicable"
        row["hard_negative_result"] = "false_positive_near_bad_point" if error <= threshold else "pass_prediction_away_from_bad_point"
    else:
        row["result"] = "pass" if error <= threshold else "fail"
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "kind",
        "item_id",
        "split",
        "video",
        "time_sec",
        "expected_x",
        "expected_y",
        "track_x",
        "track_y",
        "center_error_px",
        "threshold_px",
        "result",
        "hard_negative_result",
        "confidence",
        "track_source",
        "uncertainty_reasons",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field) for field in fields}
            out["uncertainty_reasons"] = ";".join(row.get("uncertainty_reasons") or [])
            writer.writerow(out)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    overall = summary["overall"]
    lines = [
        "# Detector Track Evaluation",
        "",
        f"- Positive labels: `{overall['positive_labels']}`",
        f"- Pass rate: `{overall['pass_rate']}`",
        f"- Mean center error px: `{overall['mean_center_error_px']}`",
        f"- P95 center error px: `{overall['p95_center_error_px']}`",
        f"- Hard-negative points: `{overall['hard_negative_points']}`",
        f"- Hard-negative false-positive rate: `{overall['hard_negative_false_positive_rate']}`",
        "",
        "## By Split",
        "",
        "| Split | Labels | Pass Rate | Mean Error Px | Hard Negatives | Hard-Neg FP Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, metrics in sorted(summary["by_split"].items()):
        lines.append(
            f"| {split} | {metrics['positive_labels']} | {metrics['pass_rate']} | "
            f"{metrics['mean_center_error_px']} | {metrics['hard_negative_points']} | "
            f"{metrics['hard_negative_false_positive_rate']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_detector_tracks(
    *,
    labels_jsonl: Path,
    tracks_root: Path,
    out_dir: Path,
    hard_negatives_jsonl: Path | None = None,
    max_time_delta_sec: float = 0.08,
    tolerance_px: float = 24.0,
    radius_multiplier: float = 1.5,
) -> dict[str, Any]:
    positive_labels = load_jsonl(labels_jsonl)
    hard_negatives = load_jsonl(hard_negatives_jsonl)
    track_cache: dict[Path, dict[str, Any] | None] = {}
    rows = [
        evaluate_label(
            label,
            track_cache,
            tracks_root,
            max_time_delta_sec=max_time_delta_sec,
            tolerance_px=tolerance_px,
            radius_multiplier=radius_multiplier,
            hard_negative=False,
        )
        for label in positive_labels
    ]
    rows.extend(
        evaluate_label(
            label,
            track_cache,
            tracks_root,
            max_time_delta_sec=max_time_delta_sec,
            tolerance_px=tolerance_px,
            radius_multiplier=radius_multiplier,
            hard_negative=True,
        )
        for label in hard_negatives
    )
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[str(row.get("split") or "unknown")].append(row)
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "labels_jsonl": portable(labels_jsonl),
        "hard_negatives_jsonl": portable(hard_negatives_jsonl),
        "tracks_root": portable(tracks_root),
        "parameters": {
            "max_time_delta_sec": max_time_delta_sec,
            "tolerance_px": tolerance_px,
            "radius_multiplier": radius_multiplier,
        },
        "overall": summarize_rows(rows),
        "by_split": {split: summarize_rows(split_rows) for split, split_rows in by_split.items()},
        "rows": rows,
        "outputs": {
            "metrics_json": portable(out_dir / "detector_track_metrics.json"),
            "metrics_csv": portable(out_dir / "detector_track_metrics.csv"),
            "metrics_md": portable(out_dir / "detector_track_metrics.md"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "detector_track_metrics.json", summary)
    write_csv(out_dir / "detector_track_metrics.csv", rows)
    write_markdown(out_dir / "detector_track_metrics.md", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detector tracks against reviewed footbag labels")
    parser.add_argument("--dataset", type=Path, help="Detector dataset directory containing reviewed_detector_labels.jsonl")
    parser.add_argument("--labels-jsonl", type=Path)
    parser.add_argument("--hard-negatives-jsonl", type=Path)
    parser.add_argument("--tracks-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-time-delta-sec", type=float, default=0.08)
    parser.add_argument("--tolerance-px", type=float, default=24.0)
    parser.add_argument("--radius-multiplier", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_jsonl = resolve_dataset_file(args.dataset, args.labels_jsonl, "reviewed_detector_labels.jsonl")
    hard_negatives_jsonl = resolve_dataset_file(args.dataset, args.hard_negatives_jsonl, "hard_negatives/points.jsonl")
    if labels_jsonl is None:
        raise SystemExit("Provide --dataset or --labels-jsonl")
    summary = evaluate_detector_tracks(
        labels_jsonl=labels_jsonl,
        hard_negatives_jsonl=hard_negatives_jsonl,
        tracks_root=args.tracks_root,
        out_dir=args.out_dir,
        max_time_delta_sec=args.max_time_delta_sec,
        tolerance_px=args.tolerance_px,
        radius_multiplier=args.radius_multiplier,
    )
    print(f"metrics: {args.out_dir / 'detector_track_metrics.json'}")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
