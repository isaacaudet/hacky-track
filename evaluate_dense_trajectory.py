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
VISIBLE_STATES = {"visible", "partially_occluded"}
NO_TARGET_STATES = {"fully_occluded", "out_of_frame"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def portable(path: Path | None, base: Path = ROOT) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return path.name


def numeric(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if math.isfinite(value_f) else None


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def normalize_prediction(row: dict[str, Any]) -> dict[str, Any] | None:
    x = numeric(row.get("x"))
    y = numeric(row.get("y"))
    if (x is None or y is None) and row.get("center") is not None:
        center = row.get("center")
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            x = numeric(center[0])
            y = numeric(center[1])
    if x is None or y is None:
        return None
    frame_index = row.get("frame_index")
    time_sec = numeric(row.get("time_sec"))
    return {
        "clip_id": row.get("clip_id"),
        "source_video": row.get("source_video") or row.get("video"),
        "frame_index": None if frame_index is None else int(frame_index),
        "time_sec": time_sec,
        "x": x,
        "y": y,
        "confidence": numeric(row.get("confidence")) if row.get("confidence") is not None else 1.0,
        "visibility": row.get("visibility"),
        "source": row.get("source", row.get("track_source", "prediction")),
    }


def predictions_from_jsonl(predictions_jsonl: Path | None) -> list[dict[str, Any]]:
    return [item for row in load_jsonl(predictions_jsonl) if (item := normalize_prediction(row)) is not None]


def predictions_from_tracks_root(tracks_root: Path | None) -> list[dict[str, Any]]:
    if tracks_root is None or not tracks_root.exists():
        return []
    predictions: list[dict[str, Any]] = []
    for track_path in sorted(tracks_root.glob("*/detector_track.json")):
        track_doc = read_json(track_path)
        video = track_doc.get("video") or track_path.parent.name
        for point in track_doc.get("track", []):
            center = point.get("center") or [None, None]
            x = numeric(center[0])
            y = numeric(center[1])
            if x is None or y is None:
                continue
            predictions.append(
                {
                    "clip_id": point.get("clip_id"),
                    "source_video": video,
                    "frame_index": int(point["frame_index"]) if point.get("frame_index") is not None else None,
                    "time_sec": numeric(point.get("time_sec")),
                    "x": x,
                    "y": y,
                    "confidence": numeric(point.get("confidence")) if point.get("confidence") is not None else 1.0,
                    "visibility": None,
                    "source": point.get("source", "track"),
                }
            )
    return predictions


def prediction_keys(prediction: dict[str, Any]) -> list[tuple[str, Any]]:
    keys: list[tuple[str, Any]] = []
    frame_index = prediction.get("frame_index")
    if frame_index is not None:
        if prediction.get("clip_id"):
            keys.append((f"clip:{prediction['clip_id']}", frame_index))
        if prediction.get("source_video"):
            keys.append((f"video:{Path(str(prediction['source_video'])).name}", frame_index))
            keys.append((f"video:{Path(str(prediction['source_video'])).stem}", frame_index))
    return keys


def build_prediction_indexes(predictions: list[dict[str, Any]]) -> tuple[dict[tuple[str, Any], list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    by_key: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        for key in prediction_keys(prediction):
            by_key[key].append(prediction)
        if prediction.get("source_video"):
            by_video[Path(str(prediction["source_video"])).name].append(prediction)
            by_video[Path(str(prediction["source_video"])).stem].append(prediction)
    return by_key, by_video


def candidate_predictions(
    label: dict[str, Any],
    by_key: dict[tuple[str, Any], list[dict[str, Any]]],
    by_video: dict[str, list[dict[str, Any]]],
    *,
    max_time_delta_sec: float,
) -> list[dict[str, Any]]:
    frame_index = label.get("frame_index")
    candidates: list[dict[str, Any]] = []
    if frame_index is not None:
        if label.get("clip_id"):
            candidates.extend(by_key.get((f"clip:{label['clip_id']}", int(frame_index)), []))
        video = str(label.get("source_video") or "")
        if video:
            candidates.extend(by_key.get((f"video:{Path(video).name}", int(frame_index)), []))
            candidates.extend(by_key.get((f"video:{Path(video).stem}", int(frame_index)), []))
    if candidates:
        return dedupe_predictions(candidates)
    time_sec = numeric(label.get("time_sec"))
    if time_sec is None:
        return []
    video = str(label.get("source_video") or "")
    pool = by_video.get(Path(video).name, []) + by_video.get(Path(video).stem, [])
    return dedupe_predictions(
        [
            prediction
            for prediction in pool
            if prediction.get("time_sec") is not None and abs(float(prediction["time_sec"]) - time_sec) <= max_time_delta_sec
        ]
    )


def dedupe_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for prediction in predictions:
        key = (prediction.get("source_video"), prediction.get("clip_id"), prediction.get("frame_index"), prediction.get("time_sec"), prediction.get("x"), prediction.get("y"))
        if key in seen:
            continue
        seen.add(key)
        out.append(prediction)
    return out


def evaluate_label(
    label: dict[str, Any],
    by_key: dict[tuple[str, Any], list[dict[str, Any]]],
    by_video: dict[str, list[dict[str, Any]]],
    *,
    tolerance_px: float,
    radius_multiplier: float,
    confidence_threshold: float,
    max_time_delta_sec: float,
) -> dict[str, Any]:
    visibility = str(label.get("visibility") or "unlabeled")
    quality = str(label.get("quality") or "pending")
    row = {
        "clip_id": label.get("clip_id"),
        "source_video": label.get("source_video"),
        "split": label.get("split", "unknown"),
        "training_use": label.get("training_use", "train_or_calibration"),
        "frame_index": label.get("frame_index"),
        "time_sec": label.get("time_sec"),
        "visibility": visibility,
        "quality": quality,
        "expected_x": label.get("x"),
        "expected_y": label.get("y"),
        "prediction_x": None,
        "prediction_y": None,
        "confidence": None,
        "center_error_px": None,
        "threshold_px": None,
        "result": "ignored",
    }
    if quality != "reviewed" or visibility not in (VISIBLE_STATES | NO_TARGET_STATES):
        return row
    candidates = candidate_predictions(label, by_key, by_video, max_time_delta_sec=max_time_delta_sec)
    confident = [item for item in candidates if (numeric(item.get("confidence")) or 0.0) >= confidence_threshold]
    if visibility in NO_TARGET_STATES:
        if confident:
            best = max(confident, key=lambda item: numeric(item.get("confidence")) or 0.0)
            row.update(
                {
                    "prediction_x": round(float(best["x"]), 3),
                    "prediction_y": round(float(best["y"]), 3),
                    "confidence": round(float(best.get("confidence") or 0.0), 6),
                    "result": "false_positive_no_target",
                }
            )
        else:
            row["result"] = "pass_no_target"
        return row

    expected_x = numeric(label.get("x"))
    expected_y = numeric(label.get("y"))
    if expected_x is None or expected_y is None:
        row["result"] = "missing_label_center"
        return row
    if not confident:
        row["result"] = "missing_prediction"
        row["threshold_px"] = round(max(tolerance_px, (numeric(label.get("radius")) or 0.0) * radius_multiplier), 3)
        return row
    expected = (expected_x, expected_y)
    best = min(confident, key=lambda item: distance(expected, (float(item["x"]), float(item["y"]))))
    error = distance(expected, (float(best["x"]), float(best["y"])))
    threshold = max(tolerance_px, (numeric(label.get("radius")) or 0.0) * radius_multiplier)
    row.update(
        {
            "prediction_x": round(float(best["x"]), 3),
            "prediction_y": round(float(best["y"]), 3),
            "confidence": round(float(best.get("confidence") or 0.0), 6),
            "center_error_px": round(error, 3),
            "threshold_px": round(threshold, 3),
            "result": "pass" if error <= threshold else "fail",
        }
    )
    return row


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    visible = [row for row in rows if row["visibility"] in VISIBLE_STATES and row["quality"] == "reviewed"]
    no_target = [row for row in rows if row["visibility"] in NO_TARGET_STATES and row["quality"] == "reviewed"]
    errors = [float(row["center_error_px"]) for row in visible if row.get("center_error_px") is not None]
    passed = sum(1 for row in visible if row["result"] == "pass")
    missing = sum(1 for row in visible if row["result"] == "missing_prediction")
    failed = sum(1 for row in visible if row["result"] == "fail")
    false_positive_no_target = sum(1 for row in no_target if row["result"] == "false_positive_no_target")
    return {
        "visible_frames": len(visible),
        "passed_visible_frames": passed,
        "failed_visible_frames": failed,
        "missing_visible_frames": missing,
        "visible_pass_rate": None if not visible else round(passed / len(visible), 6),
        "visible_missing_rate": None if not visible else round(missing / len(visible), 6),
        "mean_center_error_px": None if not errors else round(mean(errors), 3),
        "p95_center_error_px": None if not errors else round(float(percentile(errors, 0.95)), 3),
        "no_target_frames": len(no_target),
        "false_positive_no_target_frames": false_positive_no_target,
        "no_target_false_positive_rate": None if not no_target else round(false_positive_no_target / len(no_target), 6),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "clip_id",
        "source_video",
        "split",
        "training_use",
        "frame_index",
        "time_sec",
        "visibility",
        "quality",
        "expected_x",
        "expected_y",
        "prediction_x",
        "prediction_y",
        "confidence",
        "center_error_px",
        "threshold_px",
        "result",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    overall = summary["overall"]
    lines = [
        "# Dense Trajectory Evaluation",
        "",
        f"- Visible frames: `{overall['visible_frames']}`",
        f"- Visible pass rate: `{overall['visible_pass_rate']}`",
        f"- Visible missing rate: `{overall['visible_missing_rate']}`",
        f"- Mean center error px: `{overall['mean_center_error_px']}`",
        f"- P95 center error px: `{overall['p95_center_error_px']}`",
        f"- No-target frames: `{overall['no_target_frames']}`",
        f"- No-target false-positive rate: `{overall['no_target_false_positive_rate']}`",
        "",
        "## By Split",
        "",
        "| Split | Visible Frames | Pass Rate | Missing Rate | Mean Error Px | No-target Frames | No-target FP Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, metrics in sorted(summary["by_split"].items()):
        lines.append(
            f"| {split} | {metrics['visible_frames']} | {metrics['visible_pass_rate']} | "
            f"{metrics['visible_missing_rate']} | {metrics['mean_center_error_px']} | "
            f"{metrics['no_target_frames']} | {metrics['no_target_false_positive_rate']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_dense_trajectory(
    *,
    labels_jsonl: Path,
    out_dir: Path,
    predictions_jsonl: Path | None = None,
    tracks_root: Path | None = None,
    tolerance_px: float = 12.0,
    radius_multiplier: float = 1.5,
    confidence_threshold: float = 0.01,
    max_time_delta_sec: float = 0.04,
    dry_run: bool = False,
) -> dict[str, Any]:
    labels = load_jsonl(labels_jsonl)
    predictions = predictions_from_jsonl(predictions_jsonl) + predictions_from_tracks_root(tracks_root)
    by_key, by_video = build_prediction_indexes(predictions)
    rows = [
        evaluate_label(
            label,
            by_key,
            by_video,
            tolerance_px=tolerance_px,
            radius_multiplier=radius_multiplier,
            confidence_threshold=confidence_threshold,
            max_time_delta_sec=max_time_delta_sec,
        )
        for label in labels
    ]
    by_split: dict[str, dict[str, Any]] = {}
    for split in sorted({str(row.get("split") or "unknown") for row in rows}):
        by_split[split] = summarize_rows([row for row in rows if str(row.get("split") or "unknown") == split])
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "labels_jsonl": portable(labels_jsonl),
        "predictions_jsonl": portable(predictions_jsonl),
        "tracks_root": portable(tracks_root),
        "out_dir": portable(out_dir),
        "parameters": {
            "tolerance_px": tolerance_px,
            "radius_multiplier": radius_multiplier,
            "confidence_threshold": confidence_threshold,
            "max_time_delta_sec": max_time_delta_sec,
        },
        "overall": summarize_rows(rows),
        "by_split": by_split,
        "rows": rows,
        "outputs": {},
    }
    if dry_run:
        return summary
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_json = out_dir / "dense_trajectory_metrics.json"
    metrics_csv = out_dir / "dense_trajectory_metrics.csv"
    metrics_md = out_dir / "dense_trajectory_metrics.md"
    summary["outputs"] = {
        "metrics_json": portable(metrics_json),
        "metrics_csv": portable(metrics_csv),
        "metrics_md": portable(metrics_md),
    }
    write_json(metrics_json, summary)
    write_csv(metrics_csv, rows)
    write_markdown(metrics_md, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dense trajectory labels against predictions or detector tracks")
    parser.add_argument("--labels-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path)
    parser.add_argument("--tracks-root", type=Path)
    parser.add_argument("--tolerance-px", type=float, default=12.0)
    parser.add_argument("--radius-multiplier", type=float, default=1.5)
    parser.add_argument("--confidence-threshold", type=float, default=0.01)
    parser.add_argument("--max-time-delta-sec", type=float, default=0.04)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_dense_trajectory(
        labels_jsonl=args.labels_jsonl,
        out_dir=args.out_dir,
        predictions_jsonl=args.predictions_jsonl,
        tracks_root=args.tracks_root,
        tolerance_px=args.tolerance_px,
        radius_multiplier=args.radius_multiplier,
        confidence_threshold=args.confidence_threshold,
        max_time_delta_sec=args.max_time_delta_sec,
        dry_run=args.dry_run,
    )
    print(f"metrics: {args.out_dir / 'dense_trajectory_metrics.json'}")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
