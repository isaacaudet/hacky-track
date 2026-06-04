#!/usr/bin/env python3
"""Evaluate release-path stall/drop handling from existing reviewed labels.

This is separate from the contact side/surface classifier. Touch timing is
already a release path; stall and drop/floor reset need their own gate because
they are event decisions with different failure modes.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import joblib
import numpy as np

from attach_touch_l2_features import TrackFeatures, TrackPoint, attach_row_features, compute_track_features
from event_error_audit import load_video_paths, read_frame, resize_letterbox


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "runs/release-27-public/touch_corpus_v1"
DEFAULT_REVIEW_LABELS = ROOT / "outputs/review_validation/reviewed_labels.jsonl"
DEFAULT_OUT_DIR = DEFAULT_CORPUS / "release_stall_drop_classifier_v1"
DEFAULT_REVIEW_MANIFEST = ROOT / "runs/release-27-public/qa_sidefix_v7/qa_manifest.json"
DEFAULT_VIDEO_SEARCH_DIR = Path.home() / "Downloads"
DEFAULT_DETECTIONS_JSONL = [
    DEFAULT_CORPUS / "owlv2_touch_detections_v1/detections.jsonl",
    DEFAULT_CORPUS / "owlv2_touch_detections_contact_missing_v1/detections.jsonl",
]
DEFAULT_TOUCH_EVENTS_JSONL = [
    DEFAULT_CORPUS / "touch_classifier_v1/touch_classifier_frozen_events.jsonl",
    DEFAULT_CORPUS / "touch_classifier_v1/touch_classifier_oof_events.jsonl",
]
DEFAULT_THRESHOLD = 0.2
DEFAULT_BIN_SEC = 0.05
DEFAULT_MIN_EXAMPLES = 20
DEFAULT_MIN_VIDEOS = 3
DEFAULT_MIN_POSITIVE_EXAMPLES = 10
DEFAULT_MIN_NEGATIVE_EXAMPLES = 10
DEFAULT_MAX_TRACK_GAP_SEC = 0.25
DEFAULT_BREAK_TOLERANCE_SEC = 0.08
DROP_GATE_PRECISION = 0.90
DROP_GATE_RECALL = 0.90
STALL_GATE_PRECISION = 0.85
STALL_GATE_RECALL = 0.80

TARGET_KINDS = ("drop_floor", "stall")
STATUS_TO_LABEL = {"approved": 1, "rejected": 0}
NUMERIC_FEATURES = [
    "candidate_track_delta_sec",
    "candidate_track_confidence",
    "candidate_track_x",
    "candidate_track_y",
    "candidate_y_ratio",
    "detector_confidence_near_candidate",
    "event_near_touch_within_025",
    "event_next_touch_gap_sec",
    "event_no_next_touch",
    "event_no_prev_touch",
    "event_post_gap_gt_1s",
    "event_pre_gap_gt_1s",
    "event_prev_touch_gap_sec",
    "event_touch_count_after",
    "event_touch_count_before",
    "event_touch_count_total",
    "event_touch_stream_present",
    "floor_context_score",
    "floor_ground_ratio",
    "floor_sky_ratio",
    "sequence_confidence_mean_window",
    "sequence_gap_after_sec",
    "sequence_gap_before_sec",
    "sequence_longest_gap_sec",
    "sequence_low_screen_ratio_window",
    "sequence_max_speed_px_sec",
    "sequence_mean_speed_px_sec",
    "sequence_post_track_count",
    "sequence_pre_track_count",
    "sequence_track_coverage_ratio",
    "sequence_track_count_window",
    "sequence_y_ratio_max_window",
    "sequence_y_ratio_mean_window",
    "sequence_y_ratio_min_window",
    "trajectory_ax_window",
    "trajectory_ay_window",
    "trajectory_break_support",
    "trajectory_confidence_mean_window",
    "trajectory_gap_after_sec",
    "trajectory_gap_before_sec",
    "trajectory_impulse_score",
    "trajectory_local_y_quad_rms_px",
    "trajectory_max_positive_dvy",
    "trajectory_nearest_break_delta_sec",
    "trajectory_nearest_y_peak_delta_sec",
    "trajectory_nearest_y_trough_delta_sec",
    "trajectory_speed_after",
    "trajectory_speed_before",
    "trajectory_speed_delta",
    "trajectory_track_points_window",
    "trajectory_vx_after",
    "trajectory_vx_before",
    "trajectory_vx_delta",
    "trajectory_vy_after",
    "trajectory_vy_before",
    "trajectory_vy_delta",
    "trajectory_x_range_window_px",
    "trajectory_y_peak_prominence_window_px",
    "trajectory_y_position_pct_window",
    "trajectory_y_range_window_px",
    "trajectory_y_trough_prominence_window_px",
]
BOOLEAN_FEATURES = ["height_reversal"]
CATEGORICAL_FEATURES = ["trajectory_feature_status"]
MODEL_FAMILIES = ("logistic_regression", "extra_trees", "gradient_boosting")
FEATURE_MODES = {
    "l2_only": ("sequence_", "floor_", "candidate_y_ratio", "event_"),
    "sequence_window": ("event_",),
    "rally_sequence": (),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def normalize_status(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    return text if text in STATUS_TO_LABEL else None


def normalize_video_id(value: Any) -> str:
    text = Path(str(value or "")).name
    text = text.removesuffix(".MOV").removesuffix(".mov").removesuffix(".review.json")
    text = text.replace("_singular_display 2", "_singular_display-2")
    return text


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in {None, ""}:
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def canonical_review_row(row: dict[str, Any]) -> dict[str, Any] | None:
    kind = str(row.get("kind") or "")
    if kind not in TARGET_KINDS:
        return None
    status = normalize_status(row.get("status") or row.get("review_status"))
    if status is None:
        return None
    time_sec = safe_float(row.get("time_sec"))
    source_video = row.get("source_video")
    if time_sec is None or not source_video:
        return None
    return {
        "kind": kind,
        "status": status,
        "label": STATUS_TO_LABEL[status],
        "video_id": normalize_video_id(source_video),
        "video_name": Path(str(source_video)).name,
        "source_video": Path(str(source_video)).name,
        "candidate_time_sec": float(time_sec),
        "time_sec": float(time_sec),
        "review_confidence": safe_float(row.get("confidence")),
        "item_id": row.get("item_id"),
        "review_file": row.get("review_file"),
        "note": row.get("note"),
    }


def prepare_review_rows(rows: list[dict[str, Any]], *, bin_sec: float = DEFAULT_BIN_SEC) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    canonical = [item for row in rows if (item := canonical_review_row(row)) is not None]
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in canonical:
        groups[(row["video_id"], row["kind"], int(round(float(row["time_sec"]) / bin_sec)))].append(row)

    cleaned: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    merged_duplicate_rows = 0
    for key, items in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        statuses = {str(item["status"]) for item in items}
        if len(statuses) > 1:
            conflicts.append(
                {
                    "video_id": key[0],
                    "kind": key[1],
                    "time_bin": key[2],
                    "statuses": sorted(statuses),
                    "rows": len(items),
                    "times_sec": [round(float(item["time_sec"]), 6) for item in items],
                    "item_ids": [item.get("item_id") for item in items],
                }
            )
            continue
        if len(items) > 1:
            merged_duplicate_rows += len(items) - 1
        best = max(
            items,
            key=lambda item: (
                -1.0 if item.get("review_confidence") is None else float(item["review_confidence"]),
                -abs(float(item["time_sec"]) - (key[2] * bin_sec)),
            ),
        )
        cleaned.append(best)
    summary = {
        "input_rows": len(rows),
        "canonical_rows": len(canonical),
        "clean_rows": len(cleaned),
        "groups": len(groups),
        "conflict_groups": len(conflicts),
        "excluded_conflict_rows": sum(int(item["rows"]) for item in conflicts),
        "merged_duplicate_rows": merged_duplicate_rows,
        "conflicts": conflicts,
        "counts_by_kind_status": {
            f"{kind}/{status}": count
            for (kind, status), count in sorted(Counter((row["kind"], row["status"]) for row in canonical).items())
        },
        "clean_counts_by_kind_status": {
            f"{kind}/{status}": count
            for (kind, status), count in sorted(Counter((row["kind"], row["status"]) for row in cleaned).items())
        },
        "clean_videos_by_kind": {
            kind: sorted({row["video_id"] for row in cleaned if row["kind"] == kind})
            for kind in TARGET_KINDS
        },
    }
    return cleaned, summary


def top_detection_row(row: dict[str, Any], *, threshold: float) -> dict[str, Any] | None:
    detections = row.get("detections")
    if isinstance(detections, list) and detections:
        top: dict[str, Any] | None = None
        top_score = safe_float(row.get("top_score"))
        first_score = safe_float(detections[0].get("score"))
        if top_score is not None and first_score is not None and abs(first_score - top_score) <= 1e-6:
            top = detections[0]
        else:
            top = max(detections, key=lambda item: safe_float(item.get("score"), 0.0) or 0.0)
        score = safe_float(top.get("score"), 0.0) or 0.0
        if score < threshold:
            return None
        return {
            "video_id": row.get("video_id"),
            "source_video": Path(str(row.get("source_video") or row.get("video_name") or "")).name,
            "frame_index": row.get("frame_index"),
            "time_sec": float(row["time_sec"]),
            "x": float(top["x"]),
            "y": float(top["y"]),
            "confidence": float(score),
        }
    if row.get("x") is not None and row.get("y") is not None:
        score = safe_float(row.get("confidence") or row.get("score") or row.get("top_score"), 1.0) or 0.0
        if score < threshold:
            return None
        return {
            "video_id": row.get("video_id"),
            "source_video": Path(str(row.get("source_video") or row.get("video_name") or "")).name,
            "frame_index": row.get("frame_index"),
            "time_sec": float(row["time_sec"]),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "confidence": float(score),
        }
    return None


def iter_detection_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_compact_track_cache(paths: list[Path], cache_path: Path, *, threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
    deduped: dict[tuple[str, float], dict[str, Any]] = {}
    input_rows = 0
    candidate_points = 0
    for path in paths:
        for row in iter_detection_rows(path):
            input_rows += 1
            point = top_detection_row(row, threshold=threshold)
            if point is None or not point.get("source_video"):
                continue
            candidate_points += 1
            key = (str(point["source_video"]), round(float(point["time_sec"]), 6))
            prior = deduped.get(key)
            if prior is None or float(point["confidence"]) > float(prior["confidence"]):
                deduped[key] = point
    rows = sorted(deduped.values(), key=lambda item: (str(item.get("source_video")), float(item.get("time_sec") or 0.0)))
    write_jsonl(cache_path, rows)
    by_video = Counter(str(row.get("source_video")) for row in rows)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "input_files": [str(path) for path in paths],
        "input_rows": input_rows,
        "candidate_points": candidate_points,
        "kept_points": len(rows),
        "videos": dict(sorted(by_video.items())),
        "cache_path": str(cache_path),
    }
    write_json(cache_path.with_suffix(".manifest.json"), manifest)
    return manifest


def load_compact_tracks(path: Path) -> dict[str, list[TrackPoint]]:
    tracks: dict[str, list[TrackPoint]] = defaultdict(list)
    for row in read_jsonl(path):
        video_name = Path(str(row.get("source_video") or "")).name
        if not video_name:
            continue
        tracks[video_name].append(
            TrackPoint(
                time_sec=float(row["time_sec"]),
                x=float(row["x"]),
                y=float(row["y"]),
                confidence=float(row["confidence"]),
                frame_index=None if row.get("frame_index") is None else int(row["frame_index"]),
            )
        )
    for video_name, points in tracks.items():
        tracks[video_name] = sorted(points, key=lambda item: item.time_sec)
    return dict(tracks)


def load_touch_event_streams(paths: list[Path]) -> dict[str, list[float]]:
    streams: dict[str, set[float]] = defaultdict(set)
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if str(row.get("event_type") or "touch") != "touch":
                continue
            time_sec = safe_float(row.get("time_sec"))
            if time_sec is None:
                continue
            for key in ("video_name", "video_id", "source_video"):
                value = row.get(key)
                if value in {None, ""}:
                    continue
                text = str(value)
                streams[text].add(float(time_sec))
                streams[Path(text).name].add(float(time_sec))
                streams[normalize_video_id(text)].add(float(time_sec))
    return {key: sorted(values) for key, values in streams.items()}


def summarize_touch_streams(streams: dict[str, list[float]]) -> dict[str, Any]:
    canonical = {key: values for key, values in streams.items() if key.startswith("video-") and key.endswith(("singular_display", "singular_display-2"))}
    return {
        "lookup_keys": len(streams),
        "canonical_videos": len(canonical),
        "events_by_video": {key: len(values) for key, values in sorted(canonical.items())},
    }


def nearest_track_point(points: list[TrackPoint], time_sec: float) -> TrackPoint | None:
    if not points:
        return None
    return min(points, key=lambda item: abs(item.time_sec - time_sec))


def add_nearest_track_features(row: dict[str, Any], points: list[TrackPoint]) -> dict[str, Any]:
    out = dict(row)
    point = nearest_track_point(points, float(row["candidate_time_sec"]))
    if point is None:
        out.update(
            {
                "candidate_track_delta_sec": None,
                "candidate_track_confidence": None,
                "candidate_track_x": None,
                "candidate_track_y": None,
            }
        )
    else:
        out.update(
            {
                "candidate_track_delta_sec": round(abs(point.time_sec - float(row["candidate_time_sec"])), 6),
                "candidate_track_confidence": round(float(point.confidence), 6),
                "candidate_track_x": round(float(point.x), 6),
                "candidate_track_y": round(float(point.y), 6),
            }
        )
    return out


def add_touch_event_stream_features(row: dict[str, Any], touch_streams: dict[str, list[float]]) -> dict[str, Any]:
    out = dict(row)
    time_sec = float(row["candidate_time_sec"])
    times = (
        touch_streams.get(str(row.get("video_name")))
        or touch_streams.get(str(row.get("video_id")))
        or touch_streams.get(normalize_video_id(str(row.get("video_name") or "")))
        or []
    )
    prev_times = [value for value in times if value < time_sec]
    next_times = [value for value in times if value > time_sec]
    prev_gap = None if not prev_times else time_sec - prev_times[-1]
    next_gap = None if not next_times else next_times[0] - time_sec
    out.update(
        {
            "event_touch_stream_present": 1.0 if times else 0.0,
            "event_touch_count_total": len(times),
            "event_touch_count_before": len(prev_times),
            "event_touch_count_after": len(next_times),
            "event_prev_touch_gap_sec": None if prev_gap is None else round(float(prev_gap), 6),
            "event_next_touch_gap_sec": None if next_gap is None else round(float(next_gap), 6),
            "event_near_touch_within_025": 1.0 if any(abs(value - time_sec) <= 0.25 for value in times) else 0.0,
            "event_no_next_touch": 1.0 if times and not next_times else 0.0,
            "event_no_prev_touch": 1.0 if times and not prev_times else 0.0,
            "event_post_gap_gt_1s": 1.0 if times and (not next_times or float(next_gap or 0.0) > 1.0) else 0.0,
            "event_pre_gap_gt_1s": 1.0 if prev_gap is not None and float(prev_gap) > 1.0 else 0.0,
        }
    )
    return out


def frame_interval(points: list[TrackPoint]) -> float:
    deltas = [b.time_sec - a.time_sec for a, b in zip(points, points[1:]) if b.time_sec > a.time_sec]
    if not deltas:
        return 1.0 / 30.0
    return float(np.median(np.asarray(deltas, dtype=float)))


def longest_track_gap(points: list[TrackPoint], *, start: float, end: float) -> float | None:
    if not points:
        return None
    times = [start, *[point.time_sec for point in points], end]
    return max(b - a for a, b in zip(times, times[1:]))


def local_floor_context_scores(frame: np.ndarray, x: float, y: float, radius: int = 76) -> tuple[float, float]:
    h, w = frame.shape[:2]
    left = max(0, int(round(x - radius)))
    right = min(w, int(round(x + radius)))
    top = max(0, int(round(y - radius)))
    bottom = min(h, int(round(y + radius)))
    if right <= left or bottom <= top:
        return 0.0, 0.0
    patch = frame[top:bottom, left:right]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    green = (hue >= 28) & (hue <= 92) & (sat >= 22) & (val >= 35)
    tan = (hue >= 8) & (hue <= 34) & (sat >= 18) & (val >= 45)
    dark_ground = (hue >= 18) & (hue <= 105) & (sat >= 12) & (val >= 18) & (val <= 135)
    sky = ((hue >= 88) & (hue <= 128) & (sat >= 25) & (val >= 75)) | ((sat <= 28) & (val >= 170))
    total = max(1, patch.shape[0] * patch.shape[1])
    return float(np.count_nonzero(green | tan | dark_ground) / total), float(np.count_nonzero(sky) / total)


def video_paths_with_download_fallback(review_manifest: Path) -> dict[str, str]:
    video_paths = load_video_paths(review_manifest)
    for video_path in sorted(DEFAULT_VIDEO_SEARCH_DIR.glob("video-*singular_display*.MOV")):
        video_paths.setdefault(normalize_video_id(video_path.name), str(video_path))
    return video_paths


def video_dimensions(video_path: str | None) -> tuple[int | None, int | None]:
    if not video_path:
        return None, None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
    cap.release()
    return width, height


def add_sequence_window_features(
    row: dict[str, Any],
    points: list[TrackPoint],
    *,
    video_path: str | None,
    width: int | None,
    height: int | None,
    window_sec: float = 1.0,
) -> dict[str, Any]:
    out = dict(row)
    time_sec = float(row["candidate_time_sec"])
    start = time_sec - window_sec
    end = time_sec + window_sec
    window_points = [point for point in points if start <= point.time_sec <= end]
    pre_points = [point for point in window_points if point.time_sec < time_sec]
    post_points = [point for point in window_points if point.time_sec > time_sec]
    interval = frame_interval(window_points)
    expected_points = max(1.0, (2.0 * window_sec) / max(interval, 1e-6))
    y_ratios = [point.y / height for point in window_points if height]
    speeds: list[float] = []
    for a, b in zip(window_points, window_points[1:]):
        dt = b.time_sec - a.time_sec
        if dt <= 0:
            continue
        speeds.append(float(math.hypot(b.x - a.x, b.y - a.y) / dt))
    nearest = nearest_track_point(points, time_sec)
    candidate_y_ratio = None
    floor_ground_ratio = None
    floor_sky_ratio = None
    floor_context_score = None
    if nearest is not None and height:
        candidate_y_ratio = float(nearest.y / height)
    if nearest is not None and video_path:
        frame = read_frame(video_path, time_sec)
        if frame is not None:
            floor_ground_ratio, floor_sky_ratio = local_floor_context_scores(frame, nearest.x, nearest.y)
            floor_context_score = float(floor_ground_ratio - floor_sky_ratio)
    before_times = [point.time_sec for point in points if point.time_sec < time_sec]
    after_times = [point.time_sec for point in points if point.time_sec > time_sec]
    out.update(
        {
            "candidate_y_ratio": None if candidate_y_ratio is None else round(candidate_y_ratio, 6),
            "floor_ground_ratio": None if floor_ground_ratio is None else round(floor_ground_ratio, 6),
            "floor_sky_ratio": None if floor_sky_ratio is None else round(floor_sky_ratio, 6),
            "floor_context_score": None if floor_context_score is None else round(floor_context_score, 6),
            "sequence_track_count_window": len(window_points),
            "sequence_pre_track_count": len(pre_points),
            "sequence_post_track_count": len(post_points),
            "sequence_track_coverage_ratio": round(min(1.0, len(window_points) / expected_points), 6),
            "sequence_longest_gap_sec": None if not window_points else round(float(longest_track_gap(window_points, start=start, end=end) or 0.0), 6),
            "sequence_gap_before_sec": None if not before_times else round(float(time_sec - max(before_times)), 6),
            "sequence_gap_after_sec": None if not after_times else round(float(min(after_times) - time_sec), 6),
            "sequence_y_ratio_min_window": None if not y_ratios else round(float(min(y_ratios)), 6),
            "sequence_y_ratio_max_window": None if not y_ratios else round(float(max(y_ratios)), 6),
            "sequence_y_ratio_mean_window": None if not y_ratios else round(float(np.mean(y_ratios)), 6),
            "sequence_low_screen_ratio_window": None if not y_ratios else round(float(sum(1 for value in y_ratios if value >= 0.88) / len(y_ratios)), 6),
            "sequence_mean_speed_px_sec": None if not speeds else round(float(np.mean(speeds)), 6),
            "sequence_max_speed_px_sec": None if not speeds else round(float(max(speeds)), 6),
            "sequence_confidence_mean_window": None if not window_points else round(float(np.mean([point.confidence for point in window_points])), 6),
        }
    )
    return out


def attach_release_track_features(
    rows: list[dict[str, Any]],
    tracks: dict[str, list[TrackPoint]],
    *,
    video_paths: dict[str, str] | None = None,
    touch_streams: dict[str, list[float]] | None = None,
    min_points: int = 12,
    max_track_gap_sec: float = DEFAULT_MAX_TRACK_GAP_SEC,
    break_tolerance_sec: float = DEFAULT_BREAK_TOLERANCE_SEC,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    track_cache: dict[str, TrackFeatures] = {}
    video_dimension_cache: dict[str, tuple[int | None, int | None]] = {}
    out_rows: list[dict[str, Any]] = []
    per_video: dict[str, dict[str, Any]] = {}
    for row in rows:
        video_name = str(row["video_name"])
        points = tracks.get(video_name, [])
        if video_name not in track_cache:
            if points:
                track_cache[video_name] = compute_track_features(points, min_points=min_points, max_gap_sec=max_track_gap_sec)
            else:
                track_cache[video_name] = TrackFeatures("missing_detection_track", [], np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([]), 0)
        updated = attach_row_features(row, track_cache[video_name], break_tolerance_sec=break_tolerance_sec)
        updated = add_nearest_track_features(updated, points)
        video_path = None if video_paths is None else video_paths.get(str(row.get("video_id"))) or video_paths.get(normalize_video_id(video_name))
        if video_name not in video_dimension_cache:
            video_dimension_cache[video_name] = video_dimensions(video_path)
        width, height = video_dimension_cache[video_name]
        updated = add_sequence_window_features(updated, points, video_path=video_path, width=width, height=height)
        updated = add_touch_event_stream_features(updated, touch_streams or {})
        out_rows.append(updated)
        stats = per_video.setdefault(
            video_name,
            {
                "video_id": row.get("video_id"),
                "source_video": video_name,
                "rows": 0,
                "approved": 0,
                "rejected": 0,
                "track_points": len(points),
                "track_status": track_cache[video_name].status,
                "track_segments": track_cache[video_name].segments,
                "video_width": width,
                "video_height": height,
            },
        )
        stats["rows"] += 1
        if row.get("label"):
            stats["approved"] += 1
        else:
            stats["rejected"] += 1
    summary = {
        "rows": len(out_rows),
        "ok_rows": sum(1 for row in out_rows if row.get("trajectory_feature_status") == "ok"),
        "missing_track_rows": sum(1 for row in out_rows if row.get("trajectory_feature_status") == "missing_detection_track"),
        "videos": list(per_video.values()),
    }
    return out_rows, summary


def feature_value(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value in {None, ""}:
        if "delta" in key or "gap" in key:
            return 9.0
        return 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def disabled_feature(key: str, disabled_prefixes: Iterable[str]) -> bool:
    return any(key == prefix or key.startswith(prefix) for prefix in disabled_prefixes)


def feature_dict(row: dict[str, Any], *, disabled_prefixes: Iterable[str] = ()) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for key in NUMERIC_FEATURES:
        if disabled_feature(key, disabled_prefixes):
            continue
        features[key] = feature_value(row, key)
    for key in BOOLEAN_FEATURES:
        if disabled_feature(key, disabled_prefixes):
            continue
        features[key] = 1.0 if bool(row.get(key)) else 0.0
    for key in CATEGORICAL_FEATURES:
        if disabled_feature(key, disabled_prefixes):
            continue
        features[key] = str(row.get(key) or "missing")
    return features


def build_model(model_family: str):
    from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if model_family == "logistic_regression":
        return make_pipeline(
            DictVectorizer(sparse=True),
            StandardScaler(with_mean=False),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=17),
        )
    if model_family == "extra_trees":
        return make_pipeline(
            DictVectorizer(sparse=False),
            ExtraTreesClassifier(n_estimators=200, max_depth=4, class_weight="balanced", random_state=17),
        )
    if model_family == "gradient_boosting":
        return make_pipeline(
            DictVectorizer(sparse=False),
            GradientBoostingClassifier(max_depth=2, n_estimators=40, random_state=17),
        )
    raise ValueError(f"unknown model family: {model_family}")


def binary_metrics(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    tp = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 1)
    fp = sum(1 for label, pred in zip(labels, predictions) if label == 0 and pred == 1)
    fn = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 0)
    tn = sum(1 for label, pred in zip(labels, predictions) if label == 0 and pred == 0)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = None
    if precision is not None and recall is not None and precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def gate_thresholds(kind: str) -> tuple[float, float]:
    if kind == "drop_floor":
        return DROP_GATE_PRECISION, DROP_GATE_RECALL
    return STALL_GATE_PRECISION, STALL_GATE_RECALL


def train_target(
    rows: list[dict[str, Any]],
    kind: str,
    *,
    min_examples: int = DEFAULT_MIN_EXAMPLES,
    min_videos: int = DEFAULT_MIN_VIDEOS,
    min_positive_examples: int = DEFAULT_MIN_POSITIVE_EXAMPLES,
    min_negative_examples: int = DEFAULT_MIN_NEGATIVE_EXAMPLES,
    model_family: str = "logistic_regression",
    feature_mode: str = "sequence_window",
    disabled_prefixes: Iterable[str] = (),
) -> tuple[dict[str, Any], Any | None]:
    target_rows = [row for row in rows if row.get("kind") == kind]
    labels = [int(row["label"]) for row in target_rows]
    videos = sorted({str(row.get("video_id")) for row in target_rows})
    label_counts = Counter(labels)
    reasons: list[str] = []
    if len(target_rows) < min_examples:
        reasons.append(f"need at least {min_examples} reviewed {kind} rows, found {len(target_rows)}")
    if len(videos) < min_videos:
        reasons.append(f"need at least {min_videos} videos for clip-disjoint {kind} evaluation, found {len(videos)}")
    if label_counts.get(1, 0) < min_positive_examples:
        reasons.append(f"need at least {min_positive_examples} approved {kind} rows, found {label_counts.get(1, 0)}")
    if label_counts.get(0, 0) < min_negative_examples:
        reasons.append(f"need at least {min_negative_examples} rejected {kind} rows, found {label_counts.get(0, 0)}")
    if len(label_counts) < 2:
        reasons.append(f"need approved and rejected {kind} examples, found {dict(label_counts)}")
    if reasons:
        return (
            {
                "status": "not_ready",
                "kind": kind,
                "model_family": model_family,
                "feature_mode": feature_mode,
                "disabled_prefixes": list(disabled_prefixes),
                "reasons": reasons,
                "rows": len(target_rows),
                "videos": videos,
                "label_counts": {"approved": int(label_counts.get(1, 0)), "rejected": int(label_counts.get(0, 0))},
            },
            None,
        )

    predictions: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    skipped_folds: list[dict[str, Any]] = []
    for video_id in videos:
        train_rows = [row for row in target_rows if str(row.get("video_id")) != video_id]
        test_rows = [row for row in target_rows if str(row.get("video_id")) == video_id]
        train_labels = [int(row["label"]) for row in train_rows]
        if len(set(train_labels)) < 2:
            skipped_folds.append({"video_id": video_id, "rows": len(test_rows), "reason": "training fold has one class"})
            continue
        model = build_model(model_family)
        model.fit([feature_dict(row, disabled_prefixes=disabled_prefixes) for row in train_rows], train_labels)
        test_labels = [int(row["label"]) for row in test_rows]
        test_feature_rows = [feature_dict(row, disabled_prefixes=disabled_prefixes) for row in test_rows]
        fold_preds = [int(value) for value in model.predict(test_feature_rows)]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(test_feature_rows)
            classes = [int(item) for item in model.classes_]
            positive_index = classes.index(1) if 1 in classes else 0
            positive_scores = [float(row[positive_index]) for row in proba]
        else:
            positive_scores = [1.0 if pred else 0.0 for pred in fold_preds]
        metrics = binary_metrics(test_labels, fold_preds)
        folds.append(
            {
                "video_id": video_id,
                "rows": len(test_rows),
                "label_counts": {"approved": int(sum(test_labels)), "rejected": int(len(test_labels) - sum(test_labels))},
                "prediction_counts": {"approved": int(sum(fold_preds)), "rejected": int(len(fold_preds) - sum(fold_preds))},
                **metrics,
            }
        )
        for row, label, pred, score in zip(test_rows, test_labels, fold_preds, positive_scores):
            predictions.append(
                {
                    "kind": kind,
                    "feature_mode": feature_mode,
                    "model_family": model_family,
                    "video_id": row.get("video_id"),
                    "video_name": row.get("video_name"),
                    "candidate_time_sec": row.get("candidate_time_sec"),
                    "label": "approved" if label else "rejected",
                    "prediction": "approved" if pred else "rejected",
                    "positive_score": round(float(score), 6),
                    "correct": label == pred,
                    "trajectory_feature_status": row.get("trajectory_feature_status"),
                    "trajectory_impulse_score": row.get("trajectory_impulse_score"),
                    "trajectory_y_range_window_px": row.get("trajectory_y_range_window_px"),
                    "candidate_track_confidence": row.get("candidate_track_confidence"),
                    "candidate_y_ratio": row.get("candidate_y_ratio"),
                    "event_next_touch_gap_sec": row.get("event_next_touch_gap_sec"),
                    "event_no_next_touch": row.get("event_no_next_touch"),
                    "event_post_gap_gt_1s": row.get("event_post_gap_gt_1s"),
                    "event_prev_touch_gap_sec": row.get("event_prev_touch_gap_sec"),
                    "event_touch_stream_present": row.get("event_touch_stream_present"),
                    "floor_context_score": row.get("floor_context_score"),
                    "floor_ground_ratio": row.get("floor_ground_ratio"),
                    "floor_sky_ratio": row.get("floor_sky_ratio"),
                    "sequence_track_coverage_ratio": row.get("sequence_track_coverage_ratio"),
                    "sequence_low_screen_ratio_window": row.get("sequence_low_screen_ratio_window"),
                    "sequence_longest_gap_sec": row.get("sequence_longest_gap_sec"),
                    "sequence_mean_speed_px_sec": row.get("sequence_mean_speed_px_sec"),
                }
            )
    if not predictions:
        return (
            {
                "status": "not_ready",
                "kind": kind,
                "model_family": model_family,
                "feature_mode": feature_mode,
                "disabled_prefixes": list(disabled_prefixes),
                "reasons": ["all leave-one-video-out folds were skipped"],
                "rows": len(target_rows),
                "videos": videos,
                "label_counts": {"approved": int(label_counts.get(1, 0)), "rejected": int(label_counts.get(0, 0))},
                "skipped_folds": skipped_folds,
            },
            None,
        )

    all_labels = [1 if row["label"] == "approved" else 0 for row in predictions]
    all_preds = [1 if row["prediction"] == "approved" else 0 for row in predictions]
    metrics = binary_metrics(all_labels, all_preds)
    precision_threshold, recall_threshold = gate_thresholds(kind)
    precision = metrics["precision"]
    recall = metrics["recall"]
    gate_pass = precision is not None and recall is not None and precision >= precision_threshold and recall >= recall_threshold
    final_model = build_model(model_family)
    final_model.fit([feature_dict(row, disabled_prefixes=disabled_prefixes) for row in target_rows], labels)
    return (
        {
            "status": "trained",
            "kind": kind,
            "model_family": model_family,
            "feature_mode": feature_mode,
            "disabled_prefixes": list(disabled_prefixes),
            "rows": len(target_rows),
            "evaluated_rows": len(predictions),
            "videos": videos,
            "label_counts": {"approved": int(label_counts.get(1, 0)), "rejected": int(label_counts.get(0, 0))},
            "prediction_counts": dict(Counter(row["prediction"] for row in predictions)),
            "gate": "pass" if gate_pass else "fail",
            "precision_threshold": precision_threshold,
            "recall_threshold": recall_threshold,
            **metrics,
            "folds": folds,
            "skipped_folds": skipped_folds,
            "errors": [row for row in predictions if not row["correct"]],
        },
        final_model,
    )


def train_best_target(rows: list[dict[str, Any]], kind: str) -> tuple[dict[str, Any], Any | None]:
    results: dict[str, dict[str, Any]] = {}
    models: dict[str, Any] = {}
    for mode_name, disabled_prefixes in FEATURE_MODES.items():
        for family in MODEL_FAMILIES:
            result, model = train_target(
                rows,
                kind,
                model_family=family,
                feature_mode=mode_name,
                disabled_prefixes=disabled_prefixes,
            )
            model_key = f"{mode_name}/{family}"
            results[model_key] = result
            if model is not None:
                models[model_key] = model
    trained = [(model_key, result) for model_key, result in results.items() if result.get("status") == "trained"]
    mode_best: dict[str, dict[str, Any]] = {}
    family_best: dict[str, dict[str, Any]] = {}
    for mode_name in FEATURE_MODES:
        candidates = [(key, result) for key, result in trained if str(result.get("feature_mode")) == mode_name]
        if candidates:
            mode_best[mode_name] = max(candidates, key=lambda item: (float(item[1].get("f1") or 0.0), float(item[1].get("precision") or 0.0)))[1]
        else:
            first_key = f"{mode_name}/{MODEL_FAMILIES[0]}"
            if first_key in results:
                mode_best[mode_name] = results[first_key]
    for family in MODEL_FAMILIES:
        candidates = [(key, result) for key, result in trained if str(result.get("model_family")) == family]
        if candidates:
            family_best[family] = max(candidates, key=lambda item: (float(item[1].get("f1") or 0.0), float(item[1].get("precision") or 0.0)))[1]
        else:
            first_key = f"{next(iter(FEATURE_MODES))}/{family}"
            if first_key in results:
                family_best[family] = results[first_key]
    if not trained:
        first = results[f"{next(iter(FEATURE_MODES))}/{MODEL_FAMILIES[0]}"]
        first = dict(first)
        first["selected_feature_mode"] = first.get("feature_mode")
        first["selected_model_family"] = first.get("model_family")
        first["model_mode_results"] = results
        first["feature_mode_results"] = mode_best
        first["model_family_results"] = family_best
        return first, None

    def key(item: tuple[str, dict[str, Any]]) -> tuple[float, float, int, int]:
        model_key, result = item
        f1 = result.get("f1")
        precision = result.get("precision")
        mode = str(result.get("feature_mode"))
        family = str(result.get("model_family"))
        mode_preference = {"l2_only": 1, "sequence_window": 0}.get(mode, 0)
        preference = {"logistic_regression": 2, "extra_trees": 1, "gradient_boosting": 0}.get(family, 0)
        return (float(f1 or 0.0), float(precision or 0.0), mode_preference, preference)

    model_key, best = max(trained, key=key)
    out = dict(best)
    out["selected_feature_mode"] = best.get("feature_mode")
    out["selected_model_family"] = best.get("model_family")
    out["selected_model_key"] = model_key
    out["model_mode_results"] = results
    out["feature_mode_results"] = mode_best
    out["model_family_results"] = family_best
    return out, models.get(model_key)


def write_training_artifacts(out_dir: Path, models: dict[str, Any], results: dict[str, Any]) -> str | None:
    if not models:
        return None
    artifact = out_dir / "release_stall_drop_classifier.joblib"
    joblib.dump(
        {
            "schema_version": 1,
            "models": models,
            "targets": sorted(models),
            "model_results": results,
            "feature_modes": FEATURE_MODES,
            "numeric_features": NUMERIC_FEATURES,
            "boolean_features": BOOLEAN_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "threshold": DEFAULT_THRESHOLD,
        },
        artifact,
    )
    return str(artifact)


def fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def render_error_strip(
    *,
    row: dict[str, Any],
    video_path: str,
    track: list[TrackPoint],
    out_path: Path,
) -> None:
    time_sec = float(row["candidate_time_sec"])
    frame_times = [time_sec - 0.40, time_sec - 0.20, time_sec, time_sec + 0.20, time_sec + 0.40]
    frame_w, frame_h, header_h = 240, 320, 92
    canvas = np.full((header_h + frame_h, frame_w * len(frame_times), 3), 244, dtype=np.uint8)
    title = (
        f"{row.get('kind')} {row.get('video_id')} t={time_sec:.3f}s "
        f"label={row.get('label')} pred={row.get('prediction')}"
    )
    cv2.putText(canvas, title[:140], (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"score={fmt_metric(row.get('positive_score'))} impulse={fmt_metric(row.get('trajectory_impulse_score'))} y_range={fmt_metric(row.get('trajectory_y_range_window_px'))}",
        (12, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (55, 55, 55),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"y_ratio={fmt_metric(row.get('candidate_y_ratio'))} floor={fmt_metric(row.get('floor_context_score'))} low={fmt_metric(row.get('sequence_low_screen_ratio_window'))} next_touch_gap={fmt_metric(row.get('event_next_touch_gap_sec'))}",
        (12, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (55, 55, 55),
        1,
        cv2.LINE_AA,
    )
    for index, frame_time in enumerate(frame_times):
        frame = read_frame(video_path, frame_time)
        if frame is None:
            tile = np.full((frame_h, frame_w, 3), 20, dtype=np.uint8)
            cv2.putText(tile, "missing frame", (28, frame_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)
        else:
            point = nearest_track_point(track, frame_time)
            if point is not None and abs(point.time_sec - frame_time) <= 0.09:
                cv2.circle(frame, (int(round(point.x)), int(round(point.y))), 28, (255, 255, 0), 5)
                cv2.circle(frame, (int(round(point.x)), int(round(point.y))), 6, (0, 0, 255), -1)
            tile = resize_letterbox(frame, frame_w, frame_h)
        cv2.putText(tile, f"{frame_time:.2f}s", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        if abs(frame_time - time_sec) < 1e-6:
            cv2.rectangle(tile, (2, 2), (frame_w - 3, frame_h - 3), (0, 0, 255), 4)
        canvas[header_h : header_h + frame_h, index * frame_w : (index + 1) * frame_w] = tile
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def write_contact_sheet(paths: list[Path], out_path: Path) -> None:
    images: list[np.ndarray] = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        width = 420
        height = max(1, round(image.shape[0] * width / image.shape[1]))
        images.append(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
    if not images:
        return
    cols = 2
    w = max(image.shape[1] for image in images)
    h = max(image.shape[0] for image in images)
    rows = math.ceil(len(images) / cols)
    sheet = np.full((rows * h, cols * w, 3), 244, dtype=np.uint8)
    for index, image in enumerate(images):
        y = (index // cols) * h
        x = (index % cols) * w
        sheet[y : y + image.shape[0], x : x + image.shape[1]] = image
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def render_error_audit(
    rows: list[dict[str, Any]],
    results: dict[str, Any],
    tracks: dict[str, list[TrackPoint]],
    *,
    out_dir: Path,
    review_manifest: Path,
    max_errors: int = 24,
) -> dict[str, Any]:
    video_paths = load_video_paths(review_manifest)
    for video_path in sorted(DEFAULT_VIDEO_SEARCH_DIR.glob("video-*singular_display*.MOV")):
        video_paths.setdefault(normalize_video_id(video_path.name), str(video_path))
    errors: list[dict[str, Any]] = []
    for target, result in results.items():
        for error in result.get("errors") or []:
            errors.append(error)
    errors = sorted(errors, key=lambda row: (str(row.get("kind")), str(row.get("video_id")), float(row.get("candidate_time_sec") or 0.0)))[:max_errors]
    rendered: list[Path] = []
    strips_dir = out_dir / "error_audit/strips"
    for index, error in enumerate(errors, start=1):
        video_id = str(error.get("video_id"))
        video_path = video_paths.get(video_id) or video_paths.get(normalize_video_id(error.get("video_name")))
        if not video_path:
            continue
        strip = strips_dir / f"{index:03d}_{error.get('kind')}_{video_id}_{float(error.get('candidate_time_sec') or 0):.3f}.png"
        render_error_strip(row=error, video_path=video_path, track=tracks.get(str(error.get("video_name")), []), out_path=strip)
        if strip.exists():
            rendered.append(strip)
            error["strip_path"] = str(strip)
    sheet = out_dir / "error_audit/stall_drop_error_contact_sheet.jpg"
    write_contact_sheet(rendered, sheet)
    audit = {
        "errors": len(errors),
        "rendered_strips": len(rendered),
        "contact_sheet": str(sheet) if sheet.exists() else None,
        "rows_jsonl": str(out_dir / "error_audit/stall_drop_errors.jsonl"),
    }
    write_jsonl(out_dir / "error_audit/stall_drop_errors.jsonl", errors)
    return audit


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Release Stall / Drop Classifier",
        "",
        f"- Status: `{summary['status']}`",
        f"- Created: `{summary['created_at']}`",
        f"- Label source: `{summary['review_labels']}`",
        f"- Compact track cache: `{summary['compact_track_cache']}`",
        f"- Track threshold: `{summary['threshold']}`",
        "",
        "Detection inputs:",
        "",
    ]
    for detection_path in summary.get("detections_jsonl") or []:
        lines.append(f"- `{detection_path}`")
    lines.extend(["", "Touch-event stream inputs:", ""])
    for event_path in summary.get("touch_events_jsonl") or []:
        lines.append(f"- `{event_path}`")
    lines.extend(
        [
            "",
        "## Label Inventory",
        "",
        f"- Canonical reviewed rows: `{summary['review_row_summary']['canonical_rows']}`",
        f"- Clean rows after conflict exclusion: `{summary['review_row_summary']['clean_rows']}`",
        f"- Conflict groups excluded: `{summary['review_row_summary']['conflict_groups']}`",
        f"- Conflict rows excluded: `{summary['review_row_summary']['excluded_conflict_rows']}`",
        f"- Same-status duplicate rows merged: `{summary['review_row_summary']['merged_duplicate_rows']}`",
        f"- Clean counts: `{summary['review_row_summary']['clean_counts_by_kind_status']}`",
        "",
        "## Track Coverage",
        "",
        f"- Compact points: `{summary['compact_track_manifest']['kept_points']}`",
        f"- Feature rows with usable L2 status: `{summary['feature_summary']['ok_rows']}` / `{summary['feature_summary']['rows']}`",
        f"- Touch-event stream videos: `{summary.get('touch_event_stream_summary', {}).get('canonical_videos', 0)}`",
        "",
        "## Gates",
        "",
        "| target | rows | approved/rejected | feature/model | precision | recall | f1 | gate |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for target, result in summary["targets"].items():
        counts = result.get("label_counts") or {}
        feature_model = f"{result.get('selected_feature_mode') or result.get('feature_mode')}/{result.get('selected_model_family') or result.get('model_family')}"
        lines.append(
            f"| `{target}` | {result.get('rows')} | {counts.get('approved', 0)}/{counts.get('rejected', 0)} | "
            f"`{feature_model}` | {fmt_metric(result.get('precision'))} | "
            f"{fmt_metric(result.get('recall'))} | {fmt_metric(result.get('f1'))} | `{result.get('gate', result.get('status'))}` |"
        )
    lines.extend(["", "## Target Details", ""])
    for target, result in summary["targets"].items():
        lines.extend(["", f"### `{target}`", ""])
        lines.append(f"- Status: `{result.get('status')}`")
        if result.get("reasons"):
            for reason in result["reasons"]:
                lines.append(f"- Blocker: {reason}")
        if result.get("status") == "trained":
            lines.append(f"- Selected feature mode: `{result.get('selected_feature_mode') or result.get('feature_mode')}`")
            lines.append(f"- Selected model family: `{result.get('selected_model_family')}`")
            lines.append(f"- Gate thresholds: precision >= `{result.get('precision_threshold')}`, recall >= `{result.get('recall_threshold')}`")
            lines.append(f"- Confusion: TP `{result.get('true_positive')}`, FP `{result.get('false_positive')}`, FN `{result.get('false_negative')}`, TN `{result.get('true_negative')}`")
            if result.get("feature_mode_results"):
                lines.append("- Feature-mode ablation:")
                for mode_name, mode_result in result["feature_mode_results"].items():
                    lines.append(
                        f"  - `{mode_name}` / `{mode_result.get('model_family')}`: status `{mode_result.get('status')}`, "
                        f"P `{fmt_metric(mode_result.get('precision'))}`, R `{fmt_metric(mode_result.get('recall'))}`, "
                        f"F1 `{fmt_metric(mode_result.get('f1'))}`, gate `{mode_result.get('gate')}`"
                    )
            if result.get("model_family_results"):
                lines.append("- Best-per-family ablation:")
                for family, family_result in result["model_family_results"].items():
                    lines.append(
                        f"  - `{family}` / `{family_result.get('feature_mode')}`: status `{family_result.get('status')}`, "
                        f"P `{fmt_metric(family_result.get('precision'))}`, R `{fmt_metric(family_result.get('recall'))}`, "
                        f"F1 `{fmt_metric(family_result.get('f1'))}`, gate `{family_result.get('gate')}`"
                    )
            if result.get("folds"):
                lines.append("- Leave-one-video-out folds:")
                for fold in result["folds"]:
                    lines.append(
                        f"  - `{fold['video_id']}` rows `{fold['rows']}`: "
                        f"P `{fmt_metric(fold.get('precision'))}`, R `{fmt_metric(fold.get('recall'))}`, "
                        f"FP `{fold.get('false_positive')}`, FN `{fold.get('false_negative')}`"
                    )
            if result.get("errors"):
                lines.append("- First errors:")
                for error in result["errors"][:12]:
                    lines.append(
                        f"  - `{error.get('video_id')}` @ `{float(error.get('candidate_time_sec') or 0):.3f}s`: "
                        f"label `{error.get('label')}`, predicted `{error.get('prediction')}`, score `{fmt_metric(error.get('positive_score'))}`"
                    )
    lines.extend(
        [
            "",
            "## Visual Audit",
            "",
            f"- Error strips rendered: `{summary['error_audit']['rendered_strips']}`",
            f"- Contact sheet: `{summary['error_audit'].get('contact_sheet')}`",
            "",
            "## Release Interpretation",
            "",
            "- This uses existing reviewed reset/stall decisions; it does not ask for duplicate labels.",
            "- The supplemental stall/drop OWLv2 cache removes missing-track confounds.",
            "- The sequence-window/floor-context and rally-sequence ablations are compared against the prior L2-only baseline; a richer mode is selected only when leave-one-video-out metrics improve.",
            "- Failure after full coverage, floor context, and available merged-touch gap context means the remaining automatic drop/reset problem is not a simple detector-coverage or point-local reset issue.",
            "- Drop/stall badges should remain unpromoted unless the target gate passes and downstream HUD integration is explicitly wired.",
            "- Stall is expected to remain label-limited when approved examples are scarce.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stall_drop_classifier(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    review_rows_raw = read_jsonl(args.review_labels)
    review_rows, review_summary = prepare_review_rows(review_rows_raw, bin_sec=args.conflict_bin_sec)
    compact_cache = args.compact_track_cache or (out_dir / "owlv2_compact_track_points.jsonl")
    if args.force_compact_track_cache or not compact_cache.exists():
        compact_manifest = build_compact_track_cache(args.detections_jsonl, compact_cache, threshold=args.threshold)
    else:
        manifest_path = compact_cache.with_suffix(".manifest.json")
        compact_manifest = read_json(manifest_path) if manifest_path.exists() else {"cache_path": str(compact_cache), "kept_points": len(read_jsonl(compact_cache))}
    tracks = load_compact_tracks(compact_cache)
    video_paths = video_paths_with_download_fallback(args.review_manifest)
    touch_streams = load_touch_event_streams(args.touch_events_jsonl)
    feature_rows, feature_summary = attach_release_track_features(
        review_rows,
        tracks,
        video_paths=video_paths,
        touch_streams=touch_streams,
        min_points=args.min_points,
        max_track_gap_sec=args.max_track_gap_sec,
        break_tolerance_sec=args.break_tolerance_sec,
    )
    write_jsonl(out_dir / "stall_drop_training_rows.jsonl", feature_rows)
    target_results: dict[str, Any] = {}
    models: dict[str, Any] = {}
    for target in TARGET_KINDS:
        result, model = train_best_target(feature_rows, target)
        target_results[target] = result
        if model is not None:
            models[target] = model
    artifact = write_training_artifacts(out_dir, models, target_results)
    error_audit = render_error_audit(feature_rows, target_results, tracks, out_dir=out_dir, review_manifest=args.review_manifest, max_errors=args.max_error_strips)
    status = "trained"
    if not any(result.get("status") == "trained" for result in target_results.values()):
        status = "not_ready"
    elif any(result.get("gate") == "pass" for result in target_results.values()):
        status = "gate_candidate"
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "review_labels": str(args.review_labels),
        "out_dir": str(out_dir),
        "threshold": args.threshold,
        "detections_jsonl": [str(path) for path in args.detections_jsonl],
        "touch_events_jsonl": [str(path) for path in args.touch_events_jsonl],
        "touch_event_stream_summary": summarize_touch_streams(touch_streams),
        "compact_track_cache": str(compact_cache),
        "compact_track_manifest": compact_manifest,
        "review_row_summary": review_summary,
        "feature_summary": feature_summary,
        "targets": target_results,
        "model_artifact": artifact,
        "error_audit": error_audit,
        "report": str(out_dir / "release_stall_drop_classifier_report.md"),
    }
    write_json(out_dir / "release_stall_drop_classifier_summary.json", summary)
    write_report(out_dir / "release_stall_drop_classifier_report.md", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate release stall/drop classifiers from reviewed labels")
    parser.add_argument("--review-labels", type=Path, default=DEFAULT_REVIEW_LABELS)
    parser.add_argument("--detections-jsonl", type=Path, action="append", default=list(DEFAULT_DETECTIONS_JSONL))
    parser.add_argument("--touch-events-jsonl", type=Path, action="append", default=list(DEFAULT_TOUCH_EVENTS_JSONL))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--compact-track-cache", type=Path)
    parser.add_argument("--force-compact-track-cache", action="store_true")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--conflict-bin-sec", type=float, default=DEFAULT_BIN_SEC)
    parser.add_argument("--min-points", type=int, default=12)
    parser.add_argument("--max-track-gap-sec", type=float, default=DEFAULT_MAX_TRACK_GAP_SEC)
    parser.add_argument("--break-tolerance-sec", type=float, default=DEFAULT_BREAK_TOLERANCE_SEC)
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW_MANIFEST)
    parser.add_argument("--max-error-strips", type=int, default=24)
    return parser


def main() -> None:
    summary = run_stall_drop_classifier(build_parser().parse_args())
    print(f"status: {summary['status']}")
    print(f"report: {summary['report']}")
    print(f"summary: {Path(summary['out_dir']) / 'release_stall_drop_classifier_summary.json'}")
    for target, result in summary["targets"].items():
        print(
            f"{target}: status={result.get('status')} gate={result.get('gate')} "
            f"P={fmt_metric(result.get('precision'))} R={fmt_metric(result.get('recall'))} rows={result.get('rows')}"
        )


if __name__ == "__main__":
    main()
