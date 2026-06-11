#!/usr/bin/env python3
"""Attach L2 trajectory features to touch-candidate rows.

Inputs are the candidate tables from `build_touch_training_table.py` and cached
fixed-OWLv2 detections or prediction rows. The output keeps the same schema but
fills the trajectory columns that `train_touch_classifier.py` requires for the
real fused model.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from owlv2_event_eval import (
    AUDIO_BREAK_MIN_SAMPLES,
    AUDIO_BREAK_VELOCITY_WIN_SEC,
    AUDIO_FUSION_BREAK_TOL_SEC,
    LAMBDAS,
    MIN_TOUCH_GAP_SEC,
    dedupe_times,
    velocity_change,
)
from prototype_arc_touches import detect_touches, robust_clean


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = ROOT / "runs/release-27-public/touch_corpus_v1/touch_training_dataset_v1"
DEFAULT_THRESHOLD = 0.2
DEFAULT_MAX_TRACK_GAP_SEC = 0.25


@dataclass(frozen=True)
class TrackPoint:
    time_sec: float
    x: float
    y: float
    confidence: float
    frame_index: int | None = None


@dataclass(frozen=True)
class TrackFeatures:
    status: str
    breakpoints: list[dict[str, Any]]
    ts: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    confidences: np.ndarray
    segments: int = 0


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


def source_video_name(row: dict[str, Any]) -> str | None:
    value = row.get("source_video") or row.get("video_name")
    if value is None:
        return None
    return Path(str(value)).name


def top_detection(row: dict[str, Any], threshold: float) -> TrackPoint | None:
    detections = row.get("detections")
    if isinstance(detections, list):
        above = [det for det in detections if float(det.get("score") or 0.0) >= threshold]
        if not above:
            return None
        det = max(above, key=lambda item: float(item.get("score") or 0.0))
        return TrackPoint(
            time_sec=float(row["time_sec"]),
            x=float(det["x"]),
            y=float(det["y"]),
            confidence=float(det["score"]),
            frame_index=None if row.get("frame_index") is None else int(row["frame_index"]),
        )
    if row.get("x") is not None and row.get("y") is not None:
        confidence = float(row.get("confidence") or row.get("score") or 1.0)
        if confidence < threshold:
            return None
        return TrackPoint(
            time_sec=float(row["time_sec"]),
            x=float(row["x"]),
            y=float(row["y"]),
            confidence=confidence,
            frame_index=None if row.get("frame_index") is None else int(row["frame_index"]),
        )
    return None


def load_detection_tracks(paths: list[Path], threshold: float) -> dict[str, list[TrackPoint]]:
    tracks: dict[str, list[TrackPoint]] = {}
    for path in paths:
        for row in read_jsonl(path):
            video = source_video_name(row)
            if not video:
                continue
            point = top_detection(row, threshold)
            if point is None:
                continue
            tracks.setdefault(video, []).append(point)
    for video, points in tracks.items():
        deduped: dict[float, TrackPoint] = {}
        for point in points:
            key = round(point.time_sec, 6)
            prior = deduped.get(key)
            if prior is None or point.confidence > prior.confidence:
                deduped[key] = point
        tracks[video] = sorted(deduped.values(), key=lambda item: item.time_sec)
    return tracks


def detection_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for item in args.detections_jsonl or []:
        paths.append(item)
    for directory in args.detections_dir or []:
        paths.extend(sorted(directory.glob("*.jsonl")))
        paths.extend(sorted(directory.glob("**/*.jsonl")))
    unique: list[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def split_track_segments(points: list[TrackPoint], *, max_gap_sec: float) -> list[list[TrackPoint]]:
    ordered = sorted(points, key=lambda item: item.time_sec)
    segments: list[list[TrackPoint]] = []
    for point in ordered:
        if not segments or point.time_sec - segments[-1][-1].time_sec > max_gap_sec:
            segments.append([point])
        else:
            segments[-1].append(point)
    return segments


def arrays_for_points(points: list[TrackPoint]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([point.time_sec for point in points], dtype=float),
        np.asarray([point.x for point in points], dtype=float),
        np.asarray([point.y for point in points], dtype=float),
        np.asarray([point.confidence for point in points], dtype=float),
    )


def clean_segment(points: list[TrackPoint], *, min_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if len(points) < min_points:
        return None
    ts, xs, ys, confidences = arrays_for_points(points)
    keep = robust_clean(ts, xs, ys, floor_px=18.0)
    ts, xs, ys, confidences = ts[keep], xs[keep], ys[keep], confidences[keep]
    if len(ts) < min_points:
        return None
    return ts, xs, ys, confidences


def compute_track_features(points: list[TrackPoint], *, min_points: int, max_gap_sec: float = DEFAULT_MAX_TRACK_GAP_SEC) -> TrackFeatures:
    if len(points) < min_points:
        return TrackFeatures("too_few_track_points", [], np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([]), 0)

    breakpoint_rows: list[dict[str, Any]] = []
    clean_segments: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for segment_index, segment in enumerate(split_track_segments(points, max_gap_sec=max_gap_sec)):
        clean = clean_segment(segment, min_points=min_points)
        if clean is None:
            continue
        ts, xs, ys, confidences = clean
        clean_segments.append(clean)
        tc = ts - ts.mean()
        t0 = float(ts.mean())
        for lam in LAMBDAS:
            raw = detect_touches(tc, xs, ys, t0, lam)
            for time_sec in dedupe_times(raw, MIN_TOUCH_GAP_SEC):
                breakpoint_rows.append(
                    {
                        "time_sec": float(time_sec),
                        "lambda": int(lam),
                        "segment_index": segment_index,
                        "dvy": velocity_change(
                            ts,
                            ys,
                            float(time_sec),
                            win=AUDIO_BREAK_VELOCITY_WIN_SEC,
                            min_samples=AUDIO_BREAK_MIN_SAMPLES,
                        ),
                    }
                )
    if not clean_segments:
        ts_all, xs_all, ys_all, confidences_all = arrays_for_points(points)
        return TrackFeatures("too_few_segment_track_points", [], ts_all, xs_all, ys_all, confidences_all, len(split_track_segments(points, max_gap_sec=max_gap_sec)))
    ts = np.concatenate([segment[0] for segment in clean_segments])
    xs = np.concatenate([segment[1] for segment in clean_segments])
    ys = np.concatenate([segment[2] for segment in clean_segments])
    confidences = np.concatenate([segment[3] for segment in clean_segments])
    order = np.argsort(ts)
    ts, xs, ys, confidences = ts[order], xs[order], ys[order], confidences[order]
    # Dedupe equivalent breakpoint times emitted by multiple lambdas inside each
    # continuous segment while preserving per-lambda support rows near candidates.
    return TrackFeatures("ok", breakpoint_rows, ts, xs, ys, confidences, len(clean_segments))


def nearest_distance(time_sec: float, values: list[float]) -> float | None:
    if not values:
        return None
    return min(abs(time_sec - value) for value in values)


def local_slope(ts: np.ndarray, values: np.ndarray, time_sec: float, *, before: bool, win: float, min_samples: int) -> float | None:
    if before:
        mask = (ts >= time_sec - win) & (ts < time_sec)
    else:
        mask = (ts > time_sec) & (ts <= time_sec + win)
    if int(mask.sum()) < min_samples:
        return None
    return float(np.polyfit(ts[mask], values[mask], 1)[0])


def height_reversal(ts: np.ndarray, ys: np.ndarray, time_sec: float, *, win: float = 0.14) -> bool | None:
    before = local_slope(ts, ys, time_sec, before=True, win=win, min_samples=2)
    after = local_slope(ts, ys, time_sec, before=False, win=win, min_samples=2)
    if before is None or after is None:
        return None
    # Image y increases downward. A footbag touch that sends the bag upward is
    # commonly a positive-to-negative y-velocity reversal.
    return before > 0.0 and after < 0.0


def confidence_near(ts: np.ndarray, confidences: np.ndarray, time_sec: float, *, win: float = 0.08) -> float | None:
    if len(ts) == 0:
        return None
    mask = np.abs(ts - time_sec) <= win
    if not bool(mask.any()):
        index = int(np.argmin(np.abs(ts - time_sec)))
        if abs(float(ts[index]) - time_sec) > win * 2.0:
            return None
        return float(confidences[index])
    return float(np.max(confidences[mask]))


def local_track_window_features(
    ts: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    confidences: np.ndarray,
    time_sec: float,
    *,
    win: float = 0.16,
    min_samples: int = 2,
) -> dict[str, Any]:
    """Return candidate-centered track dynamics.

    Image y increases downward. Touches often produce a discontinuity in y
    velocity, but sideways redirections are common too; keep x and y velocity
    jumps separate so the classifier can learn either pattern.
    """
    if len(ts) == 0:
        return {
            "trajectory_track_points_window": 0,
            "trajectory_gap_before_sec": None,
            "trajectory_gap_after_sec": None,
            "trajectory_nearest_y_peak_delta_sec": None,
            "trajectory_nearest_y_trough_delta_sec": None,
            "trajectory_y_position_pct_window": None,
            "trajectory_y_peak_prominence_window_px": None,
            "trajectory_y_trough_prominence_window_px": None,
            "trajectory_vx_before": None,
            "trajectory_vx_after": None,
            "trajectory_vx_delta": None,
            "trajectory_vy_before": None,
            "trajectory_vy_after": None,
            "trajectory_vy_delta": None,
            "trajectory_ax_window": None,
            "trajectory_ay_window": None,
            "trajectory_speed_before": None,
            "trajectory_speed_after": None,
            "trajectory_speed_delta": None,
            "trajectory_impulse_score": None,
            "trajectory_x_range_window_px": None,
            "trajectory_y_range_window_px": None,
            "trajectory_confidence_mean_window": None,
            "trajectory_local_y_quad_rms_px": None,
        }

    before_mask = ts < time_sec
    after_mask = ts > time_sec
    before_times = ts[before_mask]
    after_times = ts[after_mask]
    gap_before = None if len(before_times) == 0 else float(time_sec - np.max(before_times))
    gap_after = None if len(after_times) == 0 else float(np.min(after_times) - time_sec)

    window_mask = np.abs(ts - time_sec) <= win
    window_count = int(window_mask.sum())
    x_range = None
    y_range = None
    conf_mean = None
    y_quad_rms = None
    y_position_pct = None
    y_peak_prominence = None
    y_trough_prominence = None
    ax_window = None
    ay_window = None
    if window_count:
        xw = xs[window_mask]
        yw = ys[window_mask]
        tw = ts[window_mask]
        cw = confidences[window_mask]
        x_range = float(np.max(xw) - np.min(xw))
        y_range = float(np.max(yw) - np.min(yw))
        conf_mean = float(np.mean(cw))
        y_at_candidate = float(np.interp(time_sec, ts, ys))
        if y_range and y_range > 1e-6:
            y_position_pct = float((y_at_candidate - np.min(yw)) / y_range)
            # Image y grows downward. "Peak" here means low-on-screen / high y;
            # "trough" means high-on-screen / low y. Keep both so graph orientation
            # does not matter to the classifier.
            y_peak_prominence = float(y_at_candidate - np.min(yw))
            y_trough_prominence = float(np.max(yw) - y_at_candidate)
        if window_count >= 5:
            centered_t = tw - time_sec
            coeffs_y = np.polyfit(centered_t, yw, 2)
            coeffs_x = np.polyfit(centered_t, xw, 2)
            residuals = yw - np.polyval(coeffs_y, centered_t)
            y_quad_rms = float(np.sqrt(np.mean(residuals * residuals)))
            ax_window = float(2.0 * coeffs_x[0])
            ay_window = float(2.0 * coeffs_y[0])

    extrema_win = max(win * 2.0, 0.32)
    local_mask = np.abs(ts - time_sec) <= extrema_win
    local_ts = ts[local_mask]
    local_ys = ys[local_mask]
    peak_times: list[float] = []
    trough_times: list[float] = []
    if len(local_ts) >= 3:
        for i in range(1, len(local_ts) - 1):
            if local_ys[i] >= local_ys[i - 1] and local_ys[i] >= local_ys[i + 1]:
                peak_times.append(float(local_ts[i]))
            if local_ys[i] <= local_ys[i - 1] and local_ys[i] <= local_ys[i + 1]:
                trough_times.append(float(local_ts[i]))
    nearest_peak_delta = nearest_distance(time_sec, peak_times)
    nearest_trough_delta = nearest_distance(time_sec, trough_times)

    vx_before = local_slope(ts, xs, time_sec, before=True, win=win, min_samples=min_samples)
    vx_after = local_slope(ts, xs, time_sec, before=False, win=win, min_samples=min_samples)
    vy_before = local_slope(ts, ys, time_sec, before=True, win=win, min_samples=min_samples)
    vy_after = local_slope(ts, ys, time_sec, before=False, win=win, min_samples=min_samples)
    speed_before = None
    speed_after = None
    if vx_before is not None and vy_before is not None:
        speed_before = float(math.hypot(vx_before, vy_before))
    if vx_after is not None and vy_after is not None:
        speed_after = float(math.hypot(vx_after, vy_after))
    impulse_score = None
    if vx_before is not None and vx_after is not None and vy_before is not None and vy_after is not None:
        impulse_score = float(math.hypot(vx_after - vx_before, vy_after - vy_before))

    return {
        "trajectory_track_points_window": window_count,
        "trajectory_gap_before_sec": None if gap_before is None else round(gap_before, 6),
        "trajectory_gap_after_sec": None if gap_after is None else round(gap_after, 6),
        "trajectory_nearest_y_peak_delta_sec": None if nearest_peak_delta is None else round(nearest_peak_delta, 6),
        "trajectory_nearest_y_trough_delta_sec": None if nearest_trough_delta is None else round(nearest_trough_delta, 6),
        "trajectory_y_position_pct_window": None if y_position_pct is None else round(y_position_pct, 6),
        "trajectory_y_peak_prominence_window_px": None if y_peak_prominence is None else round(y_peak_prominence, 6),
        "trajectory_y_trough_prominence_window_px": None if y_trough_prominence is None else round(y_trough_prominence, 6),
        "trajectory_vx_before": None if vx_before is None else round(vx_before, 6),
        "trajectory_vx_after": None if vx_after is None else round(vx_after, 6),
        "trajectory_vx_delta": None if vx_before is None or vx_after is None else round(vx_after - vx_before, 6),
        "trajectory_vy_before": None if vy_before is None else round(vy_before, 6),
        "trajectory_vy_after": None if vy_after is None else round(vy_after, 6),
        "trajectory_vy_delta": None if vy_before is None or vy_after is None else round(vy_after - vy_before, 6),
        "trajectory_ax_window": None if ax_window is None else round(ax_window, 6),
        "trajectory_ay_window": None if ay_window is None else round(ay_window, 6),
        "trajectory_speed_before": None if speed_before is None else round(speed_before, 6),
        "trajectory_speed_after": None if speed_after is None else round(speed_after, 6),
        "trajectory_speed_delta": None if speed_before is None or speed_after is None else round(speed_after - speed_before, 6),
        "trajectory_impulse_score": None if impulse_score is None else round(impulse_score, 6),
        "trajectory_x_range_window_px": None if x_range is None else round(x_range, 6),
        "trajectory_y_range_window_px": None if y_range is None else round(y_range, 6),
        "trajectory_confidence_mean_window": None if conf_mean is None else round(conf_mean, 6),
        "trajectory_local_y_quad_rms_px": None if y_quad_rms is None else round(y_quad_rms, 6),
    }


def attach_row_features(row: dict[str, Any], track: TrackFeatures, *, break_tolerance_sec: float) -> dict[str, Any]:
    out = dict(row)
    time_sec = float(row["candidate_time_sec"])
    if track.status != "ok":
        out.update(
            {
                "trajectory_feature_status": track.status,
                "trajectory_break_support": None,
                "trajectory_nearest_break_delta_sec": None,
                "trajectory_max_positive_dvy": None,
                "height_reversal": None,
                "detector_confidence_near_candidate": None,
                **local_track_window_features(track.ts, track.xs, track.ys, track.confidences, time_sec),
            }
        )
        return out

    nearby = [bp for bp in track.breakpoints if abs(float(bp["time_sec"]) - time_sec) <= break_tolerance_sec]
    dvy_values = [float(bp["dvy"]) for bp in nearby if bp.get("dvy") is not None]
    positive_dvy = [value for value in dvy_values if value > 0.0]
    breakpoint_times = [float(bp["time_sec"]) for bp in track.breakpoints]
    nearest = nearest_distance(time_sec, breakpoint_times)
    out.update(
        {
            "trajectory_feature_status": "ok",
            "trajectory_break_support": len(nearby),
            "trajectory_nearest_break_delta_sec": None if nearest is None else round(float(nearest), 6),
            "trajectory_max_positive_dvy": round(max(positive_dvy), 6) if positive_dvy else 0.0,
            "height_reversal": height_reversal(track.ts, track.ys, time_sec),
            "detector_confidence_near_candidate": None
            if (conf := confidence_near(track.ts, track.confidences, time_sec)) is None
            else round(float(conf), 6),
            **local_track_window_features(track.ts, track.xs, track.ys, track.confidences, time_sec),
        }
    )
    return out


def attach_features_to_rows(
    rows: list[dict[str, Any]],
    tracks: dict[str, list[TrackPoint]],
    *,
    min_points: int,
    break_tolerance_sec: float,
    max_track_gap_sec: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    track_cache: dict[str, TrackFeatures] = {}
    out_rows: list[dict[str, Any]] = []
    per_video: dict[str, dict[str, Any]] = {}
    for row in rows:
        video_name = str(row["video_name"])
        if video_name not in track_cache:
            points = tracks.get(video_name, [])
            if points:
                track_cache[video_name] = compute_track_features(points, min_points=min_points, max_gap_sec=max_track_gap_sec)
            else:
                track_cache[video_name] = TrackFeatures("missing_detection_track", [], np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([]))
        updated = attach_row_features(row, track_cache[video_name], break_tolerance_sec=break_tolerance_sec)
        out_rows.append(updated)
        stats = per_video.setdefault(
            video_name,
            {
                "video_name": video_name,
                "video_id": row.get("video_id"),
                "split": row.get("split"),
                "rows": 0,
                "positive_rows": 0,
                "status": updated["trajectory_feature_status"],
                "track_points": len(tracks.get(video_name, [])),
                "track_segments": track_cache[video_name].segments,
            },
        )
        stats["rows"] += 1
        stats["positive_rows"] += int(bool(row.get("label_is_touch")))
        if updated["trajectory_feature_status"] == "ok":
            stats["ok_rows"] = int(stats.get("ok_rows", 0)) + 1
    summary = {
        "videos": list(per_video.values()),
        "rows": len(out_rows),
        "ok_rows": sum(1 for row in out_rows if row.get("trajectory_feature_status") == "ok"),
        "missing_track_rows": sum(1 for row in out_rows if row.get("trajectory_feature_status") == "missing_detection_track"),
        "too_few_track_rows": sum(1 for row in out_rows if str(row.get("trajectory_feature_status", "")).startswith("too_few")),
    }
    return out_rows, summary


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Touch L2 Feature Attachment",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Dataset dir: `{manifest['dataset_dir']}`",
        f"- Detection files: `{manifest['detection_files']}`",
        f"- Train/val rows: `{manifest['train_val']['rows']}`",
        f"- Train/val rows with L2 features: `{manifest['train_val']['ok_rows']}`",
        f"- Frozen-test rows: `{manifest['test_frozen']['rows']}`",
        f"- Frozen-test rows with L2 features: `{manifest['test_frozen']['ok_rows']}`",
        f"- Max continuous-track gap: `{manifest['max_track_gap_sec']}` sec",
        "",
        "## Per-Video",
        "",
        "| video | split | rows | positives | track points | segments | status | ok rows |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in manifest["videos"]:
        lines.append(
            f"| `{row['video_name']}` | {row.get('split')} | {row['rows']} | {row['positive_rows']} | "
            f"{row['track_points']} | {row.get('track_segments', 0)} | {row['status']} | {row.get('ok_rows', 0)} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- This step only fills candidate-level L2 features; it does not train or tune the classifier.",
            "- Tracks are split at large time gaps before arc fitting, so candidate-window exports cannot fabricate breakpoints across disjoint spans.",
            "- Rows without a matching fixed-OWLv2 track remain marked `missing_detection_track`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def attach_dataset(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else dataset_dir
    paths = detection_paths(args)
    tracks = load_detection_tracks(paths, args.threshold)
    train_rows = read_jsonl(dataset_dir / "touch_training_candidates.jsonl")
    test_rows = read_jsonl(dataset_dir / "touch_training_test_frozen.jsonl")
    train_out, train_summary = attach_features_to_rows(
        train_rows,
        tracks,
        min_points=args.min_points,
        break_tolerance_sec=args.break_tolerance_sec,
        max_track_gap_sec=args.max_track_gap_sec,
    )
    test_out, test_summary = attach_features_to_rows(
        test_rows,
        tracks,
        min_points=args.min_points,
        break_tolerance_sec=args.break_tolerance_sec,
        max_track_gap_sec=args.max_track_gap_sec,
    )
    if any(row.get("split") == "test_frozen" for row in train_out):
        raise AssertionError("test_frozen row leaked into train/validation output")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "touch_training_candidates.jsonl", train_out)
    write_jsonl(out_dir / "touch_training_test_frozen.jsonl", test_out)
    videos = train_summary["videos"] + test_summary["videos"]
    status = "features_attached" if train_summary["ok_rows"] or test_summary["ok_rows"] else "no_matching_detection_tracks"
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "threshold": args.threshold,
        "break_tolerance_sec": args.break_tolerance_sec,
        "max_track_gap_sec": args.max_track_gap_sec,
        "min_points": args.min_points,
        "detection_files": [str(path) for path in paths],
        "track_videos": sorted(tracks.keys()),
        "train_val": train_summary,
        "test_frozen": test_summary,
        "videos": videos,
    }
    write_json(out_dir / "touch_l2_feature_manifest.json", manifest)
    write_report(out_dir / "touch_l2_feature_report.md", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach L2 trajectory features to touch training candidates")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--detections-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--detections-dir", type=Path, action="append", default=[])
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--break-tolerance-sec", type=float, default=AUDIO_FUSION_BREAK_TOL_SEC)
    parser.add_argument("--max-track-gap-sec", type=float, default=DEFAULT_MAX_TRACK_GAP_SEC)
    parser.add_argument("--min-points", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    manifest = attach_dataset(parse_args())
    out_dir = Path(manifest["out_dir"])
    print(f"manifest: {out_dir / 'touch_l2_feature_manifest.json'}")
    print(f"report:   {out_dir / 'touch_l2_feature_report.md'}")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "train_val_rows": manifest["train_val"]["rows"],
                "train_val_ok_rows": manifest["train_val"]["ok_rows"],
                "test_frozen_rows": manifest["test_frozen"]["rows"],
                "test_frozen_ok_rows": manifest["test_frozen"]["ok_rows"],
                "track_videos": len(manifest["track_videos"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
