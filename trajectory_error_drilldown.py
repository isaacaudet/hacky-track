#!/usr/bin/env python3
"""Drill into missed event-level touches using candidate L2 feature windows.

This diagnostic is narrower than `event_error_audit.py`: it focuses on
event-level false negatives and explains why the output layer did not emit a
touch. It joins each missed approved touch to nearby cue candidates, their
classifier score, precision/recovery gate state, audio values, and L2 trajectory
features, then renders one visual strip per miss.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from event_error_audit import (
    DEFAULT_REVIEW_MANIFEST,
    DEFAULT_TRACK_ROOT,
    TrackPoint,
    load_per_video_tracks,
    load_video_paths,
    nearest_track_point,
    read_frame,
    read_json,
    read_jsonl,
    resize_letterbox,
    track_window,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "runs/release-27-public/touch_corpus_v1"
DEFAULT_EVENTS_JSONL = DEFAULT_CORPUS / "touch_classifier_v1/touch_classifier_oof_events.jsonl"
DEFAULT_PREDICTIONS_JSONL = DEFAULT_CORPUS / "touch_classifier_v1/touch_classifier_oof_predictions.jsonl"
DEFAULT_LABELS_DIR = DEFAULT_CORPUS / "visual_touch_labels"
DEFAULT_OUT_DIR = DEFAULT_CORPUS / "trajectory_error_drilldown_v1"
MATCH_TOL_SEC = 0.20


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def predictions_by_video(path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        out[str(row.get("video_id"))].append(row)
    for rows in out.values():
        rows.sort(key=lambda row: float(row.get("candidate_time_sec") or 0.0))
    return dict(out)


def label_events(labels_dir: Path, video_id: str) -> list[dict[str, Any]]:
    path = labels_dir / f"{video_id}.events.json"
    if not path.exists():
        return []
    doc = read_json(path)
    out = []
    for rally in doc.get("rallies", []):
        for event in rally.get("events", []):
            if event.get("review_status") == "approved" and event.get("time_sec") is not None:
                out.append(dict(event))
    return sorted(out, key=lambda row: float(row.get("time_sec") or 0.0))


def nearest_event_delta(time_sec: float, events: list[dict[str, Any]], event_type: str) -> float | None:
    times = [float(row["time_sec"]) for row in events if row.get("type") == event_type and row.get("time_sec") is not None]
    if not times:
        return None
    return min(abs(time_sec - item) for item in times)


def near_candidates(rows: list[dict[str, Any]], time_sec: float, window_sec: float) -> list[dict[str, Any]]:
    return sorted(
        [row for row in rows if abs(float(row.get("candidate_time_sec") or 0.0) - time_sec) <= window_sec],
        key=lambda row: (abs(float(row.get("candidate_time_sec") or 0.0) - time_sec), float(row.get("candidate_time_sec") or 0.0)),
    )


def candidate_snapshot(row: dict[str, Any], touch_time: float) -> dict[str, Any]:
    candidate_time = float(row.get("candidate_time_sec") or 0.0)
    return {
        "candidate_time_sec": round(candidate_time, 6),
        "delta_sec": round(candidate_time - touch_time, 6),
        "predicted_is_touch": bool(row.get("predicted_is_touch")),
        "label_is_touch": bool(row.get("label_is_touch")),
        "touch_score": safe_float(row.get("touch_score")),
        "candidate_precision_gate_reason": row.get("candidate_precision_gate_reason"),
        "candidate_recall_rescue_reason": row.get("candidate_recall_rescue_reason"),
        "audio_strength": safe_float(row.get("audio_strength")),
        "trajectory_break_support": safe_float(row.get("trajectory_break_support")),
        "trajectory_nearest_break_delta_sec": safe_float(row.get("trajectory_nearest_break_delta_sec")),
        "trajectory_impulse_score": safe_float(row.get("trajectory_impulse_score")),
        "trajectory_track_points_window": safe_float(row.get("trajectory_track_points_window")),
        "trajectory_gap_before_sec": safe_float(row.get("trajectory_gap_before_sec")),
        "trajectory_gap_after_sec": safe_float(row.get("trajectory_gap_after_sec")),
        "trajectory_x_range_window_px": safe_float(row.get("trajectory_x_range_window_px")),
        "trajectory_y_range_window_px": safe_float(row.get("trajectory_y_range_window_px")),
        "trajectory_confidence_mean_window": safe_float(row.get("trajectory_confidence_mean_window")),
        "candidate_review_decision": row.get("candidate_review_decision"),
    }


def high_impulse_rescue_signature(row: dict[str, Any]) -> bool:
    return (
        (safe_float(row.get("audio_strength"), 0.0) or 0.0) >= 10.0
        and (safe_float(row.get("trajectory_impulse_score"), 0.0) or 0.0) >= 500.0
        and (safe_float(row.get("trajectory_break_support"), 0.0) or 0.0) >= 1.0
        and (safe_float(row.get("trajectory_nearest_break_delta_sec"), 999.0) or 999.0) <= 0.10
    )


def classify_miss(
    *,
    event: dict[str, Any],
    candidates: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
) -> str:
    time_sec = float(event["time_sec"])
    exact = [row for row in candidates if abs(float(row.get("candidate_time_sec") or 0.0) - time_sec) <= MATCH_TOL_SEC]
    stall_delta = nearest_event_delta(time_sec, label_rows, "stall")
    if not exact:
        return "no_candidate_within_match_window"
    best = exact[0]
    if stall_delta is not None and stall_delta <= 0.05:
        return "touch_stall_overlap"
    if best.get("candidate_precision_gate_reason") == "no_trajectory_corroboration":
        return "vetoed_no_trajectory_corroboration"
    points = safe_float(best.get("trajectory_track_points_window"), 0.0) or 0.0
    gap_before = safe_float(best.get("trajectory_gap_before_sec"), 999.0) or 999.0
    gap_after = safe_float(best.get("trajectory_gap_after_sec"), 999.0) or 999.0
    if points < 6 or gap_before > 0.12 or gap_after > 0.12:
        return "track_gap_or_missing_window"
    if high_impulse_rescue_signature(best):
        return "high_impulse_low_classifier_score"
    break_support = safe_float(best.get("trajectory_break_support"), 0.0) or 0.0
    break_delta = safe_float(best.get("trajectory_nearest_break_delta_sec"), 999.0) or 999.0
    impulse = safe_float(best.get("trajectory_impulse_score"), 0.0) or 0.0
    if break_support <= 0 or break_delta > 0.10:
        return "missing_nearby_trajectory_break"
    if impulse < 500.0:
        return "weak_trajectory_impulse"
    return "low_classifier_score"


def render_drilldown_strip(
    *,
    event: dict[str, Any],
    candidates: list[dict[str, Any]],
    mode: str,
    video_path: str,
    track: list[TrackPoint],
    out_path: Path,
    window_sec: float,
) -> None:
    time_sec = float(event["time_sec"])
    frame_times = [time_sec - 0.40, time_sec - 0.20, time_sec, time_sec + 0.20, time_sec + 0.40]
    frame_w, frame_h = 260, 360
    header_h, graph_h, table_h = 132, 210, 150
    width = frame_w * len(frame_times)
    height = header_h + frame_h + graph_h + table_h
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    title = f"{event.get('video_id')}  missed touch  t={time_sec:.3f}s  mode={mode}"
    cv2.putText(canvas, title[:150], (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "red=missed approved touch, green=predicted candidate, orange=rejected candidate, cyan=detector-track ball",
        (14, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "table: dt score audio break bdt impulse pts gate",
        (14, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )

    for index, frame_time in enumerate(frame_times):
        x0 = index * frame_w
        frame = read_frame(video_path, frame_time)
        if frame is None:
            tile = np.full((frame_h, frame_w, 3), 30, dtype=np.uint8)
            cv2.putText(tile, "missing frame", (40, frame_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        else:
            point = nearest_track_point(track, frame_time)
            if point is not None and abs(point.time_sec - frame_time) <= 0.08:
                cv2.circle(frame, (int(point.x), int(point.y)), 28, (255, 255, 0), 5)
            tile = resize_letterbox(frame, frame_w, frame_h)
        cv2.putText(tile, f"{frame_time:.2f}s", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(tile, (0, 0), (frame_w - 1, frame_h - 1), (80, 80, 80), 1)
        canvas[header_h : header_h + frame_h, x0 : x0 + frame_w] = tile

    graph_y0 = header_h + frame_h + 20
    graph_x0 = 40
    graph_w = width - 80
    graph_h_inner = graph_h - 48
    cv2.rectangle(canvas, (graph_x0, graph_y0), (graph_x0 + graph_w, graph_y0 + graph_h_inner), (210, 210, 210), 1)
    window_points = track_window(track, time_sec, window_sec)

    def sx(t: float) -> int:
        return int(graph_x0 + ((t - (time_sec - window_sec)) / (2 * window_sec)) * graph_w)

    if window_points:
        ys = [point.y for point in window_points]
        y_min, y_max = min(ys), max(ys)
        if y_max - y_min < 1:
            y_max = y_min + 1

        def sy(y_value: float) -> int:
            return int(graph_y0 + ((y_value - y_min) / (y_max - y_min)) * graph_h_inner)

        pts = [(sx(point.time_sec), sy(point.y)) for point in window_points]
        for a, b in zip(pts, pts[1:]):
            cv2.line(canvas, a, b, (70, 70, 70), 2)
        for xy in pts:
            cv2.circle(canvas, xy, 3, (255, 180, 0), -1)
    else:
        cv2.putText(canvas, "no detector-track points in graph window", (graph_x0 + 20, graph_y0 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
    touch_x = sx(time_sec)
    cv2.line(canvas, (touch_x, graph_y0), (touch_x, graph_y0 + graph_h_inner), (0, 0, 255), 3)
    for candidate in candidates:
        candidate_time = float(candidate.get("candidate_time_sec") or 0.0)
        if not (time_sec - window_sec <= candidate_time <= time_sec + window_sec):
            continue
        x = sx(candidate_time)
        color = (0, 165, 255)
        if candidate.get("predicted_is_touch"):
            color = (0, 170, 0)
        cv2.line(canvas, (x, graph_y0), (x, graph_y0 + graph_h_inner), color, 2)
    cv2.putText(canvas, "local y track + candidate markers", (graph_x0, graph_y0 + graph_h_inner + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1)

    table_y = header_h + frame_h + graph_h + 22
    headers = ["dt", "pred", "score", "audio", "br", "bdt", "imp", "pts", "gate"]
    x_cols = [20, 105, 170, 260, 350, 410, 500, 610, 680]
    for x, header in zip(x_cols, headers):
        cv2.putText(canvas, header, (x, table_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (40, 40, 40), 1, cv2.LINE_AA)
    for row_index, candidate in enumerate(candidates[:5], start=1):
        y = table_y + row_index * 24
        values = [
            f"{float(candidate.get('delta_sec') or 0.0):+.3f}",
            "Y" if candidate.get("predicted_is_touch") else "n",
            fmt(candidate.get("touch_score"), 3),
            fmt(candidate.get("audio_strength"), 1),
            fmt(candidate.get("trajectory_break_support"), 0),
            fmt(candidate.get("trajectory_nearest_break_delta_sec"), 3),
            fmt(candidate.get("trajectory_impulse_score"), 0),
            fmt(candidate.get("trajectory_track_points_window"), 0),
            str(candidate.get("candidate_precision_gate_reason") or candidate.get("candidate_recall_rescue_reason") or "")[:28],
        ]
        color = (25, 110, 25) if candidate.get("predicted_is_touch") else (80, 80, 80)
        for x, value in zip(x_cols, values):
            cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def fmt(value: Any, decimals: int) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return "-"
    return f"{numeric:.{decimals}f}"


def build_drilldown(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir.resolve()
    strips_dir = out_dir / "strips"
    event_rows = read_jsonl(args.events_jsonl)
    fn_events = [row for row in event_rows if row.get("event_match_type") == "false_negative"]
    predictions = predictions_by_video(args.predictions_jsonl)
    video_paths = load_video_paths(args.review_manifest)
    error_video_ids = {str(row.get("video_id")) for row in fn_events}
    tracks = load_per_video_tracks(args.track_root, video_paths, error_video_ids)

    drilldown_rows: list[dict[str, Any]] = []
    for index, event in enumerate(fn_events, start=1):
        video_id = str(event["video_id"])
        time_sec = float(event["time_sec"])
        label_rows = label_events(args.labels_dir, video_id)
        candidates = near_candidates(predictions.get(video_id, []), time_sec, args.candidate_window_sec)
        snapshots = [candidate_snapshot(row, time_sec) for row in candidates]
        mode = classify_miss(event=event, candidates=candidates, label_rows=label_rows)
        strip_path = strips_dir / f"{index:03d}_{video_id}_{time_sec:.3f}_{mode}.png"
        if video_id in video_paths and not args.no_render:
            render_drilldown_strip(
                event=event,
                candidates=snapshots,
                mode=mode,
                video_path=video_paths[video_id],
                track=tracks.get(video_id, []),
                out_path=strip_path,
                window_sec=args.graph_window_sec,
            )
        exact_candidates = [row for row in snapshots if abs(float(row["delta_sec"])) <= MATCH_TOL_SEC]
        drilldown_rows.append(
            {
                **event,
                "trajectory_failure_mode": mode,
                "nearby_candidate_count": len(candidates),
                "match_window_candidate_count": len(exact_candidates),
                "nearest_candidate_delta_sec": None if not snapshots else snapshots[0]["delta_sec"],
                "nearest_candidate": None if not snapshots else snapshots[0],
                "candidate_snapshots": snapshots,
                "strip_path": str(strip_path) if strip_path.exists() else None,
            }
        )

    histogram = Counter(row["trajectory_failure_mode"] for row in drilldown_rows)
    by_video: dict[str, Counter] = defaultdict(Counter)
    for row in drilldown_rows:
        by_video[str(row["video_id"])][row["trajectory_failure_mode"]] += 1
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "events_jsonl": str(args.events_jsonl),
        "predictions_jsonl": str(args.predictions_jsonl),
        "labels_dir": str(args.labels_dir),
        "out_dir": str(out_dir),
        "false_negative_events": len(drilldown_rows),
        "failure_mode_histogram": dict(histogram),
        "by_video": {video_id: dict(counter) for video_id, counter in sorted(by_video.items())},
        "rows_jsonl": str(out_dir / "trajectory_error_drilldown.jsonl"),
        "report_path": str(out_dir / "trajectory_error_drilldown_report.md"),
    }
    write_jsonl(out_dir / "trajectory_error_drilldown.jsonl", drilldown_rows)
    write_json(out_dir / "trajectory_error_drilldown_summary.json", summary)
    write_report(out_dir / "trajectory_error_drilldown_report.md", summary, drilldown_rows)
    return summary


def write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Trajectory Error Drilldown",
        "",
        f"- False-negative events: `{summary['false_negative_events']}`",
        f"- Events: `{summary['events_jsonl']}`",
        f"- Predictions: `{summary['predictions_jsonl']}`",
        "",
        "## Failure Modes",
        "",
        "| mode | count |",
        "| --- | ---: |",
    ]
    for mode, count in sorted(summary["failure_mode_histogram"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {mode} | {count} |")
    lines.extend(["", "## By Video", "", "| video | modes |", "| --- | --- |"])
    for video_id, modes in summary["by_video"].items():
        parts = ", ".join(f"{mode}: {count}" for mode, count in sorted(modes.items()))
        lines.append(f"| `{video_id}` | {parts} |")
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| video | time | mode | candidates | nearest dt | nearest score | nearest audio | nearest break | nearest bdt | nearest impulse | strip |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        nearest = row.get("nearest_candidate") or {}
        strip = Path(str(row["strip_path"])).name if row.get("strip_path") else ""
        lines.append(
            f"| `{row['video_id']}` | {float(row['time_sec']):.3f} | {row['trajectory_failure_mode']} | "
            f"{row['nearby_candidate_count']} | {nearest.get('delta_sec')} | {fmt(nearest.get('touch_score'), 3)} | "
            f"{fmt(nearest.get('audio_strength'), 1)} | {fmt(nearest.get('trajectory_break_support'), 0)} | "
            f"{fmt(nearest.get('trajectory_nearest_break_delta_sec'), 3)} | {fmt(nearest.get('trajectory_impulse_score'), 0)} | {strip} |"
        )
    lines.extend(
        [
            "",
            "## Recommended Intervention",
            "",
            "A strict high-impulse rescue is worth testing when missed candidates have strong audio, "
            "a nearby L2 breakpoint, and a large local trajectory impulse despite a low classifier score. "
            "This is an output rule over existing L2 features; it does not change OWLv2 or bridge detector silence.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drill into missed event-level touches using L2 feature windows")
    parser.add_argument("--events-jsonl", type=Path, default=DEFAULT_EVENTS_JSONL)
    parser.add_argument("--predictions-jsonl", type=Path, default=DEFAULT_PREDICTIONS_JSONL)
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW_MANIFEST)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--track-root", type=Path, default=DEFAULT_TRACK_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--candidate-window-sec", type=float, default=0.75)
    parser.add_argument("--graph-window-sec", type=float, default=1.0)
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = build_drilldown(parse_args())
    print(f"jsonl:   {summary['rows_jsonl']}")
    print(f"summary: {Path(summary['out_dir']) / 'trajectory_error_drilldown_summary.json'}")
    print(f"report:  {summary['report_path']}")
    print(json.dumps({"false_negative_events": summary["false_negative_events"], "failure_mode_histogram": summary["failure_mode_histogram"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
