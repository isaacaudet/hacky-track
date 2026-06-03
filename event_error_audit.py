#!/usr/bin/env python3
"""Render and classify event-level touch classifier mistakes.

This audit consumes the merged event output from `train_touch_classifier.py`
and produces one visual strip per event-level false positive/false negative.
It is intentionally diagnostic: it does not relabel data or retrain a model.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "runs/release-27-public/touch_corpus_v1"
DEFAULT_EVENTS_JSONL = DEFAULT_CORPUS / "touch_classifier_v1/touch_classifier_oof_events.jsonl"
DEFAULT_PREDICTIONS_JSONL = DEFAULT_CORPUS / "touch_classifier_v1/touch_classifier_oof_predictions.jsonl"
DEFAULT_REVIEW_MANIFEST = DEFAULT_CORPUS / "touch_review_manifest.json"
DEFAULT_LABELS_DIR = DEFAULT_CORPUS / "visual_touch_labels"
DEFAULT_DETECTIONS_JSONL = DEFAULT_CORPUS / "owlv2_touch_detections_v1/detections.jsonl"
DEFAULT_TRACK_ROOT = ROOT / "runs/release-27-public/detector_inference_v10_calibrated_batch_300_processed_temporal"
DEFAULT_OUT_DIR = DEFAULT_CORPUS / "event_error_audit_v1"
DETECTION_THRESHOLD = 0.2


@dataclass(frozen=True)
class TrackPoint:
    time_sec: float
    x: float
    y: float
    confidence: float
    frame_index: int | None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def label_path(labels_dir: Path, video_id: str) -> Path:
    return labels_dir / f"{video_id}.events.json"


def approved_event_times(labels_dir: Path, video_id: str, event_type: str) -> list[float]:
    path = label_path(labels_dir, video_id)
    if not path.exists():
        return []
    doc = read_json(path)
    out = []
    for rally in doc.get("rallies", []):
        for event in rally.get("events", []):
            if event.get("review_status") == "approved" and event.get("type") == event_type and event.get("time_sec") is not None:
                out.append(float(event["time_sec"]))
    return sorted(out)


def approved_stall_windows(labels_dir: Path, video_id: str) -> list[tuple[float, float]]:
    path = label_path(labels_dir, video_id)
    if not path.exists():
        return []
    doc = read_json(path)
    out = []
    for rally in doc.get("rallies", []):
        for event in rally.get("events", []):
            if event.get("review_status") != "approved" or event.get("type") != "stall" or event.get("time_sec") is None:
                continue
            start = float(event["time_sec"])
            duration = float(event.get("duration_sec") or 0.5)
            out.append((start, start + max(0.0, duration)))
    return sorted(out)


def nearest_delta(time_sec: float, times: list[float]) -> float | None:
    if not times:
        return None
    return min(abs(time_sec - item) for item in times)


def nearest_window_delta(time_sec: float, windows: list[tuple[float, float]]) -> float | None:
    if not windows:
        return None
    distances = []
    for start, end in windows:
        if start <= time_sec <= end:
            distances.append(0.0)
        else:
            distances.append(min(abs(time_sec - start), abs(time_sec - end)))
    return min(distances)


def top_detection(row: dict[str, Any]) -> TrackPoint | None:
    detections = row.get("detections") or []
    above = [det for det in detections if safe_float(det.get("score"), 0.0) >= DETECTION_THRESHOLD]
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


def load_tracks(path: Path, video_ids: set[str] | None = None) -> dict[str, list[TrackPoint]]:
    tracks: dict[str, list[TrackPoint]] = defaultdict(list)
    for row in iter_jsonl(path):
        video_id = str(row.get("video_id") or "")
        if not video_id:
            continue
        if video_ids is not None and video_id not in video_ids:
            continue
        point = top_detection(row)
        if point is not None:
            tracks[video_id].append(point)
    for video_id, points in tracks.items():
        deduped: dict[float, TrackPoint] = {}
        for point in points:
            key = round(point.time_sec, 6)
            prior = deduped.get(key)
            if prior is None or point.confidence > prior.confidence:
                deduped[key] = point
        tracks[video_id] = sorted(deduped.values(), key=lambda item: item.time_sec)
    return dict(tracks)


def load_per_video_tracks(track_root: Path, video_paths: dict[str, str], video_ids: set[str]) -> dict[str, list[TrackPoint]]:
    tracks: dict[str, list[TrackPoint]] = {}
    for video_id in sorted(video_ids):
        video_path = video_paths.get(video_id)
        if not video_path:
            continue
        stem = Path(video_path).stem
        candidates = [
            track_root / stem / "detector_track.json",
            track_root / video_id / "detector_track.json",
            track_root / video_id.replace("-", " ") / "detector_track.json",
        ]
        track_path = next((item for item in candidates if item.exists()), None)
        if track_path is None:
            continue
        doc = read_json(track_path)
        points = []
        for row in doc.get("track", []):
            center = row.get("center")
            if not center or len(center) < 2 or row.get("time_sec") is None:
                continue
            points.append(
                TrackPoint(
                    time_sec=float(row["time_sec"]),
                    x=float(center[0]),
                    y=float(center[1]),
                    confidence=float(row.get("confidence") or 0.0),
                    frame_index=None if row.get("frame_index") is None else int(row["frame_index"]),
                )
            )
        if points:
            tracks[video_id] = sorted(points, key=lambda item: item.time_sec)
    return tracks


def load_video_paths(path: Path) -> dict[str, str]:
    doc = read_json(path)
    return {str(item["video_id"]): str(item["video_path"]) for item in doc.get("items", [])}


def predictions_by_video(path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        out[str(row.get("video_id"))].append(row)
    for rows in out.values():
        rows.sort(key=lambda row: float(row.get("candidate_time_sec") or 0.0))
    return dict(out)


def classify_failure_mode(
    event: dict[str, Any],
    *,
    touch_delta: float | None,
    stall_delta: float | None,
) -> str:
    event_type = str(event.get("event_match_type"))
    if event_type == "false_negative":
        nearest_pred = safe_float(event.get("nearest_predicted_delta_sec"))
        if nearest_pred is not None and nearest_pred <= 0.45:
            return "duplicate-after-touch not merged enough"
        return "trajectory artifact"

    if touch_delta is not None and 0.20 < touch_delta <= 0.45:
        return "duplicate-after-touch not merged enough"
    if stall_delta is not None and stall_delta <= 0.35:
        return "stall/control mistaken as touch"

    audio_strength = safe_float(event.get("audio_strength"), 0.0) or 0.0
    break_support = safe_float(event.get("trajectory_break_support"), 0.0) or 0.0
    local_rms = safe_float(event.get("trajectory_local_y_quad_rms_px"), 0.0) or 0.0
    if local_rms >= 35.0 or break_support >= 5.0:
        return "trajectory artifact"
    if audio_strength >= 10.0 and break_support >= 1.0:
        return "loud footstep with ball motion nearby"
    if touch_delta is not None and touch_delta <= 0.60:
        return "true label ambiguity"
    return "trajectory artifact"


def track_window(points: list[TrackPoint], time_sec: float, window_sec: float) -> list[TrackPoint]:
    return [point for point in points if time_sec - window_sec <= point.time_sec <= time_sec + window_sec]


def nearest_track_point(points: list[TrackPoint], time_sec: float) -> TrackPoint | None:
    if not points:
        return None
    return min(points, key=lambda point: abs(point.time_sec - time_sec))


def read_frame(video_path: str, time_sec: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_sec) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame


def resize_letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(width / max(1, w), height / max(1, h))
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def render_strip(
    *,
    event: dict[str, Any],
    video_path: str,
    track: list[TrackPoint],
    touch_times: list[float],
    stall_windows: list[tuple[float, float]],
    mode: str,
    out_path: Path,
    window_sec: float,
) -> None:
    event_time = float(event["time_sec"])
    frame_times = [event_time - 0.40, event_time - 0.20, event_time, event_time + 0.20, event_time + 0.40]
    frame_w, frame_h = 260, 360
    header_h, graph_h = 120, 220
    width = frame_w * len(frame_times)
    height = header_h + frame_h + graph_h
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    title = (
        f"{event.get('video_id')}  {event.get('event_match_type')}  t={event_time:.3f}s  "
        f"mode={mode}  conf={event.get('confidence')}"
    )
    cv2.putText(canvas, title[:150], (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 2, cv2.LINE_AA)
    detail = (
        f"audio={event.get('audio_strength')} break={event.get('trajectory_break_support')} "
        f"break_dt={event.get('trajectory_nearest_break_delta_sec')} rms={event.get('trajectory_local_y_quad_rms_px')} "
        f"stall_dt={event.get('nearest_stall_delta_sec')}"
    )
    cv2.putText(canvas, detail[:160], (14, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (55, 55, 55), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "red=prediction/error time, green=reviewed touch, blue=stall window, cyan=detector-track ball",
        (14, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (80, 80, 80),
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
            raw_h, raw_w = frame.shape[:2]
            point = nearest_track_point(track, frame_time)
            if point is not None and abs(point.time_sec - frame_time) <= 0.08:
                cv2.circle(frame, (int(point.x), int(point.y)), 28, (255, 255, 0), 5)
            tile = resize_letterbox(frame, frame_w, frame_h)
            if point is not None and abs(point.time_sec - frame_time) <= 0.08:
                # ball already drawn in raw image before resize
                pass
        cv2.putText(tile, f"{frame_time:.2f}s", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(tile, (0, 0), (frame_w - 1, frame_h - 1), (80, 80, 80), 1)
        canvas[header_h : header_h + frame_h, x0 : x0 + frame_w] = tile

    graph_y0 = header_h + frame_h + 20
    graph_x0 = 40
    graph_w = width - 80
    graph_h_inner = graph_h - 50
    cv2.rectangle(canvas, (graph_x0, graph_y0), (graph_x0 + graph_w, graph_y0 + graph_h_inner), (210, 210, 210), 1)
    window_points = track_window(track, event_time, window_sec)
    if window_points:
        ys = [point.y for point in window_points]
        y_min, y_max = min(ys), max(ys)
        if y_max - y_min < 1:
            y_max = y_min + 1

        def sx(time_sec: float) -> int:
            return int(graph_x0 + ((time_sec - (event_time - window_sec)) / (2 * window_sec)) * graph_w)

        def sy(y_value: float) -> int:
            return int(graph_y0 + ((y_value - y_min) / (y_max - y_min)) * graph_h_inner)

        pts = [(sx(point.time_sec), sy(point.y)) for point in window_points]
        for a, b in zip(pts, pts[1:]):
            cv2.line(canvas, a, b, (70, 70, 70), 2)
        for point, xy in zip(window_points, pts):
            cv2.circle(canvas, xy, 3, (255, 180, 0), -1)
        for touch in touch_times:
            if event_time - window_sec <= touch <= event_time + window_sec:
                x = sx(touch)
                cv2.line(canvas, (x, graph_y0), (x, graph_y0 + graph_h_inner), (0, 170, 0), 2)
        for start, end in stall_windows:
            if end < event_time - window_sec or start > event_time + window_sec:
                continue
            x1 = sx(max(start, event_time - window_sec))
            x2 = sx(min(end, event_time + window_sec))
            cv2.rectangle(canvas, (x1, graph_y0), (x2, graph_y0 + graph_h_inner), (240, 210, 80), -1)
        x = sx(event_time)
        cv2.line(canvas, (x, graph_y0), (x, graph_y0 + graph_h_inner), (0, 0, 255), 3)
        cv2.putText(canvas, "ball y track", (graph_x0, graph_y0 + graph_h_inner + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1)
    else:
        cv2.putText(canvas, "no detector-track points in graph window", (graph_x0 + 20, graph_y0 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir.resolve()
    strips_dir = out_dir / "strips"
    event_rows = read_jsonl(args.events_jsonl)
    error_events = [row for row in event_rows if row.get("event_match_type") in {"false_positive", "false_negative"}]
    error_video_ids = {str(row.get("video_id")) for row in error_events}
    video_paths = load_video_paths(args.review_manifest)
    predictions = predictions_by_video(args.predictions_jsonl)
    tracks = load_per_video_tracks(args.track_root, video_paths, error_video_ids)
    missing_track_ids = error_video_ids - set(tracks)
    if missing_track_ids and args.detections_jsonl.exists():
        fallback_tracks = load_tracks(args.detections_jsonl, missing_track_ids)
        tracks.update(fallback_tracks)

    audit_rows: list[dict[str, Any]] = []
    for index, event in enumerate(error_events, start=1):
        video_id = str(event["video_id"])
        time_sec = float(event["time_sec"])
        touch_times = approved_event_times(args.labels_dir, video_id, "touch")
        stall_windows = approved_stall_windows(args.labels_dir, video_id)
        touch_delta = nearest_delta(time_sec, touch_times)
        stall_delta = nearest_window_delta(time_sec, stall_windows)
        mode = classify_failure_mode(event, touch_delta=touch_delta, stall_delta=stall_delta)
        strip_path = strips_dir / f"{index:03d}_{video_id}_{time_sec:.3f}_{event['event_match_type']}.png"
        if video_id in video_paths and not args.no_render:
            render_strip(
                event=event,
                video_path=video_paths[video_id],
                track=tracks.get(video_id, []),
                touch_times=touch_times,
                stall_windows=stall_windows,
                mode=mode,
                out_path=strip_path,
                window_sec=args.window_sec,
            )
        audit_rows.append(
            {
                **event,
                "failure_mode": mode,
                "nearest_touch_delta_sec": None if touch_delta is None else round(float(touch_delta), 6),
                "nearest_stall_delta_sec": None if stall_delta is None else round(float(stall_delta), 6),
                "strip_path": str(strip_path) if strip_path.exists() else None,
                "nearby_candidate_count": sum(
                    1 for row in predictions.get(video_id, []) if abs(float(row.get("candidate_time_sec") or 0.0) - time_sec) <= args.window_sec
                ),
            }
        )

    histogram = Counter(row["failure_mode"] for row in audit_rows)
    by_video: dict[str, Counter] = defaultdict(Counter)
    for row in audit_rows:
        by_video[str(row["video_id"])][row["failure_mode"]] += 1
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "events_jsonl": str(args.events_jsonl),
        "predictions_jsonl": str(args.predictions_jsonl),
        "detections_jsonl": str(args.detections_jsonl),
        "track_root": str(args.track_root),
        "track_videos_loaded": len(tracks),
        "track_videos_missing": sorted(error_video_ids - set(tracks)),
        "out_dir": str(out_dir),
        "error_events": len(audit_rows),
        "false_positive_events": sum(1 for row in audit_rows if row.get("event_match_type") == "false_positive"),
        "false_negative_events": sum(1 for row in audit_rows if row.get("event_match_type") == "false_negative"),
        "failure_mode_histogram": dict(histogram),
        "by_video": {video_id: dict(counter) for video_id, counter in sorted(by_video.items())},
        "rows_jsonl": str(out_dir / "event_error_audit.jsonl"),
        "report_path": str(out_dir / "event_error_audit_report.md"),
    }
    write_jsonl(out_dir / "event_error_audit.jsonl", audit_rows)
    write_json(out_dir / "event_error_audit_summary.json", summary)
    write_report(out_dir / "event_error_audit_report.md", summary, audit_rows)
    return summary


def write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Event Error Audit",
        "",
        f"- Error events: `{summary['error_events']}`",
        f"- False positives: `{summary['false_positive_events']}`",
        f"- False negatives: `{summary['false_negative_events']}`",
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
            "| type | video | time | mode | nearest touch | nearest stall | strip |",
            "| --- | --- | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        strip = Path(str(row["strip_path"])).name if row.get("strip_path") else ""
        lines.append(
            f"| {row['event_match_type']} | `{row['video_id']}` | {float(row['time_sec']):.3f} | "
            f"{row['failure_mode']} | {row.get('nearest_touch_delta_sec')} | {row.get('nearest_stall_delta_sec')} | {strip} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render and classify event-level touch classifier mistakes")
    parser.add_argument("--events-jsonl", type=Path, default=DEFAULT_EVENTS_JSONL)
    parser.add_argument("--predictions-jsonl", type=Path, default=DEFAULT_PREDICTIONS_JSONL)
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW_MANIFEST)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--detections-jsonl", type=Path, default=DEFAULT_DETECTIONS_JSONL)
    parser.add_argument("--track-root", type=Path, default=DEFAULT_TRACK_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = build_audit(parse_args())
    print(f"jsonl:   {summary['rows_jsonl']}")
    print(f"summary: {Path(summary['out_dir']) / 'event_error_audit_summary.json'}")
    print(f"report:  {summary['report_path']}")
    print(json.dumps({key: summary[key] for key in ("error_events", "false_positive_events", "false_negative_events", "failure_mode_histogram")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
