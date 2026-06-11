#!/usr/bin/env python3
"""Render release-HUD touch error strips with track, label, and candidate signals.

This consumes `release_rally_analytics.py` error rows and joins them back to the
merged classifier events, cue-level predictions, reviewed labels, and OWLv2
track. It produces one visual strip per false positive / missed touch so the
next precision rule can be chosen from actual failure modes.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from event_error_audit import (
    DEFAULT_DETECTIONS_JSONL,
    DEFAULT_LABELS_DIR,
    DEFAULT_REVIEW_MANIFEST,
    TrackPoint,
    approved_event_times,
    approved_stall_windows,
    load_tracks,
    load_video_paths,
    nearest_delta,
    nearest_track_point,
    nearest_window_delta,
    read_frame,
    read_json,
    read_jsonl,
    resize_letterbox,
    safe_float,
    track_window,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "runs/release-27-public/touch_corpus_v1"
DEFAULT_HUD_DIR = DEFAULT_CORPUS / "release_touch_hud_v1"
DEFAULT_ERRORS_JSONL = DEFAULT_HUD_DIR / "analytics/release_rally_event_errors.jsonl"
DEFAULT_FROZEN_EVENTS = DEFAULT_CORPUS / "touch_classifier_v1/touch_classifier_frozen_events.jsonl"
DEFAULT_OOF_EVENTS = DEFAULT_CORPUS / "touch_classifier_v1/touch_classifier_oof_events.jsonl"
DEFAULT_FROZEN_PREDICTIONS = DEFAULT_CORPUS / "touch_classifier_v1/touch_classifier_frozen_predictions.jsonl"
DEFAULT_OOF_PREDICTIONS = DEFAULT_CORPUS / "touch_classifier_v1/touch_classifier_oof_predictions.jsonl"
DEFAULT_OUT_DIR = DEFAULT_HUD_DIR / "event_error_audit"
DEFAULT_WINDOW_SEC = 1.0
DEFAULT_CANDIDATE_WINDOW_SEC = 0.85


def event_time(row: dict[str, Any]) -> float:
    return float(row.get("time_sec") if row.get("time_sec") is not None else row.get("candidate_time_sec") or 0.0)


def candidate_time(row: dict[str, Any]) -> float:
    return float(row.get("candidate_time_sec") if row.get("candidate_time_sec") is not None else row.get("time_sec") or 0.0)


def load_events(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return sorted(rows, key=lambda row: (str(row.get("video_id")), event_time(row)))


def rows_by_video(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row.get("video_id") or ""), []).append(row)
    for items in out.values():
        items.sort(key=event_time)
    return out


def nearest_row(rows: list[dict[str, Any]], time_sec: float, *, key: str = "time_sec", max_delta_sec: float = 0.04) -> dict[str, Any] | None:
    if not rows:
        return None
    best = min(rows, key=lambda row: abs(float(row.get(key) or 0.0) - time_sec))
    return best if abs(float(best.get(key) or 0.0) - time_sec) <= max_delta_sec else None


def nearby_candidates(rows: list[dict[str, Any]], time_sec: float, window_sec: float) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if abs(candidate_time(row) - time_sec) <= window_sec
    ]
    return sorted(selected, key=lambda row: (abs(candidate_time(row) - time_sec), candidate_time(row)))


def fmt(value: Any, decimals: int = 3) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return "-"
    return f"{numeric:.{decimals}f}"


def nearest_predicted_delta(time_sec: float, predicted_times: list[float]) -> float | None:
    if not predicted_times:
        return None
    return min(abs(time_sec - item) for item in predicted_times)


def classify_release_error(
    error: dict[str, Any],
    *,
    event_features: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
    touch_delta: float | None,
    stall_delta: float | None,
    predicted_delta: float | None,
) -> str:
    error_type = str(error.get("error_type") or "")
    feature = event_features or {}
    if error_type == "false_negative":
        if predicted_delta is not None and predicted_delta <= 0.60:
            return "duplicate-after-touch not merged enough"
        exact = [row for row in candidates if abs(candidate_time(row) - float(error["time_sec"])) <= 0.20]
        if not exact:
            return "trajectory artifact"
        best = exact[0]
        if best.get("candidate_precision_gate_reason") == "no_trajectory_corroboration":
            return "trajectory artifact"
        if best.get("candidate_precision_gate_reason") == "weak_audio_weak_trajectory":
            return "true label ambiguity"
        return "trajectory artifact"

    if stall_delta is not None and stall_delta <= 0.35:
        return "stall/control mistaken as touch"
    if touch_delta is not None and 0.20 < touch_delta <= 0.45:
        return "duplicate-after-touch not merged enough"

    audio_strength = safe_float(feature.get("audio_strength"), 0.0) or 0.0
    break_support = safe_float(feature.get("trajectory_break_support"), 0.0) or 0.0
    impulse = safe_float(feature.get("trajectory_impulse_score"), 0.0) or 0.0
    local_rms = safe_float(feature.get("trajectory_local_y_quad_rms_px"), 0.0) or 0.0
    review_decision = str((candidates[0].get("candidate_review_decision") if candidates else "") or "")

    if review_decision == "no_touch" and audio_strength >= 10.0 and break_support >= 1.0:
        return "loud footstep with ball motion nearby"
    if local_rms >= 35.0 or (break_support >= 5.0 and impulse >= 1500.0):
        return "trajectory artifact"
    if touch_delta is not None and touch_delta <= 0.75:
        return "true label ambiguity"
    return "trajectory artifact"


def draw_track_graph(
    canvas: np.ndarray,
    *,
    track: list[TrackPoint],
    center_time: float,
    window_sec: float,
    x0: int,
    y0: int,
    width: int,
    height: int,
    reviewed_touches: list[float],
    predicted_touches: list[float],
    stall_windows: list[tuple[float, float]],
    candidates: list[dict[str, Any]],
) -> None:
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (210, 210, 210), 1)

    def sx(time_sec: float) -> int:
        return int(x0 + ((time_sec - (center_time - window_sec)) / (2 * window_sec)) * width)

    for start, end in stall_windows:
        if end < center_time - window_sec or start > center_time + window_sec:
            continue
        left = sx(max(start, center_time - window_sec))
        right = sx(min(end, center_time + window_sec))
        cv2.rectangle(canvas, (left, y0), (right, y0 + height), (235, 225, 120), -1)

    points = track_window(track, center_time, window_sec)
    if points:
        ys = [point.y for point in points]
        y_min, y_max = min(ys), max(ys)
        if y_max - y_min < 1.0:
            y_max = y_min + 1.0

        def sy(value: float) -> int:
            return int(y0 + ((value - y_min) / (y_max - y_min)) * height)

        graph_points = [(sx(point.time_sec), sy(point.y)) for point in points]
        for a, b in zip(graph_points, graph_points[1:]):
            cv2.line(canvas, a, b, (65, 65, 65), 2)
        for point, xy in zip(points, graph_points):
            radius = 3 if point.confidence >= 0.2 else 2
            cv2.circle(canvas, xy, radius, (255, 180, 0), -1)
    else:
        cv2.putText(canvas, "no OWLv2 track points in graph window", (x0 + 20, y0 + height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 80), 2)

    for time_sec in reviewed_touches:
        if center_time - window_sec <= time_sec <= center_time + window_sec:
            x = sx(time_sec)
            cv2.line(canvas, (x, y0), (x, y0 + height), (0, 150, 0), 2)
    for time_sec in predicted_touches:
        if center_time - window_sec <= time_sec <= center_time + window_sec:
            x = sx(time_sec)
            cv2.line(canvas, (x, y0), (x, y0 + height), (210, 60, 210), 2)
    for candidate in candidates:
        time_sec = candidate_time(candidate)
        if center_time - window_sec <= time_sec <= center_time + window_sec:
            x = sx(time_sec)
            color = (0, 170, 0) if candidate.get("predicted_is_touch") else (0, 150, 255)
            cv2.line(canvas, (x, y0), (x, y0 + height), color, 1)
    x = sx(center_time)
    cv2.line(canvas, (x, y0), (x, y0 + height), (0, 0, 255), 3)
    cv2.putText(canvas, "ball y track: red=error, green=reviewed touch, purple=predicted event, orange=cue", (x0, y0 + height + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 55, 55), 1)


def render_error_strip(
    *,
    error: dict[str, Any],
    mode: str,
    event_features: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
    video_path: str,
    track: list[TrackPoint],
    reviewed_touches: list[float],
    predicted_touches: list[float],
    stall_windows: list[tuple[float, float]],
    out_path: Path,
    window_sec: float,
) -> None:
    time_sec = float(error["time_sec"])
    frame_times = [time_sec - 0.40, time_sec - 0.20, time_sec, time_sec + 0.20, time_sec + 0.40]
    frame_w, frame_h = 260, 360
    header_h, graph_h, table_h = 152, 232, 172
    width = frame_w * len(frame_times)
    height = header_h + frame_h + graph_h + table_h
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    feature = event_features or {}
    title = f"{error.get('video_id')}  {error.get('error_type')}  t={time_sec:.3f}s  mode={mode}"
    cv2.putText(canvas, title[:150], (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"nearest truth={error.get('nearest_truth_delta_sec')} nearest pred={error.get('nearest_prediction_delta_sec')} "
        f"audio={fmt(feature.get('audio_strength'), 1)} break={fmt(feature.get('trajectory_break_support'), 0)} "
        f"bdt={fmt(feature.get('trajectory_nearest_break_delta_sec'), 3)} impulse={fmt(feature.get('trajectory_impulse_score'), 0)}",
        (14, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (55, 55, 55),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(canvas, "cyan circle=nearest OWLv2 ball; table shows cue-level model features around this moment", (14, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Buckets: duplicate-after-touch / stall-control / loud footstep+motion / true label ambiguity / trajectory artifact", (14, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (70, 70, 70), 1, cv2.LINE_AA)

    for index, frame_time in enumerate(frame_times):
        x_tile = index * frame_w
        frame = read_frame(video_path, frame_time)
        if frame is None:
            tile = np.full((frame_h, frame_w, 3), 30, dtype=np.uint8)
            cv2.putText(tile, "missing frame", (36, frame_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        else:
            point = nearest_track_point(track, frame_time)
            if point is not None and abs(point.time_sec - frame_time) <= 0.08:
                cv2.circle(frame, (int(point.x), int(point.y)), 28, (255, 255, 0), 5)
                cv2.circle(frame, (int(point.x), int(point.y)), 6, (0, 0, 255), -1)
            tile = resize_letterbox(frame, frame_w, frame_h)
        cv2.putText(tile, f"{frame_time:.2f}s", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(tile, (0, 0), (frame_w - 1, frame_h - 1), (80, 80, 80), 1)
        canvas[header_h : header_h + frame_h, x_tile : x_tile + frame_w] = tile

    graph_y = header_h + frame_h + 20
    draw_track_graph(
        canvas,
        track=track,
        center_time=time_sec,
        window_sec=window_sec,
        x0=40,
        y0=graph_y,
        width=width - 80,
        height=graph_h - 52,
        reviewed_touches=reviewed_touches,
        predicted_touches=predicted_touches,
        stall_windows=stall_windows,
        candidates=candidates,
    )

    table_y = header_h + frame_h + graph_h + 20
    columns = [
        ("dt", 20),
        ("pred", 92),
        ("score", 148),
        ("audio", 230),
        ("br", 310),
        ("bdt", 360),
        ("imp", 438),
        ("rms", 530),
        ("review", 608),
        ("gate/rescue", 748),
    ]
    for label, x in columns:
        cv2.putText(canvas, label, (x, table_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (40, 40, 40), 1, cv2.LINE_AA)
    for row_index, candidate in enumerate(candidates[:6], start=1):
        y = table_y + row_index * 24
        gate = candidate.get("candidate_precision_gate_reason") or candidate.get("candidate_recall_rescue_reason") or ""
        values = {
            "dt": f"{candidate_time(candidate) - time_sec:+.3f}",
            "pred": "Y" if candidate.get("predicted_is_touch") else "n",
            "score": fmt(candidate.get("touch_score"), 3),
            "audio": fmt(candidate.get("audio_strength"), 1),
            "br": fmt(candidate.get("trajectory_break_support"), 0),
            "bdt": fmt(candidate.get("trajectory_nearest_break_delta_sec"), 3),
            "imp": fmt(candidate.get("trajectory_impulse_score"), 0),
            "rms": fmt(candidate.get("trajectory_local_y_quad_rms_px"), 1),
            "review": str(candidate.get("candidate_review_decision") or "")[:14],
            "gate/rescue": str(gate)[:34],
        }
        color = (30, 120, 30) if candidate.get("predicted_is_touch") else (80, 80, 80)
        for label, x in columns:
            cv2.putText(canvas, values[label], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def write_contact_sheet(paths: list[Path], out_path: Path) -> None:
    thumbs: list[np.ndarray] = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        width = 430
        height = max(1, round(image.shape[0] * width / image.shape[1]))
        thumbs.append(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return
    cols = 2
    thumb_h = max(item.shape[0] for item in thumbs)
    thumb_w = max(item.shape[1] for item in thumbs)
    rows = math.ceil(len(thumbs) / cols)
    sheet = np.full((rows * thumb_h, cols * thumb_w, 3), 245, dtype=np.uint8)
    for index, thumb in enumerate(thumbs):
        y = (index // cols) * thumb_h
        x = (index % cols) * thumb_w
        sheet[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def write_report(path: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Release Event Error Audit",
        "",
        f"- Created: `{manifest['created_at']}`",
        f"- Errors: `{manifest['error_events']}`",
        f"- Contact sheet: `{manifest.get('contact_sheet')}`",
        "",
        "## Failure Modes",
        "",
        "| mode | count |",
        "| --- | ---: |",
    ]
    for mode, count in sorted(manifest["failure_mode_histogram"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {mode} | {count} |")
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| type | video | time | mode | nearest truth | nearest pred | cue count | strip |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        strip = Path(str(row["strip_path"])).name if row.get("strip_path") else "-"
        lines.append(
            f"| `{row['error_type']}` | `{row['video_id']}` | {float(row['time_sec']):.3f} | {row['failure_mode']} | "
            f"{row.get('nearest_truth_delta_sec')} | {row.get('nearest_prediction_delta_sec')} | {row.get('nearby_candidate_count')} | `{strip}` |"
        )
    lines.extend(
        [
            "",
            "## Suggested Next Rule",
            "",
            "Use the dominant bucket from this report. If `loud footstep with ball motion nearby` dominates, test a stronger "
            "trajectory-cleanliness or candidate-review-derived negative feature. If `duplicate-after-touch not merged enough` "
            "dominates, tune event-level NMS/min-gap. If `trajectory artifact` dominates, inspect OWLv2/L2 track gaps before "
            "adding classifier rules.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_release_error_audit(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir.resolve()
    strips_dir = out_dir / "strips"
    errors = read_jsonl(args.errors_jsonl)
    if args.video_id:
        wanted = set(args.video_id)
        errors = [row for row in errors if str(row.get("video_id")) in wanted]
    event_rows = load_events(args.events_jsonl)
    prediction_rows = load_events(args.predictions_jsonl)
    events_by_video = rows_by_video(event_rows)
    predictions_by_vid = rows_by_video(prediction_rows)
    video_ids = {str(row.get("video_id")) for row in errors}
    video_paths = load_video_paths(args.review_manifest)
    tracks = load_tracks(args.detections_jsonl, video_ids)
    rendered_paths: list[Path] = []
    audit_rows: list[dict[str, Any]] = []
    for index, error in enumerate(errors, start=1):
        video_id = str(error.get("video_id") or "")
        time_sec = float(error["time_sec"])
        reviewed_touches = approved_event_times(args.labels_dir, video_id, "touch")
        stall_windows = approved_stall_windows(args.labels_dir, video_id)
        predicted_events = [
            row
            for row in events_by_video.get(video_id, [])
            if row.get("event_type") == "touch" and row.get("event_match_type") != "false_negative"
        ]
        predicted_times = [event_time(row) for row in predicted_events]
        feature_row = nearest_row(events_by_video.get(video_id, []), time_sec, max_delta_sec=0.04)
        candidate_rows = nearby_candidates(predictions_by_vid.get(video_id, []), time_sec, args.candidate_window_sec)
        touch_delta = nearest_delta(time_sec, reviewed_touches)
        stall_delta = nearest_window_delta(time_sec, stall_windows)
        pred_delta = nearest_predicted_delta(time_sec, predicted_times)
        if str(error.get("error_type")) == "false_positive":
            pred_delta = 0.0
        mode = classify_release_error(
            error,
            event_features=feature_row,
            candidates=candidate_rows,
            touch_delta=touch_delta,
            stall_delta=stall_delta,
            predicted_delta=pred_delta,
        )
        strip_path = strips_dir / f"{index:03d}_{video_id}_{time_sec:.3f}_{error['error_type']}.png"
        if not args.no_render and video_id in video_paths:
            render_error_strip(
                error=error,
                mode=mode,
                event_features=feature_row,
                candidates=candidate_rows,
                video_path=video_paths[video_id],
                track=tracks.get(video_id, []),
                reviewed_touches=reviewed_touches,
                predicted_touches=predicted_times,
                stall_windows=stall_windows,
                out_path=strip_path,
                window_sec=args.window_sec,
            )
            if strip_path.exists():
                rendered_paths.append(strip_path)
        audit_rows.append(
            {
                **error,
                "failure_mode": mode,
                "nearest_truth_delta_sec": None if touch_delta is None else round(float(touch_delta), 6),
                "nearest_prediction_delta_sec": None if pred_delta is None else round(float(pred_delta), 6),
                "nearest_stall_delta_sec": None if stall_delta is None else round(float(stall_delta), 6),
                "nearby_candidate_count": len(candidate_rows),
                "event_audio_strength": None if feature_row is None else feature_row.get("audio_strength"),
                "event_trajectory_break_support": None if feature_row is None else feature_row.get("trajectory_break_support"),
                "event_trajectory_impulse_score": None if feature_row is None else feature_row.get("trajectory_impulse_score"),
                "strip_path": str(strip_path) if strip_path.exists() else None,
            }
        )
    contact_sheet = out_dir / "release_event_error_audit_contact_sheet.jpg"
    write_contact_sheet(rendered_paths, contact_sheet)
    histogram = Counter(row["failure_mode"] for row in audit_rows)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "errors_jsonl": str(args.errors_jsonl),
        "events_jsonl": [str(path) for path in args.events_jsonl],
        "predictions_jsonl": [str(path) for path in args.predictions_jsonl],
        "detections_jsonl": str(args.detections_jsonl),
        "labels_dir": str(args.labels_dir),
        "out_dir": str(out_dir),
        "error_events": len(audit_rows),
        "false_positive_events": sum(1 for row in audit_rows if row.get("error_type") == "false_positive"),
        "false_negative_events": sum(1 for row in audit_rows if row.get("error_type") == "false_negative"),
        "failure_mode_histogram": dict(histogram),
        "track_points_loaded": sum(len(points) for points in tracks.values()),
        "rendered_strips": len(rendered_paths),
        "contact_sheet": str(contact_sheet) if contact_sheet.exists() else None,
        "rows_jsonl": str(out_dir / "release_event_error_audit.jsonl"),
        "summary_json": str(out_dir / "release_event_error_audit_summary.json"),
        "report_path": str(out_dir / "release_event_error_audit_report.md"),
    }
    write_jsonl(out_dir / "release_event_error_audit.jsonl", audit_rows)
    write_json(out_dir / "release_event_error_audit_summary.json", manifest)
    write_report(out_dir / "release_event_error_audit_report.md", manifest, audit_rows)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render release-HUD touch error strips with track and candidate signals")
    parser.add_argument("--errors-jsonl", type=Path, default=DEFAULT_ERRORS_JSONL)
    parser.add_argument("--events-jsonl", type=Path, action="append", default=[DEFAULT_FROZEN_EVENTS, DEFAULT_OOF_EVENTS])
    parser.add_argument("--predictions-jsonl", type=Path, action="append", default=[DEFAULT_FROZEN_PREDICTIONS, DEFAULT_OOF_PREDICTIONS])
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW_MANIFEST)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--detections-jsonl", type=Path, default=DEFAULT_DETECTIONS_JSONL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC)
    parser.add_argument("--candidate-window-sec", type=float, default=DEFAULT_CANDIDATE_WINDOW_SEC)
    parser.add_argument("--no-render", action="store_true")
    return parser


def main() -> None:
    summary = build_release_error_audit(build_parser().parse_args())
    print(f"jsonl:   {summary['rows_jsonl']}")
    print(f"summary: {summary['summary_json']}")
    print(f"report:  {summary['report_path']}")
    print(f"sheet:   {summary.get('contact_sheet')}")
    print(json.dumps({key: summary[key] for key in ("error_events", "false_positive_events", "false_negative_events", "failure_mode_histogram")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
