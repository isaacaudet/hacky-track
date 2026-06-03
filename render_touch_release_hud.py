#!/usr/bin/env python3
"""Render release HUD clips from merged touch-classifier event output.

The legacy HUD renderers consume QA event documents. This adapter converts the
release classifier's merged event-level JSONL into that HUD event-doc shape and
generates per-touch anchors from the fixed OWLv2/L2 track, so touch sparks are
placed on the detected ball instead of the old HSV fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from paint_hud import DEFAULT_ASSETS, generate_assets, render_video
from prototype_arc_touches import robust_clean


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "runs/release-27-public/touch_corpus_v1"
DEFAULT_FROZEN_EVENTS = DEFAULT_CORPUS / "touch_classifier_v1/touch_classifier_frozen_events.jsonl"
DEFAULT_OOF_EVENTS = DEFAULT_CORPUS / "touch_classifier_v1/touch_classifier_oof_events.jsonl"
DEFAULT_DETECTIONS = DEFAULT_CORPUS / "owlv2_touch_detections_v1/detections.jsonl"
DEFAULT_INVENTORY = DEFAULT_CORPUS / "touch_corpus_inventory.json"
DEFAULT_VISUAL_LABELS = DEFAULT_CORPUS / "visual_touch_labels"
DEFAULT_LEGACY_EVENTS_DIR = ROOT / "data"
DEFAULT_OUT_DIR = DEFAULT_CORPUS / "release_touch_hud_v1"
DEFAULT_THRESHOLD = 0.2
DEFAULT_MAX_TRACK_GAP_SEC = 0.25
DEFAULT_RALLY_GAP_SEC = 2.2
DEFAULT_SCALE = 0.5
LABEL_EVENT_TYPES = {"stall", "drop_floor"}
DEFAULT_TOUCH_OVERRIDE_TOLERANCE_SEC = 0.08


@dataclass(frozen=True)
class TrackPoint:
    time_sec: float
    x: float
    y: float
    confidence: float
    frame_index: int | None = None


@dataclass(frozen=True)
class Center:
    x: float
    y: float
    confidence: float | None
    source: str
    delta_sec: float


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_slug(value: str) -> str:
    return (
        str(value)
        .replace(" ", "-")
        .replace("/", "-")
        .replace("\\", "-")
        .replace("__", "_")
        .strip("-")
    )


def video_lookup(inventory_path: Path) -> dict[str, dict[str, Any]]:
    inventory = read_json(inventory_path)
    by_id: dict[str, dict[str, Any]] = {}
    for item in inventory.get("videos", []):
        by_id[str(item["video_id"])] = item
    return by_id


def event_video_ids(rows: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        video_id = str(row.get("video_id") or "")
        if video_id and video_id not in seen:
            seen.add(video_id)
            out.append(video_id)
    return out


def predicted_touch_events(rows: list[dict[str, Any]], video_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("video_id") or "") != video_id:
            continue
        if row.get("event_type") != "touch":
            continue
        if row.get("event_match_type") == "false_negative":
            # Diagnostic truth-only rows must never be rendered as product output.
            continue
        if row.get("confidence") is None:
            continue
        events.append(row)
    events.sort(key=lambda item: float(item.get("time_sec") or 0.0))
    return events


def load_touch_overrides(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"touch override file not found: {path}")
    doc = read_json(path)
    raw = doc.get("video_overrides", doc)
    if not isinstance(raw, dict):
        raise ValueError("touch override file must contain an object or video_overrides object")
    return {str(video_id): dict(value or {}) for video_id, value in raw.items()}


def apply_touch_overrides(
    touch_events: list[dict[str, Any]],
    video_id: str,
    overrides: dict[str, dict[str, Any]],
    *,
    tolerance_sec: float = DEFAULT_TOUCH_OVERRIDE_TOLERANCE_SEC,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    override = overrides.get(video_id) or {}
    remove_times = [float(value) for value in override.get("remove_touch_times_sec", [])]
    add_times = [float(value) for value in override.get("add_touch_times_sec", [])]
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for event in touch_events:
        time_sec = float(event.get("time_sec") or 0.0)
        if any(abs(time_sec - remove_time) <= tolerance_sec for remove_time in remove_times):
            removed.append(event)
            continue
        kept.append(event)
    for time_sec in add_times:
        if any(abs(float(event.get("time_sec") or 0.0) - time_sec) <= tolerance_sec for event in kept):
            continue
        kept.append(
            {
                "schema_version": 1,
                "video_id": video_id,
                "event_type": "touch",
                "time_sec": round(time_sec, 6),
                "confidence": 1.0,
                "event_match_type": "visual_override",
                "source": "visual_touch_override",
            }
        )
    kept.sort(key=lambda item: float(item.get("time_sec") or 0.0))
    return kept, {
        "removed_touch_times_sec": [round(float(event.get("time_sec") or 0.0), 6) for event in removed],
        "added_touch_times_sec": [round(time_sec, 6) for time_sec in add_times],
        "override_reason": override.get("reason"),
        "override_source": override.get("source"),
    }


def flatten_labeled_events(doc: dict[str, Any], *, event_types: set[str] | None = None) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for rally in doc.get("rallies", []):
        for event in rally.get("events", []):
            event_type = str(event.get("type") or "")
            if event_types is not None and event_type not in event_types:
                continue
            if event.get("review_status") not in {None, "", "approved", "reviewed"}:
                continue
            if event.get("time_sec") is None:
                continue
            events.append(
                {
                    **event,
                    "type": event_type,
                    "time_sec": float(event["time_sec"]),
                    "duration_sec": float(event.get("duration_sec") or (1.0 if event_type == "stall" else 0.0)),
                    "source": event.get("source") or doc.get("annotation_method") or "labeled_event",
                    "source_video": doc.get("source_video"),
                    "rally_id": rally.get("id"),
                }
            )
    return sorted(events, key=lambda item: float(item["time_sec"]))


def label_doc_candidates(video: dict[str, Any], labels_dir: Path, legacy_events_dir: Path) -> list[Path]:
    video_id = str(video.get("video_id") or "")
    video_name = str(video.get("video_name") or "")
    stem = Path(video_name).stem
    candidates = [
        labels_dir / f"{video_id}.events.json",
        labels_dir / f"{safe_slug(video_id)}.events.json",
    ]
    if stem:
        candidates.extend(
            [
                legacy_events_dir / f"{stem}.events.json",
                legacy_events_dir / f"{safe_slug(stem)}.events.json",
            ]
        )
    out: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(path)
    return out


def labeled_release_events(video: dict[str, Any], labels_dir: Path, legacy_events_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for path in label_doc_candidates(video, labels_dir, legacy_events_dir):
        if not path.exists():
            continue
        doc = read_json(path)
        for event in flatten_labeled_events(doc, event_types=LABEL_EVENT_TYPES):
            key = (str(event["type"]), round(float(event["time_sec"]) * 20))
            if key in seen:
                continue
            seen.add(key)
            events.append({**event, "label_source_path": str(path)})
    return sorted(events, key=lambda item: float(item["time_sec"]))


def selected_video_ids(
    frozen_rows: list[dict[str, Any]],
    oof_rows: list[dict[str, Any]],
    explicit_video_ids: list[str],
    extra_non_frozen: int,
) -> list[str]:
    if explicit_video_ids:
        return list(dict.fromkeys(explicit_video_ids))
    selected = event_video_ids(frozen_rows)
    for video_id in event_video_ids(oof_rows):
        if video_id in selected:
            continue
        selected.append(video_id)
        if len(selected) >= len(event_video_ids(frozen_rows)) + extra_non_frozen:
            break
    return selected


def top_detection(row: dict[str, Any], threshold: float) -> TrackPoint | None:
    detections = row.get("detections")
    if not isinstance(detections, list):
        return None
    above = [det for det in detections if float(det.get("score") or 0.0) >= threshold]
    if not above:
        return None
    best = max(above, key=lambda item: float(item.get("score") or 0.0))
    return TrackPoint(
        time_sec=float(row["time_sec"]),
        x=float(best["x"]),
        y=float(best["y"]),
        confidence=float(best["score"]),
        frame_index=None if row.get("frame_index") is None else int(row["frame_index"]),
    )


def stream_detection_tracks(path: Path, video_ids: set[str], threshold: float) -> dict[str, list[TrackPoint]]:
    tracks: dict[str, list[TrackPoint]] = {video_id: [] for video_id in video_ids}
    patterns = [f'"video_id": "{video_id}"' for video_id in video_ids]
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not any(pattern in line for pattern in patterns):
                continue
            row = json.loads(line)
            video_id = str(row.get("video_id") or "")
            if video_id not in video_ids:
                continue
            point = top_detection(row, threshold)
            if point is None:
                continue
            tracks[video_id].append(point)
    for video_id, points in tracks.items():
        by_time: dict[float, TrackPoint] = {}
        for point in points:
            key = round(point.time_sec, 6)
            prior = by_time.get(key)
            if prior is None or point.confidence > prior.confidence:
                by_time[key] = point
        tracks[video_id] = sorted(by_time.values(), key=lambda item: item.time_sec)
    return tracks


def split_segments(points: list[TrackPoint], max_gap_sec: float) -> list[list[TrackPoint]]:
    segments: list[list[TrackPoint]] = []
    for point in sorted(points, key=lambda item: item.time_sec):
        if not segments or point.time_sec - segments[-1][-1].time_sec > max_gap_sec:
            segments.append([point])
        else:
            segments[-1].append(point)
    return segments


def clean_track(points: list[TrackPoint], max_gap_sec: float) -> list[TrackPoint]:
    cleaned: list[TrackPoint] = []
    for segment in split_segments(points, max_gap_sec):
        if len(segment) < 5:
            continue
        ts = np.asarray([point.time_sec for point in segment], dtype=float)
        xs = np.asarray([point.x for point in segment], dtype=float)
        ys = np.asarray([point.y for point in segment], dtype=float)
        keep = robust_clean(ts, xs, ys, floor_px=18.0)
        for point, kept in zip(segment, keep):
            if bool(kept):
                cleaned.append(point)
    return sorted(cleaned, key=lambda item: item.time_sec)


def interpolate_center(points: list[TrackPoint], time_sec: float, max_gap_sec: float) -> Center | None:
    if not points:
        return None
    ordered = sorted(points, key=lambda item: item.time_sec)
    nearest = min(ordered, key=lambda item: abs(item.time_sec - time_sec))
    nearest_delta = abs(nearest.time_sec - time_sec)
    for left, right in zip(ordered, ordered[1:]):
        if left.time_sec <= time_sec <= right.time_sec and right.time_sec - left.time_sec <= max_gap_sec:
            ratio = 0.0 if right.time_sec == left.time_sec else (time_sec - left.time_sec) / (right.time_sec - left.time_sec)
            return Center(
                x=float(left.x + (right.x - left.x) * ratio),
                y=float(left.y + (right.y - left.y) * ratio),
                confidence=float(max(left.confidence, right.confidence)),
                source="l2_clean_interpolated",
                delta_sec=0.0,
            )
    if nearest_delta <= max_gap_sec:
        return Center(
            x=nearest.x,
            y=nearest.y,
            confidence=nearest.confidence,
            source="l2_clean_nearest",
            delta_sec=float(nearest_delta),
        )
    return None


def raw_nearest_center(points: list[TrackPoint], time_sec: float, max_delta_sec: float) -> Center | None:
    if not points:
        return None
    nearest = min(points, key=lambda item: abs(item.time_sec - time_sec))
    delta = abs(nearest.time_sec - time_sec)
    if delta > max_delta_sec:
        return None
    return Center(
        x=nearest.x,
        y=nearest.y,
        confidence=nearest.confidence,
        source="owlv2_nearest",
        delta_sec=float(delta),
    )


def center_for_event(
    clean_points: list[TrackPoint],
    raw_points: list[TrackPoint],
    time_sec: float,
    max_track_gap_sec: float,
) -> Center | None:
    center = interpolate_center(clean_points, time_sec, max_track_gap_sec)
    if center is not None:
        return center
    return raw_nearest_center(raw_points, time_sec, max_track_gap_sec)


def group_rallies(events: list[dict[str, Any]], rally_gap_sec: float) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for event in sorted(events, key=lambda item: float(item["time_sec"])):
        if not groups or float(event["time_sec"]) - float(groups[-1][-1]["time_sec"]) > rally_gap_sec:
            groups.append([event])
        else:
            groups[-1].append(event)
    return groups


def event_key(rally_id: int, event_type: str, touch_number: int | None, time_sec: float) -> str:
    number = touch_number if touch_number is not None else 0
    return f"r{rally_id}:{event_type}:{number}:{time_sec:.2f}"


def contact_label_for_touch(event: dict[str, Any]) -> str:
    for key in ("contact_label", "contact_type", "contact_side", "label"):
        value = str(event.get(key) or "").strip()
        if value and value not in {"release_touch", "touch", "unknown"}:
            return value
    return "release_touch"


def event_duration(event: dict[str, Any]) -> float:
    if event.get("duration_sec") is not None:
        return max(0.0, float(event.get("duration_sec") or 0.0))
    return 1.0 if str(event.get("type") or "") == "stall" else 0.0


def build_hud_doc_and_anchors(
    video: dict[str, Any],
    events: list[dict[str, Any]],
    raw_points: list[TrackPoint],
    clean_points: list[TrackPoint],
    *,
    rally_gap_sec: float,
    max_track_gap_sec: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    width = int(video.get("metadata", {}).get("width") or 0)
    height = int(video.get("metadata", {}).get("height") or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Missing video dimensions for {video.get('video_id')}")
    rallies: list[dict[str, Any]] = []
    anchors: list[dict[str, Any]] = []
    center_sources: dict[str, int] = {}
    missing_centers: list[dict[str, Any]] = []

    normalized_events: list[dict[str, Any]] = []
    for event in events:
        event_type = str(event.get("event_type") or event.get("type") or "touch")
        normalized_events.append({**event, "type": event_type, "time_sec": float(event["time_sec"])})

    for rally_index, group in enumerate(group_rallies(normalized_events, rally_gap_sec), start=1):
        rally_events: list[dict[str, Any]] = []
        touch_index = 0
        for event in group:
            time_sec = float(event["time_sec"])
            event_type = str(event.get("type") or "touch")
            if event_type == "touch":
                touch_index += 1
            event_touch_number = touch_index if event_type == "touch" else None
            center = center_for_event(clean_points, raw_points, time_sec, max_track_gap_sec)
            if center is None:
                missing_centers.append({"time_sec": round(time_sec, 6), "event_type": event_type, "event": event})
                continue
            key = event_key(rally_index, event_type, event_touch_number, time_sec)
            anchors.append(
                {
                    "key": key,
                    "x": round(center.x / width, 6),
                    "y": round(center.y / height, 6),
                    "source": center.source,
                    "confidence": None if center.confidence is None else round(float(center.confidence), 6),
                    "center_delta_sec": round(center.delta_sec, 6),
                    "center_x_px": round(center.x, 3),
                    "center_y_px": round(center.y, 3),
                }
            )
            center_sources[center.source] = center_sources.get(center.source, 0) + 1
            if event_type == "touch":
                rally_events.append(
                    {
                        "type": "touch",
                        "touch_number": event_touch_number,
                        "time_sec": round(time_sec, 6),
                        "label": contact_label_for_touch(event),
                        "confidence": round(float(event.get("confidence") or 0.0), 6),
                        "event_match_type": event.get("event_match_type"),
                        "audio_strength": event.get("audio_strength"),
                        "trajectory_break_support": event.get("trajectory_break_support"),
                        "trajectory_impulse_score": event.get("trajectory_impulse_score"),
                        "center_source": center.source,
                        "qa_ball_x": round(center.x, 3),
                        "qa_ball_y": round(center.y, 3),
                    }
                )
            elif event_type in LABEL_EVENT_TYPES:
                rally_events.append(
                    {
                        "type": event_type,
                        "time_sec": round(time_sec, 6),
                        "duration_sec": round(event_duration(event), 6),
                        "label": str(event.get("label") or event_type),
                        "confidence": None if event.get("confidence") is None else round(float(event.get("confidence") or 0.0), 6),
                        "review_status": event.get("review_status"),
                        "source": event.get("source"),
                        "label_source_path": event.get("label_source_path"),
                        "center_source": center.source,
                        "qa_ball_x": round(center.x, 3),
                        "qa_ball_y": round(center.y, 3),
                    }
                )
        if not rally_events:
            continue
        touch_count = sum(1 for item in rally_events if item["type"] == "touch")
        stall_count = sum(1 for item in rally_events if item["type"] == "stall")
        start_sec = max(0.0, float(rally_events[0]["time_sec"]) - 0.2)
        end_sec = min(
            float(video.get("metadata", {}).get("duration_sec") or float(rally_events[-1]["time_sec"]) + 0.5),
            float(rally_events[-1]["time_sec"]) + 0.45,
        )
        rallies.append(
            {
                "id": rally_index,
                "label": f"release rally {rally_index}",
                "start_sec": round(start_sec, 6),
                "end_sec": round(end_sec, 6),
                "expected_touches": touch_count,
                "expected_stalls": stall_count,
                "events": rally_events,
            }
        )

    flat_events = [event for rally in rallies for event in rally["events"]]
    flat_touch_events = [event for event in flat_events if event["type"] == "touch"]
    doc = {
        "schema_version": 1,
        "source_video": video["video_name"],
        "source_video_path": video["video_path"],
        "annotation_method": "release_touch_classifier_with_reviewed_stall_drop_labels_and_owlv2_l2_centers",
        "rallies": rallies,
        "events": flat_events,
        "notes": [
            "Touch times come from merged event-level classifier output.",
            "Stall and drop_floor events come from reviewed or legacy event labels when available; they are not inferred by the release touch classifier.",
            "Touch spark anchors are generated from fixed OWLv2 detections cleaned/interpolated by L2 trajectory logic.",
            "No HSV/color fallback anchors are used for release touch events.",
        ],
    }
    anchor_doc = {
        "coordinate_space": "normalized_frame",
        "source": "fixed_owlv2_l2_track_centers",
        "notes": "Coordinates are normalized x/y positions in the source frame and are used to place HUD touch sparks.",
        "anchors": anchors,
    }
    summary = {
        "video_id": video["video_id"],
        "video_name": video["video_name"],
        "split": video["split"],
        "touch_events": len(events),
        "rendered_touch_events": len(flat_touch_events),
        "rendered_stall_events": sum(1 for event in flat_events if event["type"] == "stall"),
        "rendered_drop_floor_events": sum(1 for event in flat_events if event["type"] == "drop_floor"),
        "rallies": len(rallies),
        "raw_track_points": len(raw_points),
        "clean_l2_track_points": len(clean_points),
        "anchor_count": len(anchors),
        "missing_center_count": len(missing_centers),
        "missing_centers": missing_centers,
        "center_sources": center_sources,
    }
    return doc, anchor_doc, summary


def ffprobe_streams(video_path: Path) -> list[str]:
    if not shutil.which("ffprobe"):
        return []
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,duration",
            "-of",
            "compact=p=0:nk=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def verify_render(video_path: Path, *, require_audio: bool) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open rendered HUD video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_means: list[float] = []
    for frame_index in [0, max(0, frame_count // 2), max(0, frame_count - 1)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if ok:
            sample_means.append(round(float(frame.mean()), 3))
    cap.release()
    streams = ffprobe_streams(video_path)
    has_video = width > 0 and height > 0 and frame_count > 0
    has_audio = any("|audio|" in line for line in streams)
    nonblank = bool(sample_means and max(sample_means) > 1.0)
    if not has_video or not nonblank or (require_audio and not has_audio):
        raise RuntimeError(f"HUD video failed video/audio/nonblank verification: {video_path}")
    return {
        "path": str(video_path),
        "fps": round(fps, 3),
        "frames": frame_count,
        "duration_sec": None if fps <= 0 else round(frame_count / fps, 3),
        "width": width,
        "height": height,
        "sampled_frame_means": sample_means,
        "streams": streams,
        "has_video": has_video,
        "has_audio": has_audio,
        "nonblank": nonblank,
    }


def labeled_frame(video_path: Path, time_sec: float, label: str, width: int = 420) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(time_sec * fps))))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    height = round(frame.shape[0] * width / frame.shape[1])
    thumb = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(thumb, (0, 0), (width, 42), (0, 0, 0), -1)
    cv2.putText(thumb, label[:64], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 255, 255), 2, cv2.LINE_AA)
    return thumb


def write_preview_sheet(rows: list[dict[str, Any]], out_path: Path) -> None:
    thumbs: list[np.ndarray] = []
    for row in rows:
        verify = row.get("verification") or {}
        path = Path(str(verify.get("path") or ""))
        duration = float(verify.get("duration_sec") or 0.0)
        if not path.exists() or duration <= 0:
            continue
        for time_sec in [min(duration - 0.05, max(0.0, duration * 0.18)), min(duration - 0.05, max(0.0, duration * 0.55))]:
            frame = labeled_frame(path, time_sec, f"{row['video_id']}  {time_sec:.1f}s")
            if frame is not None:
                thumbs.append(frame)
    if not thumbs:
        return
    cols = 2
    rows_n = math.ceil(len(thumbs) / cols)
    thumb_h = max(int(thumb.shape[0]) for thumb in thumbs)
    thumb_w = max(int(thumb.shape[1]) for thumb in thumbs)
    sheet = np.full((rows_n * thumb_h, cols * thumb_w, 3), 245, dtype=np.uint8)
    for index, thumb in enumerate(thumbs):
        x = (index % cols) * thumb_w
        y = (index // cols) * thumb_h
        h, w = thumb.shape[:2]
        sheet[y : y + h, x : x + w] = thumb
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Release Touch HUD Render",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Created: `{manifest['created_at']}`",
        f"- Videos rendered: `{len(manifest['videos'])}`",
        f"- Preview sheet: `{manifest.get('preview_sheet')}`",
        "",
        "## Videos",
        "",
        "| video | split | touches | stalls | drops | rallies | anchors | center sources | video/audio/nonblank | output |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in manifest["videos"]:
        verify = row.get("verification") or {}
        status = f"{bool(verify.get('has_video'))}/{bool(verify.get('has_audio'))}/{bool(verify.get('nonblank'))}"
        sources = ", ".join(f"{k}:{v}" for k, v in sorted((row.get("center_sources") or {}).items())) or "-"
        lines.append(
            f"| `{row['video_id']}` | {row['split']} | {row['rendered_touch_events']} | "
            f"{row.get('rendered_stall_events', 0)} | {row.get('rendered_drop_floor_events', 0)} | {row['rallies']} | "
            f"{row['anchor_count']} | {sources} | {status} | `{row['output_video']}` |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Event docs are generated from `touch_classifier_*_events.jsonl` merged event-level predictions.",
            "- False-negative diagnostic rows are excluded from HUD output.",
            "- Reviewed `stall` and `drop_floor` labels are rendered when they exist, but they are not classifier predictions yet.",
            "- Touch anchors are generated from OWLv2 detections cleaned/interpolated by L2 where possible; raw OWLv2 nearest-point fallback is still detector-based, not HSV.",
            "- Any missing touch center fails the render unless `--allow-missing-centers` is passed.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_release_huds(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir.resolve()
    frozen_rows = read_jsonl(args.frozen_events)
    oof_rows = read_jsonl(args.oof_events)
    all_rows = frozen_rows + oof_rows
    inventory = video_lookup(args.inventory)
    chosen_ids = selected_video_ids(frozen_rows, oof_rows, args.video_id or [], args.extra_non_frozen)
    missing_inventory = [video_id for video_id in chosen_ids if video_id not in inventory]
    if missing_inventory:
        raise RuntimeError(f"Missing inventory entries for: {missing_inventory}")
    selected = {video_id: inventory[video_id] for video_id in chosen_ids}

    print(f"selected videos: {', '.join(chosen_ids)}")
    print(f"loading OWLv2 detections from {args.detections_jsonl} ...")
    raw_tracks = stream_detection_tracks(args.detections_jsonl, set(chosen_ids), args.threshold)
    clean_tracks = {video_id: clean_track(points, args.max_track_gap_sec) for video_id, points in raw_tracks.items()}
    touch_overrides = load_touch_overrides(args.touch_overrides)
    generate_assets(args.assets)

    video_summaries: list[dict[str, Any]] = []
    for video_id in chosen_ids:
        video = selected[video_id]
        touch_events = predicted_touch_events(all_rows, video_id)
        touch_events, override_summary = apply_touch_overrides(
            touch_events,
            video_id,
            touch_overrides,
            tolerance_sec=args.touch_override_tolerance_sec,
        )
        label_events = labeled_release_events(video, args.visual_labels_dir, args.legacy_events_dir)
        events = touch_events + label_events
        events.sort(key=lambda item: float(item["time_sec"]))
        if not touch_events:
            print(f"skip {video_id}: no predicted touch events")
            continue
        video_out = out_dir / video_id
        doc, anchors, summary = build_hud_doc_and_anchors(
            video,
            events,
            raw_tracks.get(video_id, []),
            clean_tracks.get(video_id, []),
            rally_gap_sec=args.rally_gap_sec,
            max_track_gap_sec=args.max_track_gap_sec,
        )
        if summary["missing_center_count"] and not args.allow_missing_centers:
            raise RuntimeError(f"{video_id} has missing OWLv2/L2 touch centers: {summary['missing_centers'][:5]}")
        events_path = video_out / "release_touch_hud_events.json"
        anchors_path = video_out / "release_touch_hud_anchors.json"
        write_json(events_path, doc)
        write_json(anchors_path, anchors)
        output_video = render_video(
            Path(video["video_path"]),
            video_out,
            events_path,
            anchors_path,
            args.assets,
            scale=args.scale,
            max_seconds=args.max_seconds,
        )
        verification = verify_render(output_video, require_audio=not args.allow_missing_audio)
        summary.update(
            {
                "video_path": video["video_path"],
                "events_doc": str(events_path),
                "anchors_doc": str(anchors_path),
                "output_video": str(output_video),
                "preview": str(video_out / "paint_hud_preview.jpg"),
                "contact_sheet": str(video_out / "paint_hud_contact_sheet.jpg"),
                "verification": verification,
                "touch_override": override_summary,
            }
        )
        write_json(video_out / "release_touch_hud_summary.json", summary)
        video_summaries.append(summary)
        print(f"rendered {video_id}: {output_video}")

    preview_sheet = out_dir / "release_touch_hud_preview_sheet.jpg"
    write_preview_sheet(video_summaries, preview_sheet)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed"
        if video_summaries
        and all(
            (row.get("verification") or {}).get("has_video")
            and (row.get("verification") or {}).get("nonblank")
            and ((row.get("verification") or {}).get("has_audio") or args.allow_missing_audio)
            for row in video_summaries
        )
        else "failed",
        "corpus": str(DEFAULT_CORPUS),
        "frozen_events": str(args.frozen_events),
        "oof_events": str(args.oof_events),
        "detections_jsonl": str(args.detections_jsonl),
        "visual_labels_dir": str(args.visual_labels_dir),
        "legacy_events_dir": str(args.legacy_events_dir),
        "touch_overrides": None if args.touch_overrides is None else str(args.touch_overrides),
        "touch_override_tolerance_sec": args.touch_override_tolerance_sec,
        "threshold": args.threshold,
        "scale": args.scale,
        "rally_gap_sec": args.rally_gap_sec,
        "max_track_gap_sec": args.max_track_gap_sec,
        "preview_sheet": str(preview_sheet),
        "videos": video_summaries,
    }
    write_json(out_dir / "release_touch_hud_manifest.json", manifest)
    write_report(out_dir / "release_touch_hud_report.md", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render release HUD clips from merged touch-classifier events")
    parser.add_argument("--frozen-events", type=Path, default=DEFAULT_FROZEN_EVENTS)
    parser.add_argument("--oof-events", type=Path, default=DEFAULT_OOF_EVENTS)
    parser.add_argument("--detections-jsonl", type=Path, default=DEFAULT_DETECTIONS)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--visual-labels-dir", type=Path, default=DEFAULT_VISUAL_LABELS)
    parser.add_argument("--legacy-events-dir", type=Path, default=DEFAULT_LEGACY_EVENTS_DIR)
    parser.add_argument("--touch-overrides", type=Path, default=None)
    parser.add_argument("--touch-override-tolerance-sec", type=float, default=DEFAULT_TOUCH_OVERRIDE_TOLERANCE_SEC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--extra-non-frozen", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--rally-gap-sec", type=float, default=DEFAULT_RALLY_GAP_SEC)
    parser.add_argument("--max-track-gap-sec", type=float, default=DEFAULT_MAX_TRACK_GAP_SEC)
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--allow-missing-centers", action="store_true")
    parser.add_argument("--allow-missing-audio", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = render_release_huds(args)
    print(f"status: {manifest['status']}")
    print(f"manifest: {args.out_dir.resolve() / 'release_touch_hud_manifest.json'}")
    print(f"report:   {args.out_dir.resolve() / 'release_touch_hud_report.md'}")
    print(f"preview:  {manifest.get('preview_sheet')}")


if __name__ == "__main__":
    main()
