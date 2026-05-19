#!/usr/bin/env python3
"""Local touch-review app for Hacky Track candidate labels.

The detector writes candidate touches. This app keeps a separate human-review
layer so the raw model output stays reproducible while approved/rejected/manual
labels can feed the next training pass.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
REVIEWS_DIR = ROOT / "reviews"
DOWNLOADS_DIR = Path.home() / "Downloads"
DEFAULT_TRACK_SIZE = {"width": 688, "height": 912}
DEFAULT_QA_MANIFEST = OUTPUTS_DIR / "full_training_27_qa" / "qa_manifest.json"
DEFAULT_REVIEW_BATCH = OUTPUTS_DIR / "review_batches" / "latest_review_batch.json"
DEFAULT_ASSISTED_REVIEW = OUTPUTS_DIR / "review_batches" / "assisted_review_suggestions.json"
REVIEWABLE_KINDS = {"touch", "drop_floor", "stall", "around_the_world"}
CONTACT_SIDES = ("unknown", "left", "right", "center")
CONTACT_TYPES = ("unknown", "foot", "foot_candidate", "knee_candidate", "stall", "ground", "unknown_contact")
TRICK_LABELS = (
    "",
    "right_kick",
    "left_kick",
    "right_stall",
    "left_stall",
    "outer_right",
    "outer_left",
    "around_the_world_outer_right",
    "around_the_world",
)


@dataclass(frozen=True)
class ReviewableVideo:
    stem: str
    source_video: str
    video_path: Path
    event_path: Path
    review_path: Path
    candidate_source: str
    track_size: dict[str, int]
    qa_sheet_path: Path | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def load_review_batch() -> dict[str, Any] | None:
    if not DEFAULT_REVIEW_BATCH.exists():
        return None
    try:
        return read_json(DEFAULT_REVIEW_BATCH)
    except json.JSONDecodeError:
        return None


def load_assisted_review() -> dict[str, Any] | None:
    if not DEFAULT_ASSISTED_REVIEW.exists():
        return None
    try:
        return read_json(DEFAULT_ASSISTED_REVIEW)
    except json.JSONDecodeError:
        return None


def slug_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")


def maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def event_review_tags(event: dict[str, Any]) -> list[str]:
    kind = str(event.get("type") or "unknown")
    tags: list[str] = [kind]
    accuracy = str(event.get("qa_ball_accuracy") or "").lower()
    if accuracy:
        tags.append(f"ball_{accuracy}")
    source = str(event.get("qa_ball_source") or "").lower()
    if source in {"missing", "track_fallback"}:
        tags.append("ball_fallback")
    correction = maybe_float(event.get("qa_ball_correction_px"))
    if correction is not None:
        if correction >= 125:
            tags.append("huge_ball_correction")
        elif correction >= 75:
            tags.append("large_ball_correction")
    confidence = maybe_float(event.get("confidence"))
    if confidence is not None:
        if confidence < 0.58:
            tags.append("low_confidence")
        elif confidence >= 0.86:
            tags.append("high_confidence")
    contact_type = str(event.get("contact_type") or "").lower()
    if contact_type in {"unknown", "unknown_contact", "foot_candidate", "knee_candidate", "ground_candidate"}:
        tags.append("ambiguous_contact")
    if contact_type:
        tags.append(f"contact_{contact_type}")
    side = str(event.get("contact_side") or "").lower()
    if side in {"unknown", "center"}:
        tags.append(f"side_{side or 'unknown'}")
    side_confidence = maybe_float(event.get("side_confidence"))
    if side_confidence is not None and side_confidence < 0.55:
        tags.append("low_side_confidence")
    side_reason = str(event.get("side_uncertainty_reason") or "").lower()
    if side_reason:
        tags.append(f"side_{slug_id(side_reason)}")
    foot_x = maybe_float(event.get("foot_x"))
    if side in {"left", "right"} and foot_x is not None:
        center_dist = abs(foot_x - DEFAULT_TRACK_SIZE["width"] / 2)
        if center_dist < 58:
            tags.append("side_needs_review")
    y = maybe_float(event.get("qa_ball_y", event.get("y")))
    if y is not None and y / DEFAULT_TRACK_SIZE["height"] >= 0.90:
        tags.append("near_floor")
    if kind == "touch" and y is not None and y / DEFAULT_TRACK_SIZE["height"] >= 0.90:
        tags.append("floor_risk_touch")
    if kind == "drop_floor":
        tags.append("reset_candidate")
    drop_source = str(event.get("drop_source") or "").lower()
    if drop_source:
        tags.append(f"drop_{slug_id(drop_source)}")
    label = str(event.get("label") or "").lower()
    note = str(event.get("note") or "").lower()
    text = f"{label} {note}"
    if "recovered" in text:
        tags.append("recovered_touch")
    if "hidden" in text:
        tags.append("hidden_drop")
    if "reclassified" in text:
        tags.append("reclassified")
    if "track-context" in text:
        tags.append("audio_track_context")
    duration = maybe_float(event.get("duration_sec"))
    if kind == "stall":
        tags.append("stall_window")
        if duration is not None and duration < 0.18:
            tags.append("short_stall")
    if kind == "around_the_world":
        tags.append("trick_candidate")
    return sorted(dict.fromkeys(tag for tag in tags if tag))


def find_video_path(source_video: str, event_path: Path) -> Path:
    candidates = [
        DOWNLOADS_DIR / source_video,
        ROOT / source_video,
        event_path.parent / source_video,
        event_path.parent / "source.mov",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return DOWNLOADS_DIR / source_video


def discover_videos() -> dict[str, ReviewableVideo]:
    found: dict[str, ReviewableVideo] = {}
    if not OUTPUTS_DIR.exists():
        return found

    if DEFAULT_QA_MANIFEST.exists():
        try:
            manifest = read_json(DEFAULT_QA_MANIFEST)
        except json.JSONDecodeError:
            manifest = {}
        for run in manifest.get("runs", []):
            event_path = ROOT / str(run.get("qa_events_path", ""))
            if not event_path.exists():
                event_path = Path(str(run.get("qa_events_path", "")))
            if not event_path.exists():
                continue
            source_video = Path(str(run.get("video") or "")).name
            if not source_video:
                try:
                    source_video = str(read_json(event_path).get("source_video") or event_path.parent.name)
                except json.JSONDecodeError:
                    source_video = event_path.parent.name
            raw_stem = Path(source_video).stem or event_path.parent.name
            stem = slug_id(raw_stem)
            video_path = Path(str(run.get("video") or ""))
            if not video_path.exists():
                video_path = find_video_path(source_video, event_path)
            qa_sheet_path = ROOT / str(run.get("qa_sheet_path", ""))
            if not qa_sheet_path.exists():
                qa_sheet_path = None
            found[stem] = ReviewableVideo(
                stem=stem,
                source_video=source_video,
                video_path=video_path,
                event_path=event_path,
                review_path=REVIEWS_DIR / f"{stem}.review.json",
                candidate_source="qa_events.json",
                track_size=DEFAULT_TRACK_SIZE.copy(),
                qa_sheet_path=qa_sheet_path,
            )
        if found:
            return found

    for out_dir in sorted(path for path in OUTPUTS_DIR.iterdir() if path.is_dir()):
        event_path = out_dir / "trained_events.json"
        candidate_source = "trained_events.json"
        if not event_path.exists():
            event_path = out_dir / "candidate_events.json"
            candidate_source = "candidate_events.json"
        if not event_path.exists():
            continue

        try:
            doc = read_json(event_path)
        except json.JSONDecodeError:
            continue

        source_video = str(doc.get("source_video") or f"{out_dir.name}.MOV")
        raw_stem = Path(source_video).stem or out_dir.name
        stem = slug_id(raw_stem)
        if stem in found:
            continue
        found[stem] = ReviewableVideo(
            stem=stem,
            source_video=source_video,
            video_path=find_video_path(source_video, event_path),
            event_path=event_path,
            review_path=REVIEWS_DIR / f"{stem}.review.json",
            candidate_source=candidate_source,
            track_size=DEFAULT_TRACK_SIZE.copy(),
        )
    return found


def candidate_item(event: dict[str, Any], event_path: Path, rally_id: int | None = None) -> dict[str, Any] | None:
    kind = str(event.get("type") or "")
    if kind not in REVIEWABLE_KINDS:
        return None
    if rally_id is None:
        rally_id = event.get("qa_rally_id") or event.get("rally_id")
    rally_id_int = int(rally_id or 0)
    touch_number = event.get("qa_touch_index", event.get("touch_number"))
    touch_number_int = int(touch_number or 0)
    time_sec = round(float(event.get("time_sec", event.get("start_sec", 0.0)) or 0.0), 3)
    suffix = f"{kind[:5]}-{int(round(time_sec * 1000)):07d}"
    if kind == "touch":
        suffix = f"t{touch_number_int:03d}-{int(round(time_sec * 1000)):07d}"
    item_id = f"cand-r{rally_id_int:03d}-{suffix}"
    x = event.get("qa_ball_x", event.get("x"))
    y = event.get("qa_ball_y", event.get("y"))
    item = {
        "id": item_id,
        "source": "candidate",
        "candidate_file": str(event_path.relative_to(ROOT)) if event_path.is_relative_to(ROOT) else str(event_path),
        "kind": kind,
        "status": "pending",
        "rally_id": rally_id_int or None,
        "touch_number": touch_number_int or None,
        "time_sec": time_sec,
        "start_sec": None if event.get("start_sec") is None else round(float(event["start_sec"]), 3),
        "end_sec": None if event.get("end_sec") is None else round(float(event["end_sec"]), 3),
        "duration_sec": None if event.get("duration_sec") is None else round(float(event["duration_sec"]), 3),
        "confidence": event.get("confidence"),
        "audio_z": event.get("audio_z"),
        "visual_score": event.get("visual_score"),
        "motion_score": event.get("motion_score"),
        "x": x,
        "y": y,
        "ball_confidence": event.get("qa_ball_confidence"),
        "ball_accuracy": event.get("qa_ball_accuracy"),
        "foot_x": event.get("foot_x"),
        "foot_y": event.get("foot_y"),
        "foot_confidence": event.get("foot_confidence"),
        "foot_distance": event.get("foot_distance"),
        "contact_side": event.get("contact_side") or "unknown",
        "contact_type": event.get("contact_type") or ("ground" if kind == "drop_floor" else "unknown"),
        "side_confidence": event.get("side_confidence"),
        "side_source": event.get("side_source"),
        "side_uncertainty_reason": event.get("side_uncertainty_reason"),
        "detector_contact_side": event.get("contact_side") or "unknown",
        "detector_contact_type": event.get("contact_type") or ("ground" if kind == "drop_floor" else "unknown"),
        "trick_label": event.get("trick_label") or (event.get("label") if kind == "around_the_world" else ""),
        "detector_label": event.get("label"),
        "drop_source": event.get("drop_source"),
        "drop_score": event.get("drop_score"),
        "review_tags": event_review_tags(event),
        "note": "",
    }
    return item


def batch_records_for(info: ReviewableVideo) -> dict[str, dict[str, Any]]:
    batch = load_review_batch()
    if not batch:
        return {}
    records: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(batch.get("items", []), start=1):
        if record.get("review_stem") != info.stem and record.get("source_video") != info.source_video:
            continue
        item_id = str(record.get("review_item_id") or "")
        if not item_id:
            continue
        item = dict(record)
        item["batch_index"] = index
        item["batch_total"] = len(batch.get("items", []))
        item["batch_id"] = batch.get("batch_id")
        records[item_id] = item
    return records


def assisted_records_for(info: ReviewableVideo) -> dict[str, dict[str, Any]]:
    assisted = load_assisted_review()
    if not assisted:
        return {}
    records: dict[str, dict[str, Any]] = {}
    for record in assisted.get("items", []):
        if record.get("review_stem") != info.stem and record.get("source_video") != info.source_video:
            continue
        item_id = str(record.get("review_item_id") or "")
        if item_id:
            records[item_id] = dict(record)
    return records


def review_item_from_active_learning_record(record: dict[str, Any]) -> dict[str, Any] | None:
    if record.get("source") != "active_learning":
        return None
    kind = str(record.get("kind") or "touch")
    if kind not in REVIEWABLE_KINDS:
        return None
    time_sec = round(float(record.get("time_sec") or 0.0), 3)
    item_id = str(record.get("review_item_id") or f"miss-{kind}-{int(round(time_sec * 1000)):07d}")
    duration = record.get("duration_sec")
    if duration is None and kind == "stall" and record.get("end_sec") is not None:
        duration = round(float(record["end_sec"]) - time_sec, 3)
    tags = list(record.get("review_tags") or [])
    for tag in ["active_learning", f"likely_missed_{kind}"]:
        if tag not in tags:
            tags.append(tag)
    return {
        "id": item_id,
        "source": "manual",
        "active_learning_candidate": True,
        "kind": kind,
        "status": "pending",
        "proposed_status": "missing",
        "rally_id": record.get("rally_id"),
        "touch_number": record.get("touch_number"),
        "time_sec": time_sec,
        "start_sec": record.get("start_sec") if record.get("start_sec") is not None else (time_sec if kind == "stall" else None),
        "end_sec": record.get("end_sec"),
        "duration_sec": duration,
        "confidence": record.get("confidence"),
        "audio_z": record.get("audio_z"),
        "visual_score": record.get("visual_score"),
        "motion_score": record.get("motion_score"),
        "x": record.get("qa_ball_x"),
        "y": record.get("qa_ball_y"),
        "ball_confidence": record.get("qa_ball_confidence"),
        "ball_accuracy": record.get("qa_ball_accuracy"),
        "foot_x": record.get("foot_x"),
        "foot_y": record.get("foot_y"),
        "foot_confidence": record.get("foot_confidence"),
        "foot_distance": record.get("foot_distance"),
        "contact_side": record.get("contact_side") or "unknown",
        "contact_type": record.get("contact_type") or ("ground" if kind == "drop_floor" else "stall" if kind == "stall" else "unknown"),
        "side_confidence": record.get("side_confidence"),
        "side_source": record.get("side_source"),
        "side_uncertainty_reason": record.get("side_uncertainty_reason"),
        "detector_contact_side": record.get("contact_side") or "unknown",
        "detector_contact_type": record.get("contact_type") or ("ground" if kind == "drop_floor" else "stall" if kind == "stall" else "unknown"),
        "trick_label": record.get("trick_label") or ("around_the_world" if kind == "around_the_world" else ""),
        "drop_source": record.get("drop_source"),
        "drop_score": record.get("drop_score"),
        "review_tags": tags,
        "note": f"active_learning proposal: {record.get('qa_suppressed_reason') or record.get('note') or 'likely missed event'}",
    }


def apply_batch_metadata(info: ReviewableVideo, items: list[dict[str, Any]]) -> None:
    records = batch_records_for(info)
    assisted_records = assisted_records_for(info)
    if not records and not assisted_records:
        return
    existing_ids = {str(item.get("id") or "") for item in items}
    for item in items:
        item_id = str(item.get("id") or "")
        record = records.get(item_id)
        assisted = assisted_records.get(item_id)
        if not record:
            item["in_review_batch"] = False
        else:
            tags = list(item.get("review_tags") or [])
            if "review_batch" not in tags:
                tags.insert(0, "review_batch")
            item.update(
                {
                    "in_review_batch": True,
                    "batch_id": record.get("batch_id"),
                    "batch_item_id": record.get("batch_item_id"),
                    "batch_index": record.get("batch_index"),
                    "batch_total": record.get("batch_total"),
                    "batch_priority_score": record.get("priority_score"),
                    "batch_crop_path": record.get("crop_path"),
                    "batch_tile_path": record.get("tile_path"),
                    "batch_selection_reasons": record.get("selection_reasons", []),
                    "review_tags": tags,
                }
            )
    for item_id, record in records.items():
        if item_id in existing_ids:
            continue
        item = review_item_from_active_learning_record(record)
        if item is None:
            continue
        assisted = assisted_records.get(item_id)
        tags = list(item.get("review_tags") or [])
        if "review_batch" not in tags:
            tags.insert(0, "review_batch")
        item.update(
            {
                "in_review_batch": True,
                "batch_id": record.get("batch_id"),
                "batch_item_id": record.get("batch_item_id"),
                "batch_index": record.get("batch_index"),
                "batch_total": record.get("batch_total"),
                "batch_priority_score": record.get("priority_score"),
                "batch_crop_path": record.get("crop_path"),
                "batch_tile_path": record.get("tile_path"),
                "batch_selection_reasons": record.get("selection_reasons", []),
                "review_tags": tags,
            }
        )
        if assisted:
            bucket = str(assisted.get("bucket") or "")
            if bucket:
                tag = f"assist_{bucket}"
                if tag not in tags:
                    tags.append(tag)
            item.update(
                {
                    "assist_bucket": assisted.get("bucket"),
                    "assist_action": assisted.get("recommended_action"),
                    "assist_score": assisted.get("evidence_score"),
                    "assist_reasons": assisted.get("reasons", []),
                    "review_tags": tags,
                }
            )
        items.append(item)
        if assisted:
            tags = list(item.get("review_tags") or [])
            bucket = str(assisted.get("bucket") or "")
            if bucket:
                tag = f"assist_{bucket}"
                if tag not in tags:
                    tags.append(tag)
            item.update(
                {
                    "assist_bucket": assisted.get("bucket"),
                    "assist_action": assisted.get("recommended_action"),
                    "assist_score": assisted.get("evidence_score"),
                    "assist_reasons": assisted.get("reasons", []),
                    "review_tags": tags,
                }
            )


def initialize_review_doc(info: ReviewableVideo) -> dict[str, Any]:
    event_doc = read_json(info.event_path)
    items: list[dict[str, Any]] = []
    if isinstance(event_doc.get("events"), list):
        for event in event_doc.get("events", []):
            item = candidate_item(event, info.event_path)
            if item:
                items.append(item)
    else:
        for rally in event_doc.get("rallies", []):
            rally_id = int(rally.get("id") or 0)
            for event in rally.get("events", []):
                item = candidate_item(event, info.event_path, rally_id)
                if item:
                    items.append(item)

    items.sort(key=lambda item: (float(item.get("time_sec") or 0.0), item.get("id", "")))
    apply_batch_metadata(info, items)
    return {
        "version": 1,
        "source_video": info.source_video,
        "video_path": str(info.video_path),
        "candidate_source": str(info.event_path.relative_to(ROOT)) if info.event_path.is_relative_to(ROOT) else str(info.event_path),
        "qa_sheet_path": None if info.qa_sheet_path is None else str(info.qa_sheet_path),
        "track_size": info.track_size,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "items": items,
    }


def merged_review_doc(info: ReviewableVideo) -> dict[str, Any]:
    base = initialize_review_doc(info)
    if not info.review_path.exists():
        return base

    try:
        existing = read_json(info.review_path)
    except json.JSONDecodeError:
        return base

    existing_by_id = {str(item.get("id")): item for item in existing.get("items", []) if item.get("id")}
    merged: list[dict[str, Any]] = []
    base_ids: set[str] = set()

    for item in base["items"]:
        item_id = str(item["id"])
        base_ids.add(item_id)
        previous = existing_by_id.get(item_id)
        if previous:
            merged_item = {**item, **previous}
            merged.append(merged_item)
        else:
            merged.append(item)

    for item in existing.get("items", []):
        item_id = str(item.get("id") or "")
        if item_id and item_id not in base_ids and item.get("source") == "manual":
            merged.append(item)

    merged.sort(key=lambda item: (float(item.get("time_sec") or 0.0), item.get("id", "")))
    base["items"] = merged
    base["created_at"] = existing.get("created_at") or base["created_at"]
    base["updated_at"] = existing.get("updated_at") or base["updated_at"]
    return base


def review_summary(info: ReviewableVideo) -> dict[str, Any]:
    doc = merged_review_doc(info)
    counts = {"pending": 0, "approved": 0, "rejected": 0, "missing": 0}
    kind_counts: dict[str, int] = {}
    batch_total = 0
    batch_decided = 0
    batch_pending = 0
    for item in doc.get("items", []):
        status = str(item.get("status") or "pending")
        if status in counts:
            counts[status] += 1
        if item.get("in_review_batch"):
            batch_total += 1
            if status in {"approved", "rejected", "missing"}:
                batch_decided += 1
            elif status == "pending":
                batch_pending += 1
        kind = str(item.get("kind") or "unknown")
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    return {
        "stem": info.stem,
        "source_video": info.source_video,
        "video_exists": info.video_path.exists(),
        "video_path": str(info.video_path),
        "events_path": str(info.event_path.relative_to(ROOT)) if info.event_path.is_relative_to(ROOT) else str(info.event_path),
        "review_path": str(info.review_path.relative_to(ROOT)) if info.review_path.is_relative_to(ROOT) else str(info.review_path),
        "candidate_source": info.candidate_source,
        "total_items": len(doc.get("items", [])),
        "counts": counts,
        "kind_counts": kind_counts,
        "batch_total": batch_total,
        "batch_decided": batch_decided,
        "batch_pending": batch_pending,
        "qa_sheet_path": None if info.qa_sheet_path is None else str(info.qa_sheet_path),
        "updated_at": doc.get("updated_at"),
    }


def accepted_events_doc(review_doc: dict[str, Any]) -> dict[str, Any]:
    accepted = [
        item
        for item in review_doc.get("items", [])
        if item.get("kind") in REVIEWABLE_KINDS and item.get("status") in {"approved", "missing"}
    ]
    accepted.sort(key=lambda item: float(item.get("time_sec") or 0.0))

    rallies: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    rally_id = 1
    last_time: float | None = None
    for item in accepted:
        time_sec = float(item.get("time_sec") or 0.0)
        if current and last_time is not None and time_sec - last_time > 2.2:
            rallies.append(build_rally(rally_id, current))
            rally_id += 1
            current = []
        current.append(item)
        if item.get("kind") == "drop_floor":
            rallies.append(build_rally(rally_id, current))
            rally_id += 1
            current = []
            last_time = None
        else:
            last_time = time_sec
    if current:
        rallies.append(build_rally(rally_id, current))

    return {
        "source_video": review_doc.get("source_video"),
        "annotation_method": "review_app_approved",
        "created_at": utc_now(),
        "rallies": rallies,
    }


def build_rally(rally_id: int, items: list[dict[str, Any]]) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    touch_index = 0
    for item in items:
        kind = str(item.get("kind") or "touch")
        if kind == "touch":
            touch_index += 1
        event = {
            "type": kind,
            "touch_number": touch_index if kind == "touch" else None,
            "time_sec": round(float(item.get("time_sec") or 0.0), 3),
            "start_sec": None if item.get("start_sec") is None else round(float(item.get("start_sec") or 0.0), 3),
            "end_sec": None if item.get("end_sec") is None else round(float(item.get("end_sec") or 0.0), 3),
            "duration_sec": None if item.get("duration_sec") is None else round(float(item.get("duration_sec") or 0.0), 3),
            "label": "manual_missing" if item.get("status") == "missing" else "approved_candidate",
            "review_item_id": item.get("id"),
            "confidence": item.get("confidence"),
            "audio_z": item.get("audio_z"),
            "visual_score": item.get("visual_score"),
            "motion_score": item.get("motion_score"),
            "x": item.get("x"),
            "y": item.get("y"),
            "foot_x": item.get("foot_x"),
            "foot_y": item.get("foot_y"),
            "foot_confidence": item.get("foot_confidence"),
            "foot_distance": item.get("foot_distance"),
            "contact_side": item.get("contact_side"),
            "contact_type": item.get("contact_type"),
            "side_confidence": item.get("side_confidence"),
            "side_source": item.get("side_source"),
            "side_uncertainty_reason": item.get("side_uncertainty_reason"),
            "trick_label": item.get("trick_label"),
            "review_status": item.get("status"),
            "review_note": item.get("note"),
        }
        events.append({key: value for key, value in event.items() if value is not None})

    start = float(items[0].get("time_sec") or 0.0)
    end = max(
        float(item.get("end_sec") or item.get("time_sec") or start)
        for item in items
    )
    touches = sum(1 for item in items if item.get("kind") == "touch")
    stalls = sum(1 for item in items if item.get("kind") == "stall")
    drops = sum(1 for item in items if item.get("kind") == "drop_floor")
    return {
        "id": rally_id,
        "label": f"Reviewed rally {rally_id}",
        "start_sec": round(start, 3),
        "end_sec": round(end, 3),
        "expected_touches": touches,
        "expected_stalls": stalls,
        "drop_floor_events": drops,
        "events": events,
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hacky Track Review</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101112;
      --panel: #181a1d;
      --panel-2: #202329;
      --line: #30343b;
      --text: #f1f3f4;
      --muted: #a6adb7;
      --dim: #747d8a;
      --green: #76d68a;
      --red: #ff7770;
      --yellow: #ffd65c;
      --blue: #76b9ff;
      --ink: #08090a;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      height: 100vh;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
      overflow: hidden;
    }

    button, select, input, textarea {
      font: inherit;
    }

    .shell {
      height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
      min-height: 0;
    }

    header {
      display: flex;
      gap: 14px;
      align-items: center;
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      background: #141619;
    }

    .brand {
      display: flex;
      align-items: baseline;
      gap: 10px;
      min-width: 190px;
    }

    h1 {
      margin: 0;
      font-size: 16px;
      font-weight: 780;
      letter-spacing: 0;
    }

    .save-state {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }

    .top-select {
      min-width: 280px;
      color: var(--text);
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 8px 10px;
      outline: none;
    }

    .counts {
      display: flex;
      gap: 8px;
      align-items: center;
      margin-left: auto;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--muted);
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 12px;
      white-space: nowrap;
    }

    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--dim);
      flex: 0 0 auto;
    }

    .dot.pending { background: var(--yellow); }
    .dot.approved { background: var(--green); }
    .dot.rejected { background: var(--red); }
    .dot.missing { background: var(--blue); }

    main {
      display: grid;
      grid-template-columns: minmax(420px, 1fr) 430px;
      min-height: 0;
    }

    .viewer {
      min-width: 0;
      display: grid;
      grid-template-rows: 1fr auto;
      background: #0b0c0d;
      border-right: 1px solid var(--line);
    }

    .video-stage {
      min-height: 0;
      display: grid;
      place-items: center;
      padding: 16px;
      background:
        linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(0deg, rgba(255,255,255,0.025) 1px, transparent 1px),
        #0b0c0d;
      background-size: 28px 28px;
    }

    .video-wrap {
      position: relative;
      height: min(calc(100vh - 170px), 84vw);
      max-height: 100%;
      aspect-ratio: 688 / 912;
      background: #000;
      box-shadow: 0 18px 60px rgba(0,0,0,.45);
      overflow: hidden;
    }

    video {
      display: block;
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #000;
    }

    .marker {
      position: absolute;
      width: 54px;
      height: 54px;
      margin-left: -27px;
      margin-top: -27px;
      border: 3px solid var(--yellow);
      border-radius: 50%;
      box-shadow: 0 0 0 2px rgba(0,0,0,.8), 0 0 24px rgba(255,214,92,.35);
      pointer-events: none;
      opacity: 0;
      transform: scale(.9);
      transition: opacity .12s ease, transform .12s ease;
    }

    .marker.show {
      opacity: 1;
      transform: scale(1);
    }

    .control-strip {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: center;
      padding: 12px 16px;
      border-top: 1px solid var(--line);
      background: #141619;
    }

    .selected-line {
      min-width: 0;
      display: flex;
      gap: 10px;
      align-items: center;
      color: var(--muted);
      white-space: nowrap;
      overflow: hidden;
    }

    .selected-line strong {
      color: var(--text);
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .actions {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    button {
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--text);
      border-radius: 7px;
      padding: 8px 11px;
      cursor: pointer;
      min-height: 36px;
    }

    button:hover { border-color: #4b5360; background: #262a31; }
    button:active { transform: translateY(1px); }
    button.primary { background: #d9f99d; border-color: #d9f99d; color: var(--ink); font-weight: 760; }
    button.danger { background: #3a1d1e; border-color: #673031; color: #ffd0cd; }
    button.blue { background: #152a3d; border-color: #31577b; color: #d6ebff; }
    button.ghost { color: var(--muted); }

    aside {
      min-width: 0;
      min-height: 0;
      display: grid;
      grid-template-rows: auto auto 1fr auto;
      background: var(--panel);
    }

    .tool-row {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 8px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
    }

    .filter-row {
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 8px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
    }

    input, textarea {
      width: 100%;
      color: var(--text);
      background: #111317;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 9px 10px;
      outline: none;
    }

    input:focus, textarea:focus, select:focus {
      border-color: #596473;
      box-shadow: 0 0 0 2px rgba(118,185,255,.14);
    }

    .queue {
      min-height: 0;
      overflow: auto;
    }

    .row {
      width: 100%;
      display: grid;
      grid-template-columns: 28px 72px 1fr 64px;
      gap: 8px;
      align-items: center;
      padding: 9px 12px;
      border: 0;
      border-bottom: 1px solid rgba(48,52,59,.8);
      background: transparent;
      color: var(--text);
      border-radius: 0;
      text-align: left;
      min-height: 48px;
    }

    .row:hover { background: rgba(255,255,255,.035); border-color: rgba(48,52,59,.8); }
    .row.active { background: #26303a; box-shadow: inset 3px 0 0 var(--yellow); }

    .row-index {
      color: var(--dim);
      font-variant-numeric: tabular-nums;
      font-size: 12px;
    }

    .row-time {
      color: var(--text);
      font-weight: 720;
      font-variant-numeric: tabular-nums;
    }

    .row-main {
      min-width: 0;
      display: grid;
      gap: 2px;
    }

    .row-title {
      display: flex;
      gap: 6px;
      align-items: center;
      min-width: 0;
    }

    .row-title span:last-child {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .row-sub {
      color: var(--muted);
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .score {
      color: var(--muted);
      font-variant-numeric: tabular-nums;
      text-align: right;
      font-size: 12px;
    }

    .details {
      border-top: 1px solid var(--line);
      padding: 12px;
      display: grid;
      gap: 10px;
      max-height: min(390px, 43vh);
      overflow: auto;
    }

    .batch-preview {
      display: none;
      border: 1px solid var(--line);
      background: #0b0c0d;
      border-radius: 7px;
      overflow: hidden;
      height: 178px;
      place-items: center;
    }

    .batch-preview.show {
      display: grid;
    }

    .batch-preview img {
      display: block;
      width: auto;
      max-width: 100%;
      height: 176px;
      object-fit: contain;
      background: #050607;
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 8px;
    }

    .metric {
      background: #121417;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 8px;
      min-width: 0;
    }

    .metric-label {
      color: var(--dim);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }

    .metric-value {
      color: var(--text);
      font-size: 13px;
      font-weight: 720;
      font-variant-numeric: tabular-nums;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .note-row {
      display: grid;
      gap: 7px;
    }

    .field-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }

    .field {
      display: grid;
      gap: 5px;
      min-width: 0;
    }

    .field label {
      color: var(--dim);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }

    .field select {
      width: 100%;
      color: var(--text);
      background: #111317;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 8px 9px;
      outline: none;
      min-width: 0;
    }

    textarea {
      resize: vertical;
      min-height: 58px;
      max-height: 140px;
    }

    .footer-actions {
      display: flex;
      gap: 8px;
      justify-content: space-between;
      align-items: center;
    }

    .path-note {
      color: var(--dim);
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .empty {
      padding: 28px 18px;
      color: var(--muted);
    }

    @media (max-width: 980px) {
      header { align-items: stretch; flex-wrap: wrap; }
      .top-select { min-width: 100%; }
      .counts { margin-left: 0; justify-content: flex-start; }
      main { grid-template-columns: 1fr; grid-template-rows: minmax(420px, 58vh) minmax(420px, 42vh); }
      .viewer { border-right: 0; border-bottom: 1px solid var(--line); }
      aside { min-height: 420px; }
      .video-wrap { height: 100%; max-width: 100%; }
      .control-strip { grid-template-columns: 1fr; }
      .actions { justify-content: flex-start; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div class="brand">
        <h1>Hacky Track Review</h1>
        <span id="saveState" class="save-state">loading</span>
      </div>
      <select id="videoSelect" class="top-select"></select>
      <div id="counts" class="counts"></div>
    </header>

    <main>
      <section class="viewer">
        <div class="video-stage">
          <div class="video-wrap" id="videoWrap">
            <video id="video" controls playsinline preload="metadata"></video>
            <div id="marker" class="marker"></div>
          </div>
        </div>
        <div class="control-strip">
          <div class="selected-line">
            <span id="selectedDot" class="dot pending"></span>
            <strong id="selectedTitle">No event selected</strong>
            <span id="selectedTime"></span>
          </div>
          <div class="actions">
            <button id="prevBtn" class="ghost" title="Previous candidate">Prev</button>
            <button id="approveBtn" class="primary" title="Approve selected touch">Approve</button>
            <button id="rejectBtn" class="danger" title="Reject selected touch">Reject</button>
            <button id="pendingBtn" class="ghost" title="Return selected touch to pending">Pending</button>
            <button id="addMissingBtn" class="blue" title="Add touch at current video time">Add Touch</button>
            <button id="addDropBtn" class="blue" title="Add floor drop at current video time">Add Drop</button>
            <button id="addStallBtn" class="blue" title="Add stall window at current video time">Add Stall</button>
            <button id="addAtwBtn" class="blue" title="Add around-the-world event at current video time">Add ATW</button>
            <button id="nextBtn" class="ghost" title="Next candidate">Next</button>
          </div>
        </div>
      </section>

      <aside>
        <div class="tool-row">
          <button id="saveBtn">Save</button>
          <button id="exportBtn">Export Approved</button>
          <button id="playClipBtn">Play Clip</button>
        </div>
        <div class="filter-row">
          <input id="filterInput" type="search" placeholder="Filter status, kind, time, tag, note">
          <button id="batchOnlyBtn" class="ghost">Batch</button>
          <button id="pendingOnlyBtn" class="ghost">Pending</button>
        </div>
        <div id="queue" class="queue"></div>
        <div class="details">
          <div class="metrics">
            <div class="metric"><div class="metric-label">Status</div><div id="metricStatus" class="metric-value">-</div></div>
            <div class="metric"><div class="metric-label">Rally</div><div id="metricRally" class="metric-value">-</div></div>
            <div class="metric"><div class="metric-label">Conf</div><div id="metricConf" class="metric-value">-</div></div>
            <div class="metric"><div class="metric-label">Audio</div><div id="metricAudio" class="metric-value">-</div></div>
          </div>
          <div class="actions">
            <button id="nudgeBackBtn" class="ghost" title="Move selected time earlier">-0.05s</button>
            <button id="setCurrentBtn" class="ghost" title="Set selected time to current video time">Set Time</button>
            <button id="nudgeForwardBtn" class="ghost" title="Move selected time later">+0.05s</button>
            <button id="deleteManualBtn" class="danger" title="Delete selected manual touch">Delete Manual</button>
          </div>
          <div id="batchPreview" class="batch-preview">
            <img id="batchCrop" alt="Selected review batch crop">
          </div>
          <div class="field-grid">
            <div class="field">
              <label for="kindSelect">Kind</label>
              <select id="kindSelect">
                <option value="touch">touch</option>
                <option value="drop_floor">drop_floor</option>
                <option value="stall">stall</option>
                <option value="around_the_world">around_the_world</option>
              </select>
            </div>
            <div class="field">
              <label for="sideSelect">Side (0/1/2)</label>
              <select id="sideSelect">
                <option value="unknown">unknown</option>
                <option value="left">left</option>
                <option value="right">right</option>
                <option value="center">center</option>
              </select>
            </div>
            <div class="field">
              <label for="contactSelect">Contact</label>
              <select id="contactSelect">
                <option value="unknown">unknown</option>
                <option value="foot">foot</option>
                <option value="foot_candidate">foot_candidate</option>
                <option value="knee_candidate">knee_candidate</option>
                <option value="stall">stall</option>
                <option value="ground">ground</option>
                <option value="unknown_contact">unknown_contact</option>
              </select>
            </div>
            <div class="field">
              <label for="trickSelect">Move</label>
              <select id="trickSelect">
                <option value=""></option>
                <option value="right_kick">right_kick</option>
                <option value="left_kick">left_kick</option>
                <option value="right_stall">right_stall</option>
                <option value="left_stall">left_stall</option>
                <option value="outer_right">outer_right</option>
                <option value="outer_left">outer_left</option>
                <option value="around_the_world_outer_right">around_the_world_outer_right</option>
                <option value="around_the_world">around_the_world</option>
              </select>
            </div>
            <div class="field">
              <label for="startInput">Start</label>
              <input id="startInput" type="number" step="0.001" min="0" placeholder="start">
            </div>
            <div class="field">
              <label for="endInput">End</label>
              <input id="endInput" type="number" step="0.001" min="0" placeholder="end">
            </div>
          </div>
          <div class="note-row">
            <textarea id="noteInput" placeholder="Note"></textarea>
          </div>
          <div class="footer-actions">
            <span id="pathNote" class="path-note"></span>
          </div>
        </div>
      </aside>
    </main>
  </div>

  <script>
    const state = {
      videos: [],
      stem: null,
      review: null,
      selectedId: null,
      dirty: false,
      saveTimer: null,
      filter: "",
      batchOnly: false,
      pendingOnly: false,
      lastExport: ""
    };

    const $ = (id) => document.getElementById(id);
    const video = $("video");
    const marker = $("marker");

    function fmtTime(value) {
      const t = Number(value || 0);
      const m = Math.floor(t / 60);
      const s = (t - m * 60).toFixed(3).padStart(6, "0");
      return `${m}:${s}`;
    }

    function fmtShort(value) {
      if (value === null || value === undefined || value === "") return "-";
      const num = Number(value);
      if (!Number.isFinite(num)) return String(value);
      return num.toFixed(num >= 10 ? 1 : 2);
    }

    function itemLabel(item) {
      if (!item) return "No event selected";
      const source = item.source === "manual" ? "Manual" : "Candidate";
      const rally = item.rally_id ? `R${item.rally_id}` : "No rally";
      const touch = item.touch_number ? `T${item.touch_number}` : "touch";
      const kind = item.kind || "touch";
      return kind === "touch" ? `${source} ${rally} ${touch}` : `${source} ${rally} ${kind}`;
    }

    function sortedItems() {
      if (!state.review) return [];
      return [...state.review.items].sort((a, b) => {
        const dt = Number(a.time_sec || 0) - Number(b.time_sec || 0);
        return dt || String(a.id).localeCompare(String(b.id));
      });
    }

    function visibleItems() {
      const filter = state.filter.trim().toLowerCase();
      return sortedItems().filter((item) => {
        if (state.pendingOnly && item.status !== "pending") return false;
        if (state.batchOnly && !item.in_review_batch) return false;
        if (!filter) return true;
        const haystack = [
          item.in_review_batch ? "review_batch batch" : "",
          item.batch_item_id,
          item.batch_index,
          item.assist_bucket,
          item.assist_action,
          item.status,
          item.kind,
          item.source,
          item.note,
          item.contact_side,
          item.contact_type,
          item.side_source,
          item.side_uncertainty_reason,
          item.trick_label,
          ...(item.review_tags || []),
          item.rally_id,
          item.touch_number,
          fmtTime(item.time_sec),
          Number(item.time_sec || 0).toFixed(3)
        ].join(" ").toLowerCase();
        return haystack.includes(filter);
      });
    }

    function currentItem() {
      if (!state.review || !state.selectedId) return null;
      return state.review.items.find((item) => item.id === state.selectedId) || null;
    }

    function setSaveState(text) {
      $("saveState").textContent = text;
    }

    function markDirty() {
      state.dirty = true;
      setSaveState("unsaved");
      window.clearTimeout(state.saveTimer);
      state.saveTimer = window.setTimeout(saveNow, 600);
    }

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: {"Content-Type": "application/json"},
        ...options
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `${response.status} ${response.statusText}`);
      }
      return response.json();
    }

    async function loadVideos() {
      const data = await api("/api/videos");
      state.videos = data.videos;
      const select = $("videoSelect");
      select.innerHTML = "";
      for (const entry of state.videos) {
        const option = document.createElement("option");
        option.value = entry.stem;
        const batchText = entry.batch_total ? ` · batch ${entry.batch_decided}/${entry.batch_total}` : "";
        option.textContent = `${entry.source_video} (${entry.total_items}${batchText})`;
        select.appendChild(option);
      }
      if (!state.videos.length) {
        $("queue").innerHTML = '<div class="empty">No trained event files found under outputs.</div>';
        setSaveState("empty");
        return;
      }
      await loadReview(state.videos[0].stem);
    }

    async function loadReview(stem) {
      state.stem = stem;
      $("videoSelect").value = stem;
      state.review = await api(`/api/review/${encodeURIComponent(stem)}`);
      state.selectedId = null;
      state.dirty = false;
      state.lastExport = "";
      video.src = state.review.video_url;
      const firstPendingBatch = sortedItems().find((item) => item.in_review_batch && item.status === "pending");
      const firstPending = firstPendingBatch || sortedItems().find((item) => item.status === "pending");
      const first = firstPending || sortedItems()[0] || null;
      if (first) state.selectedId = first.id;
      setSaveState("saved");
      render();
      if (first) seekToItem(first, false);
    }

    async function saveNow() {
      if (!state.review || !state.stem) return;
      window.clearTimeout(state.saveTimer);
      setSaveState("saving");
      state.review.items = sortedItems();
      const result = await api(`/api/review/${encodeURIComponent(state.stem)}`, {
        method: "POST",
        body: JSON.stringify(state.review)
      });
      state.review.updated_at = result.updated_at;
      state.dirty = false;
      setSaveState("saved");
      renderCounts();
    }

    async function exportApproved() {
      await saveNow();
      const result = await api(`/api/export/${encodeURIComponent(state.stem)}`, {method: "POST"});
      state.lastExport = result.export_path;
      $("pathNote").textContent = result.export_path;
      setSaveState("exported");
    }

    function seekToItem(item, play = false) {
      if (!item) return;
      const preRoll = 0.45;
      video.currentTime = Math.max(0, Number(item.time_sec || 0) - preRoll);
      if (play) {
        const stopAt = Number(item.time_sec || 0) + 0.65;
        video.play().catch(() => {});
        const stop = () => {
          if (video.currentTime >= stopAt) {
            video.pause();
            video.removeEventListener("timeupdate", stop);
          }
        };
        video.addEventListener("timeupdate", stop);
      }
    }

    function selectItem(item, seek = true) {
      if (!item) return;
      state.selectedId = item.id;
      if (seek) seekToItem(item, false);
      render();
    }

    function selectRelative(delta) {
      const items = visibleItems();
      if (!items.length) return;
      const index = Math.max(0, items.findIndex((item) => item.id === state.selectedId));
      const next = Math.min(items.length - 1, Math.max(0, index + delta));
      selectItem(items[next], true);
    }

    function setStatus(status) {
      const item = currentItem();
      if (!item) return;
      item.status = item.source === "manual" && item.active_learning_candidate && status === "approved" ? "missing" : status;
      markDirty();
      render();
      if (item.status === "approved" || item.status === "rejected" || item.status === "missing") {
        const items = visibleItems();
        const index = items.findIndex((entry) => entry.id === item.id);
        const next = items.slice(index + 1).find((entry) => entry.status === "pending") || items[index + 1];
        if (next) selectItem(next, true);
      }
    }

    function setSide(side) {
      const item = currentItem();
      if (!item) return;
      item.contact_side = side;
      if ((item.trick_label || "") === "" && (side === "left" || side === "right")) {
        if (item.kind === "stall" || item.contact_type === "stall") item.trick_label = `${side}_stall`;
        else if (item.kind === "touch" && (item.contact_type || "").includes("foot")) item.trick_label = `${side}_kick`;
      }
      markDirty();
      render();
    }

    function addManualEvent(kind = "touch") {
      if (!state.review) return;
      const timeSec = Math.max(0, video.currentTime || 0);
      const id = `manual-${kind}-${Date.now()}-${Math.round(timeSec * 1000)}`;
      const duration = kind === "stall" ? 0.35 : null;
      const item = {
        id,
        source: "manual",
        kind,
        status: "missing",
        rally_id: nearestRallyId(timeSec),
        touch_number: null,
        time_sec: Number(timeSec.toFixed(3)),
        start_sec: kind === "stall" ? Number(timeSec.toFixed(3)) : null,
        end_sec: kind === "stall" ? Number((timeSec + duration).toFixed(3)) : null,
        duration_sec: duration,
        confidence: null,
        audio_z: null,
        visual_score: null,
        motion_score: null,
        x: null,
        y: null,
        ball_confidence: null,
        ball_accuracy: null,
        contact_side: "unknown",
        contact_type: kind === "drop_floor" ? "ground" : kind === "stall" ? "stall" : "unknown",
        trick_label: kind === "around_the_world" ? "around_the_world" : "",
        note: ""
      };
      state.review.items.push(item);
      state.selectedId = id;
      markDirty();
      render();
    }

    function addMissingTouch() {
      addManualEvent("touch");
    }

    function nearestRallyId(timeSec) {
      const candidates = sortedItems().filter((item) => item.rally_id);
      let best = null;
      for (const item of candidates) {
        const dist = Math.abs(Number(item.time_sec || 0) - timeSec);
        if (!best || dist < best.dist) best = {dist, rally_id: item.rally_id};
      }
      return best && best.dist <= 2.5 ? best.rally_id : null;
    }

    function nudgeSelected(delta) {
      const item = currentItem();
      if (!item) return;
      item.time_sec = Number(Math.max(0, Number(item.time_sec || 0) + delta).toFixed(3));
      markDirty();
      render();
      seekToItem(item, false);
    }

    function setSelectedToCurrent() {
      const item = currentItem();
      if (!item) return;
      item.time_sec = Number(Math.max(0, video.currentTime || 0).toFixed(3));
      markDirty();
      render();
    }

    function deleteManual() {
      const item = currentItem();
      if (!item || item.source !== "manual") return;
      state.review.items = state.review.items.filter((entry) => entry.id !== item.id);
      const next = sortedItems().find((entry) => Number(entry.time_sec || 0) >= Number(item.time_sec || 0)) || sortedItems()[0] || null;
      state.selectedId = next ? next.id : null;
      markDirty();
      render();
    }

    function renderCounts() {
      const counts = {pending: 0, approved: 0, rejected: 0, missing: 0};
      let batchTotal = 0;
      let batchDecided = 0;
      if (state.review) {
        for (const item of state.review.items) {
          if (counts[item.status] !== undefined) counts[item.status] += 1;
          if (item.in_review_batch) {
            batchTotal += 1;
            if (item.status === "approved" || item.status === "rejected" || item.status === "missing") batchDecided += 1;
          }
        }
      }
      const pills = Object.entries(counts).map(([key, value]) => (
        `<span class="pill"><span class="dot ${key}"></span>${key} ${value}</span>`
      ));
      if (batchTotal) pills.unshift(`<span class="pill"><span class="dot approved"></span>batch ${batchDecided}/${batchTotal}</span>`);
      $("counts").innerHTML = pills.join("");
    }

    function renderQueue() {
      const queue = $("queue");
      const items = visibleItems();
      if (!items.length) {
        queue.innerHTML = '<div class="empty">No touches match this view.</div>';
        return;
      }
      queue.innerHTML = "";
      items.forEach((item, index) => {
        const row = document.createElement("button");
        row.className = `row ${item.id === state.selectedId ? "active" : ""}`;
        row.type = "button";
        const tags = (item.review_tags || []).slice(0, 4).join(" ");
        const batchLabel = item.in_review_batch ? `batch #${item.batch_index} · ` : "";
        const assistLabel = item.assist_bucket ? `${item.assist_bucket} · ` : "";
        row.innerHTML = `
          <span class="row-index">${index + 1}</span>
          <span class="row-time">${fmtTime(item.time_sec)}</span>
          <span class="row-main">
            <span class="row-title"><span class="dot ${item.status}"></span><span>${itemLabel(item)}</span></span>
            <span class="row-sub">${escapeHtml(batchLabel)}${escapeHtml(assistLabel)}${escapeHtml(item.kind || "touch")} · ${escapeHtml(item.contact_side || "unknown")} · ${escapeHtml(item.contact_type || "unknown")}${tags ? " · " + escapeHtml(tags) : ""}${item.note ? " - " + escapeHtml(item.note) : ""}</span>
          </span>
          <span class="score">${fmtShort(item.confidence)}</span>
        `;
        row.addEventListener("click", () => selectItem(item, true));
        row.addEventListener("dblclick", () => seekToItem(item, true));
        queue.appendChild(row);
      });
    }

    function escapeHtml(value) {
      return String(value || "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }[char]));
    }

    function artifactUrl(path) {
      return `/artifact?path=${encodeURIComponent(path)}`;
    }

    function renderSelected() {
      const item = currentItem();
      $("selectedTitle").textContent = itemLabel(item);
      $("selectedTime").textContent = item ? fmtTime(item.time_sec) : "";
      $("selectedDot").className = `dot ${item ? item.status : "pending"}`;
      $("metricStatus").textContent = item ? item.status : "-";
      $("metricRally").textContent = item && item.rally_id ? `R${item.rally_id}` : "-";
      $("metricConf").textContent = item ? fmtShort(item.confidence) : "-";
      $("metricAudio").textContent = item ? fmtShort(item.audio_z) : "-";
      $("noteInput").value = item ? (item.note || "") : "";
      $("kindSelect").value = item ? (item.kind || "touch") : "touch";
      $("sideSelect").value = item ? (item.contact_side || "unknown") : "unknown";
      $("contactSelect").value = item ? (item.contact_type || "unknown") : "unknown";
      $("trickSelect").value = item ? (item.trick_label || "") : "";
      $("startInput").value = item && item.start_sec !== null && item.start_sec !== undefined ? Number(item.start_sec).toFixed(3) : "";
      $("endInput").value = item && item.end_sec !== null && item.end_sec !== undefined ? Number(item.end_sec).toFixed(3) : "";
      $("deleteManualBtn").disabled = !item || item.source !== "manual";
      if (item && item.batch_crop_path) {
        $("batchCrop").src = artifactUrl(item.batch_crop_path);
        $("batchPreview").classList.add("show");
      } else {
        $("batchCrop").removeAttribute("src");
        $("batchPreview").classList.remove("show");
      }

      const size = state.review && state.review.track_size ? state.review.track_size : {width: 688, height: 912};
      if (item && item.x !== null && item.x !== undefined && item.y !== null && item.y !== undefined) {
        marker.style.left = `${(Number(item.x) / Number(size.width || 688)) * 100}%`;
        marker.style.top = `${(Number(item.y) / Number(size.height || 912)) * 100}%`;
        marker.classList.add("show");
      } else {
        marker.classList.remove("show");
      }

      const assist = item && item.assist_bucket ? `${item.assist_bucket}: ${item.assist_action || ""}` : "";
      const path = assist || state.lastExport || (state.review ? state.review.review_path || "" : "");
      $("pathNote").textContent = path;
    }

    function render() {
      renderCounts();
      renderQueue();
      renderSelected();
      $("batchOnlyBtn").classList.toggle("primary", state.batchOnly);
      $("pendingOnlyBtn").classList.toggle("primary", state.pendingOnly);
    }

    $("videoSelect").addEventListener("change", async (event) => {
      if (state.dirty) await saveNow();
      await loadReview(event.target.value);
    });
    $("saveBtn").addEventListener("click", saveNow);
    $("exportBtn").addEventListener("click", exportApproved);
    $("playClipBtn").addEventListener("click", () => seekToItem(currentItem(), true));
    $("prevBtn").addEventListener("click", () => selectRelative(-1));
    $("nextBtn").addEventListener("click", () => selectRelative(1));
    $("approveBtn").addEventListener("click", () => setStatus("approved"));
    $("rejectBtn").addEventListener("click", () => setStatus("rejected"));
    $("pendingBtn").addEventListener("click", () => setStatus("pending"));
    $("addMissingBtn").addEventListener("click", addMissingTouch);
    $("addDropBtn").addEventListener("click", () => addManualEvent("drop_floor"));
    $("addStallBtn").addEventListener("click", () => addManualEvent("stall"));
    $("addAtwBtn").addEventListener("click", () => addManualEvent("around_the_world"));
    $("nudgeBackBtn").addEventListener("click", () => nudgeSelected(-0.05));
    $("nudgeForwardBtn").addEventListener("click", () => nudgeSelected(0.05));
    $("setCurrentBtn").addEventListener("click", setSelectedToCurrent);
    $("deleteManualBtn").addEventListener("click", deleteManual);
    $("filterInput").addEventListener("input", (event) => {
      state.filter = event.target.value;
      renderQueue();
    });
    $("pendingOnlyBtn").addEventListener("click", () => {
      state.pendingOnly = !state.pendingOnly;
      render();
    });
    $("batchOnlyBtn").addEventListener("click", () => {
      state.batchOnly = !state.batchOnly;
      render();
    });
    $("noteInput").addEventListener("input", (event) => {
      const item = currentItem();
      if (!item) return;
      item.note = event.target.value;
      markDirty();
      renderQueue();
    });

    $("kindSelect").addEventListener("change", (event) => {
      const item = currentItem();
      if (!item) return;
      item.kind = event.target.value;
      if (item.kind === "drop_floor") item.contact_type = "ground";
      if (item.kind === "stall") item.contact_type = "stall";
      markDirty();
      render();
    });
    $("sideSelect").addEventListener("change", (event) => {
      setSide(event.target.value);
    });
    $("contactSelect").addEventListener("change", (event) => {
      const item = currentItem();
      if (!item) return;
      item.contact_type = event.target.value;
      markDirty();
      renderQueue();
    });
    $("trickSelect").addEventListener("change", (event) => {
      const item = currentItem();
      if (!item) return;
      item.trick_label = event.target.value;
      markDirty();
      renderQueue();
    });
    $("startInput").addEventListener("change", (event) => {
      const item = currentItem();
      if (!item) return;
      const value = event.target.value === "" ? null : Number(event.target.value);
      item.start_sec = value;
      if (value !== null) item.time_sec = Number(value.toFixed(3));
      if (item.start_sec !== null && item.end_sec !== null && item.end_sec !== undefined) {
        item.duration_sec = Number(Math.max(0, Number(item.end_sec) - Number(item.start_sec)).toFixed(3));
      }
      markDirty();
      render();
    });
    $("endInput").addEventListener("change", (event) => {
      const item = currentItem();
      if (!item) return;
      const value = event.target.value === "" ? null : Number(event.target.value);
      item.end_sec = value;
      if (item.start_sec !== null && item.start_sec !== undefined && value !== null) {
        item.duration_sec = Number(Math.max(0, value - Number(item.start_sec)).toFixed(3));
      }
      markDirty();
      render();
    });

    window.addEventListener("keydown", (event) => {
      const tag = document.activeElement && document.activeElement.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (event.key === "a" || event.key === "A") { setStatus("approved"); event.preventDefault(); }
      if (event.key === "r" || event.key === "R") { setStatus("rejected"); event.preventDefault(); }
      if (event.key === "p" || event.key === "P") { setStatus("pending"); event.preventDefault(); }
      if (event.key === "q" || event.key === "Q") { state.batchOnly = !state.batchOnly; render(); event.preventDefault(); }
      if (event.key === "m" || event.key === "M") { addMissingTouch(); event.preventDefault(); }
      if (event.key === "d" || event.key === "D") { addManualEvent("drop_floor"); event.preventDefault(); }
      if (event.key === "s" || event.key === "S") { addManualEvent("stall"); event.preventDefault(); }
      if (event.key === "w" || event.key === "W") { addManualEvent("around_the_world"); event.preventDefault(); }
      if (event.key === "0") { setSide("unknown"); event.preventDefault(); }
      if (event.key === "1") { setSide("left"); event.preventDefault(); }
      if (event.key === "2") { setSide("right"); event.preventDefault(); }
      if (event.key === "ArrowDown" || event.key === "n" || event.key === "N") { selectRelative(1); event.preventDefault(); }
      if (event.key === "ArrowUp" || event.key === "b" || event.key === "B") { selectRelative(-1); event.preventDefault(); }
      if (event.key === "[") { nudgeSelected(-0.05); event.preventDefault(); }
      if (event.key === "]") { nudgeSelected(0.05); event.preventDefault(); }
      if (event.key === " ") {
        if (video.paused) video.play().catch(() => {});
        else video.pause();
        event.preventDefault();
      }
    });

    window.addEventListener("beforeunload", (event) => {
      if (!state.dirty) return;
      event.preventDefault();
      event.returnValue = "";
    });

    loadVideos().catch((error) => {
      console.error(error);
      $("queue").innerHTML = `<div class="empty">${escapeHtml(error.message)}</div>`;
      setSaveState("error");
    });
  </script>
</body>
</html>
"""


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "HackyTrackReview/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/":
            self.send_text(HTML, "text/html; charset=utf-8")
            return
        if path == "/api/videos":
            videos = [review_summary(info) for info in self.videos().values()]
            self.send_json({"videos": videos})
            return
        if path.startswith("/api/review/"):
            stem = slug_id(path.rsplit("/", 1)[-1])
            info = self.require_video(stem)
            if not info:
                return
            doc = merged_review_doc(info)
            doc["video_url"] = f"/media/{info.stem}"
            doc["review_path"] = str(info.review_path.relative_to(ROOT)) if info.review_path.is_relative_to(ROOT) else str(info.review_path)
            self.send_json(doc)
            return
        if path.startswith("/media/"):
            stem = slug_id(path.rsplit("/", 1)[-1])
            info = self.require_video(stem)
            if not info:
                return
            self.send_file(info.video_path)
            return
        if path == "/artifact":
            query = parse_qs(parsed.query)
            rel_path = query.get("path", [""])[0]
            artifact = (ROOT / rel_path).resolve()
            try:
                artifact.relative_to(ROOT.resolve())
            except ValueError:
                self.send_error(HTTPStatus.FORBIDDEN, "Artifact must be under the project root")
                return
            self.send_file(artifact)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path.startswith("/media/"):
            stem = slug_id(path.rsplit("/", 1)[-1])
            info = self.require_video(stem)
            if not info:
                return
            self.send_file(info.video_path, head_only=True)
            return
        if path == "/":
            body = HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path.startswith("/api/review/"):
            stem = slug_id(path.rsplit("/", 1)[-1])
            info = self.require_video(stem)
            if not info:
                return
            payload = self.read_body_json()
            if payload is None:
                return
            if not isinstance(payload.get("items"), list):
                self.send_error(HTTPStatus.BAD_REQUEST, "Review document must include an items list")
                return
            payload.pop("video_url", None)
            payload.pop("review_path", None)
            payload["source_video"] = info.source_video
            payload["video_path"] = str(info.video_path)
            payload["candidate_source"] = str(info.event_path.relative_to(ROOT)) if info.event_path.is_relative_to(ROOT) else str(info.event_path)
            payload["qa_sheet_path"] = None if info.qa_sheet_path is None else str(info.qa_sheet_path)
            payload["track_size"] = payload.get("track_size") or info.track_size
            payload["updated_at"] = utc_now()
            payload.setdefault("created_at", payload["updated_at"])
            payload["items"] = sorted(
                payload["items"],
                key=lambda item: (float(item.get("time_sec") or 0.0), str(item.get("id") or "")),
            )
            write_json(info.review_path, payload)
            self.send_json({"ok": True, "updated_at": payload["updated_at"], "review_path": str(info.review_path)})
            return

        if path.startswith("/api/export/"):
            stem = slug_id(path.rsplit("/", 1)[-1])
            info = self.require_video(stem)
            if not info:
                return
            review_doc = merged_review_doc(info)
            if info.review_path.exists():
                review_doc = read_json(info.review_path)
            export_doc = accepted_events_doc(review_doc)
            export_path = REVIEWS_DIR / f"{info.stem}.approved_events.json"
            write_json(export_path, export_doc)
            display_path = str(export_path.relative_to(ROOT)) if export_path.is_relative_to(ROOT) else str(export_path)
            self.send_json({"ok": True, "export_path": display_path, "rallies": len(export_doc["rallies"])})
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def videos(self) -> dict[str, ReviewableVideo]:
        return self.server.review_videos  # type: ignore[attr-defined]

    def require_video(self, stem: str) -> ReviewableVideo | None:
        info = self.videos().get(stem)
        if not info:
            self.send_error(HTTPStatus.NOT_FOUND, f"Unknown video: {stem}")
            return None
        return info

    def read_body_json(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON")
            return None
        if not isinstance(payload, dict):
            self.send_error(HTTPStatus.BAD_REQUEST, "Expected a JSON object")
            return None
        return payload

    def send_json(self, data: dict[str, Any]) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str, content_type: str) -> None:
        body = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, *, head_only: bool = False) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, f"File not found: {path}")
            return

        size = path.stat().st_size
        range_header = self.headers.get("Range")
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix.lower() == ".mov":
            content_type = "video/quicktime"

        start = 0
        end = size - 1
        status = HTTPStatus.OK
        if range_header:
            match = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if match:
                if match.group(1):
                    start = int(match.group(1))
                if match.group(2):
                    end = int(match.group(2))
                end = min(end, size - 1)
                status = HTTPStatus.PARTIAL_CONTENT

        if start < 0 or start >= size or end < start:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head_only:
            return

        with path.open("rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def free_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Hacky Track touch review app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--qa-manifest", type=Path, default=DEFAULT_QA_MANIFEST)
    parser.add_argument("--review-batch", type=Path, default=DEFAULT_REVIEW_BATCH)
    parser.add_argument("--assisted-review", type=Path, default=DEFAULT_ASSISTED_REVIEW)
    parser.add_argument("--reviews-dir", type=Path, default=REVIEWS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global DEFAULT_QA_MANIFEST, DEFAULT_REVIEW_BATCH, DEFAULT_ASSISTED_REVIEW, REVIEWS_DIR
    DEFAULT_QA_MANIFEST = args.qa_manifest
    DEFAULT_REVIEW_BATCH = args.review_batch
    DEFAULT_ASSISTED_REVIEW = args.assisted_review
    REVIEWS_DIR = args.reviews_dir
    videos = discover_videos()
    if not videos:
        print(f"No reviewable videos found under {OUTPUTS_DIR}")
    port = free_port(args.port) if args.host in {"127.0.0.1", "localhost"} else args.port
    server = ThreadingHTTPServer((args.host, port), ReviewHandler)
    server.review_videos = videos  # type: ignore[attr-defined]
    print(f"review app: http://{args.host}:{port}/")
    print(f"videos: {len(videos)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
