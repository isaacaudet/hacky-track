#!/usr/bin/env python3
"""Build a visual review batch from enriched Hacky Track QA events.

The detector is only useful once its failures are easy to review. This script
selects a deterministic, high-risk-but-balanced subset across the 27-video QA
manifest, writes a JSON/JSONL review queue, and renders crop sheets that make
ball-center/contact mistakes visible.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageDraw, ImageFont

from review_app import REVIEWABLE_KINDS, candidate_item, event_review_tags, slug_id


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "outputs" / "full_training_27_qa" / "qa_manifest.json"
DEFAULT_OUT_DIR = ROOT / "outputs" / "review_batches"
OUT_SIZE = (688, 912)
SHEET_COLUMNS = 4
TILE_W = 330
TILE_H = 388
CROP_W = 300
CROP_H = 252


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def event_time(event: dict[str, Any]) -> float:
    return float(event.get("time_sec", event.get("start_sec", 0.0)) or 0.0)


def event_frame_time(event: dict[str, Any]) -> float:
    return float(event.get("qa_frame_time_sec", event_time(event)) or 0.0)


def score_record(kind: str, tags: list[str], event: dict[str, Any]) -> float:
    score = {
        "touch": 1.0,
        "drop_floor": 1.35,
        "stall": 1.25,
        "around_the_world": 2.2,
    }.get(kind, 0.8)
    weights = {
        "ball_low": 1.2,
        "ball_medium": 0.55,
        "ball_fallback": 1.1,
        "huge_ball_correction": 1.2,
        "large_ball_correction": 0.75,
        "ambiguous_contact": 0.8,
        "side_needs_review": 0.85,
        "side_unknown": 0.55,
        "side_center": 0.35,
        "floor_risk_touch": 1.25,
        "near_floor": 0.45,
        "hidden_drop": 1.1,
        "reclassified": 1.0,
        "recovered_touch": 0.85,
        "audio_track_context": 0.55,
        "short_stall": 0.8,
        "trick_candidate": 1.6,
        "contact_knee_candidate": 0.95,
        "contact_unknown_contact": 0.75,
        "contact_foot_candidate": 0.45,
        "contact_ground_candidate": 0.75,
        "low_confidence": 0.65,
        "high_confidence": 0.15,
        "active_learning": 1.35,
        "likely_missed_touch": 1.15,
        "likely_missed_drop_floor": 1.25,
        "likely_missed_stall": 1.15,
        "likely_missed_around_the_world": 1.35,
        "likely_missing_floor_reset": 1.40,
        "gap_without_floor_reset": 1.30,
        "strict_best_rally_blocker": 1.10,
        "large_rally_gap": 0.50,
        "suppressed_candidate": 0.65,
        "suppressed_near_floor_weak_contact": 0.55,
        "suppressed_floor_reset_too_far_from_limb_context": 0.50,
        "suppressed_low_confidence_touch_candidate": 0.35,
        "suppressed_weak_knee_candidate": 0.90,
    }
    score += sum(weights.get(tag, 0.0) for tag in tags)
    correction = maybe_float(event.get("qa_ball_correction_px")) or 0.0
    score += min(0.5, correction / 260.0)
    audio_z = maybe_float(event.get("audio_z")) or 0.0
    if kind == "touch" and audio_z >= 24:
        score += 0.18
    return round(score, 4)


def suppressed_review_tags(event: dict[str, Any]) -> list[str]:
    tags = event_review_tags(event)
    kind = str(event.get("type") or "touch")
    tags.extend(["active_learning", "suppressed_candidate", f"likely_missed_{kind}"])
    reason = str(event.get("qa_suppressed_reason") or "").strip()
    if reason:
        tags.append(f"suppressed_{reason}")
    return sorted(dict.fromkeys(tag for tag in tags if tag))


def gap_drop_review_tags(rally: dict[str, Any]) -> list[str]:
    tags = [
        "drop_floor",
        "reset_candidate",
        "active_learning",
        "likely_missed_drop_floor",
        "gap_without_floor_reset",
        "likely_missing_floor_reset",
    ]
    gap = maybe_float(rally.get("next_contact_gap_sec")) or 0.0
    if gap >= 4.0:
        tags.append("large_rally_gap")
    if int(rally.get("touches") or 0) >= 6:
        tags.append("strict_best_rally_blocker")
    return tags


def resolve_event_path(run: dict[str, Any]) -> Path:
    raw = str(run.get("qa_events_path") or "")
    path = ROOT / raw
    if path.exists():
        return path
    return Path(raw)


def resolve_video_path(run: dict[str, Any], source_video: str) -> Path:
    raw = str(run.get("video") or "")
    path = Path(raw)
    if path.exists():
        return path
    downloads = Path.home() / "Downloads" / Path(source_video).name
    if downloads.exists():
        return downloads
    return path


def contact_events_for_rally(doc: dict[str, Any], rally_id: int) -> list[dict[str, Any]]:
    return [
        item
        for item in doc.get("events", [])
        if int(item.get("qa_rally_id") or -1) == rally_id and item.get("type") in REVIEWABLE_KINDS
    ]


def first_contact_after(doc: dict[str, Any], time_sec: float) -> dict[str, Any] | None:
    contacts = [
        item
        for item in doc.get("events", [])
        if item.get("type") in REVIEWABLE_KINDS and event_time(item) > time_sec + 0.05
    ]
    if not contacts:
        return None
    return min(contacts, key=event_time)


def gap_drop_proposal_record(
    *,
    doc: dict[str, Any],
    run: dict[str, Any],
    rally: dict[str, Any],
    event_path: Path,
    source_video: str,
    video_path: Path,
    review_stem: str,
    qa_sheet_path: Path | None,
    video_index: int,
) -> dict[str, Any] | None:
    if not rally.get("ended_by_gap_without_floor_reset"):
        return None
    rally_id = int(rally.get("id") or 0)
    if rally_id <= 0:
        return None
    end_sec = maybe_float(rally.get("end_sec"))
    gap_sec = maybe_float(rally.get("next_contact_gap_sec"))
    if end_sec is None or gap_sec is None or gap_sec < 2.25:
        return None
    proposal_time = round(end_sec + min(1.15, gap_sec * 0.55), 3)
    rally_events = contact_events_for_rally(doc, rally_id)
    last_event = max(rally_events, key=event_time) if rally_events else None
    next_event = first_contact_after(doc, end_sec)
    x = None
    y = None
    if last_event and next_event:
        last_x = maybe_float(last_event.get("qa_ball_x", last_event.get("x")))
        next_x = maybe_float(next_event.get("qa_ball_x", next_event.get("x")))
        last_y = maybe_float(last_event.get("qa_ball_y", last_event.get("y")))
        next_y = maybe_float(next_event.get("qa_ball_y", next_event.get("y")))
        if last_x is not None and next_x is not None:
            x = round((last_x + next_x) / 2, 2)
        if last_y is not None and next_y is not None:
            y = round(max(last_y, next_y, OUT_SIZE[1] * 0.78), 2)
    elif last_event:
        x = last_event.get("qa_ball_x", last_event.get("x"))
        y = max(maybe_float(last_event.get("qa_ball_y", last_event.get("y"))) or OUT_SIZE[1] * 0.78, OUT_SIZE[1] * 0.78)
    item_id = f"miss-r{rally_id:03d}-dropgap-{int(round(proposal_time * 1000)):07d}"
    tags = gap_drop_review_tags(rally)
    event = {
        "type": "drop_floor",
        "time_sec": proposal_time,
        "qa_ball_x": x,
        "qa_ball_y": y,
        "qa_ball_radius": 18.0,
        "qa_ball_confidence": 0.0,
        "qa_ball_accuracy": "unknown",
        "contact_side": "unknown",
        "contact_type": "ground",
        "contact_confidence": 0.0,
        "drop_source": "gap_without_floor_reset",
        "drop_score": None,
        "label": "active-learning missing floor reset",
        "note": f"rally ended by {gap_sec:.2f}s gap without floor reset",
    }
    record = {
        "batch_item_id": f"{review_stem}__{item_id}",
        "review_item_id": item_id,
        "review_stem": review_stem,
        "source": "active_learning",
        "proposed_status": "missing",
        "source_video": source_video,
        "video_path": str(video_path),
        "qa_events_path": rel(event_path),
        "qa_sheet_path": None if qa_sheet_path is None or not qa_sheet_path.exists() else rel(qa_sheet_path),
        "event_index": f"gap-rally-{rally_id}",
        "video_index": video_index,
        "kind": "drop_floor",
        "time_sec": proposal_time,
        "frame_time_sec": proposal_time,
        "rally_id": rally_id,
        "touch_number": None,
        "confidence": None,
        "audio_z": None,
        "visual_score": None,
        "motion_score": None,
        "qa_ball_x": x,
        "qa_ball_y": y,
        "qa_ball_radius": 18.0,
        "qa_ball_confidence": 0.0,
        "qa_ball_accuracy": "unknown",
        "qa_ball_correction_px": None,
        "foot_x": None,
        "foot_y": None,
        "foot_distance": None,
        "contact_side": "unknown",
        "contact_type": "ground",
        "contact_confidence": 0.0,
        "drop_source": "gap_without_floor_reset",
        "drop_score": None,
        "label": event["label"],
        "note": event["note"],
        "review_tags": tags,
        "gap_start_sec": end_sec,
        "gap_duration_sec": round(gap_sec, 3),
        "blocked_rally_quality_score": rally.get("quality_score"),
        "blocked_rally_touches": rally.get("touches"),
    }
    record["priority_score"] = score_record("drop_floor", tags, event) + min(0.6, gap_sec / 8.0)
    return record


def collect_records(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path)
    records: list[dict[str, Any]] = []
    for video_index, run in enumerate(manifest.get("runs", [])):
        event_path = resolve_event_path(run)
        if not event_path.exists():
            continue
        doc = read_json(event_path)
        source_video = str(doc.get("source_video") or Path(str(run.get("video") or event_path.parent.name)).name)
        video_path = resolve_video_path(run, source_video)
        review_stem = slug_id(Path(source_video).stem or event_path.parent.name)
        qa_sheet_path = ROOT / str(run.get("qa_sheet_path", ""))
        if not qa_sheet_path.exists():
            qa_sheet_path = None
        for event_index, event in enumerate(doc.get("events", [])):
            kind = str(event.get("type") or "")
            if kind not in REVIEWABLE_KINDS:
                continue
            item = candidate_item(event, event_path)
            if not item:
                continue
            tags = event_review_tags(event)
            record = {
                "batch_item_id": f"{review_stem}__{item['id']}",
                "review_item_id": item["id"],
                "review_stem": review_stem,
                "source_video": source_video,
                "video_path": str(video_path),
                "qa_events_path": rel(event_path),
                "qa_sheet_path": None if qa_sheet_path is None or not qa_sheet_path.exists() else rel(qa_sheet_path),
                "event_index": event_index,
                "video_index": video_index,
                "kind": kind,
                "time_sec": round(event_time(event), 3),
                "frame_time_sec": round(event_frame_time(event), 3),
                "rally_id": event.get("qa_rally_id") or event.get("rally_id"),
                "touch_number": event.get("qa_touch_index") or event.get("touch_number"),
                "confidence": event.get("confidence"),
                "audio_z": event.get("audio_z"),
                "visual_score": event.get("visual_score"),
                "motion_score": event.get("motion_score"),
                "qa_ball_x": event.get("qa_ball_x", event.get("x")),
                "qa_ball_y": event.get("qa_ball_y", event.get("y")),
                "qa_ball_radius": event.get("qa_ball_radius", 18.0),
                "qa_ball_confidence": event.get("qa_ball_confidence"),
                "qa_ball_accuracy": event.get("qa_ball_accuracy"),
                "qa_ball_correction_px": event.get("qa_ball_correction_px"),
                "foot_x": event.get("foot_x"),
                "foot_y": event.get("foot_y"),
                "foot_confidence": event.get("foot_confidence"),
                "foot_distance": event.get("foot_distance"),
                "contact_side": event.get("contact_side"),
                "contact_type": event.get("contact_type"),
                "contact_confidence": event.get("contact_confidence"),
                "side_confidence": event.get("side_confidence"),
                "side_source": event.get("side_source"),
                "side_uncertainty_reason": event.get("side_uncertainty_reason"),
                "drop_source": event.get("drop_source"),
                "drop_score": event.get("drop_score"),
                "precision_gate_hard_veto": event.get("precision_gate_hard_veto"),
                "precision_gate_soft_flag": event.get("precision_gate_soft_flag"),
                "precision_gate_reasons": event.get("precision_gate_reasons"),
                "precision_gate_hard_reasons": event.get("precision_gate_hard_reasons"),
                "precision_gate_soft_reasons": event.get("precision_gate_soft_reasons"),
                "precision_gate_score": event.get("precision_gate_score"),
                "label": event.get("label"),
                "note": event.get("note"),
                "review_tags": tags,
            }
            record["priority_score"] = score_record(kind, tags, event)
            records.append(record)
        for suppressed_index, event in enumerate(doc.get("suppressed_events", [])):
            kind = str(event.get("type") or "")
            if kind not in REVIEWABLE_KINDS:
                continue
            time_sec = round(event_time(event), 3)
            rally_id = int(event.get("qa_rally_id") or event.get("rally_id") or 0)
            item_id = f"miss-r{rally_id:03d}-{kind[:5]}-{int(round(time_sec * 1000)):07d}"
            tags = suppressed_review_tags(event)
            record = {
                "batch_item_id": f"{review_stem}__{item_id}",
                "review_item_id": item_id,
                "review_stem": review_stem,
                "source": "active_learning",
                "proposed_status": "missing",
                "source_video": source_video,
                "video_path": str(video_path),
                "qa_events_path": rel(event_path),
                "qa_sheet_path": None if qa_sheet_path is None or not qa_sheet_path.exists() else rel(qa_sheet_path),
                "event_index": f"suppressed-{suppressed_index}",
                "video_index": video_index,
                "kind": kind,
                "time_sec": time_sec,
                "frame_time_sec": round(event_frame_time(event), 3),
                "rally_id": rally_id or None,
                "touch_number": event.get("qa_touch_index") or event.get("touch_number"),
                "confidence": event.get("confidence"),
                "audio_z": event.get("audio_z"),
                "visual_score": event.get("visual_score"),
                "motion_score": event.get("motion_score"),
                "qa_ball_x": event.get("qa_ball_x", event.get("x")),
                "qa_ball_y": event.get("qa_ball_y", event.get("y")),
                "qa_ball_radius": event.get("qa_ball_radius", 18.0),
                "qa_ball_confidence": event.get("qa_ball_confidence"),
                "qa_ball_accuracy": event.get("qa_ball_accuracy"),
                "qa_ball_correction_px": event.get("qa_ball_correction_px"),
                "foot_x": event.get("foot_x"),
                "foot_y": event.get("foot_y"),
                "foot_confidence": event.get("foot_confidence"),
                "foot_distance": event.get("foot_distance"),
                "contact_side": event.get("contact_side"),
                "contact_type": event.get("contact_type"),
                "contact_confidence": event.get("contact_confidence"),
                "side_confidence": event.get("side_confidence"),
                "side_source": event.get("side_source"),
                "side_uncertainty_reason": event.get("side_uncertainty_reason"),
                "drop_source": event.get("drop_source"),
                "drop_score": event.get("drop_score"),
                "precision_gate_hard_veto": event.get("precision_gate_hard_veto"),
                "precision_gate_soft_flag": event.get("precision_gate_soft_flag"),
                "precision_gate_reasons": event.get("precision_gate_reasons"),
                "precision_gate_hard_reasons": event.get("precision_gate_hard_reasons"),
                "precision_gate_soft_reasons": event.get("precision_gate_soft_reasons"),
                "precision_gate_score": event.get("precision_gate_score"),
                "label": event.get("label"),
                "note": event.get("note") or f"active_learning_suppressed_{kind}",
                "qa_suppressed_reason": event.get("qa_suppressed_reason"),
                "review_tags": tags,
            }
            record["priority_score"] = score_record(kind, tags, event)
            records.append(record)
        for rally in doc.get("rallies", []):
            record = gap_drop_proposal_record(
                doc=doc,
                run=run,
                rally=rally,
                event_path=event_path,
                source_video=source_video,
                video_path=video_path,
                review_stem=review_stem,
                qa_sheet_path=qa_sheet_path,
                video_index=video_index,
            )
            if record is not None:
                records.append(record)
    return records


def add_selection(selected: dict[str, dict[str, Any]], record: dict[str, Any], reason: str, max_items: int, *, force: bool = False) -> None:
    if record["batch_item_id"] in selected:
        reasons = selected[record["batch_item_id"]].setdefault("selection_reasons", [])
        if reason not in reasons:
            reasons.append(reason)
        return
    if len(selected) >= max_items and not force:
        return
    item = dict(record)
    item["selection_reasons"] = [reason]
    selected[item["batch_item_id"]] = item


def select_records(records: list[dict[str, Any]], max_items: int, per_video: int) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_video[str(record["source_video"])].append(record)
        by_kind[str(record["kind"])].append(record)

    active_records = [record for record in records if record.get("source") == "active_learning"]
    candidate_records = [record for record in records if record.get("source") != "active_learning"]
    ranked_candidates = sorted(candidate_records, key=lambda item: (-float(item["priority_score"]), int(item["video_index"]), float(item["time_sec"])))
    ranked_active = sorted(active_records, key=lambda item: (-float(item["priority_score"]), int(item["video_index"]), float(item["time_sec"])))

    for record in sorted([item for item in by_kind.get("around_the_world", []) if item.get("source") != "active_learning"], key=lambda item: (int(item["video_index"]), float(item["time_sec"]))):
        add_selection(selected, record, "all_trick_candidates", max_items, force=True)

    gap_records = sorted(
        [record for record in active_records if "gap_without_floor_reset" in record.get("review_tags", [])],
        key=lambda item: (-float(item["priority_score"]), int(item["video_index"]), float(item["time_sec"])),
    )
    gap_target = max(6, int(max_items * 0.10))
    while sum(1 for item in selected.values() if "gap_without_floor_reset" in item.get("review_tags", [])) < gap_target and gap_records:
        add_selection(selected, gap_records.pop(0), "active_learning_gap_reset_quota", max_items)
        if len(selected) >= max_items:
            break

    active_targets = {
        "touch": max(4, int(max_items * 0.10)),
        "drop_floor": max(3, int(max_items * 0.08)),
        "stall": max(2, int(max_items * 0.05)),
        "around_the_world": max(1, int(max_items * 0.04)),
    }
    for kind, target in active_targets.items():
        ranked = sorted(
            [record for record in active_records if record["kind"] == kind],
            key=lambda item: (-float(item["priority_score"]), int(item["video_index"]), float(item["time_sec"])),
        )
        while sum(1 for item in selected.values() if item.get("source") == "active_learning" and item["kind"] == kind) < target and ranked:
            add_selection(selected, ranked.pop(0), f"active_learning_quota_{kind}", max_items)
            if len(selected) >= max_items:
                break

    for video, video_records in sorted(by_video.items()):
        del video
        candidates = [record for record in video_records if record.get("source") != "active_learning"]
        ranked = sorted(candidates or video_records, key=lambda item: (-float(item["priority_score"]), float(item["time_sec"])))
        if ranked:
            add_selection(selected, ranked[0], "per_video_minimum_coverage", max_items)

    kind_targets = {
        "touch": max(12, int(max_items * 0.42)),
        "drop_floor": max(8, int(max_items * 0.22)),
        "stall": max(8, int(max_items * 0.28)),
        "around_the_world": len(by_kind.get("around_the_world", [])),
    }
    for kind, target in kind_targets.items():
        ranked = sorted(
            [record for record in by_kind.get(kind, []) if record.get("source") != "active_learning"],
            key=lambda item: (-float(item["priority_score"]), int(item["video_index"]), float(item["time_sec"])),
        )
        while sum(1 for item in selected.values() if item["kind"] == kind) < target and ranked:
            add_selection(selected, ranked.pop(0), f"kind_quota_{kind}", max_items)
            if len(selected) >= max_items:
                break

    for video, video_records in sorted(by_video.items()):
        del video
        current_for_video = sum(1 for item in selected.values() if item["source_video"] == video_records[0]["source_video"])
        ranked = sorted(
            [record for record in video_records if record.get("source") != "active_learning"],
            key=lambda item: (-float(item["priority_score"]), float(item["time_sec"])),
        )
        for record in ranked:
            if current_for_video >= per_video or len(selected) >= max_items:
                break
            before = len(selected)
            add_selection(selected, record, "per_video_risk_coverage", max_items)
            if len(selected) > before:
                current_for_video += 1

    for record in ranked_candidates:
        if len(selected) >= max_items:
            break
        add_selection(selected, record, "priority_fill", max_items)
    for record in ranked_active:
        if len(selected) >= max_items:
            break
        add_selection(selected, record, "active_learning_priority_fill", max_items)

    return sorted(selected.values(), key=lambda item: (int(item["video_index"]), float(item["time_sec"]), str(item["review_item_id"])))


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def read_frame(cap: cv2.VideoCapture, fps: float, time_sec: float) -> Image.Image | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(time_sec * fps))))
    ok, frame = cap.read()
    if not ok:
        return None
    frame = cv2.resize(frame, OUT_SIZE, interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def wrap_text(text: str, chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def crop_event(frame: Image.Image, record: dict[str, Any]) -> tuple[Image.Image, tuple[float, float, float, float]]:
    width, height = frame.size
    cx = maybe_float(record.get("qa_ball_x"))
    cy = maybe_float(record.get("qa_ball_y"))
    if cx is None or cy is None:
        cx, cy = width / 2, height * 0.72
    crop_size = 330
    left = max(0, min(width - crop_size, int(round(cx - crop_size / 2))))
    top = max(0, min(height - crop_size, int(round(cy - crop_size / 2))))
    box = (left, top, left + crop_size, top + crop_size)
    crop = frame.crop(box).resize((CROP_W, CROP_H), Image.Resampling.LANCZOS)
    return crop, box


def draw_marker(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float], x: Any, y: Any, color: tuple[int, int, int], radius: int) -> None:
    fx = maybe_float(x)
    fy = maybe_float(y)
    if fx is None or fy is None:
        return
    left, top, right, bottom = box
    sx = CROP_W / (right - left)
    sy = CROP_H / (bottom - top)
    px = (fx - left) * sx
    py = (fy - top) * sy
    if px < -radius or px > CROP_W + radius or py < -radius or py > CROP_H + radius:
        return
    draw.ellipse((px - radius, py - radius, px + radius, py + radius), outline=(0, 0, 0), width=5)
    draw.ellipse((px - radius, py - radius, px + radius, py + radius), outline=color, width=3)


def draw_foot_marker(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float], x: Any, y: Any) -> None:
    fx = maybe_float(x)
    fy = maybe_float(y)
    if fx is None or fy is None:
        return
    left, top, right, bottom = box
    sx = CROP_W / (right - left)
    sy = CROP_H / (bottom - top)
    px = (fx - left) * sx
    py = (fy - top) * sy
    if px < -12 or px > CROP_W + 12 or py < -12 or py > CROP_H + 12:
        return
    draw.line((px - 11, py, px + 11, py), fill=(64, 180, 255), width=3)
    draw.line((px, py - 11, px, py + 11), fill=(64, 180, 255), width=3)


def marked_crop_for_record(frame: Image.Image, record: dict[str, Any]) -> Image.Image:
    crop, box = crop_event(frame, record)
    crop_draw = ImageDraw.Draw(crop)
    accuracy = str(record.get("qa_ball_accuracy") or "")
    ball_color = (110, 230, 130) if accuracy == "high" else (255, 220, 75) if accuracy == "medium" else (255, 92, 85)
    draw_marker(crop_draw, box, record.get("qa_ball_x"), record.get("qa_ball_y"), ball_color, 14)
    draw_foot_marker(crop_draw, box, record.get("foot_x"), record.get("foot_y"))
    return crop


def tile_for_record(index: int, record: dict[str, Any], frame: Image.Image | None, fonts: dict[str, ImageFont.ImageFont]) -> Image.Image:
    tile = Image.new("RGB", (TILE_W, TILE_H), (20, 22, 25))
    draw = ImageDraw.Draw(tile)
    accent = {
        "touch": (255, 218, 80),
        "drop_floor": (255, 116, 90),
        "stall": (116, 214, 138),
        "around_the_world": (255, 142, 230),
    }.get(str(record["kind"]), (220, 220, 220))
    draw.rectangle((0, 0, TILE_W, 6), fill=accent)
    prefix = "MISS?" if record.get("source") == "active_learning" else str(record["kind"])
    title = f"#{index:03d} {prefix} {float(record['time_sec']):.2f}s"
    draw.text((10, 13), title, fill=(246, 248, 250), font=fonts["bold"])
    video_name = str(record["source_video"]).replace("_singular_display", "")
    draw.text((10, 34), video_name[:42], fill=(170, 178, 188), font=fonts["small"])

    if frame is None:
        draw.rectangle((14, 58, 14 + CROP_W, 58 + CROP_H), fill=(6, 7, 8), outline=(75, 82, 92))
        draw.text((28, 166), "missing frame", fill=(230, 90, 90), font=fonts["body"])
    else:
        crop = marked_crop_for_record(frame, record)
        tile.paste(crop, (14, 58))
        draw.rectangle((14, 58, 14 + CROP_W, 58 + CROP_H), outline=(75, 82, 92), width=1)

    meta = (
        f"R{record.get('rally_id') or '-'} T{record.get('touch_number') or '-'} "
        f"{record.get('contact_side') or '?'} {record.get('contact_type') or '?'} "
        f"conf {record.get('confidence') if record.get('confidence') is not None else '-'}"
    )
    draw.text((10, 318), meta[:52], fill=(240, 240, 240), font=fonts["small"])
    tags = " ".join(record.get("review_tags", [])[:6])
    for row, line in enumerate(wrap_text(tags, 48)[:2]):
        draw.text((10, 339 + row * 17), line, fill=(166, 206, 255), font=fonts["small"])
    draw.text((10, 370), "ball circle / foot cross", fill=(122, 130, 140), font=fonts["tiny"])
    return tile


def write_contact_sheet(records: list[dict[str, Any]], out_dir: Path, columns: int = SHEET_COLUMNS) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "crops"
    tiles_dir = out_dir / "tiles"
    crops_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    fonts = {
        "bold": load_font(16, bold=True),
        "body": load_font(14),
        "small": load_font(12),
        "tiny": load_font(10),
    }
    caps: dict[str, tuple[cv2.VideoCapture, float]] = {}
    tiles: list[Image.Image] = []
    for index, record in enumerate(records, start=1):
        video_path = str(record.get("video_path") or "")
        cap_tuple = caps.get(video_path)
        if cap_tuple is None:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap_tuple = (cap, fps)
            caps[video_path] = cap_tuple
        cap, fps = cap_tuple
        frame = read_frame(cap, fps, float(record.get("frame_time_sec") or record.get("time_sec") or 0.0)) if cap.isOpened() else None
        tile = tile_for_record(index, record, frame, fonts)
        tiles.append(tile)
        tile_path = tiles_dir / f"{index:03d}_{slug_id(record['batch_item_id'])}.jpg"
        tile.save(tile_path, quality=91)
        if frame is not None:
            crop_path = crops_dir / f"{index:03d}_{slug_id(record['batch_item_id'])}.jpg"
            marked_crop_for_record(frame, record).save(crop_path, quality=91)
            record["crop_path"] = rel(crop_path)
        record["tile_path"] = rel(tile_path)
    for cap, _fps in caps.values():
        cap.release()

    rows = max(1, math.ceil(len(tiles) / columns))
    sheet = Image.new("RGB", (columns * TILE_W, rows * TILE_H), (12, 13, 15))
    for index, tile in enumerate(tiles):
        x = (index % columns) * TILE_W
        y = (index // columns) * TILE_H
        sheet.paste(tile, (x, y))
    sheet_path = out_dir / "review_batch_sheet.jpg"
    sheet.save(sheet_path, quality=92)
    return sheet_path


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def write_report(path: Path, batch: dict[str, Any]) -> None:
    kind_counts = Counter(item["kind"] for item in batch["items"])
    source_counts = Counter(item.get("source", "candidate") for item in batch["items"])
    tag_counts = Counter(tag for item in batch["items"] for tag in item.get("review_tags", []))
    video_counts = Counter(item["source_video"] for item in batch["items"])
    lines = [
        "# Hacky Track Review Batch",
        "",
        f"- Batch id: `{batch['batch_id']}`",
        f"- Items: {len(batch['items'])}",
        f"- Manifest videos: {batch['summary'].get('manifest_videos', 'n/a')}",
        f"- Videos with reviewable events: {batch['summary'].get('videos_with_reviewable_events', 'n/a')}",
        f"- Videos covered: {len(video_counts)}",
        f"- Contact sheet: `{batch['contact_sheet_path']}`",
        f"- JSONL: `{batch['jsonl_path']}`",
        "",
        "## Kind Counts",
        "",
    ]
    for kind, count in sorted(kind_counts.items()):
        lines.append(f"- {kind}: {count}")
    lines.extend(["", "## Source Counts", ""])
    for source, count in sorted(source_counts.items()):
        lines.append(f"- {source}: {count}")
    lines.extend(["", "## Top Review Tags", ""])
    for tag, count in tag_counts.most_common(18):
        lines.append(f"- {tag}: {count}")
    no_event = batch["summary"].get("no_reviewable_event_videos") or []
    if no_event:
        lines.extend(["", "## No Reviewable Event Videos", ""])
        for video in no_event:
            lines.append(f"- `{video}`")
    lines.extend(
        [
            "",
            "## Review Flow",
            "",
            "1. Open `python3 review_app.py --port 8765`.",
            "2. Pick the matching video from the dropdown using `review_stem` or `source_video`.",
            "3. Use the `Batch` filter to focus only this priority batch, or filter by high-risk tags like `active_learning`, `likely_missed_touch`, `floor_risk_touch`, `hidden_drop`, `large_ball_correction`, `ambiguous_contact`, or by the `review_item_id`.",
            "4. Keyboard shortcuts: `a` approve, `r` reject, `p` pending, `q` toggle batch, arrows/`n`/`b` move, `m` add missing touch, `d` add drop, `s` add stall. Approving an active-learning item records it as a missing event.",
            "5. Approve/reject candidates, add missing touches/drops/stalls, and correct side/contact/move labels.",
            "6. Run `python3 review_validation.py` to refresh validation and batch coverage.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_batch(manifest_path: Path, out_dir: Path, max_items: int, per_video: int) -> dict[str, Any]:
    records = collect_records(manifest_path)
    manifest = read_json(manifest_path)
    manifest_videos = [Path(str(run.get("video") or "")).name for run in manifest.get("runs", [])]
    videos_with_records = {str(record["source_video"]) for record in records}
    no_reviewable_event_videos = [video for video in manifest_videos if video and video not in videos_with_records]
    selected = select_records(records, max_items=max_items, per_video=per_video)
    batch_id = f"{manifest_path.parent.name}_{len(selected)}_review_items"
    batch_dir = out_dir / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = write_contact_sheet(selected, batch_dir)
    jsonl_path = batch_dir / "review_batch_items.jsonl"
    write_jsonl(jsonl_path, selected)
    batch = {
        "batch_id": batch_id,
        "manifest_path": rel(manifest_path),
        "total_candidate_events": len(records),
        "selected_items": len(selected),
        "contact_sheet_path": rel(sheet_path),
        "jsonl_path": rel(jsonl_path),
        "items": selected,
        "summary": {
            "kind_counts": dict(Counter(item["kind"] for item in selected)),
            "source_counts": dict(Counter(item.get("source", "candidate") for item in selected)),
            "tag_counts": dict(Counter(tag for item in selected for tag in item.get("review_tags", []))),
            "manifest_videos": len(manifest_videos),
            "videos_with_reviewable_events": len(videos_with_records),
            "videos_covered": len({item["source_video"] for item in selected}),
            "no_reviewable_event_videos": no_reviewable_event_videos,
        },
    }
    write_json(batch_dir / "review_batch.json", batch)
    write_json(out_dir / "latest_review_batch.json", batch)
    write_report(batch_dir / "review_batch_report.md", batch)
    write_report(out_dir / "latest_review_batch_report.md", batch)
    return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a balanced visual review batch from QA events")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-items", type=int, default=96)
    parser.add_argument("--per-video", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch = build_batch(args.manifest, args.out_dir, max_items=args.max_items, per_video=args.per_video)
    print(f"review batch: {args.out_dir / batch['batch_id'] / 'review_batch.json'}")
    print(f"sheet: {ROOT / batch['contact_sheet_path']}")
    print(f"items: {batch['selected_items']} of {batch['total_candidate_events']} candidate events")


if __name__ == "__main__":
    main()
