#!/usr/bin/env python3
"""Build detector-specific ball-label review batches from QA events.

The event review queue answers "was this touch/drop/stall real?". A detector
training set needs a different review target: "is the marked object actually the
footbag, and is the center correct?". This script mines QA events into a
separate pending label-review queue so more detector positives and hard
negatives can be reviewed without mixing them into clean metrics first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_SIZE = (688, 912)
POSITIVE_ACCURACY = {"high", "medium", "reviewed"}
REVIEW_ACCURACY = {"low", "unknown", "", "none"}


@dataclass(frozen=True)
class Candidate:
    record: dict[str, Any]
    video_path: Path


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


def safe_stem(text: str) -> str:
    stem = Path(text).stem or text
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", stem).strip("-") or "item"


def numeric(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if math.isfinite(value_f) else None


def resolve_video_path(raw: Any, video_root: Path | None = None) -> Path:
    path = Path(str(raw or "")).expanduser()
    candidates = [path]
    if video_root is not None:
        candidates.append(video_root.expanduser() / path.name)
        if not path.is_absolute():
            candidates.append(video_root.expanduser() / path)
    if not path.is_absolute():
        candidates.append(ROOT / path)
    candidates.append(Path.home() / "Downloads" / path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return path.resolve() if path.is_absolute() else path


def event_center(event: dict[str, Any]) -> tuple[float, float] | None:
    x = numeric(event.get("qa_ball_x")) or numeric(event.get("x"))
    y = numeric(event.get("qa_ball_y")) or numeric(event.get("y"))
    if x is None or y is None:
        return None
    return x, y


def event_time(event: dict[str, Any]) -> float | None:
    return numeric(event.get("qa_frame_time_sec")) or numeric(event.get("time_sec")) or numeric(event.get("start_sec"))


def event_radius(event: dict[str, Any], default_radius: float) -> float:
    radius = numeric(event.get("qa_ball_radius")) or numeric(event.get("radius")) or default_radius
    return max(8.0, min(48.0, radius))


def existing_reviewed_keys(reviews_dir: Path | None) -> set[tuple[str, str]]:
    if reviews_dir is None or not reviews_dir.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    for path in reviews_dir.glob("*.review.json"):
        doc = read_json(path)
        source_video = Path(str(doc.get("source_video") or path.stem)).name
        for item in doc.get("items", []):
            status = str(item.get("status") or "")
            if status in {"approved", "rejected", "missing"}:
                item_id = str(item.get("id") or "")
                if item_id:
                    keys.add((source_video, item_id))
    return keys


def label_priority(event: dict[str, Any], split: str | None = None) -> tuple[float, list[str], str]:
    reasons: list[str] = []
    priority = 0.0
    accuracy = str(event.get("qa_ball_accuracy") or "").lower()
    confidence = numeric(event.get("qa_ball_confidence")) or 0.0
    correction = numeric(event.get("qa_ball_correction_px")) or 0.0
    kind = str(event.get("type") or "event")

    suggested = "footbag"
    if accuracy in POSITIVE_ACCURACY:
        priority += 1.0
        reasons.append(f"qa_ball_{accuracy}")
    if confidence >= 0.70:
        priority += 0.9
        reasons.append("high_ball_confidence")
    elif confidence >= 0.55:
        priority += 0.45
        reasons.append("usable_ball_confidence")
    else:
        priority += 1.2
        reasons.append("low_ball_confidence_review")
        suggested = "verify_or_correct"
    if correction >= 120:
        priority += 1.4
        reasons.append("large_heuristic_correction")
        suggested = "verify_or_correct"
    elif correction >= 60:
        priority += 0.7
        reasons.append("moderate_heuristic_correction")
    if kind in {"drop_floor", "stall"}:
        priority += 0.45
        reasons.append(f"{kind}_coverage")
    if split in {"validation", "test"}:
        priority += 0.35
        reasons.append(f"{split}_split_coverage")
    if accuracy in REVIEW_ACCURACY:
        priority += 1.0
        reasons.append("unknown_ball_accuracy")
        suggested = "verify_or_correct"
    return round(priority, 4), reasons, suggested


def load_splits(dataset_manifest: Path | None) -> dict[str, str]:
    if dataset_manifest is None or not dataset_manifest.exists():
        return {}
    doc = read_json(dataset_manifest)
    return {str(video): str(split) for video, split in (doc.get("splits") or {}).items()}


def collect_candidates(
    qa_manifest: Path,
    *,
    reviews_dir: Path | None = None,
    dataset_manifest: Path | None = None,
    video_root: Path | None = None,
    default_radius: float = 22.0,
    exclude_reviewed: bool = True,
) -> list[Candidate]:
    manifest = read_json(qa_manifest)
    reviewed = existing_reviewed_keys(reviews_dir) if exclude_reviewed else set()
    splits = load_splits(dataset_manifest)
    candidates: list[Candidate] = []
    for video_index, run in enumerate(manifest.get("runs", [])):
        raw_video = run.get("video") or run.get("source_video")
        if not raw_video:
            continue
        source_video = Path(str(raw_video)).name
        video_path = resolve_video_path(raw_video, video_root)
        qa_path = Path(str(run.get("qa_events_path") or ""))
        if not qa_path.exists():
            qa_path = ROOT / qa_path
        if not qa_path.exists():
            continue
        doc = read_json(qa_path)
        review_stem = safe_stem(Path(source_video).stem)
        split = splits.get(source_video)
        for event_index, event in enumerate(doc.get("events", [])):
            center = event_center(event)
            time_sec = event_time(event)
            if center is None or time_sec is None:
                continue
            event_ms = int(round(float(time_sec) * 1000))
            event_id = str(event.get("review_item_id") or f"cand-r{int(event.get('qa_rally_id') or event.get('rally_id') or 0):03d}-{str(event.get('type') or 'event')[:5]}-{event_ms:07d}")
            if exclude_reviewed and (source_video, event_id) in reviewed:
                continue
            priority, reasons, suggested = label_priority(event, split)
            digest = hashlib.sha1(f"{source_video}:{event_id}:{time_sec:.4f}".encode("utf-8")).hexdigest()[:10]
            record = {
                "batch_item_id": f"{review_stem}__detlbl-{event_id}",
                "detector_label_id": f"detlbl-{event_id}",
                "event_item_id": event_id,
                "review_stem": review_stem,
                "source": "qa_event_detector_label_candidate",
                "source_video": source_video,
                "video": source_video,
                "video_path": portable(video_path),
                "qa_events_path": portable(qa_path),
                "video_index": video_index,
                "event_index": event_index,
                "event_type": event.get("type"),
                "time_sec": float(time_sec),
                "event_time_sec": event.get("time_sec"),
                "x": round(float(center[0]), 3),
                "y": round(float(center[1]), 3),
                "radius": round(event_radius(event, default_radius), 3),
                "qa_ball_confidence": event.get("qa_ball_confidence"),
                "qa_ball_accuracy": event.get("qa_ball_accuracy"),
                "qa_ball_source": event.get("qa_ball_source"),
                "qa_ball_correction_px": event.get("qa_ball_correction_px"),
                "contact_side": event.get("contact_side"),
                "contact_type": event.get("contact_type"),
                "split": split,
                "priority_score": priority,
                "selection_reasons": reasons,
                "suggested_detector_label": suggested,
                "review_required": True,
                "review_decision_schema": {
                    "detector_status": ["footbag", "not_footbag", "corrected", "not_visible", "skip"],
                    "required_for_corrected": ["corrected_x", "corrected_y"],
                    "optional": ["radius", "evidence", "reviewer"],
                },
                "_video_path": str(video_path),
                "_digest": digest,
            }
            candidates.append(Candidate(record, video_path))
    candidates.sort(key=lambda item: (-float(item.record["priority_score"]), item.record["video_index"], item.record["time_sec"]))
    return candidates


def read_resized_frame(video: Path, time_sec: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(time_sec * fps))))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.resize(frame, OUT_SIZE, interpolation=cv2.INTER_AREA)


def crop_with_padding(frame: np.ndarray, x: float, y: float, size: int) -> np.ndarray:
    height, width = frame.shape[:2]
    half = size // 2
    left = max(0, int(round(x)) - half)
    right = min(width, int(round(x)) + half)
    top = max(0, int(round(y)) - half)
    bottom = min(height, int(round(y)) + half)
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.copyMakeBorder(
        crop,
        0,
        max(0, size - crop.shape[0]),
        0,
        max(0, size - crop.shape[1]),
        cv2.BORDER_CONSTANT,
        value=(18, 18, 18),
    )[:size, :size]


def render_tile(frame: np.ndarray, record: dict[str, Any], index: int, crop_size: int) -> np.ndarray:
    x = float(record["x"])
    y = float(record["y"])
    radius = float(record["radius"])
    crop = crop_with_padding(frame, x, y, crop_size)
    scale_x = crop_size / max(1, frame.shape[1])
    scale_y = crop_size / max(1, frame.shape[0])
    # Recompute marker in crop coordinates.
    half = crop_size // 2
    marker = (half, half)
    cv2.circle(crop, marker, max(5, int(round(radius * 0.8))), (0, 255, 255), 2)
    cv2.drawMarker(crop, marker, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
    footer = np.zeros((74, crop_size, 3), dtype=np.uint8)
    lines = [
        f"{index:03d} {record['source_video'][:22]}",
        f"{record['event_type']} {record['time_sec']:.2f}s {record['suggested_detector_label']}",
        f"{','.join(record['selection_reasons'][:2])[:30]}",
    ]
    for line_index, text in enumerate(lines):
        cv2.putText(footer, text, (6, 18 + 20 * line_index), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
    _ = scale_x, scale_y
    return np.vstack([crop, footer])


def render_outputs(candidates: list[Candidate], out_dir: Path, *, crop_size: int, cols: int) -> tuple[list[dict[str, Any]], Path | None]:
    crops_dir = out_dir / "crops"
    tiles_dir = out_dir / "tiles"
    crops_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    tiles: list[np.ndarray] = []
    for index, candidate in enumerate(candidates, start=1):
        record = dict(candidate.record)
        frame = read_resized_frame(candidate.video_path, float(record["time_sec"]))
        if frame is None:
            record["render_status"] = "missing_frame"
            records.append({k: v for k, v in record.items() if not k.startswith("_")})
            continue
        crop = crop_with_padding(frame, float(record["x"]), float(record["y"]), crop_size)
        tile = render_tile(frame, record, index, crop_size)
        crop_name = f"{index:03d}_{safe_stem(record['source_video'])}__{safe_stem(record['detector_label_id'])}__{record['_digest']}.jpg"
        tile_name = crop_name.replace(".jpg", "_tile.jpg")
        crop_path = crops_dir / crop_name
        tile_path = tiles_dir / tile_name
        cv2.imwrite(str(crop_path), crop)
        cv2.imwrite(str(tile_path), tile)
        record["crop_path"] = portable(crop_path)
        record["tile_path"] = portable(tile_path)
        record["render_status"] = "ok"
        records.append({k: v for k, v in record.items() if not k.startswith("_")})
        tiles.append(tile)

    if not tiles:
        return records, None
    cols = max(1, cols)
    rows = math.ceil(len(tiles) / cols)
    tile_h, tile_w = tiles[0].shape[:2]
    sheet = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        sheet[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = tile
    sheet_path = out_dir / "detector_label_review_sheet.jpg"
    cv2.imwrite(str(sheet_path), sheet)
    return records, sheet_path


def build_review_batch(
    qa_manifest: Path,
    out_dir: Path,
    *,
    reviews_dir: Path | None = None,
    dataset_manifest: Path | None = None,
    video_root: Path | None = None,
    max_items: int = 160,
    per_video: int = 8,
    default_radius: float = 22.0,
    crop_size: int = 192,
    cols: int = 4,
    dry_run: bool = False,
) -> dict[str, Any]:
    candidates = collect_candidates(
        qa_manifest,
        reviews_dir=reviews_dir,
        dataset_manifest=dataset_manifest,
        video_root=video_root,
        default_radius=default_radius,
    )
    selected: list[Candidate] = []
    per_video_counts: dict[str, int] = {}
    for candidate in candidates:
        video = str(candidate.record["source_video"])
        if per_video_counts.get(video, 0) >= per_video:
            continue
        selected.append(candidate)
        per_video_counts[video] = per_video_counts.get(video, 0) + 1
        if len(selected) >= max_items:
            break
    selected_records = [{k: v for k, v in item.record.items() if not k.startswith("_")} for item in selected]
    sheet_path: Path | None = None
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        selected_records, sheet_path = render_outputs(selected, out_dir, crop_size=crop_size, cols=cols)
        jsonl_path = out_dir / "detector_label_review_items.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for record in selected_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        decision_template = out_dir / "detector_label_decisions_template.json"
        write_json(
            decision_template,
            {
                "schema_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "instructions": "Fill detector_status as footbag, not_footbag, corrected, not_visible, or skip. For corrected, provide corrected_x/corrected_y in the 688x912 QA coordinate space.",
                "decisions": [
                    {
                        "detector_label_id": record["detector_label_id"],
                        "source_video": record["source_video"],
                        "detector_status": "pending",
                        "corrected_x": None,
                        "corrected_y": None,
                        "radius": record.get("radius"),
                        "evidence": "",
                    }
                    for record in selected_records
                ],
            },
        )
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qa_manifest": portable(qa_manifest),
        "reviews_dir": portable(reviews_dir),
        "dataset_manifest": portable(dataset_manifest),
        "out_dir": portable(out_dir),
        "total_candidates": len(candidates),
        "selected_items": len(selected_records),
        "per_video_limit": per_video,
        "max_items": max_items,
        "contact_sheet_path": portable(sheet_path),
        "jsonl_path": portable(out_dir / "detector_label_review_items.jsonl") if not dry_run else None,
        "decision_template_path": portable(out_dir / "detector_label_decisions_template.json") if not dry_run else None,
        "items": selected_records,
    }
    if not dry_run:
        write_json(out_dir / "detector_label_review_manifest.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a detector-label review batch from QA ball centers")
    parser.add_argument("--qa-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reviews-dir", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--max-items", type=int, default=160)
    parser.add_argument("--per-video", type=int, default=8)
    parser.add_argument("--default-radius", type=float, default=22.0)
    parser.add_argument("--crop-size", type=int, default=192)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_review_batch(
        args.qa_manifest,
        args.out_dir,
        reviews_dir=args.reviews_dir,
        dataset_manifest=args.dataset_manifest,
        video_root=args.video_root,
        max_items=args.max_items,
        per_video=args.per_video,
        default_radius=args.default_radius,
        crop_size=args.crop_size,
        cols=args.cols,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(f"manifest: {args.out_dir / 'detector_label_review_manifest.json'}")
        print(f"sheet: {summary.get('contact_sheet_path')}")
    print(json.dumps({k: summary[k] for k in ["total_candidates", "selected_items", "contact_sheet_path"]}, indent=2))


if __name__ == "__main__":
    main()
