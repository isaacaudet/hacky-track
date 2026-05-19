#!/usr/bin/env python3
"""Build detector error-review batches from model and track evaluation failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from build_detector_label_review_batch import OUT_SIZE, numeric, portable, resolve_video_path, safe_stem, write_json


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ErrorCandidate:
    record: dict[str, Any]
    video_path: Path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def qa_video_paths(qa_manifest: Path, video_root: Path | None = None) -> dict[str, Path]:
    manifest = read_json(qa_manifest)
    videos: dict[str, Path] = {}
    for run in manifest.get("runs", []):
        raw_video = run.get("video") or run.get("source_video")
        if not raw_video:
            continue
        name = Path(str(raw_video)).name
        video_path = resolve_video_path(raw_video, video_root)
        videos[name] = video_path
        videos[Path(name).stem] = video_path
        videos[safe_stem(name)] = video_path
    return videos


def dataset_labels(dataset: Path | None) -> dict[str, dict[str, Any]]:
    if dataset is None:
        return {}
    labels_path = dataset / "reviewed_detector_labels.jsonl"
    labels = load_jsonl(labels_path)
    by_image: dict[str, dict[str, Any]] = {}
    for row in labels:
        image = str(row.get("image") or "")
        label = str(row.get("label") or "")
        if image:
            by_image[image] = row
        if label:
            by_image[label] = row
    return by_image


def candidate_digest(*parts: Any) -> str:
    return hashlib.sha1(":".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:10]


def candidate_priority(record: dict[str, Any]) -> float:
    priority = 0.0
    reason = str(record.get("selection_reason") or "")
    split = str(record.get("split") or "")
    video = str(record.get("source_video") or "")
    if reason == "track_center_fail":
        priority += 4.0
    elif reason == "missing_track_point":
        priority += 3.4
    elif reason == "hard_negative_false_positive":
        priority += 4.5
    elif reason == "model_center_fail":
        priority += 3.2
    elif reason == "low_confidence_near_label":
        priority += 1.6
    if video == "video-352_singular_display 2.MOV":
        priority += 1.2
    if split == "train":
        priority += 0.6
    elif split == "validation":
        priority += 0.4
    error = numeric(record.get("center_error_px")) or 0.0
    priority += min(2.0, error / 120.0)
    confidence = numeric(record.get("confidence")) or numeric(record.get("model_confidence")) or 0.0
    if reason == "low_confidence_near_label":
        priority += max(0.0, 0.05 - confidence)
    else:
        priority += min(0.6, confidence)
    return round(priority, 6)


def make_candidate(
    *,
    source_video: str,
    video_path: Path,
    item_id: str,
    split: str | None,
    time_sec: float,
    x: float,
    y: float,
    radius: float,
    event_type: str,
    selection_reason: str,
    suggested_label: str,
    source: str,
    track_x: float | None = None,
    track_y: float | None = None,
    model_confidence: float | None = None,
    center_error_px: float | None = None,
    threshold_px: float | None = None,
    evidence: str = "",
    training_use: str | None = None,
) -> ErrorCandidate:
    split_text = str(split or "train")
    use = training_use or ("audit_only" if split_text == "test" else "train_or_calibration")
    digest = candidate_digest(source_video, item_id, selection_reason, f"{time_sec:.4f}", f"{x:.2f}", f"{y:.2f}")
    label_id = f"v11err-{safe_stem(source_video)}-{safe_stem(item_id)}-{selection_reason}-{digest}"
    reasons = [selection_reason, "v10_detector_error", "needs_detector_label_review"]
    if use == "audit_only":
        reasons.append("audit_only_heldout_split")
    record = {
        "batch_item_id": f"{safe_stem(source_video)}__{label_id}",
        "detector_label_id": label_id,
        "event_item_id": item_id,
        "review_stem": safe_stem(Path(source_video).stem),
        "source": source,
        "source_video": source_video,
        "video": source_video,
        "video_id": safe_stem(source_video),
        "video_path": portable(video_path),
        "event_type": event_type,
        "time_sec": round(float(time_sec), 6),
        "time_s": round(float(time_sec), 6),
        "x": round(float(x), 3),
        "y": round(float(y), 3),
        "center_x": round(float(x), 3),
        "center_y": round(float(y), 3),
        "radius": round(float(max(8.0, min(48.0, radius))), 3),
        "split": split_text,
        "training_use": use,
        "track_x": None if track_x is None else round(float(track_x), 3),
        "track_y": None if track_y is None else round(float(track_y), 3),
        "model_confidence": None if model_confidence is None else round(float(model_confidence), 6),
        "detector_confidence": None if model_confidence is None else round(float(model_confidence), 6),
        "center_error_px": None if center_error_px is None else round(float(center_error_px), 3),
        "threshold_px": None if threshold_px is None else round(float(threshold_px), 3),
        "selection_reason": selection_reason,
        "selection_reasons": reasons,
        "suggested_detector_label": suggested_label,
        "review_required": True,
        "review_evidence": evidence,
        "review_decision_schema": {
            "detector_status": ["footbag", "not_footbag", "corrected", "not_visible", "skip"],
            "required_for_corrected": ["corrected_x", "corrected_y"],
            "optional": ["radius", "evidence", "reviewer"],
        },
        "_video_path": str(video_path),
        "_digest": digest,
    }
    record["priority_score"] = candidate_priority(record)
    return ErrorCandidate(record, video_path)


def collect_track_error_candidates(
    *,
    track_metrics: Path,
    qa_manifest: Path,
    video_root: Path | None = None,
    include_audit_only: bool = True,
) -> list[ErrorCandidate]:
    videos = qa_video_paths(qa_manifest, video_root)
    doc = read_json(track_metrics)
    candidates: list[ErrorCandidate] = []
    for row in doc.get("rows", []):
        kind = str(row.get("kind") or "")
        result = str(row.get("result") or "")
        hard_negative_result = str(row.get("hard_negative_result") or "")
        if kind == "positive" and result not in {"fail", "missing_track_point"}:
            continue
        if kind == "hard_negative" and hard_negative_result != "false_positive_near_bad_point":
            continue
        split = str(row.get("split") or "train")
        if split == "test" and not include_audit_only:
            continue
        source_video = str(row.get("video") or "")
        video_path = videos.get(source_video) or videos.get(Path(source_video).stem) or videos.get(safe_stem(source_video))
        if video_path is None:
            continue
        time_sec = numeric(row.get("time_sec"))
        if time_sec is None:
            continue
        expected_x = numeric(row.get("expected_x"))
        expected_y = numeric(row.get("expected_y"))
        if expected_x is None or expected_y is None:
            continue
        track_x = numeric(row.get("track_x"))
        track_y = numeric(row.get("track_y"))
        if kind == "hard_negative":
            x = track_x if track_x is not None else expected_x
            y = track_y if track_y is not None else expected_y
            suggested = "not_footbag"
            reason = "hard_negative_false_positive"
            event_type = "model_false_positive_on_hard_negative"
            evidence = "v10 detector track fired near a reviewed hard-negative point"
        else:
            x = expected_x
            y = expected_y
            suggested = "verify_or_correct"
            reason = "missing_track_point" if result == "missing_track_point" else "track_center_fail"
            event_type = "detector_track_positive_failure"
            evidence = f"v10 detector track result={result}"
        candidates.append(
            make_candidate(
                source_video=source_video,
                video_path=video_path,
                item_id=str(row.get("item_id") or f"{kind}-{time_sec:.3f}"),
                split=split,
                time_sec=float(time_sec),
                x=float(x),
                y=float(y),
                radius=float(numeric(row.get("radius")) or 22.0),
                event_type=event_type,
                selection_reason=reason,
                suggested_label=suggested,
                source="v10_detector_track_evaluation",
                track_x=track_x,
                track_y=track_y,
                model_confidence=numeric(row.get("confidence")),
                center_error_px=numeric(row.get("center_error_px")),
                threshold_px=numeric(row.get("threshold_px")),
                evidence=evidence,
            )
        )
    return candidates


def collect_model_error_candidates(
    *,
    model_metrics: Path | None,
    dataset: Path | None,
    qa_manifest: Path,
    video_root: Path | None = None,
    low_confidence_per_split: int = 0,
    include_audit_only: bool = True,
) -> list[ErrorCandidate]:
    if model_metrics is None or not model_metrics.exists() or dataset is None:
        return []
    videos = qa_video_paths(qa_manifest, video_root)
    labels_by_path = dataset_labels(dataset)
    doc = read_json(model_metrics)
    rows = doc.get("rows", [])
    selected_rows: list[dict[str, Any]] = [row for row in rows if str(row.get("result") or "") == "fail_center"]
    if low_confidence_per_split > 0:
        low_by_split: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            if str(row.get("result") or "") != "low_confidence_near_label":
                continue
            low_by_split.setdefault(str(row.get("split") or "train"), []).append(row)
        for split_rows in low_by_split.values():
            split_rows.sort(key=lambda item: float(item.get("confidence") or 1.0))
            selected_rows.extend(split_rows[:low_confidence_per_split])

    candidates: list[ErrorCandidate] = []
    seen: set[tuple[str, str]] = set()
    for row in selected_rows:
        split = str(row.get("split") or "train")
        if split == "test" and not include_audit_only:
            continue
        label_row = labels_by_path.get(str(row.get("image") or "")) or labels_by_path.get(str(row.get("label") or ""))
        if not label_row:
            continue
        source_video = str(label_row.get("source_video") or "")
        item_id = str(label_row.get("item_id") or Path(str(row.get("image") or "")).stem)
        key = (source_video, item_id)
        if key in seen:
            continue
        seen.add(key)
        video_path = videos.get(source_video) or videos.get(Path(source_video).stem) or videos.get(safe_stem(source_video))
        time_sec = numeric(label_row.get("time_sec"))
        x = numeric(row.get("expected_x")) or numeric(label_row.get("x"))
        y = numeric(row.get("expected_y")) or numeric(label_row.get("y"))
        if video_path is None or time_sec is None or x is None or y is None:
            continue
        result = str(row.get("result") or "")
        reason = "model_center_fail" if result == "fail_center" else "low_confidence_near_label"
        candidates.append(
            make_candidate(
                source_video=source_video,
                video_path=video_path,
                item_id=item_id,
                split=split,
                time_sec=float(time_sec),
                x=float(x),
                y=float(y),
                radius=float(numeric(label_row.get("radius")) or 22.0),
                event_type=f"detector_model_{reason}",
                selection_reason=reason,
                suggested_label="verify_or_correct",
                source="v10_detector_model_evaluation",
                track_x=numeric(row.get("prediction_x")),
                track_y=numeric(row.get("prediction_y")),
                model_confidence=numeric(row.get("confidence")),
                center_error_px=numeric(row.get("center_error_px")),
                threshold_px=numeric(row.get("threshold_px")),
                evidence=f"v10 detector model result={result}",
            )
        )
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


def crop_bounds(record: dict[str, Any], crop_size: int, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    points = [(float(record["x"]), float(record["y"]))]
    track_x = numeric(record.get("track_x"))
    track_y = numeric(record.get("track_y"))
    if track_x is not None and track_y is not None:
        points.append((track_x, track_y))
    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    max_span = max(max(abs(point[0] - center_x), abs(point[1] - center_y)) for point in points)
    side = max(crop_size, int(math.ceil(max_span * 2.0 + 80.0)))
    side = min(max(frame_width, frame_height), side)
    half = side // 2
    left = int(round(center_x)) - half
    top = int(round(center_y)) - half
    right = left + side
    bottom = top + side
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > frame_width:
        left = max(0, left - (right - frame_width))
        right = frame_width
    if bottom > frame_height:
        top = max(0, top - (bottom - frame_height))
        bottom = frame_height
    return left, top, right, bottom


def render_tile(frame: np.ndarray, record: dict[str, Any], index: int, crop_size: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = frame.shape[:2]
    left, top, right, bottom = crop_bounds(record, crop_size, width, height)
    crop = frame[top:bottom, left:right].copy()
    if crop.size == 0:
        crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        left = top = 0
    display = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    scale_x = crop_size / max(1, right - left)
    scale_y = crop_size / max(1, bottom - top)

    def project(x: float, y: float) -> tuple[int, int]:
        return int(round((x - left) * scale_x)), int(round((y - top) * scale_y))

    expected = project(float(record["x"]), float(record["y"]))
    radius = max(5, int(round(float(record["radius"]) * max(scale_x, scale_y) * 0.8)))
    cv2.circle(display, expected, radius, (0, 220, 0), 2)
    cv2.drawMarker(display, expected, (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)

    track_x = numeric(record.get("track_x"))
    track_y = numeric(record.get("track_y"))
    if track_x is not None and track_y is not None:
        track = project(track_x, track_y)
        cv2.drawMarker(display, track, (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=2)
        cv2.line(display, expected, track, (255, 255, 255), 1)

    footer_h = 104
    footer = np.zeros((footer_h, crop_size, 3), dtype=np.uint8)
    lines = [
        f"{index:03d} {str(record['source_video'])[:24]}",
        f"{record['selection_reason']} {record['time_sec']:.2f}s {record.get('split')} {record.get('training_use')}",
        f"suggest {record.get('suggested_detector_label')} err={record.get('center_error_px')} conf={record.get('model_confidence')}",
        "green/yellow=label red=model",
    ]
    for line_index, text in enumerate(lines):
        cv2.putText(footer, text[:46], (6, 18 + 22 * line_index), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
    return display, np.vstack([display, footer])


def render_outputs(candidates: list[ErrorCandidate], out_dir: Path, *, crop_size: int, cols: int) -> tuple[list[dict[str, Any]], Path | None]:
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
        crop, tile = render_tile(frame, record, index, crop_size)
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
    sheet_path = out_dir / "detector_error_review_sheet.jpg"
    cv2.imwrite(str(sheet_path), sheet)
    return records, sheet_path


def dedupe_candidates(candidates: list[ErrorCandidate]) -> list[ErrorCandidate]:
    best_by_key: dict[tuple[str, str, str], ErrorCandidate] = {}
    for candidate in candidates:
        record = candidate.record
        key = (
            str(record.get("source_video") or ""),
            str(record.get("event_item_id") or ""),
            str(record.get("selection_reason") or ""),
        )
        existing = best_by_key.get(key)
        if existing is None or float(record.get("priority_score") or 0.0) > float(existing.record.get("priority_score") or 0.0):
            best_by_key[key] = candidate
    return sorted(best_by_key.values(), key=lambda item: (-float(item.record.get("priority_score") or 0.0), item.record["source_video"], float(item.record["time_sec"])))


def select_candidates(candidates: list[ErrorCandidate], *, max_items: int, per_video: int) -> list[ErrorCandidate]:
    selected: list[ErrorCandidate] = []
    per_video_counts: dict[str, int] = {}
    for candidate in candidates:
        video = str(candidate.record["source_video"])
        if per_video_counts.get(video, 0) >= per_video:
            continue
        selected.append(candidate)
        per_video_counts[video] = per_video_counts.get(video, 0) + 1
        if len(selected) >= max_items:
            break
    return selected


def build_error_review_batch(
    *,
    qa_manifest: Path,
    track_metrics: Path,
    out_dir: Path,
    dataset: Path | None = None,
    model_metrics: Path | None = None,
    video_root: Path | None = None,
    max_items: int = 120,
    per_video: int = 12,
    crop_size: int = 224,
    cols: int = 4,
    low_confidence_per_split: int = 8,
    include_audit_only: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    candidates = collect_track_error_candidates(
        track_metrics=track_metrics,
        qa_manifest=qa_manifest,
        video_root=video_root,
        include_audit_only=include_audit_only,
    )
    candidates.extend(
        collect_model_error_candidates(
            model_metrics=model_metrics,
            dataset=dataset,
            qa_manifest=qa_manifest,
            video_root=video_root,
            low_confidence_per_split=low_confidence_per_split,
            include_audit_only=include_audit_only,
        )
    )
    candidates = dedupe_candidates(candidates)
    selected = select_candidates(candidates, max_items=max_items, per_video=per_video)
    selected_records = [{k: v for k, v in item.record.items() if not k.startswith("_")} for item in selected]
    sheet_path: Path | None = None
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        selected_records, sheet_path = render_outputs(selected, out_dir, crop_size=crop_size, cols=cols)
        jsonl_path = out_dir / "detector_error_review_items.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for record in selected_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        write_json(
            out_dir / "detector_error_decisions_template.json",
            {
                "schema_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "instructions": "Review v10 detector errors. Use footbag/corrected only when the marked green/yellow label center is a real footbag. Use not_footbag for the marked center if it is clearly not the bag. Keep held-out audit-only rows as skip unless intentionally creating a new split.",
                "decisions": [
                    {
                        "detector_label_id": record["detector_label_id"],
                        "source_video": record["source_video"],
                        "detector_status": "pending",
                        "corrected_x": None,
                        "corrected_y": None,
                        "radius": record.get("radius"),
                        "training_use": record.get("training_use"),
                        "suggested_detector_label": record.get("suggested_detector_label"),
                        "evidence": "",
                    }
                    for record in selected_records
                ],
            },
        )

    counts_by_reason: dict[str, int] = {}
    counts_by_split: dict[str, int] = {}
    counts_by_training_use: dict[str, int] = {}
    for record in selected_records:
        counts_by_reason[str(record.get("selection_reason") or "")] = counts_by_reason.get(str(record.get("selection_reason") or ""), 0) + 1
        counts_by_split[str(record.get("split") or "")] = counts_by_split.get(str(record.get("split") or ""), 0) + 1
        counts_by_training_use[str(record.get("training_use") or "")] = counts_by_training_use.get(str(record.get("training_use") or ""), 0) + 1
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qa_manifest": portable(qa_manifest),
        "track_metrics": portable(track_metrics),
        "model_metrics": portable(model_metrics),
        "dataset": portable(dataset),
        "out_dir": portable(out_dir),
        "total_candidates": len(candidates),
        "selected_items": len(selected_records),
        "per_video_limit": per_video,
        "max_items": max_items,
        "low_confidence_per_split": low_confidence_per_split,
        "include_audit_only": include_audit_only,
        "counts_by_reason": counts_by_reason,
        "counts_by_split": counts_by_split,
        "counts_by_training_use": counts_by_training_use,
        "contact_sheet_path": portable(sheet_path),
        "jsonl_path": portable(out_dir / "detector_error_review_items.jsonl") if not dry_run else None,
        "decision_template_path": portable(out_dir / "detector_error_decisions_template.json") if not dry_run else None,
        "items": selected_records,
    }
    if not dry_run:
        write_json(out_dir / "detector_error_review_manifest.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a detector error-review batch from v10 track/model failures")
    parser.add_argument("--qa-manifest", type=Path, required=True)
    parser.add_argument("--track-metrics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--model-metrics", type=Path)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--max-items", type=int, default=120)
    parser.add_argument("--per-video", type=int, default=12)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--low-confidence-per-split", type=int, default=8)
    parser.add_argument("--exclude-audit-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_error_review_batch(
        qa_manifest=args.qa_manifest,
        track_metrics=args.track_metrics,
        out_dir=args.out_dir,
        dataset=args.dataset,
        model_metrics=args.model_metrics,
        video_root=args.video_root,
        max_items=args.max_items,
        per_video=args.per_video,
        crop_size=args.crop_size,
        cols=args.cols,
        low_confidence_per_split=args.low_confidence_per_split,
        include_audit_only=not args.exclude_audit_only,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(f"manifest: {args.out_dir / 'detector_error_review_manifest.json'}")
        print(f"sheet: {summary.get('contact_sheet_path')}")
    print(
        json.dumps(
            {
                "total_candidates": summary["total_candidates"],
                "selected_items": summary["selected_items"],
                "counts_by_reason": summary["counts_by_reason"],
                "counts_by_split": summary["counts_by_split"],
                "counts_by_training_use": summary["counts_by_training_use"],
                "contact_sheet_path": summary["contact_sheet_path"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
