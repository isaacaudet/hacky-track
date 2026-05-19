#!/usr/bin/env python3
"""Mine detector false-positive candidates for hard-negative review."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_detector_label_review_batch import Candidate, render_outputs, resolve_video_path, safe_stem, write_json


ROOT = Path(__file__).resolve().parent
OUT_SIZE = (688, 912)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def portable(path: Path | None, base: Path = ROOT) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except (OSError, ValueError):
        return path.name if path.is_absolute() else str(path)


def numeric(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if math.isfinite(value_f) else None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_splits(dataset_manifest: Path | None) -> dict[str, str]:
    if dataset_manifest is None or not dataset_manifest.exists():
        return {}
    doc = read_json(dataset_manifest)
    return {str(video): str(split) for video, split in (doc.get("splits") or {}).items()}


def qa_runs_by_video(qa_manifest: Path, video_root: Path | None = None) -> dict[str, dict[str, Any]]:
    manifest = read_json(qa_manifest)
    runs: dict[str, dict[str, Any]] = {}
    for run in manifest.get("runs", []):
        raw_video = run.get("video") or run.get("source_video")
        if not raw_video:
            continue
        source_video = Path(str(raw_video)).name
        qa_path = Path(str(run.get("qa_events_path") or ""))
        if not qa_path.exists():
            qa_path = ROOT / qa_path
        video_path = resolve_video_path(raw_video, video_root)
        runs[source_video] = {
            "source_video": source_video,
            "qa_events_path": qa_path,
            "video_path": video_path,
        }
        runs[Path(source_video).stem] = runs[source_video]
        runs[safe_stem(source_video)] = runs[source_video]
    return runs


def qa_ball_points(qa_events_path: Path) -> list[dict[str, float]]:
    if not qa_events_path.exists():
        return []
    doc = read_json(qa_events_path)
    points: list[dict[str, float]] = []
    for event in doc.get("events", []):
        x = numeric(event.get("qa_ball_x"))
        y = numeric(event.get("qa_ball_y"))
        time_sec = numeric(event.get("qa_frame_time_sec")) or numeric(event.get("time_sec"))
        if x is None or y is None or time_sec is None:
            continue
        points.append({"x": x, "y": y, "time_sec": time_sec})
    return points


def nearest_qa_distance(x: float, y: float, time_sec: float, points: list[dict[str, float]], max_time_delta_sec: float) -> tuple[float | None, float | None]:
    best_distance: float | None = None
    best_delta: float | None = None
    for point in points:
        delta = abs(point["time_sec"] - time_sec)
        if delta > max_time_delta_sec:
            continue
        distance = math.hypot(point["x"] - x, point["y"] - y)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_delta = delta
    return best_distance, best_delta


def convert_detection_center(detection: dict[str, Any], video_info: dict[str, Any]) -> tuple[float, float, float]:
    width = float(video_info.get("width") or OUT_SIZE[0])
    height = float(video_info.get("height") or OUT_SIZE[1])
    scale_x = OUT_SIZE[0] / width
    scale_y = OUT_SIZE[1] / height
    center = detection.get("center") or [0.0, 0.0]
    bbox = detection.get("bbox") or [center[0] - 12.0, center[1] - 12.0, center[0] + 12.0, center[1] + 12.0]
    box_w = max(1.0, (float(bbox[2]) - float(bbox[0])) * scale_x)
    box_h = max(1.0, (float(bbox[3]) - float(bbox[1])) * scale_y)
    return float(center[0]) * scale_x, float(center[1]) * scale_y, max(8.0, min(48.0, max(box_w, box_h) / 2.0))


def inference_runs(inference_root: Path | None, batch_manifest: Path | None) -> list[dict[str, Any]]:
    if batch_manifest is not None:
        doc = read_json(batch_manifest)
        base = batch_manifest.parent
        runs: list[dict[str, Any]] = []
        for run in doc.get("runs", []):
            manifest_path = Path(str(run.get("manifest") or ""))
            if not manifest_path.exists():
                manifest_path = base.parent / manifest_path
            if not manifest_path.exists():
                manifest_path = ROOT / manifest_path
            runs.append({"manifest_path": manifest_path})
        return runs
    if inference_root is None:
        return []
    return [{"manifest_path": path} for path in sorted(inference_root.glob("*/detector_inference_manifest.json"))]


def collect_false_positive_candidates(
    *,
    qa_manifest: Path,
    inference_root: Path | None = None,
    batch_manifest: Path | None = None,
    dataset_manifest: Path | None = None,
    video_root: Path | None = None,
    min_confidence: float = 0.001,
    max_confidence: float = 0.08,
    time_exclusion_sec: float = 0.18,
    center_exclusion_px: float = 72.0,
    frame_bucket: int = 8,
    grid_px: int = 48,
) -> list[Candidate]:
    qa_runs = qa_runs_by_video(qa_manifest, video_root)
    splits = load_splits(dataset_manifest)
    candidates_by_bucket: dict[tuple[str, int, int, int], Candidate] = {}
    qa_cache: dict[Path, list[dict[str, float]]] = {}

    for run in inference_runs(inference_root, batch_manifest):
        manifest_path = Path(run["manifest_path"])
        if not manifest_path.exists():
            continue
        inference_doc = read_json(manifest_path)
        source_video = Path(str(inference_doc.get("video") or manifest_path.parent.name)).name
        qa_run = qa_runs.get(source_video) or qa_runs.get(Path(source_video).stem) or qa_runs.get(safe_stem(source_video))
        if qa_run is None:
            continue
        qa_path = Path(qa_run["qa_events_path"])
        if qa_path not in qa_cache:
            qa_cache[qa_path] = qa_ball_points(qa_path)
        raw_path = manifest_path.parent / "raw_model_detections.jsonl"
        video_info = inference_doc.get("video_info") or {}
        split = splits.get(source_video)
        for detection in load_jsonl(raw_path):
            confidence = float(detection.get("confidence") or 0.0)
            if confidence < min_confidence or confidence > max_confidence:
                continue
            time_sec = numeric(detection.get("time_sec"))
            frame_index = int(detection.get("frame_index") or 0)
            if time_sec is None:
                fps = numeric(video_info.get("fps")) or 30.0
                time_sec = frame_index / fps
            x, y, radius = convert_detection_center(detection, video_info)
            nearest_distance, nearest_delta = nearest_qa_distance(x, y, time_sec, qa_cache[qa_path], time_exclusion_sec)
            if nearest_distance is not None and nearest_distance <= center_exclusion_px:
                continue
            bucket = (source_video, frame_index // max(1, frame_bucket), int(x // grid_px), int(y // grid_px))
            digest = hashlib.sha1(f"{source_video}:{frame_index}:{x:.2f}:{y:.2f}:{confidence:.5f}".encode("utf-8")).hexdigest()[:10]
            record = {
                "batch_item_id": f"{safe_stem(source_video)}__detfp-{frame_index:07d}-{digest}",
                "detector_label_id": f"detfp-{safe_stem(source_video)}-{frame_index:07d}-{digest}",
                "event_item_id": None,
                "review_stem": safe_stem(Path(source_video).stem),
                "source": "model_false_positive_candidate",
                "source_video": source_video,
                "video_id": safe_stem(source_video),
                "video": source_video,
                "video_path": portable(Path(qa_run["video_path"])),
                "qa_events_path": portable(qa_path),
                "event_type": "model_false_positive",
                "time_sec": round(float(time_sec), 6),
                "time_s": round(float(time_sec), 6),
                "event_time_sec": None,
                "frame_index": frame_index,
                "x": round(x, 3),
                "y": round(y, 3),
                "center_x": round(x, 3),
                "center_y": round(y, 3),
                "radius": round(radius, 3),
                "model_confidence": round(confidence, 6),
                "detector_confidence": round(confidence, 6),
                "model_bbox": detection.get("bbox"),
                "nearest_qa_distance_px": None if nearest_distance is None else round(nearest_distance, 3),
                "nearest_qa_time_delta_sec": None if nearest_delta is None else round(nearest_delta, 6),
                "split": split,
                "priority_score": round(confidence + min(1.0, (nearest_distance or 240.0) / 240.0), 6),
                "selection_reasons": ["model_detection_far_from_qa_ball", "object_label_review_candidate", "needs_detector_label_review"],
                "selection_reason": "model_detection_far_from_qa_ball",
                "suggested_detector_label": "verify_or_correct",
                "review_required": True,
                "review_decision_schema": {
                    "detector_status": ["not_footbag", "footbag", "corrected", "not_visible", "skip"],
                    "required_for_corrected": ["corrected_x", "corrected_y"],
                    "optional": ["radius", "evidence", "reviewer"],
                },
                "_video_path": str(Path(qa_run["video_path"])),
                "_digest": digest,
            }
            candidate = Candidate(record, Path(qa_run["video_path"]))
            existing = candidates_by_bucket.get(bucket)
            if existing is None or float(record["priority_score"]) > float(existing.record["priority_score"]):
                candidates_by_bucket[bucket] = candidate
    return sorted(candidates_by_bucket.values(), key=lambda item: (-float(item.record["priority_score"]), item.record["source_video"], item.record["frame_index"]))


def build_false_positive_review_batch(
    qa_manifest: Path,
    out_dir: Path,
    *,
    inference_root: Path | None = None,
    batch_manifest: Path | None = None,
    dataset_manifest: Path | None = None,
    video_root: Path | None = None,
    max_items: int = 120,
    per_video: int = 24,
    crop_size: int = 192,
    cols: int = 4,
    min_confidence: float = 0.001,
    max_confidence: float = 0.08,
    dry_run: bool = False,
) -> dict[str, Any]:
    candidates = collect_false_positive_candidates(
        qa_manifest=qa_manifest,
        inference_root=inference_root,
        batch_manifest=batch_manifest,
        dataset_manifest=dataset_manifest,
        video_root=video_root,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
    )
    selected: list[Candidate] = []
    per_video_counts: dict[str, int] = defaultdict(int)
    for candidate in candidates:
        video = str(candidate.record["source_video"])
        if per_video_counts[video] >= per_video:
            continue
        selected.append(candidate)
        per_video_counts[video] += 1
        if len(selected) >= max_items:
            break

    selected_records = [{k: v for k, v in item.record.items() if not k.startswith("_")} for item in selected]
    sheet_path: Path | None = None
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        selected_records, sheet_path = render_outputs(selected, out_dir, crop_size=crop_size, cols=cols)
        jsonl_path = out_dir / "detector_false_positive_review_items.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for record in selected_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        write_json(
            out_dir / "detector_false_positive_decisions_template.json",
            {
                "schema_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "instructions": "Review model false-positive candidates. Use not_footbag only when the marked center is clearly not the footbag. Use footbag/corrected when the model found a real bag.",
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
        "inference_root": portable(inference_root),
        "batch_manifest": portable(batch_manifest),
        "dataset_manifest": portable(dataset_manifest),
        "out_dir": portable(out_dir),
        "total_candidates": len(candidates),
        "selected_items": len(selected_records),
        "per_video_limit": per_video,
        "max_items": max_items,
        "min_confidence": min_confidence,
        "max_confidence": max_confidence,
        "contact_sheet_path": portable(sheet_path),
        "jsonl_path": portable(out_dir / "detector_false_positive_review_items.jsonl") if not dry_run else None,
        "decision_template_path": portable(out_dir / "detector_false_positive_decisions_template.json") if not dry_run else None,
        "items": selected_records,
    }
    if not dry_run:
        write_json(out_dir / "detector_false_positive_review_manifest.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build detector false-positive hard-negative review batch")
    parser.add_argument("--qa-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path)
    parser.add_argument("--batch-manifest", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--max-items", type=int, default=120)
    parser.add_argument("--per-video", type=int, default=24)
    parser.add_argument("--crop-size", type=int, default=192)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--min-confidence", type=float, default=0.001)
    parser.add_argument("--max-confidence", type=float, default=0.08)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.inference_root and not args.batch_manifest:
        raise SystemExit("Provide --inference-root or --batch-manifest")
    summary = build_false_positive_review_batch(
        args.qa_manifest,
        args.out_dir,
        inference_root=args.inference_root,
        batch_manifest=args.batch_manifest,
        dataset_manifest=args.dataset_manifest,
        video_root=args.video_root,
        max_items=args.max_items,
        per_video=args.per_video,
        crop_size=args.crop_size,
        cols=args.cols,
        min_confidence=args.min_confidence,
        max_confidence=args.max_confidence,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(f"manifest: {args.out_dir / 'detector_false_positive_review_manifest.json'}")
        print(f"sheet: {summary.get('contact_sheet_path')}")
    print(json.dumps({k: summary[k] for k in ["total_candidates", "selected_items", "contact_sheet_path"]}, indent=2))


if __name__ == "__main__":
    main()
