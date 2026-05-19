#!/usr/bin/env python3
"""Render before/during/after frame strips for review batch items."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from qa_rally_enrichment import OUT_SIZE, read_resized_frame


ROOT = Path(__file__).resolve().parent
DEFAULT_BATCH = ROOT / "outputs" / "review_batches" / "latest_review_batch.json"
DEFAULT_SUGGESTIONS = ROOT / "outputs" / "review_batches" / "assisted_review_suggestions.json"
DEFAULT_OUT_DIR = ROOT / "outputs" / "review_batches" / "sequence_sheets"
DEFAULT_REVIEWS_DIR = ROOT / "reviews"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def review_status_lookup(reviews_dir: Path) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for path in sorted(reviews_dir.glob("*.review.json")):
        try:
            doc = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        source_video = str(doc.get("source_video") or "")
        for item in doc.get("items", []):
            item_id = item.get("id")
            if not source_video or not item_id:
                continue
            lookup[(source_video, str(item_id))] = str(item.get("status") or "pending")
    return lookup


def mark_frame(frame: np.ndarray, item: dict[str, Any], label: str) -> np.ndarray:
    out = frame.copy()
    if item.get("qa_ball_x") is not None and item.get("qa_ball_y") is not None:
        x = int(round(float(item["qa_ball_x"])))
        y = int(round(float(item["qa_ball_y"])))
        r = max(18, int(round(float(item.get("qa_ball_radius") or 18) + 8)))
        cv2.circle(out, (x, y), r, (0, 230, 255), 3, cv2.LINE_AA)
        cv2.circle(out, (x, y), 4, (0, 230, 255), -1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (OUT_SIZE[0], 32), (15, 15, 15), -1)
    cv2.putText(out, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.resize(out, (220, 292), interpolation=cv2.INTER_AREA)


def item_strip(item: dict[str, Any], suggestion: dict[str, Any], offsets: list[float]) -> np.ndarray | None:
    video = rel_path(item["video_path"])
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    base_t = float(item.get("frame_time_sec") or item["time_sec"])
    frames: list[np.ndarray] = []
    for offset in offsets:
        t = max(0.0, base_t + offset)
        frame = read_resized_frame(cap, fps, t)
        if frame is None:
            continue
        label = f"{Path(item['source_video']).stem} {item['kind']} {float(item['time_sec']):.2f}s {offset:+.2f}"
        frames.append(mark_frame(frame, item, label))
    cap.release()
    if not frames:
        return None
    strip = np.concatenate(frames, axis=1)
    footer_h = 74
    canvas = np.full((strip.shape[0] + footer_h, strip.shape[1], 3), 18, dtype=np.uint8)
    canvas[: strip.shape[0], : strip.shape[1]] = strip
    lines = [
        f"{item['review_stem']} / {item['review_item_id']}",
        f"{item['kind']} status={item.get('review_status', 'unknown')} bucket={suggestion.get('bucket')} score={suggestion.get('evidence_score')} side={item.get('contact_side')} type={item.get('contact_type')}",
        " ".join(str(tag) for tag in item.get("review_tags", [])[:8]),
    ]
    for idx, line in enumerate(lines):
        cv2.putText(canvas, line[:150], (8, strip.shape[0] + 18 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Render review sequence sheets")
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--suggestions", type=Path, default=DEFAULT_SUGGESTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--reviews-dir", type=Path, default=DEFAULT_REVIEWS_DIR)
    parser.add_argument("--kind", action="append", help="Filter kind; may be repeated")
    parser.add_argument("--bucket", action="append", help="Filter assisted bucket; may be repeated")
    parser.add_argument("--status", action="append", help="Filter current review status; may be repeated")
    parser.add_argument("--seconds", type=float, default=0.45)
    args = parser.parse_args()

    batch = read_json(args.batch)
    suggestions_doc = read_json(args.suggestions)
    suggestions = {item["batch_item_id"]: item for item in suggestions_doc["items"]}
    statuses = review_status_lookup(args.reviews_dir)
    kinds = set(args.kind or [])
    buckets = set(args.bucket or [])
    status_filter = set(args.status or [])
    offsets = [-args.seconds, -args.seconds / 2, 0.0, args.seconds / 2, args.seconds]

    strips: list[np.ndarray] = []
    for item in batch["items"]:
        suggestion = suggestions.get(item["batch_item_id"], {})
        if kinds and item["kind"] not in kinds:
            continue
        if buckets and suggestion.get("bucket") not in buckets:
            continue
        review_status = statuses.get((str(item.get("source_video") or ""), str(item.get("review_item_id") or "")), "unreviewed")
        if status_filter and review_status not in status_filter:
            continue
        item_with_status = dict(item)
        item_with_status["review_status"] = review_status
        strip = item_strip(item_with_status, suggestion, offsets)
        if strip is not None:
            strips.append(strip)

    if not strips:
        return
    strip_h = max(strip.shape[0] for strip in strips)
    strip_w = max(strip.shape[1] for strip in strips)
    rows = len(strips)
    sheet = np.full((rows * strip_h, strip_w, 3), 16, dtype=np.uint8)
    for idx, strip in enumerate(strips):
        y = idx * strip_h
        sheet[y : y + strip.shape[0], : strip.shape[1]] = strip

    name_parts = [*(args.kind or ["all"]), *(args.bucket or []), *(args.status or [])]
    out_name = "__".join(part.replace("/", "-") for part in name_parts) + "_sequence_sheet.jpg"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / out_name
    cv2.imwrite(str(out_path), sheet)
    print(f"sequence sheet: {out_path} ({len(strips)} items)")


if __name__ == "__main__":
    main()
