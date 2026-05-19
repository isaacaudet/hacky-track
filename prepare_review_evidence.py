#!/usr/bin/env python3
"""Build grouped visual evidence sheets for release review batches."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from qa_rally_enrichment import OUT_SIZE, read_resized_frame


ROOT = Path(__file__).resolve().parent
DEFAULT_BATCH = ROOT / "outputs" / "review_batches" / "latest_review_batch.json"
DEFAULT_OUT_DIR = ROOT / "outputs" / "review_batches" / "evidence_pack"
DEFAULT_REVIEWS_DIR = ROOT / "reviews"

SHEET_FRAME_SIZE = (172, 228)
STRIP_FOOTER_H = 92
BUCKETS = (
    "gap_floor_reset",
    "ball_accuracy",
    "drop_review",
    "side_contact",
    "stall_trick",
    "missed_touch",
    "all_priority",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except (OSError, ValueError):
        return str(path)


def resolve_path(raw: Any) -> Path:
    path = Path(str(raw or ""))
    if path.exists():
        return path
    root_path = ROOT / path
    if root_path.exists():
        return root_path
    return path


def maybe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def review_status_lookup(reviews_dir: Path) -> dict[tuple[str, str], str]:
    statuses: dict[tuple[str, str], str] = {}
    if not reviews_dir.exists():
        return statuses
    for path in sorted(reviews_dir.glob("*.review.json")):
        try:
            doc = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        source_video = str(doc.get("source_video") or "")
        for item in doc.get("items", []):
            item_id = str(item.get("id") or "")
            if source_video and item_id:
                statuses[(source_video, item_id)] = str(item.get("status") or "pending")
    return statuses


def load_suggestions(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path or not path.exists():
        return {}
    doc = read_json(path)
    return {str(item.get("batch_item_id")): item for item in doc.get("items", []) if item.get("batch_item_id")}


def item_tags(item: dict[str, Any]) -> set[str]:
    return {str(tag) for tag in item.get("review_tags", [])}


def item_buckets(item: dict[str, Any]) -> list[str]:
    tags = item_tags(item)
    kind = str(item.get("kind") or "")
    buckets = ["all_priority"]
    if {"gap_without_floor_reset", "likely_missing_floor_reset", "strict_best_rally_blocker"} & tags:
        buckets.append("gap_floor_reset")
    if {"huge_ball_correction", "large_ball_correction", "ball_low", "ball_fallback", "floor_risk_touch"} & tags:
        buckets.append("ball_accuracy")
    if kind == "drop_floor" or {"likely_missed_drop_floor", "hidden_drop"} & tags:
        buckets.append("drop_review")
    if {"low_side_confidence", "side_unknown", "side_needs_review", "ambiguous_contact"} & tags:
        buckets.append("side_contact")
    if kind in {"stall", "around_the_world"} or {"trick_candidate", "likely_missed_stall"} & tags:
        buckets.append("stall_trick")
    if {"likely_missed_touch", "suppressed_low_confidence_touch_candidate", "suppressed_audio_track_huge_correction_touch"} & tags:
        buckets.append("missed_touch")
    return buckets


def priority_score(item: dict[str, Any], suggestion: dict[str, Any] | None, status: str) -> float:
    tags = item_tags(item)
    score = float(item.get("priority_score") or 0.0)
    if status in {"unreviewed", "pending"}:
        score += 1.5
    if "strict_best_rally_blocker" in tags:
        score += 9.0
    if "gap_without_floor_reset" in tags:
        score += 7.0
    if "likely_missed_drop_floor" in tags:
        score += 5.0
    if "likely_missed_touch" in tags:
        score += 4.5
    if "huge_ball_correction" in tags:
        score += 4.0
    if "large_ball_correction" in tags:
        score += 2.2
    if "ball_low" in tags or "ball_fallback" in tags:
        score += 3.0
    if "side_unknown" in tags:
        score += 2.2
    if "low_side_confidence" in tags or "side_needs_review" in tags:
        score += 1.6
    if "ambiguous_contact" in tags:
        score += 2.0
    if str(item.get("kind") or "") in {"stall", "around_the_world"}:
        score += 2.0
    if suggestion:
        score += float(suggestion.get("evidence_score") or 0.0)
    return round(score, 3)


def sequence_times(item: dict[str, Any], seconds: float) -> list[float]:
    start = maybe_float(item.get("start_sec"))
    end = maybe_float(item.get("end_sec"))
    gap_start = maybe_float(item.get("gap_start_sec"))
    gap_duration = maybe_float(item.get("gap_duration_sec"))
    base = maybe_float(item.get("frame_time_sec")) or maybe_float(item.get("time_sec")) or 0.0
    if gap_start is not None and gap_duration is not None and gap_duration > 0:
        return [
            max(0.0, gap_start - 0.20),
            max(0.0, gap_start + gap_duration * 0.25),
            max(0.0, base),
            max(0.0, gap_start + gap_duration * 0.75),
            max(0.0, gap_start + gap_duration - 0.20),
        ]
    if start is not None and end is not None and end > start:
        return [
            max(0.0, start - 0.18),
            max(0.0, start),
            max(0.0, (start + end) / 2.0),
            max(0.0, end),
            max(0.0, end + 0.18),
        ]
    return [max(0.0, base + offset) for offset in (-seconds, -seconds / 2.0, 0.0, seconds / 2.0, seconds)]


def draw_event_overlay(frame: np.ndarray, item: dict[str, Any], label: str) -> np.ndarray:
    out = frame.copy()
    cv2.line(out, (OUT_SIZE[0] // 2, 0), (OUT_SIZE[0] // 2, OUT_SIZE[1]), (245, 245, 245), 1, cv2.LINE_AA)
    bx = maybe_float(item.get("qa_ball_x"))
    by = maybe_float(item.get("qa_ball_y"))
    br = maybe_float(item.get("qa_ball_radius")) or 18.0
    if bx is not None and by is not None:
        radius = max(14, int(round(br + 9)))
        cv2.circle(out, (int(round(bx)), int(round(by))), radius, (0, 235, 255), 3, cv2.LINE_AA)
        cv2.circle(out, (int(round(bx)), int(round(by))), 4, (0, 235, 255), -1, cv2.LINE_AA)
    fx = maybe_float(item.get("foot_x"))
    fy = maybe_float(item.get("foot_y"))
    if fx is not None and fy is not None:
        cv2.circle(out, (int(round(fx)), int(round(fy))), 16, (255, 0, 255), 3, cv2.LINE_AA)
    x = maybe_float(item.get("x"))
    y = maybe_float(item.get("y"))
    if x is not None and y is not None and (x != bx or y != by):
        cv2.drawMarker(out, (int(round(x)), int(round(y))), (255, 220, 60), cv2.MARKER_CROSS, 26, 2, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (OUT_SIZE[0], 34), (15, 15, 15), -1)
    cv2.putText(out, label[:92], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.resize(out, SHEET_FRAME_SIZE, interpolation=cv2.INTER_AREA)


def item_strip(item: dict[str, Any], sheet_index: int, seconds: float) -> np.ndarray | None:
    video = resolve_path(item.get("video_path"))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[np.ndarray] = []
    for t in sequence_times(item, seconds):
        frame = read_resized_frame(cap, fps, t)
        if frame is None:
            continue
        label = f"#{sheet_index:03d} {Path(str(item.get('source_video') or '')).stem} {str(item.get('kind') or '?')} {t:.2f}s"
        frames.append(draw_event_overlay(frame, item, label))
    cap.release()
    if not frames:
        return None
    strip = np.concatenate(frames, axis=1)
    canvas = np.full((strip.shape[0] + STRIP_FOOTER_H, strip.shape[1], 3), 18, dtype=np.uint8)
    canvas[: strip.shape[0], : strip.shape[1]] = strip
    tags = " ".join(str(tag) for tag in item.get("review_tags", [])[:10])
    lines = [
        f"{sheet_index:03d} {item.get('batch_item_id')} status={item.get('review_status')} score={item.get('evidence_priority')}",
        f"{item.get('kind')} t={float(item.get('time_sec') or 0.0):.3f}s side={item.get('contact_side')} type={item.get('contact_type')} ball={item.get('qa_ball_accuracy')} side_conf={item.get('side_confidence')}",
        tags,
        str(item.get("note") or item.get("label") or "")[:150],
    ]
    for idx, line in enumerate(lines):
        cv2.putText(canvas, line[:170], (8, strip.shape[0] + 18 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


def write_sheet(items: list[dict[str, Any]], out_path: Path, *, seconds: float, cols: int) -> int:
    strips: list[np.ndarray] = []
    for idx, item in enumerate(items, start=1):
        strip = item_strip(item, idx, seconds)
        if strip is not None:
            strips.append(strip)
    if not strips:
        return 0
    strip_h = max(strip.shape[0] for strip in strips)
    strip_w = max(strip.shape[1] for strip in strips)
    rows = math.ceil(len(strips) / cols)
    sheet = np.full((rows * strip_h, cols * strip_w, 3), 16, dtype=np.uint8)
    for idx, strip in enumerate(strips):
        row = idx // cols
        col = idx % cols
        y = row * strip_h
        x = col * strip_w
        sheet[y : y + strip.shape[0], x : x + strip.shape[1]] = strip
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    return len(strips)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Review Evidence Pack",
        "",
        "This pack groups the current review batch by release blockers. Yellow circles mark the QA ball center; magenta circles mark the detected foot/limb point; the white line is the frame center.",
        "",
        f"- Batch: `{payload['batch_id']}`",
        f"- Items indexed: {summary['items']}",
        f"- Sheets written: {summary['sheets_written']}",
        f"- Review status counts: {', '.join(f'{key}={value}' for key, value in sorted(summary['review_status_counts'].items())) or 'none'}",
        "",
        "## Sheets",
        "",
        "| Bucket | Items | Sheet |",
        "| --- | ---: | --- |",
    ]
    for bucket in BUCKETS:
        sheet = payload["sheets"].get(bucket)
        if not sheet:
            continue
        lines.append(f"| `{bucket}` | {sheet['items']} | `{sheet['path']}` |")
    lines.extend(
        [
            "",
            "## Highest Priority Items",
            "",
            "| Rank | Item | Kind | Time | Status | Priority | Tags |",
            "| ---: | --- | --- | ---: | --- | ---: | --- |",
        ]
    )
    for idx, item in enumerate(payload["items"][:40], start=1):
        tags = " ".join(str(tag) for tag in item.get("review_tags", [])[:8])
        lines.append(
            f"| {idx} | `{item['batch_item_id']}` | `{item.get('kind')}` | "
            f"{float(item.get('time_sec') or 0):.3f} | {item.get('review_status')} | "
            f"{float(item.get('evidence_priority') or 0):.3f} | {tags} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_evidence_pack(
    batch_path: Path,
    out_dir: Path,
    reviews_dir: Path,
    suggestions_path: Path | None,
    *,
    max_items_per_bucket: int,
    seconds: float,
    cols: int,
) -> dict[str, Any]:
    batch = read_json(batch_path)
    suggestions = load_suggestions(suggestions_path)
    statuses = review_status_lookup(reviews_dir)
    bucket_items: dict[str, list[dict[str, Any]]] = defaultdict(list)
    status_counts: Counter[str] = Counter()
    indexed: list[dict[str, Any]] = []
    for item in batch.get("items", []):
        key = (str(item.get("source_video") or ""), str(item.get("review_item_id") or ""))
        status = statuses.get(key, "unreviewed")
        suggestion = suggestions.get(str(item.get("batch_item_id") or ""))
        enriched = dict(item)
        enriched["review_status"] = status
        enriched["assist_bucket"] = None if suggestion is None else suggestion.get("bucket")
        enriched["assist_action"] = None if suggestion is None else suggestion.get("recommended_action")
        enriched["evidence_priority"] = priority_score(enriched, suggestion, status)
        enriched["evidence_buckets"] = item_buckets(enriched)
        indexed.append(enriched)
        status_counts[status] += 1
        for bucket in enriched["evidence_buckets"]:
            bucket_items[bucket].append(enriched)

    for bucket in bucket_items:
        bucket_items[bucket].sort(
            key=lambda row: (
                str(row.get("review_status")) not in {"unreviewed", "pending"},
                -float(row.get("evidence_priority") or 0.0),
                str(row.get("source_video") or ""),
                float(row.get("time_sec") or 0.0),
            )
        )
    indexed.sort(
        key=lambda row: (
            str(row.get("review_status")) not in {"unreviewed", "pending"},
            -float(row.get("evidence_priority") or 0.0),
            str(row.get("source_video") or ""),
            float(row.get("time_sec") or 0.0),
        )
    )

    sheets: dict[str, dict[str, Any]] = {}
    sheets_dir = out_dir / "sheets"
    for bucket in BUCKETS:
        selected = bucket_items.get(bucket, [])[:max_items_per_bucket]
        if not selected:
            continue
        out_path = sheets_dir / f"{bucket}_sheet.jpg"
        written = write_sheet(selected, out_path, seconds=seconds, cols=cols)
        if written:
            sheets[bucket] = {"path": rel(out_path), "items": written}

    items_out: list[dict[str, Any]] = []
    for item in indexed:
        items_out.append(
            {
                "batch_item_id": item.get("batch_item_id"),
                "review_item_id": item.get("review_item_id"),
                "source_video": item.get("source_video"),
                "kind": item.get("kind"),
                "time_sec": item.get("time_sec"),
                "review_status": item.get("review_status"),
                "evidence_priority": item.get("evidence_priority"),
                "evidence_buckets": item.get("evidence_buckets"),
                "assist_bucket": item.get("assist_bucket"),
                "assist_action": item.get("assist_action"),
                "contact_side": item.get("contact_side"),
                "contact_type": item.get("contact_type"),
                "side_confidence": item.get("side_confidence"),
                "side_source": item.get("side_source"),
                "side_uncertainty_reason": item.get("side_uncertainty_reason"),
                "qa_ball_accuracy": item.get("qa_ball_accuracy"),
                "qa_ball_correction_px": item.get("qa_ball_correction_px"),
                "foot_x": item.get("foot_x"),
                "foot_y": item.get("foot_y"),
                "foot_confidence": item.get("foot_confidence"),
                "foot_distance": item.get("foot_distance"),
                "review_tags": item.get("review_tags", []),
                "note": item.get("note"),
                "crop_path": item.get("crop_path"),
                "tile_path": item.get("tile_path"),
            }
        )
    payload = {
        "schema_version": 1,
        "batch_path": rel(batch_path),
        "batch_id": batch.get("batch_id"),
        "reviews_dir": rel(reviews_dir),
        "suggestions_path": None if suggestions_path is None else rel(suggestions_path),
        "summary": {
            "items": len(items_out),
            "sheets_written": len(sheets),
            "bucket_counts": {bucket: len(bucket_items.get(bucket, [])) for bucket in BUCKETS},
            "review_status_counts": dict(sorted(status_counts.items())),
        },
        "sheets": sheets,
        "items": items_out,
    }
    write_json(out_dir / "review_evidence_manifest.json", payload)
    write_report(out_dir / "review_evidence_report.md", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build grouped review evidence sheets")
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--reviews-dir", type=Path, default=DEFAULT_REVIEWS_DIR)
    parser.add_argument("--suggestions", type=Path)
    parser.add_argument("--max-items-per-bucket", type=int, default=36)
    parser.add_argument("--seconds", type=float, default=0.56)
    parser.add_argument("--cols", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_evidence_pack(
        args.batch,
        args.out_dir,
        args.reviews_dir,
        args.suggestions,
        max_items_per_bucket=args.max_items_per_bucket,
        seconds=args.seconds,
        cols=args.cols,
    )
    print(f"manifest: {args.out_dir / 'review_evidence_manifest.json'}")
    print(f"report: {args.out_dir / 'review_evidence_report.md'}")
    for bucket, sheet in payload["sheets"].items():
        print(f"{bucket}: {sheet['path']} ({sheet['items']} items)")


if __name__ == "__main__":
    main()
