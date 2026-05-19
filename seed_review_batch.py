#!/usr/bin/env python3
"""Seed review files with the current priority batch as pending items.

This does not approve, reject, or mark anything missing. It only makes the
selected batch measurable in validation and ready for the review app while
preserving any existing decisions.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import review_app
from review_app import ROOT, merged_review_doc, utc_now, write_json


DEFAULT_BATCH = ROOT / "outputs" / "review_batches" / "latest_review_batch.json"
DEFAULT_ASSISTED_REVIEW = ROOT / "outputs" / "review_batches" / "assisted_review_suggestions.json"
DEFAULT_REVIEWS_DIR = ROOT / "reviews"
DEFAULT_QA_MANIFEST = ROOT / "outputs" / "full_training_27_qa" / "qa_manifest.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def keep_existing_item(item: dict[str, Any], batch_ids: set[str]) -> bool:
    item_id = str(item.get("id") or "")
    if item_id in batch_ids:
        return False
    if item.get("source") == "manual":
        return True
    return str(item.get("status") or "pending") in {"approved", "rejected", "missing"}


def assisted_lookup(path: Path = DEFAULT_ASSISTED_REVIEW) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        doc = read_json(path)
    except json.JSONDecodeError:
        return {}
    return {str(item.get("batch_item_id") or ""): item for item in doc.get("items", []) if item.get("batch_item_id")}


def seed_batch(
    batch_path: Path,
    *,
    qa_manifest: Path | None = None,
    reviews_dir: Path | None = None,
    assisted_review: Path = DEFAULT_ASSISTED_REVIEW,
    dry_run: bool = False,
) -> dict[str, Any]:
    batch = read_json(batch_path)
    reviews_dir = reviews_dir or DEFAULT_REVIEWS_DIR
    review_app.DEFAULT_QA_MANIFEST = qa_manifest or DEFAULT_QA_MANIFEST
    review_app.DEFAULT_REVIEW_BATCH = batch_path
    review_app.DEFAULT_ASSISTED_REVIEW = assisted_review
    review_app.REVIEWS_DIR = reviews_dir
    assisted_by_batch_id = assisted_lookup(assisted_review)
    videos = review_app.discover_videos()
    by_stem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in batch.get("items", []):
        stem = str(record.get("review_stem") or "")
        if stem:
            by_stem[stem].append(record)

    summary = {
        "batch_path": str(batch_path),
        "qa_manifest": str(review_app.DEFAULT_QA_MANIFEST),
        "reviews_dir": str(reviews_dir),
        "assisted_review": str(assisted_review),
        "batch_id": batch.get("batch_id"),
        "batch_items": len(batch.get("items", [])),
        "videos_seen": 0,
        "review_files_written": 0,
        "seeded_pending_items": 0,
        "preserved_decided_or_manual_items": 0,
        "missing_video_stems": [],
        "written_files": [],
        "dry_run": dry_run,
    }

    for stem, records in sorted(by_stem.items()):
        info = videos.get(stem)
        if not info:
            summary["missing_video_stems"].append(stem)
            continue
        summary["videos_seen"] += 1
        base_doc = merged_review_doc(info)
        base_by_id = {str(item.get("id") or ""): item for item in base_doc.get("items", []) if item.get("id")}
        batch_ids = {str(record.get("review_item_id") or "") for record in records if record.get("review_item_id")}

        existing_items: list[dict[str, Any]] = []
        if info.review_path.exists():
            try:
                existing_items = read_json(info.review_path).get("items", [])
            except json.JSONDecodeError:
                existing_items = []
        existing_by_id = {str(item.get("id") or ""): item for item in existing_items if item.get("id")}

        output_items: list[dict[str, Any]] = []
        for item in existing_items:
            if keep_existing_item(item, batch_ids):
                output_items.append(item)
                summary["preserved_decided_or_manual_items"] += 1

        for record in sorted(records, key=lambda item: (float(item.get("time_sec") or 0.0), str(item.get("review_item_id") or ""))):
            item_id = str(record.get("review_item_id") or "")
            if not item_id:
                continue
            seeded = dict(base_by_id.get(item_id) or {})
            if not seeded:
                continue
            previous = existing_by_id.get(item_id)
            if previous:
                seeded.update(previous)
            seeded.setdefault("status", "pending")
            seeded["in_review_batch"] = True
            seeded["batch_id"] = record.get("batch_id") or batch.get("batch_id")
            seeded["batch_item_id"] = record.get("batch_item_id")
            seeded["batch_priority_score"] = record.get("priority_score")
            seeded["batch_crop_path"] = record.get("crop_path")
            seeded["batch_tile_path"] = record.get("tile_path")
            seeded["batch_selection_reasons"] = record.get("selection_reasons", [])
            tags = list(seeded.get("review_tags") or [])
            if "review_batch" not in tags:
                tags.insert(0, "review_batch")
            assisted = assisted_by_batch_id.get(str(record.get("batch_item_id") or ""))
            if assisted:
                seeded["assist_bucket"] = assisted.get("bucket")
                seeded["assist_action"] = assisted.get("recommended_action")
                seeded["assist_score"] = assisted.get("evidence_score")
                seeded["assist_reasons"] = assisted.get("reasons", [])
                bucket_tag = f"assist_{assisted.get('bucket')}"
                if assisted.get("bucket") and bucket_tag not in tags:
                    tags.append(bucket_tag)
            seeded["review_tags"] = tags
            output_items.append(seeded)
            if str(seeded.get("status") or "pending") == "pending":
                summary["seeded_pending_items"] += 1

        output_items.sort(key=lambda item: (float(item.get("time_sec") or 0.0), str(item.get("id") or "")))
        doc = {
            "version": 1,
            "source_video": info.source_video,
            "video_path": str(info.video_path),
            "candidate_source": str(info.event_path.relative_to(ROOT)) if info.event_path.is_relative_to(ROOT) else str(info.event_path),
            "qa_sheet_path": None if info.qa_sheet_path is None else str(info.qa_sheet_path),
            "track_size": info.track_size,
            "created_at": base_doc.get("created_at") or utc_now(),
            "updated_at": utc_now(),
            "items": output_items,
        }
        summary["written_files"].append(str(info.review_path.relative_to(ROOT)) if info.review_path.is_relative_to(ROOT) else str(info.review_path))
        if not dry_run:
            write_json(info.review_path, doc)
        summary["review_files_written"] += 1

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed priority review-batch items as pending review records")
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--qa-manifest", type=Path, default=DEFAULT_QA_MANIFEST)
    parser.add_argument("--reviews-dir", type=Path, default=DEFAULT_REVIEWS_DIR)
    parser.add_argument("--assisted-review", type=Path, default=DEFAULT_ASSISTED_REVIEW)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = seed_batch(
        args.batch,
        qa_manifest=args.qa_manifest,
        reviews_dir=args.reviews_dir,
        assisted_review=args.assisted_review,
        dry_run=args.dry_run,
    )
    out_path = args.out or args.reviews_dir / "seed_review_batch_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"seed summary: {out_path}")
    print(f"review files written: {summary['review_files_written']}")
    print(f"seeded pending items: {summary['seeded_pending_items']}")
    if summary["missing_video_stems"]:
        print(f"missing stems: {', '.join(summary['missing_video_stems'])}")


if __name__ == "__main__":
    main()
