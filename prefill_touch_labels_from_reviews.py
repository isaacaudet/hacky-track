#!/usr/bin/env python3
"""Prefill draft touch-review labels from existing QA review subsets.

This helper is intentionally conservative. The older `reviews/*.review.json`
files contain partial visual QA decisions, not a full muted touch-label pass.
They can reduce review effort by pre-checking matching candidate moments, but
they must not satisfy the release-gate label requirement on their own.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from build_touch_training_table import cluster_candidates


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "runs/release-27-public/touch_corpus_v1"
DEFAULT_REVIEW_MANIFEST = DEFAULT_CORPUS / "touch_review_manifest.json"
DEFAULT_CANDIDATES_DIR = DEFAULT_CORPUS / "audio_candidates"
DEFAULT_LABELS_DIR = DEFAULT_CORPUS / "visual_touch_labels"
DEFAULT_REVIEWS_DIR = ROOT / "reviews"
EVENT_TYPES = {"touch", "drop_floor", "stall"}
USABLE_REVIEW_STATUSES = {"approved", "rejected"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def label_path(labels_dir: Path, video_id: str) -> Path:
    return labels_dir / f"{safe_slug(video_id)}.events.json"


def candidate_path(candidates_dir: Path, video_id: str) -> Path:
    return candidates_dir / f"{safe_slug(video_id)}.touch_candidates.json"


def review_path(reviews_dir: Path, video_id: str) -> Path:
    return reviews_dir / f"{safe_slug(video_id)}.review.json"


def usable_review_items(doc: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in doc.get("items", []):
        status = str(item.get("status") or "")
        if status not in USABLE_REVIEW_STATUSES:
            continue
        if item.get("time_sec") is None:
            continue
        rows.append(
            {
                "id": item.get("id"),
                "kind": str(item.get("kind") or ""),
                "status": status,
                "time_sec": round(float(item["time_sec"]), 3),
                "duration_sec": item.get("duration_sec"),
                "reviewed_by": item.get("reviewed_by"),
                "reviewed_at": item.get("reviewed_at"),
                "review_evidence": item.get("review_evidence") or item.get("note"),
            }
        )
    return sorted(rows, key=lambda row: (float(row["time_sec"]), row["kind"], row["status"]))


def conflicting_item_ids(items: list[dict[str, Any]], *, tolerance_sec: float) -> set[Any]:
    conflicted: set[Any] = set()
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            if left["kind"] != right["kind"]:
                continue
            if abs(float(left["time_sec"]) - float(right["time_sec"])) > tolerance_sec:
                continue
            statuses = {left["status"], right["status"]}
            if statuses == {"approved", "rejected"}:
                conflicted.add(left.get("id"))
                conflicted.add(right.get("id"))
    return conflicted


def nearest_review_item(
    time_sec: float,
    items: list[dict[str, Any]],
    *,
    tolerance_sec: float,
) -> tuple[dict[str, Any] | None, float | None]:
    candidates = [(item, abs(float(item["time_sec"]) - time_sec)) for item in items]
    candidates = [(item, delta) for item, delta in candidates if delta <= tolerance_sec]
    if not candidates:
        return None, None
    item, delta = min(candidates, key=lambda pair: (pair[1], 0 if pair[0]["status"] == "approved" else 1))
    return item, delta


def event_from_review_item(item: dict[str, Any]) -> dict[str, Any]:
    event = {
        "type": item["kind"],
        "time_sec": round(float(item["time_sec"]), 3),
        "review_status": "approved",
        "source": "review_subset_prefill",
    }
    if item.get("duration_sec") not in (None, ""):
        event["duration_sec"] = round(float(item["duration_sec"]), 3)
    return event


def prefill_doc_for_item(
    *,
    manifest_item: dict[str, Any],
    review_doc: dict[str, Any],
    candidates_payload: dict[str, Any],
    candidate_match_tolerance_sec: float,
    conflict_tolerance_sec: float,
    cluster_gap_sec: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    video_id = str(manifest_item["video_id"])
    items = usable_review_items(review_doc)
    conflicted_ids = conflicting_item_ids(items, tolerance_sec=conflict_tolerance_sec)
    usable = [item for item in items if item.get("id") not in conflicted_ids]

    events = []
    seen_events: set[tuple[str, float]] = set()
    for item in usable:
        if item["status"] != "approved" or item["kind"] not in EVENT_TYPES:
            continue
        event = event_from_review_item(item)
        key = (event["type"], float(event["time_sec"]))
        if key in seen_events:
            continue
        seen_events.add(key)
        events.append(event)

    candidate_reviews = []
    for cluster in cluster_candidates(candidates_payload, cluster_gap_sec=cluster_gap_sec):
        matched, delta = nearest_review_item(cluster.time_sec, usable, tolerance_sec=candidate_match_tolerance_sec)
        if not matched:
            continue
        decision = "touch" if matched["kind"] == "touch" and matched["status"] == "approved" else "no_touch"
        candidate_reviews.append(
            {
                "time_sec": round(float(cluster.time_sec), 3),
                "decision": decision,
                "review_status": "reviewed",
                "source": "review_subset_prefill",
                "source_review_item_id": matched.get("id"),
                "source_review_item_time_sec": matched["time_sec"],
                "source_review_item_kind": matched["kind"],
                "source_review_item_status": matched["status"],
                "source_review_delta_sec": round(float(delta or 0.0), 3),
            }
        )

    created_at = now_iso()
    doc = {
        "schema_version": 1,
        "source_video": manifest_item["video_name"],
        "source_video_path": manifest_item["video_path"],
        "split": manifest_item["split"],
        "annotation_method": "muted_visual_touch_review",
        "audio_muted_during_review_required": True,
        "candidate_review_complete": False,
        "review_status": "in_progress",
        "review_completed_at": None,
        "created_at": created_at,
        "updated_at": created_at,
        "review_note": (
            "Draft prefilled from an older partial QA review subset. This is not a complete "
            "muted review; open touch_review_app.py, verify every hint visually, and only then "
            "save complete."
        ),
        "prefill_import": {
            "source": "reviews_subset",
            "candidate_match_tolerance_sec": candidate_match_tolerance_sec,
            "cluster_gap_sec": cluster_gap_sec,
            "conflict_tolerance_sec": conflict_tolerance_sec,
            "conflicted_review_item_ids": sorted(str(item_id) for item_id in conflicted_ids if item_id is not None),
            "usable_review_items": len(usable),
            "ignored_review_items": len(review_doc.get("items", [])) - len(usable),
        },
        "candidate_reviews": sorted(candidate_reviews, key=lambda row: float(row["time_sec"])),
        "rallies": [
            {
                "id": 1,
                "label": "partial QA review prefill",
                "start_sec": 0.0,
                "end_sec": None,
                "events": sorted(events, key=lambda row: (float(row["time_sec"]), row["type"])),
            }
        ],
    }
    stats = {
        "video_id": video_id,
        "video_name": manifest_item["video_name"],
        "split": manifest_item["split"],
        "review_items": len(review_doc.get("items", [])),
        "usable_review_items": len(usable),
        "conflicted_review_items": len(conflicted_ids),
        "prefilled_events": len(events),
        "prefilled_touches": sum(1 for event in events if event["type"] == "touch"),
        "candidate_reviews": len(candidate_reviews),
        "candidate_touch_reviews": sum(1 for row in candidate_reviews if row["decision"] == "touch"),
    }
    return doc, stats


def prefill_labels(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_json(args.review_manifest)
    labels_dir = args.labels_dir.resolve()
    candidates_dir = args.candidates_dir.resolve()
    reviews_dir = args.reviews_dir.resolve()
    rows = []
    for item in manifest.get("items", []):
        if args.split and item.get("split") != args.split:
            continue
        video_id = str(item["video_id"])
        source_review = review_path(reviews_dir, video_id)
        if not source_review.exists():
            rows.append({"video_id": video_id, "video_name": item["video_name"], "status": "missing_review_file"})
            continue
        out_path = label_path(labels_dir, video_id)
        if out_path.exists() and not args.force:
            rows.append({"video_id": video_id, "video_name": item["video_name"], "status": "skipped_existing_label", "label_path": str(out_path)})
            continue
        candidates_file = candidate_path(candidates_dir, video_id)
        if not candidates_file.exists():
            raise FileNotFoundError(f"missing candidates for {video_id}: {candidates_file}")
        review_doc = read_json(source_review)
        if Path(str(review_doc.get("source_video") or "")).name != item["video_name"]:
            raise ValueError(
                f"{source_review}: source_video {review_doc.get('source_video')!r} "
                f"does not match manifest video {item['video_name']!r}"
            )
        doc, stats = prefill_doc_for_item(
            manifest_item=item,
            review_doc=review_doc,
            candidates_payload=read_json(candidates_file),
            candidate_match_tolerance_sec=args.candidate_match_tolerance_sec,
            conflict_tolerance_sec=args.conflict_tolerance_sec,
            cluster_gap_sec=args.cluster_gap_sec,
        )
        if stats["usable_review_items"] == 0 and not args.write_empty:
            rows.append({"status": "skipped_no_usable_review_items", "review_path": str(source_review), **stats})
            continue
        if not args.dry_run:
            write_json(out_path, doc)
        rows.append(
            {
                "status": "prefilled" if not args.dry_run else "would_prefill",
                "review_path": str(source_review),
                "label_path": str(out_path),
                **stats,
            }
        )
    summary = {
        "schema_version": 1,
        "created_at": now_iso(),
        "review_manifest": str(args.review_manifest),
        "reviews_dir": str(reviews_dir),
        "candidates_dir": str(candidates_dir),
        "labels_dir": str(labels_dir),
        "split": args.split,
        "dry_run": args.dry_run,
        "force": args.force,
        "prefilled": sum(1 for row in rows if row["status"] == "prefilled"),
        "would_prefill": sum(1 for row in rows if row["status"] == "would_prefill"),
        "skipped_existing_label": sum(1 for row in rows if row["status"] == "skipped_existing_label"),
        "missing_review_file": sum(1 for row in rows if row["status"] == "missing_review_file"),
        "prefilled_events": sum(int(row.get("prefilled_events") or 0) for row in rows),
        "prefilled_touches": sum(int(row.get("prefilled_touches") or 0) for row in rows),
        "candidate_reviews": sum(int(row.get("candidate_reviews") or 0) for row in rows),
        "candidate_touch_reviews": sum(int(row.get("candidate_touch_reviews") or 0) for row in rows),
        "conflicted_review_items": sum(int(row.get("conflicted_review_items") or 0) for row in rows),
        "videos": rows,
    }
    if not args.dry_run:
        write_json(labels_dir / "review_subset_prefill_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefill draft visual touch labels from old QA review subsets")
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW_MANIFEST)
    parser.add_argument("--reviews-dir", type=Path, default=DEFAULT_REVIEWS_DIR)
    parser.add_argument("--candidates-dir", type=Path, default=DEFAULT_CANDIDATES_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--split", default="test_frozen", help="manifest split to prefill; empty string means all splits")
    parser.add_argument("--candidate-match-tolerance-sec", type=float, default=0.20)
    parser.add_argument("--conflict-tolerance-sec", type=float, default=0.05)
    parser.add_argument("--cluster-gap-sec", type=float, default=0.04)
    parser.add_argument("--write-empty", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split == "":
        args.split = None
    summary = prefill_labels(args)
    print(
        json.dumps(
            {
                "prefilled": summary["prefilled"],
                "would_prefill": summary["would_prefill"],
                "skipped_existing_label": summary["skipped_existing_label"],
                "missing_review_file": summary["missing_review_file"],
                "prefilled_events": summary["prefilled_events"],
                "prefilled_touches": summary["prefilled_touches"],
                "candidate_reviews": summary["candidate_reviews"],
                "candidate_touch_reviews": summary["candidate_touch_reviews"],
                "conflicted_review_items": summary["conflicted_review_items"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"summary: {args.labels_dir / 'review_subset_prefill_summary.json'}")


if __name__ == "__main__":
    main()
