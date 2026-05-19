#!/usr/bin/env python3
"""Apply review decisions back into QA events and rally analytics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qa_rally_enrichment import assign_rallies, event_time, write_events_csv
from review_app import REVIEWABLE_KINDS, candidate_item, slug_id


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "outputs" / "full_training_27_qa" / "qa_manifest.json"
DEFAULT_REVIEWS = ROOT / "reviews"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "review_applied_qa"


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


def resolve_path(raw: Any) -> Path:
    path = Path(str(raw or ""))
    if path.exists():
        return path
    root_path = ROOT / path
    if root_path.exists():
        return root_path
    return path


def load_review_docs(reviews_dir: Path) -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    if not reviews_dir.exists():
        return docs
    for path in sorted(reviews_dir.glob("*.review.json")):
        try:
            doc = read_json(path)
        except json.JSONDecodeError:
            continue
        source_video = str(doc.get("source_video") or "")
        stem = path.name.removesuffix(".review.json")
        keys = {stem, slug_id(stem)}
        if source_video:
            keys.add(source_video)
            keys.add(Path(source_video).name)
            keys.add(slug_id(Path(source_video).stem))
        for key in keys:
            if key:
                item = dict(doc)
                item["_review_path"] = str(path)
                docs[key] = item
    return docs


def review_doc_for_source(review_docs: dict[str, dict[str, Any]], source_video: str) -> dict[str, Any] | None:
    keys = [source_video, Path(source_video).name, slug_id(Path(source_video).stem)]
    for key in keys:
        doc = review_docs.get(key)
        if doc is not None:
            return doc
    return None


def review_items_by_id(review_doc: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not review_doc:
        return {}
    return {str(item.get("id") or ""): item for item in review_doc.get("items", []) if item.get("id")}


def update_event_from_review(event: dict[str, Any], review_item: dict[str, Any]) -> dict[str, Any]:
    updated = dict(event)
    for field in (
        "contact_side",
        "contact_type",
        "side_confidence",
        "side_source",
        "side_uncertainty_reason",
        "trick_label",
        "start_sec",
        "end_sec",
        "duration_sec",
        "x",
        "y",
        "foot_x",
        "foot_y",
        "foot_confidence",
        "foot_distance",
    ):
        if review_item.get(field) not in (None, ""):
            updated[field] = review_item[field]
    if review_item.get("x") not in (None, ""):
        updated["qa_ball_x"] = review_item["x"]
    if review_item.get("y") not in (None, ""):
        updated["qa_ball_y"] = review_item["y"]
    updated["review_status"] = str(review_item.get("status") or "approved")
    updated["review_item_id"] = review_item.get("id")
    updated["review_note"] = review_item.get("note")
    return updated


def missing_review_item_to_event(review_item: dict[str, Any]) -> dict[str, Any] | None:
    kind = str(review_item.get("kind") or "")
    if kind not in REVIEWABLE_KINDS:
        return None
    status = str(review_item.get("status") or "")
    source = str(review_item.get("source") or "")
    active_approved = bool(review_item.get("active_learning_candidate")) and status == "approved"
    if not ((source == "manual" and status == "missing") or active_approved):
        return None
    time_sec = round(float(review_item.get("time_sec") or review_item.get("start_sec") or 0.0), 3)
    x = review_item.get("x")
    y = review_item.get("y")
    contact_type = review_item.get("contact_type") or ("ground" if kind == "drop_floor" else "stall" if kind == "stall" else "unknown")
    event = {
        "type": kind,
        "time_sec": time_sec,
        "label": "review_missing_event",
        "note": review_item.get("note") or "added from review decision",
        "review_status": "missing",
        "review_item_id": review_item.get("id"),
        "review_source": "manual_missing",
        "x": x,
        "y": y,
        "qa_ball_x": x,
        "qa_ball_y": y,
        "qa_ball_radius": review_item.get("qa_ball_radius") or 18.0,
        "qa_ball_confidence": review_item.get("ball_confidence") or 0.0,
        "qa_ball_accuracy": review_item.get("ball_accuracy") or "reviewed",
        "foot_x": review_item.get("foot_x"),
        "foot_y": review_item.get("foot_y"),
        "foot_confidence": review_item.get("foot_confidence"),
        "foot_distance": review_item.get("foot_distance"),
        "contact_side": review_item.get("contact_side") or "unknown",
        "contact_type": contact_type,
        "contact_confidence": review_item.get("contact_confidence") or 1.0,
        "side_confidence": review_item.get("side_confidence"),
        "side_source": review_item.get("side_source"),
        "side_uncertainty_reason": review_item.get("side_uncertainty_reason"),
        "confidence": review_item.get("confidence") or 1.0,
        "audio_z": review_item.get("audio_z"),
        "visual_score": review_item.get("visual_score"),
        "motion_score": review_item.get("motion_score"),
        "trick_label": review_item.get("trick_label"),
        "drop_source": review_item.get("drop_source") or ("review_missing" if kind == "drop_floor" else None),
        "drop_score": review_item.get("drop_score"),
    }
    if review_item.get("start_sec") is not None:
        event["start_sec"] = review_item.get("start_sec")
    if review_item.get("end_sec") is not None:
        event["end_sec"] = review_item.get("end_sec")
    if review_item.get("duration_sec") is not None:
        event["duration_sec"] = review_item.get("duration_sec")
    return {key: value for key, value in event.items() if value is not None}


def apply_reviews_to_doc(doc: dict[str, Any], event_path: Path, review_doc: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, int]]:
    by_id = review_items_by_id(review_doc)
    stats = {
        "approved_candidates": 0,
        "rejected_candidates": 0,
        "pending_candidates": 0,
        "manual_missing_inserted": 0,
    }
    merged_events: list[dict[str, Any]] = []
    existing_review_ids = {str(item.get("review_item_id") or "") for item in doc.get("events", []) if item.get("review_item_id")}
    for event in doc.get("events", []):
        item = candidate_item(event, event_path)
        review_item = by_id.get(str(item.get("id") if item else ""))
        if review_item is None:
            merged_events.append(dict(event))
            continue
        status = str(review_item.get("status") or "pending")
        if status == "rejected":
            stats["rejected_candidates"] += 1
            continue
        if status == "approved":
            stats["approved_candidates"] += 1
            merged_events.append(update_event_from_review(event, review_item))
        else:
            stats["pending_candidates"] += 1
            merged_events.append(dict(event))

    for review_item in by_id.values():
        event = missing_review_item_to_event(review_item)
        if event is None:
            continue
        review_id = str(event.get("review_item_id") or "")
        if review_id and review_id in existing_review_ids:
            continue
        merged_events.append(event)
        stats["manual_missing_inserted"] += 1

    merged_events.sort(key=event_time)
    enriched_events, rallies = assign_rallies(merged_events)
    out_doc = dict(doc)
    out_doc["annotation_method"] = "qa_events_with_review_decisions_applied"
    out_doc["events"] = enriched_events
    out_doc["rallies"] = rallies
    out_doc["review_application"] = stats
    return out_doc, stats


def run_apply(manifest_path: Path, reviews_dir: Path, out_root: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    review_docs = load_review_docs(reviews_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    totals = {
        "videos": 0,
        "approved_candidates": 0,
        "rejected_candidates": 0,
        "pending_candidates": 0,
        "manual_missing_inserted": 0,
    }
    for run in manifest.get("runs", []):
        qa_path = resolve_path(run.get("qa_events_path"))
        if not qa_path.exists():
            continue
        doc = read_json(qa_path)
        source_video = str(doc.get("source_video") or Path(str(run.get("video") or qa_path.parent.name)).name)
        review_doc = review_doc_for_source(review_docs, source_video)
        out_doc, stats = apply_reviews_to_doc(doc, qa_path, review_doc)
        slug = slug_id(Path(source_video).stem or qa_path.parent.name)
        out_dir = out_root / slug
        out_events = out_dir / "qa_events.json"
        out_csv = out_dir / "qa_events.csv"
        write_json(out_events, out_doc)
        write_events_csv(out_doc.get("events", []), out_csv)
        for key in totals:
            if key == "videos":
                continue
            totals[key] += int(stats.get(key) or 0)
        totals["videos"] += 1
        runs.append(
            {
                "video": run.get("video") or source_video,
                "qa_events_path": rel(out_events),
                "qa_csv_path": rel(out_csv),
                "qa_sheet_path": run.get("qa_sheet_path"),
                "touch_candidates": sum(1 for item in out_doc.get("events", []) if item.get("type") == "touch"),
                "ground_hit_candidates": sum(1 for item in out_doc.get("events", []) if item.get("type") == "drop_floor"),
                "stall_candidates": sum(1 for item in out_doc.get("events", []) if item.get("type") == "stall"),
                "around_the_world_candidates": sum(1 for item in out_doc.get("events", []) if item.get("type") == "around_the_world"),
                "rallies": len(out_doc.get("rallies", [])),
                "reviews_applied": stats,
            }
        )
    out_manifest = {
        "schema_version": 1,
        "source_manifest": rel(manifest_path),
        "reviews_dir": rel(reviews_dir),
        "annotation_method": "qa_events_with_review_decisions_applied",
        "summary": totals,
        "runs": runs,
    }
    manifest_out = out_root / "qa_manifest.json"
    write_json(manifest_out, out_manifest)
    return {"manifest_path": manifest_out, "summary": totals, "runs": runs}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply review decisions to QA events and recompute rallies")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reviews-dir", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_apply(args.manifest, args.reviews_dir, args.out_root)
    print(f"review-applied manifest: {result['manifest_path']}")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
