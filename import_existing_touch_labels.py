#!/usr/bin/env python3
"""Import legacy `data/*.events.json` touch labels into the review-label format.

The current touch classifier pipeline consumes completed muted-review label
files with candidate-level touch/no-touch decisions. Earlier work stored human
touch labels directly in `data/*.events.json`; this importer preserves that
work by converting those events into the newer review format and deriving
candidate decisions from the approved touch times.
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
EVENT_TYPES = {"touch", "drop_floor", "stall"}


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


def iter_legacy_events(doc: dict[str, Any]) -> list[dict[str, Any]]:
    events = []
    for rally in doc.get("rallies", []):
        for event in rally.get("events", []):
            if event.get("type") not in EVENT_TYPES:
                continue
            if event.get("time_sec") is None:
                raise ValueError(f"legacy event missing time_sec: {event}")
            row = {
                "type": event["type"],
                "time_sec": round(float(event["time_sec"]), 3),
                "review_status": "approved",
                "source": "legacy_events_import",
            }
            if event.get("duration_sec") not in (None, ""):
                row["duration_sec"] = round(float(event["duration_sec"]), 3)
            for key in ("label", "confidence", "note", "touch_number"):
                if event.get(key) is not None:
                    row[f"legacy_{key}"] = event[key]
            events.append(row)
    return sorted(events, key=lambda row: (float(row["time_sec"]), row["type"]))


def candidate_reviews_from_legacy(
    candidates_payload: dict[str, Any],
    legacy_events: list[dict[str, Any]],
    *,
    touch_tolerance_sec: float,
    cluster_gap_sec: float,
) -> list[dict[str, Any]]:
    touch_times = [float(row["time_sec"]) for row in legacy_events if row["type"] == "touch"]
    reviews = []
    for cluster in cluster_candidates(
        candidates_payload,
        cluster_gap_sec=cluster_gap_sec,
        include_generated_event_hints=False,
    ):
        is_touch = any(abs(cluster.time_sec - touch_time) <= touch_tolerance_sec for touch_time in touch_times)
        reviews.append(
            {
                "time_sec": round(float(cluster.time_sec), 3),
                "decision": "touch" if is_touch else "no_touch",
                "review_status": "reviewed",
                "source": "legacy_events_import",
            }
        )
    return reviews


def import_one(
    item: dict[str, Any],
    *,
    candidates_dir: Path,
    labels_dir: Path,
    touch_tolerance_sec: float,
    cluster_gap_sec: float,
    force: bool,
    dry_run: bool,
) -> dict[str, Any] | None:
    video_id = str(item["video_id"])
    existing_events_path = item.get("existing_events_path")
    if not existing_events_path:
        return None
    source_path = Path(existing_events_path)
    if not source_path.exists():
        raise FileNotFoundError(f"missing legacy events for {video_id}: {source_path}")
    out_path = label_path(labels_dir, video_id)
    if out_path.exists() and not force:
        return {
            "video_id": video_id,
            "video_name": item["video_name"],
            "status": "skipped_existing_label",
            "label_path": str(out_path),
        }
    candidates_file = candidate_path(candidates_dir, video_id)
    if not candidates_file.exists():
        raise FileNotFoundError(f"missing candidates for {video_id}: {candidates_file}")
    legacy_doc = read_json(source_path)
    if Path(str(legacy_doc.get("source_video") or "")).name != item["video_name"]:
        raise ValueError(
            f"{source_path}: source_video {legacy_doc.get('source_video')!r} "
            f"does not match manifest video {item['video_name']!r}"
        )
    legacy_events = iter_legacy_events(legacy_doc)
    candidates_payload = read_json(candidates_file)
    candidate_reviews = candidate_reviews_from_legacy(
        candidates_payload,
        legacy_events,
        touch_tolerance_sec=touch_tolerance_sec,
        cluster_gap_sec=cluster_gap_sec,
    )
    imported_at = now_iso()
    doc = {
        "schema_version": 1,
        "source_video": item["video_name"],
        "source_video_path": item["video_path"],
        "split": item["split"],
        "annotation_method": "muted_visual_touch_review",
        "audio_muted_during_review_required": True,
        "candidate_review_complete": True,
        "review_status": "complete",
        "review_completed_at": imported_at,
        "created_at": imported_at,
        "updated_at": imported_at,
        "review_note": (
            "Imported from legacy events.json so existing human touch labels can be used by "
            "the candidate-based classifier pipeline. Candidate decisions were derived from "
            "legacy approved touch times; inspect legacy_import before using as final release evidence."
        ),
        "legacy_import": {
            "source_events_path": str(source_path),
            "source_annotation_method": legacy_doc.get("annotation_method"),
            "source_caveat": legacy_doc.get("caveat"),
            "touch_tolerance_sec": touch_tolerance_sec,
            "cluster_gap_sec": cluster_gap_sec,
            "audio_assisted_source": "audio" in str(legacy_doc.get("annotation_method") or "").lower(),
        },
        "candidate_reviews": candidate_reviews,
        "rallies": [
            {
                "id": 1,
                "label": "legacy imported full-video review",
                "start_sec": 0.0,
                "end_sec": None,
                "events": legacy_events,
            }
        ],
    }
    if not dry_run:
        write_json(out_path, doc)
    return {
        "video_id": video_id,
        "video_name": item["video_name"],
        "split": item["split"],
        "status": "imported" if not dry_run else "would_import",
        "source_events_path": str(source_path),
        "label_path": str(out_path),
        "approved_events": len(legacy_events),
        "approved_touches": sum(1 for row in legacy_events if row["type"] == "touch"),
        "candidate_reviews": len(candidate_reviews),
        "candidate_touch_reviews": sum(1 for row in candidate_reviews if row["decision"] == "touch"),
        "audio_assisted_source": doc["legacy_import"]["audio_assisted_source"],
    }


def import_existing_labels(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_json(args.review_manifest)
    labels_dir = args.labels_dir.resolve()
    candidates_dir = args.candidates_dir.resolve()
    rows = []
    for item in manifest.get("items", []):
        result = import_one(
            item,
            candidates_dir=candidates_dir,
            labels_dir=labels_dir,
            touch_tolerance_sec=args.touch_tolerance_sec,
            cluster_gap_sec=args.cluster_gap_sec,
            force=args.force,
            dry_run=args.dry_run,
        )
        if result:
            rows.append(result)
    summary = {
        "schema_version": 1,
        "created_at": now_iso(),
        "review_manifest": str(args.review_manifest),
        "candidates_dir": str(candidates_dir),
        "labels_dir": str(labels_dir),
        "touch_tolerance_sec": args.touch_tolerance_sec,
        "cluster_gap_sec": args.cluster_gap_sec,
        "dry_run": args.dry_run,
        "force": args.force,
        "videos_with_legacy_events": len(rows),
        "imported": sum(1 for row in rows if row["status"] == "imported"),
        "skipped_existing_label": sum(1 for row in rows if row["status"] == "skipped_existing_label"),
        "approved_touches": sum(int(row.get("approved_touches") or 0) for row in rows),
        "candidate_reviews": sum(int(row.get("candidate_reviews") or 0) for row in rows),
        "audio_assisted_imports": [row["video_id"] for row in rows if row.get("audio_assisted_source")],
        "videos": rows,
    }
    if not args.dry_run:
        write_json(labels_dir / "legacy_import_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import legacy touch event labels into visual review format")
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW_MANIFEST)
    parser.add_argument("--candidates-dir", type=Path, default=DEFAULT_CANDIDATES_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--touch-tolerance-sec", type=float, default=0.20)
    parser.add_argument("--cluster-gap-sec", type=float, default=0.04)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = import_existing_labels(parse_args())
    print(
        json.dumps(
            {
                "videos_with_legacy_events": summary["videos_with_legacy_events"],
                "imported": summary["imported"],
                "skipped_existing_label": summary["skipped_existing_label"],
                "approved_touches": summary["approved_touches"],
                "candidate_reviews": summary["candidate_reviews"],
                "audio_assisted_imports": summary["audio_assisted_imports"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
