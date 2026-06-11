#!/usr/bin/env python3
"""Prefill TRAIN-only draft touch-review suggestions.

This script is intentionally not a labeling step. It writes candidate-level
`review_status: "suggested"` rows so the review app can show a default decision,
while the training/export code continues to ignore them until a human confirms
the clip and saves reviewed candidate decisions.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from attach_touch_l2_features import (
    DEFAULT_MAX_TRACK_GAP_SEC,
    TrackFeatures,
    attach_row_features,
    compute_track_features,
    load_detection_tracks,
)
from owlv2_event_eval import AUDIO_FUSION_BREAK_TOL_SEC
from build_touch_training_table import cluster_candidates


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "runs/release-27-public/touch_corpus_v1"
DEFAULT_REVIEW_MANIFEST = DEFAULT_CORPUS / "touch_review_manifest.json"
DEFAULT_CANDIDATES_DIR = DEFAULT_CORPUS / "audio_candidates"
DEFAULT_LABELS_DIR = DEFAULT_CORPUS / "visual_touch_labels"
DEFAULT_DETECTIONS_JSONL = [
    ROOT / "runs/release-27-public/owlv2_l2_eval_v1/owlv2_raw_predictions.jsonl",
    DEFAULT_CORPUS / "owlv2_touch_detections_v1/detections.jsonl",
]
SUGGESTION_SOURCE = "train_audio_trajectory_draft_prefill"
STRICT_VISUAL_METHOD = "muted_visual_touch_review"
TOUCH_HINT_TYPES = {"touch"}
NON_TOUCH_HINT_TYPES = {"drop_floor", "stall"}


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


def empty_draft_doc(item: dict[str, Any], created_at: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_video": item["video_name"],
        "source_video_path": item["video_path"],
        "split": item["split"],
        "annotation_method": STRICT_VISUAL_METHOD,
        "audio_muted_during_review_required": True,
        "candidate_review_complete": False,
        "review_status": "in_progress",
        "review_completed_at": None,
        "created_at": created_at,
        "updated_at": created_at,
        "review_note": (
            "TRAIN-only draft suggestions from loose audio onsets plus OWLv2/L2 trajectory "
            "breaks. These are not labels. Confirm from muted video before completing."
        ),
        "candidate_reviews": [],
        "candidate_suggestions": [],
        "rallies": [
            {
                "id": 1,
                "label": "draft suggestion prefill only",
                "start_sec": 0.0,
                "end_sec": None,
                "events": [],
            }
        ],
    }


def load_or_create_doc(path: Path, item: dict[str, Any], *, created_at: str) -> dict[str, Any]:
    if not path.exists():
        return empty_draft_doc(item, created_at)
    doc = read_json(path)
    source_video = Path(str(doc.get("source_video") or "")).name
    if source_video != item["video_name"]:
        raise ValueError(f"{path}: source_video {source_video!r} does not match manifest video {item['video_name']!r}")
    if doc.get("split") != item["split"]:
        raise ValueError(f"{path}: split {doc.get('split')!r} does not match manifest split {item['split']!r}")
    return doc


def empty_track(status: str) -> TrackFeatures:
    return TrackFeatures(status, [], np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([]), 0)


def track_for_video(
    item: dict[str, Any],
    tracks: dict[str, list[Any]],
    *,
    min_points: int,
    max_track_gap_sec: float,
) -> TrackFeatures:
    points = tracks.get(item["video_name"], [])
    if not points:
        return empty_track("missing_detection_track")
    return compute_track_features(points, min_points=min_points, max_gap_sec=max_track_gap_sec)


def candidate_feature_row(item: dict[str, Any], cluster: Any, index: int, clusters: list[Any]) -> dict[str, Any]:
    prev_time = None if index == 0 else clusters[index - 1].time_sec
    next_time = None if index + 1 >= len(clusters) else clusters[index + 1].time_sec
    return {
        "schema_version": 1,
        "video_id": item["video_id"],
        "video_name": item["video_name"],
        "split": item["split"],
        "candidate_time_sec": round(float(cluster.time_sec), 6),
        "has_audio": cluster.has_audio,
        "audio_strength": None if cluster.audio_strength is None else round(float(cluster.audio_strength), 6),
        "has_existing_hint": cluster.has_existing_hint,
        "existing_hint_types": list(cluster.existing_hint_types),
        "raw_sources": list(cluster.raw_sources),
        "time_since_prev_candidate_sec": None if prev_time is None else round(float(cluster.time_sec - prev_time), 6),
        "time_to_next_candidate_sec": None if next_time is None else round(float(next_time - cluster.time_sec), 6),
        "nearest_drop_delta_sec": None,
        "nearest_stall_delta_sec": None,
        "in_stall_window": False,
    }


def decide_suggestion(row: dict[str, Any]) -> tuple[str, str]:
    hint_types = set(str(value) for value in row.get("existing_hint_types") or [])
    if hint_types & TOUCH_HINT_TYPES:
        return "touch", "existing_or_generated_touch_hint"
    if hint_types & NON_TOUCH_HINT_TYPES:
        return "no_touch", "existing_non_touch_hint"

    if row.get("trajectory_feature_status") != "ok":
        return "no_touch", str(row.get("trajectory_feature_status") or "missing_trajectory")

    support = int(row.get("trajectory_break_support") or 0)
    positive_dvy = float(row.get("trajectory_max_positive_dvy") or 0.0)
    height_reversal = row.get("height_reversal") is True
    if row.get("has_audio") and support > 0 and (positive_dvy > 0.0 or height_reversal):
        return "touch", "audio_onset_near_l2_break"
    return "no_touch", "audio_without_l2_break"


def suggestion_row(row: dict[str, Any], reason: str, decision: str) -> dict[str, Any]:
    return {
        "time_sec": round(float(row["candidate_time_sec"]), 3),
        "decision": decision,
        "review_status": "suggested",
        "source": SUGGESTION_SOURCE,
        "suggestion_reason": reason,
        "has_audio": bool(row.get("has_audio")),
        "audio_strength": row.get("audio_strength"),
        "existing_hint_types": row.get("existing_hint_types") or [],
        "raw_sources": row.get("raw_sources") or [],
        "trajectory_feature_status": row.get("trajectory_feature_status"),
        "trajectory_break_support": row.get("trajectory_break_support"),
        "trajectory_nearest_break_delta_sec": row.get("trajectory_nearest_break_delta_sec"),
        "trajectory_max_positive_dvy": row.get("trajectory_max_positive_dvy"),
        "height_reversal": row.get("height_reversal"),
        "detector_confidence_near_candidate": row.get("detector_confidence_near_candidate"),
    }


def reviewed_or_human_review(row: dict[str, Any]) -> bool:
    return row.get("review_status") in (None, "reviewed") and row.get("source") != SUGGESTION_SOURCE


def merge_suggestions(doc: dict[str, Any], suggestions: list[dict[str, Any]]) -> dict[str, Any]:
    existing = [row for row in doc.get("candidate_reviews", []) if reviewed_or_human_review(row)]
    blocked_times = [float(row["time_sec"]) for row in existing if row.get("time_sec") is not None]
    merged = list(existing)
    for suggestion in suggestions:
        time_sec = float(suggestion["time_sec"])
        if any(abs(time_sec - existing_time) <= 0.05 for existing_time in blocked_times):
            continue
        merged.append(suggestion)
    doc["candidate_reviews"] = sorted(merged, key=lambda row: float(row["time_sec"]))
    doc["candidate_suggestions"] = sorted(suggestions, key=lambda row: float(row["time_sec"]))
    doc["candidate_review_complete"] = False
    doc["review_status"] = "in_progress"
    doc["review_completed_at"] = None
    doc["updated_at"] = now_iso()
    doc["suggestion_prefill"] = {
        "source": SUGGESTION_SOURCE,
        "updated_at": doc["updated_at"],
        "suggestions": len(suggestions),
        "suggested_touches": sum(1 for row in suggestions if row["decision"] == "touch"),
        "suggested_no_touches": sum(1 for row in suggestions if row["decision"] == "no_touch"),
        "note": "Draft only; training/readiness ignores review_status='suggested'.",
    }
    return doc


def suggestions_for_item(
    item: dict[str, Any],
    candidates_payload: dict[str, Any],
    track: TrackFeatures,
    *,
    cluster_gap_sec: float,
    break_tolerance_sec: float,
) -> list[dict[str, Any]]:
    clusters = cluster_candidates(candidates_payload, cluster_gap_sec=cluster_gap_sec)
    suggestions: list[dict[str, Any]] = []
    for index, cluster in enumerate(clusters):
        base = candidate_feature_row(item, cluster, index, clusters)
        featured = attach_row_features(base, track, break_tolerance_sec=break_tolerance_sec)
        decision, reason = decide_suggestion(featured)
        suggestions.append(suggestion_row(featured, reason, decision))
    return suggestions


def prefill_train_suggestions(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_json(args.review_manifest)
    detection_paths = [path for path in args.detections_jsonl if path.exists()]
    tracks = load_detection_tracks(detection_paths, args.threshold) if detection_paths else {}
    rows: list[dict[str, Any]] = []
    for item in manifest.get("items", []):
        video_id = str(item["video_id"])
        if item.get("split") != "train":
            rows.append({"video_id": video_id, "video_name": item["video_name"], "split": item.get("split"), "status": "skipped_non_train"})
            continue

        out_path = label_path(args.labels_dir, video_id)
        if out_path.exists():
            prior = read_json(out_path)
            if prior.get("candidate_review_complete") and not args.force:
                rows.append({"video_id": video_id, "video_name": item["video_name"], "split": "train", "status": "skipped_complete_label"})
                continue

        candidates_file = candidate_path(args.candidates_dir, video_id)
        if not candidates_file.exists():
            raise FileNotFoundError(f"missing candidates for train video {video_id}: {candidates_file}")

        created_at = now_iso()
        doc = load_or_create_doc(out_path, item, created_at=created_at)
        track = track_for_video(item, tracks, min_points=args.min_points, max_track_gap_sec=args.max_track_gap_sec)
        suggestions = suggestions_for_item(
            item,
            read_json(candidates_file),
            track,
            cluster_gap_sec=args.cluster_gap_sec,
            break_tolerance_sec=args.break_tolerance_sec,
        )
        merged = merge_suggestions(doc, suggestions)
        if not args.dry_run:
            write_json(out_path, merged)
        rows.append(
            {
                "video_id": video_id,
                "video_name": item["video_name"],
                "split": "train",
                "status": "would_prefill" if args.dry_run else "prefilled",
                "label_path": str(out_path),
                "suggestions": len(suggestions),
                "suggested_touches": sum(1 for row in suggestions if row["decision"] == "touch"),
                "suggested_no_touches": sum(1 for row in suggestions if row["decision"] == "no_touch"),
                "track_status": track.status,
                "track_points": len(tracks.get(item["video_name"], [])),
                "track_segments": track.segments,
            }
        )

    if any(row.get("split") != "train" and row.get("status") in {"prefilled", "would_prefill"} for row in rows):
        raise AssertionError("non-train prefill attempted")
    summary = {
        "schema_version": 1,
        "created_at": now_iso(),
        "source": SUGGESTION_SOURCE,
        "review_manifest": str(args.review_manifest),
        "candidates_dir": str(args.candidates_dir),
        "labels_dir": str(args.labels_dir),
        "detections_jsonl": [str(path) for path in detection_paths],
        "threshold": args.threshold,
        "cluster_gap_sec": args.cluster_gap_sec,
        "break_tolerance_sec": args.break_tolerance_sec,
        "dry_run": args.dry_run,
        "force": args.force,
        "prefilled": sum(1 for row in rows if row["status"] == "prefilled"),
        "would_prefill": sum(1 for row in rows if row["status"] == "would_prefill"),
        "skipped_complete_label": sum(1 for row in rows if row["status"] == "skipped_complete_label"),
        "skipped_non_train": sum(1 for row in rows if row["status"] == "skipped_non_train"),
        "suggestions": sum(int(row.get("suggestions") or 0) for row in rows),
        "suggested_touches": sum(int(row.get("suggested_touches") or 0) for row in rows),
        "suggested_no_touches": sum(int(row.get("suggested_no_touches") or 0) for row in rows),
        "videos": rows,
    }
    if not args.dry_run:
        write_json(args.labels_dir / "train_suggestion_prefill_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefill TRAIN-only draft candidate decisions from audio+trajectory signals")
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW_MANIFEST)
    parser.add_argument("--candidates-dir", type=Path, default=DEFAULT_CANDIDATES_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--detections-jsonl", type=Path, action="append", default=list(DEFAULT_DETECTIONS_JSONL))
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--cluster-gap-sec", type=float, default=0.04)
    parser.add_argument("--break-tolerance-sec", type=float, default=AUDIO_FUSION_BREAK_TOL_SEC)
    parser.add_argument("--max-track-gap-sec", type=float, default=DEFAULT_MAX_TRACK_GAP_SEC)
    parser.add_argument("--min-points", type=int, default=12)
    parser.add_argument("--force", action="store_true", help="replace prior draft suggestions, but still preserve reviewed rows")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = prefill_train_suggestions(parse_args())
    print(
        json.dumps(
            {
                "prefilled": summary["prefilled"],
                "would_prefill": summary["would_prefill"],
                "skipped_complete_label": summary["skipped_complete_label"],
                "skipped_non_train": summary["skipped_non_train"],
                "suggestions": summary["suggestions"],
                "suggested_touches": summary["suggested_touches"],
                "suggested_no_touches": summary["suggested_no_touches"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"summary: {DEFAULT_LABELS_DIR / 'train_suggestion_prefill_summary.json'}")


if __name__ == "__main__":
    main()
