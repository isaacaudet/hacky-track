#!/usr/bin/env python3
"""Audit visual touch-label readiness for the fixed-OWLv2 touch pipeline.

This is a read-only checklist generator. It answers the immediate Phase 1
question: which clips still need muted visual review before strict training and
frozen-test evaluation can run?
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_touch_training_table import (
    DEFAULT_CANDIDATES_DIR,
    DEFAULT_CORPUS,
    DEFAULT_LABELS_DIR,
    DEFAULT_REVIEW_MANIFEST,
    STRICT_VISUAL_METHOD,
    approved_events,
    candidate_path,
    candidate_reviews,
    cluster_candidates,
    label_path,
    read_json,
    validate_review_doc,
    write_json,
)


DEFAULT_OUT_DIR = DEFAULT_CORPUS
STATUS_ORDER = {
    "complete_ready": 0,
    "complete_but_unchecked_hints": 1,
    "draft_incomplete": 2,
    "missing_label_file": 3,
    "missing_candidate_file": 4,
    "malformed_label_file": 5,
}


def write_markdown(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Touch Label Readiness",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Review manifest: `{manifest['review_manifest']}`",
        f"- Labels dir: `{manifest['labels_dir']}`",
        f"- Candidates dir: `{manifest['candidates_dir']}`",
        f"- Complete ready videos: `{manifest['summary']['complete_ready_videos']}` / `{manifest['summary']['videos']}`",
        f"- Non-test ready videos: `{manifest['summary']['non_test_complete_ready_videos']}`",
        f"- Frozen-test ready videos: `{manifest['summary']['frozen_test_complete_ready_videos']}`",
        f"- Total hints: `{manifest['summary']['total_hints']}`",
        f"- Unchecked hints: `{manifest['summary']['unchecked_hints']}`",
        f"- Likely/model unchecked hints: `{manifest['summary']['likely_unchecked_hints']}`",
        f"- Audio-tail unchecked hints: `{manifest['summary']['audio_only_unchecked_hints']}`",
        f"- Candidate reviews recorded: `{manifest['summary']['candidate_reviews']}`",
        "",
        "## Next Clips To Review",
        "",
        "| priority | video | split | status | hints | unchecked | likely left | audio-tail left | reason |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in manifest["next_clips"]:
        lines.append(
            f"| {row['priority']} | `{row['video_name']}` | {row['split']} | {row['status']} | "
            f"{row['total_hint_count']} | {row['unchecked_hint_count']} | "
            f"{row['likely_unchecked_hint_count']} | {row['audio_only_unchecked_hint_count']} | {row['reason']} |"
        )
    if not manifest["next_clips"]:
        lines.append("|  |  |  |  |  |  |  |  | All manifest clips are complete-ready. |")

    lines.extend(
        [
            "",
            "## Split Summary",
            "",
            "| split | videos | ready | missing | draft | unchecked | malformed | hints | unchecked hints | likely left | audio-tail left |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split, row in sorted(manifest["splits"].items()):
        lines.append(
            f"| {split} | {row['videos']} | {row['complete_ready']} | {row['missing_label_file']} | "
            f"{row['draft_incomplete']} | {row['complete_but_unchecked_hints']} | {row['malformed_label_file']} | "
            f"{row['total_hints']} | {row['unchecked_hints']} | "
            f"{row['likely_unchecked_hints']} | {row['audio_only_unchecked_hints']} |"
        )

    lines.extend(
        [
            "",
            "## Per-Video Checklist",
            "",
            "| video | split | status | hints | checked | unchecked | likely left | audio-tail left | approved touches | candidate reviews | label file |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in manifest["videos"]:
        lines.append(
            f"| `{row['video_name']}` | {row['split']} | {row['status']} | "
            f"{row['total_hint_count']} | {row['checked_hint_count']} | {row['unchecked_hint_count']} | "
            f"{row['likely_unchecked_hint_count']} | {row['audio_only_unchecked_hint_count']} | "
            f"{row['approved_touches']} | {row['candidate_review_count']} | `{row['label_path']}` |"
        )

    lines.extend(
        [
            "",
            "## Commands",
            "",
            "```bash",
            "python3 touch_review_app.py",
            "python3 run_touch_pipeline.py",
            "python3 run_touch_pipeline.py --export-owlv2-detections",
            "```",
            "",
            "Notes:",
            "- Hints are audio/on-existing-event prompts only; labels must come from muted visual review.",
            "- A clip is ready only when `candidate_review_complete=true` and every clustered candidate moment has an explicit `candidate_reviews` decision.",
            f"- Strict labels must use `annotation_method={STRICT_VISUAL_METHOD}` and `audio_muted_during_review_required=true`.",
            "- Frozen-test clips are prioritized because release-gate evaluation requires visually reviewed held-out rows.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_items(path: Path) -> list[dict[str, Any]]:
    doc = read_json(path)
    items = list(doc.get("items", []))
    seen: set[str] = set()
    for item in items:
        video_id = str(item["video_id"])
        if video_id in seen:
            raise ValueError(f"duplicate video_id in review manifest: {video_id}")
        seen.add(video_id)
    return items


def clustered_review_hints(
    payload: dict[str, Any],
    *,
    cluster_gap_sec: float,
    include_generated_event_hints: bool = True,
) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    for cluster in cluster_candidates(
        payload,
        cluster_gap_sec=cluster_gap_sec,
        include_generated_event_hints=include_generated_event_hints,
    ):
        hints.append(
            {
                "time_sec": float(cluster.time_sec),
                "source": "+".join(cluster.raw_sources),
                "has_audio": cluster.has_audio,
                "audio_strength": cluster.audio_strength,
                "has_existing_hint": cluster.has_existing_hint,
                "existing_hint_types": list(cluster.existing_hint_types),
                "raw_sources": list(cluster.raw_sources),
            }
        )
    return hints


def count_checked_hints(hints: list[dict[str, Any]], reviews: list[dict[str, Any]], *, tolerance_sec: float) -> int:
    count = 0
    review_times = [float(row["time_sec"]) for row in reviews]
    for hint in hints:
        hint_time = float(hint["time_sec"])
        if any(abs(hint_time - review_time) <= tolerance_sec for review_time in review_times):
            count += 1
    return count


def likely_touch_hint(hint: dict[str, Any]) -> bool:
    return "touch" in {str(item) for item in hint.get("existing_hint_types", [])}


def hint_review_stats(hints: list[dict[str, Any]], reviews: list[dict[str, Any]], *, tolerance_sec: float) -> dict[str, int]:
    review_times = [float(row["time_sec"]) for row in reviews]
    stats = {
        "checked_hint_count": 0,
        "unchecked_hint_count": 0,
        "likely_hint_count": 0,
        "likely_checked_hint_count": 0,
        "likely_unchecked_hint_count": 0,
        "audio_only_hint_count": 0,
        "audio_only_checked_hint_count": 0,
        "audio_only_unchecked_hint_count": 0,
    }
    for hint in hints:
        checked = any(abs(float(hint["time_sec"]) - review_time) <= tolerance_sec for review_time in review_times)
        likely = likely_touch_hint(hint)
        if checked:
            stats["checked_hint_count"] += 1
        else:
            stats["unchecked_hint_count"] += 1
        if likely:
            stats["likely_hint_count"] += 1
            stats["likely_checked_hint_count" if checked else "likely_unchecked_hint_count"] += 1
        else:
            stats["audio_only_hint_count"] += 1
            stats["audio_only_checked_hint_count" if checked else "audio_only_unchecked_hint_count"] += 1
    return stats


def empty_split_row() -> dict[str, int]:
    return {
        "videos": 0,
        "complete_ready": 0,
        "complete_but_unchecked_hints": 0,
        "draft_incomplete": 0,
        "missing_label_file": 0,
        "missing_candidate_file": 0,
        "malformed_label_file": 0,
        "total_hints": 0,
        "unchecked_hints": 0,
        "likely_hints": 0,
        "likely_unchecked_hints": 0,
        "audio_only_hints": 0,
        "audio_only_unchecked_hints": 0,
        "candidate_reviews": 0,
        "approved_touches": 0,
    }


def classify_video(
    *,
    item: dict[str, Any],
    labels_dir: Path,
    candidates_dir: Path,
    review_match_tolerance_sec: float,
    candidate_cluster_gap_sec: float = 0.04,
) -> dict[str, Any]:
    video_id = str(item["video_id"])
    labels_file = label_path(labels_dir, video_id)
    candidates_file = candidate_path(candidates_dir, video_id)
    row = {
        "video_id": video_id,
        "video_name": item["video_name"],
        "split": item["split"],
        "label_path": str(labels_file),
        "candidate_path": str(candidates_file),
        "status": "missing_label_file",
        "ready_for_training_table": False,
        "error": None,
        "audio_candidate_count": 0,
        "existing_event_hint_count": 0,
        "generated_event_hint_count": 0,
        "total_hint_count": 0,
        "checked_hint_count": 0,
        "unchecked_hint_count": 0,
        "likely_hint_count": 0,
        "likely_checked_hint_count": 0,
        "likely_unchecked_hint_count": 0,
        "audio_only_hint_count": 0,
        "audio_only_checked_hint_count": 0,
        "audio_only_unchecked_hint_count": 0,
        "candidate_review_count": 0,
        "candidate_review_complete": False,
        "review_status": None,
        "approved_events": 0,
        "approved_touches": 0,
        "approved_drops": 0,
        "approved_stalls": 0,
    }

    try:
        if not candidates_file.exists():
            row["status"] = "missing_candidate_file"
            row["error"] = f"missing candidate file: {candidates_file}"
            return row
        candidate_payload = read_json(candidates_file)
        row["audio_candidate_count"] = len(candidate_payload.get("audio_candidates", []))
        row["existing_event_hint_count"] = len(candidate_payload.get("existing_event_hints", []))
        row["generated_event_hint_count"] = len(candidate_payload.get("generated_event_hints", []))

        if not labels_file.exists():
            hints = clustered_review_hints(candidate_payload, cluster_gap_sec=candidate_cluster_gap_sec)
            row["total_hint_count"] = len(hints)
            row.update(hint_review_stats(hints, [], tolerance_sec=review_match_tolerance_sec))
            return row

        labels_doc = read_json(labels_file)
        validate_review_doc(labels_file, labels_doc, item, strict_visual=True)
        include_generated = not bool(labels_doc.get("legacy_import"))
        hints = clustered_review_hints(
            candidate_payload,
            cluster_gap_sec=candidate_cluster_gap_sec,
            include_generated_event_hints=include_generated,
        )
        row["total_hint_count"] = len(hints)
        events = approved_events(labels_doc)
        reviews = candidate_reviews(labels_doc)
        row.update(hint_review_stats(hints, reviews, tolerance_sec=review_match_tolerance_sec))
        row["candidate_review_count"] = len(reviews)
        row["candidate_review_complete"] = bool(labels_doc.get("candidate_review_complete"))
        row["review_status"] = labels_doc.get("review_status")
        row["approved_events"] = len(events)
        row["approved_touches"] = sum(1 for event in events if event["type"] == "touch")
        row["approved_drops"] = sum(1 for event in events if event["type"] == "drop_floor")
        row["approved_stalls"] = sum(1 for event in events if event["type"] == "stall")
        if not row["candidate_review_complete"]:
            row["status"] = "draft_incomplete"
        elif row["unchecked_hint_count"]:
            row["status"] = "complete_but_unchecked_hints"
        else:
            row["status"] = "complete_ready"
            row["ready_for_training_table"] = True
        return row
    except Exception as exc:
        row["status"] = "malformed_label_file"
        row["ready_for_training_table"] = False
        row["error"] = str(exc)
        return row


def make_next_clips(videos: list[dict[str, Any]], *, min_non_test_videos: int, min_frozen_test_videos: int) -> list[dict[str, Any]]:
    non_test_ready = sum(1 for row in videos if row["split"] != "test_frozen" and row["status"] == "complete_ready")
    frozen_ready = sum(1 for row in videos if row["split"] == "test_frozen" and row["status"] == "complete_ready")
    next_clips = []
    for row in videos:
        if row["status"] == "complete_ready":
            continue
        if row["split"] == "test_frozen" and frozen_ready < min_frozen_test_videos:
            priority = 0
            reason = "frozen-test visual labels are required for release-gate evaluation"
        elif row["split"] != "test_frozen" and non_test_ready < min_non_test_videos:
            priority = 1
            reason = f"need at least {min_non_test_videos} complete train/validation clips before strict training"
        elif row["split"] == "test_frozen":
            priority = 2
            reason = "more frozen-test coverage improves final gate confidence"
        else:
            priority = 3
            reason = "more visual labels improve leave-clips-out training coverage"
        item = dict(row)
        item["priority"] = priority
        item["reason"] = reason
        next_clips.append(item)
    return sorted(
        next_clips,
        key=lambda row: (
            int(row["priority"]),
            STATUS_ORDER.get(str(row["status"]), 99),
            int(row["total_hint_count"]),
            str(row["video_name"]),
        ),
    )


def build_readiness_report(args: argparse.Namespace) -> dict[str, Any]:
    review_manifest = args.review_manifest.resolve()
    labels_dir = args.labels_dir.resolve()
    candidates_dir = args.candidates_dir.resolve()
    out_dir = args.out_dir.resolve()
    candidate_cluster_gap_sec = float(getattr(args, "candidate_cluster_gap_sec", 0.04))
    items = load_items(review_manifest)
    videos = [
        classify_video(
            item=item,
            labels_dir=labels_dir,
            candidates_dir=candidates_dir,
            review_match_tolerance_sec=args.review_match_tolerance_sec,
            candidate_cluster_gap_sec=candidate_cluster_gap_sec,
        )
        for item in items
    ]

    splits: dict[str, dict[str, int]] = {}
    for row in videos:
        split_row = splits.setdefault(str(row["split"]), empty_split_row())
        split_row["videos"] += 1
        status = str(row["status"])
        if status in split_row:
            split_row[status] += 1
        split_row["total_hints"] += int(row["total_hint_count"])
        split_row["unchecked_hints"] += int(row["unchecked_hint_count"])
        split_row["likely_hints"] += int(row["likely_hint_count"])
        split_row["likely_unchecked_hints"] += int(row["likely_unchecked_hint_count"])
        split_row["audio_only_hints"] += int(row["audio_only_hint_count"])
        split_row["audio_only_unchecked_hints"] += int(row["audio_only_unchecked_hint_count"])
        split_row["candidate_reviews"] += int(row["candidate_review_count"])
        split_row["approved_touches"] += int(row["approved_touches"])

    non_test_ready = sum(1 for row in videos if row["split"] != "test_frozen" and row["status"] == "complete_ready")
    frozen_ready = sum(1 for row in videos if row["split"] == "test_frozen" and row["status"] == "complete_ready")
    complete_ready = sum(1 for row in videos if row["status"] == "complete_ready")
    minimum_ready = bool(non_test_ready >= args.min_non_test_videos and frozen_ready >= args.min_frozen_test_videos)
    status = "minimum_label_set_ready" if minimum_ready else "waiting_for_visual_labels"
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "review_manifest": str(review_manifest),
        "labels_dir": str(labels_dir),
        "candidates_dir": str(candidates_dir),
        "review_match_tolerance_sec": args.review_match_tolerance_sec,
        "candidate_cluster_gap_sec": candidate_cluster_gap_sec,
        "min_non_test_videos": args.min_non_test_videos,
        "min_frozen_test_videos": args.min_frozen_test_videos,
        "readiness_json": str(out_dir / "touch_label_readiness.json"),
        "readiness_report": str(out_dir / "touch_label_readiness.md"),
        "summary": {
            "videos": len(videos),
            "complete_ready_videos": complete_ready,
            "non_test_complete_ready_videos": non_test_ready,
            "frozen_test_complete_ready_videos": frozen_ready,
            "missing_label_files": sum(1 for row in videos if row["status"] == "missing_label_file"),
            "draft_incomplete_videos": sum(1 for row in videos if row["status"] == "draft_incomplete"),
            "complete_but_unchecked_videos": sum(1 for row in videos if row["status"] == "complete_but_unchecked_hints"),
            "malformed_label_files": sum(1 for row in videos if row["status"] == "malformed_label_file"),
            "total_hints": sum(int(row["total_hint_count"]) for row in videos),
            "checked_hints": sum(int(row["checked_hint_count"]) for row in videos),
            "unchecked_hints": sum(int(row["unchecked_hint_count"]) for row in videos),
            "likely_hints": sum(int(row["likely_hint_count"]) for row in videos),
            "likely_checked_hints": sum(int(row["likely_checked_hint_count"]) for row in videos),
            "likely_unchecked_hints": sum(int(row["likely_unchecked_hint_count"]) for row in videos),
            "audio_only_hints": sum(int(row["audio_only_hint_count"]) for row in videos),
            "audio_only_checked_hints": sum(int(row["audio_only_checked_hint_count"]) for row in videos),
            "audio_only_unchecked_hints": sum(int(row["audio_only_unchecked_hint_count"]) for row in videos),
            "candidate_reviews": sum(int(row["candidate_review_count"]) for row in videos),
            "approved_touches": sum(int(row["approved_touches"]) for row in videos),
            "minimum_strict_prerequisites_met": minimum_ready,
        },
        "splits": splits,
        "next_clips": make_next_clips(
            videos,
            min_non_test_videos=args.min_non_test_videos,
            min_frozen_test_videos=args.min_frozen_test_videos,
        ),
        "videos": sorted(videos, key=lambda row: (str(row["split"]), STATUS_ORDER.get(str(row["status"]), 99), str(row["video_name"]))),
    }
    write_json(out_dir / "touch_label_readiness.json", manifest)
    write_markdown(out_dir / "touch_label_readiness.md", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit muted visual touch-label readiness")
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW_MANIFEST)
    parser.add_argument("--candidates-dir", type=Path, default=DEFAULT_CANDIDATES_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--review-match-tolerance-sec", type=float, default=0.05)
    parser.add_argument("--candidate-cluster-gap-sec", type=float, default=0.04)
    parser.add_argument("--min-non-test-videos", type=int, default=3)
    parser.add_argument("--min-frozen-test-videos", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    manifest = build_readiness_report(parse_args())
    print(f"json:   {manifest['readiness_json']}")
    print(f"report: {manifest['readiness_report']}")
    print(json.dumps({"status": manifest["status"], **manifest["summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
