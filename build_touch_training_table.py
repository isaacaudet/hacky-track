#!/usr/bin/env python3
"""Build the touch-classifier candidate table from muted visual labels.

This is the bridge between Phase 1 labeling and Phase 2 training. It does not
run OWLv2 or train a model. It validates the human-reviewed event files, joins
them to the high-recall candidate hints, and writes a clip-disjoint table that a
fused touch classifier can consume once L2 trajectory features are attached.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "runs/release-27-public/touch_corpus_v1"
DEFAULT_REVIEW_MANIFEST = DEFAULT_CORPUS / "touch_review_manifest.json"
DEFAULT_CANDIDATES_DIR = DEFAULT_CORPUS / "audio_candidates"
DEFAULT_LABELS_DIR = DEFAULT_CORPUS / "visual_touch_labels"
DEFAULT_OUT_DIR = DEFAULT_CORPUS / "touch_training_dataset_v1"
EVENT_TYPES = {"touch", "drop_floor", "stall"}
STRICT_VISUAL_METHOD = "muted_visual_touch_review"


@dataclass(frozen=True)
class CandidateCluster:
    time_sec: float
    has_audio: bool
    audio_strength: float | None
    has_existing_hint: bool
    existing_hint_types: tuple[str, ...]
    raw_sources: tuple[str, ...]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def label_path(labels_dir: Path, video_id: str) -> Path:
    return labels_dir / f"{safe_slug(video_id)}.events.json"


def candidate_path(candidates_dir: Path, video_id: str) -> Path:
    return candidates_dir / f"{safe_slug(video_id)}.touch_candidates.json"


def approved_events(doc: dict[str, Any]) -> list[dict[str, Any]]:
    events = [event for rally in doc.get("rallies", []) for event in rally.get("events", [])]
    out = []
    for event in events:
        if event.get("review_status") != "approved":
            continue
        event_type = event.get("type")
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unsupported approved event type: {event_type}")
        if event.get("time_sec") is None:
            raise ValueError(f"approved {event_type} missing time_sec")
        out.append(event)
    return sorted(out, key=lambda event: float(event["time_sec"]))


def validate_review_doc(path: Path, doc: dict[str, Any], item: dict[str, Any], *, strict_visual: bool) -> None:
    source_video = Path(str(doc.get("source_video") or "")).name
    if source_video != item["video_name"]:
        raise ValueError(f"{path}: source_video {source_video!r} does not match manifest video {item['video_name']!r}")
    if doc.get("split") != item["split"]:
        raise ValueError(f"{path}: split {doc.get('split')!r} does not match manifest split {item['split']!r}")
    if strict_visual:
        if doc.get("annotation_method") != STRICT_VISUAL_METHOD:
            raise ValueError(f"{path}: annotation_method must be {STRICT_VISUAL_METHOD!r}")
        if doc.get("audio_muted_during_review_required") is not True:
            raise ValueError(f"{path}: audio_muted_during_review_required must be true")


def review_complete(doc: dict[str, Any]) -> bool:
    return bool(doc.get("candidate_review_complete"))


def candidate_reviews(doc: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in doc.get("candidate_reviews", []):
        if row.get("review_status") not in (None, "reviewed"):
            continue
        if row.get("time_sec") is None:
            raise ValueError("candidate review missing time_sec")
        decision = str(row.get("decision") or "")
        if decision not in {"touch", "no_touch"}:
            raise ValueError(f"unsupported candidate review decision: {decision}")
        out.append({"time_sec": float(row["time_sec"]), "decision": decision})
    return sorted(out, key=lambda row: float(row["time_sec"]))


def load_manifest(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    doc = read_json(path)
    items = list(doc.get("items", []))
    by_id = {str(item["video_id"]): item for item in items}
    if len(by_id) != len(items):
        raise ValueError("duplicate video_id in review manifest")
    return items, by_id


def validate_label_files(labels_dir: Path, by_id: dict[str, dict[str, Any]]) -> None:
    if not labels_dir.exists():
        return
    known_names = {f"{safe_slug(video_id)}.events.json" for video_id in by_id}
    unknown = sorted(path.name for path in labels_dir.glob("*.events.json") if path.name not in known_names)
    if unknown:
        raise ValueError(f"label files not present in manifest: {unknown}")


def nearest(time_sec: float, times: list[float]) -> tuple[float | None, float | None]:
    if not times:
        return None, None
    match = min(times, key=lambda item: abs(item - time_sec))
    return match, abs(match - time_sec)


def nearest_candidate_review(time_sec: float, reviews: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float | None]:
    if not reviews:
        return None, None
    match = min(reviews, key=lambda row: abs(float(row["time_sec"]) - time_sec))
    return match, abs(float(match["time_sec"]) - time_sec)


def event_windows(events: list[dict[str, Any]], event_type: str) -> list[tuple[float, float]]:
    windows = []
    for event in events:
        if event.get("type") != event_type:
            continue
        start = float(event["time_sec"])
        duration = float(event.get("duration_sec") or 0.0)
        windows.append((start, start + max(0.0, duration)))
    return windows


def min_window_distance(time_sec: float, windows: list[tuple[float, float]]) -> float | None:
    if not windows:
        return None
    distances = []
    for start, end in windows:
        if start <= time_sec <= end:
            distances.append(0.0)
        else:
            distances.append(min(abs(time_sec - start), abs(time_sec - end)))
    return min(distances)


def cluster_candidates(
    payload: dict[str, Any],
    *,
    cluster_gap_sec: float,
    include_generated_event_hints: bool = True,
) -> list[CandidateCluster]:
    raw: list[dict[str, Any]] = []
    for item in payload.get("audio_candidates", []):
        raw.append(
            {
                "time_sec": float(item["time_sec"]),
                "source": item.get("source") or "loose_audio_onset",
                "audio_strength": None if item.get("strength") is None else float(item["strength"]),
                "existing_hint_type": None,
            }
        )
    for item in payload.get("existing_event_hints", []):
        raw.append(
            {
                "time_sec": float(item["time_sec"]),
                "source": item.get("source") or "existing_event_hint",
                "audio_strength": None,
                "existing_hint_type": item.get("event_type"),
            }
        )
    if include_generated_event_hints:
        for item in payload.get("generated_event_hints", []):
            raw.append(
                {
                    "time_sec": float(item["time_sec"]),
                    "source": item.get("source") or "generated_event_hint",
                    "audio_strength": None,
                    "existing_hint_type": item.get("event_type"),
                }
            )
    raw.sort(key=lambda item: float(item["time_sec"]))
    clusters: list[list[dict[str, Any]]] = []
    for item in raw:
        if not clusters or float(item["time_sec"]) - float(clusters[-1][-1]["time_sec"]) > cluster_gap_sec:
            clusters.append([item])
        else:
            clusters[-1].append(item)

    out: list[CandidateCluster] = []
    for cluster in clusters:
        audio_items = [item for item in cluster if item.get("audio_strength") is not None]
        strongest_audio = max((float(item["audio_strength"]) for item in audio_items), default=None)
        has_existing = any(item.get("existing_hint_type") for item in cluster)
        if audio_items:
            time_sec = float(max(audio_items, key=lambda item: float(item["audio_strength"]))["time_sec"])
        else:
            time_sec = float(cluster[0]["time_sec"])
        out.append(
            CandidateCluster(
                time_sec=time_sec,
                has_audio=bool(audio_items),
                audio_strength=strongest_audio,
                has_existing_hint=has_existing,
                existing_hint_types=tuple(sorted({str(item["existing_hint_type"]) for item in cluster if item.get("existing_hint_type")})),
                raw_sources=tuple(sorted({str(item["source"]) for item in cluster})),
            )
        )
    return out


def add_approved_touch_clusters(
    clusters: list[CandidateCluster],
    touch_times: list[float],
    *,
    merge_tolerance_sec: float,
) -> tuple[list[CandidateCluster], int]:
    """Ensure every approved visual touch can become a training row.

    Candidate generation is intentionally high-recall, but it can still miss a
    visually reviewed touch. A human-approved touch is stronger evidence than a
    missing hint, so add a synthetic candidate at that exact time when no
    existing candidate cluster is already essentially colocated with it.
    """
    out = list(clusters)
    injected = 0
    for time_sec in touch_times:
        if any(abs(float(cluster.time_sec) - time_sec) <= merge_tolerance_sec for cluster in out):
            continue
        out.append(
            CandidateCluster(
                time_sec=float(time_sec),
                has_audio=False,
                audio_strength=None,
                has_existing_hint=True,
                existing_hint_types=("touch",),
                raw_sources=("approved_manual_touch",),
            )
        )
        injected += 1
    out.sort(key=lambda cluster: float(cluster.time_sec))
    return out, injected


def candidate_rows_for_video(
    *,
    item: dict[str, Any],
    labels_doc: dict[str, Any],
    candidates_payload: dict[str, Any],
    touch_tolerance_sec: float,
    cluster_gap_sec: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = approved_events(labels_doc)
    reviews = candidate_reviews(labels_doc)
    has_candidate_review_audit = bool(reviews)
    touch_times = [float(event["time_sec"]) for event in events if event["type"] == "touch"]
    drop_times = [float(event["time_sec"]) for event in events if event["type"] == "drop_floor"]
    stall_windows = event_windows(events, "stall")
    clusters = cluster_candidates(
        candidates_payload,
        cluster_gap_sec=cluster_gap_sec,
        include_generated_event_hints=not bool(labels_doc.get("legacy_import")),
    )
    clusters, approved_touch_rows_injected = add_approved_touch_clusters(
        clusters,
        touch_times,
        merge_tolerance_sec=cluster_gap_sec if has_candidate_review_audit else touch_tolerance_sec,
    )
    rows: list[dict[str, Any]] = []
    for index, cluster in enumerate(clusters):
        prev_time = None if index == 0 else clusters[index - 1].time_sec
        next_time = None if index + 1 >= len(clusters) else clusters[index + 1].time_sec
        matched_touch, touch_delta = nearest(cluster.time_sec, touch_times)
        _, drop_delta = nearest(cluster.time_sec, drop_times)
        stall_delta = min_window_distance(cluster.time_sec, stall_windows)
        review, review_delta = nearest_candidate_review(cluster.time_sec, reviews)
        in_stall = bool(stall_delta == 0.0)
        candidate_reviewed = True
        candidate_review_source = "legacy_full_clip_complete"
        is_touch = bool(touch_delta is not None and touch_delta <= touch_tolerance_sec)
        candidate_review_decision = "touch" if is_touch else "no_touch"
        if has_candidate_review_audit:
            if "approved_manual_touch" in cluster.raw_sources:
                candidate_reviewed = True
                candidate_review_source = "approved_manual_event"
                candidate_review_decision = "touch"
                is_touch = True
            else:
                review_match_tolerance = max(cluster_gap_sec, 0.05)
                candidate_reviewed = bool(review_delta is not None and review_delta <= review_match_tolerance)
                if candidate_reviewed and review is not None:
                    candidate_review_source = "candidate_review"
                    candidate_review_decision = str(review["decision"])
                    is_touch = candidate_review_decision == "touch"
                else:
                    candidate_review_source = "missing_candidate_review"
                    candidate_review_decision = None
                    is_touch = False
        label_touch_time = None
        if is_touch:
            if matched_touch is not None and touch_delta is not None and touch_delta <= touch_tolerance_sec:
                label_touch_time = float(matched_touch)
            elif candidate_reviewed and review is not None:
                label_touch_time = float(review["time_sec"])
            else:
                label_touch_time = float(cluster.time_sec)
        rows.append(
            {
                "schema_version": 1,
                "video_id": item["video_id"],
                "video_name": item["video_name"],
                "split": item["split"],
                "candidate_time_sec": round(cluster.time_sec, 6),
                "label_is_touch": is_touch,
                "label_touch_time_sec": None if label_touch_time is None else round(label_touch_time, 6),
                "label_touch_delta_sec": None if touch_delta is None else round(float(touch_delta), 6),
                "candidate_reviewed": candidate_reviewed,
                "candidate_review_decision": candidate_review_decision,
                "candidate_review_source": candidate_review_source,
                "candidate_review_delta_sec": None if review_delta is None else round(float(review_delta), 6),
                "has_audio": cluster.has_audio,
                "audio_strength": None if cluster.audio_strength is None else round(cluster.audio_strength, 6),
                "has_existing_hint": cluster.has_existing_hint,
                "existing_hint_types": list(cluster.existing_hint_types),
                "raw_sources": list(cluster.raw_sources),
                "time_since_prev_candidate_sec": None if prev_time is None else round(cluster.time_sec - prev_time, 6),
                "time_to_next_candidate_sec": None if next_time is None else round(next_time - cluster.time_sec, 6),
                "nearest_drop_delta_sec": None if drop_delta is None else round(float(drop_delta), 6),
                "nearest_stall_delta_sec": None if stall_delta is None else round(float(stall_delta), 6),
                "in_stall_window": in_stall,
                "trajectory_feature_status": "missing_l2_features",
                "trajectory_break_support": None,
                "trajectory_nearest_break_delta_sec": None,
                "trajectory_max_positive_dvy": None,
                "height_reversal": None,
                "detector_confidence_near_candidate": None,
            }
        )
    covered = sum(1 for touch in touch_times if any(abs(cluster.time_sec - touch) <= touch_tolerance_sec for cluster in clusters))
    stats = {
        "approved_events": len(events),
        "approved_touches": len(touch_times),
        "approved_drops": len(drop_times),
        "approved_stalls": sum(1 for event in events if event["type"] == "stall"),
        "candidate_reviews": len(reviews),
        "candidate_rows": len(rows),
        "positive_candidate_rows": sum(1 for row in rows if row["label_is_touch"]),
        "reviewed_candidate_rows": sum(1 for row in rows if row["candidate_reviewed"]),
        "unreviewed_candidate_rows": sum(1 for row in rows if not row["candidate_reviewed"]),
        "negative_candidate_rows_without_review": sum(1 for row in rows if not row["label_is_touch"] and not row["candidate_reviewed"]),
        "touches_with_candidate": covered,
        "touches_without_candidate": len(touch_times) - covered,
        "approved_touch_rows_injected": approved_touch_rows_injected,
    }
    return rows, stats


def split_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        split = str(row["split"])
        item = out.setdefault(split, {"rows": 0, "positives": 0, "negatives": 0, "videos": 0})
        item["rows"] += 1
        item["positives" if row["label_is_touch"] else "negatives"] += 1
    for split in out:
        out[split]["videos"] = len({row["video_id"] for row in rows if row["split"] == split})
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Touch Training Candidate Table",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Review manifest: `{manifest['review_manifest']}`",
        f"- Labels dir: `{manifest['labels_dir']}`",
        f"- Candidates dir: `{manifest['candidates_dir']}`",
        f"- Complete labeled videos used: `{manifest['labeled_videos']}`",
        f"- Draft/incomplete label files skipped: `{manifest['skipped_incomplete_video_count']}`",
        f"- Candidate rows: `{manifest['candidate_rows']}`",
        f"- Positive candidate rows: `{manifest['positive_candidate_rows']}`",
        f"- Candidate reviews recorded: `{manifest['candidate_reviews']}`",
        f"- Reviewed candidate rows: `{manifest['reviewed_candidate_rows']}`",
        f"- Negative candidate rows without review: `{manifest['negative_candidate_rows_without_review']}`",
        f"- Approved touches: `{manifest['approved_touches']}`",
        f"- Touches without any candidate within tolerance: `{manifest['touches_without_candidate']}`",
        f"- Approved touch rows injected: `{manifest['approved_touch_rows_injected']}`",
        "",
        "## Split Counts",
        "",
        "| split | videos | rows | positives | negatives |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split, row in sorted(manifest["split_counts"].items()):
        lines.append(f"| {split} | {row['videos']} | {row['rows']} | {row['positives']} | {row['negatives']} |")
    if manifest["skipped_incomplete_videos"]:
        lines.extend(
            [
                "",
                "## Skipped Draft Labels",
                "",
                "| video | split | reason |",
                "| --- | --- | --- |",
            ]
        )
        for row in manifest["skipped_incomplete_videos"]:
            lines.append(f"| `{row['video_name']}` | {row['split']} | {row['reason']} |")
    lines.extend(
        [
            "",
            "## Per-Video Coverage",
            "",
        "| video | split | touches | covered | missing | reviews | candidates | reviewed | positives |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in manifest["videos"]:
        lines.append(
            f"| `{row['video_name']}` | {row['split']} | {row['approved_touches']} | "
            f"{row['touches_with_candidate']} | {row['touches_without_candidate']} | "
            f"{row['candidate_reviews']} | {row['candidate_rows']} | "
            f"{row['reviewed_candidate_rows']} | {row['positive_candidate_rows']} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- When `candidate_reviews` are present, explicit cue-level review decisions are the training labels.",
            "- Legacy/imported label files without `candidate_reviews` fall back to matching approved muted-visual touch events by time.",
            "- Audio candidates are high-recall hints, not ground truth.",
            "- Draft label files are ignored until `candidate_review_complete=true`, so unreviewed hints do not become false negatives.",
            "- New label files should include `candidate_reviews`; unreviewed negative candidates are skipped in strict mode.",
            "- Trajectory feature columns are present but intentionally null until L2 feature extraction is attached.",
            "- `test_frozen` rows are exported separately and must not be used for training or tuning.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_training_table(args: argparse.Namespace) -> dict[str, Any]:
    items, by_id = load_manifest(args.review_manifest)
    labels_dir = args.labels_dir.resolve()
    candidates_dir = args.candidates_dir.resolve()
    out_dir = args.out_dir.resolve()
    validate_label_files(labels_dir, by_id)

    train_val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    video_stats: list[dict[str, Any]] = []
    labeled_videos = 0
    skipped_incomplete_videos: list[dict[str, Any]] = []
    for item in items:
        video_id = str(item["video_id"])
        labels_path = label_path(labels_dir, video_id)
        if not labels_path.exists():
            continue
        labels_doc = read_json(labels_path)
        validate_review_doc(labels_path, labels_doc, item, strict_visual=not args.allow_non_visual_labels)
        if not review_complete(labels_doc) and not args.allow_incomplete_labels:
            skipped_incomplete_videos.append(
                {
                    "video_id": video_id,
                    "video_name": item["video_name"],
                    "split": item["split"],
                    "label_path": str(labels_path),
                    "reason": "candidate_review_complete is not true",
                }
            )
            continue
        candidates_file = candidate_path(candidates_dir, video_id)
        if not candidates_file.exists():
            raise FileNotFoundError(f"missing candidates for labeled video {video_id}: {candidates_file}")
        candidates_payload = read_json(candidates_file)
        rows, stats = candidate_rows_for_video(
            item=item,
            labels_doc=labels_doc,
            candidates_payload=candidates_payload,
            touch_tolerance_sec=args.touch_tolerance_sec,
            cluster_gap_sec=args.cluster_gap_sec,
        )
        if stats["negative_candidate_rows_without_review"] and not args.allow_incomplete_labels:
            skipped_incomplete_videos.append(
                {
                    "video_id": video_id,
                    "video_name": item["video_name"],
                    "split": item["split"],
                    "label_path": str(labels_path),
                    "reason": f"{stats['negative_candidate_rows_without_review']} negative candidates lack candidate_reviews",
                }
            )
            continue
        labeled_videos += 1
        target = test_rows if item["split"] == "test_frozen" else train_val_rows
        target.extend(rows)
        video_stats.append({"video_id": video_id, "video_name": item["video_name"], "split": item["split"], **stats})

    all_rows = train_val_rows + test_rows
    if args.require_labels and not labeled_videos:
        raise ValueError(f"no reviewed labels found in {labels_dir}")
    if any(row["split"] == "test_frozen" for row in train_val_rows):
        raise AssertionError("test_frozen row leaked into train/validation rows")

    status = "ready_for_feature_attachment" if labeled_videos else "waiting_for_visual_labels"
    if all_rows and all(row["trajectory_feature_status"] == "missing_l2_features" for row in all_rows):
        status = "waiting_for_l2_features"
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "review_manifest": str(args.review_manifest),
        "labels_dir": str(labels_dir),
        "candidates_dir": str(candidates_dir),
        "touch_tolerance_sec": args.touch_tolerance_sec,
        "cluster_gap_sec": args.cluster_gap_sec,
        "candidate_rows_jsonl": str(out_dir / "touch_training_candidates.jsonl"),
        "candidate_rows_csv": str(out_dir / "touch_training_candidates.csv"),
        "test_frozen_jsonl": str(out_dir / "touch_training_test_frozen.jsonl"),
        "labeled_videos": labeled_videos,
        "skipped_incomplete_videos": skipped_incomplete_videos,
        "skipped_incomplete_video_count": len(skipped_incomplete_videos),
        "candidate_rows": len(all_rows),
        "train_val_candidate_rows": len(train_val_rows),
        "test_frozen_candidate_rows": len(test_rows),
        "positive_candidate_rows": sum(1 for row in all_rows if row["label_is_touch"]),
        "approved_touches": sum(row["approved_touches"] for row in video_stats),
        "touches_with_candidate": sum(row["touches_with_candidate"] for row in video_stats),
        "touches_without_candidate": sum(row["touches_without_candidate"] for row in video_stats),
        "approved_touch_rows_injected": sum(row["approved_touch_rows_injected"] for row in video_stats),
        "candidate_reviews": sum(row["candidate_reviews"] for row in video_stats),
        "reviewed_candidate_rows": sum(row["reviewed_candidate_rows"] for row in video_stats),
        "unreviewed_candidate_rows": sum(row["unreviewed_candidate_rows"] for row in video_stats),
        "negative_candidate_rows_without_review": sum(row["negative_candidate_rows_without_review"] for row in video_stats),
        "split_counts": split_counts(all_rows),
        "videos": video_stats,
        "training_guardrails": {
            "test_frozen_excluded_from_train_val": True,
            "strict_visual_labels_required": not args.allow_non_visual_labels,
            "complete_clip_review_required": not args.allow_incomplete_labels,
            "candidate_review_audit_required_for_new_labels": not args.allow_incomplete_labels,
            "trajectory_features_required_before_fused_training": True,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "touch_training_candidates.jsonl", train_val_rows)
    write_csv(out_dir / "touch_training_candidates.csv", train_val_rows)
    write_jsonl(out_dir / "touch_training_test_frozen.jsonl", test_rows)
    write_json(out_dir / "touch_training_dataset_manifest.json", manifest)
    write_report(out_dir / "touch_training_dataset_report.md", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build touch-classifier candidate table from muted visual labels")
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW_MANIFEST)
    parser.add_argument("--candidates-dir", type=Path, default=DEFAULT_CANDIDATES_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--touch-tolerance-sec", type=float, default=0.20)
    parser.add_argument("--cluster-gap-sec", type=float, default=0.04)
    parser.add_argument("--allow-non-visual-labels", action="store_true")
    parser.add_argument("--allow-incomplete-labels", action="store_true", help="consume draft label files; smoke/debug only")
    parser.add_argument("--require-labels", action="store_true")
    return parser.parse_args()


def main() -> None:
    manifest = build_training_table(parse_args())
    print(f"manifest: {manifest['candidate_rows_jsonl']}")
    print(f"report:   {Path(manifest['candidate_rows_jsonl']).parent / 'touch_training_dataset_report.md'}")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "labeled_videos": manifest["labeled_videos"],
                "candidate_rows": manifest["candidate_rows"],
                "positive_candidate_rows": manifest["positive_candidate_rows"],
                "approved_touches": manifest["approved_touches"],
                "touches_without_candidate": manifest["touches_without_candidate"],
                "split_counts": manifest["split_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
