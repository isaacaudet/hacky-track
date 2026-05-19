#!/usr/bin/env python3
"""Release-quality evaluation for Hacky Track reviewed labels.

This is intentionally separate from the prototype detector metrics. The model
trainer can report patch-classifier quality, but release readiness depends on
reviewed event behavior: touches, drops, stalls, duplicates, side/type labels,
and trick candidates on clean held-out review data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_REVIEWS = ROOT / "reviews"
DEFAULT_OUT = ROOT / "outputs" / "release_evaluation"
DEFAULT_BATCH = ROOT / "outputs" / "review_batches" / "latest_review_batch.json"
KINDS = ("touch", "drop_floor", "stall", "around_the_world")
DECIDED_STATUSES = {"approved", "rejected", "missing"}
SIDE_VALUES = {"left", "right", "center"}
TARGETS = {
    "ball_event_center_pass_rate": 0.95,
    "touch_precision": 0.90,
    "touch_recall": 0.85,
    "duplicate_touch_rate": 0.05,
    "drop_floor_precision": 0.90,
    "drop_floor_recall": 0.90,
    "stall_precision": 0.85,
    "stall_recall": 0.80,
    "side_accuracy": 0.85,
    "contact_type_accuracy": 0.85,
    "knee_precision": 0.80,
    "knee_min_examples": 20,
    "trick_precision": 0.80,
}


@dataclass(frozen=True)
class BatchIndex:
    batch_id: str | None
    keys: frozenset[str]
    videos: frozenset[str]
    path: Path | None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def portable_path(path: Any, *, base: Path = ROOT) -> str | None:
    if path in (None, ""):
        return None
    candidate = Path(str(path)).expanduser()
    try:
        return str(candidate.resolve().relative_to(base.resolve()))
    except (OSError, ValueError):
        pass
    if candidate.is_absolute():
        return candidate.name
    return str(candidate)


def percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def review_key(source_video: Any, item_id: Any) -> str:
    return f"{source_video or ''}::{item_id or ''}"


def norm_kind(item: dict[str, Any]) -> str:
    kind = str(item.get("kind") or "touch")
    return kind if kind in KINDS else "touch"


def norm_status(item: dict[str, Any]) -> str:
    return str(item.get("status") or "pending")


def has_text_flag(item: dict[str, Any], needle: str) -> bool:
    text = " ".join(
        [
            str(item.get("note") or ""),
            str(item.get("review_evidence") or ""),
            " ".join(str(tag) for tag in item.get("review_tags", []) or []),
        ]
    ).lower()
    return needle.lower() in text


def compatible_contact_type(detector_type: Any, reviewed_type: Any) -> bool:
    detector = str(detector_type or "").strip()
    reviewed = str(reviewed_type or "").strip()
    if not detector or not reviewed:
        return True
    if detector == reviewed:
        return True
    compatible = {
        "foot_candidate": "foot",
        "knee_candidate": "knee",
        "ground_candidate": "ground",
    }
    return compatible.get(detector) == reviewed


def load_batch(path: Path | None) -> BatchIndex:
    if not path or not path.exists():
        return BatchIndex(None, frozenset(), frozenset(), None)
    doc = read_json(path)
    items = doc.get("items", [])
    keys = {
        review_key(item.get("source_video"), item.get("review_item_id"))
        for item in items
        if item.get("review_item_id")
    }
    videos = {
        str(item.get("source_video"))
        for item in items
        if str(item.get("source_video") or "").strip()
    }
    return BatchIndex(doc.get("batch_id"), frozenset(keys), frozenset(videos), path)


def label_tier(item: dict[str, Any], source_video: str, batch: BatchIndex) -> str:
    if norm_status(item) not in DECIDED_STATUSES:
        return "pending"
    key = review_key(source_video, item.get("id"))
    if key in batch.keys:
        return "clean_current_batch"
    if item.get("in_review_batch") and (not batch.batch_id or item.get("batch_id") == batch.batch_id):
        return "clean_current_batch"
    if str(item.get("review_decision_source") or "").strip():
        return "exploratory_reviewed"
    if item.get("reviewed_at") or item.get("reviewed_by"):
        return "exploratory_reviewed"
    return "exploratory_reviewed"


def reviewed_rows(review_paths: list[Path], batch: BatchIndex) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for review_path in review_paths:
        doc = read_json(review_path)
        source_video = str(doc.get("source_video") or review_path.stem)
        for item in doc.get("items", []):
            status = norm_status(item)
            kind = norm_kind(item)
            row = {
                "review_file": portable_path(review_path),
                "source_video": source_video,
                "item_id": item.get("id"),
                "source": str(item.get("source") or "candidate"),
                "kind": kind,
                "status": status,
                "label_tier": label_tier(item, source_video, batch),
                "time_sec": item.get("time_sec"),
                "start_sec": item.get("start_sec"),
                "end_sec": item.get("end_sec"),
                "duration_sec": item.get("duration_sec"),
                "contact_side": item.get("contact_side", "unknown"),
                "detector_contact_side": item.get("detector_contact_side"),
                "side_confidence": item.get("side_confidence"),
                "side_source": item.get("side_source"),
                "side_uncertainty_reason": item.get("side_uncertainty_reason"),
                "contact_type": item.get("contact_type", "unknown"),
                "detector_contact_type": item.get("detector_contact_type"),
                "trick_label": item.get("trick_label", ""),
                "confidence": item.get("confidence"),
                "review_tags": item.get("review_tags", []),
                "note": item.get("note", ""),
                "duplicate_rejection": kind == "touch" and status == "rejected" and has_text_flag(item, "duplicate"),
                "ambiguous_contact": has_text_flag(item, "ambiguous"),
                "has_stall_window": kind == "stall" and item.get("end_sec") is not None,
            }
            rows.append(row)
    return rows


def deterministic_video_splits(videos: list[str], seed: int) -> dict[str, str]:
    ordered = sorted(set(videos))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    count = len(ordered)
    if count == 0:
        return {}
    if count == 1:
        return {ordered[0]: "test"}
    if count == 2:
        return {ordered[0]: "train", ordered[1]: "test"}
    test_count = max(1, round(count * 0.15))
    val_count = max(1, round(count * 0.15))
    if test_count + val_count >= count:
        test_count = 1
        val_count = 1
    train_count = count - test_count - val_count
    result: dict[str, str] = {}
    for idx, video in enumerate(ordered):
        if idx < train_count:
            split = "train"
        elif idx < train_count + val_count:
            split = "validation"
        else:
            split = "test"
        result[video] = split
    return result


def split_summary(rows: list[dict[str, Any]], assignments: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        split_videos = sorted(video for video, assigned in assignments.items() if assigned == split)
        split_rows = [row for row in rows if assignments.get(str(row.get("source_video"))) == split]
        clean_rows = [row for row in split_rows if row["label_tier"] == "clean_current_batch"]
        out[split] = {
            "videos": split_videos,
            "reviewed_items": len(split_rows),
            "clean_items": len(clean_rows),
            "decided_clean_items": sum(1 for row in clean_rows if row["status"] in DECIDED_STATUSES),
        }
    return out


def safe_div(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return round(float(numerator) / float(denominator), 4)


def metric_rows(rows: list[dict[str, Any]], split: str | None = None, *, tier: str = "clean_current_batch") -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("label_tier") == tier and row.get("status") in DECIDED_STATUSES and (split is None or row.get("split") == split)
    ]


def kind_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for kind in KINDS:
        kind_rows = [row for row in rows if row["kind"] == kind]
        approved = sum(1 for row in kind_rows if row["source"] == "candidate" and row["status"] == "approved")
        rejected = sum(1 for row in kind_rows if row["source"] == "candidate" and row["status"] == "rejected")
        missing = sum(1 for row in kind_rows if row["source"] == "manual" and row["status"] == "missing")
        metrics[kind] = {
            "approved_candidates": approved,
            "rejected_candidates": rejected,
            "manual_missing": missing,
            "reviewed_decisions": len(kind_rows),
            "precision": safe_div(approved, approved + rejected),
            "recall": safe_div(approved, approved + missing),
        }
    return metrics


def classification_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted_candidates = [row for row in rows if row["source"] == "candidate" and row["status"] == "approved"]
    side_rows = [
        row
        for row in accepted_candidates
        if not row.get("ambiguous_contact")
        and str(row.get("contact_side") or "unknown") in SIDE_VALUES
        and str(row.get("detector_contact_side") or "unknown") not in {"", "unknown"}
    ]
    side_correct = sum(1 for row in side_rows if row.get("contact_side") == row.get("detector_contact_side"))
    type_rows = [
        row
        for row in accepted_candidates
        if not row.get("ambiguous_contact")
        and str(row.get("contact_type") or "unknown") not in {"", "unknown"}
        and str(row.get("detector_contact_type") or "unknown") not in {"", "unknown"}
    ]
    type_correct = sum(1 for row in type_rows if compatible_contact_type(row.get("detector_contact_type"), row.get("contact_type")))
    unknown_side_rows = [
        row
        for row in accepted_candidates
        if row.get("ambiguous_contact") and str(row.get("detector_contact_side") or "unknown") == "unknown"
    ]
    return {
        "side": {
            "evaluated": len(side_rows),
            "correct": side_correct,
            "accuracy": safe_div(side_correct, len(side_rows)),
            "ambiguous_kept_unknown": len(unknown_side_rows),
        },
        "contact_type": {
            "evaluated": len(type_rows),
            "correct": type_correct,
            "accuracy": safe_div(type_correct, len(type_rows)),
        },
    }


def duplicate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    approved_touches = sum(1 for row in rows if row["kind"] == "touch" and row["source"] == "candidate" and row["status"] == "approved")
    duplicate_rejections = sum(1 for row in rows if row.get("duplicate_rejection"))
    return {
        "approved_touches": approved_touches,
        "duplicate_touch_rejections": duplicate_rejections,
        "duplicate_touch_rate_vs_accepted": safe_div(duplicate_rejections, approved_touches),
    }


def release_condition_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    knee_rows = [
        row
        for row in rows
        if str(row.get("contact_type") or "") == "knee"
        or str(row.get("detector_contact_type") or "") == "knee"
        or str(row.get("detector_contact_type") or "") == "knee_candidate"
    ]
    knee_approved = sum(1 for row in knee_rows if row["source"] == "candidate" and row["status"] == "approved")
    knee_rejected = sum(1 for row in knee_rows if row["source"] == "candidate" and row["status"] == "rejected")
    knee_precision = safe_div(knee_approved, knee_approved + knee_rejected)

    trick_rows = [row for row in rows if row["kind"] == "around_the_world" or row.get("trick_label")]
    trick_approved = sum(1 for row in trick_rows if row["source"] == "candidate" and row["status"] == "approved")
    trick_rejected = sum(1 for row in trick_rows if row["source"] == "candidate" and row["status"] == "rejected")
    trick_precision = safe_div(trick_approved, trick_approved + trick_rejected)

    stall_rows = [row for row in rows if row["kind"] == "stall"]
    approved_stall_windows = sum(1 for row in stall_rows if row["status"] == "approved" and row.get("has_stall_window"))
    approved_stalls_without_window = sum(1 for row in stall_rows if row["status"] == "approved" and not row.get("has_stall_window"))

    return {
        "knee": {
            "reviewed_examples": len(knee_rows),
            "approved_candidates": knee_approved,
            "rejected_candidates": knee_rejected,
            "precision": knee_precision,
            "release_state": "released"
            if len(knee_rows) >= TARGETS["knee_min_examples"] and knee_precision is not None and knee_precision >= TARGETS["knee_precision"]
            else "candidate_only",
        },
        "tricks": {
            "reviewed_examples": len(trick_rows),
            "approved_candidates": trick_approved,
            "rejected_candidates": trick_rejected,
            "precision": trick_precision,
            "release_state": "released" if trick_rows and trick_precision is not None and trick_precision >= TARGETS["trick_precision"] else "candidate_only",
        },
        "stall_windows": {
            "approved_windows_with_end": approved_stall_windows,
            "approved_stalls_without_window": approved_stalls_without_window,
        },
    }


def model_artifacts(training_manifest: Path | None, model_dir: Path | None) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    training_report: dict[str, Any] | None = None
    manifest_path: str | None = None
    if training_manifest and training_manifest.exists():
        manifest_path = portable_path(training_manifest)
        manifest = read_json(training_manifest)
        model_path = manifest.get("model_path")
        if model_path:
            path = Path(str(model_path))
            artifacts.append({"kind": "patch_classifier", "path": portable_path(path), "exists": path.exists()})
        training_report = manifest.get("training_report")
    search_dir = model_dir or (training_manifest.parent.parent / "models" if training_manifest else None)
    if search_dir and search_dir.exists():
        for path in sorted(search_dir.glob("*.joblib")):
            if not any(item.get("path") == portable_path(path) for item in artifacts):
                artifacts.append({"kind": "model_file", "path": portable_path(path), "exists": path.exists()})
        report_path = search_dir / "training_report.json"
        if training_report is None and report_path.exists():
            training_report = read_json(report_path)
    return {
        "training_manifest": manifest_path,
        "artifacts": artifacts,
        "has_saved_model_artifact": any(item["exists"] for item in artifacts),
        "patch_classifier_report": training_report,
    }


def ball_audit_metrics(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None
    doc = read_json(path)
    return {
        "path": portable_path(path),
        "sampled_events": doc.get("sampled_events"),
        "pass_rate": doc.get("pass_rate"),
        "target_pass_rate": doc.get("target_pass_rate"),
        "target_met": doc.get("target_met"),
        "status_counts": doc.get("status_counts", {}),
        "kind_status_counts": doc.get("kind_status_counts", {}),
    }


def gate(name: str, value: float | None, target: float, *, minimum: bool = True, no_data_message: str | None = None) -> dict[str, Any]:
    if value is None:
        return {"name": name, "status": "no_data", "value": None, "target": target, "message": no_data_message or "No held-out reviewed denominator."}
    passed = value >= target if minimum else value < target
    return {"name": name, "status": "pass" if passed else "fail", "value": value, "target": target}


def build_target_gates(test_metrics: dict[str, Any], all_metrics: dict[str, Any], ball_audit: dict[str, Any] | None, artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    test_kinds = test_metrics["kinds"]
    test_class = test_metrics["classification"]
    test_duplicates = test_metrics["duplicates"]
    all_release = all_metrics["release_conditions"]
    gates = [
        gate("touch precision on held-out reviewed labels", test_kinds["touch"]["precision"], TARGETS["touch_precision"]),
        gate("touch recall on held-out reviewed labels", test_kinds["touch"]["recall"], TARGETS["touch_recall"]),
        gate("duplicate touch rate", test_duplicates["duplicate_touch_rate_vs_accepted"], TARGETS["duplicate_touch_rate"], minimum=False),
        gate("drop/floor precision on held-out reviewed labels", test_kinds["drop_floor"]["precision"], TARGETS["drop_floor_precision"]),
        gate("drop/floor recall on held-out reviewed labels", test_kinds["drop_floor"]["recall"], TARGETS["drop_floor_recall"]),
        gate("stall precision on held-out reviewed labels", test_kinds["stall"]["precision"], TARGETS["stall_precision"]),
        gate("stall recall on held-out reviewed labels", test_kinds["stall"]["recall"], TARGETS["stall_recall"]),
        gate("side classification accuracy on held-out reviewed labels", test_class["side"]["accuracy"], TARGETS["side_accuracy"]),
        gate("contact type accuracy on held-out reviewed labels", test_class["contact_type"]["accuracy"], TARGETS["contact_type_accuracy"]),
        {
            "name": "knee release guard",
            "status": "pass" if all_release["knee"]["release_state"] == "released" else "candidate_only",
            "value": all_release["knee"],
            "target": {"min_examples": TARGETS["knee_min_examples"], "precision": TARGETS["knee_precision"]},
        },
        {
            "name": "trick release guard",
            "status": "pass" if all_release["tricks"]["release_state"] == "released" else "candidate_only",
            "value": all_release["tricks"],
            "target": {"precision": TARGETS["trick_precision"]},
        },
        {
            "name": "model artifact saved",
            "status": "pass" if artifacts.get("has_saved_model_artifact") else "fail",
            "value": artifacts.get("artifacts", []),
            "target": "at least one saved model artifact",
        },
    ]
    if ball_audit:
        gates.insert(0, gate("ball event-center pass rate", ball_audit.get("pass_rate"), TARGETS["ball_event_center_pass_rate"]))
    else:
        gates.insert(
            0,
            {
                "name": "ball event-center pass rate",
                "status": "no_data",
                "value": None,
                "target": TARGETS["ball_event_center_pass_rate"],
                "message": "No ball audit metrics supplied.",
            },
        )
    return gates


def compute_metrics(rows: list[dict[str, Any]], split: str | None = None) -> dict[str, Any]:
    selected = metric_rows(rows, split)
    return {
        "items": len(selected),
        "kinds": kind_metrics(selected),
        "classification": classification_metrics(selected),
        "duplicates": duplicate_metrics(selected),
        "release_conditions": release_condition_metrics(selected),
    }


def evaluation_doc(
    *,
    review_paths: list[Path],
    batch: BatchIndex,
    training_manifest: Path | None,
    model_dir: Path | None,
    ball_audit: Path | None,
    seed: int,
) -> dict[str, Any]:
    rows = reviewed_rows(review_paths, batch)
    clean_videos = sorted({str(row["source_video"]) for row in rows if row["label_tier"] == "clean_current_batch"})
    split_videos = sorted(batch.videos) or clean_videos
    split_population = "review_batch_videos" if batch.videos else "decided_clean_label_videos"
    assignments = deterministic_video_splits(split_videos, seed)
    for row in rows:
        row["split"] = assignments.get(str(row.get("source_video")), "exploratory")

    tiers = Counter(row["label_tier"] for row in rows)
    statuses = Counter(row["status"] for row in rows)
    all_clean = compute_metrics(rows, None)
    split_metrics = {split: compute_metrics(rows, split) for split in ("train", "validation", "test")}
    artifacts = model_artifacts(training_manifest, model_dir)
    audit = ball_audit_metrics(ball_audit)
    gates = build_target_gates(split_metrics["test"], all_clean, audit, artifacts)
    release_blockers = [
        gate_item
        for gate_item in gates
        if gate_item["status"] in {"fail", "no_data", "candidate_only"}
    ]
    dataset_id_source = json.dumps(
        {
            "review_files": [portable_path(path) for path in review_paths],
            "batch": portable_path(batch.path) if batch.path else None,
            "seed": seed,
        },
        sort_keys=True,
    )
    return {
        "schema_version": 1,
        "evaluation_id": hashlib.sha1(dataset_id_source.encode("utf-8")).hexdigest()[:12],
        "seed": seed,
        "inputs": {
            "reviews_dir": portable_path(review_paths[0].parent) if review_paths else None,
            "review_files": [portable_path(path) for path in review_paths],
            "batch": portable_path(batch.path) if batch.path else None,
            "batch_id": batch.batch_id,
            "training_manifest": portable_path(training_manifest) if training_manifest else None,
            "ball_audit": portable_path(ball_audit) if ball_audit else None,
        },
        "label_inventory": {
            "review_files": len(review_paths),
            "reviewed_items": len(rows),
            "label_tiers": dict(sorted(tiers.items())),
            "statuses": dict(sorted(statuses.items())),
        },
        "dataset_splits": {
            "assignment_method": "deterministic video-level shuffle",
            "assignment_population": split_population,
            "source_video_count": len(split_videos),
            "assignments": assignments,
            "summary": split_summary(rows, assignments),
        },
        "metrics": {
            "all_clean_current_batch": all_clean,
            "splits": split_metrics,
            "ball_tracking": audit,
            "model_artifacts": artifacts,
        },
        "target_gates": gates,
        "release_blockers": release_blockers,
        "reviewed_labels": rows,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_report(doc: dict[str, Any], path: Path) -> None:
    all_clean = doc["metrics"]["all_clean_current_batch"]
    test = doc["metrics"]["splits"]["test"]
    lines = [
        "# Release Evaluation Report",
        "",
        f"- Evaluation ID: `{doc['evaluation_id']}`",
        f"- Review files: {doc['label_inventory']['review_files']}",
        f"- Reviewed items: {doc['label_inventory']['reviewed_items']}",
        f"- Clean current-batch items: {doc['label_inventory']['label_tiers'].get('clean_current_batch', 0)}",
        f"- Exploratory reviewed items: {doc['label_inventory']['label_tiers'].get('exploratory_reviewed', 0)}",
        "",
        "## Held-Out Test Metrics",
        "",
        "| Metric | Value | Target |",
        "| --- | ---: | ---: |",
        f"| Touch precision | {percent(test['kinds']['touch']['precision'])} | {percent(TARGETS['touch_precision'])} |",
        f"| Touch recall | {percent(test['kinds']['touch']['recall'])} | {percent(TARGETS['touch_recall'])} |",
        f"| Drop precision | {percent(test['kinds']['drop_floor']['precision'])} | {percent(TARGETS['drop_floor_precision'])} |",
        f"| Drop recall | {percent(test['kinds']['drop_floor']['recall'])} | {percent(TARGETS['drop_floor_recall'])} |",
        f"| Stall precision | {percent(test['kinds']['stall']['precision'])} | {percent(TARGETS['stall_precision'])} |",
        f"| Stall recall | {percent(test['kinds']['stall']['recall'])} | {percent(TARGETS['stall_recall'])} |",
        f"| Side accuracy | {percent(test['classification']['side']['accuracy'])} | {percent(TARGETS['side_accuracy'])} |",
        f"| Contact type accuracy | {percent(test['classification']['contact_type']['accuracy'])} | {percent(TARGETS['contact_type_accuracy'])} |",
        f"| Duplicate touch rate | {percent(test['duplicates']['duplicate_touch_rate_vs_accepted'])} | < {percent(TARGETS['duplicate_touch_rate'])} |",
        "",
        "## All Clean Labels",
        "",
        "| Kind | Approved | Rejected | Manual missing | Precision | Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kind, stats in all_clean["kinds"].items():
        lines.append(
            f"| `{kind}` | {stats['approved_candidates']} | {stats['rejected_candidates']} | "
            f"{stats['manual_missing']} | {percent(stats['precision'])} | {percent(stats['recall'])} |"
        )
    lines.extend(["", "## Target Gates", ""])
    for gate_item in doc["target_gates"]:
        lines.append(f"- {gate_item['status'].upper()}: {gate_item['name']}")
    if doc["release_blockers"]:
        lines.extend(["", "## Release Blockers", ""])
        for blocker in doc["release_blockers"]:
            message = blocker.get("message")
            suffix = f" - {message}" if message else ""
            lines.append(f"- {blocker['name']}: {blocker['status']}{suffix}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Clean metrics use only reviewed items from the supplied review batch.",
            "- Older reviewed decisions remain in the JSONL export as exploratory labels but are not mixed into held-out release targets.",
            "- Recall is limited by manually added missing events; more active-learning review is required before release claims are strong.",
            "- Knee and trick classifiers remain candidate-only until their reviewed example and precision guards pass.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Hacky Track release readiness from reviewed labels")
    parser.add_argument("--reviews-dir", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--ball-audit", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review_paths = sorted(args.reviews_dir.glob("*.review.json"))
    batch = load_batch(args.batch)
    doc = evaluation_doc(
        review_paths=review_paths,
        batch=batch,
        training_manifest=args.training_manifest,
        model_dir=args.model_dir,
        ball_audit=args.ball_audit,
        seed=args.seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "release_metrics.json"
    splits_path = args.out_dir / "dataset_splits.json"
    labels_path = args.out_dir / "reviewed_labels.release.jsonl"
    report_path = args.out_dir / "release_evaluation_report.md"
    write_json(metrics_path, {key: value for key, value in doc.items() if key != "reviewed_labels"})
    write_json(splits_path, doc["dataset_splits"])
    write_jsonl(labels_path, doc["reviewed_labels"])
    write_report(doc, report_path)
    print(f"metrics: {metrics_path}")
    print(f"splits: {splits_path}")
    print(f"labels: {labels_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
