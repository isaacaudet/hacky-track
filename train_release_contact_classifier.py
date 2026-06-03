#!/usr/bin/env python3
"""Evaluate/train release contact side/type classification when labels exist.

The current release touch path predicts timing only. This script is the separate
promotion gate for contact intelligence: left/right, foot/knee, stall, etc. It
uses pose/body proximity as soft features when present, and reports `not_ready`
instead of fabricating labels when the corpus does not contain reviewed contact
annotations.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "runs/release-27-public/touch_corpus_v1"
DEFAULT_DATASET_DIR = DEFAULT_CORPUS / "touch_training_dataset_v1"
DEFAULT_LABELS_DIR = DEFAULT_CORPUS / "visual_touch_labels"
DEFAULT_OUT_DIR = DEFAULT_CORPUS / "release_contact_classifier_v1"
DEFAULT_MIN_EXAMPLES = 20
DEFAULT_MIN_VIDEOS = 3

POSE_FEATURES = [
    "pose_nearest_foot_conf",
    "pose_nearest_foot_dist_px",
    "pose_nearest_foot_dist_norm_shank",
    "pose_nearest_lower_conf",
    "pose_nearest_lower_dist_px",
    "pose_nearest_lower_dist_norm_shank",
]
CONTACT_TYPE_KEYS = ("contact_type", "label_contact_type", "reviewed_contact_type")
CONTACT_SIDE_KEYS = ("contact_side", "label_contact_side", "reviewed_contact_side")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def first_value(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value not in {None, ""}:
            return str(value)
    return None


def normalize_type(value: str | None, event_type: str | None = None) -> str | None:
    raw = (value or event_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "foot": "kick",
        "toe": "kick",
        "ankle": "kick",
        "heel": "kick",
        "left_kick": "kick",
        "right_kick": "kick",
        "touch": None,
        "release_touch": None,
        "unknown": None,
    }
    if raw in aliases:
        return aliases[raw]
    if raw in {"kick", "knee", "stall", "drop_floor", "chest", "hand"}:
        return raw
    return raw or None


def normalize_side(value: str | None) -> str | None:
    raw = (value or "").strip().lower()
    if raw in {"l", "left"}:
        return "left"
    if raw in {"r", "right"}:
        return "right"
    return None


def pose_candidate(row: dict[str, Any]) -> dict[str, str | None]:
    part = str(row.get("pose_nearest_lower_part") or row.get("pose_nearest_foot_part") or "").lower()
    contact_type = None
    if "knee" in part:
        contact_type = "knee"
    elif any(token in part for token in ("toe", "heel", "ankle", "foot")):
        contact_type = "kick"
    side = None
    if "left" in part:
        side = "left"
    elif "right" in part:
        side = "right"
    return {"pose_candidate_type": contact_type, "pose_candidate_side": side}


def pose_feature_present(row: dict[str, Any]) -> bool:
    return any(key in row and row.get(key) is not None for key in POSE_FEATURES)


def load_label_contact_examples(labels_dir: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for path in sorted(labels_dir.glob("*.events.json")):
        doc = read_json(path)
        source_video = str(doc.get("source_video") or path.name)
        for rally in doc.get("rallies", []):
            for event in rally.get("events", []):
                if event.get("review_status") not in {None, "", "approved", "reviewed"}:
                    continue
                contact_type = normalize_type(first_value(event, CONTACT_TYPE_KEYS), str(event.get("type") or ""))
                contact_side = normalize_side(first_value(event, CONTACT_SIDE_KEYS))
                if contact_type is None and contact_side is None:
                    continue
                examples.append(
                    {
                        "video_id": path.name.removesuffix(".events.json"),
                        "source_video": source_video,
                        "time_sec": float(event.get("time_sec") or 0.0),
                        "event_type": event.get("type"),
                        "contact_type": contact_type,
                        "contact_side": contact_side,
                    }
                )
    return examples


def rows_with_contact_labels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        contact_type = normalize_type(first_value(row, CONTACT_TYPE_KEYS))
        contact_side = normalize_side(first_value(row, CONTACT_SIDE_KEYS))
        if contact_type is None and contact_side is None:
            continue
        out.append({**row, "contact_type": contact_type, "contact_side": contact_side, **pose_candidate(row)})
    return out


def readiness_summary(rows: list[dict[str, Any]], label_examples: list[dict[str, Any]], *, min_examples: int, min_videos: int) -> dict[str, Any]:
    pose_rows = sum(1 for row in rows if pose_feature_present(row))
    labeled_rows = rows_with_contact_labels(rows)
    label_counts = Counter(row.get("contact_type") for row in labeled_rows if row.get("contact_type"))
    side_counts = Counter(row.get("contact_side") for row in labeled_rows if row.get("contact_side"))
    label_event_counts = Counter(example.get("contact_type") for example in label_examples if example.get("contact_type"))
    videos_with_labels = sorted({str(row.get("video_id")) for row in labeled_rows})
    reasons: list[str] = []
    if pose_rows == 0:
        reasons.append("pose/body proximity columns are missing; run run_touch_pipeline.py --attach-pose-features first")
    if len(labeled_rows) < min_examples:
        reasons.append(f"need at least {min_examples} training rows with reviewed contact labels, found {len(labeled_rows)}")
    if len(videos_with_labels) < min_videos:
        reasons.append(f"need at least {min_videos} labeled videos for clip-disjoint contact evaluation, found {len(videos_with_labels)}")
    if len(label_counts) < 2 and len(side_counts) < 2:
        reasons.append("need at least two contact classes or two side classes to train/evaluate")
    status = "ready_for_training" if not reasons else "not_ready"
    return {
        "status": status,
        "reasons": reasons,
        "candidate_rows": len(rows),
        "rows_with_pose_features": pose_rows,
        "rows_with_contact_labels": len(labeled_rows),
        "videos_with_contact_labels": videos_with_labels,
        "contact_type_counts_in_rows": dict(label_counts),
        "contact_side_counts_in_rows": dict(side_counts),
        "contact_type_counts_in_label_files": dict(label_event_counts),
        "label_file_contact_examples": len(label_examples),
        "pose_feature_names": POSE_FEATURES,
    }


def value_as_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        return 9999.0 if "dist" in key else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def leave_one_video_out_majority_baseline(rows: list[dict[str, Any]], label_key: str) -> dict[str, Any] | None:
    labeled = [row for row in rows if row.get(label_key)]
    videos = sorted({str(row.get("video_id")) for row in labeled})
    if len(videos) < 2:
        return None
    total = 0
    correct = 0
    folds = []
    for video_id in videos:
        train = [row for row in labeled if str(row.get("video_id")) != video_id]
        test = [row for row in labeled if str(row.get("video_id")) == video_id]
        if not train or not test:
            continue
        majority = Counter(str(row[label_key]) for row in train).most_common(1)[0][0]
        fold_correct = sum(1 for row in test if str(row[label_key]) == majority)
        total += len(test)
        correct += fold_correct
        folds.append({"video_id": video_id, "rows": len(test), "majority_label": majority, "accuracy": fold_correct / len(test)})
    return {"accuracy": correct / total if total else None, "rows": total, "folds": folds}


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Release Contact Classifier",
        "",
        f"- Status: `{summary['status']}`",
        f"- Created: `{summary['created_at']}`",
        f"- Candidate rows: `{summary['candidate_rows']}`",
        f"- Rows with pose features: `{summary['rows_with_pose_features']}`",
        f"- Rows with reviewed contact labels: `{summary['rows_with_contact_labels']}`",
        "",
    ]
    if summary["reasons"]:
        lines.extend(["## Blockers", ""])
        for reason in summary["reasons"]:
            lines.append(f"- {reason}")
        lines.append("")
    lines.extend(
        [
            "## Label Counts",
            "",
            f"- Contact type rows: `{summary['contact_type_counts_in_rows']}`",
            f"- Contact side rows: `{summary['contact_side_counts_in_rows']}`",
            f"- Contact labels found in event files: `{summary['contact_type_counts_in_label_files']}`",
            "",
            "## Notes",
            "",
            "- Pose/body proximity is treated as a soft feature source, never a hard touch rule.",
            "- This classifier is separate from the release touch-timing model; generic touches remain unlabeled until this gate is real.",
        ]
    )
    if summary.get("majority_baselines"):
        lines.extend(["", "## Baselines", ""])
        for label, result in summary["majority_baselines"].items():
            lines.append(f"- `{label}` leave-one-video-out majority baseline: `{result}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_contact_classifier(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.dataset_dir / "touch_training_candidates.jsonl")
    rows += read_jsonl(args.dataset_dir / "touch_training_test_frozen.jsonl")
    label_examples = load_label_contact_examples(args.labels_dir)
    summary = readiness_summary(rows, label_examples, min_examples=args.min_examples, min_videos=args.min_videos)
    labeled_rows = rows_with_contact_labels(rows)
    majority_baselines = {}
    if summary["status"] == "ready_for_training":
        for key in ("contact_type", "contact_side"):
            result = leave_one_video_out_majority_baseline(labeled_rows, key)
            if result:
                majority_baselines[key] = result
    summary.update(
        {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_dir": str(args.dataset_dir),
            "labels_dir": str(args.labels_dir),
            "majority_baselines": majority_baselines,
            "report": str(args.out_dir / "release_contact_classifier_report.md"),
        }
    )
    write_json(args.out_dir / "release_contact_classifier_summary.json", summary)
    write_report(args.out_dir / "release_contact_classifier_report.md", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate release contact type/side classifier when labels exist")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-examples", type=int, default=DEFAULT_MIN_EXAMPLES)
    parser.add_argument("--min-videos", type=int, default=DEFAULT_MIN_VIDEOS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_contact_classifier(args)
    print(f"status: {summary['status']}")
    print(f"report: {summary['report']}")
    if summary["reasons"]:
        print("reasons:")
        for reason in summary["reasons"]:
            print(f"- {reason}")


if __name__ == "__main__":
    main()
