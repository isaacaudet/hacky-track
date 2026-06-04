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
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "runs/release-27-public/touch_corpus_v1"
DEFAULT_DATASET_DIR = DEFAULT_CORPUS / "touch_training_dataset_v1"
DEFAULT_LABELS_DIR = DEFAULT_CORPUS / "visual_touch_labels"
DEFAULT_OUT_DIR = DEFAULT_CORPUS / "release_contact_classifier_v1"
DEFAULT_MIN_EXAMPLES = 20
DEFAULT_MIN_VIDEOS = 3
CONTACT_CLASS_MIN_EXAMPLES = 20

POSE_FEATURES = [
    "pose_nearest_foot_conf",
    "pose_nearest_foot_dist_px",
    "pose_nearest_foot_dist_norm_shank",
    "pose_nearest_lower_conf",
    "pose_nearest_lower_dist_px",
    "pose_nearest_lower_dist_norm_shank",
    "pose_left_foot_min_dist_px",
    "pose_right_foot_min_dist_px",
    "pose_left_right_foot_min_dist_delta_px",
    "pose_geometry_side_margin_px",
    "pose_geometry_surface_margin_px",
]
POSE_ATTACHMENT_KEYS = [
    "pose_feature_status",
    "pose_frame_index",
    "pose_ball_x",
    "pose_ball_y",
    "pose_ball_missing",
    "pose_missing",
    "pose_present",
    "pose_person_boxes",
    "pose_lower_body_present",
    "pose_foot_present",
    "pose_shank_length_px",
    *POSE_FEATURES,
]
CONTACT_TYPE_KEYS = ("contact_type", "label_contact_type", "reviewed_contact_type")
CONTACT_SIDE_KEYS = ("contact_side", "label_contact_side", "reviewed_contact_side")
CONTACT_SURFACE_KEYS = ("contact_surface", "label_contact_surface", "reviewed_contact_surface")
CONTACT_SIDE_BASIS_KEYS = ("contact_side_basis", "label_contact_side_basis", "reviewed_contact_side_basis", "side_label_basis")
CONTACT_SIDE_TRAINABLE_BASES = {"wearer_limb", "legacy_unspecified"}

NUMERIC_FEATURES = [
    "audio_attack_ratio",
    "audio_high_band_ratio",
    "audio_low_band_ratio",
    "audio_mfcc_1",
    "audio_mfcc_2",
    "audio_mfcc_3",
    "audio_mfcc_4",
    "audio_mfcc_5",
    "audio_mfcc_6",
    "audio_mid_band_ratio",
    "audio_peak_to_rms",
    "audio_post_rms",
    "audio_pre_rms",
    "audio_spectral_bandwidth",
    "audio_spectral_centroid",
    "audio_spectral_flatness",
    "audio_spectral_rolloff85",
    "audio_strength",
    "audio_window_peak",
    "audio_window_rms",
    "audio_zero_crossing_rate",
    "detector_confidence_near_candidate",
    "pose_ball_score",
    "pose_nearest_foot_conf",
    "pose_nearest_foot_dist_px",
    "pose_nearest_foot_dist_norm_shank",
    "pose_nearest_lower_conf",
    "pose_nearest_lower_dist_px",
    "pose_nearest_lower_dist_norm_shank",
    "pose_person_boxes",
    "pose_shank_length_px",
    "time_since_prev_candidate_sec",
    "time_to_next_candidate_sec",
    "trajectory_ax_window",
    "trajectory_ay_window",
    "trajectory_break_support",
    "trajectory_confidence_mean_window",
    "trajectory_gap_after_sec",
    "trajectory_gap_before_sec",
    "trajectory_impulse_score",
    "trajectory_local_y_quad_rms_px",
    "trajectory_max_positive_dvy",
    "trajectory_nearest_break_delta_sec",
    "trajectory_nearest_y_peak_delta_sec",
    "trajectory_nearest_y_trough_delta_sec",
    "trajectory_speed_after",
    "trajectory_speed_before",
    "trajectory_speed_delta",
    "trajectory_track_points_window",
    "trajectory_vx_after",
    "trajectory_vx_before",
    "trajectory_vx_delta",
    "trajectory_vy_after",
    "trajectory_vy_before",
    "trajectory_vy_delta",
    "trajectory_x_range_window_px",
    "trajectory_y_peak_prominence_window_px",
    "trajectory_y_position_pct_window",
    "trajectory_y_range_window_px",
    "trajectory_y_trough_prominence_window_px",
]
BOOLEAN_FEATURES = [
    "has_audio",
    "has_existing_hint",
    "height_reversal",
    "in_stall_window",
    "pose_ball_missing",
    "pose_foot_present",
    "pose_lower_body_present",
    "pose_missing",
    "pose_present",
]
CATEGORICAL_FEATURES = [
    "audio_timbre_status",
    "pose_feature_status",
    "pose_geometry_nearest_side",
    "pose_geometry_nearest_surface",
    "pose_nearest_foot_part",
    "pose_nearest_lower_part",
    "trajectory_feature_status",
    "visual_crop_feature_status",
    "vision_embedding_feature_status",
    "vision_embedding_model",
]
AUTO_FEATURE_PREFIXES = (
    "pose_left_",
    "pose_right_",
    "pose_geometry_",
    "foot_track_",
    "crop_ball_foot_",
    "visual_",
    "vision_",
)
CONTACT_TARGETS = ("contact_type", "contact_side", "contact_surface")
CONTACT_GATE_ACCURACY = 0.85
CONTACT_RELEASE_CLASSES = {
    "contact_type": ("kick", "stall", "knee", "drop_floor"),
    "contact_side": ("left", "right"),
    "contact_surface": ("inner", "outer"),
}
CONTACT_FEATURE_MODES = {
    "all_features": (),
    "no_foot_track": ("foot_track_",),
    "no_visual_crop": ("visual_",),
    "no_visual_crop_no_foot_track": ("visual_", "foot_track_"),
    "no_vision_embedding": ("vision_",),
    "no_vision_embedding_no_foot_track": ("vision_", "foot_track_"),
    "no_visual_features": ("visual_", "vision_"),
    "no_visual_features_no_foot_track": ("visual_", "vision_", "foot_track_"),
    "pose_only": ("__pose_only__",),
}
CONTACT_MODEL_FAMILIES = ("logistic_regression", "ridge_classifier", "linear_svc", "extra_trees", "gradient_boosting")
LINEAR_SVC_FEATURE_MODES = {"no_visual_features", "no_visual_features_no_foot_track", "pose_only"}
CONTACT_SIDE_SEQUENCE_STATES = ("left", "right")
CONTACT_SIDE_SEQUENCE_SMOOTHING = {
    "alpha": 1.0,
    "emission_temperature": 1.0,
    "transition_weight": 1.0,
}


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
        "left_knee": "knee",
        "right_knee": "knee",
        "left_inner_kick": "kick",
        "left_outer_kick": "kick",
        "right_inner_kick": "kick",
        "right_outer_kick": "kick",
        "left_inner_knee": "knee",
        "left_outer_knee": "knee",
        "right_inner_knee": "knee",
        "right_outer_knee": "knee",
        "left_stall": "stall",
        "right_stall": "stall",
        "left_inner_stall": "stall",
        "left_outer_stall": "stall",
        "right_inner_stall": "stall",
        "right_outer_stall": "stall",
        "touch": None,
        "release_touch": None,
        "unknown": None,
    }
    if raw in aliases:
        return aliases[raw]
    if raw in {"kick", "knee", "stall", "drop_floor", "chest", "hand"}:
        return raw
    return raw or None


def normalize_side(value: str | None, trick_label: str | None = None) -> str | None:
    raw = (value or trick_label or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in {"l", "left"}:
        return "left"
    if raw in {"r", "right"}:
        return "right"
    if raw.startswith("left_") or "_left" in raw:
        return "left"
    if raw.startswith("right_") or "_right" in raw:
        return "right"
    return None


def normalize_surface(value: str | None, trick_label: str | None = None) -> str | None:
    raw = (value or trick_label or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in {"inner", "inside"} or "_inner_" in raw or raw.startswith("inner_"):
        return "inner"
    if raw in {"outer", "outside"} or "_outer_" in raw or raw.startswith("outer_"):
        return "outer"
    return None


def side_from_trick_label(trick_label: str | None) -> str | None:
    return normalize_side(None, trick_label)


def normalize_side_basis(value: str | None, *, contact_side: str | None = None, trick_label: str | None = None) -> str:
    raw = (value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not raw and contact_side in {"left", "right"} and side_from_trick_label(trick_label) == contact_side:
        return "wearer_limb"
    aliases = {
        "": "legacy_unspecified",
        "unknown": "unknown",
        "unclear": "ambiguous",
        "ambiguous": "ambiguous",
        "contacting_limb": "wearer_limb",
        "wearer": "wearer_limb",
        "wearer_side": "wearer_limb",
        "wearer_limb": "wearer_limb",
        "body_side": "wearer_limb",
        "screen": "screen_position",
        "screen_side": "screen_position",
        "screen_position": "screen_position",
        "pose": "pose_anatomical",
        "pose_side": "pose_anatomical",
        "pose_anatomical": "pose_anatomical",
        "legacy": "legacy_unspecified",
        "legacy_unspecified": "legacy_unspecified",
    }
    return aliases.get(raw, raw or "legacy_unspecified")


def side_label_is_trainable(side_basis: str | None, *, contact_side: str | None = None, trick_label: str | None = None) -> bool:
    return normalize_side_basis(side_basis, contact_side=contact_side, trick_label=trick_label) in CONTACT_SIDE_TRAINABLE_BASES


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


def pose_columns_attached(row: dict[str, Any]) -> bool:
    return any(key in row for key in POSE_ATTACHMENT_KEYS)


def visual_crop_feature_present(row: dict[str, Any]) -> bool:
    return str(row.get("visual_crop_feature_status") or "") == "ok"


def vision_embedding_feature_present(row: dict[str, Any]) -> bool:
    return bool(row.get("vision_embedding_present"))


def release_class_coverage(label_counts: Counter[str], label_key: str, *, min_examples_per_class: int = CONTACT_CLASS_MIN_EXAMPLES) -> dict[str, Any] | None:
    release_classes = CONTACT_RELEASE_CLASSES.get(label_key)
    if not release_classes:
        return None
    class_counts = {klass: int(label_counts.get(klass, 0)) for klass in release_classes}
    blockers = [
        f"need at least {min_examples_per_class} `{klass}` labels, found {count}"
        for klass, count in class_counts.items()
        if count < min_examples_per_class
    ]
    return {
        "release_classes": list(release_classes),
        "min_examples_per_class": min_examples_per_class,
        "class_counts": class_counts,
        "passes": not blockers,
        "blockers": blockers,
    }


def video_match_keys(row: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for key in ("video_id", "video_name", "source_video"):
        value = row.get(key)
        if value not in {None, ""}:
            text = str(value)
            keys.add(text)
            keys.add(text.removesuffix(".MOV").removesuffix(".mov"))
    return keys


def load_label_contact_examples(labels_dir: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for path in sorted(labels_dir.glob("*.events.json")):
        doc = read_json(path)
        source_video = str(doc.get("source_video") or path.name)
        video_id = path.name.removesuffix(".events.json")
        for rally in doc.get("rallies", []):
            for event in rally.get("events", []):
                if event.get("review_status") not in {None, "", "approved", "reviewed"}:
                    continue
                trick_label = str(event.get("trick_label") or "")
                contact_type = normalize_type(first_value(event, CONTACT_TYPE_KEYS) or trick_label, str(event.get("type") or ""))
                contact_side = normalize_side(first_value(event, CONTACT_SIDE_KEYS), trick_label)
                contact_side_basis = normalize_side_basis(first_value(event, CONTACT_SIDE_BASIS_KEYS), contact_side=contact_side, trick_label=trick_label)
                if contact_side and not side_label_is_trainable(contact_side_basis, contact_side=contact_side, trick_label=trick_label):
                    contact_side = None
                contact_surface = normalize_surface(first_value(event, CONTACT_SURFACE_KEYS), trick_label)
                if contact_type is None and contact_side is None and contact_surface is None:
                    continue
                examples.append(
                    {
                        "video_id": video_id,
                        "source_video": source_video,
                        "time_sec": float(event.get("time_sec") or 0.0),
                        "event_type": event.get("type"),
                        "contact_type": contact_type,
                        "contact_side": contact_side,
                        "contact_side_basis": contact_side_basis,
                        "contact_surface": contact_surface,
                        "trick_label": trick_label or None,
                    }
                )
    return examples


def contact_label_file_inventory(labels_dir: Path) -> dict[str, Any]:
    """Summarize reviewed contact labels in event files before row matching.

    This inventory intentionally distinguishes generic side/type labels from
    inner/outer surface labels. A reviewed `right_kick` is useful for side/type
    training, but it is not a surface example unless it says inner/outer.
    """

    files: list[dict[str, Any]] = []
    aggregate_counts: Counter[str] = Counter()
    aggregate_type_counts: Counter[str] = Counter()
    aggregate_side_counts: Counter[str] = Counter()
    aggregate_surface_counts: Counter[str] = Counter()
    for path in sorted(labels_dir.glob("*.events.json")):
        doc = read_json(path)
        file_counts: Counter[str] = Counter()
        type_counts: Counter[str] = Counter()
        side_counts: Counter[str] = Counter()
        surface_counts: Counter[str] = Counter()
        trick_counts: Counter[str] = Counter()
        for rally in doc.get("rallies", []):
            for event in rally.get("events", []):
                if event.get("review_status") not in {None, "", "approved", "reviewed"}:
                    continue
                file_counts["reviewed_events"] += 1
                trick_label = str(event.get("trick_label") or "")
                contact_type = normalize_type(first_value(event, CONTACT_TYPE_KEYS) or trick_label, str(event.get("type") or ""))
                contact_side = normalize_side(first_value(event, CONTACT_SIDE_KEYS), trick_label)
                contact_side_basis = normalize_side_basis(first_value(event, CONTACT_SIDE_BASIS_KEYS), contact_side=contact_side, trick_label=trick_label)
                if contact_side and not side_label_is_trainable(contact_side_basis, contact_side=contact_side, trick_label=trick_label):
                    file_counts["excluded_non_wearer_side_labels"] += 1
                    contact_side = None
                raw_surface = first_value(event, CONTACT_SURFACE_KEYS)
                contact_surface = normalize_surface(raw_surface, trick_label)
                if raw_surface and str(raw_surface).strip().lower() in {"unknown", "unclear", "ambiguous"}:
                    file_counts["explicit_unknown_surface_labels"] += 1
                if contact_type:
                    file_counts["contact_type_labels"] += 1
                    type_counts[contact_type] += 1
                if contact_side:
                    file_counts["contact_side_labels"] += 1
                    side_counts[contact_side] += 1
                if contact_surface:
                    file_counts["contact_surface_labels"] += 1
                    surface_counts[contact_surface] += 1
                if trick_label:
                    trick_counts[trick_label] += 1
        row = {
            "file": path.name,
            "reviewed_events": int(file_counts["reviewed_events"]),
            "contact_type_labels": int(file_counts["contact_type_labels"]),
            "contact_side_labels": int(file_counts["contact_side_labels"]),
            "contact_surface_labels": int(file_counts["contact_surface_labels"]),
            "explicit_unknown_surface_labels": int(file_counts["explicit_unknown_surface_labels"]),
            "excluded_non_wearer_side_labels": int(file_counts["excluded_non_wearer_side_labels"]),
            "type_counts": dict(type_counts),
            "side_counts": dict(side_counts),
            "surface_counts": dict(surface_counts),
            "trick_label_counts": dict(trick_counts),
        }
        files.append(row)
        aggregate_counts.update({key: value for key, value in file_counts.items() if isinstance(value, int)})
        aggregate_type_counts.update(type_counts)
        aggregate_side_counts.update(side_counts)
        aggregate_surface_counts.update(surface_counts)
    return {
        "files": files,
        "aggregate": {
            "reviewed_events": int(aggregate_counts["reviewed_events"]),
            "contact_type_labels": int(aggregate_counts["contact_type_labels"]),
            "contact_side_labels": int(aggregate_counts["contact_side_labels"]),
            "contact_surface_labels": int(aggregate_counts["contact_surface_labels"]),
            "explicit_unknown_surface_labels": int(aggregate_counts["explicit_unknown_surface_labels"]),
            "excluded_non_wearer_side_labels": int(aggregate_counts["excluded_non_wearer_side_labels"]),
            "type_counts": dict(aggregate_type_counts),
            "side_counts": dict(aggregate_side_counts),
            "surface_counts": dict(aggregate_surface_counts),
        },
    }


def attach_event_file_contact_labels(
    rows: list[dict[str, Any]],
    label_examples: list[dict[str, Any]],
    *,
    tolerance_sec: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Match reviewed contact/stall/drop labels onto candidate feature rows.

    The touch training table owns candidate-level features; the visual event files
    own human labels. Matching by clip and near-time keeps those responsibilities
    separate and avoids requiring the review UI to duplicate labels into every row.
    """

    out = [dict(row) for row in rows]
    matched_row_indexes: set[int] = set()
    unmatched: list[dict[str, Any]] = []
    for example in label_examples:
        example_keys = video_match_keys(example)
        best_index: int | None = None
        best_delta = tolerance_sec + 1.0
        example_time = float(example.get("time_sec") or 0.0)
        for index, row in enumerate(out):
            if index in matched_row_indexes:
                continue
            if not (video_match_keys(row) & example_keys):
                continue
            try:
                delta = abs(float(row.get("candidate_time_sec")) - example_time)
            except (TypeError, ValueError):
                continue
            if delta <= tolerance_sec and delta < best_delta:
                best_index = index
                best_delta = delta
        if best_index is None:
            unmatched.append(example)
            continue
        matched_row_indexes.add(best_index)
        row = out[best_index]
        if example.get("contact_type"):
            row["contact_type"] = example["contact_type"]
            row["label_contact_type"] = example["contact_type"]
        if example.get("contact_side"):
            row["contact_side"] = example["contact_side"]
            row["label_contact_side"] = example["contact_side"]
            row["contact_side_basis"] = example.get("contact_side_basis") or "legacy_unspecified"
            row["label_contact_side_basis"] = example.get("contact_side_basis") or "legacy_unspecified"
        if example.get("contact_surface"):
            row["contact_surface"] = example["contact_surface"]
            row["label_contact_surface"] = example["contact_surface"]
        if example.get("trick_label"):
            row["trick_label"] = example["trick_label"]
        row["contact_label_source"] = "visual_event_file"
        row["contact_label_event_type"] = example.get("event_type")
        row["contact_label_time_sec"] = example_time
        row["contact_label_match_delta_sec"] = round(best_delta, 6)
    summary = {
        "label_examples": len(label_examples),
        "matched_examples": len(matched_row_indexes),
        "unmatched_examples": len(unmatched),
        "unmatched": unmatched[:20],
        "match_tolerance_sec": tolerance_sec,
    }
    return out, summary


def rows_with_contact_labels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        contact_type = normalize_type(first_value(row, CONTACT_TYPE_KEYS) or str(row.get("trick_label") or ""))
        contact_side = normalize_side(first_value(row, CONTACT_SIDE_KEYS), str(row.get("trick_label") or ""))
        contact_side_basis = normalize_side_basis(first_value(row, CONTACT_SIDE_BASIS_KEYS), contact_side=contact_side, trick_label=str(row.get("trick_label") or ""))
        if contact_side and not side_label_is_trainable(contact_side_basis, contact_side=contact_side, trick_label=str(row.get("trick_label") or "")):
            contact_side = None
        contact_surface = normalize_surface(first_value(row, CONTACT_SURFACE_KEYS), str(row.get("trick_label") or ""))
        if contact_type is None and contact_side is None and contact_surface is None:
            continue
        out.append({**row, "contact_type": contact_type, "contact_side": contact_side, "contact_side_basis": contact_side_basis, "contact_surface": contact_surface, **pose_candidate(row)})
    return out


def contact_label_gap_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labeled_rows = rows_with_contact_labels(rows)
    out: dict[str, Any] = {}
    for target in CONTACT_TARGETS:
        counts = Counter(str(row[target]) for row in labeled_rows if row.get(target))
        coverage = release_class_coverage(counts, target)
        if coverage is None:
            continue
        out[target] = {
            "class_counts": coverage["class_counts"],
            "min_examples_per_class": coverage["min_examples_per_class"],
            "additional_needed": {
                klass: max(0, int(coverage["min_examples_per_class"]) - int(count))
                for klass, count in coverage["class_counts"].items()
            },
            "passes": coverage["passes"],
            "blockers": coverage["blockers"],
        }
    return out


def pose_coverage_by_contact_target(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labeled_rows = rows_with_contact_labels(rows)
    out: dict[str, Any] = {}
    for target in CONTACT_TARGETS:
        target_rows = [row for row in labeled_rows if row.get(target)]
        by_label: dict[str, Any] = {}
        for label in sorted({str(row[target]) for row in target_rows}):
            label_rows = [row for row in target_rows if str(row[target]) == label]
            by_label[label] = {
                "rows": len(label_rows),
                "pose_present_rows": sum(1 for row in label_rows if row.get("pose_present")),
                "usable_pose_distance_rows": sum(1 for row in label_rows if pose_feature_present(row)),
                "pose_status_counts": dict(Counter(str(row.get("pose_feature_status") or "missing") for row in label_rows)),
            }
        out[target] = {
            "rows": len(target_rows),
            "pose_present_rows": sum(1 for row in target_rows if row.get("pose_present")),
            "usable_pose_distance_rows": sum(1 for row in target_rows if pose_feature_present(row)),
            "pose_status_counts": dict(Counter(str(row.get("pose_feature_status") or "missing") for row in target_rows)),
            "by_label": by_label,
        }
    return out


def readiness_summary(rows: list[dict[str, Any]], label_examples: list[dict[str, Any]], *, min_examples: int, min_videos: int) -> dict[str, Any]:
    pose_attached_rows = sum(1 for row in rows if pose_columns_attached(row))
    pose_present_rows = sum(1 for row in rows if row.get("pose_present"))
    pose_rows = sum(1 for row in rows if pose_feature_present(row))
    visual_crop_rows = sum(1 for row in rows if visual_crop_feature_present(row))
    vision_embedding_rows = sum(1 for row in rows if vision_embedding_feature_present(row))
    labeled_rows = rows_with_contact_labels(rows)
    label_counts = Counter(row.get("contact_type") for row in labeled_rows if row.get("contact_type"))
    side_counts = Counter(row.get("contact_side") for row in labeled_rows if row.get("contact_side"))
    side_basis_counts = Counter(row.get("contact_side_basis") for row in labeled_rows if row.get("contact_side") and row.get("contact_side_basis"))
    surface_counts = Counter(row.get("contact_surface") for row in labeled_rows if row.get("contact_surface"))
    label_event_counts = Counter(example.get("contact_type") for example in label_examples if example.get("contact_type"))
    surface_event_counts = Counter(example.get("contact_surface") for example in label_examples if example.get("contact_surface"))
    videos_with_labels = sorted({str(row.get("video_id")) for row in labeled_rows})
    reasons: list[str] = []
    if pose_attached_rows == 0:
        reasons.append("pose/body proximity columns are missing; run run_touch_pipeline.py --attach-pose-features first")
    elif pose_rows == 0:
        reasons.append("pose/body proximity columns are attached but no usable ball-to-body distance rows are present; rerun pose with fresh OWLv2/L2 detections instead of a stale cache")
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
        "rows_with_pose_columns": pose_attached_rows,
        "rows_with_pose_present": pose_present_rows,
        "rows_with_pose_features": pose_rows,
        "rows_with_visual_crop_features": visual_crop_rows,
        "rows_with_vision_embedding_features": vision_embedding_rows,
        "rows_with_contact_labels": len(labeled_rows),
        "videos_with_contact_labels": videos_with_labels,
        "contact_type_counts_in_rows": dict(label_counts),
        "contact_side_counts_in_rows": dict(side_counts),
        "contact_side_basis_counts_in_rows": dict(side_basis_counts),
        "contact_surface_counts_in_rows": dict(surface_counts),
        "contact_type_counts_in_label_files": dict(label_event_counts),
        "contact_surface_counts_in_label_files": dict(surface_event_counts),
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


def auto_feature_value(value: Any) -> float | str | None:
    if value in {None, ""}:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not np_is_finite(value):
            return None
        return float(value)
    text = str(value)
    try:
        numeric = float(text)
    except ValueError:
        return text
    return numeric if np_is_finite(numeric) else None


def np_is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def is_disabled_feature(key: str, disabled_prefixes: tuple[str, ...]) -> bool:
    return bool(disabled_prefixes and key.startswith(disabled_prefixes))


def contact_feature_dict(row: dict[str, Any], *, disabled_prefixes: tuple[str, ...] = ()) -> dict[str, Any]:
    pose_only = disabled_prefixes == ("__pose_only__",)
    features: dict[str, Any] = {}
    for key in NUMERIC_FEATURES:
        if is_disabled_feature(key, disabled_prefixes):
            continue
        features[key] = value_as_float(row, key)
    for key in BOOLEAN_FEATURES:
        if is_disabled_feature(key, disabled_prefixes):
            continue
        features[key] = 1.0 if bool(row.get(key)) else 0.0
    for key in CATEGORICAL_FEATURES:
        if is_disabled_feature(key, disabled_prefixes):
            continue
        features[key] = str(row.get(key) or "missing")
    for key, value in pose_candidate(row).items():
        if is_disabled_feature(key, disabled_prefixes):
            continue
        features[key] = value or "missing"
    raw_sources = row.get("raw_sources") or []
    if isinstance(raw_sources, str):
        raw_sources = [raw_sources]
    for source in raw_sources:
        features[f"raw_source={source}"] = 1.0
    for key, value in row.items():
        if not key.startswith(AUTO_FEATURE_PREFIXES) or key in features:
            continue
        if is_disabled_feature(key, disabled_prefixes):
            continue
        parsed = auto_feature_value(value)
        if parsed is not None:
            features[key] = parsed
    if pose_only:
        return {key: value for key, value in features.items() if key.startswith("pose")}
    return features


def build_model(model_family: str = "logistic_regression"):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    if model_family == "logistic_regression":
        return make_pipeline(
            DictVectorizer(sparse=True),
            StandardScaler(with_mean=False),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=11),
        )
    if model_family == "ridge_classifier":
        return make_pipeline(
            DictVectorizer(sparse=True),
            StandardScaler(with_mean=False),
            RidgeClassifier(class_weight="balanced"),
        )
    if model_family == "linear_svc":
        return make_pipeline(
            DictVectorizer(sparse=True),
            StandardScaler(with_mean=False),
            LinearSVC(class_weight="balanced", C=0.2, random_state=11, max_iter=50000),
        )
    if model_family == "extra_trees":
        return make_pipeline(
            DictVectorizer(sparse=False),
            ExtraTreesClassifier(n_estimators=200, max_depth=4, class_weight="balanced", random_state=11),
        )
    if model_family == "gradient_boosting":
        return make_pipeline(
            DictVectorizer(sparse=False),
            GradientBoostingClassifier(max_depth=2, n_estimators=40, random_state=11),
        )
    raise ValueError(f"unknown contact model family: {model_family}")


def prediction_confidences(model: Any, features: list[dict[str, Any]]) -> list[float]:
    if not features:
        return []
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        return [float(max(row)) for row in probabilities]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        margins: list[float] = []
        for row in scores:
            if isinstance(row, (list, tuple)):
                values = [float(value) for value in row]
            elif hasattr(row, "tolist"):
                listed = row.tolist()
                values = [float(value) for value in listed] if isinstance(listed, list) else [float(listed)]
            else:
                values = [float(row)]
            if len(values) == 1:
                margins.append(abs(values[0]))
            else:
                sorted_values = sorted(values)
                margins.append(sorted_values[-1] - sorted_values[-2])
        positive = [margin for margin in margins if margin > 0]
        scale = sorted(positive)[len(positive) // 2] if positive else 1.0
        if scale <= 0:
            scale = 1.0
        return [float(1.0 / (1.0 + math.exp(-(margin / scale)))) for margin in margins]
    return [1.0 for _ in features]


def rows_for_target(rows: list[dict[str, Any]], label_key: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get(label_key)]


def confusion_counts(labels: Iterable[str], preds: Iterable[str]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for label, pred in zip(labels, preds):
        matrix[str(label)][str(pred)] += 1
    return {label: dict(preds_by_label) for label, preds_by_label in sorted(matrix.items())}


def classification_quality(labels: Iterable[str], preds: Iterable[str]) -> dict[str, Any]:
    label_list = [str(label) for label in labels]
    pred_list = [str(pred) for pred in preds]
    total = len(label_list)
    correct = sum(1 for label, pred in zip(label_list, pred_list) if label == pred)
    confusion = confusion_counts(label_list, pred_list)
    per_class_recall: dict[str, float | None] = {}
    for label, pred_counts in confusion.items():
        class_total = sum(int(count) for count in pred_counts.values())
        per_class_recall[label] = None if class_total <= 0 else float(pred_counts.get(label, 0) / class_total)
    valid_recalls = [value for value in per_class_recall.values() if value is not None]
    return {
        "accuracy": correct / total if total else None,
        "balanced_accuracy": sum(valid_recalls) / len(valid_recalls) if valid_recalls else None,
        "per_class_recall": per_class_recall,
        "confusion": confusion,
    }


def selective_accuracy_rows(predictions: list[dict[str, Any]], thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)) -> list[dict[str, Any]]:
    rows = []
    total = len(predictions)
    for threshold in thresholds:
        kept = [row for row in predictions if float(row.get("confidence") or 0.0) >= threshold]
        if not kept:
            rows.append({"threshold": threshold, "kept": 0, "coverage": 0.0, "accuracy": None})
            continue
        correct = sum(1 for row in kept if row["correct"])
        rows.append(
            {
                "threshold": threshold,
                "kept": len(kept),
                "coverage": len(kept) / total if total else 0.0,
                "accuracy": correct / len(kept),
            }
        )
    return rows


def sequence_transition_priors(
    rows: list[dict[str, Any]],
    label_key: str,
    *,
    states: tuple[str, ...],
    alpha: float,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    init_counts: Counter[str] = Counter()
    transition_counts: dict[str, Counter[str]] = {state: Counter() for state in states}
    for video_id in sorted({str(row.get("video_id")) for row in rows}):
        sequence = [
            str(row[label_key])
            for row in sorted(
                (row for row in rows if str(row.get("video_id")) == video_id and str(row.get(label_key)) in states),
                key=lambda item: float(item.get("candidate_time_sec") or 0.0),
            )
        ]
        if not sequence:
            continue
        init_counts[sequence[0]] += 1
        for prior, current in zip(sequence, sequence[1:]):
            transition_counts[prior][current] += 1
    init_total = sum(init_counts.values()) + alpha * len(states)
    init = {state: (init_counts[state] + alpha) / init_total for state in states}
    transitions: dict[str, dict[str, float]] = {}
    for prior in states:
        denom = sum(transition_counts[prior].values()) + alpha * len(states)
        transitions[prior] = {
            current: (transition_counts[prior][current] + alpha) / denom
            for current in states
        }
    return init, transitions


def confidence_emission_probs(
    *,
    prediction: str,
    confidence: float,
    states: tuple[str, ...],
    temperature: float,
) -> dict[str, float]:
    clipped = max(0.501, min(0.999, float(confidence)))
    if temperature <= 0:
        temperature = 1.0
    logit = math.log(clipped / (1.0 - clipped)) * temperature
    adjusted = 1.0 / (1.0 + math.exp(-logit))
    other = (1.0 - adjusted) / max(1, len(states) - 1)
    return {state: adjusted if state == prediction else other for state in states}


def viterbi_smooth_sequence(
    predictions: list[str],
    confidences: list[float],
    *,
    init: dict[str, float],
    transitions: dict[str, dict[str, float]],
    states: tuple[str, ...],
    emission_temperature: float,
    transition_weight: float,
) -> list[str]:
    if not predictions:
        return []
    dp: list[dict[str, float]] = []
    back: list[dict[str, str | None]] = []
    for index, (prediction, confidence) in enumerate(zip(predictions, confidences)):
        emissions = confidence_emission_probs(
            prediction=prediction,
            confidence=confidence,
            states=states,
            temperature=emission_temperature,
        )
        if index == 0:
            dp.append(
                {
                    state: math.log(max(1e-12, init[state])) + math.log(max(1e-12, emissions[state]))
                    for state in states
                }
            )
            back.append({state: None for state in states})
            continue
        current_scores: dict[str, float] = {}
        current_back: dict[str, str | None] = {}
        for state in states:
            candidates = []
            for prior in states:
                score = (
                    dp[-1][prior]
                    + transition_weight * math.log(max(1e-12, transitions[prior][state]))
                    + math.log(max(1e-12, emissions[state]))
                )
                candidates.append((score, prior))
            best_score, best_prior = max(candidates, key=lambda item: item[0])
            current_scores[state] = best_score
            current_back[state] = best_prior
        dp.append(current_scores)
        back.append(current_back)
    final_state = max(states, key=lambda state: dp[-1][state])
    sequence = [final_state]
    for index in range(len(predictions) - 1, 0, -1):
        prior = back[index][sequence[-1]]
        if prior is None:
            break
        sequence.append(prior)
    return list(reversed(sequence))


def train_single_contact_target(
    rows: list[dict[str, Any]],
    label_key: str,
    *,
    min_examples: int,
    min_videos: int,
    feature_mode: str = "all_features",
    disabled_prefixes: tuple[str, ...] = (),
    model_family: str = "logistic_regression",
) -> tuple[dict[str, Any], Any | None]:
    labeled = rows_for_target(rows, label_key)
    labels = [str(row[label_key]) for row in labeled]
    label_counts = Counter(labels)
    videos = sorted({str(row.get("video_id")) for row in labeled})
    reasons: list[str] = []
    if len(labeled) < min_examples:
        reasons.append(f"need at least {min_examples} rows for {label_key}, found {len(labeled)}")
    if len(videos) < min_videos:
        reasons.append(f"need at least {min_videos} videos for {label_key}, found {len(videos)}")
    if len(label_counts) < 2:
        reasons.append(f"need at least 2 classes for {label_key}, found {dict(label_counts)}")
    if reasons:
        return (
            {
                "status": "not_ready",
                "reasons": reasons,
                "rows": len(labeled),
                "videos": videos,
                "label_counts": dict(label_counts),
            },
            None,
        )

    predictions: list[dict[str, Any]] = []
    sequence_predictions: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    skipped_folds: list[dict[str, Any]] = []
    for video_id in videos:
        train = [row for row in labeled if str(row.get("video_id")) != video_id]
        test = sorted(
            (row for row in labeled if str(row.get("video_id")) == video_id),
            key=lambda row: float(row.get("candidate_time_sec") or 0.0),
        )
        train_labels = [str(row[label_key]) for row in train]
        if len(set(train_labels)) < 2:
            skipped_folds.append({"video_id": video_id, "rows": len(test), "reason": "training fold has one class"})
            continue
        model = build_model(model_family)
        train_features = [contact_feature_dict(row, disabled_prefixes=disabled_prefixes) for row in train]
        test_features = [contact_feature_dict(row, disabled_prefixes=disabled_prefixes) for row in test]
        model.fit(train_features, train_labels)
        test_labels = [str(row[label_key]) for row in test]
        fold_preds = [str(value) for value in model.predict(test_features)]
        fold_confidences = prediction_confidences(model, test_features)
        fold_sequence_preds: list[str] | None = None
        if label_key == "contact_side":
            init, transitions = sequence_transition_priors(
                train,
                label_key,
                states=CONTACT_SIDE_SEQUENCE_STATES,
                alpha=float(CONTACT_SIDE_SEQUENCE_SMOOTHING["alpha"]),
            )
            fold_sequence_preds = viterbi_smooth_sequence(
                fold_preds,
                fold_confidences,
                init=init,
                transitions=transitions,
                states=CONTACT_SIDE_SEQUENCE_STATES,
                emission_temperature=float(CONTACT_SIDE_SEQUENCE_SMOOTHING["emission_temperature"]),
                transition_weight=float(CONTACT_SIDE_SEQUENCE_SMOOTHING["transition_weight"]),
            )
        fold_correct = sum(1 for label, pred in zip(test_labels, fold_preds) if label == pred)
        folds.append(
            {
                "video_id": video_id,
                "rows": len(test),
                "accuracy": fold_correct / len(test) if test else None,
                "label_counts": dict(Counter(test_labels)),
                "pred_counts": dict(Counter(fold_preds)),
            }
        )
        for row, label, pred, confidence in zip(test, test_labels, fold_preds, fold_confidences):
            predictions.append(
                {
                    "video_id": row.get("video_id"),
                    "candidate_time_sec": row.get("candidate_time_sec"),
                    "label": label,
                    "prediction": pred,
                    "confidence": round(confidence, 6),
                    "correct": label == pred,
                }
            )
        if fold_sequence_preds is not None:
            for row, label, pred, raw_pred, confidence in zip(test, test_labels, fold_sequence_preds, fold_preds, fold_confidences):
                sequence_predictions.append(
                    {
                        "video_id": row.get("video_id"),
                        "candidate_time_sec": row.get("candidate_time_sec"),
                        "label": label,
                        "prediction": pred,
                        "raw_prediction": raw_pred,
                        "confidence": round(confidence, 6),
                        "correct": label == pred,
                    }
                )

    if not predictions:
        return (
            {
                "status": "not_ready",
                "reasons": ["all leave-one-video-out folds were skipped"],
                "rows": len(labeled),
                "videos": videos,
                "label_counts": dict(label_counts),
                "skipped_folds": skipped_folds,
            },
            None,
        )

    quality = classification_quality((row["label"] for row in predictions), (row["prediction"] for row in predictions))
    accuracy = float(quality["accuracy"] or 0.0)
    final_model = build_model(model_family)
    final_model.fit([contact_feature_dict(row, disabled_prefixes=disabled_prefixes) for row in labeled], labels)
    gate = "pass" if accuracy >= CONTACT_GATE_ACCURACY else "fail"
    gate_blockers: list[str] = []
    class_coverage = release_class_coverage(label_counts, label_key)
    release_scope_gate = "pass" if gate == "pass" and (class_coverage is None or class_coverage["passes"]) else "fail"
    release_scope_blockers = list((class_coverage or {}).get("blockers") or [])
    side_basis_counts: dict[str, int] | None = None
    explicit_wearer_limb_rows: int | None = None
    if label_key == "contact_side":
        side_basis_counter = Counter(
            normalize_side_basis(row.get("contact_side_basis"), contact_side=row.get("contact_side"), trick_label=row.get("trick_label"))
            for row in labeled
            if row.get("contact_side")
        )
        side_basis_counts = dict(side_basis_counter)
        explicit_wearer_limb_rows = side_basis_counter.get("wearer_limb", 0)
        if explicit_wearer_limb_rows < min_examples:
            gate = "fail"
            gate_blockers.append(
                f"need at least {min_examples} explicit wearer_limb side labels, found {explicit_wearer_limb_rows}; legacy_unspecified side labels are analysis-only"
            )
            release_scope_gate = "fail"
            release_scope_blockers.append(gate_blockers[-1])
    if gate != "pass":
        release_scope_gate = "fail"
        accuracy_blocker = f"accuracy gate failed: {accuracy:.3f} < {CONTACT_GATE_ACCURACY:.2f}"
        if accuracy_blocker not in release_scope_blockers:
            release_scope_blockers.insert(0, accuracy_blocker)
    result = {
        "status": "trained",
        "feature_mode": feature_mode,
        "model_family": model_family,
        "disabled_prefixes": list(disabled_prefixes),
        "gate": gate,
        "gate_blockers": gate_blockers,
        "gate_accuracy_threshold": CONTACT_GATE_ACCURACY,
        "release_scope_gate": release_scope_gate,
        "release_scope_blockers": release_scope_blockers,
        "release_class_coverage": class_coverage,
        "accuracy": accuracy,
        "balanced_accuracy": quality["balanced_accuracy"],
        "per_class_recall": quality["per_class_recall"],
        "rows": len(predictions),
        "training_rows": len(labeled),
        "videos": videos,
        "label_counts": dict(label_counts),
        "prediction_counts": dict(Counter(row["prediction"] for row in predictions)),
        "confusion": quality["confusion"],
        "selective_accuracy": selective_accuracy_rows(predictions),
        "folds": folds,
        "skipped_folds": skipped_folds,
        "errors": [row for row in predictions if not row["correct"]][:50],
    }
    if sequence_predictions:
        sequence_quality = classification_quality(
            (row["label"] for row in sequence_predictions),
            (row["prediction"] for row in sequence_predictions),
        )
        result["sequence_smoothed"] = {
            "status": "diagnostic_only",
            "note": (
                "Temporal smoothing is evaluated on leave-one-video-out side predictions only; "
                "automatic HUD side badges remain blocked unless the release gate passes."
            ),
            "params": dict(CONTACT_SIDE_SEQUENCE_SMOOTHING),
            "states": list(CONTACT_SIDE_SEQUENCE_STATES),
            "accuracy": sequence_quality["accuracy"],
            "balanced_accuracy": sequence_quality["balanced_accuracy"],
            "per_class_recall": sequence_quality["per_class_recall"],
            "confusion": sequence_quality["confusion"],
            "rows": len(sequence_predictions),
            "prediction_counts": dict(Counter(row["prediction"] for row in sequence_predictions)),
            "changed_rows": sum(1 for row in sequence_predictions if row["prediction"] != row["raw_prediction"]),
            "improves_raw_accuracy": (sequence_quality["accuracy"] or 0.0) > accuracy,
            "errors": [row for row in sequence_predictions if not row["correct"]][:50],
        }
    if side_basis_counts is not None:
        result["side_basis_counts"] = side_basis_counts
        result["explicit_wearer_limb_rows"] = explicit_wearer_limb_rows
        result["side_basis_min_examples"] = min_examples
    return result, final_model


def train_single_contact_target_best_mode(
    rows: list[dict[str, Any]],
    label_key: str,
    *,
    min_examples: int,
    min_videos: int,
) -> tuple[dict[str, Any], Any | None]:
    combo_results: dict[str, dict[str, Any]] = {}
    combo_models: dict[str, Any] = {}
    for mode, disabled_prefixes in CONTACT_FEATURE_MODES.items():
        for model_family in CONTACT_MODEL_FAMILIES:
            if model_family == "linear_svc" and mode not in LINEAR_SVC_FEATURE_MODES:
                combo_results[f"{mode}/{model_family}"] = {
                    "status": "not_ready",
                    "reasons": [
                        "linear_svc is only evaluated on bounded low-dimensional modes "
                        f"{sorted(LINEAR_SVC_FEATURE_MODES)}"
                    ],
                    "feature_mode": mode,
                    "model_family": model_family,
                }
                continue
            result, model = train_single_contact_target(
                rows,
                label_key,
                min_examples=min_examples,
                min_videos=min_videos,
                feature_mode=mode,
                disabled_prefixes=disabled_prefixes,
                model_family=model_family,
            )
            key = f"{mode}/{model_family}"
            combo_results[key] = result
            if model is not None:
                combo_models[key] = model
    trained = [(key, result) for key, result in combo_results.items() if result.get("status") == "trained"]
    if not trained:
        first_key = next(iter(combo_results))
        result = dict(combo_results[first_key])
        result["model_mode_results"] = combo_results
        return result, None

    def selection_key(item: tuple[str, dict[str, Any]]) -> tuple[float, float, int, int]:
        _key, result = item
        feature_mode = str(result.get("feature_mode") or "")
        model_family = str(result.get("model_family") or "")
        feature_preference = {
            "no_visual_features_no_foot_track": 4,
            "no_visual_features": 3,
            "pose_only": 3,
            "no_visual_crop_no_foot_track": 3,
            "no_vision_embedding_no_foot_track": 2,
            "no_foot_track": 2,
            "no_visual_crop": 2,
            "no_vision_embedding": 1,
            "all_features": 0,
        }.get(feature_mode, 0)
        model_preference = {
            "logistic_regression": 2,
            "ridge_classifier": 2,
            "linear_svc": 2,
            "extra_trees": 1,
            "gradient_boosting": 0,
        }.get(model_family, 0)
        return (
            float(result.get("accuracy") or 0.0),
            float(result.get("balanced_accuracy") or 0.0),
            feature_preference,
            model_preference,
        )

    best_key, best_result = max(
        trained,
        key=selection_key,
    )
    result = dict(best_result)
    result["selected_feature_mode"] = result.get("feature_mode")
    result["selected_model_family"] = result.get("model_family")
    result["selected_model_mode_key"] = best_key
    result["model_mode_results"] = combo_results
    feature_mode_results: dict[str, dict[str, Any]] = {}
    model_family_results: dict[str, dict[str, Any]] = {}
    for mode in CONTACT_FEATURE_MODES:
        mode_trained = [(key, combo_results[key]) for key in combo_results if key.startswith(f"{mode}/") and combo_results[key].get("status") == "trained"]
        if mode_trained:
            _mode_key, mode_result = max(mode_trained, key=selection_key)
            feature_mode_results[mode] = mode_result
    for family in CONTACT_MODEL_FAMILIES:
        family_trained = [(key, combo_results[key]) for key in combo_results if key.endswith(f"/{family}") and combo_results[key].get("status") == "trained"]
        if family_trained:
            _family_key, family_result = max(family_trained, key=selection_key)
            model_family_results[family] = family_result
    result["feature_mode_results"] = feature_mode_results
    result["model_family_results"] = model_family_results
    return result, combo_models.get(best_key)


def train_contact_models(
    labeled_rows: list[dict[str, Any]],
    *,
    out_dir: Path,
    min_examples: int,
    min_videos: int,
    targets: tuple[str, ...] = CONTACT_TARGETS,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    target_results: dict[str, Any] = {}
    models: dict[str, Any] = {}
    model_feature_modes: dict[str, str] = {}
    model_families: dict[str, str] = {}
    for target in targets:
        result, model = train_single_contact_target_best_mode(
            labeled_rows,
            target,
            min_examples=min_examples,
            min_videos=min_videos,
        )
        target_results[target] = result
        if model is not None:
            models[target] = model
            model_feature_modes[target] = str(result.get("selected_feature_mode") or result.get("feature_mode") or "all_features")
            model_families[target] = str(result.get("selected_model_family") or result.get("model_family") or "logistic_regression")
    artifact_path = out_dir / "release_contact_classifier.joblib"
    if models:
        joblib.dump(
            {
                "schema_version": 1,
                "models": models,
                "model_feature_modes": model_feature_modes,
                "model_families": model_families,
                "feature_modes": CONTACT_FEATURE_MODES,
                "model_family_options": CONTACT_MODEL_FAMILIES,
                "targets": sorted(models),
                "numeric_features": NUMERIC_FEATURES,
                "boolean_features": BOOLEAN_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
                "gate_accuracy_threshold": CONTACT_GATE_ACCURACY,
            },
            artifact_path,
        )
    trained = {target: result for target, result in target_results.items() if result.get("status") == "trained"}
    return {
        "status": "trained" if trained else "not_ready",
        "artifact": str(artifact_path) if models else None,
        "targets": target_results,
    }


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
        f"- Rows with pose columns attached: `{summary['rows_with_pose_columns']}`",
        f"- Rows with pose present: `{summary['rows_with_pose_present']}`",
        f"- Rows with usable pose distance features: `{summary['rows_with_pose_features']}`",
        f"- Rows with visual crop features: `{summary.get('rows_with_visual_crop_features', 0)}`",
        f"- Rows with vision embedding features: `{summary.get('rows_with_vision_embedding_features', 0)}`",
        f"- Rows with reviewed contact labels: `{summary['rows_with_contact_labels']}`",
        f"- Event-file contact labels matched to rows: `{summary['event_file_label_match']['matched_examples']}` / `{summary['event_file_label_match']['label_examples']}`",
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
            f"- Contact side basis rows: `{summary.get('contact_side_basis_counts_in_rows', {})}`",
            f"- Contact surface rows: `{summary['contact_surface_counts_in_rows']}`",
            f"- Contact labels found in event files: `{summary['contact_type_counts_in_label_files']}`",
            f"- Contact surfaces found in event files: `{summary['contact_surface_counts_in_label_files']}`",
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
    if summary.get("contact_label_gaps"):
        lines.extend(["", "## Release Label Gaps", ""])
        lines.extend(
            [
                "| target | class counts | additional needed | status |",
                "| --- | --- | --- | --- |",
            ]
        )
        for target, gap in summary["contact_label_gaps"].items():
            status = "pass" if gap.get("passes") else "blocked"
            lines.append(
                f"| `{target}` | `{gap.get('class_counts')}` | "
                f"`{gap.get('additional_needed')}` | {status} |"
            )
    if summary.get("pose_coverage_by_contact_target"):
        lines.extend(["", "## Pose Coverage For Reviewed Contact Rows", ""])
        lines.extend(
            [
                "| target | rows | pose present | usable pose | pose status counts |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for target, coverage in summary["pose_coverage_by_contact_target"].items():
            lines.append(
                f"| `{target}` | {coverage.get('rows')} | {coverage.get('pose_present_rows')} | "
                f"{coverage.get('usable_pose_distance_rows')} | `{coverage.get('pose_status_counts')}` |"
            )
    if summary.get("contact_label_file_inventory"):
        inventory = summary["contact_label_file_inventory"]
        lines.extend(["", "## Event-File Contact Label Inventory", ""])
        lines.append(f"- Aggregate: `{inventory.get('aggregate')}`")
        lines.extend(
            [
                "",
                "| file | type | side | surface | unknown surface | trick labels |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in inventory.get("files", []):
            if not (row.get("contact_type_labels") or row.get("contact_side_labels") or row.get("contact_surface_labels")):
                continue
            lines.append(
                f"| `{row.get('file')}` | {row.get('contact_type_labels')} | {row.get('contact_side_labels')} | "
                f"{row.get('contact_surface_labels')} | {row.get('explicit_unknown_surface_labels')} | "
                f"`{row.get('trick_label_counts')}` |"
            )
    if summary.get("trained_models"):
        lines.extend(["", "## Trained Models", ""])
        trained_models = summary["trained_models"]
        lines.append(f"- Artifact: `{trained_models.get('artifact')}`")
        for target, result in (trained_models.get("targets") or {}).items():
            lines.extend(["", f"### `{target}`", ""])
            lines.append(f"- Status: `{result.get('status')}`")
            if result.get("reasons"):
                for reason in result["reasons"]:
                    lines.append(f"- Blocker: {reason}")
            if result.get("status") == "trained":
                lines.append(f"- Selected feature mode: `{result.get('selected_feature_mode') or result.get('feature_mode')}`")
                lines.append(f"- Selected model family: `{result.get('selected_model_family') or result.get('model_family')}`")
                lines.append(f"- Leave-one-video-out accuracy: `{result.get('accuracy'):.3f}`")
                if result.get("balanced_accuracy") is not None:
                    lines.append(f"- Leave-one-video-out balanced accuracy: `{result.get('balanced_accuracy'):.3f}`")
                lines.append(f"- Gate: `{result.get('gate')}` at >= `{result.get('gate_accuracy_threshold')}`")
                lines.append(f"- Release-scope gate: `{result.get('release_scope_gate')}`")
                if result.get("sequence_smoothed"):
                    smoothed = result["sequence_smoothed"]
                    lines.append(
                        f"- Sequence-smoothed side diagnostic: accuracy `{smoothed.get('accuracy'):.3f}`, "
                        f"balanced `{smoothed.get('balanced_accuracy'):.3f}`, "
                        f"changed rows `{smoothed.get('changed_rows')}`"
                    )
                    lines.append(f"- Sequence-smoothed side status: `{smoothed.get('status')}`; automatic side badges remain unpromoted.")
                for blocker in result.get("gate_blockers") or []:
                    lines.append(f"- Gate blocker: {blocker}")
                if result.get("release_scope_blockers"):
                    for blocker in result["release_scope_blockers"]:
                        lines.append(f"- Release-scope blocker: {blocker}")
                if result.get("side_basis_counts"):
                    lines.append(f"- Side basis counts: `{result.get('side_basis_counts')}`")
                if result.get("release_class_coverage"):
                    coverage = result["release_class_coverage"]
                    lines.append(
                        f"- Release class coverage: `{coverage.get('class_counts')}` "
                        f"(min `{coverage.get('min_examples_per_class')}` per class)"
                    )
                lines.append(f"- Rows: `{result.get('rows')}`")
                lines.append(f"- Label counts: `{result.get('label_counts')}`")
                lines.append(f"- Prediction counts: `{result.get('prediction_counts')}`")
                lines.append(f"- Confusion: `{result.get('confusion')}`")
                if result.get("per_class_recall"):
                    lines.append(f"- Per-class recall: `{result.get('per_class_recall')}`")
                if result.get("feature_mode_results"):
                    lines.append("- Feature-mode ablation:")
                    for mode, mode_result in result["feature_mode_results"].items():
                        if mode_result.get("status") == "trained":
                            lines.append(
                                f"  - `{mode}`: accuracy `{mode_result.get('accuracy'):.3f}`, "
                                f"balanced `{mode_result.get('balanced_accuracy'):.3f}`, "
                                f"model `{mode_result.get('selected_model_family') or mode_result.get('model_family')}`, "
                                f"gate `{mode_result.get('gate')}`"
                            )
                        else:
                            lines.append(f"  - `{mode}`: status `{mode_result.get('status')}`")
                if result.get("model_family_results"):
                    lines.append("- Model-family ablation:")
                    for family, family_result in result["model_family_results"].items():
                        if family_result.get("status") == "trained":
                            lines.append(
                                f"  - `{family}`: accuracy `{family_result.get('accuracy'):.3f}`, "
                                f"balanced `{family_result.get('balanced_accuracy'):.3f}`, "
                                f"feature mode `{family_result.get('selected_feature_mode') or family_result.get('feature_mode')}`, "
                                f"gate `{family_result.get('gate')}`"
                            )
                        else:
                            lines.append(f"  - `{family}`: status `{family_result.get('status')}`")
                if result.get("selective_accuracy"):
                    lines.append("- Selective accuracy by confidence:")
                    for row in result["selective_accuracy"]:
                        accuracy = row.get("accuracy")
                        accuracy_text = "n/a" if accuracy is None else f"{accuracy:.3f}"
                        lines.append(
                            f"  - conf >= `{row.get('threshold')}`: kept `{row.get('kept')}`, "
                            f"coverage `{row.get('coverage'):.3f}`, accuracy `{accuracy_text}`"
                        )
                if result.get("errors"):
                    lines.append("- First errors:")
                    for error in result["errors"][:12]:
                        lines.append(
                            f"  - {error.get('video_id')} @ {error.get('candidate_time_sec')}s: "
                            f"label `{error.get('label')}`, predicted `{error.get('prediction')}`"
                        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_contact_classifier(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.dataset_dir / "touch_training_candidates.jsonl")
    rows += read_jsonl(args.dataset_dir / "touch_training_test_frozen.jsonl")
    label_examples = load_label_contact_examples(args.labels_dir)
    label_inventory = contact_label_file_inventory(args.labels_dir)
    rows, label_match = attach_event_file_contact_labels(rows, label_examples, tolerance_sec=args.label_match_tolerance_sec)
    summary = readiness_summary(rows, label_examples, min_examples=args.min_examples, min_videos=args.min_videos)
    labeled_rows = rows_with_contact_labels(rows)
    majority_baselines = {}
    if summary["status"] == "ready_for_training":
        for key in ("contact_type", "contact_side", "contact_surface"):
            result = leave_one_video_out_majority_baseline(labeled_rows, key)
            if result:
                majority_baselines[key] = result
    trained_models = None
    if summary["status"] == "ready_for_training":
        trained_models = train_contact_models(
            labeled_rows,
            out_dir=args.out_dir,
            min_examples=args.min_examples,
            min_videos=args.min_videos,
        )
    summary.update(
        {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_dir": str(args.dataset_dir),
            "labels_dir": str(args.labels_dir),
            "event_file_label_match": label_match,
            "contact_label_file_inventory": label_inventory,
            "contact_label_gaps": contact_label_gap_summary(rows),
            "pose_coverage_by_contact_target": pose_coverage_by_contact_target(rows),
            "majority_baselines": majority_baselines,
            "trained_models": trained_models,
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
    parser.add_argument("--label-match-tolerance-sec", type=float, default=0.08)
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
