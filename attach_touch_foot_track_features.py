#!/usr/bin/env python3
"""Attach per-clip foot identity/continuity features to touch candidates.

This stage does not run a new detector. It builds on RTMW foot geometry already
attached by ``attach_touch_pose_features.py`` and adds temporal features that
describe which visible foot-like track the ball is nearest across nearby
candidate frames.

Important release distinction:
- ``foot_track_*`` features are label-free and can be used by automatic models.
- ``manual_foot_*`` fields are calibrated from reviewed contact labels in the
  same clip. They are useful for visual-corrected/manual HUD workflows and
  diagnostics only, and are intentionally excluded from automatic model features.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import train_release_contact_classifier as contact


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "runs/release-27-public/touch_corpus_v1"
DEFAULT_DATASET_DIR = DEFAULT_CORPUS / "touch_training_dataset_v1"
DEFAULT_LABELS_DIR = DEFAULT_CORPUS / "visual_touch_labels"
FOOT_TRACK_FEATURE_VERSION = 1
SIDE_VALUES = {"left", "right"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finite_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def valid_side(value: Any) -> str | None:
    text = str(value or "").lower()
    return text if text in SIDE_VALUES else None


def other_side(side: str | None) -> str | None:
    if side == "left":
        return "right"
    if side == "right":
        return "left"
    return None


def row_time(row: dict[str, Any]) -> float:
    return float(row.get("candidate_time_sec") or 0.0)


def nearest_pose_side(row: dict[str, Any]) -> str | None:
    side = valid_side(row.get("pose_geometry_nearest_side"))
    if side:
        return side
    left_dist = finite_float(row.get("pose_left_foot_min_dist_px"))
    right_dist = finite_float(row.get("pose_right_foot_min_dist_px"))
    if left_dist is None and right_dist is None:
        return None
    if left_dist is None:
        return "right"
    if right_dist is None:
        return "left"
    return "left" if left_dist <= right_dist else "right"


def side_distance(row: dict[str, Any], side: str) -> float | None:
    return finite_float(row.get(f"pose_{side}_foot_min_dist_px"))


def side_norm_distance(row: dict[str, Any], side: str) -> float | None:
    return finite_float(row.get(f"pose_{side}_foot_min_dist_norm_shank"))


def side_surface_guess(row: dict[str, Any], side: str) -> str | None:
    value = str(row.get(f"pose_{side}_edge_surface_guess") or "")
    return value if value in {"inner", "outer"} else None


def side_axis_feature(row: dict[str, Any], side: str, suffix: str) -> float | None:
    return finite_float(row.get(f"pose_{side}_{suffix}"))


def base_features(status: str) -> dict[str, Any]:
    return {
        "foot_track_feature_version": FOOT_TRACK_FEATURE_VERSION,
        "foot_track_feature_status": status,
        "foot_track_nearest_pose_side": None,
        "foot_track_nearest_pose_side_is_left": False,
        "foot_track_nearest_pose_side_is_right": False,
        "foot_track_nearest_pose_side_confidence": None,
        "foot_track_pose_side_margin_px": None,
        "foot_track_pose_side_margin_norm_nearest": None,
        "foot_track_left_dist_px": None,
        "foot_track_right_dist_px": None,
        "foot_track_left_right_dist_delta_px": None,
        "foot_track_abs_left_right_dist_delta_px": None,
        "foot_track_nearest_dist_px": None,
        "foot_track_nearest_dist_norm_shank": None,
        "foot_track_nearest_surface_guess": None,
        "foot_track_nearest_surface_margin_px": None,
        "foot_track_nearest_ball_axis_projection": None,
        "foot_track_nearest_ball_axis_lateral_px": None,
        "foot_track_prev_pose_side": None,
        "foot_track_next_pose_side": None,
        "foot_track_prev_same_pose_side": False,
        "foot_track_next_same_pose_side": False,
        "foot_track_prev_time_delta_sec": None,
        "foot_track_next_time_delta_sec": None,
        "foot_track_window_valid_count": 0,
        "foot_track_window_left_count": 0,
        "foot_track_window_right_count": 0,
        "foot_track_window_nearest_side_fraction": None,
        "foot_track_window_switch_count": 0,
        "foot_track_left_dist_velocity_before_px_s": None,
        "foot_track_right_dist_velocity_before_px_s": None,
        "foot_track_left_dist_velocity_after_px_s": None,
        "foot_track_right_dist_velocity_after_px_s": None,
        "foot_track_nearest_dist_velocity_before_px_s": None,
        "foot_track_nearest_dist_velocity_after_px_s": None,
    }


def sorted_video_rows(rows: list[dict[str, Any]]) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row.get("video_id") or row.get("video_name") or "unknown")].append((index, row))
    for video_id in grouped:
        grouped[video_id].sort(key=lambda item: row_time(item[1]))
    return grouped


def nearest_valid_neighbor(
    sequence: list[tuple[int, dict[str, Any]]],
    pos: int,
    *,
    direction: int,
) -> tuple[int, dict[str, Any], str] | None:
    step = 1 if direction >= 0 else -1
    cursor = pos + step
    while 0 <= cursor < len(sequence):
        side = nearest_pose_side(sequence[cursor][1])
        if side:
            index, row = sequence[cursor]
            return index, row, side
        cursor += step
    return None


def distance_velocity(current: dict[str, Any], neighbor: dict[str, Any] | None, side: str) -> float | None:
    if neighbor is None:
        return None
    cur = side_distance(current, side)
    prior = side_distance(neighbor, side)
    if cur is None or prior is None:
        return None
    dt = row_time(current) - row_time(neighbor)
    if abs(dt) <= 1e-6:
        return None
    return (cur - prior) / dt


def add_temporal_features(
    rows: list[dict[str, Any]],
    *,
    window_sec: float,
) -> list[dict[str, Any]]:
    out = [dict(row) for row in rows]
    grouped = sorted_video_rows(out)
    for _video_id, sequence in grouped.items():
        for pos, (original_index, row) in enumerate(sequence):
            side = nearest_pose_side(row)
            left_dist = side_distance(row, "left")
            right_dist = side_distance(row, "right")
            if side is None:
                status = "missing_pose_side" if row.get("pose_feature_status") == "ok" else str(row.get("pose_feature_status") or "missing_pose")
                out[original_index].update(base_features(status))
                continue

            features = base_features("ok")
            nearest_dist = side_distance(row, side)
            other_dist = side_distance(row, other_side(side) or "")
            margin = finite_float(row.get("pose_geometry_side_margin_px"))
            if margin is None and left_dist is not None and right_dist is not None:
                margin = abs(left_dist - right_dist)
            confidence = None
            if nearest_dist is not None and other_dist is not None:
                confidence = abs(other_dist - nearest_dist) / max(1.0, other_dist + nearest_dist)
            elif margin is not None and nearest_dist is not None:
                confidence = margin / max(1.0, margin + nearest_dist)

            features.update(
                {
                    "foot_track_nearest_pose_side": side,
                    "foot_track_nearest_pose_side_is_left": side == "left",
                    "foot_track_nearest_pose_side_is_right": side == "right",
                    "foot_track_nearest_pose_side_confidence": confidence,
                    "foot_track_pose_side_margin_px": margin,
                    "foot_track_pose_side_margin_norm_nearest": None if margin is None or nearest_dist is None else margin / max(1.0, nearest_dist),
                    "foot_track_left_dist_px": left_dist,
                    "foot_track_right_dist_px": right_dist,
                    "foot_track_left_right_dist_delta_px": None if left_dist is None or right_dist is None else left_dist - right_dist,
                    "foot_track_abs_left_right_dist_delta_px": None if left_dist is None or right_dist is None else abs(left_dist - right_dist),
                    "foot_track_nearest_dist_px": nearest_dist,
                    "foot_track_nearest_dist_norm_shank": side_norm_distance(row, side),
                    "foot_track_nearest_surface_guess": side_surface_guess(row, side),
                    "foot_track_nearest_surface_margin_px": side_axis_feature(row, side, "edge_surface_margin_px"),
                    "foot_track_nearest_ball_axis_projection": side_axis_feature(row, side, "ball_axis_projection"),
                    "foot_track_nearest_ball_axis_lateral_px": side_axis_feature(row, side, "ball_axis_lateral_px"),
                }
            )

            prev = nearest_valid_neighbor(sequence, pos, direction=-1)
            next_item = nearest_valid_neighbor(sequence, pos, direction=1)
            prev_row = prev[1] if prev else None
            next_row = next_item[1] if next_item else None
            prev_side = prev[2] if prev else None
            next_side = next_item[2] if next_item else None
            features["foot_track_prev_pose_side"] = prev_side
            features["foot_track_next_pose_side"] = next_side
            features["foot_track_prev_same_pose_side"] = prev_side == side
            features["foot_track_next_same_pose_side"] = next_side == side
            features["foot_track_prev_time_delta_sec"] = None if prev_row is None else row_time(row) - row_time(prev_row)
            features["foot_track_next_time_delta_sec"] = None if next_row is None else row_time(next_row) - row_time(row)
            for track_side in ("left", "right"):
                before = distance_velocity(row, prev_row, track_side)
                next_velocity = None if next_row is None else distance_velocity(next_row, row, track_side)
                after = None if next_velocity is None else -next_velocity
                features[f"foot_track_{track_side}_dist_velocity_before_px_s"] = before
                features[f"foot_track_{track_side}_dist_velocity_after_px_s"] = after
            features["foot_track_nearest_dist_velocity_before_px_s"] = features.get(f"foot_track_{side}_dist_velocity_before_px_s")
            features["foot_track_nearest_dist_velocity_after_px_s"] = features.get(f"foot_track_{side}_dist_velocity_after_px_s")

            window_sides: list[str] = []
            last_side = None
            switch_count = 0
            current_time = row_time(row)
            for _neighbor_index, neighbor in sequence:
                if abs(row_time(neighbor) - current_time) > window_sec:
                    continue
                neighbor_side = nearest_pose_side(neighbor)
                if not neighbor_side:
                    continue
                window_sides.append(neighbor_side)
                if last_side is not None and neighbor_side != last_side:
                    switch_count += 1
                last_side = neighbor_side
            counts = Counter(window_sides)
            features["foot_track_window_valid_count"] = len(window_sides)
            features["foot_track_window_left_count"] = counts.get("left", 0)
            features["foot_track_window_right_count"] = counts.get("right", 0)
            features["foot_track_window_nearest_side_fraction"] = None if not window_sides else counts.get(side, 0) / len(window_sides)
            features["foot_track_window_switch_count"] = switch_count
            out[original_index].update(features)
    return out


def row_key(row: dict[str, Any]) -> tuple[str, float]:
    return str(row.get("video_id") or row.get("video_name") or "unknown"), round(row_time(row), 6)


def manual_calibration_by_video(rows: list[dict[str, Any]], labels_dir: Path, tolerance_sec: float) -> dict[str, dict[str, Any]]:
    label_examples = contact.load_label_contact_examples(labels_dir)
    matched_rows, _summary = contact.attach_event_file_contact_labels(rows, label_examples, tolerance_sec=tolerance_sec)
    labeled = contact.rows_with_contact_labels(matched_rows)
    votes: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: {"left": Counter(), "right": Counter()})
    for row in labeled:
        wearer_side = valid_side(row.get("contact_side"))
        pose_side = nearest_pose_side(row)
        video_id = str(row.get("video_id") or row.get("video_name") or "unknown")
        if wearer_side and pose_side:
            votes[video_id][pose_side][wearer_side] += 1

    calibration: dict[str, dict[str, Any]] = {}
    for video_id, side_votes in votes.items():
        mapping: dict[str, str | None] = {}
        confidence: dict[str, float | None] = {}
        counts: dict[str, dict[str, int]] = {}
        for pose_side in ("left", "right"):
            counter = side_votes[pose_side]
            counts[pose_side] = dict(counter)
            if not counter:
                mapping[pose_side] = None
                confidence[pose_side] = None
                continue
            label, count = counter.most_common(1)[0]
            mapping[pose_side] = label
            confidence[pose_side] = count / sum(counter.values())
        calibration[video_id] = {
            "video_id": video_id,
            "status": "calibrated" if any(mapping.values()) else "no_pose_label_votes",
            "mapping": mapping,
            "confidence": confidence,
            "counts": counts,
        }
    return calibration


def add_manual_calibration_fields(
    rows: list[dict[str, Any]],
    *,
    calibration: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        video_id = str(row.get("video_id") or row.get("video_name") or "unknown")
        pose_side = nearest_pose_side(row)
        item = calibration.get(video_id)
        mapped_side = None
        mapped_conf = None
        status = "no_clip_calibration"
        if item and pose_side:
            mapped_side = (item.get("mapping") or {}).get(pose_side)
            mapped_conf = (item.get("confidence") or {}).get(pose_side)
            status = "calibrated" if mapped_side else "no_pose_side_mapping"
        elif item:
            status = "missing_pose_side"
        new_row.update(
            {
                "manual_foot_calibration_status": status,
                "manual_foot_pose_side": pose_side,
                "manual_foot_calibrated_side": mapped_side,
                "manual_foot_calibrated_side_confidence": mapped_conf,
            }
        )
        out.append(new_row)
    return out


def attach_dataset(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else dataset_dir
    train_path = dataset_dir / "touch_training_candidates.jsonl"
    test_path = dataset_dir / "touch_training_test_frozen.jsonl"
    train_rows = read_jsonl(train_path)
    test_rows = read_jsonl(test_path)
    all_rows_for_calibration = train_rows + test_rows
    calibration = manual_calibration_by_video(all_rows_for_calibration, args.labels_dir.resolve(), args.label_match_tolerance_sec)

    train_out = add_temporal_features(train_rows, window_sec=args.window_sec)
    test_out = add_temporal_features(test_rows, window_sec=args.window_sec)
    train_out = add_manual_calibration_fields(train_out, calibration=calibration)
    test_out = add_manual_calibration_fields(test_out, calibration=calibration)
    if any(row.get("split") == "test_frozen" for row in train_out):
        raise AssertionError("test_frozen row leaked into train/validation foot-track output")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "touch_training_candidates.jsonl", train_out)
    write_jsonl(out_dir / "touch_training_test_frozen.jsonl", test_out)
    combined = train_out + test_out
    per_video = []
    for video_id, sequence in sorted(sorted_video_rows(combined).items()):
        video_rows = [row for _idx, row in sequence]
        per_video.append(
            {
                "video_id": video_id,
                "rows": len(video_rows),
                "ok_rows": sum(1 for row in video_rows if row.get("foot_track_feature_status") == "ok"),
                "pose_side_rows": sum(1 for row in video_rows if valid_side(row.get("foot_track_nearest_pose_side"))),
                "manual_calibrated_rows": sum(1 for row in video_rows if row.get("manual_foot_calibration_status") == "calibrated"),
                "calibration": calibration.get(video_id),
            }
        )
    manifest = {
        "schema_version": 1,
        "foot_track_feature_version": FOOT_TRACK_FEATURE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "features_attached" if any(row.get("foot_track_feature_status") == "ok" for row in combined) else "no_foot_track_features",
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "labels_dir": str(args.labels_dir.resolve()),
        "label_match_tolerance_sec": args.label_match_tolerance_sec,
        "window_sec": args.window_sec,
        "train_val": {
            "rows": len(train_out),
            "ok_rows": sum(1 for row in train_out if row.get("foot_track_feature_status") == "ok"),
            "manual_calibrated_rows": sum(1 for row in train_out if row.get("manual_foot_calibration_status") == "calibrated"),
        },
        "test_frozen": {
            "rows": len(test_out),
            "ok_rows": sum(1 for row in test_out if row.get("foot_track_feature_status") == "ok"),
            "manual_calibrated_rows": sum(1 for row in test_out if row.get("manual_foot_calibration_status") == "calibrated"),
        },
        "videos": per_video,
        "notes": [
            "foot_track_* fields are label-free automatic features.",
            "manual_foot_* fields are derived from reviewed labels and must not be used for automatic release metrics.",
            "No CoTracker/SAM2 dependency is required for this baseline; RTMW foot geometry supplies the foot candidates.",
        ],
    }
    write_json(out_dir / "touch_foot_track_feature_manifest.json", manifest)
    write_report(out_dir / "touch_foot_track_feature_report.md", manifest)
    return manifest


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Touch Foot-Track Feature Attachment",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Dataset dir: `{manifest['dataset_dir']}`",
        f"- Out dir: `{manifest['out_dir']}`",
        f"- Window: `{manifest['window_sec']}` sec",
        f"- Train/val rows: `{manifest['train_val']['rows']}`",
        f"- Train/val foot-track rows: `{manifest['train_val']['ok_rows']}`",
        f"- Train/val manual-calibrated rows: `{manifest['train_val']['manual_calibrated_rows']}`",
        f"- Frozen-test rows: `{manifest['test_frozen']['rows']}`",
        f"- Frozen-test foot-track rows: `{manifest['test_frozen']['ok_rows']}`",
        f"- Frozen-test manual-calibrated rows: `{manifest['test_frozen']['manual_calibrated_rows']}`",
        "",
        "## Per-Video",
        "",
        "| video | rows | foot-track rows | manual-calibrated rows | calibration |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in manifest["videos"]:
        calibration = row.get("calibration") or {}
        lines.append(
            f"| `{row['video_id']}` | {row['rows']} | {row['ok_rows']} | "
            f"{row['manual_calibrated_rows']} | `{calibration.get('mapping')}` |"
        )
    lines.extend(["", "## Notes", ""])
    for note in manifest.get("notes", []):
        lines.append(f"- {note}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach per-clip foot identity/continuity features to touch candidates")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--label-match-tolerance-sec", type=float, default=0.08)
    parser.add_argument("--window-sec", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    manifest = attach_dataset(parse_args())
    out_dir = Path(manifest["out_dir"])
    print(f"manifest: {out_dir / 'touch_foot_track_feature_manifest.json'}")
    print(f"report:   {out_dir / 'touch_foot_track_feature_report.md'}")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "train_val_rows": manifest["train_val"]["rows"],
                "train_val_ok_rows": manifest["train_val"]["ok_rows"],
                "test_frozen_rows": manifest["test_frozen"]["rows"],
                "test_frozen_ok_rows": manifest["test_frozen"]["ok_rows"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
