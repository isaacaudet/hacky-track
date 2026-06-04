#!/usr/bin/env python3
"""Render visual audit strips for release contact classifier errors.

This is deliberately diagnostic: it does not change labels or model outputs. It
replays the current clip-disjoint contact classifier errors and writes a compact
image strip per error so side/surface/stall failures can be inspected from the
actual video frame instead of from scalar metrics alone.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import train_release_contact_classifier as contact


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "runs/release-27-public/touch_corpus_v1"
DEFAULT_DATASET_DIR = DEFAULT_CORPUS / "touch_training_dataset_v1"
DEFAULT_LABELS_DIR = DEFAULT_CORPUS / "visual_touch_labels"
DEFAULT_REVIEW_MANIFEST = DEFAULT_CORPUS / "touch_review_manifest.json"
DEFAULT_OUT_DIR = DEFAULT_CORPUS / "release_contact_classifier_v1" / "contact_error_audit"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def video_paths_by_name(review_manifest: Path) -> dict[str, Path]:
    doc = read_json(review_manifest)
    return {Path(str(item["video_name"])).name: Path(str(item["video_path"])) for item in doc.get("items", [])}


def row_lookup_key(row: dict[str, Any]) -> tuple[str, float]:
    return str(row.get("video_id")), round(float(row.get("candidate_time_sec") or 0.0), 6)


def load_labeled_rows(dataset_dir: Path, labels_dir: Path, tolerance_sec: float) -> list[dict[str, Any]]:
    rows = contact.read_jsonl(dataset_dir / "touch_training_candidates.jsonl")
    rows += contact.read_jsonl(dataset_dir / "touch_training_test_frozen.jsonl")
    labels = contact.load_label_contact_examples(labels_dir)
    matched, _summary = contact.attach_event_file_contact_labels(rows, labels, tolerance_sec=tolerance_sec)
    return contact.rows_with_contact_labels(matched)


def infer_failure_bucket(target: str, row: dict[str, Any], label: str, prediction: str) -> str:
    if target == "contact_side":
        pose_side = row.get("pose_geometry_nearest_side") or row.get("pose_candidate_side")
        if pose_side and pose_side != label:
            return "pose_side_disagreement"
        if row.get("pose_missing") or row.get("pose_feature_status") != "ok":
            return "pose_missing"
        return "side_visual_ambiguity"
    if target == "contact_surface":
        pose_surface = row.get("pose_geometry_nearest_surface")
        if pose_surface and pose_surface != label:
            return "pose_surface_disagreement"
        if row.get("contact_surface") in {"inner", "outer"} and row.get("visual_crop_feature_status") != "ok":
            return "surface_missing_crop"
        return "surface_label_or_geometry_ambiguity"
    if target == "contact_type":
        if row.get("in_stall_window"):
            return "stall_window_confusion"
        if row.get("pose_nearest_lower_part") and "knee" in str(row.get("pose_nearest_lower_part")):
            return "pose_knee_confusion"
        return "type_motion_ambiguity"
    return "unknown"


def read_frame_at(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def ball_xy(row: dict[str, Any]) -> tuple[float, float] | None:
    for prefix in ("visual_crop_ball", "pose_ball", "flow_ball"):
        x = row.get(f"{prefix}_x")
        y = row.get(f"{prefix}_y")
        if x is None or y is None:
            continue
        try:
            return float(x), float(y)
        except (TypeError, ValueError):
            continue
    return None


def draw_error_frame(frame: np.ndarray, row: dict[str, Any], target: str, label: str, prediction: str, delta_label: str) -> np.ndarray:
    out = frame.copy()
    xy = ball_xy(row)
    if xy is not None:
        x, y = int(round(xy[0])), int(round(xy[1]))
        cv2.circle(out, (x, y), 24, (0, 255, 255), 3)
        cv2.circle(out, (x, y), 4, (0, 0, 255), -1)
    title = f"{target}: label={label} pred={prediction} {delta_label}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 128), (0, 0, 0), -1)
    cv2.putText(out, title, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    meta = (
        f"t={float(row.get('candidate_time_sec') or 0):.3f}s "
        f"pose_side={row.get('pose_geometry_nearest_side')} pose_surface={row.get('pose_geometry_nearest_surface')} "
        f"crop={row.get('visual_crop_feature_status')}"
    )
    cv2.putText(out, meta, (24, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (210, 230, 255), 2, cv2.LINE_AA)
    return out


def render_error_strip(
    *,
    row: dict[str, Any],
    target: str,
    label: str,
    prediction: str,
    video_path: Path,
    out_path: Path,
    width_px: int,
) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    center = int(round(float(row.get("candidate_time_sec") or 0.0) * fps))
    frames: list[np.ndarray] = []
    try:
        for offset in (-3, 0, 3):
            frame = read_frame_at(cap, max(0, center + offset))
            if frame is None:
                continue
            drawn = draw_error_frame(frame, row, target, label, prediction, f"frame {offset:+d}")
            scale = width_px / max(1, drawn.shape[1])
            resized = cv2.resize(drawn, (width_px, int(round(drawn.shape[0] * scale))), interpolation=cv2.INTER_AREA)
            frames.append(resized)
    finally:
        cap.release()
    if not frames:
        return False
    max_h = max(frame.shape[0] for frame in frames)
    padded = []
    for frame in frames:
        if frame.shape[0] < max_h:
            pad = np.zeros((max_h - frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            frame = np.vstack([frame, pad])
        padded.append(frame)
    strip = np.hstack(padded)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), strip))


def build_error_rows(
    labeled_rows: list[dict[str, Any]],
    *,
    min_examples: int,
    min_videos: int,
) -> list[dict[str, Any]]:
    by_key = {row_lookup_key(row): row for row in labeled_rows}
    errors: list[dict[str, Any]] = []
    for target in contact.CONTACT_TARGETS:
        result, _model = contact.train_single_contact_target_best_mode(
            labeled_rows,
            target,
            min_examples=min_examples,
            min_videos=min_videos,
        )
        for error in result.get("errors") or []:
            key = (str(error.get("video_id")), round(float(error.get("candidate_time_sec") or 0.0), 6))
            row = by_key.get(key)
            if row is None:
                continue
            bucket = infer_failure_bucket(target, row, str(error.get("label")), str(error.get("prediction")))
            errors.append(
                {
                    **error,
                    "target": target,
                    "selected_feature_mode": result.get("selected_feature_mode") or result.get("feature_mode"),
                    "failure_bucket": bucket,
                    "video_name": row.get("video_name"),
                    "split": row.get("split"),
                    "pose_geometry_nearest_side": row.get("pose_geometry_nearest_side"),
                    "pose_geometry_nearest_surface": row.get("pose_geometry_nearest_surface"),
                    "pose_nearest_lower_part": row.get("pose_nearest_lower_part"),
                    "visual_crop_feature_status": row.get("visual_crop_feature_status"),
                }
            )
    return errors


def write_report(path: Path, summary: dict[str, Any], errors: list[dict[str, Any]]) -> None:
    by_target = Counter(row["target"] for row in errors)
    by_bucket = Counter(row["failure_bucket"] for row in errors)
    lines = [
        "# Contact Classifier Error Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Errors: `{len(errors)}`",
        f"- Contact sheet dir: `{summary['strip_dir']}`",
        "",
        "## Counts",
        "",
        f"- By target: `{dict(by_target)}`",
        f"- By failure bucket: `{dict(by_bucket)}`",
        "",
        "## Error Rows",
        "",
        "| target | video | time | label | pred | bucket | strip |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for row in errors:
        strip = row.get("strip_path")
        strip_cell = f"[open]({strip})" if strip else ""
        lines.append(
            f"| {row['target']} | `{row.get('video_id')}` | {float(row.get('candidate_time_sec') or 0):.3f} | "
            f"{row.get('label')} | {row.get('prediction')} | {row.get('failure_bucket')} | {strip_cell} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir.resolve()
    strip_dir = out_dir / "strips"
    if strip_dir.exists():
        for stale_path in strip_dir.glob("*.jpg"):
            stale_path.unlink()
    labeled_rows = load_labeled_rows(args.dataset_dir.resolve(), args.labels_dir.resolve(), args.label_match_tolerance_sec)
    errors = build_error_rows(labeled_rows, min_examples=args.min_examples, min_videos=args.min_videos)
    video_paths = video_paths_by_name(args.review_manifest.resolve())
    for index, error in enumerate(errors):
        video_name = str(error.get("video_name") or "")
        video_path = video_paths.get(video_name)
        if video_path is None:
            error["strip_status"] = "missing_video_path"
            continue
        safe_video = str(error.get("video_id") or "video").replace("/", "-")
        strip_path = strip_dir / f"{index:03d}_{error['target']}_{safe_video}_{float(error.get('candidate_time_sec') or 0):.3f}.jpg"
        ok = render_error_strip(
            row=next(row for row in labeled_rows if row_lookup_key(row) == (str(error.get("video_id")), round(float(error.get("candidate_time_sec") or 0.0), 6))),
            target=str(error["target"]),
            label=str(error["label"]),
            prediction=str(error["prediction"]),
            video_path=video_path,
            out_path=strip_path,
            width_px=args.strip_panel_width_px,
        )
        error["strip_status"] = "ok" if ok else "render_failed"
        if ok:
            error["strip_path"] = str(strip_path)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "dataset_dir": str(args.dataset_dir),
        "labels_dir": str(args.labels_dir),
        "review_manifest": str(args.review_manifest),
        "out_dir": str(out_dir),
        "strip_dir": str(strip_dir),
        "errors": len(errors),
        "by_target": dict(Counter(row["target"] for row in errors)),
        "by_failure_bucket": dict(Counter(row["failure_bucket"] for row in errors)),
    }
    write_json(out_dir / "contact_error_audit_summary.json", summary)
    write_jsonl(out_dir / "contact_error_audit.jsonl", errors)
    write_report(out_dir / "contact_error_audit_report.md", summary, errors)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render visual strips for release contact classifier errors")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--label-match-tolerance-sec", type=float, default=0.08)
    parser.add_argument("--min-examples", type=int, default=contact.DEFAULT_MIN_EXAMPLES)
    parser.add_argument("--min-videos", type=int, default=contact.DEFAULT_MIN_VIDEOS)
    parser.add_argument("--strip-panel-width-px", type=int, default=420)
    return parser.parse_args()


def main() -> None:
    summary = run_audit(parse_args())
    print(f"status: {summary['status']}")
    print(f"report: {Path(summary['out_dir']) / 'contact_error_audit_report.md'}")
    print(json.dumps({"errors": summary["errors"], "by_target": summary["by_target"], "by_failure_bucket": summary["by_failure_bucket"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
