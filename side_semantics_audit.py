#!/usr/bin/env python3
"""Audit what reviewed `left/right` side labels mean in egocentric footage.

The release side classifier is failing below the majority baseline. Before
adding more model capacity, this diagnostic compares reviewed side labels
against several possible conventions:

- RTMW anatomical left/right from nearest foot keypoints.
- Flipped RTMW side.
- Screen side of the tracked ball.
- Flipped screen side.

If no convention aligns well per-video, the problem is semantic/label
definition rather than another scalar feature.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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
DEFAULT_OUT_DIR = DEFAULT_CORPUS / "release_contact_classifier_v1" / "side_semantics_audit"
SIDE_VALUES = {"left", "right"}


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


def video_paths_by_name(review_manifest: Path) -> dict[str, Path]:
    doc = read_json(review_manifest)
    return {Path(str(item["video_name"])).name: Path(str(item["video_path"])) for item in doc.get("items", [])}


def opposite(side: str | None) -> str | None:
    if side == "left":
        return "right"
    if side == "right":
        return "left"
    return None


def valid_side(value: Any) -> str | None:
    text = str(value or "").lower()
    return text if text in SIDE_VALUES else None


def screen_ball_x_norm(row: dict[str, Any]) -> float | None:
    value = row.get("visual_ball_x_norm")
    if value is None:
        x = row.get("visual_crop_ball_x") or row.get("pose_ball_x") or row.get("flow_ball_x")
        width = row.get("visual_frame_width_px")
        if x is not None and width:
            try:
                value = float(x) / float(width)
            except (TypeError, ValueError, ZeroDivisionError):
                value = None
    if value is None:
        return None
    try:
        x_norm = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x_norm):
        return None
    return x_norm


def screen_side(row: dict[str, Any], *, deadzone: float = 0.0) -> str | None:
    x_norm = screen_ball_x_norm(row)
    if x_norm is None:
        return None
    if abs(x_norm - 0.5) <= deadzone:
        return None
    return "left" if x_norm < 0.5 else "right"


def pose_side(row: dict[str, Any]) -> str | None:
    return valid_side(row.get("pose_geometry_nearest_side") or row.get("pose_candidate_side"))


def side_predictions(row: dict[str, Any]) -> dict[str, str | None]:
    anatomical = pose_side(row)
    screen = screen_side(row)
    screen_deadzone = screen_side(row, deadzone=0.08)
    return {
        "pose_anatomical": anatomical,
        "pose_flipped": opposite(anatomical),
        "screen_ball": screen,
        "screen_ball_flipped": opposite(screen),
        "screen_ball_deadzone_0_08": screen_deadzone,
        "screen_ball_deadzone_0_08_flipped": opposite(screen_deadzone),
    }


def load_side_rows(dataset_dir: Path, labels_dir: Path, tolerance_sec: float) -> list[dict[str, Any]]:
    rows = contact.read_jsonl(dataset_dir / "touch_training_candidates.jsonl")
    rows += contact.read_jsonl(dataset_dir / "touch_training_test_frozen.jsonl")
    labels = contact.load_label_contact_examples(labels_dir)
    matched, _summary = contact.attach_event_file_contact_labels(rows, labels, tolerance_sec=tolerance_sec)
    return [row for row in contact.rows_with_contact_labels(matched) if valid_side(row.get("contact_side"))]


def score_predictor(rows: list[dict[str, Any]], predictor: str) -> dict[str, Any]:
    covered = 0
    correct = 0
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        label = valid_side(row.get("contact_side"))
        pred = side_predictions(row).get(predictor)
        if label is None or pred is None:
            continue
        covered += 1
        correct += int(label == pred)
        confusion[label][pred] += 1
    return {
        "predictor": predictor,
        "rows": len(rows),
        "covered": covered,
        "coverage": covered / len(rows) if rows else 0.0,
        "correct": correct,
        "accuracy": correct / covered if covered else None,
        "confusion": {label: dict(counter) for label, counter in confusion.items()},
    }


def score_all(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    predictors = [
        "pose_anatomical",
        "pose_flipped",
        "screen_ball",
        "screen_ball_flipped",
        "screen_ball_deadzone_0_08",
        "screen_ball_deadzone_0_08_flipped",
    ]
    return {predictor: score_predictor(rows, predictor) for predictor in predictors}


def best_mapping(scores: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    covered = [row for row in scores.values() if row["covered"]]
    if not covered:
        return None
    return max(covered, key=lambda row: (float(row["accuracy"] or 0.0), float(row["coverage"] or 0.0), row["covered"]))


def draw_side_frame(frame: np.ndarray, row: dict[str, Any], predictions: dict[str, str | None], delta_label: str) -> np.ndarray:
    out = frame.copy()
    x = row.get("visual_crop_ball_x") or row.get("pose_ball_x") or row.get("flow_ball_x")
    y = row.get("visual_crop_ball_y") or row.get("pose_ball_y") or row.get("flow_ball_y")
    if x is not None and y is not None:
        cv2.circle(out, (int(round(float(x))), int(round(float(y)))), 24, (0, 255, 255), 3)
        cv2.circle(out, (int(round(float(x))), int(round(float(y)))), 4, (0, 0, 255), -1)
    label = valid_side(row.get("contact_side"))
    title = f"label={label} screen={predictions.get('screen_ball')} pose={predictions.get('pose_anatomical')} {delta_label}"
    meta = (
        f"t={float(row.get('candidate_time_sec') or 0):.3f}s "
        f"x={screen_ball_x_norm(row)} flip_screen={predictions.get('screen_ball_flipped')} "
        f"flip_pose={predictions.get('pose_flipped')}"
    )
    cv2.rectangle(out, (0, 0), (out.shape[1], 128), (0, 0, 0), -1)
    cv2.putText(out, title, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, meta, (24, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (210, 230, 255), 2, cv2.LINE_AA)
    return out


def render_side_strip(row: dict[str, Any], video_path: Path, out_path: Path, width_px: int) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    center = int(round(float(row.get("candidate_time_sec") or 0.0) * fps))
    predictions = side_predictions(row)
    frames: list[np.ndarray] = []
    try:
        for offset in (-3, 0, 3):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, center + offset))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            drawn = draw_side_frame(frame, row, predictions, f"frame {offset:+d}")
            scale = width_px / max(1, drawn.shape[1])
            frames.append(cv2.resize(drawn, (width_px, int(round(drawn.shape[0] * scale))), interpolation=cv2.INTER_AREA))
    finally:
        cap.release()
    if not frames:
        return False
    max_h = max(frame.shape[0] for frame in frames)
    padded = []
    for frame in frames:
        if frame.shape[0] < max_h:
            frame = np.vstack([frame, np.zeros((max_h - frame.shape[0], frame.shape[1], 3), dtype=np.uint8)])
        padded.append(frame)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), np.hstack(padded)))


def disagreement_reason(row: dict[str, Any]) -> str:
    label = valid_side(row.get("contact_side"))
    predictions = side_predictions(row)
    if predictions["pose_anatomical"] and predictions["pose_anatomical"] != label:
        return "pose_anatomical_disagrees"
    if predictions["screen_ball"] and predictions["screen_ball"] != label:
        return "screen_side_disagrees"
    if predictions["pose_anatomical"] is None and predictions["screen_ball"] is None:
        return "no_side_proxy"
    return "mixed_or_low_confidence"


def build_rows(rows: list[dict[str, Any]], *, video_paths: dict[str, Path], out_dir: Path, max_strips: int, width_px: int) -> list[dict[str, Any]]:
    audit_rows: list[dict[str, Any]] = []
    strips = 0
    for row in rows:
        predictions = side_predictions(row)
        label = valid_side(row.get("contact_side"))
        reason = disagreement_reason(row)
        out = {
            "video_id": row.get("video_id"),
            "video_name": row.get("video_name"),
            "split": row.get("split"),
            "candidate_time_sec": row.get("candidate_time_sec"),
            "label_side": label,
            "trick_label": row.get("trick_label"),
            "screen_ball_x_norm": screen_ball_x_norm(row),
            "reason": reason,
            **{f"pred_{key}": value for key, value in predictions.items()},
        }
        if reason != "mixed_or_low_confidence" and strips < max_strips:
            video_path = video_paths.get(str(row.get("video_name")))
            if video_path is not None:
                strip_path = out_dir / "strips" / f"{str(row.get('video_id')).replace('/', '-')}_{float(row.get('candidate_time_sec') or 0):.3f}_{reason}.jpg"
                if render_side_strip(row, video_path, strip_path, width_px):
                    out["strip_path"] = str(strip_path)
                    strips += 1
        audit_rows.append(out)
    return audit_rows


def write_report(path: Path, summary: dict[str, Any], per_video: list[dict[str, Any]]) -> None:
    lines = [
        "# Side Semantics Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Side-labeled rows: `{summary['side_rows']}`",
        f"- Label counts: `{summary['label_counts']}`",
        f"- Best global mapping: `{summary['best_global_mapping']}`",
        f"- Strip dir: `{summary['strip_dir']}`",
        "",
        "## Global Mapping Scores",
        "",
        "| predictor | covered | coverage | accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, score in summary["global_scores"].items():
        accuracy = score["accuracy"]
        accuracy_text = "" if accuracy is None else f"{accuracy:.3f}"
        lines.append(f"| `{name}` | {score['covered']} | {score['coverage']:.3f} | {accuracy_text} |")
    lines.extend(
        [
            "",
            "## Per-Video Best Mapping",
            "",
            "| video | rows | labels | best predictor | coverage | accuracy | interpretation |",
            "| --- | ---: | --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in per_video:
        best = row.get("best_mapping") or {}
        accuracy = best.get("accuracy")
        accuracy_text = "" if accuracy is None else f"{accuracy:.3f}"
        lines.append(
            f"| `{row['video_id']}` | {row['rows']} | `{row['label_counts']}` | "
            f"`{best.get('predictor')}` | {float(best.get('coverage') or 0):.3f} | {accuracy_text} | {row['interpretation']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If `screen_ball` wins, labels likely mean visible screen side.",
            "- If `pose_anatomical` wins, RTMW anatomical side is aligned with labels.",
            "- If a flipped predictor wins, the corresponding convention is reversed.",
            "- If no predictor reaches high accuracy per-video, side labels need a clearer definition before more modeling.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def interpret_best(best: dict[str, Any] | None) -> str:
    if not best or best.get("accuracy") is None:
        return "no usable side proxy"
    accuracy = float(best["accuracy"])
    coverage = float(best.get("coverage") or 0.0)
    predictor = str(best.get("predictor"))
    if accuracy >= 0.85 and coverage >= 0.5:
        return f"consistent with {predictor}"
    if accuracy >= 0.75:
        return f"weakly consistent with {predictor}"
    return "mixed side convention or insufficient proxy"


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_side_rows(args.dataset_dir.resolve(), args.labels_dir.resolve(), args.label_match_tolerance_sec)
    out_dir = args.out_dir.resolve()
    video_paths = video_paths_by_name(args.review_manifest.resolve())
    global_scores = score_all(rows)
    best_global = best_mapping(global_scores)
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_video[str(row.get("video_id"))].append(row)
    per_video = []
    for video_id, video_rows in sorted(by_video.items()):
        scores = score_all(video_rows)
        best = best_mapping(scores)
        per_video.append(
            {
                "video_id": video_id,
                "rows": len(video_rows),
                "label_counts": dict(Counter(valid_side(row.get("contact_side")) for row in video_rows)),
                "scores": scores,
                "best_mapping": best,
                "interpretation": interpret_best(best),
            }
        )
    audit_rows = build_rows(rows, video_paths=video_paths, out_dir=out_dir, max_strips=args.max_strips, width_px=args.strip_panel_width_px)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "dataset_dir": str(args.dataset_dir),
        "labels_dir": str(args.labels_dir),
        "review_manifest": str(args.review_manifest),
        "out_dir": str(out_dir),
        "strip_dir": str(out_dir / "strips"),
        "side_rows": len(rows),
        "label_counts": dict(Counter(valid_side(row.get("contact_side")) for row in rows)),
        "global_scores": global_scores,
        "best_global_mapping": None if best_global is None else best_global,
        "per_video": per_video,
        "disagreement_counts": dict(Counter(row["reason"] for row in audit_rows)),
    }
    write_json(out_dir / "side_semantics_audit_summary.json", summary)
    write_jsonl(out_dir / "side_semantics_audit.jsonl", audit_rows)
    write_report(out_dir / "side_semantics_audit_report.md", summary, per_video)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit reviewed side-label semantics against screen and pose conventions")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--label-match-tolerance-sec", type=float, default=0.08)
    parser.add_argument("--max-strips", type=int, default=48)
    parser.add_argument("--strip-panel-width-px", type=int, default=420)
    return parser.parse_args()


def main() -> None:
    summary = run_audit(parse_args())
    print(f"status: {summary['status']}")
    print(f"report: {Path(summary['out_dir']) / 'side_semantics_audit_report.md'}")
    print(
        json.dumps(
            {
                "side_rows": summary["side_rows"],
                "label_counts": summary["label_counts"],
                "best_global_mapping": summary["best_global_mapping"],
                "disagreement_counts": summary["disagreement_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
