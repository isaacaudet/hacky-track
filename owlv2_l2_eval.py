#!/usr/bin/env python3
"""Evaluate OWLv2 raw detections plus a conservative L2 fill/clean pass.

The safety rule is the point of this script: OWLv2 silence is treated as a
real absence signal. L2 may fill only short gaps bounded by confident anchors,
and by default each filled frame must have low-confidence detector support near
the interpolated trajectory. This prevents the fill step from reintroducing the
old sustained lock-on failure mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

import evaluate_dense_trajectory


ROOT = Path(__file__).resolve().parent
DEFAULT_LABELS = ROOT / "runs/release-27-public/dense_trajectory_review_v2_with_sources/dense_trajectory_labels.reviewed.jsonl"
DEFAULT_DETECTIONS = ROOT / "runs/release-27-public/oracle_owlv2_full_v1/detections.jsonl"
DEFAULT_OUT = ROOT / "runs/release-27-public/owlv2_l2_eval_v1"
VISIBLE_STATES = {"visible", "partially_occluded"}
NO_TARGET_STATES = {"fully_occluded", "out_of_frame"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def portable(path: Path | None, base: Path = ROOT) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except (OSError, ValueError):
        return str(path)


def distance(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def top_detection(record: dict[str, Any], threshold: float) -> dict[str, float] | None:
    above = [det for det in record.get("detections", []) if float(det["score"]) >= threshold]
    if not above:
        return None
    return max(above, key=lambda det: float(det["score"]))


def best_near_detection(record: dict[str, Any], x: float, y: float, *, min_score: float, max_dist_px: float) -> dict[str, float] | None:
    candidates = [
        det
        for det in record.get("detections", [])
        if float(det["score"]) >= min_score and distance(float(det["x"]), float(det["y"]), x, y) <= max_dist_px
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda det: distance(float(det["x"]), float(det["y"]), x, y) - 10.0 * float(det["score"]))


def prediction_from_detection(record: dict[str, Any], det: dict[str, float], *, source: str) -> dict[str, Any]:
    return {
        "clip_id": record.get("clip_id"),
        "source_video": record.get("source_video"),
        "frame_index": int(record["frame_index"]),
        "time_sec": record.get("time_sec"),
        "x": round(float(det["x"]), 3),
        "y": round(float(det["y"]), 3),
        "confidence": round(float(det["score"]), 6),
        "visibility": "predicted",
        "source": source,
    }


def raw_predictions(records: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        det = top_detection(record, threshold)
        if det is not None:
            rows.append(prediction_from_detection(record, det, source="owlv2_raw"))
    return rows


def robust_clean_anchors(anchors: list[dict[str, Any]], *, win: int = 6, k: float = 3.5, floor_px: float = 18.0, iters: int = 2) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(anchors) < max(12, 2 * win + 1):
        return anchors, []
    t = np.asarray([float(anchor["time_sec"]) for anchor in anchors], dtype=float)
    x = np.asarray([float(anchor["x"]) for anchor in anchors], dtype=float)
    y = np.asarray([float(anchor["y"]) for anchor in anchors], dtype=float)
    keep = np.ones(len(anchors), dtype=bool)
    for _ in range(iters):
        idx = np.where(keep)[0]
        drop: list[int] = []
        for pos, i in enumerate(idx):
            nb = idx[max(0, pos - win) : pos + win + 1]
            nb = nb[nb != i]
            if len(nb) < 6:
                continue
            degree = 2 if len(nb) >= 8 else 1
            cy = np.polyfit(t[nb], y[nb], degree)
            cx = np.polyfit(t[nb], x[nb], min(degree, 1))
            res = np.hypot(y[nb] - np.polyval(cy, t[nb]), x[nb] - np.polyval(cx, t[nb]))
            scale = 1.4826 * np.median(res)
            residual = float(np.hypot(y[i] - np.polyval(cy, t[i]), x[i] - np.polyval(cx, t[i])))
            if residual > k * scale + floor_px:
                drop.append(int(i))
        if not drop:
            break
        keep[drop] = False
    kept = [anchor for anchor, ok in zip(anchors, keep) if ok]
    dropped = [anchor for anchor, ok in zip(anchors, keep) if not ok]
    return kept, dropped


def interpolate_position(left: dict[str, Any], right: dict[str, Any], frame_index: int) -> tuple[float, float]:
    span = int(right["frame_index"]) - int(left["frame_index"])
    if span <= 0:
        return float(left["x"]), float(left["y"])
    alpha = (frame_index - int(left["frame_index"])) / span
    return (
        float(left["x"]) + alpha * (float(right["x"]) - float(left["x"])),
        float(left["y"]) + alpha * (float(right["y"]) - float(left["y"])),
    )


def l2_fill_and_clean(
    records: list[dict[str, Any]],
    *,
    threshold: float,
    candidate_floor: float,
    support_radius_px: float,
    max_gap_frames: int,
    max_gap_sec: float,
    clean: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_clip[str(record["clip_id"])].append(record)

    predictions: list[dict[str, Any]] = []
    diagnostics = {
        "clips": {},
        "raw_anchor_count": 0,
        "kept_anchor_count": 0,
        "dropped_anchor_count": 0,
        "filled_count": 0,
        "fill_rejected_count": 0,
    }
    for clip_id, clip_records in sorted(by_clip.items()):
        ordered = sorted(clip_records, key=lambda item: int(item["frame_index"]))
        record_by_frame = {int(record["frame_index"]): record for record in ordered}
        anchors: list[dict[str, Any]] = []
        for record in ordered:
            det = top_detection(record, threshold)
            if det is None:
                continue
            anchors.append(
                {
                    "clip_id": clip_id,
                    "source_video": record.get("source_video"),
                    "frame_index": int(record["frame_index"]),
                    "time_sec": float(record.get("time_sec") or 0.0),
                    "x": float(det["x"]),
                    "y": float(det["y"]),
                    "score": float(det["score"]),
                    "record": record,
                }
            )
        kept, dropped = robust_clean_anchors(anchors) if clean else (anchors, [])
        kept_by_frame = {int(anchor["frame_index"]): anchor for anchor in kept}
        for anchor in kept:
            predictions.append(
                {
                    "clip_id": anchor["clip_id"],
                    "source_video": anchor["source_video"],
                    "frame_index": anchor["frame_index"],
                    "time_sec": anchor["time_sec"],
                    "x": round(anchor["x"], 3),
                    "y": round(anchor["y"], 3),
                    "confidence": round(anchor["score"], 6),
                    "visibility": "predicted",
                    "source": "owlv2_l2_anchor",
                }
            )

        filled = 0
        rejected = 0
        for left, right in zip(kept, kept[1:]):
            left_frame = int(left["frame_index"])
            right_frame = int(right["frame_index"])
            missing_frames = [frame for frame in range(left_frame + 1, right_frame) if frame in record_by_frame and frame not in kept_by_frame]
            if not missing_frames:
                continue
            gap_frames = right_frame - left_frame - 1
            gap_sec = float(right["time_sec"]) - float(left["time_sec"])
            if gap_frames > max_gap_frames or gap_sec > max_gap_sec:
                rejected += len(missing_frames)
                continue
            for frame_index in missing_frames:
                record = record_by_frame[frame_index]
                interp_x, interp_y = interpolate_position(left, right, frame_index)
                support = best_near_detection(
                    record,
                    interp_x,
                    interp_y,
                    min_score=candidate_floor,
                    max_dist_px=support_radius_px,
                )
                if support is None:
                    rejected += 1
                    continue
                confidence = min(float(left["score"]), float(right["score"]), max(threshold, float(support["score"])))
                # Blend toward the low-confidence candidate when it exists near
                # the trajectory; this keeps L2 anchored in detector evidence.
                x = 0.65 * float(support["x"]) + 0.35 * interp_x
                y = 0.65 * float(support["y"]) + 0.35 * interp_y
                predictions.append(
                    {
                        "clip_id": record.get("clip_id"),
                        "source_video": record.get("source_video"),
                        "frame_index": int(record["frame_index"]),
                        "time_sec": record.get("time_sec"),
                        "x": round(x, 3),
                        "y": round(y, 3),
                        "confidence": round(confidence, 6),
                        "visibility": "predicted",
                        "source": "owlv2_l2_fill",
                    }
                )
                filled += 1
        diagnostics["clips"][clip_id] = {
            "records": len(ordered),
            "raw_anchors": len(anchors),
            "kept_anchors": len(kept),
            "dropped_anchors": len(dropped),
            "filled": filled,
            "fill_rejected": rejected,
        }
        diagnostics["raw_anchor_count"] += len(anchors)
        diagnostics["kept_anchor_count"] += len(kept)
        diagnostics["dropped_anchor_count"] += len(dropped)
        diagnostics["filled_count"] += filled
        diagnostics["fill_rejected_count"] += rejected
    predictions.sort(key=lambda row: (str(row.get("clip_id")), int(row["frame_index"]), str(row.get("source"))))
    return predictions, diagnostics


def build_prediction_map(predictions: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for prediction in predictions:
        key = (str(prediction.get("clip_id")), int(prediction["frame_index"]))
        current = out.get(key)
        if current is None or float(prediction.get("confidence") or 0.0) > float(current.get("confidence") or 0.0):
            out[key] = prediction
    return out


def longest_run(flags: list[bool]) -> int:
    best = cur = 0
    for flag in flags:
        if flag:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def lock_on_metrics(labels: list[dict[str, Any]], predictions: list[dict[str, Any]], *, tolerance_px: float) -> dict[str, Any]:
    prediction_map = build_prediction_map(predictions)
    by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for label in labels:
        if label.get("quality") != "reviewed":
            continue
        if label.get("visibility") not in (VISIBLE_STATES | NO_TARGET_STATES):
            continue
        by_clip[str(label["clip_id"])].append(label)

    no_target_fp = 0
    filled_no_target_fp = 0
    bad_prediction = 0
    filled_predictions = 0
    clip_rows: list[dict[str, Any]] = []
    longest_no_target = 0
    longest_bad = 0
    longest_filled_no_target = 0
    for clip_id, clip_labels in sorted(by_clip.items()):
        ordered = sorted(clip_labels, key=lambda item: int(item["frame_index"]))
        no_target_flags: list[bool] = []
        bad_flags: list[bool] = []
        filled_no_target_flags: list[bool] = []
        for label in ordered:
            pred = prediction_map.get((clip_id, int(label["frame_index"])))
            is_filled = bool(pred and str(pred.get("source")) == "owlv2_l2_fill")
            if is_filled:
                filled_predictions += 1
            no_target_bad = False
            bad = False
            if label.get("visibility") in NO_TARGET_STATES:
                no_target_bad = pred is not None
                bad = no_target_bad
                if no_target_bad:
                    no_target_fp += 1
                    if is_filled:
                        filled_no_target_fp += 1
            elif label.get("visibility") in VISIBLE_STATES and pred is not None and label.get("x") is not None and label.get("y") is not None:
                err = distance(float(pred["x"]), float(pred["y"]), float(label["x"]), float(label["y"]))
                bad = err > tolerance_px
            no_target_flags.append(no_target_bad)
            bad_flags.append(bad)
            filled_no_target_flags.append(no_target_bad and is_filled)
            if bad:
                bad_prediction += 1
        clip_no_target = longest_run(no_target_flags)
        clip_bad = longest_run(bad_flags)
        clip_filled_no_target = longest_run(filled_no_target_flags)
        longest_no_target = max(longest_no_target, clip_no_target)
        longest_bad = max(longest_bad, clip_bad)
        longest_filled_no_target = max(longest_filled_no_target, clip_filled_no_target)
        clip_rows.append(
            {
                "clip_id": clip_id,
                "longest_no_target_fp_run": clip_no_target,
                "longest_bad_prediction_run": clip_bad,
                "longest_filled_no_target_fp_run": clip_filled_no_target,
            }
        )
    return {
        "no_target_fp_frames": no_target_fp,
        "filled_no_target_fp_frames": filled_no_target_fp,
        "filled_predictions": filled_predictions,
        "bad_prediction_frames": bad_prediction,
        "longest_no_target_fp_run": longest_no_target,
        "longest_bad_prediction_run": longest_bad,
        "longest_filled_no_target_fp_run": longest_filled_no_target,
        "by_clip": clip_rows,
    }


def evaluate_predictions(
    *,
    labels_jsonl: Path,
    predictions_jsonl: Path,
    out_dir: Path,
    tolerance_px: float,
) -> dict[str, Any]:
    return evaluate_dense_trajectory.evaluate_dense_trajectory(
        labels_jsonl=labels_jsonl,
        predictions_jsonl=predictions_jsonl,
        out_dir=out_dir,
        tolerance_px=tolerance_px,
        radius_multiplier=0.0,
        confidence_threshold=0.01,
        max_time_delta_sec=0.0,
    )


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# OWLv2 + L2 Fill Evaluation",
        "",
        f"- OWLv2 threshold: `{summary['parameters']['threshold']}`",
        f"- Candidate floor: `{summary['parameters']['candidate_floor']}`",
        f"- Max gap frames: `{summary['parameters']['max_gap_frames']}`",
        f"- Support radius px: `{summary['parameters']['support_radius_px']}`",
        "",
        "| track | tolerance | visible pass | missing | mean err | p95 err | no-target FP | longest no-target run | filled no-target FP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ["raw", "l2"]:
        for tol, item in summary["evaluations"][key].items():
            metrics = item["dense_metrics"]["overall"]
            lock = item["lock_on"]
            lines.append(
                f"| {key} | {tol} | {metrics['visible_pass_rate']} | {metrics['visible_missing_rate']} | "
                f"{metrics['mean_center_error_px']} | {metrics['p95_center_error_px']} | "
                f"{metrics['no_target_false_positive_rate']} | {lock['longest_no_target_fp_run']} | "
                f"{lock['filled_no_target_fp_frames']} |"
            )
    lines.extend(
        [
            "",
            "## L2 Diagnostics",
            "",
            f"- Raw anchors: `{summary['l2_diagnostics']['raw_anchor_count']}`",
            f"- Kept anchors: `{summary['l2_diagnostics']['kept_anchor_count']}`",
            f"- Dropped anchors: `{summary['l2_diagnostics']['dropped_anchor_count']}`",
            f"- Filled frames: `{summary['l2_diagnostics']['filled_count']}`",
            f"- Fill rejected frames: `{summary['l2_diagnostics']['fill_rejected_count']}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_owlv2_l2_eval(
    *,
    labels_jsonl: Path,
    detections_jsonl: Path,
    out_dir: Path,
    threshold: float,
    candidate_floor: float,
    support_radius_px: float,
    max_gap_frames: int,
    max_gap_sec: float,
    tolerances: tuple[float, ...],
    clean: bool,
) -> dict[str, Any]:
    labels = read_jsonl(labels_jsonl)
    detections = read_jsonl(detections_jsonl)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = raw_predictions(detections, threshold)
    l2, l2_diagnostics = l2_fill_and_clean(
        detections,
        threshold=threshold,
        candidate_floor=candidate_floor,
        support_radius_px=support_radius_px,
        max_gap_frames=max_gap_frames,
        max_gap_sec=max_gap_sec,
        clean=clean,
    )
    raw_path = out_dir / "owlv2_raw_predictions.jsonl"
    l2_path = out_dir / "owlv2_l2_predictions.jsonl"
    write_jsonl(raw_path, raw)
    write_jsonl(l2_path, l2)

    evaluations: dict[str, dict[str, Any]] = {"raw": {}, "l2": {}}
    for name, predictions, predictions_path in [("raw", raw, raw_path), ("l2", l2, l2_path)]:
        for tolerance in tolerances:
            eval_dir = out_dir / f"eval_{name}_tol{int(tolerance)}"
            dense_metrics = evaluate_predictions(
                labels_jsonl=labels_jsonl,
                predictions_jsonl=predictions_path,
                out_dir=eval_dir,
                tolerance_px=tolerance,
            )
            lock_metrics = lock_on_metrics(labels, predictions, tolerance_px=tolerance)
            evaluations[name][str(tolerance)] = {
                "dense_metrics": {
                    "overall": dense_metrics["overall"],
                    "by_split": dense_metrics["by_split"],
                    "outputs": dense_metrics["outputs"],
                },
                "lock_on": lock_metrics,
            }

    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "labels_jsonl": portable(labels_jsonl),
        "detections_jsonl": portable(detections_jsonl),
        "out_dir": portable(out_dir),
        "parameters": {
            "threshold": threshold,
            "candidate_floor": candidate_floor,
            "support_radius_px": support_radius_px,
            "max_gap_frames": max_gap_frames,
            "max_gap_sec": max_gap_sec,
            "tolerances": list(tolerances),
            "clean": clean,
        },
        "prediction_counts": {
            "raw": len(raw),
            "l2": len(l2),
        },
        "l2_diagnostics": l2_diagnostics,
        "evaluations": evaluations,
        "outputs": {
            "raw_predictions": portable(raw_path),
            "l2_predictions": portable(l2_path),
            "summary_json": portable(out_dir / "summary.json"),
            "report_md": portable(out_dir / "report.md"),
        },
    }
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "report.md", summary)
    return summary


def parse_tolerances(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one tolerance is required")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate raw OWLv2 and conservative L2-filled tracks")
    parser.add_argument("--labels-jsonl", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--detections-jsonl", type=Path, default=DEFAULT_DETECTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--candidate-floor", type=float, default=0.1)
    parser.add_argument("--support-radius-px", type=float, default=25.0)
    parser.add_argument("--max-gap-frames", type=int, default=3)
    parser.add_argument("--max-gap-sec", type=float, default=0.18)
    parser.add_argument("--tolerances", type=parse_tolerances, default=(12.0, 15.0))
    parser.add_argument("--no-clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_owlv2_l2_eval(
        labels_jsonl=args.labels_jsonl,
        detections_jsonl=args.detections_jsonl,
        out_dir=args.out_dir,
        threshold=args.threshold,
        candidate_floor=args.candidate_floor,
        support_radius_px=args.support_radius_px,
        max_gap_frames=args.max_gap_frames,
        max_gap_sec=args.max_gap_sec,
        tolerances=args.tolerances,
        clean=not args.no_clean,
    )
    print(f"summary: {args.out_dir / 'summary.json'}")
    print((args.out_dir / "report.md").read_text(encoding="utf-8"))
    raw_best = summary["evaluations"]["raw"][str(summary["parameters"]["tolerances"][-1])]["dense_metrics"]["overall"]
    l2_best = summary["evaluations"]["l2"][str(summary["parameters"]["tolerances"][-1])]["dense_metrics"]["overall"]
    print(json.dumps({"raw": raw_best, "l2": l2_best}, indent=2))


if __name__ == "__main__":
    main()
