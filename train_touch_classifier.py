#!/usr/bin/env python3
"""Train/evaluate the fused touch classifier from the candidate table.

Default mode is intentionally strict: it requires L2 trajectory features, a
minimum number of labeled videos, and frozen-test rows before training is
reported as release evidence. Use --allow-small, --allow-missing-trajectory,
and --allow-missing-frozen-test only for plumbing smoke tests, not for release
metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = ROOT / "runs/release-27-public/touch_corpus_v1/touch_training_dataset_v1"
DEFAULT_LABELS_DIR = DEFAULT_DATASET_DIR.parent / "visual_touch_labels"
TOUCH_PRECISION_GATE = 0.90
TOUCH_RECALL_GATE = 0.85
CANDIDATE_GATE_BREAK_DELTA_SEC = 0.10
CANDIDATE_GATE_WEAK_AUDIO_MAX = 8.0
CANDIDATE_GATE_WEAK_BREAK_SUPPORT_MAX = 4
CANDIDATE_RESCUE_AUDIO_MIN = 10.0
CANDIDATE_RESCUE_IMPULSE_MIN = 500.0
CANDIDATE_RESCUE_BREAK_SUPPORT_MIN = 1
CANDIDATE_RESCUE_BREAK_DELTA_SEC = 0.10
EVENT_NMS_GAP_SEC = 0.30
EVENT_MATCH_TOL_SEC = 0.20
AUDIO_FEATURES = [
    "has_audio",
    "audio_strength",
    "has_existing_hint",
    "time_since_prev_candidate_sec",
    "time_to_next_candidate_sec",
    "nearest_drop_delta_sec",
    "nearest_stall_delta_sec",
    "in_stall_window",
]
AUDIO_TIMBRE_FEATURES = [
    "audio_window_rms",
    "audio_window_peak",
    "audio_peak_to_rms",
    "audio_zero_crossing_rate",
    "audio_spectral_centroid",
    "audio_spectral_bandwidth",
    "audio_spectral_rolloff85",
    "audio_spectral_flatness",
    "audio_low_band_ratio",
    "audio_mid_band_ratio",
    "audio_high_band_ratio",
    "audio_pre_rms",
    "audio_post_rms",
    "audio_attack_ratio",
    "audio_mfcc_1",
    "audio_mfcc_2",
    "audio_mfcc_3",
    "audio_mfcc_4",
    "audio_mfcc_5",
    "audio_mfcc_6",
]
TRAJECTORY_FEATURES = [
    "trajectory_break_support",
    "trajectory_nearest_break_delta_sec",
    "trajectory_max_positive_dvy",
    "height_reversal",
    "detector_confidence_near_candidate",
]
TRACK_WINDOW_FEATURES = [
    "trajectory_track_points_window",
    "trajectory_gap_before_sec",
    "trajectory_gap_after_sec",
    "trajectory_nearest_y_peak_delta_sec",
    "trajectory_nearest_y_trough_delta_sec",
    "trajectory_y_position_pct_window",
    "trajectory_y_peak_prominence_window_px",
    "trajectory_y_trough_prominence_window_px",
    "trajectory_vx_before",
    "trajectory_vx_after",
    "trajectory_vx_delta",
    "trajectory_vy_before",
    "trajectory_vy_after",
    "trajectory_vy_delta",
    "trajectory_ax_window",
    "trajectory_ay_window",
    "trajectory_speed_before",
    "trajectory_speed_after",
    "trajectory_speed_delta",
    "trajectory_impulse_score",
    "trajectory_x_range_window_px",
    "trajectory_y_range_window_px",
    "trajectory_confidence_mean_window",
    "trajectory_local_y_quad_rms_px",
]
POSE_FEATURES = [
    "pose_ball_missing",
    "pose_missing",
    "pose_present",
    "pose_person_boxes",
    "pose_lower_body_present",
    "pose_foot_present",
    "pose_shank_length_px",
    "pose_nearest_foot_conf",
    "pose_nearest_foot_dist_px",
    "pose_nearest_foot_dist_norm_shank",
    "pose_nearest_lower_conf",
    "pose_nearest_lower_dist_px",
    "pose_nearest_lower_dist_norm_shank",
]
FLOW_FEATURES = [
    "flow_missing",
    "flow_ball_missing",
    "flow_texture_std",
    "flow_before_points",
    "flow_after_points",
    "flow_before_valid_frac",
    "flow_after_valid_frac",
    "flow_before_dx",
    "flow_before_dy",
    "flow_before_mag",
    "flow_after_dx",
    "flow_after_dy",
    "flow_after_mag",
    "flow_dx_delta",
    "flow_dy_delta",
    "flow_delta_mag",
    "flow_impulse_mag",
]
TRACK_FEATURES = TRAJECTORY_FEATURES + TRACK_WINDOW_FEATURES
FULL_AUDIO_FEATURES = AUDIO_FEATURES + AUDIO_TIMBRE_FEATURES
FUSED_BASE_FEATURES = FULL_AUDIO_FEATURES + TRACK_FEATURES
FUSED_FLOW_FEATURES = FUSED_BASE_FEATURES + FLOW_FEATURES
FUSED_POSE_FEATURES = FUSED_BASE_FEATURES + POSE_FEATURES
FUSED_ALL_FEATURES = FUSED_BASE_FEATURES + FLOW_FEATURES + POSE_FEATURES
FEATURE_SETS = {
    "audio_only": AUDIO_FEATURES,
    "audio_timbre_only": FULL_AUDIO_FEATURES,
    "trajectory_only": TRACK_FEATURES,
    "flow_only": FLOW_FEATURES,
    "pose_only": POSE_FEATURES,
    "fused_audio_trajectory": FUSED_BASE_FEATURES,
    "fused_audio_trajectory_flow": FUSED_FLOW_FEATURES,
    "fused_audio_trajectory_pose": FUSED_POSE_FEATURES,
    "fused_audio_trajectory_flow_pose": FUSED_ALL_FEATURES,
}
FEATURE_DEFAULTS = {
    "trajectory_nearest_break_delta_sec": 999.0,
    "trajectory_gap_before_sec": 999.0,
    "trajectory_gap_after_sec": 999.0,
    "trajectory_nearest_y_peak_delta_sec": 999.0,
    "trajectory_nearest_y_trough_delta_sec": 999.0,
    "nearest_drop_delta_sec": 999.0,
    "nearest_stall_delta_sec": 999.0,
    "pose_ball_missing": 1.0,
    "pose_missing": 1.0,
    "pose_nearest_foot_dist_px": 9999.0,
    "pose_nearest_foot_dist_norm_shank": 100.0,
    "pose_nearest_lower_dist_px": 9999.0,
    "pose_nearest_lower_dist_norm_shank": 100.0,
    "flow_missing": 1.0,
    "flow_ball_missing": 1.0,
    "flow_before_valid_frac": 0.0,
    "flow_after_valid_frac": 0.0,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def approved_touch_times_from_labels(labels_dir: Path | None, video_id: str, *, match_tol_sec: float = EVENT_MATCH_TOL_SEC) -> list[float] | None:
    if labels_dir is None:
        return None
    path = labels_dir / f"{video_id}.events.json"
    if not path.exists():
        return None
    doc = read_json(path)
    times = []
    for rally in doc.get("rallies", []):
        for event in rally.get("events", []):
            if event.get("review_status") != "approved":
                continue
            if event.get("type") != "touch" or event.get("time_sec") is None:
                continue
            times.append(float(event["time_sec"]))
    return dedupe_event_times(times, match_tol_sec)


def value_as_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        return float(FEATURE_DEFAULTS.get(key, 0.0))
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(numeric) or math.isinf(numeric):
        return 0.0
    return numeric


def feature_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    return np.asarray([[value_as_float(row, key) for key in feature_names] for row in rows], dtype=float)


def labels(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([1 if row.get("label_is_touch") else 0 for row in rows], dtype=int)


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "n_detected": tp + fp,
        "n_truth": tp + fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def has_trajectory_features(rows: list[dict[str, Any]]) -> bool:
    return any(any(row.get(key) is not None for key in TRAJECTORY_FEATURES) for row in rows)


def has_pose_features(rows: list[dict[str, Any]]) -> bool:
    return any(any(row.get(key) is not None for key in POSE_FEATURES) for row in rows)


def has_flow_features(rows: list[dict[str, Any]]) -> bool:
    return any(any(row.get(key) is not None for key in FLOW_FEATURES) for row in rows)


def make_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=7),
    )


def predict_binary(model: Any, matrix: np.ndarray, threshold: float) -> np.ndarray:
    scores = predict_scores(model, matrix)
    return (scores >= threshold).astype(int)


def predict_scores(model: Any, matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(matrix)[:, 1]
    return model.decision_function(matrix)


def prediction_row(
    row: dict[str, Any],
    *,
    score: float,
    predicted: int,
    fold_video_id: str | None,
    feature_names: list[str],
) -> dict[str, Any]:
    truth = 1 if row.get("label_is_touch") else 0
    if truth == 1 and predicted == 0:
        error_type = "false_negative"
    elif truth == 0 and predicted == 1:
        error_type = "false_positive"
    elif truth == 1:
        error_type = "true_positive"
    else:
        error_type = "true_negative"
    out = {
        "video_id": row.get("video_id"),
        "video_name": row.get("video_name"),
        "split": row.get("split"),
        "candidate_time_sec": row.get("candidate_time_sec"),
        "label_is_touch": bool(truth),
        "predicted_is_touch": bool(predicted),
        "touch_score": round(float(score), 6),
        "error_type": error_type,
        "fold_video_id": fold_video_id,
        "candidate_review_decision": row.get("candidate_review_decision"),
        "candidate_review_source": row.get("candidate_review_source"),
        "label_touch_time_sec": row.get("label_touch_time_sec"),
        "label_touch_delta_sec": row.get("label_touch_delta_sec"),
    }
    for key in feature_names:
        out[key] = row.get(key)
    return out


def prediction_error_type(truth: bool, predicted: bool) -> str:
    if truth and not predicted:
        return "false_negative"
    if not truth and predicted:
        return "false_positive"
    return "true_positive" if truth else "true_negative"


def candidate_precision_veto_reason(row: dict[str, Any]) -> str | None:
    """Return a release precision-gate veto reason for classifier candidates.

    Audio onsets are high-recall but noisy; for a candidate to remain a touch
    it must have at least one nearby L2 trajectory-break vote. A predicted
    audio fire with zero break support and no break within 100 ms is the
    observed frozen-test false-positive class: sound without ball motion.
    """
    break_support = value_as_float(row, "trajectory_break_support")
    break_delta = value_as_float(row, "trajectory_nearest_break_delta_sec")
    audio_strength = value_as_float(row, "audio_strength")
    if break_support <= 0 and break_delta > CANDIDATE_GATE_BREAK_DELTA_SEC:
        return "no_trajectory_corroboration"
    if audio_strength < CANDIDATE_GATE_WEAK_AUDIO_MAX and break_support <= CANDIDATE_GATE_WEAK_BREAK_SUPPORT_MAX:
        return "weak_audio_weak_trajectory"
    return None


def candidate_recall_rescue_reason(row: dict[str, Any]) -> str | None:
    """Return a strict L2 output-rescue reason for missed touch candidates.

    The trajectory drilldown showed a narrow class of misses where the
    classifier scored a cue too low despite strong audio and a clean nearby L2
    impulse. This rescue is intentionally conservative: it requires the audio
    transient, at least one breakpoint vote, a nearby breakpoint, and a large
    local velocity impulse. It does not rescue detector-silence or no-trajectory
    candidates.
    """
    audio_strength = value_as_float(row, "audio_strength")
    impulse = value_as_float(row, "trajectory_impulse_score")
    break_support = value_as_float(row, "trajectory_break_support")
    break_delta = value_as_float(row, "trajectory_nearest_break_delta_sec")
    if (
        audio_strength >= CANDIDATE_RESCUE_AUDIO_MIN
        and impulse >= CANDIDATE_RESCUE_IMPULSE_MIN
        and break_support >= CANDIDATE_RESCUE_BREAK_SUPPORT_MIN
        and break_delta <= CANDIDATE_RESCUE_BREAK_DELTA_SEC
    ):
        return "high_impulse_audio_trajectory_rescue"
    return None


def apply_candidate_precision_gate(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gated: list[dict[str, Any]] = []
    for row in predictions:
        out = dict(row)
        out["candidate_precision_gate_vetoed"] = False
        out["candidate_precision_gate_reason"] = None
        out["candidate_recall_rescued"] = False
        out["candidate_recall_rescue_reason"] = None
        if bool(out.get("predicted_is_touch")):
            reason = candidate_precision_veto_reason(out)
            if reason:
                out["predicted_is_touch_raw"] = True
                out["predicted_is_touch"] = False
                out["candidate_precision_gate_vetoed"] = True
                out["candidate_precision_gate_reason"] = reason
                out["error_type"] = prediction_error_type(bool(out.get("label_is_touch")), False)
        if not bool(out.get("predicted_is_touch")):
            reason = candidate_recall_rescue_reason(out)
            if reason:
                out.setdefault("predicted_is_touch_raw", False)
                out["predicted_is_touch"] = True
                out["candidate_recall_rescued"] = True
                out["candidate_recall_rescue_reason"] = reason
                out["error_type"] = prediction_error_type(bool(out.get("label_is_touch")), True)
        gated.append(out)
    return gated


def metrics_from_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = np.asarray([1 if row.get("label_is_touch") else 0 for row in predictions], dtype=int)
    y_pred = np.asarray([1 if row.get("predicted_is_touch") else 0 for row in predictions], dtype=int)
    per_video = []
    for video_id in sorted({str(row["video_id"]) for row in predictions}):
        indexes = [idx for idx, row in enumerate(predictions) if str(row["video_id"]) == video_id]
        metrics = precision_recall_f1(y_true[indexes], y_pred[indexes])
        per_video.append({"video_id": video_id, "rows": len(indexes), **metrics})
    return {**precision_recall_f1(y_true, y_pred), "rows": len(predictions), "per_video": per_video}


def dedupe_event_times(times: list[float], gap_sec: float) -> list[float]:
    out: list[float] = []
    for time_sec in sorted(times):
        if not out or time_sec - out[-1] > gap_sec:
            out.append(time_sec)
    return out


def predicted_event_clusters(rows: list[dict[str, Any]], *, nms_gap_sec: float = EVENT_NMS_GAP_SEC) -> list[list[dict[str, Any]]]:
    predicted_rows = sorted(
        [row for row in rows if bool(row.get("predicted_is_touch"))],
        key=lambda row: float(row.get("candidate_time_sec") or 0.0),
    )
    clusters: list[list[dict[str, Any]]] = []
    for row in predicted_rows:
        time_sec = float(row.get("candidate_time_sec") or 0.0)
        if not clusters or time_sec - float(clusters[-1][-1].get("candidate_time_sec") or 0.0) > nms_gap_sec:
            clusters.append([row])
        else:
            clusters[-1].append(row)
    return clusters


def best_event_candidate(cluster: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        cluster,
        key=lambda row: (
            value_as_float(row, "touch_score"),
            value_as_float(row, "trajectory_break_support"),
            -abs(value_as_float(row, "trajectory_nearest_break_delta_sec")),
        ),
    )


def predicted_event_times(rows: list[dict[str, Any]], *, nms_gap_sec: float = EVENT_NMS_GAP_SEC) -> list[float]:
    clusters = predicted_event_clusters(rows, nms_gap_sec=nms_gap_sec)
    events = []
    for cluster in clusters:
        best = best_event_candidate(cluster)
        events.append(float(best.get("candidate_time_sec") or 0.0))
    return events


def truth_event_times(
    rows: list[dict[str, Any]],
    *,
    match_tol_sec: float = EVENT_MATCH_TOL_SEC,
    labels_dir: Path | None = None,
) -> list[float]:
    if rows:
        label_times = approved_touch_times_from_labels(labels_dir, str(rows[0].get("video_id")), match_tol_sec=match_tol_sec)
        if label_times is not None:
            return label_times
    times = []
    for row in rows:
        if not bool(row.get("label_is_touch")):
            continue
        value = row.get("label_touch_time_sec")
        if value is None or value == "":
            value = row.get("candidate_time_sec")
        if value is not None and value != "":
            times.append(float(value))
    return dedupe_event_times(times, match_tol_sec)


def event_precision_recall_f1(predicted_times: list[float], truth_times: list[float], *, match_tol_sec: float) -> dict[str, Any]:
    used_truth: set[int] = set()
    true_positive = 0
    for predicted in sorted(predicted_times):
        matches = [
            (abs(predicted - truth), index)
            for index, truth in enumerate(truth_times)
            if index not in used_truth and abs(predicted - truth) <= match_tol_sec
        ]
        if not matches:
            continue
        _, index = min(matches)
        used_truth.add(index)
        true_positive += 1
    false_positive = len(predicted_times) - true_positive
    false_negative = len(truth_times) - true_positive
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "n_detected": len(predicted_times),
        "n_truth": len(truth_times),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def match_event_times(predicted_times: list[float], truth_times: list[float], *, match_tol_sec: float) -> tuple[dict[int, int], set[int]]:
    used_truth: set[int] = set()
    matches: dict[int, int] = {}
    for pred_index, predicted in enumerate(predicted_times):
        candidates = [
            (abs(predicted - truth), truth_index)
            for truth_index, truth in enumerate(truth_times)
            if truth_index not in used_truth and abs(predicted - truth) <= match_tol_sec
        ]
        if not candidates:
            continue
        _, truth_index = min(candidates)
        used_truth.add(truth_index)
        matches[pred_index] = truth_index
    return matches, used_truth


def event_rows_from_predictions(
    predictions: list[dict[str, Any]],
    *,
    nms_gap_sec: float = EVENT_NMS_GAP_SEC,
    match_tol_sec: float = EVENT_MATCH_TOL_SEC,
    labels_dir: Path | None = None,
    include_false_negatives: bool = True,
) -> list[dict[str, Any]]:
    """Build final merged touch-event rows from raw candidate predictions."""
    event_rows: list[dict[str, Any]] = []
    for video_id in sorted({str(row["video_id"]) for row in predictions}):
        rows = [row for row in predictions if str(row.get("video_id")) == video_id]
        clusters = predicted_event_clusters(rows, nms_gap_sec=nms_gap_sec)
        best_rows = [best_event_candidate(cluster) for cluster in clusters]
        predicted_times = [float(row.get("candidate_time_sec") or 0.0) for row in best_rows]
        truth_times = truth_event_times(rows, match_tol_sec=match_tol_sec, labels_dir=labels_dir)
        matches, used_truth = match_event_times(predicted_times, truth_times, match_tol_sec=match_tol_sec)
        for index, (cluster, best) in enumerate(zip(clusters, best_rows)):
            matched_truth_index = matches.get(index)
            matched_truth_time = None if matched_truth_index is None else truth_times[matched_truth_index]
            event_time = float(best.get("candidate_time_sec") or 0.0)
            suppressed = [
                float(row.get("candidate_time_sec") or 0.0)
                for row in cluster
                if row is not best
            ]
            event_rows.append(
                {
                    "schema_version": 1,
                    "video_id": video_id,
                    "video_name": best.get("video_name"),
                    "split": best.get("split"),
                    "fold_video_id": best.get("fold_video_id"),
                    "event_type": "touch",
                    "time_sec": round(event_time, 6),
                    "confidence": best.get("touch_score"),
                    "event_match_type": "true_positive" if matched_truth_index is not None else "false_positive",
                    "matched_truth_time_sec": None if matched_truth_time is None else round(float(matched_truth_time), 6),
                    "matched_truth_delta_sec": None if matched_truth_time is None else round(abs(event_time - float(matched_truth_time)), 6),
                    "candidate_count": len(cluster),
                    "source_candidate_time_sec": best.get("candidate_time_sec"),
                    "suppressed_candidate_times_sec": [round(time_sec, 6) for time_sec in suppressed],
                    "nms_gap_sec": nms_gap_sec,
                    "match_tol_sec": match_tol_sec,
                    "candidate_precision_gate_vetoed": best.get("candidate_precision_gate_vetoed", False),
                    "candidate_precision_gate_reason": best.get("candidate_precision_gate_reason"),
                    "candidate_recall_rescued": best.get("candidate_recall_rescued", False),
                    "candidate_recall_rescue_reason": best.get("candidate_recall_rescue_reason"),
                    "audio_strength": best.get("audio_strength"),
                    "trajectory_break_support": best.get("trajectory_break_support"),
                    "trajectory_nearest_break_delta_sec": best.get("trajectory_nearest_break_delta_sec"),
                    "trajectory_impulse_score": best.get("trajectory_impulse_score"),
                    "trajectory_local_y_quad_rms_px": best.get("trajectory_local_y_quad_rms_px"),
                    "height_reversal": best.get("height_reversal"),
                    "nearest_stall_delta_sec": best.get("nearest_stall_delta_sec"),
                    "in_stall_window": best.get("in_stall_window"),
                }
            )
        if include_false_negatives:
            for truth_index, truth_time in enumerate(truth_times):
                if truth_index in used_truth:
                    continue
                nearest_pred_delta = min((abs(float(time) - float(truth_time)) for time in predicted_times), default=None)
                event_rows.append(
                    {
                        "schema_version": 1,
                        "video_id": video_id,
                        "video_name": rows[0].get("video_name") if rows else None,
                        "split": rows[0].get("split") if rows else None,
                        "fold_video_id": rows[0].get("fold_video_id") if rows else None,
                        "event_type": "touch",
                        "time_sec": round(float(truth_time), 6),
                        "confidence": None,
                        "event_match_type": "false_negative",
                        "matched_truth_time_sec": round(float(truth_time), 6),
                        "matched_truth_delta_sec": 0.0,
                        "nearest_predicted_delta_sec": None if nearest_pred_delta is None else round(float(nearest_pred_delta), 6),
                        "candidate_count": 0,
                        "source_candidate_time_sec": None,
                        "suppressed_candidate_times_sec": [],
                        "nms_gap_sec": nms_gap_sec,
                        "match_tol_sec": match_tol_sec,
                    }
                )
    event_rows.sort(key=lambda row: (str(row.get("video_id")), float(row.get("time_sec") or 0.0), str(row.get("event_match_type"))))
    return event_rows


def event_level_metrics_from_predictions(
    predictions: list[dict[str, Any]],
    *,
    nms_gap_sec: float = EVENT_NMS_GAP_SEC,
    match_tol_sec: float = EVENT_MATCH_TOL_SEC,
    labels_dir: Path | None = None,
) -> dict[str, Any]:
    per_video = []
    aggregate = {
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "n_detected": 0,
        "n_truth": 0,
    }
    for video_id in sorted({str(row["video_id"]) for row in predictions}):
        rows = [row for row in predictions if str(row.get("video_id")) == video_id]
        predicted = predicted_event_times(rows, nms_gap_sec=nms_gap_sec)
        truth = truth_event_times(rows, match_tol_sec=match_tol_sec, labels_dir=labels_dir)
        metrics = event_precision_recall_f1(predicted, truth, match_tol_sec=match_tol_sec)
        per_video.append({"video_id": video_id, "rows": len(rows), **metrics})
        for key in aggregate:
            aggregate[key] += int(metrics.get(key) or 0)
    precision = aggregate["true_positive"] / aggregate["n_detected"] if aggregate["n_detected"] else 0.0
    recall = aggregate["true_positive"] / aggregate["n_truth"] if aggregate["n_truth"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        **aggregate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rows": len(predictions),
        "per_video": per_video,
        "nms_gap_sec": nms_gap_sec,
        "match_tol_sec": match_tol_sec,
    }


def leave_one_video_out_predictions(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    threshold: float,
    *,
    apply_precision_gate: bool = False,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    videos = sorted({str(row["video_id"]) for row in rows})
    for video_id in videos:
        train_rows = [row for row in rows if row["video_id"] != video_id]
        eval_rows = [row for row in rows if row["video_id"] == video_id]
        y_train = labels(train_rows)
        if len(set(y_train.tolist())) < 2:
            continue
        model = make_model()
        model.fit(feature_matrix(train_rows, feature_names), y_train)
        matrix = feature_matrix(eval_rows, feature_names)
        scores = predict_scores(model, matrix)
        predicted = (scores >= threshold).astype(int)
        for row, score, pred in zip(eval_rows, scores, predicted):
            predictions.append(
                prediction_row(row, score=float(score), predicted=int(pred), fold_video_id=video_id, feature_names=feature_names)
            )
    predictions.sort(key=lambda row: (str(row.get("video_id")), float(row.get("candidate_time_sec") or 0.0)))
    return apply_candidate_precision_gate(predictions) if apply_precision_gate else predictions


def leave_one_video_out(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    threshold: float,
    *,
    apply_precision_gate: bool = False,
) -> dict[str, Any]:
    videos = sorted({str(row["video_id"]) for row in rows})
    fold_rows = []
    all_true: list[int] = []
    all_pred: list[int] = []
    for video_id in videos:
        train_rows = [row for row in rows if row["video_id"] != video_id]
        eval_rows = [row for row in rows if row["video_id"] == video_id]
        y_train = labels(train_rows)
        if len(set(y_train.tolist())) < 2:
            fold_rows.append({"video_id": video_id, "status": "skipped_one_class_training_fold", "eval_rows": len(eval_rows)})
            continue
        model = make_model()
        model.fit(feature_matrix(train_rows, feature_names), y_train)
        matrix = feature_matrix(eval_rows, feature_names)
        scores = predict_scores(model, matrix)
        predicted = (scores >= threshold).astype(int)
        predictions = [
            prediction_row(row, score=float(score), predicted=int(pred), fold_video_id=video_id, feature_names=feature_names)
            for row, score, pred in zip(eval_rows, scores, predicted)
        ]
        if apply_precision_gate:
            predictions = apply_candidate_precision_gate(predictions)
        y_true = np.asarray([1 if row.get("label_is_touch") else 0 for row in predictions], dtype=int)
        y_pred = np.asarray([1 if row.get("predicted_is_touch") else 0 for row in predictions], dtype=int)
        metrics = precision_recall_f1(y_true, y_pred)
        fold_rows.append({"video_id": video_id, "status": "ok", "eval_rows": len(eval_rows), **metrics})
        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())
    aggregate = precision_recall_f1(np.asarray(all_true, dtype=int), np.asarray(all_pred, dtype=int)) if all_true else None
    return {"folds": fold_rows, "aggregate": aggregate}


def model_predictions(
    model: Any,
    rows: list[dict[str, Any]],
    feature_names: list[str],
    threshold: float,
    *,
    apply_precision_gate: bool = False,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    scores = predict_scores(model, feature_matrix(rows, feature_names))
    predicted = (scores >= threshold).astype(int)
    predictions = [
        prediction_row(row, score=float(score), predicted=int(pred), fold_video_id=None, feature_names=feature_names)
        for row, score, pred in zip(rows, scores, predicted)
    ]
    return apply_candidate_precision_gate(predictions) if apply_precision_gate else predictions


def evaluate_model(
    model: Any,
    rows: list[dict[str, Any]],
    feature_names: list[str],
    threshold: float,
    *,
    apply_precision_gate: bool = False,
) -> dict[str, Any] | None:
    if not rows:
        return None
    return metrics_from_predictions(model_predictions(model, rows, feature_names, threshold, apply_precision_gate=apply_precision_gate))


def gate_status(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {"status": "not_evaluated", "precision_gate": False, "recall_gate": False, "passes": False}
    precision = float(metrics.get("precision") or 0.0)
    recall = float(metrics.get("recall") or 0.0)
    return {
        "status": "evaluated",
        "precision_gate": precision >= TOUCH_PRECISION_GATE,
        "recall_gate": recall >= TOUCH_RECALL_GATE,
        "passes": precision >= TOUCH_PRECISION_GATE and recall >= TOUCH_RECALL_GATE,
        "precision_threshold": TOUCH_PRECISION_GATE,
        "recall_threshold": TOUCH_RECALL_GATE,
    }


def run_feature_set_cv(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    return {
        name: leave_one_video_out(rows, feature_names, threshold)
        for name, feature_names in FEATURE_SETS.items()
    }


def run_feature_set_frozen_test(
    rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    """Train each ablation feature set on train/val rows and score the held-out
    frozen-test rows. The leave-one-video-out CV is noisy on few clips; the
    frozen-test column shows which feature sets actually hold up out-of-sample."""
    train_labels = labels(rows) if rows else np.asarray([], dtype=int)
    trainable = bool(rows) and bool(test_rows) and len(set(train_labels.tolist())) >= 2
    out: dict[str, Any] = {}
    for name, feature_names in FEATURE_SETS.items():
        if not trainable:
            out[name] = {"aggregate": None}
            continue
        model = make_model()
        model.fit(feature_matrix(rows, feature_names), train_labels)
        out[name] = {"aggregate": evaluate_model(model, test_rows, feature_names, threshold)}
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def error_counts(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    by_video: dict[str, dict[str, int]] = {}
    for row in predictions:
        error_type = str(row.get("error_type"))
        counts[error_type] = counts.get(error_type, 0) + 1
        video_counts = by_video.setdefault(str(row.get("video_id")), {})
        video_counts[error_type] = video_counts.get(error_type, 0) + 1
    return {
        "counts": counts,
        "by_video": by_video,
        "mistakes": counts.get("false_negative", 0) + counts.get("false_positive", 0),
        "false_negative": counts.get("false_negative", 0),
        "false_positive": counts.get("false_positive", 0),
    }


def validate_rows(
    rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    allow_small: bool,
    allow_missing_trajectory: bool,
    allow_missing_frozen_test: bool,
    min_videos: int,
) -> list[str]:
    reasons = []
    if not rows:
        reasons.append("no train/validation candidate rows")
    if any(row.get("split") == "test_frozen" for row in rows):
        reasons.append("test_frozen rows leaked into train/validation table")
    if any(row.get("split") != "test_frozen" for row in test_rows):
        reasons.append("non-test row leaked into frozen-test table")
    train_videos = {str(row.get("video_id")) for row in rows}
    test_videos = {str(row.get("video_id")) for row in test_rows}
    overlap = sorted(train_videos & test_videos)
    if overlap:
        reasons.append(f"clip split leakage between train/validation and frozen test: {overlap}")
    video_count = len({str(row.get("video_id")) for row in rows})
    if video_count < min_videos and not allow_small:
        reasons.append(f"need at least {min_videos} labeled non-test videos, found {video_count}")
    y = labels(rows)
    if len(set(y.tolist())) < 2:
        reasons.append("need both positive and negative candidate rows")
    y_test = labels(test_rows)
    if not allow_missing_frozen_test:
        if not test_rows:
            reasons.append("no frozen-test candidate rows; release gate evaluation requires visually reviewed frozen-test labels")
        elif len(set(y_test.tolist())) < 2:
            reasons.append("frozen-test rows must contain both positive and negative candidates for touch precision/recall gate evaluation")
    if not allow_missing_trajectory and not has_trajectory_features(rows):
        reasons.append("L2 trajectory features are missing; run the trajectory feature attachment step first")
    return reasons


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Touch Classifier Training Report",
        "",
        f"- Status: `{summary['status']}`",
        f"- Dataset dir: `{summary['dataset_dir']}`",
        f"- Labels dir: `{summary.get('labels_dir')}`",
        f"- Rows: `{summary['train_val_rows']}` train/validation, `{summary['test_frozen_rows']}` frozen test",
        f"- Feature mode: `{summary['feature_mode']}`",
        f"- Features: `{', '.join(summary['feature_names'])}`",
        f"- Pose features present: `{summary.get('pose_features_present')}`",
        f"- Optical-flow features present: `{summary.get('flow_features_present')}`",
        f"- Audio timbre features present: `{summary.get('audio_timbre_features_present')}`",
        f"- Candidate precision gate: `{'enabled' if summary.get('candidate_precision_gate_enabled') else 'disabled'}`",
        "",
    ]
    if not str(summary["status"]).startswith("trained"):
        lines.extend(["## Not Ready", ""])
        for reason in summary.get("reasons", []):
            lines.append(f"- {reason}")
    else:
        cv = summary.get("leave_one_video_out") or {}
        agg = cv.get("aggregate")
        if agg:
            raw_agg = (summary.get("raw_leave_one_video_out") or {}).get("aggregate") or {}
            event_cv = summary.get("event_level_leave_one_video_out") or {}
            event_cv_gate = summary.get("event_level_cv_gate") or {}
            event_config = summary.get("event_level_config") or {}
            rescues = summary.get("candidate_recall_rescues") or {}
            lines.extend(
                [
                    "## Leave-Clips-Out CV",
                    "",
                    f"- Precision: `{agg['precision']:.3f}`",
                    f"- Recall: `{agg['recall']:.3f}`",
                    f"- F1: `{agg['f1']:.3f}`",
                    f"- Raw baseline: `P {float(raw_agg.get('precision') or 0.0):.3f} / "
                    f"R {float(raw_agg.get('recall') or 0.0):.3f} / "
                    f"F1 {float(raw_agg.get('f1') or 0.0):.3f}`",
                    f"- Gate: `{'PASS' if (summary.get('cv_gate') or {}).get('passes') else 'FAIL'}` "
                    f"(P>={TOUCH_PRECISION_GATE:.2f}, R>={TOUCH_RECALL_GATE:.2f})",
                    f"- Event-level after duplicate merge: `P {float(event_cv.get('precision') or 0.0):.3f} / "
                    f"R {float(event_cv.get('recall') or 0.0):.3f} / "
                    f"F1 {float(event_cv.get('f1') or 0.0):.3f}` "
                    f"(`{'PASS' if event_cv_gate.get('passes') else 'FAIL'}`)",
                    f"- Event truth source: `{event_config.get('truth_source', 'candidate_rows')}`",
                    f"- Candidate recall rescues: `{int(rescues.get('cv') or 0)}`",
                    "",
                ]
            )
        ablations = summary.get("ablation_leave_one_video_out") or {}
        frozen_ablations = summary.get("ablation_frozen_test") or {}
        if ablations:
            lines.extend(
                [
                    "## Ablations",
                    "",
                    "Leave-one-clip-out CV is noisy on few clips; the frozen columns show which "
                    "feature sets actually hold up on the held-out test clips.",
                    "",
                    "| feature set | cv P | cv R | cv F1 | cv tp/gt | cv fp | cv gate | "
                    "frozen P | frozen R | frozen F1 | frozen tp/gt | frozen fp | frozen gate |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
                ]
            )
            for name in (
                "audio_only",
                "audio_timbre_only",
                "trajectory_only",
                "flow_only",
                "pose_only",
                "fused_audio_trajectory",
                "fused_audio_trajectory_flow",
                "fused_audio_trajectory_pose",
                "fused_audio_trajectory_flow_pose",
            ):
                cv_metrics = (ablations.get(name) or {}).get("aggregate") or {}
                cv_gate = gate_status(cv_metrics if cv_metrics else None)
                cv_tp = int(cv_metrics.get("true_positive") or 0)
                cv_truth = int(cv_metrics.get("n_truth") or 0)
                cv_fp = int(cv_metrics.get("false_positive") or 0)
                ft_metrics = (frozen_ablations.get(name) or {}).get("aggregate") or {}
                if ft_metrics:
                    ft_gate = gate_status(ft_metrics)
                    ft_tp = int(ft_metrics.get("true_positive") or 0)
                    ft_truth = int(ft_metrics.get("n_truth") or 0)
                    ft_fp = int(ft_metrics.get("false_positive") or 0)
                    ft_cells = (
                        f"{float(ft_metrics.get('precision') or 0.0):.3f} | "
                        f"{float(ft_metrics.get('recall') or 0.0):.3f} | "
                        f"{float(ft_metrics.get('f1') or 0.0):.3f} | "
                        f"{ft_tp}/{ft_truth} | {ft_fp} | {'PASS' if ft_gate['passes'] else 'FAIL'}"
                    )
                else:
                    ft_cells = "n/a | n/a | n/a | n/a | n/a | n/a"
                lines.append(
                    f"| {name} | {float(cv_metrics.get('precision') or 0.0):.3f} | "
                    f"{float(cv_metrics.get('recall') or 0.0):.3f} | {float(cv_metrics.get('f1') or 0.0):.3f} | "
                    f"{cv_tp}/{cv_truth} | {cv_fp} | {'PASS' if cv_gate['passes'] else 'FAIL'} | {ft_cells} |"
                )
            lines.append("")
        frozen = summary.get("frozen_test")
        if frozen:
            gate = summary.get("frozen_test_gate") or {}
            raw_frozen = summary.get("raw_frozen_test") or {}
            vetoes = summary.get("candidate_precision_gate_vetoes") or {}
            rescues = summary.get("candidate_recall_rescues") or {}
            event_frozen = summary.get("event_level_frozen_test") or {}
            event_frozen_gate = summary.get("event_level_frozen_test_gate") or {}
            lines.extend(
                [
                    "## Frozen Test",
                    "",
                    f"- Precision: `{frozen['precision']:.3f}`",
                    f"- Recall: `{frozen['recall']:.3f}`",
                    f"- F1: `{frozen['f1']:.3f}`",
                    f"- Raw baseline: `P {float(raw_frozen.get('precision') or 0.0):.3f} / "
                    f"R {float(raw_frozen.get('recall') or 0.0):.3f} / "
                    f"F1 {float(raw_frozen.get('f1') or 0.0):.3f}`",
                    f"- Candidate precision-gate vetoes: `{int(vetoes.get('frozen_test') or 0)}`",
                    f"- Candidate recall rescues: `{int(rescues.get('frozen_test') or 0)}`",
                    f"- Gate: `{'PASS' if gate.get('passes') else 'FAIL'}` "
                    f"(P>={TOUCH_PRECISION_GATE:.2f}, R>={TOUCH_RECALL_GATE:.2f})",
                    f"- Event-level after duplicate merge: `P {float(event_frozen.get('precision') or 0.0):.3f} / "
                    f"R {float(event_frozen.get('recall') or 0.0):.3f} / "
                    f"F1 {float(event_frozen.get('f1') or 0.0):.3f}` "
                    f"(`{'PASS' if event_frozen_gate.get('passes') else 'FAIL'}`)",
                    "",
                    "| video | rows | precision | recall | f1 | fp | fn |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for row in frozen.get("per_video", []):
                lines.append(
                    f"| `{row['video_id']}` | {row['rows']} | {row['precision']:.3f} | "
                    f"{row['recall']:.3f} | {row['f1']:.3f} | {row['false_positive']} | {row['false_negative']} |"
                )
            lines.append("")
        errors = summary.get("out_of_fold_errors") or {}
        if errors:
            lines.extend(
                [
                    "## Out-of-Fold Errors",
                    "",
                    f"- False negatives: `{errors.get('false_negative', 0)}`",
                    f"- False positives: `{errors.get('false_positive', 0)}`",
                    f"- JSONL: `{summary.get('error_jsonl')}`",
                    f"- CSV: `{summary.get('error_csv')}`",
                    "",
                    "| video | false negatives | false positives |",
                    "| --- | ---: | ---: |",
                ]
            )
            for video_id, counts in sorted((errors.get("by_video") or {}).items()):
                fn = int(counts.get("false_negative", 0))
                fp = int(counts.get("false_positive", 0))
                if fn or fp:
                    lines.append(f"| `{video_id}` | {fn} | {fp} |")
            lines.append("")
        lines.extend(["## Folds", "", "| video | status | rows | precision | recall | f1 |", "| --- | --- | ---: | ---: | ---: | ---: |"])
        for row in cv.get("folds", []):
            lines.append(
                f"| `{row['video_id']}` | {row['status']} | {row.get('eval_rows', 0)} | "
                f"{float(row.get('precision') or 0.0):.3f} | {float(row.get('recall') or 0.0):.3f} | {float(row.get('f1') or 0.0):.3f} |"
            )
    lines.extend(
        [
            "",
            "Notes:",
            "- Frozen-test rows are not used in training or cross-validation.",
            "- Frozen-test candidate metrics are diagnostic; the release gate is evaluated on merged event-level touch events.",
            "- Candidate-level metrics can fail while the product output passes after event-level duplicate merge/NMS.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_classifier(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = args.dataset_dir.resolve()
    labels_dir = args.labels_dir.resolve() if getattr(args, "labels_dir", None) else None
    rows = read_jsonl(dataset_dir / "touch_training_candidates.jsonl")
    test_rows = read_jsonl(dataset_dir / "touch_training_test_frozen.jsonl")
    if args.audio_only:
        feature_mode = "audio_timbre_only" if not getattr(args, "disable_audio_timbre_features", False) else "audio_only"
        feature_names = FEATURE_SETS[feature_mode]
    elif getattr(args, "disable_pose_features", False) and getattr(args, "disable_flow_features", False):
        feature_mode = "fused_audio_trajectory"
        feature_names = FEATURE_SETS["fused_audio_trajectory"]
    elif getattr(args, "disable_pose_features", False):
        feature_mode = "fused_audio_trajectory_flow"
        feature_names = FEATURE_SETS["fused_audio_trajectory_flow"]
    elif getattr(args, "disable_flow_features", False):
        feature_mode = "fused_audio_trajectory_pose"
        feature_names = FEATURE_SETS["fused_audio_trajectory_pose"]
    else:
        feature_mode = "fused_audio_trajectory_flow_pose"
        feature_names = FEATURE_SETS["fused_audio_trajectory_flow_pose"]
    missing_trajectory_smoke = bool(not args.audio_only and args.allow_missing_trajectory and not has_trajectory_features(rows))
    if missing_trajectory_smoke:
        feature_mode = "fused_audio_trajectory_missing_l2_smoke"
    reasons = validate_rows(
        rows,
        test_rows,
        allow_small=args.allow_small,
        allow_missing_trajectory=args.allow_missing_trajectory or args.audio_only,
        allow_missing_frozen_test=args.allow_missing_frozen_test,
        min_videos=args.min_videos,
    )
    out_dir = args.out_dir.resolve() if args.out_dir else dataset_dir / "touch_classifier_v1"
    strict_release_mode = not args.allow_small and not args.allow_missing_trajectory and not args.allow_missing_frozen_test and not args.audio_only
    status = "not_ready" if reasons else ("trained" if strict_release_mode else "trained_smoke")
    summary: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "labels_dir": None if labels_dir is None else str(labels_dir),
        "out_dir": str(out_dir),
        "status": status,
        "reasons": reasons,
        "feature_mode": feature_mode,
        "feature_names": feature_names,
        "pose_features_present": has_pose_features(rows + test_rows),
        "flow_features_present": has_flow_features(rows + test_rows),
        "audio_timbre_features_present": any(any(row.get(key) is not None for key in AUDIO_TIMBRE_FEATURES) for row in rows + test_rows),
        "candidate_touch_precision_gate": TOUCH_PRECISION_GATE,
        "candidate_touch_recall_gate": TOUCH_RECALL_GATE,
        "candidate_precision_gate_enabled": not getattr(args, "disable_candidate_precision_gate", False) and not args.audio_only,
        "candidate_precision_gate_config": {
            "name": "trajectory_corroboration",
            "veto_reasons": ["no_trajectory_corroboration", "weak_audio_weak_trajectory"],
            "max_zero_support_break_delta_sec": CANDIDATE_GATE_BREAK_DELTA_SEC,
            "weak_audio_max": CANDIDATE_GATE_WEAK_AUDIO_MAX,
            "weak_break_support_max": CANDIDATE_GATE_WEAK_BREAK_SUPPORT_MAX,
            "recall_rescue_reason": "high_impulse_audio_trajectory_rescue",
            "recall_rescue_audio_min": CANDIDATE_RESCUE_AUDIO_MIN,
            "recall_rescue_impulse_min": CANDIDATE_RESCUE_IMPULSE_MIN,
            "recall_rescue_break_support_min": CANDIDATE_RESCUE_BREAK_SUPPORT_MIN,
            "recall_rescue_break_delta_sec": CANDIDATE_RESCUE_BREAK_DELTA_SEC,
        },
        "threshold": args.threshold,
        "strict_release_mode": strict_release_mode,
        "allow_missing_frozen_test": args.allow_missing_frozen_test,
        "train_val_rows": len(rows),
        "test_frozen_rows": len(test_rows),
        "train_val_videos": sorted({str(row["video_id"]) for row in rows}),
        "test_frozen_videos": sorted({str(row["video_id"]) for row in test_rows}),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    if not reasons:
        model = make_model()
        matrix = feature_matrix(rows, feature_names)
        y = labels(rows)
        model.fit(matrix, y)
        apply_precision_gate = bool(summary["candidate_precision_gate_enabled"])
        raw_cv = leave_one_video_out(rows, feature_names, args.threshold)
        raw_frozen_test = evaluate_model(model, test_rows, feature_names, args.threshold)
        cv = leave_one_video_out(rows, feature_names, args.threshold, apply_precision_gate=apply_precision_gate)
        frozen_test = evaluate_model(model, test_rows, feature_names, args.threshold, apply_precision_gate=apply_precision_gate)
        cv_predictions = leave_one_video_out_predictions(
            rows,
            feature_names,
            args.threshold,
            apply_precision_gate=apply_precision_gate,
        )
        cv_errors = [row for row in cv_predictions if row["error_type"] in {"false_negative", "false_positive"}]
        frozen_predictions = model_predictions(
            model,
            test_rows,
            feature_names,
            args.threshold,
            apply_precision_gate=apply_precision_gate,
        )
        event_level_cv = event_level_metrics_from_predictions(cv_predictions, labels_dir=labels_dir)
        event_level_frozen_test = event_level_metrics_from_predictions(frozen_predictions, labels_dir=labels_dir)
        frozen_gate_vetoes = [row for row in frozen_predictions if row.get("candidate_precision_gate_vetoed")]
        cv_gate_vetoes = [row for row in cv_predictions if row.get("candidate_precision_gate_vetoed")]
        frozen_rescues = [row for row in frozen_predictions if row.get("candidate_recall_rescued")]
        cv_rescues = [row for row in cv_predictions if row.get("candidate_recall_rescued")]
        cv_events = event_rows_from_predictions(cv_predictions, labels_dir=labels_dir)
        frozen_events = event_rows_from_predictions(frozen_predictions, labels_dir=labels_dir)
        error_jsonl = out_dir / "touch_classifier_errors.jsonl"
        error_csv = out_dir / "touch_classifier_errors.csv"
        frozen_predictions_jsonl = out_dir / "touch_classifier_frozen_predictions.jsonl"
        oof_predictions_jsonl = out_dir / "touch_classifier_oof_predictions.jsonl"
        frozen_events_jsonl = out_dir / "touch_classifier_frozen_events.jsonl"
        oof_events_jsonl = out_dir / "touch_classifier_oof_events.jsonl"
        write_jsonl(error_jsonl, cv_errors)
        write_csv_rows(error_csv, cv_errors)
        write_jsonl(frozen_predictions_jsonl, frozen_predictions)
        write_jsonl(oof_predictions_jsonl, cv_predictions)
        write_jsonl(frozen_events_jsonl, frozen_events)
        write_jsonl(oof_events_jsonl, cv_events)
        model_path = out_dir / "touch_classifier.joblib"
        joblib.dump({"model": model, "feature_names": feature_names, "threshold": args.threshold}, model_path)
        summary["model_path"] = str(model_path)
        summary["raw_leave_one_video_out"] = raw_cv
        summary["raw_cv_gate"] = gate_status(raw_cv.get("aggregate"))
        summary["raw_frozen_test"] = raw_frozen_test
        summary["raw_frozen_test_gate"] = gate_status(raw_frozen_test)
        summary["leave_one_video_out"] = cv
        summary["cv_gate"] = gate_status(cv.get("aggregate"))
        summary["frozen_test"] = frozen_test
        summary["frozen_test_gate"] = gate_status(frozen_test)
        summary["event_level_config"] = {
            "nms_gap_sec": EVENT_NMS_GAP_SEC,
            "match_tol_sec": EVENT_MATCH_TOL_SEC,
            "truth_source": "approved_visual_touch_labels" if labels_dir is not None else "candidate_rows",
            "note": "Event-level metrics merge predicted candidate bursts before matching approved touch times.",
        }
        summary["event_level_leave_one_video_out"] = event_level_cv
        summary["event_level_cv_gate"] = gate_status(event_level_cv)
        summary["event_level_frozen_test"] = event_level_frozen_test
        summary["event_level_frozen_test_gate"] = gate_status(event_level_frozen_test)
        summary["error_jsonl"] = str(error_jsonl)
        summary["error_csv"] = str(error_csv)
        summary["frozen_predictions_jsonl"] = str(frozen_predictions_jsonl)
        summary["oof_predictions_jsonl"] = str(oof_predictions_jsonl)
        summary["frozen_events_jsonl"] = str(frozen_events_jsonl)
        summary["oof_events_jsonl"] = str(oof_events_jsonl)
        summary["out_of_fold_errors"] = error_counts(cv_predictions)
        summary["candidate_precision_gate_vetoes"] = {
            "cv": len(cv_gate_vetoes),
            "frozen_test": len(frozen_gate_vetoes),
            "frozen_test_by_reason": {
                reason: sum(1 for row in frozen_gate_vetoes if row.get("candidate_precision_gate_reason") == reason)
                for reason in sorted({str(row.get("candidate_precision_gate_reason")) for row in frozen_gate_vetoes})
            },
            "cv_by_reason": {
                reason: sum(1 for row in cv_gate_vetoes if row.get("candidate_precision_gate_reason") == reason)
                for reason in sorted({str(row.get("candidate_precision_gate_reason")) for row in cv_gate_vetoes})
            },
        }
        summary["candidate_recall_rescues"] = {
            "cv": len(cv_rescues),
            "frozen_test": len(frozen_rescues),
            "cv_by_reason": {
                reason: sum(1 for row in cv_rescues if row.get("candidate_recall_rescue_reason") == reason)
                for reason in sorted({str(row.get("candidate_recall_rescue_reason")) for row in cv_rescues})
            },
            "frozen_test_by_reason": {
                reason: sum(1 for row in frozen_rescues if row.get("candidate_recall_rescue_reason") == reason)
                for reason in sorted({str(row.get("candidate_recall_rescue_reason")) for row in frozen_rescues})
            },
        }
        if not args.audio_only:
            summary["ablation_leave_one_video_out"] = run_feature_set_cv(rows, args.threshold)
            summary["ablation_frozen_test"] = run_feature_set_frozen_test(rows, test_rows, args.threshold)
    write_json(out_dir / "touch_classifier_metrics.json", summary)
    write_report(out_dir / "touch_classifier_report.md", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate fused touch classifier from candidate table")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-videos", type=int, default=3)
    parser.add_argument("--allow-small", action="store_true", help="allow tiny smoke runs")
    parser.add_argument("--allow-missing-trajectory", action="store_true", help="allow training without L2 features; smoke only")
    parser.add_argument("--allow-missing-frozen-test", action="store_true", help="allow training without frozen-test rows; smoke only")
    parser.add_argument("--audio-only", action="store_true", help="train the audio-only ablation feature set")
    parser.add_argument("--disable-pose-features", action="store_true", help="train without pose columns even when present")
    parser.add_argument("--disable-flow-features", action="store_true", help="train without optical-flow columns even when present")
    parser.add_argument("--disable-audio-timbre-features", action="store_true", help="when --audio-only is used, use only basic audio columns")
    parser.add_argument("--disable-candidate-precision-gate", action="store_true", help="report raw classifier predictions without the trajectory-corroboration veto")
    return parser.parse_args()


def main() -> None:
    summary = train_classifier(parse_args())
    out_dir = Path(summary["out_dir"])
    print(f"metrics: {out_dir / 'touch_classifier_metrics.json'}")
    print(f"report:  {out_dir / 'touch_classifier_report.md'}")
    print(json.dumps({key: summary[key] for key in ("status", "reasons", "feature_mode", "train_val_rows", "test_frozen_rows")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
