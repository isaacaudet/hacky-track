#!/usr/bin/env python3
"""Shadow precision gate for QA candidate events.

This module is intentionally side-effect free. It scores an event and returns
diagnostic veto reasons, but it does not remove or mutate events by itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


OUT_SIZE = (688.0, 912.0)
AMBIGUOUS_TOUCH_TYPES = {"unknown_contact", "ground_candidate", "foot_candidate", "knee_candidate"}
NONFOOT_TOUCH_TYPES = {"unknown_contact", "ground_candidate"}


@dataclass(frozen=True)
class GateDecision:
    veto: bool
    hard_veto: bool
    soft_flag: bool
    score: float
    hard_reasons: tuple[str, ...]
    soft_reasons: tuple[str, ...]
    reasons: tuple[str, ...]
    features: dict[str, float | str | None]


def number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def event_kind(event: dict[str, Any]) -> str:
    kind = str(event.get("kind") or event.get("type") or "touch")
    if kind in {"drop", "floor", "floor_reset"}:
        return "drop_floor"
    return kind


def y_ratio(event: dict[str, Any]) -> float | None:
    y = number(event.get("qa_ball_y", event.get("candidate_base_y", event.get("y"))))
    return None if y is None else y / OUT_SIZE[1]


def tracker_disagreement(event: dict[str, Any]) -> float | None:
    values = [
        number(event.get("calibrated_agreement_px")),
        number(event.get("yolo_v10t_agreement_px")),
        number(event.get("yolo_v11_agreement_px")),
        number(event.get("yolo_v10g_agreement_px")),
    ]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return min(values)


def gate_touch(event: dict[str, Any]) -> tuple[float, list[str], float, list[str]]:
    hard_reasons: list[str] = []
    soft_reasons: list[str] = []
    hard_score = 0.0
    soft_score = 0.0
    contact_type = str(event.get("contact_type") or event.get("detector_contact_type") or "")
    contact_conf = number(event.get("contact_confidence")) or 0.0
    foot_distance = number(event.get("foot_distance"))
    foot_distance_f = 999.0 if foot_distance is None else foot_distance
    audio_z = number(event.get("audio_z")) or 0.0
    visual_score = number(event.get("visual_score")) or 0.0
    motion_score = number(event.get("motion_score")) or 0.0
    disagreement = tracker_disagreement(event)
    dy = number(event.get("velocity_dvy_at_touch_moment"))
    yr = y_ratio(event)

    if contact_type in NONFOOT_TOUCH_TYPES and (foot_distance_f >= 220 or contact_conf < 0.82):
        soft_reasons.append("nonfoot_contact_far_or_weak")
        soft_score += 0.46

    if contact_type == "foot_candidate" and foot_distance_f >= 145 and disagreement is not None and disagreement >= 180:
        soft_reasons.append("ambiguous_foot_candidate_tracker_disagreement")
        soft_score += 0.40

    if contact_type in AMBIGUOUS_TOUCH_TYPES and foot_distance_f >= 240:
        soft_reasons.append("ambiguous_contact_far_from_limb")
        soft_score += 0.34

    if contact_type != "foot" and audio_z < 16 and visual_score < 0.45 and motion_score < 0.35:
        soft_reasons.append("weak_multimodal_support")
        soft_score += 0.24

    if yr is not None and yr >= 0.94 and contact_type != "foot" and foot_distance_f >= 180:
        soft_reasons.append("near_floor_nonfoot_touch")
        soft_score += 0.30

    if disagreement is not None and disagreement >= 300 and contact_type != "foot":
        soft_reasons.append("severe_tracker_disagreement")
        soft_score += 0.30

    if dy is not None and dy > 150 and contact_type != "foot":
        soft_reasons.append("no_upward_rebound_signature")
        soft_score += 0.16

    # Hard touch vetoes need a much higher bar than review-routing flags. The
    # broad ambiguous/far signatures catch real touches in sequence context, so
    # hard suppression only fires on weak non-foot candidates with multiple
    # independent negatives.
    weak_nonfoot_combo = (
        contact_type in NONFOOT_TOUCH_TYPES
        and foot_distance_f >= 240
        and audio_z < 16
        and visual_score < 0.45
        and motion_score < 0.35
    )
    floor_nonfoot_combo = (
        contact_type in NONFOOT_TOUCH_TYPES
        and yr is not None
        and yr >= 0.94
        and foot_distance_f >= 220
        and visual_score < 0.75
    )
    severe_disagreement_combo = (
        contact_type in NONFOOT_TOUCH_TYPES
        and disagreement is not None
        and disagreement >= 320
        and (audio_z < 18 or motion_score < 0.35)
    )
    if weak_nonfoot_combo:
        hard_reasons.append("hard_weak_nonfoot_multimodal")
        hard_score += 0.50
    if floor_nonfoot_combo:
        hard_reasons.append("hard_near_floor_nonfoot")
        hard_score += 0.44
    if severe_disagreement_combo:
        hard_reasons.append("hard_nonfoot_tracker_disagreement")
        hard_score += 0.44

    return min(1.0, hard_score), hard_reasons, min(1.0, max(hard_score, soft_score)), soft_reasons


def gate_drop(event: dict[str, Any]) -> tuple[float, list[str], float, list[str]]:
    reasons: list[str] = []
    score = 0.0
    time_sec = number(event.get("time_sec")) or 0.0
    foot_distance = number(event.get("foot_distance"))
    foot_distance_f = 0.0 if foot_distance is None else foot_distance
    visual_score = number(event.get("visual_score")) or number(event.get("drop_score")) or 0.0
    disagreement = tracker_disagreement(event)
    dy = number(event.get("velocity_dvy_at_touch_moment"))
    yr = y_ratio(event)
    source = str(event.get("drop_source") or event.get("note") or "").lower()

    if time_sec < 1.60:
        reasons.append("setup_phase_floor_reset")
        score += 0.40

    if disagreement is not None and disagreement >= 420:
        reasons.append("severe_floor_tracker_disagreement")
        score += 0.46

    if yr is not None and yr < 0.90:
        reasons.append("floor_reset_not_near_floor")
        score += 0.36

    if foot_distance_f >= 450:
        reasons.append("floor_reset_far_from_limb_context")
        score += 0.22

    if dy is not None and dy > 450 and visual_score < 0.98:
        reasons.append("unstable_floor_motion_signature")
        score += 0.22

    if "track-context" in source:
        reasons.append("audio_track_context_not_floor_reset")
        score += 0.34

    return min(1.0, score), reasons, min(1.0, score), reasons


def evaluate_precision_gate(event: dict[str, Any]) -> GateDecision:
    """Return the shadow precision-gate decision for an event-like mapping."""
    kind = event_kind(event)
    if kind == "touch":
        hard_score, hard_reasons, score, soft_reasons = gate_touch(event)
    elif kind == "drop_floor":
        hard_score, hard_reasons, score, soft_reasons = gate_drop(event)
    else:
        hard_score, hard_reasons, score, soft_reasons = 0.0, [], 0.0, []

    hard_veto = hard_score >= 0.40
    soft_flag = bool(hard_veto or soft_reasons)
    reasons = tuple(dict.fromkeys([*hard_reasons, *soft_reasons]))
    features: dict[str, float | str | None] = {
        "kind": kind,
        "contact_type": event.get("contact_type") or event.get("detector_contact_type"),
        "foot_distance": number(event.get("foot_distance")),
        "audio_z": number(event.get("audio_z")),
        "visual_score": number(event.get("visual_score")),
        "motion_score": number(event.get("motion_score")),
        "tracker_disagreement_px": tracker_disagreement(event),
        "velocity_dvy": number(event.get("velocity_dvy_at_touch_moment")),
        "y_ratio": y_ratio(event),
    }
    return GateDecision(
        veto=hard_veto,
        hard_veto=hard_veto,
        soft_flag=soft_flag,
        score=round(score, 4),
        hard_reasons=tuple(hard_reasons),
        soft_reasons=tuple(soft_reasons),
        reasons=reasons,
        features=features,
    )
