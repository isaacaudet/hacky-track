#!/usr/bin/env python3
"""Visual QA and rally metric enrichment for Hacky Track batch outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from detect_atw_overlay import detect_ball, detect_foot, red_ball_mask, shoe_mask
from full_training_run import read_track_csv, reviewed_stall_events, speed_series
from paint_hud import detect_bag_center
from precision_gate import evaluate_precision_gate
from scan_training_data import video_meta
from train_multimodal_detector import AudioPeak, Detection, audio_peaks_for_video


DEFAULT_MANIFEST = Path("outputs/full_training_27/full_training_manifest.json")
DEFAULT_OUT_ROOT = Path("outputs/full_training_27_qa")
OUT_SIZE = (688, 912)
ROOT = Path(__file__).resolve().parent


@dataclass
class BallCandidate:
    x: float
    y: float
    radius: float
    source: str
    confidence: float


@dataclass
class BallFix:
    x: float | None
    y: float | None
    radius: float
    confidence: float
    source: str
    frame_time_sec: float
    correction_px: float | None
    foot_x: float | None
    foot_y: float | None
    foot_confidence: float
    foot_distance: float | None
    contact_side: str
    contact_type: str
    contact_confidence: float
    side_confidence: float
    side_source: str
    side_uncertainty_reason: str | None
    ball_accuracy: str


@dataclass
class ContactFoot:
    x: float
    y: float
    side_x: float
    confidence: float
    distance: float


@dataclass
class FloorResetCandidate:
    time_sec: float
    x: float
    y: float
    radius: float
    confidence: float
    source: str
    frame_time_sec: float
    foot_x: float | None
    foot_y: float | None
    foot_confidence: float
    foot_distance: float | None
    contact_side: str
    score: float
    speed: float
    reason: str
    ground_ratio: float
    sky_ratio: float
    airborne_red_blob: bool


def event_time(event: dict[str, Any]) -> float:
    return float(event.get("time_sec", event.get("start_sec", 0.0)))


def resolve_existing_path(raw: Any) -> Path:
    path = Path(str(raw or ""))
    if path.exists():
        return path
    root_path = ROOT / path
    if root_path.exists():
        return root_path
    return path


def resolve_video_path(raw: Any) -> Path:
    path = Path(str(raw or ""))
    if path.exists():
        return path
    root_path = ROOT / path
    if root_path.exists():
        return root_path
    downloads = Path.home() / "Downloads" / path.name
    if downloads.exists():
        return downloads
    return path


def read_resized_frame(cap: cv2.VideoCapture, fps: float, time_sec: float) -> np.ndarray | None:
    target_frame = max(0, int(round(time_sec * fps)))
    current_frame = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    frame = None
    ok = False
    if 0 <= target_frame - current_frame <= 12:
        for _ in range(target_frame - current_frame + 1):
            ok, frame = cap.read()
            if not ok:
                return None
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ok, frame = cap.read()
    if not ok:
        return None
    return cv2.resize(frame, OUT_SIZE, interpolation=cv2.INTER_AREA)


def nearest_detection(detections: list[Detection], time_sec: float) -> Detection | None:
    if not detections:
        return None
    idx = min(range(len(detections)), key=lambda item: abs(detections[item].time_sec - time_sec))
    return detections[idx]


def dedupe_candidates(candidates: list[BallCandidate]) -> list[BallCandidate]:
    out: list[BallCandidate] = []
    for cand in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        if any(math.hypot(cand.x - kept.x, cand.y - kept.y) < 12 for kept in out):
            continue
        out.append(cand)
    return out


def local_red_snap(frame: np.ndarray, cand: BallCandidate, search_radius: float = 92.0) -> BallCandidate:
    """Prefer a nearby red/orange sack patch over broad multicolor blobs.

    The broad color-blob detector is useful for finding the sack family of
    colors, but it can land on shoe fabric, pants, or grass in POV footage.
    A nearby compact red/orange patch is stronger evidence that the point is on
    the actual footbag, so use it to snap the event center onto the visible bag.
    """

    if cand.source.startswith("red"):
        return cand
    mask = red_ball_mask(frame)
    gate = np.zeros_like(mask)
    cv2.circle(gate, (int(round(cand.x)), int(round(cand.y))), int(round(search_radius)), 255, -1)
    local = cv2.bitwise_and(mask, gate)
    contours, _hierarchy = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[float, float, float, float, float] | None = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 10 or area > 5200:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 1:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius < 3.5 or radius > 62:
            continue
        dist = math.hypot(float(x) - cand.x, float(y) - cand.y)
        if dist > search_radius:
            continue
        score = area * max(0.25, circularity) * math.exp(-(dist * dist) / (2 * 56 * 56)) / max(radius, 1.0)
        if best is None or score > best[0]:
            best = (score, float(x), float(y), float(radius), area)
    if best is None:
        return cand
    score, x, y, radius, area = best
    confidence = max(cand.confidence, min(0.92, 0.56 + min(0.26, area / 1800.0) + min(0.10, score / 260.0)))
    return BallCandidate(x, y, max(8.0, radius), "local_red_snap", confidence)


def ball_candidates(frame: np.ndarray, predicted: tuple[float, float] | None) -> list[BallCandidate]:
    candidates: list[BallCandidate] = []
    ball, conf = detect_ball(frame, None)
    if ball is not None and conf >= 0.04:
        candidates.append(BallCandidate(ball[0], ball[1], ball[2], "red_global", min(1.0, 0.2 + conf * 0.85)))

    if predicted is not None:
        near_ball, near_conf = detect_ball(frame, (predicted[0], predicted[1], 20.0))
        if near_ball is not None and near_conf >= 0.04:
            candidates.append(
                BallCandidate(near_ball[0], near_ball[1], near_ball[2], "red_track_near", min(1.0, 0.16 + near_conf * 0.84))
            )

    center = detect_bag_center(frame)
    if center is not None:
        candidates.append(BallCandidate(center[0], center[1], 18.0, "color_blob", 0.58))

    if predicted is not None:
        candidates.append(BallCandidate(predicted[0], predicted[1], 18.0, "track_fallback", 0.28))
    return dedupe_candidates([local_red_snap(frame, cand) for cand in candidates])


def detect_contact_foot(frame: np.ndarray, ball: tuple[float, float, float]) -> ContactFoot | None:
    """Find the lower-body contour that is actually closest to the sack."""

    mask = shoe_mask(frame)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = frame.shape[:2]
    bx, by, _br = ball
    best: tuple[float, ContactFoot] | None = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 160 or area > 65000:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 12 or bh < 12:
            continue
        cx, cy = x + bw / 2.0, y + bh / 2.0
        if cy < h * 0.38:
            continue
        signed_distance = cv2.pointPolygonTest(contour, (float(bx), float(by)), True)
        edge_distance = 0.0 if signed_distance >= 0 else abs(float(signed_distance))
        center_distance = math.hypot(cx - bx, cy - by)
        if edge_distance > 220 and center_distance > 280:
            continue
        proximity = math.exp(-(edge_distance * edge_distance) / (2 * 76 * 76))
        center_proximity = math.exp(-(max(0.0, center_distance - 50.0) ** 2) / (2 * 150 * 150))
        area_score = min(1.0, area / 18000.0)
        lower_score = min(1.0, max(0.0, (cy - h * 0.38) / max(1.0, h * 0.42)))
        score = proximity * 0.62 + center_proximity * 0.18 + area_score * 0.12 + lower_score * 0.08
        if edge_distance <= 8:
            score += 0.08
        if x <= 2 or x + bw >= w - 2:
            score *= 0.90
        side_x = limb_side_anchor_x(mask, contour)
        confidence = min(1.0, score)
        if best is None or score > best[0]:
            best = (score, ContactFoot(float(cx), float(cy), float(side_x), float(confidence), float(edge_distance)))
    return None if best is None else best[1]


def limb_side_anchor_x(mask: np.ndarray, contour: np.ndarray) -> float:
    """Estimate which body side owns a foot contour from its lower limb root."""

    x, y, bw, bh = cv2.boundingRect(contour)
    cx = x + bw / 2.0
    if bh <= 0:
        return float(cx)

    pts = contour.reshape(-1, 2)
    bottom_y = float(np.max(pts[:, 1]))
    lower_y = max(float(y) + bh * 0.60, bottom_y - 56.0)

    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)
    yy, xx = np.nonzero(contour_mask)
    lower_x = xx[yy >= lower_y]
    if len(lower_x) >= 18:
        return float(np.median(lower_x))

    lower_pts = pts[pts[:, 1] >= lower_y]
    if len(lower_pts):
        return float(np.median(lower_pts[:, 0]))
    return float(cx)


def contact_side_from_geometry(
    foot_x: float | None,
    ball_x: float | None,
    reliable: bool,
    plausible: bool = False,
) -> str:
    if foot_x is None:
        return "unknown"
    if not reliable and not plausible:
        return "unknown"

    center_x = OUT_SIZE[0] / 2
    foot_delta = float(foot_x) - center_x
    ball_delta = 0.0 if ball_x is None else float(ball_x) - center_x

    if reliable:
        if abs(foot_delta) >= 18:
            return "left" if foot_delta < 0 else "right"
        if ball_x is not None and foot_delta * ball_delta > 0 and abs(foot_delta) >= 16 and abs(ball_delta) >= 60:
            return "left" if foot_delta < 0 else "right"
        blended_delta = foot_delta * 0.68 + ball_delta * 0.32
        if foot_delta * ball_delta > 0 and abs(blended_delta) >= 32:
            return "left" if blended_delta < 0 else "right"
        return "unknown"

    if abs(foot_delta) >= 86:
        return "left" if foot_delta < 0 else "right"
    if ball_x is not None and foot_delta * ball_delta > 0 and abs(foot_delta) >= 34 and abs(ball_delta) >= 34:
        return "left" if foot_delta < 0 else "right"
    blended_delta = foot_delta * 0.72 + ball_delta * 0.28
    if abs(blended_delta) >= 58:
        return "left" if blended_delta < 0 else "right"
    return "unknown"


def contact_side_from_foot(foot_x: float | None, reliable: bool) -> str:
    return contact_side_from_geometry(foot_x, None, reliable)


def side_evidence_from_contact(
    foot_x: float | None,
    foot_confidence: float,
    foot_distance: float | None,
    contact_side: str,
    contact_type: str,
    ball_x: float | None = None,
    side_x: float | None = None,
) -> tuple[float, str, str | None]:
    if contact_type == "ground":
        return 0.0, "ground_reset", "not_applicable_ground"
    if foot_x is None:
        return 0.0, "none", "no_limb_detected"

    center_x = OUT_SIZE[0] / 2
    side_reference_x = float(foot_x if side_x is None else side_x)
    side_delta = side_reference_x - center_x
    center_dist = abs(side_reference_x - center_x)
    ball_delta = None if ball_x is None else float(ball_x) - center_x
    close_limb = foot_distance is not None and foot_distance <= 76 and foot_confidence >= 0.20
    plausible_limb = foot_distance is not None and foot_distance <= 150 and foot_confidence >= 0.12
    if contact_side in {"left", "right"}:
        if center_dist < 24 and ball_delta is not None and side_delta * ball_delta > 0 and abs(ball_delta) >= 42:
            confidence = min(
                0.78,
                0.44
                + min(0.22, float(foot_confidence) * 0.18)
                + min(0.10, abs(ball_delta) / 500.0)
                + (0.04 if close_limb else 0.0),
            )
            return float(confidence), "ball_limb_geometry", "ball_side_cue_needs_review"
        confidence = min(
            0.97,
            0.48
            + min(0.30, float(foot_confidence) * 0.26)
            + min(0.12, center_dist / 340.0)
            + (0.07 if close_limb else 0.0),
        )
        source = "limb_contour_geometry" if close_limb else "plausible_limb_geometry"
        return float(confidence), source, None
    if plausible_limb and center_dist < 46:
        return 0.42, "limb_contour_geometry", "limb_near_frame_centerline"
    if plausible_limb:
        return 0.35, "plausible_limb_geometry", "limb_plausible_not_confirmed"
    return 0.0, "limb_detector", "no_reliable_limb_contact"


def classify_contact(
    frame: np.ndarray,
    event_type: str,
    cand: BallCandidate,
) -> tuple[float | None, float | None, float, float | None, float | None, str, str, float]:
    ball = (cand.x, cand.y, cand.radius)
    contact_foot = detect_contact_foot(frame, ball)
    fallback_foot, fallback_conf = detect_foot(frame, ball, None)
    if contact_foot is not None:
        foot_x = contact_foot.x
        foot_y = contact_foot.y
        side_x = contact_foot.side_x
        foot_conf = contact_foot.confidence
        foot_distance = contact_foot.distance
    elif fallback_foot is not None:
        foot_x = float(fallback_foot[0])
        foot_y = float(fallback_foot[1])
        side_x = foot_x
        foot_conf = float(fallback_conf)
        foot_distance = math.hypot(foot_x - cand.x, foot_y - cand.y)
    else:
        foot_x = None
        foot_y = None
        side_x = None
        foot_conf = 0.0
        foot_distance = None

    y_ratio = cand.y / OUT_SIZE[1]
    playable_foot_height = y_ratio >= 0.56
    near_foot = foot_distance is not None and foot_distance <= 76 and foot_conf >= 0.20 and (event_type == "stall" or playable_foot_height)
    plausible_foot = foot_distance is not None and foot_distance <= 150 and foot_conf >= 0.12 and (event_type == "stall" or playable_foot_height)
    side = contact_side_from_geometry(side_x, cand.x, near_foot or (event_type == "stall" and plausible_foot), plausible_foot)
    if event_type == "drop_floor":
        contact_type = "ground"
        contact_conf = min(1.0, 0.45 + max(0.0, y_ratio - 0.72) * 1.4)
    elif event_type == "stall":
        contact_type = "stall"
        contact_conf = min(1.0, 0.50 + (0.35 if near_foot else 0.0) + cand.confidence * 0.2)
    elif near_foot:
        contact_type = "foot"
        contact_conf = min(1.0, 0.45 + foot_conf * 0.35 + cand.confidence * 0.30)
    elif plausible_foot:
        contact_type = "foot_candidate"
        contact_conf = min(1.0, 0.30 + foot_conf * 0.25 + cand.confidence * 0.25)
    elif y_ratio < 0.60:
        contact_type = "knee_candidate"
        contact_conf = min(0.75, 0.25 + cand.confidence * 0.45)
    elif y_ratio > 0.88 and (foot_distance is None or foot_distance > 250):
        contact_type = "ground_candidate"
        contact_conf = min(0.82, 0.25 + cand.confidence * 0.35)
    else:
        contact_type = "unknown_contact"
        contact_conf = min(0.62, 0.18 + cand.confidence * 0.35)

    return (
        None if foot_x is None else float(foot_x),
        None if foot_y is None else float(foot_y),
        float(foot_conf),
        None if foot_distance is None else float(foot_distance),
        None if side_x is None else float(side_x),
        side,
        contact_type,
        float(contact_conf),
    )


def floor_context_scores(frame: np.ndarray, x: float, y: float, radius: int = 76) -> tuple[float, float]:
    h, w = frame.shape[:2]
    left = max(0, int(round(x - radius)))
    right = min(w, int(round(x + radius)))
    top = max(0, int(round(y - radius)))
    bottom = min(h, int(round(y + radius)))
    if right <= left or bottom <= top:
        return 0.0, 0.0
    patch = frame[top:bottom, left:right]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    green = (hue >= 28) & (hue <= 92) & (sat >= 22) & (val >= 35)
    tan = (hue >= 8) & (hue <= 34) & (sat >= 18) & (val >= 45)
    dark_ground = (hue >= 18) & (hue <= 105) & (sat >= 12) & (val >= 18) & (val <= 135)
    sky = ((hue >= 88) & (hue <= 128) & (sat >= 25) & (val >= 75)) | ((sat <= 28) & (val >= 170))
    total = max(1, patch.shape[0] * patch.shape[1])
    ground_ratio = float(np.count_nonzero(green | tan | dark_ground) / total)
    sky_ratio = float(np.count_nonzero(sky) / total)
    return ground_ratio, sky_ratio


def has_airborne_red_blob(frame: np.ndarray, x: float, y: float) -> bool:
    mask = red_ball_mask(frame)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 28 or area > 5200:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius < 3.0 or radius > 62:
            continue
        dist = math.hypot(float(cx) - x, float(cy) - y)
        if dist >= 120 and float(cy) <= y - 125:
            return True
    return False


def has_visible_airborne_ball(frame: np.ndarray, x: float, y: float) -> bool:
    ball, conf = detect_ball(frame, None)
    if ball is None or conf < 0.12:
        return False
    bx, by, br = ball
    if br < 5.0:
        return False
    return math.hypot(float(bx) - x, float(by) - y) >= 110 and float(by) <= y - 120


def refine_ball_for_event(
    cap: cv2.VideoCapture,
    fps: float,
    detections: list[Detection],
    event: dict[str, Any],
) -> BallFix:
    t = event_time(event)
    det = nearest_detection(detections, t)
    predicted: tuple[float, float] | None = None
    if event.get("x") is not None and event.get("y") is not None:
        predicted = (float(event["x"]), float(event["y"]))
    elif det and det.x is not None and det.y is not None:
        predicted = (float(det.x), float(det.y))

    offsets = (-0.07, -0.035, 0.0, 0.04, 0.08)
    if event.get("type") in {"stall", "drop_floor"}:
        offsets = (-0.04, 0.0, 0.04)

    best: tuple[float, BallCandidate, np.ndarray, float, tuple[Any, ...]] | None = None
    for offset in offsets:
        frame_time = max(0.0, t + offset)
        frame = read_resized_frame(cap, fps, frame_time)
        if frame is None:
            continue
        for cand in ball_candidates(frame, predicted):
            correction = 0.0 if predicted is None else math.hypot(cand.x - predicted[0], cand.y - predicted[1])
            foot_x, foot_y, foot_conf, foot_distance, side_x, side, contact_type, contact_conf = classify_contact(frame, str(event.get("type", "touch")), cand)
            source_bonus = {
                "red_global": 0.24,
                "red_track_near": 0.22,
                "local_red_snap": 0.28,
                "color_blob": 0.10,
                "track_fallback": -0.08,
            }.get(cand.source, 0.0)
            track_bonus = 0.0 if predicted is None else 0.15 * math.exp(-(correction * correction) / (2 * 170 * 170))
            contact_bonus = 0.15 if contact_type in {"foot", "stall"} else 0.08 if contact_type in {"foot_candidate", "knee_candidate"} else 0.0
            score = cand.confidence + source_bonus + track_bonus + contact_bonus
            if event.get("type") == "drop_floor" and cand.y / OUT_SIZE[1] > 0.78:
                score += 0.08
            if best is None or score > best[0]:
                best = (score, cand, frame, frame_time, (foot_x, foot_y, foot_conf, foot_distance, side_x, side, contact_type, contact_conf))

    if best is None:
        return BallFix(
            None,
            None,
            0.0,
            0.0,
            "missing",
            t,
            None,
            None,
            None,
            0.0,
            None,
            "unknown",
            "unknown_contact",
            0.0,
            0.0,
            "none",
            "missing_ball",
            "low",
        )

    _score, cand, _frame, frame_time, contact = best
    correction = None if predicted is None else math.hypot(cand.x - predicted[0], cand.y - predicted[1])
    foot_x, foot_y, foot_conf, foot_distance, side_x, side, contact_type, contact_conf = contact
    side_conf, side_source, side_uncertainty = side_evidence_from_contact(foot_x, foot_conf, foot_distance, side, contact_type, cand.x, side_x)
    if cand.confidence >= 0.70 and (correction is None or correction <= 120 or cand.source.startswith("red")):
        accuracy = "high"
    elif cand.confidence >= 0.46:
        accuracy = "medium"
    else:
        accuracy = "low"
    return BallFix(
        cand.x,
        cand.y,
        cand.radius,
        cand.confidence,
        cand.source,
        frame_time,
        correction,
        foot_x,
        foot_y,
        foot_conf,
        foot_distance,
        side,
        contact_type,
        contact_conf,
        side_conf,
        side_source,
        side_uncertainty,
        accuracy,
    )


def enrich_event_with_ball(event: dict[str, Any], fix: BallFix) -> dict[str, Any]:
    enriched = dict(event)
    enriched.update(
        {
            "qa_ball_x": None if fix.x is None else round(fix.x, 2),
            "qa_ball_y": None if fix.y is None else round(fix.y, 2),
            "qa_ball_radius": round(fix.radius, 2),
            "qa_ball_confidence": round(fix.confidence, 3),
            "qa_ball_source": fix.source,
            "qa_frame_time_sec": round(fix.frame_time_sec, 3),
            "qa_ball_correction_px": None if fix.correction_px is None else round(fix.correction_px, 2),
            "qa_ball_accuracy": fix.ball_accuracy,
            "contact_side": fix.contact_side,
            "contact_type": fix.contact_type,
            "contact_confidence": round(fix.contact_confidence, 3),
            "side_confidence": round(fix.side_confidence, 3),
            "side_source": fix.side_source,
            "side_uncertainty_reason": fix.side_uncertainty_reason,
            "foot_x": None if fix.foot_x is None else round(fix.foot_x, 2),
            "foot_y": None if fix.foot_y is None else round(fix.foot_y, 2),
            "foot_confidence": round(fix.foot_confidence, 3),
            "foot_distance": None if fix.foot_distance is None else round(fix.foot_distance, 2),
        }
    )
    if event.get("x") is None and fix.x is not None:
        enriched["x"] = round(fix.x, 2)
        enriched["y"] = round(fix.y, 2)
    return enriched


def merge_stall_events(existing: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = list(existing)
    for cand in sorted(candidates, key=event_time):
        c_start = float(cand["time_sec"])
        c_end = float(cand.get("end_sec", c_start))
        duplicate = False
        for item in merged:
            start = float(item["time_sec"])
            end = float(item.get("end_sec", start + item.get("duration_sec", 0.0)))
            overlap = max(0.0, min(c_end, end) - max(c_start, start))
            if abs(c_start - start) <= 0.45 or overlap > 0.05:
                duplicate = True
                break
        if not duplicate:
            merged.append(cand)
    return sorted(merged, key=event_time)


def detect_extra_stalls(video: Path, detections: list[Detection]) -> list[dict[str, Any]]:
    speeds = speed_series(detections, OUT_SIZE)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    runs: list[list[dict[str, float]]] = []
    current: list[dict[str, float]] = []
    for idx in range(0, len(detections), 3):
        det = detections[idx]
        if det.x is None or det.y is None or det.score < 0.38:
            if current:
                runs.append(current)
                current = []
            continue
        if det.y / OUT_SIZE[1] < 0.46 or speeds[idx] > 520:
            if current:
                runs.append(current)
                current = []
            continue
        frame = read_resized_frame(cap, fps, det.time_sec)
        if frame is None:
            continue
        predicted = (float(det.x), float(det.y))
        cand = local_red_snap(
            frame,
            BallCandidate(predicted[0], predicted[1], 18.0, "stall_track_window", min(0.90, 0.25 + float(det.score) * 0.75)),
            search_radius=56.0,
        )
        foot_x, foot_y, foot_conf, foot_distance, side_x, side, contact_type, contact_conf = classify_contact(frame, "stall", cand)
        del foot_y
        side_conf, side_source, side_uncertainty = side_evidence_from_contact(foot_x, foot_conf, foot_distance, side, contact_type, cand.x, side_x)
        near_limb = foot_distance is not None and foot_distance <= 95 and foot_conf >= 0.06
        stillish = speeds[idx] <= 150 or (speeds[idx] <= 230 and near_limb and cand.confidence >= 0.64)
        if near_limb and stillish and cand.y / OUT_SIZE[1] >= 0.48:
            current.append(
                {
                    "time_sec": float(det.time_sec),
                    "x": float(cand.x),
                    "y": float(cand.y),
                    "radius": float(cand.radius),
                    "source": cand.source,
                    "ball_confidence": float(cand.confidence),
                    "foot_distance": float(foot_distance or 999),
                    "foot_confidence": float(foot_conf),
                    "contact_confidence": float(contact_conf),
                    "contact_side": side,
                    "contact_type": contact_type,
                    "side_confidence": float(side_conf),
                    "side_source": side_source,
                    "side_uncertainty_reason": side_uncertainty or "",
                    "speed": float(speeds[idx]),
                }
            )
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    cap.release()

    stalls: list[dict[str, Any]] = []
    for run in runs:
        start = run[0]["time_sec"]
        end = run[-1]["time_sec"]
        duration = end - start
        if duration < 0.28 or len(run) < 3:
            continue
        confidence = float(np.mean([item["contact_confidence"] for item in run]))
        mean_foot_distance = float(np.mean([item["foot_distance"] for item in run]))
        max_motion = 0.0
        for idx, item in enumerate(run):
            for other in run[idx + 1 :]:
                max_motion = max(max_motion, math.hypot(item["x"] - other["x"], item["y"] - other["y"]))
        if confidence < 0.78 or mean_foot_distance > 78 or max_motion > 48:
            continue
        side_counts: dict[str, int] = {}
        for item in run:
            side_counts[item["contact_side"]] = side_counts.get(item["contact_side"], 0) + 1
        side = max(side_counts, key=side_counts.get) if side_counts else "unknown"
        side_confidence = float(np.mean([float(item["side_confidence"]) for item in run]))
        side_sources = [str(item["side_source"]) for item in run if item.get("side_source")]
        side_source = max(set(side_sources), key=side_sources.count) if side_sources else "unknown"
        uncertainty_reasons = [str(item["side_uncertainty_reason"]) for item in run if item.get("side_uncertainty_reason")]
        side_uncertainty = max(set(uncertainty_reasons), key=uncertainty_reasons.count) if uncertainty_reasons else None
        mean_ball_conf = float(np.mean([item["ball_confidence"] for item in run]))
        stalls.append(
            {
                "type": "stall",
                "time_sec": round(start, 3),
                "end_sec": round(end, 3),
                "duration_sec": round(duration, 3),
                "confidence": round(confidence, 3),
                "label": "qa detected stall",
                "note": "low ball speed near foot/limb",
                "x": round(float(np.mean([item["x"] for item in run])), 2),
                "y": round(float(np.mean([item["y"] for item in run])), 2),
                "qa_ball_x": round(float(np.mean([item["x"] for item in run])), 2),
                "qa_ball_y": round(float(np.mean([item["y"] for item in run])), 2),
                "qa_ball_radius": round(float(np.mean([item["radius"] for item in run])), 2),
                "qa_ball_source": "stall_track_window",
                "qa_frame_time_sec": round(float((start + end) / 2.0), 3),
                "qa_ball_accuracy": "high" if mean_ball_conf >= 0.70 else "medium" if mean_ball_conf >= 0.46 else "low",
                "contact_side": side,
                "contact_type": "stall",
                "side_confidence": round(side_confidence, 3),
                "side_source": side_source,
                "side_uncertainty_reason": side_uncertainty,
                "qa_ball_confidence": round(mean_ball_conf, 3),
                "foot_distance": round(mean_foot_distance, 2),
                "stall_max_motion_px": round(float(max_motion), 2),
            }
        )
    return stalls


def detect_event_centered_stalls(video: Path, detections: list[Detection], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find short held contacts around detected touches.

    Track-only stall windows miss brief toe catches when the ball is visually
    pinned to the shoe but the tracker jitters. This second pass samples around
    each touch and requires several nearby frames to keep the ball close to the
    same foot/limb.
    """

    touch_events = [item for item in events if item.get("type") == "touch"]
    if not touch_events:
        return []
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stalls: list[dict[str, Any]] = []
    for event in touch_events:
        t = event_time(event)
        good: list[dict[str, float | str]] = []
        predicted: tuple[float, float] | None = None
        if event.get("qa_ball_x") is not None and event.get("qa_ball_y") is not None:
            predicted = (float(event["qa_ball_x"]), float(event["qa_ball_y"]))
        elif event.get("x") is not None and event.get("y") is not None:
            predicted = (float(event["x"]), float(event["y"]))
        else:
            det = nearest_detection(detections, t)
            if det and det.x is not None and det.y is not None:
                predicted = (float(det.x), float(det.y))
        for offset in (-0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12, 0.16, 0.20):
            sample_t = max(0.0, t + offset)
            frame = read_resized_frame(cap, fps, sample_t)
            if frame is None:
                continue
            best_sample: tuple[float, BallCandidate, tuple[Any, ...]] | None = None
            for cand in ball_candidates(frame, predicted):
                foot_x, foot_y, foot_conf, foot_distance, side_x, side, contact_type, contact_conf = classify_contact(frame, "stall", cand)
                del foot_y
                side_conf, side_source, side_uncertainty = side_evidence_from_contact(foot_x, foot_conf, foot_distance, side, contact_type, cand.x, side_x)
                correction = 0.0 if predicted is None else math.hypot(cand.x - predicted[0], cand.y - predicted[1])
                score = cand.confidence + 0.16 * math.exp(-(correction * correction) / (2 * 150 * 150)) + contact_conf * 0.24
                if best_sample is None or score > best_sample[0]:
                    best_sample = (score, cand, (foot_conf, foot_distance, side, contact_type, contact_conf, side_conf, side_source, side_uncertainty))
            if best_sample is None:
                continue
            _score, cand, contact = best_sample
            foot_conf, foot_distance, side, contact_type, contact_conf, side_conf, side_source, side_uncertainty = contact
            foot_distance = foot_distance if foot_distance is not None else 999.0
            close = foot_distance <= 82 and foot_conf >= 0.06
            playable_height = cand.y / OUT_SIZE[1] >= 0.56
            if cand.confidence >= 0.46 and close and playable_height and contact_type in {"foot", "foot_candidate", "stall"}:
                good.append(
                    {
                        "time_sec": sample_t,
                        "x": float(cand.x),
                        "y": float(cand.y),
                        "radius": float(cand.radius),
                        "source": cand.source,
                        "confidence": float(contact_conf),
                        "ball_confidence": float(cand.confidence),
                        "foot_distance": foot_distance,
                        "side": side,
                        "side_confidence": float(side_conf),
                        "side_source": str(side_source),
                        "side_uncertainty_reason": str(side_uncertainty or ""),
                    }
                )
        if len(good) < 5:
            continue
        span = float(good[-1]["time_sec"]) - float(good[0]["time_sec"])
        if span < 0.18:
            continue
        points = [(float(item["x"]), float(item["y"])) for item in good]
        max_motion = 0.0
        for idx, a in enumerate(points):
            for b in points[idx + 1 :]:
                max_motion = max(max_motion, math.hypot(a[0] - b[0], a[1] - b[1]))
        if max_motion > 42:
            continue
        y_std = float(np.std([float(item["y"]) for item in good]))
        mean_foot_distance = float(np.mean([float(item["foot_distance"]) for item in good]))
        if y_std > 22 or mean_foot_distance > 72:
            continue
        side_counts: dict[str, int] = {}
        for item in good:
            side = str(item["side"])
            side_counts[side] = side_counts.get(side, 0) + 1
        side = max(side_counts, key=side_counts.get) if side_counts else str(event.get("contact_side", "unknown"))
        side_confidence = float(np.mean([float(item["side_confidence"]) for item in good]))
        side_sources = [str(item["side_source"]) for item in good if item.get("side_source")]
        side_source = max(set(side_sources), key=side_sources.count) if side_sources else "unknown"
        uncertainty_reasons = [str(item["side_uncertainty_reason"]) for item in good if item.get("side_uncertainty_reason")]
        side_uncertainty = max(set(uncertainty_reasons), key=uncertainty_reasons.count) if uncertainty_reasons else None
        confidence = float(np.mean([float(item["confidence"]) for item in good]))
        mean_ball_conf = float(np.mean([float(item["ball_confidence"]) for item in good]))
        mean_x = float(np.mean([float(item["x"]) for item in good]))
        mean_y = float(np.mean([float(item["y"]) for item in good]))
        mean_radius = float(np.mean([float(item["radius"]) for item in good]))
        stalls.append(
            {
                "type": "stall",
                "time_sec": round(float(good[0]["time_sec"]), 3),
                "end_sec": round(float(good[-1]["time_sec"]), 3),
                "duration_sec": round(span, 3),
                "confidence": round(confidence, 3),
                "label": "qa event-centered stall",
                "note": "ball remained close to foot across adjacent frames",
                "x": round(mean_x, 2),
                "y": round(mean_y, 2),
                "qa_ball_x": round(mean_x, 2),
                "qa_ball_y": round(mean_y, 2),
                "qa_ball_radius": round(mean_radius, 2),
                "qa_ball_source": "event_centered_stall",
                "qa_frame_time_sec": round(float(np.mean([float(item["time_sec"]) for item in good])), 3),
                "qa_ball_accuracy": "high" if mean_ball_conf >= 0.70 else "medium" if mean_ball_conf >= 0.46 else "low",
                "contact_side": side,
                "contact_type": "stall",
                "side_confidence": round(side_confidence, 3),
                "side_source": side_source,
                "side_uncertainty_reason": side_uncertainty,
                "qa_ball_confidence": round(mean_ball_conf, 3),
                "foot_distance": round(mean_foot_distance, 2),
                "stall_max_motion_px": round(float(max_motion), 2),
                "stall_y_std_px": round(float(y_std), 2),
            }
        )
    cap.release()
    return stalls


def recover_missing_touches(video: Path, detections: list[Detection], existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing_times = [event_time(item) for item in existing if item.get("type") in {"touch", "drop_floor", "stall"}]
    try:
        peaks: list[AudioPeak] = audio_peaks_for_video(video, min_z=6.0, min_gap_sec=0.20)
    except Exception:
        return []
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    recovered: list[dict[str, Any]] = []
    for peak in peaks:
        if peak.z_score < 14.0 or any(abs(peak.time_sec - t) <= 0.26 for t in existing_times):
            continue
        dummy = {"type": "touch", "time_sec": peak.time_sec, "audio_z": peak.z_score}
        fix = refine_ball_for_event(cap, fps, detections, dummy)
        if fix.x is None or fix.confidence < 0.46:
            continue
        if fix.contact_type in {"ground", "ground_candidate"}:
            continue
        floorish_far_contact = (
            fix.y is not None
            and fix.y / OUT_SIZE[1] >= 0.93
            and (fix.foot_distance is None or fix.foot_distance >= 170)
        )
        if floorish_far_contact:
            continue
        if fix.contact_type == "unknown_contact" and peak.z_score < 24.0:
            continue
        if fix.contact_type not in {"foot", "foot_candidate", "knee_candidate", "stall"} and fix.contact_confidence < 0.52:
            continue
        event = {
            "type": "touch",
            "time_sec": round(float(peak.time_sec), 3),
            "label": "qa recovered audio+visual contact",
            "confidence": round(min(1.0, 0.35 + min(1.0, peak.z_score / 35.0) * 0.35 + fix.contact_confidence * 0.30), 3),
            "audio_z": round(float(peak.z_score), 2),
            "visual_score": 0.0,
            "motion_score": 0.0,
            "note": "not in trained event list; recovered by QA pass",
        }
        recovered.append(enrich_event_with_ball(event, fix))
        existing_times.append(float(peak.time_sec))
    cap.release()
    return recovered


def should_reclassify_touch_as_floor(event: dict[str, Any]) -> bool:
    if event.get("type") != "touch" or event.get("qa_ball_y") is None:
        return False
    y_ratio = float(event["qa_ball_y"]) / OUT_SIZE[1]
    foot_distance = event.get("foot_distance")
    foot_distance_f = None if foot_distance is None else float(foot_distance)
    contact_type = str(event.get("contact_type") or "")
    contact_conf = float(event.get("contact_confidence") or 0.0)
    audio_z = float(event.get("audio_z") or 0.0)
    correction = float(event.get("qa_ball_correction_px") or 0.0)
    far_from_foot = foot_distance_f is None or foot_distance_f >= 240
    weak_contact = contact_conf < 0.58 or contact_type in {"ground_candidate", "unknown_contact"}
    if correction >= 120 and event.get("qa_ball_accuracy") != "high":
        return False
    return (
        (contact_type == "ground_candidate" and y_ratio >= 0.955 and far_from_foot)
        or (y_ratio >= 0.975 and far_from_foot and weak_contact)
        or (y_ratio >= 0.965 and far_from_foot and contact_type == "unknown_contact" and audio_z >= 24.0)
    )


def reclassify_floor_touches(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    reclassified: list[dict[str, Any]] = []
    count = 0
    for event in events:
        if should_reclassify_touch_as_floor(event):
            item = dict(event)
            item["type"] = "drop_floor"
            item["label"] = "qa reclassified floor reset"
            item["note"] = "qa_reclassified_touch_as_floor_reset"
            item["contact_type"] = "ground"
            item["contact_confidence"] = round(max(float(item.get("contact_confidence") or 0.0), 0.68), 3)
            item["drop_source"] = "reclassified_touch"
            reclassified.append(item)
            count += 1
        else:
            reclassified.append(event)
    return reclassified, count


def touch_suppression_reason(event: dict[str, Any]) -> str | None:
    if event.get("type") != "touch":
        return None
    contact_type = str(event.get("contact_type") or "")
    contact_conf = float(event.get("contact_confidence") or 0.0)
    foot_distance = event.get("foot_distance")
    foot_distance_f = 999.0 if foot_distance is None else float(foot_distance)
    y = event.get("qa_ball_y", event.get("y"))
    y_ratio = 0.0 if y is None else float(y) / OUT_SIZE[1]
    correction = float(event.get("qa_ball_correction_px") or 0.0)
    audio_z = float(event.get("audio_z") or 0.0)
    accuracy = str(event.get("qa_ball_accuracy") or "")
    ball_radius = float(event.get("qa_ball_radius") or 0.0)
    ball_source = str(event.get("qa_ball_source") or "")
    foot_contact_score = float(event.get("foot_contact_score") or 0.0)

    if contact_type == "ground_candidate":
        return "ground_candidate_not_touch"
    if contact_type == "knee_candidate" and (contact_conf < 0.72 or y_ratio < 0.55):
        return "weak_knee_candidate"
    if ball_source == "local_red_snap" and ball_radius >= 55 and audio_z < 18:
        return "large_skin_like_ball_snap_low_audio"
    if contact_type == "foot_candidate" and contact_conf < 0.70 and (
        correction >= 150 or foot_contact_score < 0.10 or ball_radius >= 48
    ):
        return "weak_foot_candidate_visual_mismatch"
    if contact_type == "unknown_contact" and not (contact_conf >= 0.82 and foot_distance_f <= 180 and audio_z >= 18):
        return "unknown_contact_not_supported"
    if foot_distance_f > 235 and not (contact_type == "foot" and contact_conf >= 0.92 and accuracy == "high"):
        return "too_far_from_limb"
    if y_ratio >= 0.94 and foot_distance_f >= 180 and contact_type != "foot":
        return "floor_risk_touch_far_from_foot"
    if y_ratio >= 0.975 and contact_type in {"foot_candidate", "unknown_contact"} and contact_conf < 0.75:
        return "near_floor_weak_contact"
    if correction >= 450 and accuracy != "high" and foot_distance_f > 125:
        return "huge_ball_correction_weak_contact"
    if audio_z < 8 and contact_type != "foot" and contact_conf < 0.78:
        return "weak_audio_and_nonfoot_contact"
    return None


def filter_touch_events(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for event in sorted(events, key=event_time):
        reason = touch_suppression_reason(event)
        if reason is None:
            kept.append(event)
            continue
        item = dict(event)
        item["qa_suppressed_reason"] = reason
        suppressed.append(item)

    deduped: list[dict[str, Any]] = []
    for event in kept:
        if event.get("type") != "touch" or not deduped:
            deduped.append(event)
            continue
        prev = deduped[-1]
        if prev.get("type") != "touch" or event_time(event) - event_time(prev) > 0.16:
            deduped.append(event)
            continue
        prev_score = float(prev.get("contact_confidence", prev.get("confidence", 0.0)) or 0.0) + min(
            0.20, float(prev.get("audio_z") or 0.0) / 180.0
        )
        event_score = float(event.get("contact_confidence", event.get("confidence", 0.0)) or 0.0) + min(
            0.20, float(event.get("audio_z") or 0.0) / 180.0
        )
        dropped = dict(prev if event_score > prev_score else event)
        dropped["qa_suppressed_reason"] = "duplicate_touch_within_160ms"
        suppressed.append(dropped)
        if event_score > prev_score:
            deduped[-1] = event
    return deduped, suppressed


def quality_suppression_reason(event: dict[str, Any]) -> str | None:
    kind = str(event.get("type") or "")
    text = f"{event.get('label', '')} {event.get('note', '')} {event.get('drop_source', '')}".lower()
    if kind == "touch":
        confidence = float(event.get("confidence") or 0.0)
        correction = float(event.get("qa_ball_correction_px") or 0.0)
        audio_z = float(event.get("audio_z") or 0.0)
        contact_type = str(event.get("contact_type") or "")
        contact_confidence = float(event.get("contact_confidence") or 0.0)
        if confidence < 0.58:
            if not (audio_z >= 30 and contact_type == "foot" and contact_confidence >= 0.88):
                return "low_confidence_touch_candidate"
        if correction >= 220 and "track-context" in text:
            return "audio_track_huge_correction_touch"
    elif kind == "drop_floor":
        foot_distance = event.get("foot_distance")
        foot_distance_f = 0.0 if foot_distance is None else float(foot_distance)
        y = event.get("qa_ball_y", event.get("y"))
        y_ratio = 0.0 if y is None else float(y) / OUT_SIZE[1]
        if "track-context" in text:
            return "audio_track_context_not_floor_reset"
        if event_time(event) < 1.6:
            return "setup_phase_floor_reset"
        if foot_distance_f >= 450:
            return "floor_reset_too_far_from_limb_context"
        if y_ratio < 0.90:
            return "floor_reset_not_near_floor"
    elif kind == "stall":
        y = event.get("qa_ball_y", event.get("y"))
        y_ratio = 0.0 if y is None else float(y) / OUT_SIZE[1]
        if event_time(event) < 1.3:
            return "setup_phase_stall_candidate"
        if y_ratio >= 0.89:
            return "near_floor_stall_candidate"
    elif kind == "around_the_world":
        if "reviewed_seed_label" not in text:
            return "automatic_atw_candidate_needs_review"
    return None


def filter_quality_events(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    last_drop_time: float | None = None
    contact_since_last_drop = True
    for event in sorted(events, key=event_time):
        reason = quality_suppression_reason(event)
        kind = str(event.get("type") or "")
        if reason is None and kind == "drop_floor" and last_drop_time is not None and not contact_since_last_drop:
            reason = "duplicate_floor_reset_without_intervening_contact"
        if reason is None:
            kept.append(event)
            if kind == "drop_floor":
                last_drop_time = event_time(event)
                contact_since_last_drop = False
            elif kind in {"touch", "stall"}:
                contact_since_last_drop = True
            continue
        item = dict(event)
        item["qa_suppressed_reason"] = reason
        suppressed.append(item)
    return kept, suppressed


def annotate_precision_gate(event: dict[str, Any]) -> dict[str, Any]:
    item = dict(event)
    decision = evaluate_precision_gate(item)
    item["precision_gate_hard_veto"] = bool(decision.hard_veto)
    item["precision_gate_soft_flag"] = bool(decision.soft_flag)
    item["precision_gate_score"] = round(float(decision.score), 4)
    item["precision_gate_reasons"] = ";".join(decision.reasons)
    item["precision_gate_hard_reasons"] = ";".join(decision.hard_reasons)
    item["precision_gate_soft_reasons"] = ";".join(decision.soft_reasons)
    disagreement = decision.features.get("tracker_disagreement_px")
    item["precision_gate_tracker_disagreement_px"] = None if disagreement is None else round(float(disagreement), 4)
    y_ratio_value = decision.features.get("y_ratio")
    item["precision_gate_y_ratio"] = None if y_ratio_value is None else round(float(y_ratio_value), 4)
    return item


def filter_precision_gate_events(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for event in sorted(events, key=event_time):
        item = annotate_precision_gate(event)
        if not item.get("precision_gate_hard_veto"):
            kept.append(item)
            continue
        item["qa_suppressed_reason"] = f"precision_gate:{item.get('precision_gate_hard_reasons') or item.get('precision_gate_reasons')}"
        suppressed.append(item)
    return kept, suppressed


def floor_candidate_at(
    cap: cv2.VideoCapture,
    fps: float,
    detections: list[Detection],
    speeds: np.ndarray,
    time_sec: float,
    reason: str,
) -> FloorResetCandidate | None:
    det = nearest_detection(detections, time_sec)
    predicted: tuple[float, float] | None = None
    det_speed = 0.0
    if det and det.x is not None and det.y is not None and det.score >= 0.18:
        predicted = (float(det.x), float(det.y))
        det_idx = min(range(len(detections)), key=lambda item: abs(detections[item].time_sec - time_sec)) if detections else 0
        det_speed = float(speeds[det_idx]) if det_idx < len(speeds) else 0.0
    frame = read_resized_frame(cap, fps, time_sec)
    if frame is None:
        return None

    best: FloorResetCandidate | None = None
    for cand in ball_candidates(frame, predicted):
        y_ratio = cand.y / OUT_SIZE[1]
        if y_ratio < 0.88 or cand.confidence < 0.38:
            continue
        foot_x, foot_y, foot_conf, foot_distance, _side_x, side, contact_type, _contact_conf = classify_contact(frame, "touch", cand)
        ground_ratio, sky_ratio = floor_context_scores(frame, cand.x, cand.y)
        airborne_red = has_airborne_red_blob(frame, cand.x, cand.y)
        visible_airborne_ball = has_visible_airborne_ball(frame, cand.x, cand.y)
        if visible_airborne_ball:
            continue
        if foot_distance is not None and foot_distance <= 70 and foot_conf >= 0.035:
            continue
        if reason == "visual_floor_after_last_contact" and foot_distance is not None and foot_distance <= 85 and foot_conf >= 0.20:
            continue
        if ground_ratio < 0.16:
            continue
        if sky_ratio >= 0.42 and ground_ratio < 0.30:
            continue
        deep_floor = y_ratio >= 0.96 and ground_ratio >= 0.26 and sky_ratio < 0.30
        if airborne_red and not deep_floor:
            continue
        red_floor_ok = (
            cand.source in {"red_global", "red_track_near", "local_red_snap"}
            and ground_ratio >= 0.18
            and sky_ratio < 0.44
        )
        color_blob_floor_ok = (
            cand.source == "color_blob"
            and y_ratio >= 0.95
            and (ground_ratio >= 0.32 or deep_floor)
            and sky_ratio < 0.34
            and (not airborne_red or deep_floor)
        )
        track_floor_ok = (
            cand.source == "track_fallback"
            and y_ratio >= 0.94
            and ground_ratio >= 0.30
            and sky_ratio < 0.28
            and det_speed <= 220
            and not airborne_red
        )
        if not (red_floor_ok or color_blob_floor_ok or track_floor_ok):
            continue
        near_limb = foot_distance is not None and foot_distance <= 125 and foot_conf >= 0.08 and contact_type in {"foot", "foot_candidate", "stall"}
        floor_depth = min(1.0, max(0.0, (y_ratio - 0.88) / 0.10))
        stationary_bonus = 0.10 if det_speed <= 260 else 0.0
        far_bonus = 0.16 if not near_limb else -0.14
        source_bonus = 0.08 if cand.source in {"red_global", "red_track_near", "local_red_snap"} else 0.02 if color_blob_floor_ok else -0.04
        score = (
            cand.confidence * 0.34
            + floor_depth * 0.38
            + far_bonus
            + stationary_bonus
            + source_bonus
            + min(0.14, ground_ratio * 0.18)
            - min(0.22, sky_ratio * 0.22)
        )
        if airborne_red:
            score -= 0.12 if deep_floor else 0.35
        if y_ratio >= 0.965 and foot_distance is not None and foot_distance >= 90:
            score += 0.06
        if score < 0.52:
            continue
        candidate = FloorResetCandidate(
            time_sec=time_sec,
            x=float(cand.x),
            y=float(cand.y),
            radius=float(cand.radius),
            confidence=float(cand.confidence),
            source=cand.source,
            frame_time_sec=time_sec,
            foot_x=None if foot_x is None else float(foot_x),
            foot_y=None if foot_y is None else float(foot_y),
            foot_confidence=float(foot_conf),
            foot_distance=None if foot_distance is None else float(foot_distance),
            contact_side=side,
            score=float(score),
            speed=det_speed,
            reason=reason,
            ground_ratio=float(ground_ratio),
            sky_ratio=float(sky_ratio),
            airborne_red_blob=bool(airborne_red),
        )
        if best is None or candidate.score > best.score:
            best = candidate
    return best


def floor_event_from_candidate(candidate: FloorResetCandidate) -> dict[str, Any]:
    confidence = min(1.0, 0.42 + candidate.score * 0.58)
    accuracy = "high" if candidate.confidence >= 0.70 else "medium" if candidate.confidence >= 0.46 else "low"
    side_conf, side_source, side_uncertainty = side_evidence_from_contact(
        candidate.foot_x,
        candidate.foot_confidence,
        candidate.foot_distance,
        candidate.contact_side,
        "ground",
        candidate.x,
    )
    return {
        "type": "drop_floor",
        "time_sec": round(float(candidate.time_sec), 3),
        "label": "qa hidden floor reset",
        "confidence": round(float(confidence), 3),
        "visual_score": round(float(candidate.score), 3),
        "motion_score": 0.0,
        "x": round(float(candidate.x), 2),
        "y": round(float(candidate.y), 2),
        "qa_ball_x": round(float(candidate.x), 2),
        "qa_ball_y": round(float(candidate.y), 2),
        "qa_ball_radius": round(float(candidate.radius), 2),
        "qa_ball_confidence": round(float(candidate.confidence), 3),
        "qa_ball_source": candidate.source,
        "qa_frame_time_sec": round(float(candidate.frame_time_sec), 3),
        "qa_ball_correction_px": None,
        "qa_ball_accuracy": accuracy,
        "contact_side": candidate.contact_side,
        "contact_type": "ground",
        "contact_confidence": round(float(confidence), 3),
        "side_confidence": round(float(side_conf), 3),
        "side_source": side_source,
        "side_uncertainty_reason": side_uncertainty,
        "foot_x": None if candidate.foot_x is None else round(float(candidate.foot_x), 2),
        "foot_y": None if candidate.foot_y is None else round(float(candidate.foot_y), 2),
        "foot_confidence": round(float(candidate.foot_confidence), 3),
        "foot_distance": None if candidate.foot_distance is None else round(float(candidate.foot_distance), 2),
        "drop_source": candidate.reason,
        "drop_score": round(float(candidate.score), 3),
        "track_speed": round(float(candidate.speed), 2),
        "floor_ground_context": round(float(candidate.ground_ratio), 3),
        "floor_sky_context": round(float(candidate.sky_ratio), 3),
        "floor_airborne_red_blob": bool(candidate.airborne_red_blob),
        "note": "qa_hidden_floor_reset_from_visual_continuity",
    }


def earliest_credible_floor_candidate(candidates: list[FloorResetCandidate], *, min_score: float) -> FloorResetCandidate | None:
    if not candidates:
        return None
    best = max(candidates, key=lambda item: item.score)
    # Reset timing should represent first credible floor contact, not the later
    # frame where the bag is easiest to see resting on the ground.
    strong_enough = max(min_score, best.score - 0.28)
    for candidate in sorted(candidates, key=lambda item: item.time_sec):
        if candidate.score >= strong_enough:
            return candidate
    return best if best.score >= min_score else None


def hidden_drop_sample_times(
    detections: list[Detection],
    speeds: np.ndarray,
    start: float,
    end: float,
    *,
    base_step: float = 0.12,
) -> list[float]:
    sample_times = {round(float(t), 4) for t in np.arange(start, end + 0.001, base_step)}
    for idx, det in enumerate(detections):
        if det.time_sec < start or det.time_sec > end:
            continue
        if det.x is None or det.y is None or det.score < 0.18:
            continue
        speed = float(speeds[idx]) if idx < len(speeds) else 9999.0
        if speed <= 280 and det.y / OUT_SIZE[1] >= 0.70:
            sample_times.add(round(float(det.time_sec), 4))
    return sorted(sample_times)


def detect_hidden_drops(video: Path, detections: list[Detection], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add reset events when the ball visibly hits/rests on the floor."""

    contacts = [
        item
        for item in sorted(events, key=event_time)
        if item.get("type") in {"touch", "stall", "drop_floor"}
    ]
    if not contacts:
        return []
    existing_drop_times = [event_time(item) for item in contacts if item.get("type") == "drop_floor"]
    contact_times = [event_time(item) for item in contacts]
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps if fps else 0.0
    speeds = speed_series(detections, OUT_SIZE)

    def add_if_new(candidate: FloorResetCandidate | None, out: list[dict[str, Any]]) -> None:
        if candidate is None:
            return
        if any(abs(candidate.time_sec - t) <= 0.48 for t in existing_drop_times):
            return
        if any(abs(candidate.time_sec - t) <= 0.22 for t in contact_times):
            return
        out.append(floor_event_from_candidate(candidate))
        existing_drop_times.append(candidate.time_sec)
        contact_times.append(candidate.time_sec)

    hidden: list[dict[str, Any]] = []
    for prev, nxt in zip(contacts, contacts[1:]):
        if prev.get("type") == "drop_floor" or nxt.get("type") == "drop_floor":
            continue
        prev_t = event_time(prev)
        next_t = event_time(nxt)
        gap = next_t - prev_t
        if gap < 1.38:
            continue
        sample_start = prev_t + 0.28
        sample_end = next_t - 0.28
        if sample_end <= sample_start:
            continue
        candidates: list[FloorResetCandidate] = []
        for sample_t in hidden_drop_sample_times(detections, speeds, sample_start, sample_end):
            candidate = floor_candidate_at(cap, fps, detections, speeds, float(sample_t), "visual_floor_between_contacts")
            if candidate is not None:
                candidates.append(candidate)
        best = earliest_credible_floor_candidate(candidates, min_score=0.62)
        if best is not None and best.score >= 0.62:
            add_if_new(best, hidden)
        elif gap >= 2.05:
            reset_t = prev_t + min(gap * 0.55, 1.15)
            det = nearest_detection(detections, reset_t)
            if det and det.x is not None and det.y is not None and float(det.y) / OUT_SIZE[1] >= 0.90:
                frame = read_resized_frame(cap, fps, reset_t)
                ground_ratio = 0.0
                sky_ratio = 1.0
                airborne_red = True
                if frame is not None:
                    ground_ratio, sky_ratio = floor_context_scores(frame, float(det.x), float(det.y))
                    airborne_red = has_airborne_red_blob(frame, float(det.x), float(det.y))
                if ground_ratio < 0.30 or sky_ratio >= 0.30 or airborne_red:
                    continue
                add_if_new(
                    FloorResetCandidate(
                        time_sec=reset_t,
                        x=float(det.x),
                        y=float(det.y),
                        radius=18.0,
                        confidence=max(0.38, float(det.score)),
                        source="track_gap_reset",
                        frame_time_sec=reset_t,
                        foot_x=None,
                        foot_y=None,
                        foot_confidence=0.0,
                        foot_distance=None,
                        contact_side="unknown",
                        score=0.54,
                        speed=0.0,
                        reason="long_gap_reset",
                        ground_ratio=float(ground_ratio),
                        sky_ratio=float(sky_ratio),
                        airborne_red_blob=bool(airborne_red),
                    ),
                    hidden,
                )

    last_non_drop = next((item for item in reversed(contacts) if item.get("type") != "drop_floor"), None)
    if last_non_drop is not None:
        last_t = event_time(last_non_drop)
        sample_end = min(duration, last_t + 3.10)
        candidates = []
        for sample_t in hidden_drop_sample_times(detections, speeds, last_t + 0.30, sample_end):
            candidate = floor_candidate_at(cap, fps, detections, speeds, float(sample_t), "visual_floor_after_last_contact")
            if candidate is not None:
                candidates.append(candidate)
        best = earliest_credible_floor_candidate(candidates, min_score=0.64)
        if best is not None and best.score >= 0.64:
            add_if_new(best, hidden)

    cap.release()
    return sorted(hidden, key=event_time)


def assign_rallies(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contact_types = {"touch", "stall", "drop_floor"}
    ordered = sorted(events, key=event_time)
    rallies: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    last_contact_time: float | None = None

    def append_rally(reason: str, *, next_gap: float | None = None) -> None:
        if not current:
            return
        meta: dict[str, Any] = {"end_reason": reason}
        if next_gap is not None:
            meta["next_contact_gap_sec"] = round(float(next_gap), 3)
        rallies.append((list(current), meta))

    for event in ordered:
        if event.get("type") not in contact_types:
            if current:
                current.append(event)
            continue
        t = event_time(event)
        if current and last_contact_time is not None and t - last_contact_time > 2.25:
            append_rally("gap_without_floor_reset", next_gap=t - last_contact_time)
            current = []
        current.append(event)
        if event.get("type") == "drop_floor":
            append_rally("drop_floor")
            current = []
            last_contact_time = None
        else:
            last_contact_time = t
    if current:
        append_rally("end_of_video")

    rally_summaries: list[dict[str, Any]] = []
    enriched_events: list[dict[str, Any]] = []
    for rally_idx, (rally_events, rally_meta) in enumerate(rallies, start=1):
        contacts = [item for item in rally_events if item.get("type") in contact_types]
        touch_events = [item for item in contacts if item.get("type") == "touch"]
        stall_events = [item for item in contacts if item.get("type") == "stall"]
        trick_events = [item for item in rally_events if item.get("type") == "around_the_world"]
        if not contacts:
            continue
        start = event_time(contacts[0])
        end = max(event_time(item) if item.get("type") != "stall" else float(item.get("end_sec", event_time(item))) for item in contacts)
        duration = max(0.0, end - start)
        prev_touch_time: float | None = None
        airtimes: list[float] = []
        touch_index = 0
        for item in rally_events:
            item = dict(item)
            item["qa_rally_id"] = rally_idx
            item["qa_rally_duration_sec"] = round(duration, 3)
            if item.get("type") == "touch":
                touch_index += 1
                item["qa_touch_index"] = touch_index
                item["qa_rally_touch_count"] = len(touch_events)
                if prev_touch_time is not None:
                    airtime = event_time(item) - prev_touch_time
                    item["airtime_since_prev_sec"] = round(airtime, 3)
                    airtimes.append(airtime)
                else:
                    item["airtime_since_prev_sec"] = None
                prev_touch_time = event_time(item)
            enriched_events.append(item)

        confidences = [
            float(item.get("contact_confidence", item.get("confidence", 0.5)) or 0.5)
            for item in rally_events
            if item.get("type") in {"touch", "stall"}
        ]
        ball_conf = [
            float(item.get("qa_ball_confidence", 0.4) or 0.4)
            for item in rally_events
            if item.get("type") in {"touch", "stall"}
        ]
        avg_conf = float(np.mean(confidences)) if confidences else 0.5
        avg_ball = float(np.mean(ball_conf)) if ball_conf else 0.4
        if airtimes and np.mean(airtimes) > 0:
            consistency = float(max(0.0, 1.0 - min(1.0, float(np.std(airtimes) / max(np.mean(airtimes), 0.01)))))
        else:
            consistency = 0.35
        low_accuracy = sum(1 for item in rally_events if item.get("qa_ball_accuracy") == "low")
        quality = (
            16
            + len(touch_events) * 3.8
            + duration * 0.9
            + len(stall_events) * 7.0
            + len(trick_events) * 9.0
            + consistency * 12.0
            + avg_conf * 10.0
            + avg_ball * 8.0
            - low_accuracy * 4.0
        )
        quality = float(np.clip(quality, 0, 100))
        if quality >= 85:
            grade = "S"
        elif quality >= 72:
            grade = "A"
        elif quality >= 58:
            grade = "B"
        elif quality >= 42:
            grade = "C"
        else:
            grade = "D"
        rally_summaries.append(
            {
                "id": rally_idx,
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "duration_sec": round(duration, 3),
                "touches": len(touch_events),
                "stalls": len(stall_events),
                "around_the_world": len(trick_events),
                "mean_airtime_sec": None if not airtimes else round(float(np.mean(airtimes)), 3),
                "airtime_consistency": round(consistency, 3),
                "quality_score": round(quality, 1),
                "quality_grade": grade,
                "mean_contact_confidence": round(avg_conf, 3),
                "mean_ball_confidence": round(avg_ball, 3),
                "low_ball_accuracy_events": low_accuracy,
                "end_reason": rally_meta.get("end_reason", "unknown"),
                "ended_by_gap_without_floor_reset": rally_meta.get("end_reason") == "gap_without_floor_reset",
                "next_contact_gap_sec": rally_meta.get("next_contact_gap_sec"),
            }
        )
    return enriched_events, rally_summaries


def write_events_csv(events: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "type",
        "qa_rally_id",
        "qa_touch_index",
        "time_sec",
        "end_sec",
        "duration_sec",
        "airtime_since_prev_sec",
        "contact_side",
        "contact_type",
        "contact_confidence",
        "side_confidence",
        "side_source",
        "side_uncertainty_reason",
        "qa_ball_x",
        "qa_ball_y",
        "qa_ball_confidence",
        "qa_ball_radius",
        "qa_ball_source",
        "qa_frame_time_sec",
        "qa_ball_accuracy",
        "qa_ball_correction_px",
        "foot_x",
        "foot_y",
        "foot_confidence",
        "foot_distance",
        "audio_z",
        "confidence",
        "drop_source",
        "drop_score",
        "track_speed",
        "precision_gate_hard_veto",
        "precision_gate_soft_flag",
        "precision_gate_score",
        "precision_gate_reasons",
        "precision_gate_hard_reasons",
        "precision_gate_soft_reasons",
        "precision_gate_tracker_disagreement_px",
        "precision_gate_y_ratio",
        "label",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(events)


def draw_qa_sheet(video: Path, events: list[dict[str, Any]], out_path: Path) -> None:
    contacts = [item for item in events if item.get("type") in {"touch", "stall", "drop_floor"}]
    if not contacts:
        return
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    thumbs: list[np.ndarray] = []
    for item in contacts[:120]:
        t = float(item.get("qa_frame_time_sec", event_time(item)))
        frame = read_resized_frame(cap, fps, t)
        if frame is None:
            continue
        old_x = item.get("x")
        old_y = item.get("y")
        if old_x is not None and old_y is not None:
            cv2.circle(frame, (int(round(float(old_x))), int(round(float(old_y)))), 20, (0, 255, 255), 2, cv2.LINE_AA)
        if item.get("qa_ball_x") is not None and item.get("qa_ball_y") is not None:
            cx = int(round(float(item["qa_ball_x"])))
            cy = int(round(float(item["qa_ball_y"])))
            color = (30, 220, 30) if item.get("qa_ball_accuracy") == "high" else (0, 165, 255) if item.get("qa_ball_accuracy") == "medium" else (0, 0, 255)
            cv2.circle(frame, (cx, cy), 24, color, 3, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 4, color, -1, cv2.LINE_AA)
        label = (
            f"R{item.get('qa_rally_id','?')} {item.get('type')} {event_time(item):.2f}s "
            f"{item.get('contact_side','?')} {item.get('contact_type','?')} "
            f"b{float(item.get('qa_ball_confidence') or 0):.2f}"
        )
        cv2.putText(frame, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 255, 80), 2, cv2.LINE_AA)
        thumbs.append(cv2.resize(frame, (344, 456), interpolation=cv2.INTER_AREA))
    cap.release()
    if not thumbs:
        return
    cols = 5
    rows = math.ceil(len(thumbs) / cols)
    sheet = np.full((rows * 456, cols * 344, 3), 255, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * 344
        y = (idx // cols) * 456
        sheet[y : y + 456, x : x + 344] = thumb
    cv2.imwrite(str(out_path), sheet)


def process_run(run: dict[str, Any], out_root: Path) -> dict[str, Any]:
    video = resolve_video_path(run["video"])
    full_events_path = resolve_existing_path(run["full_events_path"])
    source_dir = full_events_path.parent
    out_dir = out_root / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    source_doc = json.loads(full_events_path.read_text(encoding="utf-8"))
    detections = read_track_csv(source_dir / "trained_track.csv")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    enriched: list[dict[str, Any]] = []
    for event in source_doc.get("events", []):
        if event.get("type") in {"touch", "drop_floor", "stall"}:
            fix = refine_ball_for_event(cap, fps, detections, event)
            enriched.append(enrich_event_with_ball(event, fix))
        else:
            enriched.append(dict(event))
    cap.release()

    enriched, suppressed_touches = filter_touch_events(enriched)
    recovered = recover_missing_touches(video, detections, enriched)
    enriched.extend(recovered)
    enriched, recovered_suppressed_touches = filter_touch_events(enriched)
    suppressed_touches.extend(recovered_suppressed_touches)
    existing_stalls = [item for item in enriched if item.get("type") == "stall"]
    extra_stalls = detect_extra_stalls(video, detections)
    event_stalls = detect_event_centered_stalls(video, detections, enriched)
    reviewed = reviewed_stall_events(video)
    if reviewed:
        cap = cv2.VideoCapture(str(video))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            existing_stalls = [
                enrich_event_with_ball(event, refine_ball_for_event(cap, fps, detections, event))
                for event in reviewed
            ]
            cap.release()
        else:
            existing_stalls = reviewed
    all_stalls = merge_stall_events(existing_stalls, extra_stalls + event_stalls)
    enriched = [item for item in enriched if item.get("type") != "stall"]
    enriched.extend(all_stalls)
    enriched, reclassified_floor_touches = reclassify_floor_touches(enriched)
    hidden_drops = detect_hidden_drops(video, detections, enriched)
    enriched.extend(hidden_drops)
    enriched, suppressed_quality_events = filter_quality_events(enriched)
    enriched, suppressed_precision_events = filter_precision_gate_events(enriched)
    suppressed_touches.extend([item for item in suppressed_quality_events if item.get("type") == "touch"])
    suppressed_touches.extend([item for item in suppressed_precision_events if item.get("type") == "touch"])
    enriched.sort(key=event_time)
    enriched, rallies = assign_rallies(enriched)
    enriched.sort(key=event_time)
    suppressed_events = sorted(
        [
            *suppressed_touches,
            *[item for item in suppressed_quality_events if item.get("type") != "touch"],
            *[item for item in suppressed_precision_events if item.get("type") != "touch"],
        ],
        key=event_time,
    )
    suppressed_events = [annotate_precision_gate(item) for item in suppressed_events]

    doc = {
        "source_video": video.name,
        "annotation_method": "qa_visual_ball_refinement_hidden_drop_detection_rally_metrics",
        "video_meta": video_meta(video),
        "summary": {
            "touch_candidates": sum(1 for item in enriched if item.get("type") == "touch"),
            "ground_hit_candidates": sum(1 for item in enriched if item.get("type") == "drop_floor"),
            "hidden_drop_candidates": len(hidden_drops),
            "reclassified_floor_touches": reclassified_floor_touches,
            "stall_candidates": sum(1 for item in enriched if item.get("type") == "stall"),
            "around_the_world_candidates": sum(1 for item in enriched if item.get("type") == "around_the_world"),
            "rallies": len(rallies),
            "recovered_touches": len(recovered),
            "suppressed_touch_candidates": len(suppressed_touches),
            "suppressed_drop_candidates": sum(1 for item in suppressed_quality_events if item.get("type") == "drop_floor"),
            "precision_gate_suppressed_candidates": sum(1 for item in suppressed_events if item.get("precision_gate_hard_veto")),
            "precision_gate_suppressed_touches": sum(1 for item in suppressed_events if item.get("type") == "touch" and item.get("precision_gate_hard_veto")),
            "precision_gate_suppressed_drops": sum(1 for item in suppressed_events if item.get("type") == "drop_floor" and item.get("precision_gate_hard_veto")),
            "precision_gate_soft_flags": sum(1 for item in [*enriched, *suppressed_events] if item.get("precision_gate_soft_flag")),
            "suppressed_stall_candidates": sum(1 for item in suppressed_quality_events if item.get("type") == "stall"),
            "suppressed_trick_candidates": sum(1 for item in suppressed_quality_events if item.get("type") == "around_the_world"),
            "low_ball_accuracy_events": sum(1 for item in enriched if item.get("qa_ball_accuracy") == "low"),
            "large_ball_corrections": sum(1 for item in enriched if float(item.get("qa_ball_correction_px") or 0) >= 90),
            "mean_quality_score": None if not rallies else round(float(np.mean([item["quality_score"] for item in rallies])), 1),
        },
        "rallies": rallies,
        "events": enriched,
        "suppressed_events": suppressed_events,
    }
    json_path = out_dir / "qa_events.json"
    csv_path = out_dir / "qa_events.csv"
    sheet_path = out_dir / "qa_touch_sheet.jpg"
    json_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    write_events_csv(enriched, csv_path)
    draw_qa_sheet(video, enriched, sheet_path)
    return {
        "video": str(video),
        "qa_events_path": str(json_path),
        "qa_csv_path": str(csv_path),
        "qa_sheet_path": str(sheet_path),
        **doc["summary"],
    }


def build_report(manifest: dict[str, Any], path: Path) -> None:
    contact_sides: dict[str, int] = {}
    contact_types: dict[str, int] = {}
    quality_grades: dict[str, int] = {}
    knee_candidates: list[tuple[str, float, str]] = []
    for item in manifest["runs"]:
        doc_path = Path(item["qa_events_path"])
        if not doc_path.exists():
            continue
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        for event in doc.get("events", []):
            if event.get("type") != "touch":
                continue
            side = str(event.get("contact_side", "unknown"))
            contact_type = str(event.get("contact_type", "unknown"))
            contact_sides[side] = contact_sides.get(side, 0) + 1
            contact_types[contact_type] = contact_types.get(contact_type, 0) + 1
            if "knee" in contact_type:
                knee_candidates.append((Path(item["video"]).name, event_time(event), side))
        for rally in doc.get("rallies", []):
            grade = str(rally.get("quality_grade", "unknown"))
            quality_grades[grade] = quality_grades.get(grade, 0) + 1

    lines = [
        "# QA Rally Enrichment Report",
        "",
        f"- Videos processed: {len(manifest['runs'])}",
        f"- Total touches: {sum(item['touch_candidates'] for item in manifest['runs'])}",
        f"- Total ground resets: {sum(item['ground_hit_candidates'] for item in manifest['runs'])}",
        f"- Hidden visual drop resets: {sum(item.get('hidden_drop_candidates', 0) for item in manifest['runs'])}",
        f"- Reclassified floor touches: {sum(item.get('reclassified_floor_touches', 0) for item in manifest['runs'])}",
        f"- Total stalls: {sum(item['stall_candidates'] for item in manifest['runs'])}",
        f"- Total recovered touches: {sum(item['recovered_touches'] for item in manifest['runs'])}",
        f"- Suppressed weak/non-contact touches: {sum(item.get('suppressed_touch_candidates', 0) for item in manifest['runs'])}",
        f"- Suppressed floor reset candidates: {sum(item.get('suppressed_drop_candidates', 0) for item in manifest['runs'])}",
        f"- Suppressed stall candidates: {sum(item.get('suppressed_stall_candidates', 0) for item in manifest['runs'])}",
        f"- Suppressed trick candidates: {sum(item.get('suppressed_trick_candidates', 0) for item in manifest['runs'])}",
        f"- Total large ball corrections: {sum(item['large_ball_corrections'] for item in manifest['runs'])}",
        "",
        "## Contact Rollup",
        "",
        f"- Side counts: {', '.join(f'{key}={value}' for key, value in sorted(contact_sides.items())) or 'none'}",
        f"- Contact type counts: {', '.join(f'{key}={value}' for key, value in sorted(contact_types.items())) or 'none'}",
        f"- Rally grades: {', '.join(f'{key}={value}' for key, value in sorted(quality_grades.items())) or 'none'}",
        f"- Knee candidates: {len(knee_candidates)}",
        "",
        "| Video | Touch | Ground | Hidden drop | Reclass floor | Supp touch | Supp drop | Supp stall | Supp trick | Stall | ATW | Rallies | Avg quality | Low ball | Big fixes | Sheet |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in manifest["runs"]:
        lines.append(
            f"| `{Path(item['video']).name}` | {item['touch_candidates']} | {item['ground_hit_candidates']} | "
            f"{item.get('hidden_drop_candidates', 0)} | {item.get('reclassified_floor_touches', 0)} | "
            f"{item.get('suppressed_touch_candidates', 0)} | "
            f"{item.get('suppressed_drop_candidates', 0)} | {item.get('suppressed_stall_candidates', 0)} | "
            f"{item.get('suppressed_trick_candidates', 0)} | "
            f"{item['stall_candidates']} | {item['around_the_world_candidates']} | {item['rallies']} | "
            f"{item['mean_quality_score']} | {item['low_ball_accuracy_events']} | {item['large_ball_corrections']} | "
            f"`{item['qa_sheet_path']}` |"
        )
    lines.extend(
        [
            "",
            "## QA Notes",
            "",
            "- Green circles in QA sheets are high-confidence corrected ball centers; yellow circles are the prior event center.",
            "- Orange/red corrected centers mean the ball is still visually uncertain and should be reviewed manually.",
            "- Hidden drop resets are inserted from visual continuity when the sack is low/on the floor during a gap or after the final contact.",
            "- Stall detection is intentionally conservative: a stall must survive a duration/low-motion/near-foot window check.",
            "- Left/right and knee labels are geometric CV estimates from POV footage, not final biomechanical truth.",
            "- Rally quality is a heuristic combining touch count, duration, stalls/tricks, airtime consistency, and visual confidence.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QA enrichment on full training outputs")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.out_root.mkdir(parents=True, exist_ok=True)
    runs = []
    for run in source_manifest.get("runs", []):
        result = process_run(run, args.out_root)
        runs.append(result)
        print(
            f"{Path(result['video']).name}: touch={result['touch_candidates']} ground={result['ground_hit_candidates']} "
            f"stall={result['stall_candidates']} rallies={result['rallies']} quality={result['mean_quality_score']}"
        )
    manifest = {
        "source_manifest": str(args.manifest),
        "annotation_method": "qa_visual_ball_refinement_hidden_drop_detection_rally_metrics",
        "runs": runs,
    }
    manifest_path = args.out_root / "qa_manifest.json"
    report_path = args.out_root / "qa_report.md"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    build_report(manifest, report_path)
    print(f"manifest: {manifest_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
