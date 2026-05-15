#!/usr/bin/env python3
"""Weakly train and run a multimodal footbag touch detector.

This is a real end-to-end training/evaluation pass, but it is not a final
supervised model. It uses the checked video-50 labels plus high-confidence
auto candidates from new clips to train a visual patch classifier, tracks the
bag frame-by-frame, then fuses visual motion with contact-sound candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from hacky_mvp import AudioPeak, detect_audio_peaks, extract_mono_wav
from paint_hud import detect_bag_center
from scan_training_data import candidate_event_doc, group_peaks, video_meta


DEFAULT_CHECKED_EVENTS = Path("data/video-50_singular_display.events.json")
DEFAULT_CHECKED_ANCHORS = Path("data/video-50_singular_display.paint_anchors.json")
DEFAULT_CHECKED_VIDEO = Path("/Users/isaacaudet/Downloads/video-50_singular_display.MOV")


@dataclass
class Detection:
    frame_idx: int
    time_sec: float
    x: float | None
    y: float | None
    score: float
    source: str


@dataclass
class TouchCandidate:
    time_sec: float
    score: float
    source: str
    audio_z: float
    visual_score: float
    motion_score: float
    x: float | None
    y: float | None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_anchors(path: Path) -> dict[str, tuple[float, float]]:
    if not path.exists():
        return {}
    data = read_json(path)
    return {str(item["key"]): (float(item["x"]), float(item["y"])) for item in data.get("anchors", [])}


def event_key(rally_id: int, event: dict[str, Any]) -> str:
    number = event.get("touch_number") or 0
    return f"r{rally_id}:{event['type']}:{number}:{float(event['time_sec']):.2f}"


def read_resized_frame(cap: cv2.VideoCapture, time_sec: float, out_size: tuple[int, int]) -> np.ndarray | None:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, round(time_sec * fps)))
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)


def crop_patch(frame: np.ndarray, x: float, y: float, size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    half = size // 2
    x0, x1 = int(round(x)) - half, int(round(x)) + half
    y0, y1 = int(round(y)) - half, int(round(y)) + half
    pad_left, pad_top = max(0, -x0), max(0, -y0)
    pad_right, pad_bottom = max(0, x1 - w), max(0, y1 - h)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    patch = frame[y0:y1, x0:x1]
    if any([pad_left, pad_top, pad_right, pad_bottom]):
        patch = cv2.copyMakeBorder(patch, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
    if patch.shape[0] != size or patch.shape[1] != size:
        patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    return patch


def patch_features(patch: np.ndarray) -> np.ndarray:
    patch = cv2.resize(patch, (48, 48), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    features: list[np.ndarray] = []

    hist_h = cv2.calcHist([hsv], [0], None, [18], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [10], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [10], [0, 256]).flatten()
    for hist in (hist_h, hist_s, hist_v):
        hist = hist.astype(np.float32)
        hist /= max(float(hist.sum()), 1.0)
        features.append(hist)

    means = np.concatenate([patch.mean(axis=(0, 1)), patch.std(axis=(0, 1)), hsv.mean(axis=(0, 1)), hsv.std(axis=(0, 1))]).astype(np.float32)
    features.append(means / np.array([255, 255, 255, 128, 128, 128, 180, 255, 255, 64, 96, 96], dtype=np.float32))

    small = cv2.resize(patch, (12, 12), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1) / 255.0
    features.append(small)

    edges = cv2.Canny(gray, 60, 140)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    color_mask = ((sat > 45) & (val > 45)).astype(np.uint8)
    extras = np.array(
        [
            edges.mean() / 255.0,
            grad.mean() / 255.0,
            grad.std() / 255.0,
            color_mask.mean(),
            np.var(gray) / (255.0 * 255.0),
            np.mean(sat > 80),
            np.mean(val > 160),
        ],
        dtype=np.float32,
    )
    features.append(extras)
    return np.concatenate(features).astype(np.float32)


def colorful_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    yellow_green = (h >= 14) & (h <= 96)
    blue = (h >= 96) & (h <= 138)
    mask = ((s > 42) & (v > 38) & (yellow_green | blue)).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def region_stats(contour: np.ndarray) -> tuple[float, float, int, int, int, int, float, float]:
    area = float(cv2.contourArea(contour))
    x, y, w, h = cv2.boundingRect(contour)
    peri = cv2.arcLength(contour, True)
    circularity = 4 * math.pi * area / (peri * peri + 1e-6)
    aspect = w / max(1, h)
    return area, aspect, x, y, w, h, circularity, peri


def proposals_for_frame(frame: np.ndarray, previous: tuple[float, float] | None = None, max_global: int = 24) -> list[tuple[float, float, str]]:
    mask = colorful_mask(frame)
    h, w = frame.shape[:2]
    proposals: list[tuple[float, float, str]] = []
    scored: list[tuple[float, float, float, str]] = []
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    del labels
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        order = np.argsort(areas)[::-1][: max_global * 5]
    else:
        order = []
    for raw_idx in order:
        idx = int(raw_idx) + 1
        area = float(stats[idx, cv2.CC_STAT_AREA])
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        aspect = bw / max(1, bh)
        if area < 9 or area > 2600:
            continue
        if bw < 4 or bh < 4 or bw > 92 or bh > 92:
            continue
        if aspect < 0.28 or aspect > 3.2:
            continue
        cx, cy = float(centroids[idx][0]), float(centroids[idx][1])
        compactness = min(bw, bh) / max(bw, bh, 1)
        score = area * (0.65 + compactness)
        if previous is not None:
            distance = math.hypot(cx - previous[0], cy - previous[1])
            score *= 0.75 + 1.2 * math.exp(-(distance * distance) / (2 * 95 * 95))
        scored.append((score, cx, cy, "mask"))
    scored.sort(reverse=True, key=lambda item: item[0])
    proposals.extend((cx, cy, source) for _, cx, cy, source in scored[:max_global])

    if previous is not None:
        px, py = previous
        for dy in (-52, 0, 52):
            for dx in (-52, 0, 52):
                cx, cy = px + dx, py + dy
                if 20 <= cx <= w - 20 and 20 <= cy <= h - 20:
                    proposals.append((cx, cy, "track"))
    return proposals


def best_proposal(
    frame: np.ndarray,
    model: HistGradientBoostingClassifier,
    previous: tuple[float, float] | None,
    patch_size: int,
) -> tuple[float | None, float | None, float, str]:
    proposals = proposals_for_frame(frame, previous)
    if not proposals:
        return None, None, 0.0, "none"
    features = np.stack([patch_features(crop_patch(frame, x, y, patch_size)) for x, y, _ in proposals])
    probs = model.predict_proba(features)[:, 1]
    adjusted = probs.copy()
    if previous is not None:
        for idx, (x, y, _) in enumerate(proposals):
            dist = math.hypot(x - previous[0], y - previous[1])
            adjusted[idx] = min(1.0, adjusted[idx] + 0.18 * math.exp(-(dist * dist) / (2 * 90 * 90)))
    best_idx = int(np.argmax(adjusted))
    x, y, source = proposals[best_idx]
    return x, y, float(adjusted[best_idx]), source


def sample_random_negatives(frame: np.ndarray, avoid: list[tuple[float, float]], rng: np.random.Generator, count: int, patch_size: int) -> list[np.ndarray]:
    h, w = frame.shape[:2]
    samples: list[np.ndarray] = []
    tries = 0
    while len(samples) < count and tries < count * 50:
        tries += 1
        x = float(rng.integers(patch_size // 2, max(patch_size // 2 + 1, w - patch_size // 2)))
        y = float(rng.integers(patch_size // 2, max(patch_size // 2 + 1, h - patch_size // 2)))
        if any(math.hypot(x - ax, y - ay) < patch_size * 1.3 for ax, ay in avoid):
            continue
        samples.append(crop_patch(frame, x, y, patch_size))
    return samples


def audio_peaks_for_video(video: Path, min_z: float, min_gap_sec: float) -> list[AudioPeak]:
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        extract_mono_wav(video, wav)
        return detect_audio_peaks(wav, min_z=min_z, min_gap_sec=min_gap_sec)


def high_confidence_auto_centers(video: Path, peaks: list[AudioPeak], out_size: tuple[int, int], z_min: float) -> list[tuple[float, float, float]]:
    cap = cv2.VideoCapture(str(video))
    centers: list[tuple[float, float, float]] = []
    for peak in peaks:
        if peak.z_score < z_min:
            continue
        frame = read_resized_frame(cap, peak.time_sec + 0.06, out_size)
        if frame is None:
            continue
        center = detect_bag_center(frame)
        if center is None:
            continue
        centers.append((peak.time_sec + 0.06, center[0], center[1]))
    cap.release()
    return centers


def collect_training_samples(
    videos: list[Path],
    out_size: tuple[int, int],
    patch_size: int,
    checked_video: Path,
    checked_events: Path,
    checked_anchors: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(13)
    positives: list[np.ndarray] = []
    negatives: list[np.ndarray] = []
    report: dict[str, Any] = {"positive_sources": {}, "negative_sources": {}}

    anchors = normalized_anchors(checked_anchors)
    checked_doc = read_json(checked_events)
    cap = cv2.VideoCapture(str(checked_video))
    checked_count = 0
    for rally in checked_doc["rallies"]:
        for event in rally["events"]:
            if event["type"] not in {"touch", "stall"}:
                continue
            key = event_key(int(rally["id"]), event)
            if key not in anchors:
                continue
            for offset in (-0.04, 0.0, 0.05):
                frame = read_resized_frame(cap, float(event["time_sec"]) + offset, out_size)
                if frame is None:
                    continue
                x = anchors[key][0] * out_size[0]
                y = anchors[key][1] * out_size[1]
                positives.append(crop_patch(frame, x, y, patch_size))
                negatives.extend(sample_random_negatives(frame, [(x, y)], rng, 5, patch_size))
                checked_count += 1
    cap.release()
    report["positive_sources"]["checked_video_50_anchor_samples"] = checked_count
    report["negative_sources"]["checked_video_50_random"] = len(negatives)

    for video in videos:
        peaks = audio_peaks_for_video(video, min_z=6.0, min_gap_sec=0.32)
        centers = high_confidence_auto_centers(video, peaks, out_size, z_min=8.0)
        cap = cv2.VideoCapture(str(video))
        pos_count_before = len(positives)
        neg_count_before = len(negatives)
        for time_sec, x, y in centers:
            frame = read_resized_frame(cap, time_sec, out_size)
            if frame is None:
                continue
            positives.append(crop_patch(frame, x, y, patch_size))
            negatives.extend(sample_random_negatives(frame, [(x, y)], rng, 4, patch_size))
            # Add hard negatives from other colorful blobs in the same frame.
            for px, py, _ in proposals_for_frame(frame)[:10]:
                if math.hypot(px - x, py - y) >= patch_size * 1.4:
                    negatives.append(crop_patch(frame, px, py, patch_size))
        cap.release()
        report["positive_sources"][video.name] = len(positives) - pos_count_before
        report["negative_sources"][video.name] = len(negatives) - neg_count_before

    x = np.stack([patch_features(patch) for patch in positives + negatives])
    y = np.array([1] * len(positives) + [0] * len(negatives), dtype=np.uint8)
    report["total_positive_samples"] = len(positives)
    report["total_negative_samples"] = len(negatives)
    return x, y, report


def train_model(
    videos: list[Path],
    model_dir: Path,
    *,
    out_size: tuple[int, int],
    patch_size: int,
    checked_video: Path,
    checked_events: Path,
    checked_anchors: Path,
) -> tuple[HistGradientBoostingClassifier, Path, dict[str, Any]]:
    x, y, sample_report = collect_training_samples(videos, out_size, patch_size, checked_video, checked_events, checked_anchors)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=7, stratify=y)
    model = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        class_weight="balanced",
        random_state=7,
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    report = {
        "sample_report": sample_report,
        "test_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "out_size": {"width": out_size[0], "height": out_size[1]},
        "patch_size": patch_size,
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "footbag_patch_hgb.joblib"
    joblib.dump({"model": model, "report": report}, model_path)
    (model_dir / "training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return model, model_path, report


def smooth_series(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="same")


def track_video(video: Path, model: HistGradientBoostingClassifier, out_dir: Path, *, out_size: tuple[int, int], patch_size: int) -> list[Detection]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detections: list[Detection] = []
    previous: tuple[float, float] | None = None
    miss_count = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)
        x, y, score, source = best_proposal(frame, model, previous, patch_size)
        if x is not None and score >= 0.36:
            detections.append(Detection(frame_idx, frame_idx / fps, x, y, score, source))
            previous = (x, y)
            miss_count = 0
        else:
            detections.append(Detection(frame_idx, frame_idx / fps, None, None, score, source))
            miss_count += 1
            if miss_count > 8:
                previous = None
        frame_idx += 1
    cap.release()

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "trained_track.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_idx", "time_sec", "x", "y", "score", "source"])
        writer.writeheader()
        for det in detections:
            writer.writerow(
                {
                    "frame_idx": det.frame_idx,
                    "time_sec": round(det.time_sec, 4),
                    "x": "" if det.x is None else round(det.x, 2),
                    "y": "" if det.y is None else round(det.y, 2),
                    "score": round(det.score, 4),
                    "source": det.source,
                }
            )
    return detections


def track_video_strided(
    video: Path,
    model: HistGradientBoostingClassifier,
    out_dir: Path,
    *,
    out_size: tuple[int, int],
    patch_size: int,
    frame_stride: int,
) -> list[Detection]:
    if frame_stride <= 1:
        return track_video(video, model, out_dir, out_size=out_size, patch_size=patch_size)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detections: list[Detection] = []
    previous: tuple[float, float] | None = None
    previous_det: Detection | None = None
    miss_count = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_stride == 0 or previous_det is None:
            frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)
            x, y, score, source = best_proposal(frame, model, previous, patch_size)
            if x is not None and score >= 0.36:
                det = Detection(frame_idx, frame_idx / fps, x, y, score, source)
                previous = (x, y)
                miss_count = 0
            else:
                det = Detection(frame_idx, frame_idx / fps, None, None, score, source)
                miss_count += 1
                if miss_count > 4:
                    previous = None
            previous_det = det
        else:
            decay = max(0.0, previous_det.score - 0.025 * (frame_idx % frame_stride))
            det = Detection(
                frame_idx,
                frame_idx / fps,
                previous_det.x,
                previous_det.y,
                decay,
                "stride",
            )
        detections.append(det)
        frame_idx += 1
    cap.release()

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "trained_track.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_idx", "time_sec", "x", "y", "score", "source"])
        writer.writeheader()
        for det in detections:
            writer.writerow(
                {
                    "frame_idx": det.frame_idx,
                    "time_sec": round(det.time_sec, 4),
                    "x": "" if det.x is None else round(det.x, 2),
                    "y": "" if det.y is None else round(det.y, 2),
                    "score": round(det.score, 4),
                    "source": det.source,
                }
            )
    return detections


def interpolated_track(detections: list[Detection], out_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times = np.array([det.time_sec for det in detections], dtype=np.float32)
    scores = np.array([det.score if det.x is not None else 0.0 for det in detections], dtype=np.float32)
    valid = np.array([det.x is not None and det.y is not None and det.score >= 0.36 for det in detections])
    xs = np.array([det.x if det.x is not None else np.nan for det in detections], dtype=np.float32)
    ys = np.array([det.y if det.y is not None else np.nan for det in detections], dtype=np.float32)
    if valid.sum() < 2:
        xs = np.full_like(times, out_size[0] / 2)
        ys = np.full_like(times, out_size[1] * 0.65)
        return times, xs, ys, scores
    xs_interp = np.interp(times, times[valid], xs[valid])
    ys_interp = np.interp(times, times[valid], ys[valid])
    xs_smooth = smooth_series(xs_interp, 5)
    ys_smooth = smooth_series(ys_interp, 5)
    return times, xs_smooth, ys_smooth, scores


def motion_scores(detections: list[Detection], out_size: tuple[int, int]) -> np.ndarray:
    times, xs, ys, scores = interpolated_track(detections, out_size)
    if len(times) < 5:
        return np.zeros(len(times), dtype=np.float32)
    dt = max(float(np.median(np.diff(times))), 1 / 30)
    vx = np.gradient(xs, dt)
    vy = np.gradient(ys, dt)
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    accel = np.sqrt(ax * ax + ay * ay)
    accel_norm = np.clip(accel / max(float(np.percentile(accel, 96)), 1.0), 0, 1)
    low = np.clip((ys / out_size[1] - 0.35) / 0.45, 0, 1)
    score = 0.65 * accel_norm + 0.35 * low
    score *= np.clip(scores / 0.55, 0, 1)
    return score.astype(np.float32)


def merge_candidates(candidates: list[TouchCandidate], min_gap_sec: float = 0.32) -> list[TouchCandidate]:
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda item: item.time_sec)
    merged: list[TouchCandidate] = []
    cluster: list[TouchCandidate] = [candidates[0]]
    for cand in candidates[1:]:
        if cand.time_sec - cluster[-1].time_sec <= min_gap_sec:
            cluster.append(cand)
        else:
            merged.append(max(cluster, key=lambda item: item.score))
            cluster = [cand]
    merged.append(max(cluster, key=lambda item: item.score))
    return merged


def detect_touches(
    video: Path,
    detections: list[Detection],
    *,
    out_size: tuple[int, int],
    audio_min_z: float,
    audio_gap_sec: float,
) -> list[TouchCandidate]:
    peaks = audio_peaks_for_video(video, min_z=audio_min_z, min_gap_sec=audio_gap_sec)
    motion = motion_scores(detections, out_size)
    frame_times = np.array([det.time_sec for det in detections], dtype=np.float32)
    candidates: list[TouchCandidate] = []
    z_values = np.array([peak.z_score for peak in peaks], dtype=np.float32)
    z_scale = max(float(np.percentile(z_values, 90)) if len(z_values) else 1.0, 6.0)

    for peak in peaks:
        idx = int(np.clip(np.searchsorted(frame_times, peak.time_sec), 0, len(detections) - 1))
        det = detections[idx]
        visual = det.score if det.x is not None else 0.0
        motion_score = float(motion[idx])
        audio_score = min(1.0, peak.z_score / z_scale)
        in_contact_zone = det.y is not None and det.y / out_size[1] >= 0.32
        away_from_edge = det.x is not None and out_size[0] * 0.035 <= det.x <= out_size[0] * 0.965
        if not (in_contact_zone and away_from_edge):
            continue
        score = 0.38 * audio_score + 0.36 * visual + 0.26 * motion_score
        if visual >= 0.34 and score >= 0.42:
            candidates.append(
                TouchCandidate(
                    time_sec=peak.time_sec,
                    score=score,
                    source="audio+vision",
                    audio_z=peak.z_score,
                    visual_score=visual,
                    motion_score=motion_score,
                    x=det.x,
                    y=det.y,
                )
            )

    # Add visual-only bounces to catch quieter contacts.
    times, xs, ys, scores = interpolated_track(detections, out_size)
    if len(times) >= 5:
        dt = max(float(np.median(np.diff(times))), 1 / 30)
        vy = np.gradient(ys, dt)
        for i in range(2, len(times) - 2):
            falling_then_up = vy[i - 1] > 35 and vy[i + 1] < -35
            local_low = ys[i] >= ys[i - 1] and ys[i] >= ys[i + 1]
            if not (falling_then_up or local_low):
                continue
            if ys[i] / out_size[1] < 0.40 or xs[i] < out_size[0] * 0.045 or xs[i] > out_size[0] * 0.955:
                continue
            if scores[i] < 0.64 or motion[i] < 0.58:
                continue
            if any(abs(times[i] - cand.time_sec) <= 0.28 for cand in candidates):
                continue
            candidates.append(
                TouchCandidate(
                    time_sec=float(times[i]),
                    score=float(0.55 * scores[i] + 0.45 * motion[i]),
                    source="visual-motion",
                    audio_z=0.0,
                    visual_score=float(scores[i]),
                    motion_score=float(motion[i]),
                    x=float(xs[i]),
                    y=float(ys[i]),
                )
            )

    merged = merge_candidates(candidates, min_gap_sec=0.32)
    return [cand for cand in merged if cand.score >= 0.44]


def events_doc_from_touches(video: Path, touches: list[TouchCandidate], duration_sec: float) -> dict[str, Any]:
    groups = group_touch_candidates(touches, rally_gap_sec=2.2)
    rallies: list[dict[str, Any]] = []
    for rally_idx, group in enumerate(groups, start=1):
        events = []
        for touch_idx, cand in enumerate(group, start=1):
            events.append(
                {
                    "type": "touch",
                    "touch_number": touch_idx,
                    "time_sec": round(cand.time_sec, 3),
                    "label": cand.source,
                    "confidence": round(cand.score, 3),
                    "audio_z": round(cand.audio_z, 2),
                    "visual_score": round(cand.visual_score, 3),
                    "motion_score": round(cand.motion_score, 3),
                    "x": None if cand.x is None else round(cand.x, 2),
                    "y": None if cand.y is None else round(cand.y, 2),
                }
            )
        rallies.append(
            {
                "id": rally_idx,
                "label": f"trained candidate rally {rally_idx}",
                "start_sec": round(max(0.0, group[0].time_sec - 0.35), 3),
                "end_sec": round(min(duration_sec, group[-1].time_sec + 0.75), 3),
                "expected_touches": len(group),
                "expected_stalls": 0,
                "events": events,
            }
        )
    return {
        "source_video": video.name,
        "annotation_method": "Weakly trained visual patch classifier + bag tracking + audio/contact-motion fusion. Review candidates, not final ground truth.",
        "rallies": rallies,
    }


def group_touch_candidates(touches: list[TouchCandidate], rally_gap_sec: float) -> list[list[TouchCandidate]]:
    groups: list[list[TouchCandidate]] = []
    current: list[TouchCandidate] = []
    for touch in sorted(touches, key=lambda item: item.time_sec):
        if current and touch.time_sec - current[-1].time_sec > rally_gap_sec:
            groups.append(current)
            current = []
        current.append(touch)
    if current:
        groups.append(current)
    return groups


def write_events_csv(doc: dict[str, Any], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rally_id",
                "touch_number",
                "time_sec",
                "confidence",
                "audio_z",
                "visual_score",
                "motion_score",
                "x",
                "y",
                "label",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        for rally in doc["rallies"]:
            for event in rally["events"]:
                row = dict(event)
                row["rally_id"] = rally["id"]
                writer.writerow(row)


def read_frame_by_time(cap: cv2.VideoCapture, time_sec: float, out_size: tuple[int, int]) -> np.ndarray | None:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, round(time_sec * fps)))
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)


def write_touch_sheet(video: Path, doc: dict[str, Any], out_path: Path, out_size: tuple[int, int], *, max_events: int = 90) -> None:
    cap = cv2.VideoCapture(str(video))
    thumbs: list[np.ndarray] = []
    flat = [(rally["id"], event) for rally in doc["rallies"] for event in rally["events"]][:max_events]
    for rally_id, event in flat:
        frame = read_frame_by_time(cap, float(event["time_sec"]) + 0.05, out_size)
        if frame is None:
            continue
        if event.get("x") is not None and event.get("y") is not None:
            cx, cy = int(round(float(event["x"]))), int(round(float(event["y"])))
            cv2.circle(frame, (cx, cy), 22, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1, cv2.LINE_AA)
        label = (
            f"R{rally_id} T{event['touch_number']} {float(event['time_sec']):.2f}s "
            f"c{float(event['confidence']):.2f} z{float(event['audio_z']):.1f}"
        )
        cv2.putText(frame, label, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 255, 255), 2, cv2.LINE_AA)
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


def render_debug_overlay(
    video: Path,
    detections: list[Detection],
    doc: dict[str, Any],
    out_path: Path,
    out_size: tuple[int, int],
) -> None:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
    flat_events = [event for rally in doc["rallies"] for event in rally["events"]]
    event_times = [float(event["time_sec"]) for event in flat_events]
    frame_idx = 0
    touch_count = 0
    next_event_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame_idx >= len(detections):
            break
        frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)
        t = frame_idx / fps
        while next_event_idx < len(event_times) and event_times[next_event_idx] <= t + 0.03:
            touch_count += 1
            next_event_idx += 1
        det = detections[frame_idx]
        if det.x is not None and det.y is not None and det.score >= 0.36:
            cv2.circle(frame, (int(det.x), int(det.y)), 18, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"bag {det.score:.2f}", (int(det.x) + 20, int(det.y) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, f"bag {det.score:.2f}", (int(det.x) + 20, int(det.y) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        recent = [event for event in flat_events if 0 <= t - float(event["time_sec"]) <= 0.32]
        if recent:
            event = recent[-1]
            if event.get("x") is not None and event.get("y") is not None:
                cx, cy = int(float(event["x"])), int(float(event["y"]))
                cv2.circle(frame, (cx, cy), 38, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, f"TOUCH {touch_count}", (24, 116), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 0), 7, cv2.LINE_AA)
            cv2.putText(frame, f"TOUCH {touch_count}", (24, 116), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.rectangle(frame, (12, 12), (320, 70), (0, 0, 0), -1)
        cv2.putText(frame, f"trained candidates: {touch_count}", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        frame_idx += 1
    writer.release()
    cap.release()


def process_video(
    video: Path,
    model: HistGradientBoostingClassifier,
    out_root: Path,
    *,
    out_size: tuple[int, int],
    patch_size: int,
    audio_min_z: float,
    audio_gap_sec: float,
    render_overlay: bool,
    frame_stride: int,
) -> dict[str, Any]:
    out_dir = out_root / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    detections = track_video_strided(video, model, out_dir, out_size=out_size, patch_size=patch_size, frame_stride=frame_stride)
    touches = detect_touches(video, detections, out_size=out_size, audio_min_z=audio_min_z, audio_gap_sec=audio_gap_sec)
    meta = video_meta(video)
    doc = events_doc_from_touches(video, touches, float(meta["duration_sec"]))
    events_path = out_dir / "trained_events.json"
    events_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    write_events_csv(doc, out_dir / "trained_events.csv")
    write_touch_sheet(video, doc, out_dir / "trained_touch_sheet.jpg", out_size)
    overlay_path = None
    if render_overlay:
        overlay_path = out_dir / "trained_overlay.mp4"
        render_debug_overlay(video, detections, doc, overlay_path, out_size)
    return {
        "video": str(video),
        "duration_sec": round(float(meta["duration_sec"]), 3),
        "touch_candidates": sum(len(rally["events"]) for rally in doc["rallies"]),
        "candidate_rallies": len(doc["rallies"]),
        "events_path": str(events_path),
        "csv_path": str(out_dir / "trained_events.csv"),
        "track_path": str(out_dir / "trained_track.csv"),
        "sheet_path": str(out_dir / "trained_touch_sheet.jpg"),
        "overlay_path": str(overlay_path) if overlay_path else None,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Weakly train and run a multimodal footbag touch detector.")
    parser.add_argument("videos", type=Path, nargs="+", help="Full videos to process")
    parser.add_argument("--out-root", type=Path, default=Path("outputs"), help="Output root")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Model output dir")
    parser.add_argument("--checked-video", type=Path, default=DEFAULT_CHECKED_VIDEO)
    parser.add_argument("--checked-events", type=Path, default=DEFAULT_CHECKED_EVENTS)
    parser.add_argument("--checked-anchors", type=Path, default=DEFAULT_CHECKED_ANCHORS)
    parser.add_argument("--width", type=int, default=688)
    parser.add_argument("--height", type=int, default=912)
    parser.add_argument("--patch-size", type=int, default=74)
    parser.add_argument("--audio-min-z", type=float, default=4.8)
    parser.add_argument("--audio-gap-sec", type=float, default=0.24)
    parser.add_argument("--render-overlay", action="store_true")
    parser.add_argument("--frame-stride", type=int, default=3, help="Classify every Nth frame and interpolate between")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_size = (args.width, args.height)
    model, model_path, report = train_model(
        args.videos,
        args.model_dir,
        out_size=out_size,
        patch_size=args.patch_size,
        checked_video=args.checked_video,
        checked_events=args.checked_events,
        checked_anchors=args.checked_anchors,
    )
    summaries = [
        process_video(
            video,
            model,
            args.out_root,
            out_size=out_size,
            patch_size=args.patch_size,
            audio_min_z=args.audio_min_z,
            audio_gap_sec=args.audio_gap_sec,
            render_overlay=args.render_overlay,
            frame_stride=args.frame_stride,
        )
        for video in args.videos
    ]
    manifest = args.out_root / "trained_multimodal_manifest.json"
    manifest.write_text(json.dumps({"model_path": str(model_path), "training_report": report, "runs": summaries}, indent=2) + "\n", encoding="utf-8")
    print(f"model: {model_path}")
    print(f"training positives: {report['sample_report']['total_positive_samples']}")
    print(f"training negatives: {report['sample_report']['total_negative_samples']}")
    for item in summaries:
        print(f"{Path(item['video']).name}: {item['touch_candidates']} touch candidates, {item['candidate_rallies']} candidate rallies")
        print(f"  sheet: {item['sheet_path']}")
        if item["overlay_path"]:
            print(f"  overlay: {item['overlay_path']}")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
