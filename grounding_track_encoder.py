#!/usr/bin/env python3
"""Encode footbag tracks with an open-vocabulary detector plus temporal gating.

The earlier tracker used color/blob heuristics and regularly locked onto
shoes, furniture, plants, and other high-contrast objects. This encoder uses
GroundingDINO to propose footbag boxes, links those detections through time,
then only emits rallies where the bag track and audio transients agree.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from hacky_mvp import AudioPeak, detect_audio_peaks, extract_mono_wav
from scan_training_data import video_meta


PROMPT = "a small hacky sack ball. a small footbag. a small patterned ball. a shoe. a hand. a coffee table."
MODEL_ID = "IDEA-Research/grounding-dino-tiny"


@dataclass
class Candidate:
    time_sec: float
    box: tuple[float, float, float, float]
    score: float
    raw_score: float
    appearance_score: float
    label: str

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0

    @property
    def side(self) -> float:
        x0, y0, x1, y1 = self.box
        return math.sqrt(max((x1 - x0) * (y1 - y0), 0.0))


@dataclass
class Track:
    id: int
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def first_time(self) -> float:
        return self.candidates[0].time_sec

    @property
    def last_time(self) -> float:
        return self.candidates[-1].time_sec

    @property
    def mean_score(self) -> float:
        return float(np.mean([candidate.score for candidate in self.candidates])) if self.candidates else 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.last_time - self.first_time)

    def add(self, candidate: Candidate) -> None:
        self.candidates.append(candidate)

    def center_at(self, time_sec: float) -> tuple[float, float] | None:
        candidate = nearest_candidate(self.candidates, time_sec, max_dt=0.7)
        return candidate.center if candidate else None

    def box_at(self, time_sec: float) -> tuple[float, float, float, float] | None:
        candidate = nearest_candidate(self.candidates, time_sec, max_dt=0.7)
        return candidate.box if candidate else None


def run_model_batch(
    processor: Any,
    model: Any,
    images: list[Image.Image],
    device: str,
) -> list[dict[str, Any]]:
    inputs = processor(images=images, text=[PROMPT] * len(images), return_tensors="pt")
    inputs = {key: (value.to(device) if hasattr(value, "to") else value) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"].cpu(),
        threshold=0.11,
        text_threshold=0.11,
        target_sizes=[image.size[::-1] for image in images],
    )


def score_candidate(
    box: tuple[float, float, float, float],
    raw_score: float,
    label: str,
    image_size: tuple[int, int],
    appearance_score: float,
) -> float:
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        return 0.0

    side = math.sqrt(width * height)
    if side < 8 or side > 115:
        size_score = 0.12
    else:
        size_score = math.exp(-((math.log(side / 42.0)) ** 2) / (2 * 0.72 * 0.72))

    aspect = width / max(height, 1.0)
    aspect_score = math.exp(-abs(math.log(max(aspect, 0.05))) * 1.15)

    label_lower = label.lower()
    label_score = 1.0 if any(word in label_lower for word in ("ball", "footbag", "hacky", "patterned")) else 0.35
    if "shoe" in label_lower and not any(word in label_lower for word in ("ball", "footbag", "hacky", "patterned")):
        label_score *= 0.12
    if "coffee table" in label_lower:
        label_score *= 0.25

    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    image_w, image_h = image_size
    border_score = 0.35 if cx < 10 or cy < 10 or cx > image_w - 10 or cy > image_h - 10 else 1.0
    return float(raw_score) * size_score * aspect_score * label_score * border_score * appearance_score


def footbag_appearance_score(image: Image.Image, box: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = [int(round(value)) for value in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(image.width, x1), min(image.height, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0.2

    crop = np.array(image.crop((x0, y0, x1, y1)).convert("RGB"))
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    colorful = (sat > 45) & (val > 45)
    yellow = (hue >= 13) & (hue <= 42)
    green_cyan = (hue >= 42) & (hue <= 96)
    blue = (hue >= 96) & (hue <= 138)
    target = colorful & (yellow | green_cyan | blue)
    target_frac = float(np.mean(target))
    colorful_frac = float(np.mean(colorful))
    dark_frac = float(np.mean(val < 42))
    white_frac = float(np.mean((sat < 28) & (val > 165)))

    score = 0.28 + min(0.72, target_frac * 3.4 + colorful_frac * 0.35)
    if target_frac < 0.035 and colorful_frac < 0.14:
        score *= 0.35
    if dark_frac > 0.66 or white_frac > 0.58:
        score *= 0.55
    return float(np.clip(score, 0.12, 1.25))


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return inter / max(area_a + area_b - inter, 1e-6)


def nms(candidates: list[Candidate], max_items: int) -> list[Candidate]:
    kept: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if any(iou(candidate.box, existing.box) > 0.42 for existing in kept):
            continue
        kept.append(candidate)
        if len(kept) >= max_items:
            break
    return kept


def read_detector_frame(
    cap: cv2.VideoCapture,
    time_sec: float,
    fps: float,
    max_side: int,
) -> tuple[Image.Image, tuple[int, int]] | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(time_sec * fps))))
    ok, frame = cap.read()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame.shape[:2]
    scale = max_side / max(width, height)
    resized = cv2.resize(frame, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
    return Image.fromarray(resized), (width, height)


def make_detection_times(duration: float, peaks: list[AudioPeak], sample_fps: float) -> list[float]:
    times = {round(i / sample_fps, 3) for i in range(int(duration * sample_fps) + 1)}
    for peak in peaks:
        if peak.z_score < 4.5:
            continue
        for offset in (-0.04, 0.0, 0.06):
            t = round(min(max(0.0, peak.time_sec + offset), duration), 3)
            times.add(t)
    return sorted(times)


def detect_candidates(
    video: Path,
    times: list[float],
    *,
    processor: Any,
    model: Any,
    device: str,
    detector_max_side: int,
    output_size: tuple[int, int],
    batch_size: int,
) -> dict[float, list[Candidate]]:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    by_time: dict[float, list[Candidate]] = {}
    out_w, out_h = output_size

    for start in range(0, len(times), batch_size):
        batch_times = times[start : start + batch_size]
        images: list[Image.Image] = []
        image_meta: list[tuple[float, int, int]] = []
        for time_sec in batch_times:
            frame = read_detector_frame(cap, time_sec, fps, detector_max_side)
            if frame is None:
                continue
            image, _source_size = frame
            images.append(image)
            image_meta.append((time_sec, image.width, image.height))
        if not images:
            continue

        results = run_model_batch(processor, model, images, device)
        for image, result, (time_sec, det_w, det_h) in zip(images, results, image_meta):
            candidates: list[Candidate] = []
            scale_x, scale_y = out_w / det_w, out_h / det_h
            for raw_box, raw_score, raw_label in zip(result["boxes"], result["scores"], result["labels"]):
                x0, y0, x1, y1 = [float(value) for value in raw_box]
                label = str(raw_label)
                box = (x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y)
                appearance = footbag_appearance_score(image, (x0, y0, x1, y1))
                score = score_candidate((x0, y0, x1, y1), float(raw_score), label, (det_w, det_h), appearance)
                if score < 0.045:
                    continue
                candidates.append(Candidate(time_sec, box, score, float(raw_score), appearance, label))
            by_time[time_sec] = nms(candidates, max_items=6)
        done = min(start + batch_size, len(times))
        print(f"{video.name}: detected {done}/{len(times)} frames", flush=True)
    cap.release()
    return by_time


def link_tracks(candidates_by_time: dict[float, list[Candidate]]) -> list[Track]:
    tracks: list[Track] = []
    active: list[Track] = []
    next_id = 1
    max_gap = 1.15

    for time_sec in sorted(candidates_by_time):
        candidates = sorted(candidates_by_time[time_sec], key=lambda item: item.score, reverse=True)
        assigned: set[int] = set()

        for track in sorted(active, key=lambda item: item.mean_score, reverse=True):
            dt = time_sec - track.last_time
            if dt <= 0 or dt > max_gap:
                continue
            last_x, last_y = track.candidates[-1].center
            best_index: int | None = None
            best_cost = 1e9
            for index, candidate in enumerate(candidates):
                if index in assigned or candidate.score < 0.075:
                    continue
                cx, cy = candidate.center
                dist = math.hypot(cx - last_x, cy - last_y)
                max_dist = 95 + 520 * dt
                if dist > max_dist:
                    continue
                cost = dist / max_dist - candidate.score * 0.65
                if cost < best_cost:
                    best_cost = cost
                    best_index = index
            if best_index is not None:
                track.add(candidates[best_index])
                assigned.add(best_index)

        for index, candidate in enumerate(candidates):
            if index in assigned or candidate.score < 0.12:
                continue
            track = Track(next_id, [candidate])
            next_id += 1
            tracks.append(track)
            active.append(track)

        active = [track for track in active if time_sec - track.last_time <= max_gap]
    return tracks


def track_path_length(track: Track) -> float:
    if len(track.candidates) < 2:
        return 0.0
    centers = [candidate.center for candidate in track.candidates]
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(centers, centers[1:])))


def audio_count_near(track: Track, peaks: list[AudioPeak], margin: float = 0.45) -> int:
    return sum(1 for peak in peaks if track.first_time - margin <= peak.time_sec <= track.last_time + margin and peak.z_score >= 4.8)


def is_active_track(track: Track, peaks: list[AudioPeak]) -> bool:
    if len(track.candidates) < 2:
        return False
    if track.mean_score < 0.12:
        return False
    path = track_path_length(track)
    audio_hits = audio_count_near(track, peaks)
    if track.duration >= 0.65 and path >= 70:
        return True
    if audio_hits >= 2 and track.duration >= 0.35:
        return True
    if len(track.candidates) >= 3 and track.mean_score >= 0.22 and path >= 35:
        return True
    return False


def active_intervals(tracks: list[Track], peaks: list[AudioPeak]) -> list[tuple[float, float, list[Track]]]:
    active_tracks = [track for track in tracks if is_active_track(track, peaks)]
    active_tracks.sort(key=lambda track: track.first_time)
    intervals: list[tuple[float, float, list[Track]]] = []
    for track in active_tracks:
        start, end = max(0.0, track.first_time - 0.35), track.last_time + 0.35
        if intervals and start - intervals[-1][1] <= 1.1:
            prev_start, prev_end, prev_tracks = intervals[-1]
            intervals[-1] = (prev_start, max(prev_end, end), prev_tracks + [track])
        else:
            intervals.append((start, end, [track]))
    return intervals


def nearest_candidate(candidates: list[Candidate], time_sec: float, max_dt: float) -> Candidate | None:
    if not candidates:
        return None
    best = min(candidates, key=lambda candidate: abs(candidate.time_sec - time_sec))
    return best if abs(best.time_sec - time_sec) <= max_dt else None


def find_track_at(tracks: list[Track], time_sec: float, max_dt: float = 0.55) -> Track | None:
    viable: list[tuple[float, float, Track]] = []
    for track in tracks:
        if track.first_time - max_dt <= time_sec <= track.last_time + max_dt:
            candidate = nearest_candidate(track.candidates, time_sec, max_dt=max_dt)
            if candidate:
                viable.append((abs(candidate.time_sec - time_sec), -candidate.score, track))
    if not viable:
        return None
    return sorted(viable)[0][2]


def detect_touch_events(
    intervals: list[tuple[float, float, list[Track]]],
    peaks: list[AudioPeak],
    *,
    min_z: float,
    min_gap_sec: float,
    min_box_score: float,
    min_track_score: float,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for start, end, tracks in intervals:
        interval_peaks = [peak for peak in peaks if start <= peak.time_sec <= end and peak.z_score >= min_z]
        for peak in interval_peaks:
            if events and peak.time_sec - float(events[-1]["time_sec"]) < min_gap_sec:
                if peak.z_score <= float(events[-1]["audio_z"]):
                    continue
                events.pop()
            track = find_track_at(tracks, peak.time_sec)
            if not track:
                continue
            if track.mean_score < min_track_score:
                continue
            candidate = nearest_candidate(track.candidates, peak.time_sec, max_dt=0.55)
            if not candidate:
                continue
            if candidate.score < min_box_score:
                continue
            x0, y0, x1, y1 = candidate.box
            events.append(
                {
                    "type": "touch",
                    "time_sec": round(peak.time_sec, 3),
                    "label": "grounded-audio+box",
                    "confidence": round(min(1.0, 0.35 + candidate.score + peak.z_score / 28.0), 3),
                    "audio_z": round(peak.z_score, 2),
                    "box_score": round(candidate.score, 3),
                    "appearance_score": round(candidate.appearance_score, 3),
                    "track_id": track.id,
                    "x": round((x0 + x1) / 2.0, 2),
                    "y": round((y0 + y1) / 2.0, 2),
                    "box": [round(value, 2) for value in candidate.box],
                }
            )
    return events


def group_events_into_rallies(events: list[dict[str, Any]], gap_sec: float) -> dict[str, Any]:
    rallies: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda item: float(item["time_sec"])):
        if current and float(event["time_sec"]) - float(current[-1]["time_sec"]) > gap_sec:
            rallies.append(build_rally(len(rallies) + 1, current))
            current = []
        current.append(event)
    if current:
        rallies.append(build_rally(len(rallies) + 1, current))
    return {"rallies": rallies}


def build_rally(rally_id: int, events: list[dict[str, Any]]) -> dict[str, Any]:
    output_events: list[dict[str, Any]] = []
    for index, event in enumerate(events, start=1):
        item = dict(event)
        item["touch_number"] = index
        output_events.append(item)

    start = float(output_events[0]["time_sec"])
    end = float(output_events[-1]["time_sec"])
    final = output_events[-1]
    ground_reset = None
    if final.get("y") is not None and float(final["y"]) > 912 * 0.62:
        ground_reset = {
            "type": "ground_candidate",
            "time_sec": round(end + 0.32, 3),
            "label": "rally reset candidate",
            "source_touch": final.get("touch_number"),
        }
        output_events.append(ground_reset)
    return {
        "id": rally_id,
        "label": f"Grounded rally {rally_id}",
        "start_sec": round(start, 3),
        "end_sec": round(end, 3),
        "expected_touches": len(events),
        "expected_stalls": 0,
        "events": output_events,
    }


def intervals_from_rallies(rallies: list[dict[str, Any]], duration: float, pad: float = 0.65) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for rally in rallies:
        start = max(0.0, float(rally["start_sec"]) - pad)
        end = min(duration, float(rally["end_sec"]) + pad)
        if intervals and start - intervals[-1][1] <= 0.75:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
        else:
            intervals.append((start, end))
    return intervals


def write_track_csv(path: Path, tracks: list[Track]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["track_id", "time_sec", "score", "raw_score", "appearance_score", "label", "x0", "y0", "x1", "y1", "cx", "cy"],
        )
        writer.writeheader()
        for track in tracks:
            for candidate in track.candidates:
                x0, y0, x1, y1 = candidate.box
                cx, cy = candidate.center
                writer.writerow(
                    {
                        "track_id": track.id,
                        "time_sec": round(candidate.time_sec, 3),
                        "score": round(candidate.score, 5),
                        "raw_score": round(candidate.raw_score, 5),
                        "appearance_score": round(candidate.appearance_score, 5),
                        "label": candidate.label,
                        "x0": round(x0, 2),
                        "y0": round(y0, 2),
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "cx": round(cx, 2),
                        "cy": round(cy, 2),
                    }
                )


def write_events_csv(path: Path, doc: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["rally_id", "touch_number", "type", "time_sec", "confidence", "audio_z", "box_score", "appearance_score", "track_id", "x", "y"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rally in doc["rallies"]:
            for event in rally["events"]:
                row = {key: event.get(key) for key in fieldnames}
                row["rally_id"] = rally["id"]
                writer.writerow(row)


def read_output_frame(cap: cv2.VideoCapture, frame_idx: int, output_size: tuple[int, int]) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)


def render_overlay(
    video: Path,
    out_path: Path,
    tracks: list[Track],
    intervals: list[tuple[float, float]],
    doc: dict[str, Any],
    output_size: tuple[int, int],
    output_fps: float,
) -> None:
    meta = video_meta(video)
    source_fps = float(meta["fps"])
    duration = float(meta["duration_sec"])
    frame_step = max(1, round(source_fps / output_fps))
    actual_fps = source_fps / frame_step
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), actual_fps, output_size)
    cap = cv2.VideoCapture(str(video))
    events = [event for rally in doc["rallies"] for event in rally["events"] if event["type"] == "touch"]
    event_times = [float(event["time_sec"]) for event in events]

    frame_idx = 0
    touch_count = 0
    while frame_idx / source_fps <= duration:
        frame = read_output_frame(cap, frame_idx, output_size)
        if frame is None:
            break
        t = frame_idx / source_fps
        is_active = any(start <= t <= end for start, end in intervals)
        color = (92, 214, 118) if is_active else (120, 120, 120)
        track = find_track_at(tracks, t, max_dt=0.55) if is_active else None
        if track:
            box = track.box_at(t)
            candidate = nearest_candidate(track.candidates, t, max_dt=0.55)
            if box and candidate:
                x0, y0, x1, y1 = [int(round(value)) for value in box]
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 3, cv2.LINE_AA)
                cv2.putText(frame, f"bag {candidate.score:.2f}", (x0, max(22, y0 - 9)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, f"bag {candidate.score:.2f}", (x0, max(22, y0 - 9)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        recent = [event for event in events if 0 <= t - float(event["time_sec"]) <= 0.28]
        if recent:
            event = recent[-1]
            touch_count = event_times.index(float(event["time_sec"])) + 1
            cx, cy = int(float(event["x"])), int(float(event["y"]))
            cv2.circle(frame, (cx, cy), 42, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, "TOUCH", (24, 112), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 7, cv2.LINE_AA)
            cv2.putText(frame, "TOUCH", (24, 112), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3, cv2.LINE_AA)

        cv2.rectangle(frame, (12, 12), (410, 76), (0, 0, 0), -1)
        mode = "ACTIVE RALLY" if is_active else "NO RALLY / WALKING"
        cv2.putText(frame, mode, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"touches: {touch_count}", (24, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2, cv2.LINE_AA)
        writer.write(frame)
        frame_idx += frame_step

    cap.release()
    writer.release()


def write_touch_sheet(video: Path, doc: dict[str, Any], out_path: Path, output_size: tuple[int, int], max_events: int = 80) -> None:
    events = [event for rally in doc["rallies"] for event in rally["events"] if event["type"] == "touch"]
    events = events[:max_events]
    if not events:
        return
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    thumbs: list[Image.Image] = []
    for event in events:
        frame = read_output_frame(cap, max(0, int(round(float(event["time_sec"]) * fps))), output_size)
        if frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame).resize((344, 456), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(image)
        box = event.get("box")
        if box:
            scale_x, scale_y = 344 / output_size[0], 456 / output_size[1]
            x0, y0, x1, y1 = [float(value) for value in box]
            draw.rectangle([x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y], outline=(255, 230, 0), width=4)
        draw.text((8, 8), f"R? T{event.get('touch_number', '?')} {event['time_sec']:.2f}s z{event.get('audio_z', 0)}", fill=(255, 255, 0))
        thumbs.append(image)
    cap.release()
    if not thumbs:
        return
    cols = 5
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 344, rows * 456), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % cols) * 344, (index // cols) * 456))
    sheet.save(out_path, quality=92)


def process_video(
    video: Path,
    *,
    processor: Any,
    model: Any,
    device: str,
    out_root: Path,
    sample_fps: float,
    detector_max_side: int,
    output_size: tuple[int, int],
    batch_size: int,
    touch_min_z: float,
    touch_min_box_score: float,
    render: bool,
) -> dict[str, Any]:
    out_dir = out_root / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = video_meta(video)

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / f"{video.stem}.wav"
        extract_mono_wav(video, wav)
        peaks = detect_audio_peaks(wav, min_z=4.5, min_gap_sec=0.22)

    times = make_detection_times(float(meta["duration_sec"]), peaks, sample_fps=sample_fps)
    candidates_by_time = detect_candidates(
        video,
        times,
        processor=processor,
        model=model,
        device=device,
        detector_max_side=detector_max_side,
        output_size=output_size,
        batch_size=batch_size,
    )
    tracks = link_tracks(candidates_by_time)
    track_intervals = active_intervals(tracks, peaks)
    events = detect_touch_events(
        track_intervals,
        peaks,
        min_z=touch_min_z,
        min_gap_sec=0.30,
        min_box_score=touch_min_box_score,
        min_track_score=0.32,
    )
    rallies = group_events_into_rallies(events, gap_sec=2.25)["rallies"]
    play_intervals = intervals_from_rallies(rallies, float(meta["duration_sec"]))
    event_doc = {
        "source_video": video.name,
        "annotation_method": "grounding_dino_temporal_audio",
        "track_model": MODEL_ID,
        "sample_fps": sample_fps,
        "rallies": rallies,
        "active_intervals": [
            {"start_sec": round(start, 3), "end_sec": round(end, 3)}
            for start, end in play_intervals
        ],
        "track_intervals": [
            {"start_sec": round(start, 3), "end_sec": round(end, 3), "track_ids": [track.id for track in group]}
            for start, end, group in track_intervals
        ],
    }

    events_path = out_dir / "grounded_events.json"
    events_path.write_text(json.dumps(event_doc, indent=2) + "\n", encoding="utf-8")
    write_events_csv(out_dir / "grounded_events.csv", event_doc)
    write_track_csv(out_dir / "grounded_track.csv", tracks)
    write_touch_sheet(video, event_doc, out_dir / "grounded_touch_sheet.jpg", output_size)

    overlay_path = None
    if render:
        overlay_path = out_dir / "grounded_overlay.mp4"
        render_overlay(video, overlay_path, tracks, play_intervals, event_doc, output_size, output_fps=30.0)

    return {
        "video": str(video),
        "events_path": str(events_path),
        "overlay_path": str(overlay_path) if overlay_path else None,
        "tracks": len(tracks),
        "active_intervals": len(play_intervals),
        "touch_events": len(events),
        "rallies": len(event_doc["rallies"]),
        "detection_frames": len(times),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GroundingDINO footbag tracking encoder")
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--out-root", type=Path, default=Path("outputs"))
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--detector-max-side", type=int, default=720)
    parser.add_argument("--width", type=int, default=688)
    parser.add_argument("--height", type=int, default=912)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--touch-min-z", type=float, default=4.8)
    parser.add_argument("--touch-min-box-score", type=float, default=0.24)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--render-overlay", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    if device == "auto":
        device = "cpu"
    print(f"loading {MODEL_ID} on {device}", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
    model.eval()
    output_size = (args.width, args.height)

    summaries = [
        process_video(
            video,
            processor=processor,
            model=model,
            device=device,
            out_root=args.out_root,
            sample_fps=args.sample_fps,
            detector_max_side=args.detector_max_side,
            output_size=output_size,
            batch_size=args.batch_size,
            touch_min_z=args.touch_min_z,
            touch_min_box_score=args.touch_min_box_score,
            render=args.render_overlay,
        )
        for video in args.videos
    ]
    manifest = args.out_root / "grounded_manifest.json"
    manifest.write_text(json.dumps({"model": MODEL_ID, "runs": summaries}, indent=2) + "\n", encoding="utf-8")
    for summary in summaries:
        print(
            f"{Path(summary['video']).name}: {summary['touch_events']} touches, "
            f"{summary['rallies']} rallies, {summary['active_intervals']} active intervals",
            flush=True,
        )
        print(f"  events: {summary['events_path']}", flush=True)
        if summary["overlay_path"]:
            print(f"  overlay: {summary['overlay_path']}", flush=True)
    print(f"manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
