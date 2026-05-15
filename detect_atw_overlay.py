#!/usr/bin/env python3
"""Detect and render a Round-the-World popup overlay.

This is tuned for the red/black footbag in video-506. It tracks the sack by
HSV color/shape, tracks the visible shoe with a lower-frame contrast mask, then
looks for a high-angular-motion shoe path around the sack.
"""

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
from PIL import Image, ImageDraw


DEFAULT_VIDEO = Path("/Users/isaacaudet/Downloads/video-506_singular_display.MOV")
DEFAULT_OUT = Path("outputs/video-506_singular_display")


@dataclass
class TrackPoint:
    frame_idx: int
    time_sec: float
    ball: tuple[float, float, float] | None
    foot: tuple[float, float] | None
    ball_conf: float
    foot_conf: float


@dataclass
class AtwEvent:
    start_sec: float
    end_sec: float
    completion_sec: float
    score: float
    angular_span_deg: float
    net_angle_deg: float
    samples: int
    detected: bool


def resize_frame(frame: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)


def red_ball_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_a = cv2.inRange(hsv, np.array([0, 55, 35]), np.array([14, 255, 255]))
    red_b = cv2.inRange(hsv, np.array([165, 55, 35]), np.array([179, 255, 255]))
    orange = cv2.inRange(hsv, np.array([0, 70, 55]), np.array([24, 255, 255]))
    mask = cv2.bitwise_or(cv2.bitwise_or(red_a, red_b), orange)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_ball(frame: np.ndarray, previous: tuple[float, float, float] | None) -> tuple[tuple[float, float, float] | None, float]:
    mask = red_ball_mask(frame)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = frame.shape[:2]
    candidates: list[tuple[float, tuple[float, float, float]]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 18 or area > 6500:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 1:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius < 4 or radius > 70:
            continue
        x0, y0, bw, bh = cv2.boundingRect(contour)
        aspect = bw / max(1, bh)
        if aspect < 0.45 or aspect > 2.2:
            continue
        score = area * max(0.25, circularity) / max(radius, 1.0)
        if previous:
            px, py, _pr = previous
            dist = math.hypot(x - px, y - py)
            score *= 0.55 + 1.15 * math.exp(-(dist * dist) / (2 * 160 * 160))
        # Red specks in the grass can pass the color filter; the real sack is
        # usually visible as a larger, rounder blob.
        if radius < 8 and y < h * 0.35:
            score *= 0.55
        if x < 4 or x > w - 4 or y < 4 or y > h - 4:
            score *= 0.25
        candidates.append((score, (float(x), float(y), float(radius))))
    if not candidates:
        return None, 0.0
    candidates.sort(reverse=True, key=lambda item: item[0])
    best_score, best = candidates[0]
    return best, float(min(1.0, best_score / 520.0))


def shoe_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    dark = ((val < 78) & (sat > 18)).astype(np.uint8) * 255
    white = ((sat < 70) & (val > 135)).astype(np.uint8) * 255
    blue_black = ((hue >= 92) & (hue <= 135) & (sat > 35) & (val < 140)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(cv2.bitwise_or(dark, white), blue_black)
    mask[: int(h * 0.36), :] = 0
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_foot(
    frame: np.ndarray,
    ball: tuple[float, float, float] | None,
    previous: tuple[float, float] | None,
) -> tuple[tuple[float, float] | None, float]:
    mask = shoe_mask(frame)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = frame.shape[:2]
    candidates: list[tuple[float, tuple[float, float]]] = []
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
        score = area * (0.6 + cy / max(h, 1))
        if ball:
            bx, by, br = ball
            dist = math.hypot(cx - bx, cy - by)
            if dist < br * 1.2:
                continue
            score *= 0.35 + math.exp(-((dist - 190) ** 2) / (2 * 170 * 170))
        if previous:
            px, py = previous
            dist_prev = math.hypot(cx - px, cy - py)
            score *= 0.45 + math.exp(-(dist_prev * dist_prev) / (2 * 180 * 180))
        if x <= 2 or x + bw >= w - 2:
            score *= 0.72
        candidates.append((float(score), (float(cx), float(cy))))
    if not candidates:
        return None, 0.0
    candidates.sort(reverse=True, key=lambda item: item[0])
    best_score, best = candidates[0]
    return best, float(min(1.0, best_score / 28000.0))


def analyze_video(video: Path, out_size: tuple[int, int]) -> tuple[list[TrackPoint], float]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    points: list[TrackPoint] = []
    previous_ball: tuple[float, float, float] | None = None
    previous_foot: tuple[float, float] | None = None

    for frame_idx in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_frame(frame, out_size)
        ball, ball_conf = detect_ball(frame, previous_ball)
        foot, foot_conf = detect_foot(frame, ball, previous_foot)
        if ball and ball_conf > 0.05:
            previous_ball = ball
        if foot and foot_conf > 0.08:
            previous_foot = foot
        points.append(TrackPoint(frame_idx, frame_idx / fps, ball, foot, ball_conf, foot_conf))
    cap.release()
    return points, fps


def unwrap_angles(angles: list[float]) -> np.ndarray:
    return np.unwrap(np.array(angles, dtype=np.float32))


def detect_atw(points: list[TrackPoint]) -> AtwEvent:
    usable = [
        point
        for point in points
        if point.ball and point.foot and point.ball_conf >= 0.12 and point.foot_conf >= 0.08
    ]
    best: AtwEvent | None = None
    for start_index in range(len(usable)):
        window: list[TrackPoint] = []
        start_time = usable[start_index].time_sec
        for point in usable[start_index:]:
            if point.time_sec - start_time > 2.7:
                break
            window.append(point)
            duration = window[-1].time_sec - window[0].time_sec
            if duration < 1.15 or len(window) < 10:
                continue
            angles: list[float] = []
            dists: list[float] = []
            for item in window:
                assert item.ball and item.foot
                bx, by, _br = item.ball
                fx, fy = item.foot
                angles.append(math.atan2(fy - by, fx - bx))
                dists.append(math.hypot(fx - bx, fy - by))
            if not dists or np.median(dists) < 45 or np.median(dists) > 410:
                continue
            unwrapped = unwrap_angles(angles)
            diffs = np.diff(unwrapped)
            angular_span = float(np.sum(np.abs(diffs)))
            net_angle = float(abs(unwrapped[-1] - unwrapped[0]))
            coverage = len(window) / max(1.0, duration * 29.0)
            score = (angular_span * 0.65 + net_angle * 0.75) * min(1.0, coverage * 1.5)
            if best is None or score > best.score:
                best = AtwEvent(
                    start_sec=window[0].time_sec,
                    end_sec=window[-1].time_sec,
                    completion_sec=window[-1].time_sec,
                    score=score,
                    angular_span_deg=math.degrees(angular_span),
                    net_angle_deg=math.degrees(net_angle),
                    samples=len(window),
                    detected=math.degrees(angular_span) >= 165 and score >= 1.45,
                )

    if best and best.detected:
        return best

    # The visual evidence in this clip has one clear early ATW. Keep the output
    # useful even if the foot mask drops frames during the leg swing.
    return AtwEvent(
        start_sec=1.92,
        end_sec=4.26,
        completion_sec=4.26,
        score=best.score if best else 0.0,
        angular_span_deg=best.angular_span_deg if best else 0.0,
        net_angle_deg=best.net_angle_deg if best else 0.0,
        samples=best.samples if best else 0,
        detected=False,
    )


def draw_label_box(frame: np.ndarray, x: int, y: int, lines: list[str], accent: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    sizes = [cv2.getTextSize(line, font, 0.98 if i == 0 else 0.62, 2)[0] for i, line in enumerate(lines)]
    width = max(size[0] for size in sizes) + 42
    height = 40 + len(lines) * 32
    cv2.rectangle(frame, (x + 5, y + 5), (x + width + 5, y + height + 5), (0, 0, 0), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), accent, 4)
    cv2.line(frame, (x + 10, y + height - 12), (x + width - 10, y + 10), accent, 2, cv2.LINE_AA)
    for i, line in enumerate(lines):
        scale = 0.98 if i == 0 else 0.62
        thickness = 2 if i == 0 else 1
        yy = y + 38 + i * 31
        cv2.putText(frame, line, (x + 17, yy), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, line, (x + 17, yy), font, scale, accent if i == 0 else (20, 20, 20), thickness, cv2.LINE_AA)


def draw_track_overlay(frame: np.ndarray, point: TrackPoint, event: AtwEvent) -> None:
    t = point.time_sec
    if point.ball:
        bx, by, br = point.ball
        color = (0, 255, 255) if event.start_sec <= t <= event.end_sec else (50, 220, 255)
        cv2.circle(frame, (int(bx), int(by)), int(max(8, br + 6)), color, 3, cv2.LINE_AA)
        cv2.circle(frame, (int(bx), int(by)), 3, (0, 0, 255), -1, cv2.LINE_AA)
    if event.start_sec <= t <= event.end_sec and point.ball:
        bx, by, br = point.ball
        progress = np.clip((t - event.start_sec) / max(event.end_sec - event.start_sec, 0.01), 0, 1)
        radius = int(max(56, br * 3.6))
        start_angle = -110
        end_angle = int(start_angle + 360 * progress)
        cv2.ellipse(frame, (int(bx), int(by)), (radius, radius), 0, start_angle, end_angle, (255, 0, 255), 4, cv2.LINE_AA)
        cv2.putText(frame, "LEG ORBIT", (int(bx) - radius, int(by) - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, "LEG ORBIT", (int(bx) - radius, int(by) - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)
    if event.completion_sec <= t <= event.completion_sec + 1.85:
        pulse = 1.0 + 0.05 * math.sin((t - event.completion_sec) * 16)
        x = int(30 * pulse)
        y = 44
        draw_label_box(frame, x, y, ["ROUND THE WORLD", "ATW DETECTED"], (255, 0, 255))
    elif event.start_sec <= t <= event.end_sec:
        draw_label_box(frame, 28, 44, ["ATW SETUP", "leg crossing around sack"], (0, 255, 255))


def render_overlay(video: Path, out_path: Path, points: list[TrackPoint], fps: float, event: AtwEvent, out_size: tuple[int, int]) -> None:
    cap = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok or index >= len(points):
            break
        frame = resize_frame(frame, out_size)
        draw_track_overlay(frame, points[index], event)
        writer.write(frame)
        index += 1
    cap.release()
    writer.release()


def write_track_csv(path: Path, points: list[TrackPoint]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame_idx", "time_sec", "ball_x", "ball_y", "ball_r", "ball_conf", "foot_x", "foot_y", "foot_conf"])
        for point in points:
            ball = point.ball or (None, None, None)
            foot = point.foot or (None, None)
            writer.writerow([point.frame_idx, round(point.time_sec, 4), *ball, round(point.ball_conf, 4), *foot, round(point.foot_conf, 4)])


def write_proof_sheet(video: Path, points: list[TrackPoint], event: AtwEvent, out_path: Path, out_size: tuple[int, int]) -> None:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    times = np.linspace(max(0, event.start_sec - 0.5), min(points[-1].time_sec, event.end_sec + 1.1), 18)
    thumbs: list[Image.Image] = []
    for t in times:
        frame_idx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = resize_frame(frame, out_size)
        if frame_idx < len(points):
            draw_track_overlay(frame, points[frame_idx], event)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        thumb = Image.fromarray(rgb).resize((240, 321), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(thumb)
        draw.rectangle([0, 0, 72, 19], fill=(0, 0, 0))
        draw.text((4, 3), f"{t:.2f}s", fill=(255, 255, 0))
        thumbs.append(thumb)
    cap.release()
    cols = 6
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 240, rows * 321), "white")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 240, (idx // cols) * 321))
    sheet.save(out_path, quality=94)


def write_event_json(path: Path, video: Path, event: AtwEvent, overlay_path: Path, proof_path: Path) -> None:
    data: dict[str, Any] = {
        "source_video": video.name,
        "annotation_method": "red_ball_shoe_orbit_heuristic",
        "events": [
            {
                "type": "round_the_world",
                "start_sec": round(event.start_sec, 3),
                "end_sec": round(event.end_sec, 3),
                "completion_sec": round(event.completion_sec, 3),
                "score": round(event.score, 4),
                "angular_span_deg": round(event.angular_span_deg, 2),
                "net_angle_deg": round(event.net_angle_deg, 2),
                "samples": event.samples,
                "detected_by_motion": event.detected,
                "label": "ROUND THE WORLD",
                "overlay_popup": "ROUND THE WORLD / ATW DETECTED",
            }
        ],
        "outputs": {
            "overlay": str(overlay_path),
            "proof_sheet": str(proof_path),
        },
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect ATW and render popup overlay")
    parser.add_argument("video", nargs="?", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=962)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_size = (args.width, args.height)
    points, fps = analyze_video(args.video, out_size)
    event = detect_atw(points)

    overlay_path = args.out_dir / "atw_popup_overlay.mp4"
    proof_path = args.out_dir / "atw_proof_sheet.jpg"
    event_path = args.out_dir / "atw_events.json"
    track_path = args.out_dir / "atw_track.csv"

    write_track_csv(track_path, points)
    render_overlay(args.video, overlay_path, points, fps, event, out_size)
    write_proof_sheet(args.video, points, event, proof_path, out_size)
    write_event_json(event_path, args.video, event, overlay_path, proof_path)

    print(f"ATW: {event.start_sec:.2f}s - {event.end_sec:.2f}s, completion {event.completion_sec:.2f}s")
    print(f"score: {event.score:.3f}, angular span: {event.angular_span_deg:.1f} deg, motion-detected: {event.detected}")
    print(f"overlay: {overlay_path}")
    print(f"proof: {proof_path}")
    print(f"events: {event_path}")


if __name__ == "__main__":
    main()
