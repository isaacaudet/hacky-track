#!/usr/bin/env python3
"""Render video-506 with the existing sketch HUD and corrected ATW labels."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from paint_hud import (
    BLACK,
    CYAN,
    GREEN,
    PINK,
    YELLOW,
    AssetBank,
    Event,
    Rally,
    draw_timeline,
    generate_assets,
    mux_audio,
    overlay_rgba,
)
from detect_atw_overlay import detect_ball


DEFAULT_VIDEO = Path("/Users/isaacaudet/Downloads/video-506_singular_display.MOV")
DEFAULT_ASSETS = Path("assets/ms_paint_hud")
DEFAULT_OUT = Path("outputs/video-506_singular_display")

# Ground-truth correction from the user:
# 19 touches, one stall at about 3s, ATW immediately after through about 4s.
TOUCH_TIMES = [
    1.15,
    1.96,
    4.36,
    4.98,
    5.30,
    5.62,
    6.33,
    7.18,
    8.07,
    8.81,
    9.46,
    10.07,
    10.97,
    11.74,
    12.63,
    13.32,
    14.01,
    14.69,
    15.19,
]
STALL_TIME = 3.00
STALL_DURATION = 0.78
ATW_START = 3.55
ATW_END = 4.28
DROP_FROM_HAND_TIME = 0.86
MOVE_TIMELINE = [
    (DROP_FROM_HAND_TIME, "drop_from_hand"),
    (1.15, "right_kick"),
    (1.96, "left_kick"),
    (STALL_TIME, "right_stall"),
    (ATW_START, "around_the_world"),
    (4.36, "right_kick"),
    (4.98, "right_kick"),
    (5.30, "left_kick"),
    (5.62, "outer_right"),
    (6.33, "outer_right"),
    (7.18, "right_kick"),
    (8.07, "left_kick"),
    (8.81, "right_kick"),
    (9.46, "outer_right"),
    (10.07, "right_kick"),
    (10.97, "outer_left"),
]
MOVE_ASSET_BY_KIND = {
    "ready": "moves/move_ready.png",
    "drop_from_hand": "moves/move_drop_from_hand.png",
    "right_kick": "moves/move_right_kick.png",
    "left_kick": "moves/move_left_kick.png",
    "outer_right": "moves/move_outer_right.png",
    "outer_left": "moves/move_outer_left.png",
    "right_stall": "moves/move_right_stall.png",
    "around_the_world": "moves/move_around_the_world.png",
}


def corrected_event_doc(video: Path) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for index, time_sec in enumerate(TOUCH_TIMES, start=1):
        events.append(
            {
                "type": "touch",
                "touch_number": index,
                "time_sec": time_sec,
                "label": "corrected_touch",
            }
        )
    events.append(
        {
            "type": "stall",
            "time_sec": STALL_TIME,
            "duration_sec": STALL_DURATION,
            "label": "toe stall",
        }
    )
    events.sort(key=lambda item: (float(item["time_sec"]), 0 if item["type"] == "touch" else 1))
    return {
        "source_video": video.name,
        "annotation_method": "user_corrected_hud_ground_truth",
        "rallies": [
            {
                "id": 1,
                "label": "Video 506 corrected rally",
                "start_sec": min(TOUCH_TIMES) - 0.45,
                "end_sec": max(TOUCH_TIMES) + 0.75,
                "expected_touches": len(TOUCH_TIMES),
                "expected_stalls": 1,
                "events": events,
            }
        ],
        "special_events": [
            {
                "type": "around_the_world",
                "start_sec": ATW_START,
                "end_sec": ATW_END,
                "label": "AROUND THE WORLD",
            }
        ],
    }


def write_corrected_event_files(video: Path, out_dir: Path) -> Path:
    doc = corrected_event_doc(video)
    out_dir.mkdir(parents=True, exist_ok=True)
    event_path = out_dir / "video506_corrected_hud_events.json"
    event_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    data_path = Path("data/video-506_singular_display.events.json")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return event_path


def build_rally() -> Rally:
    events: list[Event] = []
    for index, time_sec in enumerate(TOUCH_TIMES, start=1):
        events.append(
            Event(
                rally_id=1,
                rally_label="Video 506 corrected rally",
                type="touch",
                time_sec=time_sec,
                touch_number=index,
                label="corrected_touch",
            )
        )
    events.append(
        Event(
            rally_id=1,
            rally_label="Video 506 corrected rally",
            type="stall",
            time_sec=STALL_TIME,
            duration_sec=STALL_DURATION,
            label="toe stall",
        )
    )
    events.sort(key=lambda item: (item.time_sec, 0 if item.type == "touch" else 1))
    return Rally(
        id=1,
        label="Video 506 corrected rally",
        start_sec=min(TOUCH_TIMES) - 0.45,
        end_sec=max(TOUCH_TIMES) + 0.75,
        expected_touches=len(TOUCH_TIMES),
        expected_stalls=1,
        events=tuple(events),
    )


def state_at(t: float, rally: Rally) -> dict[str, Any]:
    count_t = t + 0.045
    touches = [event for event in rally.events if event.type == "touch"]
    current_touches = sum(1 for event in touches if event.time_sec <= count_t)
    stall_active = STALL_TIME <= t <= STALL_TIME + STALL_DURATION
    atw_active = ATW_START <= t <= ATW_END
    atw_complete = ATW_END <= t <= ATW_END + 1.05
    recent_touch = next((event for event in reversed(touches) if 0 <= t - event.time_sec <= 0.28), None)
    move_t = t + 0.045
    current_move_kind = "ready"
    for move_time, move_kind in MOVE_TIMELINE:
        if move_time <= move_t:
            current_move_kind = move_kind
        else:
            break
    return {
        "current_touches": current_touches,
        "total_touches": current_touches,
        "stall_active": stall_active,
        "atw_active": atw_active,
        "atw_complete": atw_complete,
        "recent_touch": recent_touch,
        "current_move_kind": current_move_kind,
    }


def draw_impact_ticks(frame: np.ndarray, center: tuple[int, int], age: float, scale_ui: float, seed: int) -> None:
    progress = min(1.0, max(0.0, age / 0.22))
    if progress >= 1.0:
        return
    cx, cy = center
    base = int(round((10 + progress * 14) * scale_ui))
    length = int(round((18 - progress * 7) * scale_ui))
    thickness = max(2, int(round((3 - progress) * scale_ui)))
    rng = np.random.default_rng(seed * 1009 + 17)
    rotation = float(rng.uniform(-85, 85))
    ray_count = int(rng.integers(4, 7))
    rays = np.linspace(-62, 74, ray_count) + rotation + rng.uniform(-14, 14, ray_count)
    for idx, degrees in enumerate(rays):
        angle = math.radians(degrees)
        jitter = math.sin(age * 80 + idx * 1.7) * 3 * scale_ui
        x1 = int(round(cx + math.cos(angle) * (base + jitter)))
        y1 = int(round(cy + math.sin(angle) * (base + jitter)))
        x2 = int(round(cx + math.cos(angle) * (base + length + jitter)))
        y2 = int(round(cy + math.sin(angle) * (base + length + jitter)))
        color = (30, 245, 255) if idx % 2 else (255, 255, 255)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness + 4, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def draw_hud_only(
    frame: np.ndarray,
    t: float,
    rally: Rally,
    events: list[Event],
    duration: float,
    assets: AssetBank,
    touch_centers: dict[int, tuple[int, int]],
) -> None:
    h, w = frame.shape[:2]
    scale_ui = w / 688.0

    def s(value: float) -> int:
        return int(round(value * scale_ui))

    state = state_at(t, rally)

    overlay_rgba(frame, assets.rgba("panels/panel_live_counter.png"), s(14), s(16), scale_ui)
    overlay_rgba(frame, assets.rgba("labels/label_live_count.png"), s(32), s(28), scale_ui * 0.78)
    overlay_rgba(frame, assets.digit_string(f"{state['current_touches']}/19", s(68)), s(30), s(66), 1.0)

    pip_y = s(138)
    pip_x = s(36)
    pip_gap = s(18)
    for idx in range(10):
        rel = "pips/pip_on.png" if idx < min(state["current_touches"], 10) else "pips/pip_off.png"
        overlay_rgba(frame, assets.rgba(rel), pip_x + idx * pip_gap, pip_y, scale_ui * 0.68)

    chips = [
        ("RALLY 1/1", YELLOW),
        (f"TOTAL {state['total_touches']}", CYAN),
    ]
    chip_x = s(288)
    chip_y = s(20)
    for text, color in chips:
        chip = assets.chip(text, color, BLACK, s(20))
        if chip_x + chip.shape[1] > w - s(8):
            chip_x = s(288)
            chip_y += s(42)
        overlay_rgba(frame, chip, chip_x, chip_y, 1.0)
        chip_x += int(chip.shape[1] + s(7))
    move_asset = MOVE_ASSET_BY_KIND.get(str(state["current_move_kind"]), MOVE_ASSET_BY_KIND["ready"])
    move = assets.rgba(move_asset)
    move_y = s(58)
    move_x_base = s(288)
    max_move_w = max(1, w - move_x_base - s(12))
    move_scale = min(scale_ui * 0.9, max_move_w / move.shape[1])
    move_scale = max(scale_ui * 0.58, move_scale)
    move_w = int(round(move.shape[1] * move_scale))
    move_x = min(move_x_base, w - move_w - s(8))
    overlay_rgba(frame, move, move_x, move_y, move_scale)

    if state["recent_touch"] is not None:
        age = t - state["recent_touch"].time_sec
        pulse = 1.0 + 0.06 * math.sin(age * 34)
        plus = assets.chip("+1", YELLOW, BLACK, s(36))
        overlay_rgba(frame, plus, s(218), s(88), pulse)
        touch_number = state["recent_touch"].touch_number or 0
        center = touch_centers.get(touch_number)
        if center is not None:
            x = min(max(s(34), center[0] + s(46)), w - s(34))
            y = min(max(s(62), center[1] - s(42)), h - s(74))
            draw_impact_ticks(frame, (x, y), age, scale_ui, touch_number)

    if state["stall_active"] and not state["atw_active"]:
        overlay_rgba(frame, assets.chip("STALL 1", CYAN, BLACK, s(28)), w - s(246), s(218), 1.0)

    if state["atw_active"]:
        overlay_rgba(frame, assets.rgba("badges/badge_around_the_world.png"), w - s(392), s(120), scale_ui * 0.86)
        progress = min(1.0, max(0.0, (t - ATW_START) / max(ATW_END - ATW_START, 0.01)))
        bar_x, bar_y, bar_w, bar_h = w - s(384), s(186), s(300), s(18)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (bar_x + s(3), bar_y + s(3)), (bar_x + int(bar_w * progress), bar_y + bar_h - s(3)), (255, 0, 255), -1)
        star_scale = scale_ui * (0.48 + 0.04 * math.sin((t - ATW_START) * 22))
        overlay_rgba(frame, assets.rgba("effects/effect_atw_star.png"), w - s(140), s(208), star_scale)
    elif state["atw_complete"]:
        overlay_rgba(frame, assets.rgba("badges/badge_atw_clean.png"), w - s(288), s(120), scale_ui * 0.9)
        overlay_rgba(frame, assets.rgba("effects/effect_atw_star.png"), w - s(140), s(184), scale_ui * 0.48)

    draw_timeline(frame, t, events, duration, scale_ui)


def detect_touch_centers(video: Path, events: list[Event], out_size: tuple[int, int], fps: float) -> dict[int, tuple[int, int]]:
    cap = cv2.VideoCapture(str(video))
    centers: dict[int, tuple[int, int]] = {}
    if not cap.isOpened():
        return centers
    previous: tuple[float, float, float] | None = None
    offsets = [0.0, -0.035, 0.035, -0.07, 0.07]
    for event in events:
        if event.type != "touch" or event.touch_number is None:
            continue
        best_ball: tuple[float, float, float] | None = None
        best_conf = 0.0
        for offset in offsets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round((event.time_sec + offset) * fps))))
            ok, frame = cap.read()
            if not ok:
                continue
            if (frame.shape[1], frame.shape[0]) != out_size:
                frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)
            ball, conf = detect_ball(frame, previous)
            if ball is not None and conf > best_conf:
                best_ball = ball
                best_conf = conf
        if best_ball is not None:
            previous = best_ball
            centers[event.touch_number] = (int(round(best_ball[0])), int(round(best_ball[1])))
    cap.release()
    return centers


def render(video: Path, out_dir: Path, asset_dir: Path, scale: float) -> Path:
    generate_assets(asset_dir)
    assets = AssetBank(asset_dir)
    rally = build_rally()
    events = sorted(rally.events, key=lambda item: item.time_sec)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = max(2, int(round(in_w * scale)) // 2 * 2)
    out_h = max(2, int(round(in_h * scale)) // 2 * 2)
    duration = total_frames / fps if total_frames else max(TOUCH_TIMES) + 1.0
    touch_centers = detect_touch_centers(video, events, (out_w, out_h), fps)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "video506_corrected_paint_hud_overlay.mp4"

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "hud_no_audio.mp4"
        writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create writer: {raw_path}")
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != (out_w, out_h):
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            t = frame_idx / fps
            draw_hud_only(frame, t, rally, events, duration, assets, touch_centers)
            writer.write(frame)
            frame_idx += 1
        writer.release()
        cap.release()
        mux_audio(raw_path, video, final_path)
    return final_path


def write_preview(video: Path, overlay: Path, out_dir: Path) -> None:
    cap = cv2.VideoCapture(str(overlay))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    times = [1.18, 2.0, 3.15, 3.65, 4.15, 4.38, 8.8, 12.6, 15.2, 16.1]
    frames: list[np.ndarray] = []
    for time_sec in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(time_sec * fps)))
        ok, frame = cap.read()
        if ok:
            cv2.putText(frame, f"{time_sec:.2f}s", (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, f"{time_sec:.2f}s", (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
            thumb_w = 320
            thumb_h = round(frame.shape[0] * thumb_w / frame.shape[1])
            frames.append(cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        return
    cols = 2
    rows = math.ceil(len(frames) / cols)
    th, tw = frames[0].shape[:2]
    sheet = np.full((rows * th, cols * tw, 3), 255, dtype=np.uint8)
    for idx, frame in enumerate(frames):
        x = (idx % cols) * tw
        y = (idx // cols) * th
        sheet[y : y + th, x : x + tw] = frame
    cv2.imwrite(str(out_dir / "video506_corrected_paint_hud_sheet.jpg"), sheet)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render corrected video-506 with the sketch HUD only")
    parser.add_argument("video", nargs="?", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--scale", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event_path = write_corrected_event_files(args.video, args.out_dir)
    overlay = render(args.video, args.out_dir, args.assets, args.scale)
    write_preview(args.video, overlay, args.out_dir)
    print(f"events: {event_path}")
    print(f"overlay: {overlay}")
    print(f"preview: {args.out_dir / 'video506_corrected_paint_hud_sheet.jpg'}")


if __name__ == "__main__":
    main()
