#!/usr/bin/env python3
"""MVP rally tracker for Meta Glasses footbag footage.

This first pass uses audio onsets as contact candidates and a checked event
file as the source of truth for rally/touch/stall labeling. It produces JSON,
CSV, an event proof sheet, and an annotated video for inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_EVENTS = Path("data/video-50_singular_display.events.json")


@dataclass(frozen=True)
class AudioPeak:
    time_sec: float
    z_score: float
    rms: float


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def extract_mono_wav(video: Path, wav_path: Path) -> None:
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(wav_path),
        ]
    )


def detect_audio_peaks(wav_path: Path, min_z: float = 6.0, min_gap_sec: float = 0.15) -> list[AudioPeak]:
    with wave.open(str(wav_path), "rb") as wav:
        rate = wav.getframerate()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)

    audio = samples.astype(np.float32) / 32768.0
    win = int(rate * 0.025)
    hop = int(rate * 0.010)
    rms: list[float] = []
    for start in range(0, len(audio) - win, hop):
        chunk = audio[start : start + win]
        rms.append(math.sqrt(float(np.mean(chunk * chunk))))

    rms_arr = np.array(rms, dtype=np.float32)
    median = float(np.median(rms_arr))
    mad = float(np.median(np.abs(rms_arr - median))) + 1e-9
    z_scores = (rms_arr - median) / (1.4826 * mad)

    peaks: list[AudioPeak] = []
    for i in range(2, len(z_scores) - 2):
        z = float(z_scores[i])
        is_peak = z >= min_z and z >= float(z_scores[i - 1]) and z >= float(z_scores[i + 1])
        if not is_peak:
            continue
        t = i * hop / rate
        if peaks and t - peaks[-1].time_sec <= min_gap_sec:
            if z > peaks[-1].z_score:
                peaks[-1] = AudioPeak(t, z, float(rms_arr[i]))
            continue
        peaks.append(AudioPeak(t, z, float(rms_arr[i])))
    return peaks


def load_events(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_events(event_doc: dict[str, Any]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for rally in event_doc["rallies"]:
        for event in rally["events"]:
            item = dict(event)
            item["rally_id"] = rally["id"]
            item["rally_label"] = rally["label"]
            flat.append(item)
    return sorted(flat, key=lambda item: item["time_sec"])


def summarize(event_doc: dict[str, Any]) -> dict[str, Any]:
    rallies = []
    for rally in event_doc["rallies"]:
        touches = [e for e in rally["events"] if e["type"] == "touch"]
        stalls = [e for e in rally["events"] if e["type"] == "stall"]
        resets = [e for e in rally["events"] if e["type"] == "drop_floor"]
        rallies.append(
            {
                "id": rally["id"],
                "label": rally["label"],
                "touches": len(touches),
                "stalls": len(stalls),
                "start_sec": rally["start_sec"],
                "end_sec": rally["end_sec"],
                "reset": resets[-1]["time_sec"] if resets else None,
            }
        )
    return {
        "source_video": event_doc["source_video"],
        "total_rallies": len(event_doc["rallies"]),
        "total_touches": sum(r["touches"] for r in rallies),
        "total_stalls": sum(r["stalls"] for r in rallies),
        "rallies": rallies,
    }


def nearest_peak(time_sec: float, peaks: list[AudioPeak]) -> dict[str, float | None]:
    if not peaks:
        return {"audio_peak_sec": None, "audio_delta_sec": None, "audio_z": None}
    peak = min(peaks, key=lambda p: abs(p.time_sec - time_sec))
    return {
        "audio_peak_sec": round(peak.time_sec, 3),
        "audio_delta_sec": round(time_sec - peak.time_sec, 3),
        "audio_z": round(peak.z_score, 1),
    }


def write_json_outputs(event_doc: dict[str, Any], peaks: list[AudioPeak], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = flatten_events(event_doc)
    enriched = []
    for event in flat:
        item = dict(event)
        item.update(nearest_peak(float(event["time_sec"]), peaks))
        enriched.append(item)

    payload = {
        "summary": summarize(event_doc),
        "events": enriched,
        "audio_candidates": [
            {"time_sec": round(p.time_sec, 3), "z_score": round(p.z_score, 2), "rms": round(p.rms, 5)}
            for p in peaks
        ],
    }
    (out_dir / "rally_events.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with (out_dir / "rally_events.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "rally_id",
            "rally_label",
            "type",
            "time_sec",
            "touch_number",
            "label",
            "audio_peak_sec",
            "audio_delta_sec",
            "audio_z",
            "confidence",
            "note",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for event in enriched:
            writer.writerow(event)


def put_label(
    frame: np.ndarray,
    text: str,
    xy: tuple[int, int],
    scale: float,
    fg: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int] = (0, 0, 0),
    thickness: int = 2,
) -> None:
    x, y = xy
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, bg, thickness + 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thickness, cv2.LINE_AA)


def blend_rect(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def rounded_rect(
    target: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    radius: int,
    color: tuple[int, int, int],
    thickness: int = -1,
) -> None:
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(target.shape[1] - 1, x2), min(target.shape[0] - 1, y2)
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if radius == 0:
        cv2.rectangle(target, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        return

    if thickness < 0:
        cv2.rectangle(target, (x1 + radius, y1), (x2 - radius, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(target, (x1, y1 + radius), (x2, y2 - radius), color, -1, cv2.LINE_AA)
        cv2.circle(target, (x1 + radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(target, (x2 - radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(target, (x1 + radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(target, (x2 - radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
        return

    cv2.line(target, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
    cv2.line(target, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
    cv2.line(target, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.line(target, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.ellipse(target, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(target, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(target, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(target, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)


def glass_panel(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    radius: int = 18,
    fill: tuple[int, int, int] = (12, 16, 20),
    alpha: float = 0.62,
    outline: tuple[int, int, int] | None = None,
    outline_alpha: float = 0.9,
) -> None:
    overlay = frame.copy()
    rounded_rect(overlay, x1, y1, x2, y2, radius, fill, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    if outline:
        outline_color = tuple(round(c * outline_alpha + 255 * (1 - outline_alpha)) for c in outline)
        rounded_rect(frame, x1, y1, x2, y2, radius, outline_color, 1)


def draw_bar(
    frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    progress: float,
    color: tuple[int, int, int],
) -> None:
    progress = max(0.0, min(1.0, progress))
    cv2.rectangle(frame, (x, y), (x + width, y + height), (80, 80, 80), -1)
    cv2.rectangle(frame, (x, y), (x + round(width * progress), y + height), color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (230, 230, 230), 1)


def draw_metric_card(
    frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    value: str,
    accent: tuple[int, int, int],
) -> None:
    glass_panel(frame, x, y, x + width, y + height, radius=13, alpha=0.56, outline=accent, outline_alpha=0.55)
    put_label(frame, title, (x + 12, y + 23), 0.42, fg=(190, 200, 206), thickness=1)
    put_label(frame, value, (x + 12, y + height - 16), 0.86, fg=accent, thickness=2)


def draw_touch_pips(
    frame: np.ndarray,
    x: int,
    y: int,
    expected: int,
    current: int,
    accent: tuple[int, int, int],
) -> None:
    if expected <= 0:
        return
    gap = 15
    for i in range(expected):
        color = accent if i < current else (95, 105, 112)
        cv2.circle(frame, (x + i * gap, y), 4, color, -1, cv2.LINE_AA)


def event_color(event_type: str) -> tuple[int, int, int]:
    if event_type == "touch":
        return (70, 210, 255)
    if event_type == "stall":
        return (120, 255, 120)
    if event_type == "drop_floor":
        return (80, 80, 255)
    return (220, 220, 220)


def active_rally_at(event_doc: dict[str, Any], time_sec: float) -> dict[str, Any] | None:
    for rally in event_doc["rallies"]:
        if float(rally["start_sec"]) <= time_sec <= float(rally["end_sec"]):
            return rally
    return None


def count_events_before(events: list[dict[str, Any]], event_type: str, time_sec: float) -> int:
    return sum(1 for event in events if event["type"] == event_type and float(event["time_sec"]) <= time_sec)


def mux_audio(video_without_audio: Path, source_video: Path, final_path: Path) -> Path:
    if shutil.which("ffmpeg"):
        try:
            run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(video_without_audio),
                    "-i",
                    str(source_video),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-c:a",
                    "aac",
                    "-shortest",
                    str(final_path),
                ]
            )
            video_without_audio.unlink(missing_ok=True)
            return final_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    video_without_audio.rename(final_path)
    return final_path


def render_hud_video(video: Path, event_doc: dict[str, Any], out_dir: Path, scale: float = 0.5) -> Path:
    flat = flatten_events(event_doc)
    touch_events = [event for event in flat if event["type"] == "touch"]
    stall_events = [event for event in flat if event["type"] == "stall"]
    reset_events = [event for event in flat if event["type"] == "drop_floor"]

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    out_size = (round(width * scale), round(height * scale))

    silent_path = out_dir / "hud_overlay_silent.mp4"
    final_path = out_dir / "hud_overlay.mp4"
    writer = cv2.VideoWriter(str(silent_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter for HUD overlay")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = frame_idx / fps
        frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        active = active_rally_at(event_doc, t)
        active_events = active["events"] if active else []
        current_touches = count_events_before(active_events, "touch", t) if active else 0
        current_stalls = count_events_before(active_events, "stall", t) if active else 0
        total_touches = count_events_before(touch_events, "touch", t)
        total_stalls = count_events_before(stall_events, "stall", t)
        completed_or_current = []
        for rally in event_doc["rallies"]:
            if float(rally["start_sec"]) <= t:
                completed_or_current.append(count_events_before(rally["events"], "touch", t))
        best_count = max(completed_or_current, default=0)

        recent = [event for event in flat if 0 <= t - float(event["time_sec"]) <= 0.45]
        recent_event = recent[-1] if recent else None
        future = [event for event in flat if float(event["time_sec"]) > t]
        next_event = future[0] if future else None

        accent = (86, 226, 255)
        lime = (135, 255, 135)
        amber = (255, 214, 92)
        coral = (255, 110, 110)

        # Top HUD band.
        blend_rect(frame, 0, 0, w, 132, (5, 7, 10), 0.58)
        glass_panel(frame, 14, 12, 178, 44, radius=16, alpha=0.52, outline=accent, outline_alpha=0.35)
        put_label(frame, "HACKY TRACK", (28, 35), 0.55, fg=(246, 250, 252), thickness=2)
        glass_panel(frame, w - 126, 12, w - 14, 44, radius=16, alpha=0.42, outline=(190, 200, 210), outline_alpha=0.25)
        put_label(frame, f"{t:05.2f}s", (w - 110, 35), 0.50, fg=(220, 226, 232), thickness=1)

        if active:
            rally_value = f"{active['id']}/{len(event_doc['rallies'])}"
            touch_value = f"{current_touches}/{active['expected_touches']}"
            progress = (t - float(active["start_sec"])) / max(float(active["end_sec"]) - float(active["start_sec"]), 0.001)
        else:
            rally_value = "-"
            touch_value = "0/0"
            progress = 0.0

        gap = 9
        card_w = (w - 28 - gap * 3) // 4
        card_y = 56
        draw_metric_card(frame, 14, card_y, card_w, 58, "RALLY", rally_value, accent)
        draw_metric_card(frame, 14 + (card_w + gap), card_y, card_w, 58, "RUN", touch_value, lime)
        draw_metric_card(frame, 14 + (card_w + gap) * 2, card_y, card_w, 58, "TOTAL", str(total_touches), amber)
        draw_metric_card(frame, 14 + (card_w + gap) * 3, card_y, card_w, 58, "BEST", str(best_count), (146, 174, 255))

        # Live counter panel on the right side.
        panel_w = min(214, w - 40)
        panel_x = w - panel_w - 18
        panel_y = 154
        glass_panel(frame, panel_x, panel_y, panel_x + panel_w, panel_y + 190, radius=24, alpha=0.54, outline=amber, outline_alpha=0.65)
        put_label(frame, "LIVE COUNTER", (panel_x + 20, panel_y + 34), 0.48, fg=(222, 230, 236), thickness=1)
        put_label(frame, str(current_touches), (panel_x + 22, panel_y + 128), 2.75, fg=lime, thickness=5)
        expected = active["expected_touches"] if active else 0
        put_label(frame, f"/ {expected}", (panel_x + 128, panel_y + 112), 0.86, fg=(230, 236, 240), thickness=2)
        draw_touch_pips(frame, panel_x + 22, panel_y + 148, expected, current_touches, lime)
        if active and active.get("expected_stalls", 0):
            put_label(frame, f"stall {current_stalls}/{active['expected_stalls']}", (panel_x + 22, panel_y + 174), 0.55, fg=lime, thickness=1)
        elif active:
            put_label(frame, "rally live", (panel_x + 22, panel_y + 174), 0.55, fg=accent, thickness=1)
        else:
            put_label(frame, "waiting", (panel_x + 22, panel_y + 174), 0.55, fg=(200, 206, 212), thickness=1)

        # Event flash.
        if recent_event:
            color = event_color(str(recent_event["type"]))
            if recent_event["type"] == "touch":
                flash = f"TOUCH {recent_event['touch_number']}"
            elif recent_event["type"] == "stall":
                flash = "STALL"
            elif recent_event["type"] == "drop_floor":
                flash = "FLOOR RESET"
            else:
                flash = str(recent_event["type"]).upper()
            flash_x2 = min(w - panel_w - 34, 426)
            glass_panel(frame, 18, 158, flash_x2, 238, radius=22, alpha=0.52, outline=color, outline_alpha=0.45)
            put_label(frame, flash, (36, 212), 1.35, fg=color, thickness=4)
        elif active:
            status = "STALL HOLD" if current_stalls else "IN RALLY"
            glass_panel(frame, 18, 158, min(w - panel_w - 34, 312), 214, radius=20, alpha=0.42, outline=accent, outline_alpha=0.25)
            put_label(frame, status, (36, 195), 0.72, fg=(230, 236, 240), thickness=2)

        if next_event:
            next_delta = float(next_event["time_sec"]) - t
            next_label = "next: "
            if next_event["type"] == "touch":
                next_label += f"R{next_event['rally_id']} touch {next_event['touch_number']}"
            elif next_event["type"] == "drop_floor":
                next_label += "floor reset"
            else:
                next_label += str(next_event["type"])
            put_label(frame, f"{next_label} in {next_delta:.1f}s", (24, h - 96), 0.50, fg=(230, 235, 240), thickness=1)

        # Bottom timeline and rally progress.
        timeline_y = h - 50
        glass_panel(frame, 12, h - 82, w - 12, h - 12, radius=20, alpha=0.50, outline=(210, 218, 226), outline_alpha=0.25)
        draw_bar(frame, 24, h - 70, w - 48, 7, t / max(duration, 0.001), accent)
        if active:
            draw_bar(frame, 24, h - 58, w - 48, 6, progress, lime)
        cv2.line(frame, (24, timeline_y), (w - 24, timeline_y), (218, 226, 232), 1)
        for event in flat:
            x = int(24 + (w - 48) * (float(event["time_sec"]) / max(duration, 0.001)))
            color = event_color(str(event["type"]))
            radius = 5 if event["type"] == "touch" else 7
            cv2.circle(frame, (x, timeline_y), radius, color, -1)
        for event in reset_events:
            x = int(24 + (w - 48) * (float(event["time_sec"]) / max(duration, 0.001)))
            cv2.line(frame, (x, timeline_y - 16), (x, timeline_y + 16), coral, 2)
        cursor_x = int(24 + (w - 48) * t / max(duration, 0.001))
        cv2.circle(frame, (cursor_x, timeline_y), 7, (255, 255, 255), -1)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return mux_audio(silent_path, video, final_path)


def annotate_video(video: Path, event_doc: dict[str, Any], out_dir: Path, scale: float = 0.5) -> Path:
    flat = flatten_events(event_doc)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    out_size = (round(width * scale), round(height * scale))

    silent_path = out_dir / "annotated_silent.mp4"
    final_path = out_dir / "annotated_rallies.mp4"
    writer = cv2.VideoWriter(str(silent_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter for annotated video")

    current_touch_count = 0
    current_rally = None
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]

        active_rally = None
        for rally in event_doc["rallies"]:
            if float(rally["start_sec"]) <= t <= float(rally["end_sec"]):
                active_rally = rally
                break
        if active_rally is not current_rally:
            current_rally = active_rally
            current_touch_count = 0
        if active_rally:
            current_touch_count = sum(
                1
                for event in active_rally["events"]
                if event["type"] == "touch" and float(event["time_sec"]) <= t
            )

        cv2.rectangle(frame, (0, 0), (w, 96), (0, 0, 0), -1)
        put_label(frame, f"{t:05.2f}s", (16, 34), 0.85)
        if active_rally:
            label = f"Rally {active_rally['id']}: touches {current_touch_count}/{active_rally['expected_touches']}"
            if active_rally.get("expected_stalls"):
                stalls_done = sum(
                    1
                    for event in active_rally["events"]
                    if event["type"] == "stall" and float(event["time_sec"]) <= t
                )
                label += f" | stalls {stalls_done}/{active_rally['expected_stalls']}"
            put_label(frame, label, (16, 76), 0.75, fg=(70, 210, 255))
        else:
            put_label(frame, "No active rally", (16, 76), 0.75, fg=(210, 210, 210))

        nearby = [event for event in flat if abs(float(event["time_sec"]) - t) <= 0.18]
        for i, event in enumerate(nearby[:3]):
            color = event_color(str(event["type"]))
            if event["type"] == "touch":
                msg = f"R{event['rally_id']} TOUCH {event['touch_number']}"
            elif event["type"] == "stall":
                msg = f"R{event['rally_id']} STALL"
            elif event["type"] == "drop_floor":
                msg = f"R{event['rally_id']} FLOOR RESET"
            else:
                msg = f"R{event['rally_id']} {event['type']}"
            put_label(frame, msg, (16, 150 + i * 44), 0.9, fg=color)

        # Bottom event timeline.
        y = h - 36
        cv2.line(frame, (18, y), (w - 18, y), (230, 230, 230), 2)
        for event in flat:
            x = int(18 + (w - 36) * (float(event["time_sec"]) / max(duration, 0.001)))
            color = event_color(str(event["type"]))
            cv2.circle(frame, (x, y), 5 if event["type"] == "touch" else 7, color, -1)
        cv2.circle(frame, (int(18 + (w - 36) * t / max(duration, 0.001)), y), 7, (255, 255, 255), -1)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    return mux_audio(silent_path, video, final_path)


def write_event_sheet(video: Path, event_doc: dict[str, Any], out_dir: Path) -> Path:
    flat = flatten_events(event_doc)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    thumbs = []
    for event in flat:
        frame_idx = max(0, min(len(frames) - 1, round(float(event["time_sec"]) * fps)))
        frame = frames[frame_idx].copy()
        if event["type"] == "touch":
            title = f"R{event['rally_id']} touch {event['touch_number']} @ {event['time_sec']:.2f}s"
        elif event["type"] == "stall":
            title = f"R{event['rally_id']} stall @ {event['time_sec']:.2f}s"
        else:
            title = f"R{event['rally_id']} {event['type']} @ {event['time_sec']:.2f}s"
        color = event_color(str(event["type"]))
        put_label(frame, title, (20, 58), 1.35, fg=color, thickness=3)
        thumb_w = 344
        thumb_h = round(frame.shape[0] * thumb_w / frame.shape[1])
        thumbs.append(cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA))

    cols = 4
    rows = math.ceil(len(thumbs) / cols)
    th, tw = thumbs[0].shape[:2]
    sheet = np.full((rows * th, cols * tw, 3), 255, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = thumb

    path = out_dir / "event_sheet.jpg"
    cv2.imwrite(str(path), sheet)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Track footbag rally events in a Meta Glasses video.")
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS, help="Checked event annotation JSON")
    parser.add_argument("--out", type=Path, default=Path("outputs/video-50_singular_display"), help="Output directory")
    parser.add_argument("--scale", type=float, default=0.5, help="Annotated video scale factor")
    args = parser.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(args.video)
    if not args.events.exists():
        raise FileNotFoundError(args.events)
    args.out.mkdir(parents=True, exist_ok=True)

    event_doc = load_events(args.events)
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"
        extract_mono_wav(args.video, wav_path)
        peaks = detect_audio_peaks(wav_path)

    write_json_outputs(event_doc, peaks, args.out)
    sheet = write_event_sheet(args.video, event_doc, args.out)
    annotated = annotate_video(args.video, event_doc, args.out, scale=args.scale)
    hud = render_hud_video(args.video, event_doc, args.out, scale=args.scale)

    summary = summarize(event_doc)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out / 'rally_events.json'}")
    print(f"wrote {args.out / 'rally_events.csv'}")
    print(f"wrote {sheet}")
    print(f"wrote {annotated}")
    print(f"wrote {hud}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
