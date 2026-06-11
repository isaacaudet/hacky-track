#!/usr/bin/env python3
"""Render the best QA rally with the MS Paint-style sprite HUD."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
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
    overlay_rgba,
)


DEFAULT_MANIFEST = Path("outputs/full_training_27_qa/qa_manifest.json")
DEFAULT_ASSETS = Path("assets/ms_paint_hud")
DEFAULT_OUT_DIR = Path("outputs/best_rally_hud")
QA_SIZE = (688, 912)
ROOT = Path(__file__).resolve().parent
STRICT_AMBIGUOUS_CONTACT_TYPES = {"unknown_contact", "foot_candidate", "knee_candidate", "ground_candidate"}


@dataclass
class BestRally:
    video: Path
    qa_path: Path
    source_doc: dict[str, Any]
    rally: dict[str, Any]
    events: list[dict[str, Any]]
    selection_notes: dict[str, Any]


def resolve_existing_path(raw: Any, manifest_path: Path) -> Path:
    path = Path(str(raw or ""))
    if path.exists():
        return path
    root_path = ROOT / path
    if root_path.exists():
        return root_path
    manifest_relative = manifest_path.parent / path
    if manifest_relative.exists():
        return manifest_relative
    return path


def resolve_video_path(raw: Any, source_video: str) -> Path:
    path = Path(str(raw or ""))
    if path.exists():
        return path
    root_path = ROOT / path
    if root_path.exists():
        return root_path
    downloads = Path.home() / "Downloads" / Path(source_video).name
    if downloads.exists():
        return downloads
    return path


def event_time(event: dict[str, Any]) -> float:
    return float(event.get("time_sec", event.get("start_sec", 0.0)))


def rally_selection_notes(events: list[dict[str, Any]], rally: dict[str, Any] | None = None) -> dict[str, Any]:
    touches = [item for item in events if item.get("type") == "touch"]
    touch_times = [event_time(item) for item in touches]
    airtime_gaps = [round(touch_times[idx] - touch_times[idx - 1], 3) for idx in range(1, len(touch_times))]
    hidden_drop_gaps = [gap for gap in airtime_gaps if gap > 1.65]
    contact_events = [item for item in events if item.get("type") in {"touch", "stall", "drop_floor"}]
    explicit_drops = [item for item in contact_events if item.get("type") == "drop_floor"]
    terminal_drop = bool(contact_events and contact_events[-1].get("type") == "drop_floor")
    terminal_drop_events = [contact_events[-1]] if terminal_drop else []
    nonterminal_drops = explicit_drops[:-1] if terminal_drop else explicit_drops
    low_ball = [item for item in events if item.get("qa_ball_accuracy") == "low"]
    ambiguous_contacts = [
        item
        for item in touches
        if item.get("contact_type") in STRICT_AMBIGUOUS_CONTACT_TYPES
    ]
    ended_by_gap_without_floor_reset = bool((rally or {}).get("ended_by_gap_without_floor_reset"))
    complete = (
        len(touches) >= 6
        and not nonterminal_drops
        and not hidden_drop_gaps
        and not low_ball
        and not ambiguous_contacts
        and not ended_by_gap_without_floor_reset
    )
    return {
        "strict_complete": complete,
        "touches": len(touches),
        "explicit_drop_floor_events": len(explicit_drops),
        "terminal_drop_floor_events": len(terminal_drop_events),
        "nonterminal_drop_floor_events": len(nonterminal_drops),
        "max_airtime_gap_sec": None if not airtime_gaps else max(airtime_gaps),
        "hidden_drop_gap_count": len(hidden_drop_gaps),
        "low_ball_accuracy_events": len(low_ball),
        "ambiguous_contact_events": len(ambiguous_contacts),
        "ended_by_gap_without_floor_reset": ended_by_gap_without_floor_reset,
        "end_reason": (rally or {}).get("end_reason"),
        "next_contact_gap_sec": (rally or {}).get("next_contact_gap_sec"),
    }


def load_best_rally(manifest_path: Path, strict_complete: bool = True) -> BestRally:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    best: tuple[float, int, float, str, dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]] | None = None
    fallback: tuple[float, int, float, str, dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]] | None = None
    for run in manifest["runs"]:
        qa_path = resolve_existing_path(run["qa_events_path"], manifest_path)
        doc = json.loads(qa_path.read_text(encoding="utf-8"))
        for rally in doc.get("rallies", []):
            events = [item for item in doc.get("events", []) if int(item.get("qa_rally_id") or -1) == int(rally["id"])]
            notes = rally_selection_notes(events, rally)
            score = float(rally.get("quality_score") or 0.0)
            touches = int(rally.get("touches") or 0)
            duration = float(rally.get("duration_sec") or 0.0)
            key = (score, touches, duration, doc["source_video"])
            candidate = (score, touches, duration, doc["source_video"], run, doc, rally, events, notes)
            if fallback is None or key > fallback[:4]:
                fallback = candidate
            if strict_complete and not notes["strict_complete"]:
                continue
            if best is None or key > best[:4]:
                best = candidate
    if best is None:
        if strict_complete:
            raise RuntimeError("No strict complete rallies found in QA manifest")
        if fallback is None:
            raise RuntimeError("No rallies found in QA manifest")
        best = fallback
    _score, _touches, _duration, _source, run, doc, rally, events, notes = best
    qa_path = resolve_existing_path(run["qa_events_path"], manifest_path)
    video = resolve_video_path(run.get("video"), str(doc.get("source_video") or Path(str(run.get("video") or "")).name))
    return BestRally(video, qa_path, doc, rally, sorted(events, key=event_time), notes)


def rel_event(event: dict[str, Any], segment_start: float) -> dict[str, Any]:
    item = dict(event)
    if "time_sec" in item and item["time_sec"] is not None:
        item["time_sec"] = round(float(item["time_sec"]) - segment_start, 3)
    if "start_sec" in item and item["start_sec"] is not None:
        item["start_sec"] = round(float(item["start_sec"]) - segment_start, 3)
    if "end_sec" in item and item["end_sec"] is not None:
        item["end_sec"] = round(float(item["end_sec"]) - segment_start, 3)
    return item


def build_render_model(best: BestRally, pad_before: float, pad_after: float) -> tuple[float, float, Rally, list[Event], list[dict[str, Any]]]:
    start_abs = max(0.0, float(best.rally["start_sec"]) - pad_before)
    render_events = [item for item in best.events if item.get("type") != "drop_floor"]
    render_end = max((event_time(item) if item.get("type") != "stall" else float(item.get("end_sec", event_time(item))) for item in render_events), default=float(best.rally["end_sec"]))
    end_abs = render_end + pad_after
    rel_events = [rel_event(item, start_abs) for item in render_events]
    paint_events: list[Event] = []
    touch_number = 0
    for item in rel_events:
        event_type = str(item.get("type"))
        if event_type not in {"touch", "stall", "drop_floor"}:
            continue
        if event_type == "touch":
            touch_number += 1
            number = touch_number
        else:
            number = None
        duration = 0.0
        if event_type == "stall":
            duration = max(0.12, float(item.get("duration_sec") or 0.0))
            if item.get("end_sec") is not None:
                duration = max(duration, float(item["end_sec"]) - float(item["time_sec"]))
        paint_events.append(
            Event(
                rally_id=1,
                rally_label=f"Best rally R{best.rally['id']}",
                type=event_type,
                time_sec=float(item["time_sec"]),
                touch_number=number,
                duration_sec=duration,
                label=str(item.get("contact_type") or item.get("label") or event_type),
            )
        )
    paint_events.sort(key=lambda item: (item.time_sec, 0 if item.type == "touch" else 1))
    rally = Rally(
        id=1,
        label=f"Best rally from {best.video.name}",
        start_sec=max(0.0, float(best.rally["start_sec"]) - start_abs),
        end_sec=float(best.rally["end_sec"]) - start_abs,
        expected_touches=int(best.rally.get("touches") or sum(1 for item in rel_events if item.get("type") == "touch")),
        expected_stalls=int(best.rally.get("stalls") or sum(1 for item in rel_events if item.get("type") == "stall")),
        events=tuple(paint_events),
    )
    return start_abs, end_abs, rally, paint_events, rel_events


def center_map(rel_events: list[dict[str, Any]], out_size: tuple[int, int]) -> dict[int, tuple[int, int]]:
    out_w, out_h = out_size
    sx = out_w / QA_SIZE[0]
    sy = out_h / QA_SIZE[1]
    centers: dict[int, tuple[int, int]] = {}
    idx = 0
    for item in rel_events:
        if item.get("type") != "touch":
            continue
        idx += 1
        x = item.get("qa_ball_x", item.get("x"))
        y = item.get("qa_ball_y", item.get("y"))
        if x is None or y is None:
            continue
        centers[idx] = (int(round(float(x) * sx)), int(round(float(y) * sy)))
    return centers


def current_state(t: float, rally: Rally, events: list[Event], rel_events: list[dict[str, Any]]) -> dict[str, Any]:
    count_t = t + 0.045
    touches_done = sum(1 for item in events if item.type == "touch" and item.time_sec <= count_t)
    stalls_done = sum(1 for item in events if item.type == "stall" and item.time_sec <= count_t)
    active_stall = next((item for item in events if item.type == "stall" and item.time_sec <= t <= item.time_sec + max(item.duration_sec, 0.18)), None)
    recent_touch = next((item for item in reversed(events) if item.type == "touch" and 0 <= t - item.time_sec <= 0.42), None)
    recent_stall = next((item for item in reversed(events) if item.type == "stall" and 0 <= t - item.time_sec <= 0.50), None)
    current_detail = None
    for item in rel_events:
        if item.get("type") == "touch" and float(item["time_sec"]) <= count_t:
            current_detail = item
    return {
        "touches_done": touches_done,
        "stalls_done": stalls_done,
        "active_stall": active_stall,
        "recent_touch": recent_touch,
        "recent_stall": recent_stall,
        "current_detail": current_detail,
    }


def draw_impact_ticks(frame: np.ndarray, center: tuple[int, int], age: float, scale_ui: float, seed: int) -> None:
    progress = min(1.0, max(0.0, age / 0.24))
    if progress >= 1.0:
        return
    cx, cy = center
    rng = np.random.default_rng(seed * 7919 + 43)
    # Offset the burst away from the ball so it reads as an accent instead of a target.
    cx += int(round(float(rng.choice([-1, 1])) * (34 + 10 * progress) * scale_ui))
    cy -= int(round((22 + 8 * progress) * scale_ui))
    ray_count = int(rng.integers(4, 7))
    base = int(round((8 + progress * 9) * scale_ui))
    length = int(round((18 - progress * 7) * scale_ui))
    thickness = max(2, int(round((3 - progress) * scale_ui)))
    for idx in range(ray_count):
        degrees = float(rng.uniform(-145, 35) + idx * rng.uniform(18, 34))
        angle = math.radians(degrees)
        x1 = int(round(cx + math.cos(angle) * base))
        y1 = int(round(cy + math.sin(angle) * base))
        x2 = int(round(cx + math.cos(angle) * (base + length)))
        y2 = int(round(cy + math.sin(angle) * (base + length)))
        color = (30, 245, 255) if idx % 2 else (255, 255, 255)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness + 4, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def move_asset_for(detail: dict[str, Any] | None, active_stall: Event | None) -> str | None:
    if active_stall is not None:
        return "moves/move_right_stall.png"
    if not detail:
        return "moves/move_ready.png"
    side = str(detail.get("contact_side") or "")
    contact_type = str(detail.get("contact_type") or "")
    if "knee" in contact_type:
        return None
    if side == "left":
        return "moves/move_left_kick.png"
    if side == "right":
        return "moves/move_right_kick.png"
    return "labels/label_touch.png"


def draw_best_hud(
    frame: np.ndarray,
    t: float,
    rally: Rally,
    paint_events: list[Event],
    rel_events: list[dict[str, Any]],
    rally_meta: dict[str, Any],
    assets: AssetBank,
    centers: dict[int, tuple[int, int]],
    duration: float,
) -> None:
    h, w = frame.shape[:2]
    scale_ui = w / 688.0

    def s(value: float) -> int:
        return int(round(value * scale_ui))

    state = current_state(t, rally, paint_events, rel_events)

    # Primary counter panel.
    overlay_rgba(frame, assets.rgba("panels/panel_live_counter.png"), s(14), s(16), scale_ui)
    overlay_rgba(frame, assets.rgba("labels/label_live_count.png"), s(32), s(27), scale_ui * 0.74)
    count_img = assets.digit_string(f"{state['touches_done']}/{rally.expected_touches}", s(50))
    count_x, count_y = s(30), s(67)
    plus_x = s(216)
    max_count_w = max(1, plus_x - count_x - s(10))
    count_scale = min(1.0, max_count_w / max(1, count_img.shape[1]))
    overlay_rgba(frame, count_img, count_x, count_y, count_scale)
    plus = assets.rgba("badges/badge_plus_one.png")
    if state["recent_touch"] is not None:
        age = t - state["recent_touch"].time_sec
        plus_scale = scale_ui * (0.49 + 0.035 * math.sin(age * 32))
        overlay_rgba(frame, plus, plus_x, s(80), plus_scale)

    # Pips are two rows so 16 touches does not crowd the panel.
    pip_x, pip_y = s(36), s(136)
    pip_gap = s(18)
    for idx in range(rally.expected_touches):
        rel = "pips/pip_on.png" if idx < state["touches_done"] else "pips/pip_off.png"
        x = pip_x + (idx % 8) * pip_gap
        y = pip_y + (idx // 8) * s(17)
        overlay_rgba(frame, assets.rgba(rel), x, y, scale_ui * 0.58)

    # Top stat chips.
    chips = [
        (f"BEST RALLY", YELLOW),
        (f"GRADE {rally_meta.get('quality_grade', '?')}", PINK),
        (f"SCORE {float(rally_meta.get('quality_score') or 0):.0f}", CYAN),
        (f"STALL {state['stalls_done']}/{rally.expected_stalls}", GREEN),
    ]
    chip_x, chip_y = s(284), s(18)
    for text, color in chips:
        chip = assets.chip(text, color, BLACK, s(19))
        if chip_x + chip.shape[1] > w - s(10):
            chip_x = s(284)
            chip_y += s(38)
        overlay_rgba(frame, chip, chip_x, chip_y, 1.0)
        chip_x += chip.shape[1] + s(7)

    # Current move label from the sprite sheet where possible.
    move_asset = move_asset_for(state["current_detail"], state["active_stall"])
    move_y = s(82)
    if move_asset is not None:
        move = assets.rgba(move_asset)
        max_w = w - s(292) - s(16)
        move_scale = min(scale_ui * 0.72, max_w / max(1, move.shape[1]))
        overlay_rgba(frame, move, s(288), move_y, max(scale_ui * 0.48, move_scale))
    elif state["current_detail"]:
        label = assets.chip("KNEE?", PINK, BLACK, s(22))
        overlay_rgba(frame, label, s(290), move_y, 1.0)

    # Airtime and rally duration live readout.
    detail = state["current_detail"]
    if detail is not None:
        air = detail.get("airtime_since_prev_sec")
        air_text = "AIR --" if air is None else f"AIR {float(air):.2f}s"
        side = str(detail.get("contact_side") or "?").upper()
        overlay_rgba(frame, assets.chip(f"{side}  {air_text}", CYAN, BLACK, s(18)), s(290), s(124), 1.0)
    elapsed = min(duration, max(0.0, t))
    overlay_rgba(frame, assets.chip(f"TIME {elapsed:04.1f}/{duration:04.1f}", YELLOW, BLACK, s(17)), s(290), s(160), 1.0)

    # Recent touch accents. Use corrected QA ball centers, offset away from the bag.
    if state["recent_touch"] is not None:
        touch_num = state["recent_touch"].touch_number or 0
        center = centers.get(touch_num)
        if center is not None:
            draw_impact_ticks(frame, center, t - state["recent_touch"].time_sec, scale_ui, touch_num)

    if state["active_stall"] is not None:
        overlay_rgba(frame, assets.rgba("badges/badge_stall.png"), w - s(166), s(214), scale_ui * 0.68)
    elif state["recent_stall"] is not None:
        overlay_rgba(frame, assets.rgba("badges/badge_stall.png"), w - s(152), s(222), scale_ui * 0.50)

    draw_timeline(frame, t, paint_events, duration, scale_ui)


def mux_segment_audio(raw_video: Path, source_video: Path, output_video: Path, start_sec: float) -> None:
    if not shutil.which("ffmpeg"):
        shutil.copyfile(raw_video, output_video)
        return
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(raw_video),
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            str(source_video),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_video),
        ],
        check=True,
    )


def render(best: BestRally, out_dir: Path, asset_dir: Path, scale: float, pad_before: float, pad_after: float) -> tuple[Path, Path, Path]:
    generate_assets(asset_dir)
    assets = AssetBank(asset_dir)
    start_abs, end_abs, rally, paint_events, rel_events = build_render_model(best, pad_before, pad_after)
    duration = max(0.1, end_abs - start_abs)

    cap = cv2.VideoCapture(str(best.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {best.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = max(2, int(round(in_w * scale)) // 2 * 2)
    out_h = max(2, int(round(in_h * scale)) // 2 * 2)
    centers = center_map(rel_events, (out_w, out_h))

    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "best_rally_sprite_hud_overlay.mp4"
    preview_path = out_dir / "best_rally_sprite_hud_preview.jpg"
    summary_path = out_dir / "best_rally_sprite_hud_summary.json"
    summary = {
        "source_video": str(best.video),
        "qa_events": str(best.qa_path),
        "selected_rally": best.rally,
        "selection_notes": best.selection_notes,
        "segment_start_sec": round(start_abs, 3),
        "segment_end_sec": round(end_abs, 3),
        "segment_duration_sec": round(duration, 3),
        "output_video": str(final_path),
        "preview": str(preview_path),
        "hud_assets": str(asset_dir),
        "events": rel_events,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "best_rally_hud_no_audio.mp4"
        writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create writer: {raw_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(start_abs * fps))))
        frame_idx = 0
        max_frames = int(math.ceil(duration * fps))
        while frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != (out_w, out_h):
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            t = frame_idx / fps
            draw_best_hud(frame, t, rally, paint_events, rel_events, best.rally, assets, centers, duration)
            writer.write(frame)
            frame_idx += 1
        writer.release()
        cap.release()
        mux_segment_audio(raw_path, best.video, final_path, start_abs)

    write_preview(final_path, preview_path)
    return final_path, preview_path, summary_path


def write_preview(video: Path, out_path: Path) -> None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames_count / fps if fps > 0 else 0.0
    times = np.linspace(0.6, max(0.7, duration - 0.6), 8)
    frames: list[np.ndarray] = []
    for time_sec in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(float(time_sec) * fps)))
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.putText(frame, f"{float(time_sec):.1f}s", (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, f"{float(time_sec):.1f}s", (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
        thumb_w = 344
        thumb_h = round(frame.shape[0] * thumb_w / frame.shape[1])
        frames.append(cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        return
    cols = 4
    rows = math.ceil(len(frames) / cols)
    th, tw = frames[0].shape[:2]
    sheet = np.full((rows * th, cols * tw, 3), 255, dtype=np.uint8)
    for idx, frame in enumerate(frames):
        x = (idx % cols) * tw
        y = (idx // cols) * th
        sheet[y : y + th, x : x + tw] = frame
    cv2.imwrite(str(out_path), sheet)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the best QA rally with the sprite HUD")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--pad-before", type=float, default=0.18)
    parser.add_argument("--pad-after", type=float, default=0.28)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best = load_best_rally(args.manifest, strict_complete=not args.allow_incomplete)
    overlay, preview, summary = render(best, args.out_dir, args.assets, args.scale, args.pad_before, args.pad_after)
    print(f"selected: {best.video.name} rally {best.rally['id']} score {best.rally['quality_score']}")
    print(f"selection: {best.selection_notes}")
    print(f"overlay: {overlay}")
    print(f"preview: {preview}")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
