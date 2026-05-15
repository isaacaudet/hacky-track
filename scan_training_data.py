#!/usr/bin/env python3
"""Create candidate touch datasets from raw Meta Glasses footbag clips.

This is not a trained detector. It uses audio transients to produce reviewable
candidate touches, groups them into rough rallies, and writes proof sheets so
bad candidates can be accepted/rejected later.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hacky_mvp import AudioPeak, detect_audio_peaks, extract_mono_wav
from paint_hud import DEFAULT_ASSETS, DEFAULT_ANCHORS, detect_bag_center, generate_assets, render_video


def video_meta(video: Path) -> dict[str, float | int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": frame_count / fps if fps else 0.0,
    }


def group_peaks(peaks: list[AudioPeak], rally_gap_sec: float) -> list[list[AudioPeak]]:
    groups: list[list[AudioPeak]] = []
    current: list[AudioPeak] = []
    for peak in peaks:
        if current and peak.time_sec - current[-1].time_sec > rally_gap_sec:
            groups.append(current)
            current = []
        current.append(peak)
    if current:
        groups.append(current)
    return groups


def candidate_event_doc(
    video: Path,
    peaks: list[AudioPeak],
    *,
    rally_gap_sec: float,
    duration_sec: float,
) -> dict[str, Any]:
    rallies: list[dict[str, Any]] = []
    for rally_idx, group in enumerate(group_peaks(peaks, rally_gap_sec), start=1):
        events: list[dict[str, Any]] = []
        for touch_idx, peak in enumerate(group, start=1):
            events.append(
                {
                    "type": "touch",
                    "touch_number": touch_idx,
                    "time_sec": round(peak.time_sec, 3),
                    "label": "candidate audio transient",
                    "confidence": "candidate",
                    "audio_z": round(peak.z_score, 2),
                    "audio_rms": round(peak.rms, 5),
                    "note": "Auto-generated candidate from audio transient; needs visual review before treating as ground truth.",
                }
            )
        rallies.append(
            {
                "id": rally_idx,
                "label": f"candidate rally {rally_idx}",
                "start_sec": round(max(0.0, group[0].time_sec - 0.35), 3),
                "end_sec": round(min(duration_sec, group[-1].time_sec + 0.85), 3),
                "expected_touches": len(group),
                "expected_stalls": 0,
                "events": events,
            }
        )
    return {
        "source_video": video.name,
        "annotation_method": "Auto-generated review candidates from audio transient peaks; not ground truth.",
        "candidate_settings": {
            "rally_gap_sec": rally_gap_sec,
        },
        "rallies": rallies,
    }


def write_candidate_csv(doc: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rally_id",
                "touch_number",
                "time_sec",
                "audio_z",
                "audio_rms",
                "confidence",
                "label",
                "note",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        for rally in doc["rallies"]:
            for event in rally["events"]:
                row = dict(event)
                row["rally_id"] = rally["id"]
                writer.writerow(row)


def read_frame_at(cap: cv2.VideoCapture, time_sec: float) -> np.ndarray | None:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, round(time_sec * fps)))
    ok, frame = cap.read()
    return frame if ok else None


def write_candidate_sheet(video: Path, doc: dict[str, Any], out_path: Path, *, max_frames: int = 80) -> None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return
    thumbs: list[np.ndarray] = []
    flat_events = [
        (rally["id"], event)
        for rally in doc["rallies"]
        for event in rally["events"]
    ][:max_frames]
    for rally_id, event in flat_events:
        frame = read_frame_at(cap, float(event["time_sec"]) + 0.06)
        if frame is None:
            continue
        frame = cv2.resize(frame, (344, 456), interpolation=cv2.INTER_AREA)
        center = detect_bag_center(frame)
        if center is not None:
            cx, cy = int(round(center[0])), int(round(center[1]))
            cv2.circle(frame, (cx, cy), 18, (0, 255, 255), 2, cv2.LINE_AA)
        label = f"R{rally_id} C{event['touch_number']} {float(event['time_sec']):.2f}s z{float(event['audio_z']):.1f}"
        cv2.putText(frame, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (40, 255, 255), 2, cv2.LINE_AA)
        thumbs.append(frame)
    cap.release()
    if not thumbs:
        return

    cols = 5
    rows = math.ceil(len(thumbs) / cols)
    h, w = thumbs[0].shape[:2]
    sheet = np.full((rows * h, cols * w, 3), 255, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * w
        y = (idx // cols) * h
        sheet[y : y + h, x : x + w] = thumb
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def scan_video(
    video: Path,
    out_root: Path,
    *,
    min_z: float,
    min_gap_sec: float,
    rally_gap_sec: float,
    render_overlay: bool,
    overlay_scale: float,
    max_overlay_seconds: float | None,
) -> dict[str, Any]:
    meta = video_meta(video)
    out_dir = out_root / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        extract_mono_wav(video, wav)
        peaks = detect_audio_peaks(wav, min_z=min_z, min_gap_sec=min_gap_sec)

    doc = candidate_event_doc(video, peaks, rally_gap_sec=rally_gap_sec, duration_sec=float(meta["duration_sec"]))
    doc["candidate_settings"].update({"min_z": min_z, "min_gap_sec": min_gap_sec})
    doc["video_meta"] = meta

    events_path = out_dir / "candidate_events.json"
    events_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    write_candidate_csv(doc, out_dir / "candidate_events.csv")
    write_candidate_sheet(video, doc, out_dir / "candidate_sheet.jpg")

    overlay_path = None
    if render_overlay and peaks:
        generate_assets(DEFAULT_ASSETS)
        overlay_path = render_video(
            video,
            out_dir,
            events_path,
            DEFAULT_ANCHORS.with_name("__missing_candidate_anchors.json"),
            DEFAULT_ASSETS,
            scale=overlay_scale,
            max_seconds=max_overlay_seconds,
        )

    return {
        "video": str(video),
        "duration_sec": round(float(meta["duration_sec"]), 3),
        "candidates": len(peaks),
        "candidate_rallies": len(doc["rallies"]),
        "events_path": str(events_path),
        "csv_path": str(out_dir / "candidate_events.csv"),
        "sheet_path": str(out_dir / "candidate_sheet.jpg"),
        "overlay_path": str(overlay_path) if overlay_path else None,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan raw hacky sack videos into reviewable candidate touch datasets.")
    parser.add_argument("videos", type=Path, nargs="+", help="Input video paths")
    parser.add_argument("--out-root", type=Path, default=Path("outputs"), help="Output root directory")
    parser.add_argument("--min-z", type=float, default=6.0, help="Audio transient z-score threshold")
    parser.add_argument("--min-gap-sec", type=float, default=0.32, help="Minimum gap between audio candidates")
    parser.add_argument("--rally-gap-sec", type=float, default=2.2, help="Gap used to split candidate rallies")
    parser.add_argument("--render-overlay", action="store_true", help="Render a paint HUD overlay using candidate events")
    parser.add_argument("--overlay-scale", type=float, default=0.5, help="Overlay render scale")
    parser.add_argument("--max-overlay-seconds", type=float, default=None, help="Render only the first N seconds of each overlay")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = [
        scan_video(
            video,
            args.out_root,
            min_z=args.min_z,
            min_gap_sec=args.min_gap_sec,
            rally_gap_sec=args.rally_gap_sec,
            render_overlay=args.render_overlay,
            overlay_scale=args.overlay_scale,
            max_overlay_seconds=args.max_overlay_seconds,
        )
        for video in args.videos
    ]
    manifest = args.out_root / "training_candidate_manifest.json"
    manifest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for item in summary:
        print(
            f"{Path(item['video']).name}: {item['candidates']} candidates, "
            f"{item['candidate_rallies']} candidate rallies"
        )
        print(f"  sheet: {item['sheet_path']}")
        if item["overlay_path"]:
            print(f"  overlay: {item['overlay_path']}")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
