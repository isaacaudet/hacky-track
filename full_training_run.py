#!/usr/bin/env python3
"""Run the full Hacky Track training and event-enrichment pass."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from detect_atw_overlay import analyze_video, detect_atw, detect_foot
from hacky_mvp import AudioPeak, detect_audio_peaks, extract_mono_wav
from scan_training_data import video_meta
from train_multimodal_detector import Detection, process_video, train_model


DEFAULT_DOWNLOADS = Path("/Users/isaacaudet/Downloads")
DEFAULT_OUT_ROOT = Path("outputs/full_training_improved")
DEFAULT_MODEL_DIR = Path("models/full_training_improved")
DEFAULT_VIDEOS = tuple(sorted(DEFAULT_DOWNLOADS.glob("video-*_singular_display.MOV")))
REVIEWED_EVENT_FILES = {
    "video-50_singular_display.MOV": Path("data/video-50_singular_display.events.json"),
    "video-506_singular_display.MOV": Path("data/video-506_singular_display.events.json"),
    "video-506_singular_display 2.MOV": Path("data/video-506_singular_display.events.json"),
}


@dataclass
class VideoStatus:
    path: str
    readable: bool
    duration_sec: float = 0.0
    reason: str = ""


def readable_video(path: Path) -> VideoStatus:
    if not path.exists() or path.stat().st_size == 0:
        return VideoStatus(str(path), False, reason="missing_or_zero_bytes")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return VideoStatus(str(path), False, reason="opencv_open_failed")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    ok, _frame = cap.read()
    cap.release()
    if not ok or fps <= 0 or frames <= 0:
        return VideoStatus(str(path), False, reason="no_decodable_frames")
    return VideoStatus(str(path), True, duration_sec=float(frames / fps))


def read_track_csv(path: Path) -> list[Detection]:
    detections: list[Detection] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            x = None if row["x"] == "" else float(row["x"])
            y = None if row["y"] == "" else float(row["y"])
            detections.append(
                Detection(
                    frame_idx=int(row["frame_idx"]),
                    time_sec=float(row["time_sec"]),
                    x=x,
                    y=y,
                    score=float(row["score"]),
                    source=row["source"],
                )
            )
    return detections


def audio_peaks(video: Path, min_z: float, min_gap_sec: float) -> list[AudioPeak]:
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        extract_mono_wav(video, wav)
        return detect_audio_peaks(wav, min_z=min_z, min_gap_sec=min_gap_sec)


def read_frame(video: Path, time_sec: float, out_size: tuple[int, int]) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(time_sec * fps))))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)


def reviewed_event_path(video: Path) -> Path | None:
    path = REVIEWED_EVENT_FILES.get(video.name)
    if path and path.exists():
        return path
    return None


def reviewed_atw_event(video: Path) -> dict[str, Any] | None:
    path = reviewed_event_path(video)
    if not path:
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    for event in doc.get("special_events", []):
        if event.get("type") != "around_the_world":
            continue
        return {
            "type": "around_the_world",
            "start_sec": round(float(event["start_sec"]), 3),
            "end_sec": round(float(event["end_sec"]), 3),
            "completion_sec": round(float(event.get("completion_sec", event["end_sec"])), 3),
            "confidence": 1.0,
            "score": None,
            "angular_span_deg": None,
            "net_angle_deg": None,
            "samples": None,
            "detected_by_motion": False,
            "label": event.get("label", "reviewed around the world"),
            "note": "reviewed_seed_label",
        }
    return None


def reviewed_stall_events(video: Path) -> list[dict[str, Any]]:
    path = reviewed_event_path(video)
    if not path:
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    stalls: list[dict[str, Any]] = []
    for rally in doc.get("rallies", []):
        for event in rally.get("events", []):
            if event.get("type") != "stall":
                continue
            stalls.append(
                {
                    "type": "stall",
                    "time_sec": round(float(event["time_sec"]), 3),
                    "end_sec": round(float(event["time_sec"]) + float(event.get("duration_sec", 0.0)), 3),
                    "duration_sec": round(float(event.get("duration_sec", 0.0)), 3),
                    "confidence": 1.0,
                    "label": event.get("label", "reviewed stall"),
                    "note": "reviewed_seed_label",
                }
            )
    return stalls


def nearest_detection(detections: list[Detection], time_sec: float) -> Detection | None:
    if not detections:
        return None
    idx = min(range(len(detections)), key=lambda item: abs(detections[item].time_sec - time_sec))
    return detections[idx]


def speed_series(detections: list[Detection], out_size: tuple[int, int]) -> np.ndarray:
    times = np.array([item.time_sec for item in detections], dtype=np.float32)
    xs = np.array([np.nan if item.x is None else item.x for item in detections], dtype=np.float32)
    ys = np.array([np.nan if item.y is None else item.y for item in detections], dtype=np.float32)
    valid = np.isfinite(xs) & np.isfinite(ys)
    if valid.sum() < 3:
        return np.zeros(len(detections), dtype=np.float32)
    xs = np.interp(times, times[valid], xs[valid])
    ys = np.interp(times, times[valid], ys[valid])
    dt = max(float(np.median(np.diff(times))), 1 / 30)
    return np.sqrt(np.gradient(xs, dt) ** 2 + np.gradient(ys, dt) ** 2)


def classify_touch_event(
    video: Path,
    event: dict[str, Any],
    detections: list[Detection],
    out_size: tuple[int, int],
) -> dict[str, Any]:
    t = float(event["time_sec"])
    det = nearest_detection(detections, t)
    x = event.get("x")
    y = event.get("y")
    if (x is None or y is None) and det and det.x is not None and det.y is not None:
        x, y = det.x, det.y
    frame = read_frame(video, t + 0.035, out_size)
    ball = None if x is None or y is None else (float(x), float(y), 18.0)
    foot, foot_conf = detect_foot(frame, ball, None) if frame is not None else (None, 0.0)
    foot_distance = None
    if foot is not None and ball is not None:
        foot_distance = math.hypot(float(foot[0]) - ball[0], float(foot[1]) - ball[1])

    y_ratio = 0.0 if y is None else float(y) / out_size[1]
    audio_z = float(event.get("audio_z") or 0.0)
    visual_score = float(event.get("visual_score") or 0.0)
    near_foot = foot_distance is not None and foot_distance <= 175 and foot_conf >= 0.08
    far_from_foot = foot_distance is None or foot_distance >= 250
    likely_ground = (
        audio_z >= 5.5
        and not near_foot
        and far_from_foot
        and (y_ratio >= 0.90 or (y_ratio >= 0.79 and visual_score < 0.68))
    )
    event_type = "drop_floor" if likely_ground else "touch"

    enriched = dict(event)
    enriched.update(
        {
            "type": event_type,
            "foot_x": None if foot is None else round(float(foot[0]), 2),
            "foot_y": None if foot is None else round(float(foot[1]), 2),
            "foot_confidence": round(float(foot_conf), 3),
            "foot_distance": None if foot_distance is None else round(float(foot_distance), 2),
            "foot_contact_score": round(float(min(1.0, (foot_conf / 0.35) * (1.0 if near_foot else 0.45))), 3),
            "ground_score": round(float((0.55 * max(0.0, y_ratio - 0.72) / 0.28) + (0.45 * min(1.0, audio_z / 18.0))), 3),
            "visual_score": round(visual_score, 3),
            "note": "candidate_ground_hit" if likely_ground else "candidate_touch_near_foot" if near_foot else "candidate_touch_needs_review",
        }
    )
    return enriched


def detect_stall_windows(video: Path, detections: list[Detection], out_size: tuple[int, int]) -> list[dict[str, Any]]:
    speeds = speed_series(detections, out_size)
    runs: list[list[tuple[Detection, float, float]]] = []
    current: list[tuple[Detection, float, float]] = []
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    for idx in range(0, len(detections), 4):
        det = detections[idx]
        if det.x is None or det.y is None or det.score < 0.42 or speeds[idx] > 145:
            if current:
                runs.append(current)
                current = []
            continue
        frame = None
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(det.time_sec * fps))))
            ok, raw_frame = cap.read()
            if ok:
                frame = cv2.resize(raw_frame, out_size, interpolation=cv2.INTER_AREA)
        ball = (float(det.x), float(det.y), 18.0)
        foot, foot_conf = detect_foot(frame, ball, None) if frame is not None else (None, 0.0)
        dist = math.inf if foot is None else math.hypot(float(foot[0]) - ball[0], float(foot[1]) - ball[1])
        if foot_conf >= 0.09 and dist <= 125:
            current.append((det, float(dist), float(foot_conf)))
        elif current:
            runs.append(current)
            current = []
    cap.release()
    if current:
        runs.append(current)

    stalls: list[dict[str, Any]] = []
    for run in runs:
        start, end = run[0][0].time_sec, run[-1][0].time_sec
        duration = end - start
        if duration < 0.20:
            continue
        stalls.append(
            {
                "type": "stall",
                "time_sec": round(start, 3),
                "end_sec": round(end, 3),
                "duration_sec": round(duration, 3),
                "confidence": round(float(np.mean([item[2] for item in run])), 3),
                "mean_foot_distance": round(float(np.mean([item[1] for item in run])), 2),
                "label": "candidate foot stall",
            }
        )
    return stalls


def detect_atw_event(video: Path, out_size: tuple[int, int]) -> dict[str, Any] | None:
    reviewed = reviewed_atw_event(video)
    try:
        points, _fps = analyze_video(video, out_size)
        if not points:
            return reviewed
        event = detect_atw(points)
    except Exception as exc:  # noqa: BLE001 - diagnostic output for long batch jobs
        if reviewed:
            reviewed["detector_error"] = str(exc)
            return reviewed
        return {"type": "around_the_world_error", "error": str(exc)}
    if not event.detected or event.score < 1.45:
        return reviewed
    return {
        "type": "around_the_world",
        "start_sec": round(float(event.start_sec), 3),
        "end_sec": round(float(event.end_sec), 3),
        "completion_sec": round(float(event.completion_sec), 3),
        "confidence": round(float(min(1.0, event.score / 3.5)), 3),
        "score": round(float(event.score), 4),
        "angular_span_deg": round(float(event.angular_span_deg), 2),
        "net_angle_deg": round(float(event.net_angle_deg), 2),
        "samples": int(event.samples),
        "detected_by_motion": bool(event.detected),
        "label": "candidate around the world",
    }


def enrich_video_run(video: Path, out_dir: Path, out_size: tuple[int, int]) -> dict[str, Any]:
    trained_doc = json.loads((out_dir / "trained_events.json").read_text(encoding="utf-8"))
    detections = read_track_csv(out_dir / "trained_track.csv")
    enriched_events: list[dict[str, Any]] = []
    for rally in trained_doc.get("rallies", []):
        for event in rally.get("events", []):
            enriched = classify_touch_event(video, event, detections, out_size)
            enriched["rally_id"] = rally.get("id")
            enriched_events.append(enriched)
    reviewed_stalls = reviewed_stall_events(video)
    enriched_events.extend(reviewed_stalls or detect_stall_windows(video, detections, out_size))
    atw = detect_atw_event(video, out_size)
    if atw:
        enriched_events.append(atw)
    enriched_events.sort(key=lambda item: float(item.get("time_sec", item.get("start_sec", 0.0))))

    summary = {
        "touch_candidates": sum(1 for item in enriched_events if item["type"] == "touch"),
        "ground_hit_candidates": sum(1 for item in enriched_events if item["type"] == "drop_floor"),
        "stall_candidates": sum(1 for item in enriched_events if item["type"] == "stall"),
        "around_the_world_candidates": sum(1 for item in enriched_events if item["type"] == "around_the_world"),
    }
    doc = {
        "source_video": video.name,
        "annotation_method": "full_training_improved_patch_tracker_with_foot_ground_stall_atw_postprocessing",
        "video_meta": video_meta(video),
        "summary": summary,
        "events": enriched_events,
    }
    full_json = out_dir / "full_events.json"
    full_json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    write_full_events_csv(enriched_events, out_dir / "full_events.csv")
    return {"full_events_path": str(full_json), "full_csv_path": str(out_dir / "full_events.csv"), **summary}


def write_full_events_csv(events: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "type",
        "rally_id",
        "touch_number",
        "time_sec",
        "end_sec",
        "duration_sec",
        "confidence",
        "audio_z",
        "visual_score",
        "motion_score",
        "ground_score",
        "foot_contact_score",
        "foot_confidence",
        "foot_distance",
        "x",
        "y",
        "foot_x",
        "foot_y",
        "label",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for event in events:
            writer.writerow(event)


def build_markdown_report(manifest: dict[str, Any], path: Path) -> None:
    lines = [
        "# Full Training Run Report",
        "",
        "## Input Inventory",
        "",
        f"- Readable videos: {sum(1 for item in manifest['video_inventory'] if item['readable'])}",
        f"- Skipped videos: {sum(1 for item in manifest['video_inventory'] if not item['readable'])}",
        f"- Model: `{manifest['model_path']}`",
        "",
        "## Per-Video Results",
        "",
        "| Video | Touch | Ground | Stall | ATW | Events |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for run in manifest["runs"]:
        lines.append(
            f"| `{Path(run['video']).name}` | {run['touch_candidates']} | {run['ground_hit_candidates']} | "
            f"{run['stall_candidates']} | {run['around_the_world_candidates']} | `{run['full_events_path']}` |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Ground hits are candidate resets, not final reviewed labels.",
            "- Foot contact uses a shoe/foot proximity heuristic and should be reviewed visually.",
            "- Strong impact recovery uses audio spikes plus track context; it is not accepted as sound-only unless the bag track is in a playable zone.",
            "- Reviewed stall/around-the-world seed labels are used for clips that already have reviewed annotations.",
            "- Trick detection currently only emits around-the-world candidates; broader trick classification still needs labeled examples.",
            "- The zero-byte Downloads clips were skipped and need to be re-exported or downloaded locally before training can use them.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Hacky Track training and event enrichment")
    parser.add_argument("videos", nargs="*", type=Path, default=list(DEFAULT_VIDEOS))
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--width", type=int, default=688)
    parser.add_argument("--height", type=int, default=912)
    parser.add_argument("--patch-size", type=int, default=74)
    parser.add_argument("--audio-min-z", type=float, default=4.8)
    parser.add_argument("--audio-gap-sec", type=float, default=0.24)
    parser.add_argument("--frame-stride", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_size = (args.width, args.height)
    inventory = [readable_video(path) for path in args.videos]
    videos = [Path(item.path) for item in inventory if item.readable]
    if not videos:
        raise SystemExit("No readable training videos found.")

    model, model_path, report = train_model(
        videos,
        args.model_dir,
        out_size=out_size,
        patch_size=args.patch_size,
        checked_video=Path("/Users/isaacaudet/Downloads/video-50_singular_display.MOV"),
        checked_events=Path("data/video-50_singular_display.events.json"),
        checked_anchors=Path("data/video-50_singular_display.paint_anchors.json"),
    )

    runs: list[dict[str, Any]] = []
    for video in videos:
        summary = process_video(
            video,
            model,
            args.out_root,
            out_size=out_size,
            patch_size=args.patch_size,
            audio_min_z=args.audio_min_z,
            audio_gap_sec=args.audio_gap_sec,
            render_overlay=False,
            frame_stride=args.frame_stride,
        )
        full = enrich_video_run(video, Path(summary["events_path"]).parent, out_size)
        runs.append({**summary, **full})
        print(
            f"{video.name}: touch={full['touch_candidates']} ground={full['ground_hit_candidates']} "
            f"stall={full['stall_candidates']} atw={full['around_the_world_candidates']}"
        )

    manifest = {
        "model_path": str(model_path),
        "training_report": report,
        "video_inventory": [asdict(item) for item in inventory],
        "runs": runs,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_root / "full_training_manifest.json"
    report_path = args.out_root / "full_training_report.md"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    build_markdown_report(manifest, report_path)
    print(f"manifest: {manifest_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
