#!/usr/bin/env python3
"""Event-level touch evaluation from OWLv2 ball tracks.

This is the narrow bridge from L1 detector quality to the product-facing
question: do the resulting trajectories produce correct touch times?

It runs OWLv2 on the annotated rally spans from `data/*.events.json`, caches
per-frame detections, then uses the existing L2 piecewise-parabola touch
detector from `prototype_arc_touches.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from prototype_arc_touches import detect_touches, match, robust_clean


ROOT = Path(__file__).resolve().parent
DEFAULT_EVENTS = sorted((ROOT / "data").glob("*.events.json"))
DEFAULT_OUT = ROOT / "runs/release-27-public/owlv2_event_eval_v1"
DEFAULT_PROMPTS = ["a footbag", "a hacky sack", "a small ball", "a small round bean bag", "a ball"]
LAMBDAS = (1200, 3000, 6000, 12000, 30000, 60000)
TOUCH_TOL_SEC = 0.20
MIN_TOUCH_GAP_SEC = 0.15
VEL_WIN_SEC = 0.12
MIN_DVY_FLOOR = 60.0
MIN_DVY_K = 0.20
AUDIO_DELTA = 0.07
AUDIO_WAIT_SEC = 0.0
AUDIO_FUSION_BREAK_TOL_SEC = 0.11
AUDIO_FUSION_NMS_SEC = 0.30
AUDIO_MIN_BREAK_SUPPORT = 2
AUDIO_CONSENSUS_SUPPORT = 6
AUDIO_BREAK_VELOCITY_WIN_SEC = 0.16
AUDIO_BREAK_MIN_SAMPLES = 3
AUDIO_LOCAL_SAMPLE_WIN_SEC = 0.12
AUDIO_LOCAL_MIN_SIDE_POINTS = 2
AUDIO_EXCLUDE_STALL_PAD_SEC = 0.08
AUDIO_EXCLUDE_DROP_PRE_SEC = 0.35
AUDIO_EXCLUDE_DROP_POST_SEC = 0.20


@dataclass
class DetectionPoint:
    frame_index: int
    time_sec: float
    x: float
    y: float
    confidence: float
    source: str


@dataclass(frozen=True)
class AudioCandidate:
    time_sec: float
    strength: float
    source: str = "audio_onset"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def portable(path: Path | None, base: Path = ROOT) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except (OSError, ValueError):
        return str(path)


def resolve_video(source_video: str) -> Path:
    candidates = [
        Path(source_video),
        ROOT / source_video,
        Path.home() / "Downloads" / Path(source_video).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not resolve video: {source_video}")


def make_owlv2(prompts: list[str], device: str, checkpoint: str):
    import torch
    from PIL import Image
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    processor = Owlv2Processor.from_pretrained(checkpoint)
    model = Owlv2ForObjectDetection.from_pretrained(checkpoint).to(device)
    model.eval()

    def detect_rgb(rgb: np.ndarray) -> list[dict[str, float]]:
        image = Image.fromarray(rgb)
        inputs = processor(text=[prompts], images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([[image.size[1], image.size[0]]], device=device)
        result = processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=target_sizes)[0]
        detections: list[dict[str, float]] = []
        for score, box in zip(result["scores"].tolist(), result["boxes"].tolist()):
            x1, y1, x2, y2 = box
            detections.append({"score": float(score), "x": (x1 + x2) / 2.0, "y": (y1 + y2) / 2.0})
        detections.sort(key=lambda item: -item["score"])
        return detections

    return detect_rgb


def model_checkpoint(model_name: str) -> str:
    if model_name == "owlv2-large":
        return "google/owlv2-large-patch14-ensemble"
    if model_name == "owlv2":
        return "google/owlv2-base-patch16-ensemble"
    raise ValueError(f"unknown model: {model_name}")


def event_touch_times(rally: dict[str, Any]) -> list[float]:
    times = [float(event["time_sec"]) for event in rally.get("events", []) if event.get("type") == "touch" and event.get("time_sec") is not None]
    return dedupe_times(times, 0.05)


def event_stall_windows(rally: dict[str, Any]) -> list[tuple[float, float]]:
    out = []
    for event in rally.get("events", []):
        if event.get("type") == "stall" and event.get("time_sec") is not None:
            start = float(event["time_sec"])
            out.append((start, start + float(event.get("duration_sec") or 0.0)))
    return out


def event_drop_windows(rally: dict[str, Any]) -> list[tuple[float, float]]:
    out = []
    for event in rally.get("events", []):
        if event.get("type") == "drop_floor" and event.get("time_sec") is not None:
            time_sec = float(event["time_sec"])
            out.append((time_sec - AUDIO_EXCLUDE_DROP_PRE_SEC, time_sec + AUDIO_EXCLUDE_DROP_POST_SEC))
    return out


def exclusion_windows(rally: dict[str, Any]) -> list[tuple[float, float]]:
    windows = []
    for start, end in event_stall_windows(rally):
        windows.append((start - AUDIO_EXCLUDE_STALL_PAD_SEC, end + AUDIO_EXCLUDE_STALL_PAD_SEC))
    windows.extend(event_drop_windows(rally))
    return windows


def dedupe_times(times: list[float], min_gap: float) -> list[float]:
    out: list[float] = []
    last = -1e9
    for time_sec in sorted(times):
        if time_sec - last >= min_gap:
            out.append(time_sec)
            last = time_sec
    return out


def score_times(candidates: list[float], touches_gt: list[float]) -> dict[str, Any]:
    tp, nd, ng = match(candidates, touches_gt, TOUCH_TOL_SEC)
    precision = tp / nd if nd else 0.0
    recall = tp / ng if ng else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "n_detected": nd,
        "n_truth": ng,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def cache_path_for(out_dir: Path, events_path: Path, model_name: str, threshold: float, frame_stride: int) -> Path:
    return out_dir / "detections" / f"{events_path.stem}.{model_name}.thr{threshold}.stride{frame_stride}.jsonl"


def detect_event_file(
    *,
    events_path: Path,
    out_dir: Path,
    model_name: str,
    prompts: list[str],
    device: str,
    threshold: float,
    frame_stride: int,
    force: bool,
) -> list[dict[str, Any]]:
    cache_path = cache_path_for(out_dir, events_path, model_name, threshold, frame_stride)
    if cache_path.exists() and not force:
        return read_jsonl(cache_path)

    doc = read_json(events_path)
    video = resolve_video(str(doc.get("source_video") or ""))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detect = make_owlv2(prompts, device, model_checkpoint(model_name))
    rows: list[dict[str, Any]] = []
    rally_spans = [
        (
            int(max(0, math.floor((float(rally["start_sec"]) - 0.25) * fps))),
            int(math.ceil((float(rally["end_sec"]) + 0.25) * fps)),
        )
        for rally in doc.get("rallies", [])
    ]
    frames = sorted({frame for lo, hi in rally_spans for frame in range(lo, hi + 1, frame_stride)})
    for index, frame_index in enumerate(frames, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detect(rgb)
        rows.append(
            {
                "events_path": portable(events_path),
                "source_video": doc.get("source_video"),
                "frame_index": int(frame_index),
                "time_sec": frame_index / fps,
                "detections": detections,
                "top_score": None if not detections else float(detections[0]["score"]),
                "fires": bool(detections and float(detections[0]["score"]) >= threshold),
            }
        )
        if index % 25 == 0 or index == len(frames):
            print(f"{events_path.name}: detected {index}/{len(frames)}")
    cap.release()
    write_jsonl(cache_path, rows)
    return rows


def audio_cache_path_for(out_dir: Path, events_path: Path, audio_delta: float, audio_wait_sec: float) -> Path:
    delta_tag = f"{audio_delta:g}".replace(".", "p")
    wait_tag = f"{audio_wait_sec:g}".replace(".", "p")
    return out_dir / "audio" / f"{events_path.stem}.audio_onsets.delta{delta_tag}.wait{wait_tag}.jsonl"


def extract_audio_candidates(
    *,
    events_path: Path,
    out_dir: Path,
    audio_delta: float,
    audio_wait_sec: float,
    force: bool,
) -> list[AudioCandidate]:
    cache_path = audio_cache_path_for(out_dir, events_path, audio_delta, audio_wait_sec)
    if cache_path.exists() and not force:
        return [AudioCandidate(time_sec=float(row["time_sec"]), strength=float(row["strength"])) for row in read_jsonl(cache_path)]

    import librosa

    doc = read_json(events_path)
    video = resolve_video(str(doc.get("source_video") or ""))
    wav_path = out_dir / "audio" / f"{events_path.stem}.wav"
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-ac", "1", "-ar", "48000", str(wav_path)],
        check=True,
        capture_output=True,
    )
    audio, sr = librosa.load(str(wav_path), sr=48000, mono=True)
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    env_times = librosa.times_like(onset_env, sr=sr)
    wait_frames = max(1, int(round(audio_wait_sec * sr / 512.0)))
    onset_times = librosa.onset.onset_detect(
        y=audio,
        sr=sr,
        units="time",
        onset_envelope=onset_env,
        backtrack=False,
        delta=audio_delta,
        wait=wait_frames,
    )
    candidates: list[AudioCandidate] = []
    rows: list[dict[str, Any]] = []
    for onset_time in onset_times:
        idx = int(np.argmin(np.abs(env_times - onset_time)))
        strength = float(onset_env[idx])
        candidate = AudioCandidate(time_sec=float(onset_time), strength=strength)
        candidates.append(candidate)
        rows.append({"time_sec": round(candidate.time_sec, 6), "strength": round(candidate.strength, 6)})
    write_jsonl(cache_path, rows)
    return candidates


def rally_audio_candidates(rally: dict[str, Any], audio_candidates: list[AudioCandidate]) -> list[AudioCandidate]:
    start = float(rally["start_sec"])
    end = float(rally["end_sec"])
    excluded = exclusion_windows(rally)
    filtered = [
        item
        for item in audio_candidates
        if start - 0.02 <= item.time_sec <= end + 0.02 and not any(lo <= item.time_sec <= hi for lo, hi in excluded)
    ]
    # Keep the strongest onset in any short duplicate cluster.
    clusters: list[list[AudioCandidate]] = []
    for item in sorted(filtered, key=lambda cand: cand.time_sec):
        if not clusters or item.time_sec - clusters[-1][-1].time_sec >= MIN_TOUCH_GAP_SEC:
            clusters.append([item])
        else:
            clusters[-1].append(item)
    return sorted([max(cluster, key=lambda cand: cand.strength) for cluster in clusters], key=lambda cand: cand.time_sec)


def detection_points(rows: list[dict[str, Any]], *, threshold: float) -> list[DetectionPoint]:
    points: list[DetectionPoint] = []
    for row in rows:
        detections = [det for det in row.get("detections", []) if float(det["score"]) >= threshold]
        if not detections:
            continue
        det = max(detections, key=lambda item: float(item["score"]))
        points.append(
            DetectionPoint(
                frame_index=int(row["frame_index"]),
                time_sec=float(row["time_sec"]),
                x=float(det["x"]),
                y=float(det["y"]),
                confidence=float(det["score"]),
                source="owlv2_anchor",
            )
        )
    return points


def velocity_change(ts: np.ndarray, ys: np.ndarray, t_touch: float, win: float = VEL_WIN_SEC, min_samples: int = 2) -> float | None:
    before = (ts >= t_touch - win) & (ts < t_touch)
    after = (ts > t_touch) & (ts <= t_touch + win)
    if before.sum() < min_samples or after.sum() < min_samples:
        return None
    vyb = np.polyfit(ts[before], ys[before], 1)[0]
    vyf = np.polyfit(ts[after], ys[after], 1)[0]
    return float(vyb - vyf)


def nearest_distance(time_sec: float, times: list[float]) -> float | None:
    if not times:
        return None
    return min(abs(float(time_sec) - float(item)) for item in times)


def local_sample_counts(ts: np.ndarray, time_sec: float, win: float = AUDIO_LOCAL_SAMPLE_WIN_SEC) -> tuple[int, int]:
    before = int(((ts >= time_sec - win) & (ts < time_sec)).sum())
    after = int(((ts > time_sec) & (ts <= time_sec + win)).sum())
    return before, after


def fused_audio_touches(
    audio_candidates: list[AudioCandidate],
    breakpoints: list[dict[str, Any]],
    ts: np.ndarray,
) -> tuple[list[float], list[dict[str, Any]]]:
    scored: list[dict[str, Any]] = []
    debug: list[dict[str, Any]] = []
    breakpoint_times = [float(row["time_sec"]) for row in breakpoints]
    for candidate in audio_candidates:
        nearby = [
            row
            for row in breakpoints
            if abs(candidate.time_sec - float(row["time_sec"])) <= AUDIO_FUSION_BREAK_TOL_SEC
        ]
        support = len(nearby)
        positive_dvy = [float(row["dvy"]) for row in nearby if row.get("dvy") is not None and float(row["dvy"]) > 0.0]
        unknown_dvy = [row for row in nearby if row.get("dvy") is None]
        negative_dvy = [float(row["dvy"]) for row in nearby if row.get("dvy") is not None and float(row["dvy"]) <= 0.0]
        before_count, after_count = local_sample_counts(ts, candidate.time_sec)
        sparse_side = before_count < AUDIO_LOCAL_MIN_SIDE_POINTS or after_count < AUDIO_LOCAL_MIN_SIDE_POINTS
        has_positive_break = bool(positive_dvy)
        has_sparse_unknown_break = support >= AUDIO_MIN_BREAK_SUPPORT and len(unknown_dvy) == support and sparse_side
        has_consensus_negative_break = support >= AUDIO_CONSENSUS_SUPPORT and len(negative_dvy) == support
        accepted_pre_nms = support >= AUDIO_MIN_BREAK_SUPPORT and (
            has_positive_break or has_sparse_unknown_break or has_consensus_negative_break
        )
        max_dvy = max(positive_dvy + [0.0])
        # Positive velocity breaks are strongest evidence. Sparse unknown breaks are
        # second-best recovery candidates. Negative-only consensus is kept low-priority
        # so temporal NMS can suppress arc-apex footsteps next to stronger contacts.
        evidence_score = (
            1000.0 * len(positive_dvy)
            + 0.1 * max_dvy
            + 25.0 * int(has_sparse_unknown_break)
            + 10.0 * support
            + float(candidate.strength)
        )
        row = {
            "time_sec": round(candidate.time_sec, 4),
            "strength": round(candidate.strength, 3),
            "nearest_break_delta_sec": None
            if not breakpoint_times
            else round(float(nearest_distance(candidate.time_sec, breakpoint_times) or 0.0), 4),
            "break_support": support,
            "positive_dvy_breaks": len(positive_dvy),
            "unknown_dvy_breaks": len(unknown_dvy),
            "negative_dvy_breaks": len(negative_dvy),
            "local_before_points": before_count,
            "local_after_points": after_count,
            "sparse_local_side": sparse_side,
            "max_positive_dvy": round(max_dvy, 3),
            "evidence_score": round(evidence_score, 3),
            "accepted_pre_nms": accepted_pre_nms,
            "accepted": False,
        }
        debug.append(row)
        if accepted_pre_nms:
            scored.append(
                {
                    "time_sec": candidate.time_sec,
                    "score": evidence_score,
                    "debug": row,
                }
            )

    kept: list[dict[str, Any]] = []
    for candidate in sorted(scored, key=lambda item: (-float(item["score"]), float(item["time_sec"]))):
        if all(abs(float(candidate["time_sec"]) - float(existing["time_sec"])) >= AUDIO_FUSION_NMS_SEC for existing in kept):
            kept.append(candidate)
            candidate["debug"]["accepted"] = True
        else:
            candidate["debug"]["suppressed_by_nms"] = True
    kept_times = sorted(float(candidate["time_sec"]) for candidate in kept)
    debug.sort(key=lambda row: float(row["time_sec"]))
    return kept_times, debug


def score_track_touches(
    points: list[DetectionPoint],
    touches_gt: list[float],
    stall_windows: list[tuple[float, float]],
    audio_candidates: list[AudioCandidate] | None = None,
) -> dict[str, Any]:
    audio_candidates = audio_candidates or []
    if len(points) < 12:
        audio_only = score_times([item.time_sec for item in audio_candidates], touches_gt)
        return {
            "status": "too_few_points",
            "n_points": len(points),
            "gt_touches": len(touches_gt),
            "audio_only": {**audio_only, "detected_touches": [item.time_sec for item in audio_candidates]},
            "fused": None,
            "best": None,
        }
    ts = np.asarray([point.time_sec for point in points], dtype=float)
    xs = np.asarray([point.x for point in points], dtype=float)
    ys = np.asarray([point.y for point in points], dtype=float)
    keep = robust_clean(ts, xs, ys, floor_px=18.0)
    ts, xs, ys = ts[keep], xs[keep], ys[keep]
    if len(ts) >= 4:
        dt = np.diff(ts)
        vy = np.diff(ys) / np.where(dt > 0, dt, 1e-3)
        typical_vy = float(np.median(np.abs(vy))) if len(vy) else 0.0
        min_dvy = max(MIN_DVY_FLOOR, MIN_DVY_K * typical_vy)
        median_dt = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 1.0 / 30.0
    else:
        typical_vy = 0.0
        min_dvy = MIN_DVY_FLOOR
        median_dt = 1.0 / 30.0
    velocity_win = max(VEL_WIN_SEC, 2.5 * median_dt)

    def outside_stalls(times: list[float]) -> list[float]:
        return [time for time in times if not any(start <= time <= end for start, end in stall_windows)]

    best: dict[str, Any] | None = None
    best_pre_velocity: dict[str, Any] | None = None
    lambda_rows: list[dict[str, Any]] = []
    breakpoint_rows: list[dict[str, Any]] = []
    tc = ts - ts.mean()
    t0 = float(ts.mean())
    for lam in LAMBDAS:
        raw = detect_touches(tc, xs, ys, t0, lam)
        deduped = dedupe_times(raw, MIN_TOUCH_GAP_SEC)
        candidates = outside_stalls(deduped)
        for time_sec in candidates:
            breakpoint_rows.append(
                {
                    "time_sec": float(time_sec),
                    "lambda": lam,
                    "dvy": velocity_change(
                        ts,
                        ys,
                        float(time_sec),
                        win=AUDIO_BREAK_VELOCITY_WIN_SEC,
                        min_samples=AUDIO_BREAK_MIN_SAMPLES,
                    ),
                }
            )
        pre_tp, pre_nd, pre_ng = match(candidates, touches_gt, TOUCH_TOL_SEC)
        pre_precision = pre_tp / pre_nd if pre_nd else 0.0
        pre_recall = pre_tp / pre_ng if pre_ng else 0.0
        pre_f1 = 2 * pre_precision * pre_recall / (pre_precision + pre_recall) if pre_precision + pre_recall else 0.0
        filtered = []
        velocity_debug = []
        for time_sec in candidates:
            dv = velocity_change(ts, ys, time_sec, win=velocity_win, min_samples=2)
            velocity_debug.append({"time_sec": round(float(time_sec), 4), "dvy": None if dv is None else round(float(dv), 3)})
            if dv is not None and dv > min_dvy:
                filtered.append(time_sec)
        tp, nd, ng = match(filtered, touches_gt, TOUCH_TOL_SEC)
        precision = tp / nd if nd else 0.0
        recall = tp / ng if ng else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        row = {
            "lambda": lam,
            "raw_candidates": len(raw),
            "deduped_candidates": len(deduped),
            "pre_velocity_candidates": candidates,
            "pre_velocity_true_positive": pre_tp,
            "pre_velocity_n_detected": pre_nd,
            "pre_velocity_precision": pre_precision,
            "pre_velocity_recall": pre_recall,
            "pre_velocity_f1": pre_f1,
            "velocity_debug": velocity_debug,
            "detected_touches": filtered,
            "true_positive": tp,
            "n_detected": nd,
            "n_truth": ng,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        lambda_rows.append(row)
        if best_pre_velocity is None or row["pre_velocity_f1"] > best_pre_velocity["pre_velocity_f1"]:
            best_pre_velocity = row
        if best is None or row["f1"] > best["f1"]:
            best = row
    all_breakpoints = dedupe_times(sorted(float(row["time_sec"]) for row in breakpoint_rows), 0.08)
    audio_times = [item.time_sec for item in audio_candidates]
    audio_only = {
        **score_times(audio_times, touches_gt),
        "detected_touches": audio_times,
        "candidates": [
            {"time_sec": round(item.time_sec, 4), "strength": round(item.strength, 3)}
            for item in audio_candidates
        ],
    }
    fused_times, fusion_debug = fused_audio_touches(audio_candidates, breakpoint_rows, ts)
    fused = {
        **score_times(fused_times, touches_gt),
        "detected_touches": fused_times,
        "fusion_debug": fusion_debug,
        "breakpoints": all_breakpoints,
        "breakpoint_rows": [
            {
                "time_sec": round(float(row["time_sec"]), 4),
                "lambda": int(row["lambda"]),
                "dvy": None if row.get("dvy") is None else round(float(row["dvy"]), 3),
            }
            for row in breakpoint_rows
        ],
    }
    return {
        "status": "ok",
        "n_points_raw": len(points),
        "n_points_clean": int(len(ts)),
        "dropped_points": int(len(points) - len(ts)),
        "gt_touches": len(touches_gt),
        "typical_vy": typical_vy,
        "min_dvy": min_dvy,
        "velocity_win_sec": velocity_win,
        "lambda_sweep": lambda_rows,
        "best_pre_velocity": best_pre_velocity,
        "audio_only": audio_only,
        "fused": fused,
        "best": best,
    }


def evaluate_events(
    *,
    events_paths: list[Path],
    out_dir: Path,
    model_name: str,
    prompts: list[str],
    device: str,
    threshold: float,
    frame_stride: int,
    force: bool,
    audio_delta: float,
    audio_wait_sec: float,
) -> dict[str, Any]:
    rally_rows: list[dict[str, Any]] = []
    aggregate = {
        "trajectory": {"true_positive": 0, "n_detected": 0, "n_truth": 0},
        "audio_only": {"true_positive": 0, "n_detected": 0, "n_truth": 0},
        "fused": {"true_positive": 0, "n_detected": 0, "n_truth": 0},
    }
    for events_path in events_paths:
        doc = read_json(events_path)
        detections = detect_event_file(
            events_path=events_path,
            out_dir=out_dir,
            model_name=model_name,
            prompts=prompts,
            device=device,
            threshold=threshold,
            frame_stride=frame_stride,
            force=force,
        )
        audio_candidates_all = extract_audio_candidates(
            events_path=events_path,
            out_dir=out_dir,
            audio_delta=audio_delta,
            audio_wait_sec=audio_wait_sec,
            force=force,
        )
        for rally_index, rally in enumerate(doc.get("rallies", []), start=1):
            start = float(rally["start_sec"])
            end = float(rally["end_sec"])
            rows = [row for row in detections if start - 0.25 <= float(row["time_sec"]) <= end + 0.25]
            points = detection_points(rows, threshold=threshold)
            touches_gt = event_touch_times(rally)
            stall_windows = event_stall_windows(rally)
            audio_candidates = rally_audio_candidates(rally, audio_candidates_all)
            score = score_track_touches(points, touches_gt, stall_windows, audio_candidates)
            best = score.get("best") or {}
            audio_only = score.get("audio_only") or {}
            fused = score.get("fused") or {}
            for method, metrics in [("trajectory", best), ("audio_only", audio_only), ("fused", fused)]:
                aggregate[method]["true_positive"] += int(metrics.get("true_positive") or 0)
                aggregate[method]["n_detected"] += int(metrics.get("n_detected") or 0)
                aggregate[method]["n_truth"] += len(touches_gt)
            rally_rows.append(
                {
                    "events_path": portable(events_path),
                    "source_video": doc.get("source_video"),
                    "rally_index": rally_index,
                    "start_sec": start,
                    "end_sec": end,
                    "gt_touches": len(touches_gt),
                    "detector_frames": len(rows),
                    **score,
                }
            )

    def finalize(counts: dict[str, int]) -> dict[str, Any]:
        precision = counts["true_positive"] / counts["n_detected"] if counts["n_detected"] else 0.0
        recall = counts["true_positive"] / counts["n_truth"] if counts["n_truth"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {**counts, "precision": precision, "recall": recall, "f1": f1}
    aggregate_by_method = {method: finalize(counts) for method, counts in aggregate.items()}
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "threshold": threshold,
        "frame_stride": frame_stride,
        "touch_tolerance_sec": TOUCH_TOL_SEC,
        "audio_parameters": {
            "delta": audio_delta,
            "wait_sec": audio_wait_sec,
            "fusion_break_tolerance_sec": AUDIO_FUSION_BREAK_TOL_SEC,
            "fusion_nms_sec": AUDIO_FUSION_NMS_SEC,
            "min_break_support": AUDIO_MIN_BREAK_SUPPORT,
            "consensus_support": AUDIO_CONSENSUS_SUPPORT,
            "break_velocity_win_sec": AUDIO_BREAK_VELOCITY_WIN_SEC,
            "exclude_stall_pad_sec": AUDIO_EXCLUDE_STALL_PAD_SEC,
            "exclude_drop_pre_sec": AUDIO_EXCLUDE_DROP_PRE_SEC,
            "exclude_drop_post_sec": AUDIO_EXCLUDE_DROP_POST_SEC,
        },
        "events_paths": [portable(path) for path in events_paths],
        "aggregate": aggregate_by_method["fused"],
        "aggregate_by_method": aggregate_by_method,
        "rallies": rally_rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "event_eval_summary.json", summary)
    return summary


def format_metric_triplet(metrics: dict[str, Any]) -> str:
    return (
        f"{float(metrics.get('precision') or 0.0):.3f}/"
        f"{float(metrics.get('recall') or 0.0):.3f}/"
        f"{float(metrics.get('f1') or 0.0):.3f}"
    )


def write_report(path: Path, summary: dict[str, Any]) -> None:
    aggregate_by_method = summary.get("aggregate_by_method") or {}
    agg = summary["aggregate"]
    audio_params = summary.get("audio_parameters") or {}
    lines = [
        "# OWLv2 Audio-Fused Touch Evaluation",
        "",
        f"- Model: `{summary['model']}`",
        f"- Threshold: `{summary['threshold']}`",
        f"- Frame stride: `{summary['frame_stride']}`",
        f"- Touch tolerance: `{summary['touch_tolerance_sec']}` sec",
        f"- Audio onset delta/wait: `{audio_params.get('delta')}` / `{audio_params.get('wait_sec')}` sec",
        f"- Audio-to-trajectory break tolerance: `{audio_params.get('fusion_break_tolerance_sec')}` sec",
        f"- Fusion NMS: `{audio_params.get('fusion_nms_sec')}` sec",
        "",
        f"Fused aggregate: P `{agg['precision']:.3f}`, R `{agg['recall']:.3f}`, F1 `{agg['f1']:.3f}` "
        f"({agg['true_positive']}/{agg['n_truth']} GT, {agg['n_detected'] - agg['true_positive']} FP)",
        "",
        "## Aggregate Ablation",
        "",
        "| method | precision | recall | f1 | tp/gt | fp | detections |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in ("trajectory", "audio_only", "fused"):
        metrics = aggregate_by_method.get(method) or {}
        tp = int(metrics.get("true_positive") or 0)
        detected = int(metrics.get("n_detected") or 0)
        truth = int(metrics.get("n_truth") or 0)
        lines.append(
            f"| {method} | {float(metrics.get('precision') or 0.0):.3f} | "
            f"{float(metrics.get('recall') or 0.0):.3f} | "
            f"{float(metrics.get('f1') or 0.0):.3f} | "
            f"{tp}/{truth} | {detected - tp} | {detected} |"
        )
    lines.extend(
        [
            "",
            "## Per-Rally Ablation",
            "",
        "| video | rally | gt | trajectory P/R/F1 | audio-only P/R/F1 | fused P/R/F1 | audio candidates | fused times | points |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for row in summary["rallies"]:
        trajectory = row.get("best") or {}
        audio_only = row.get("audio_only") or {}
        fused = row.get("fused") or {}
        audio_candidate_count = len(audio_only.get("candidates") or [])
        fused_times = ", ".join(f"{float(time_sec):.3f}" for time_sec in fused.get("detected_touches", []))
        lines.append(
            f"| {row.get('source_video')} | {row.get('rally_index')} | {row.get('gt_touches')} | "
            f"{format_metric_triplet(trajectory)} | "
            f"{format_metric_triplet(audio_only)} | "
            f"{format_metric_triplet(fused)} | "
            f"{audio_candidate_count} | {fused_times} | {row.get('n_points_clean', 0)} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- `trajectory` is the prior L2 breakpoint plus velocity-magnitude filter.",
            "- `audio_only` is loose onset extraction after rally, stall, and drop-window filtering; it should be high-recall and low-precision.",
            "- `fused` uses audio onset timing only when corroborated by supported L2 breakpoints and local velocity evidence.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OWLv2 tracks at touch-event level")
    parser.add_argument("--events", type=Path, action="append", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", choices=["owlv2", "owlv2-large"], default="owlv2")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    parser.add_argument("--audio-delta", type=float, default=AUDIO_DELTA)
    parser.add_argument("--audio-wait-sec", type=float, default=AUDIO_WAIT_SEC)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = args.events or DEFAULT_EVENTS
    summary = evaluate_events(
        events_paths=events,
        out_dir=args.out_dir,
        model_name=args.model,
        prompts=args.prompts,
        device=args.device,
        threshold=args.threshold,
        frame_stride=args.frame_stride,
        force=args.force,
        audio_delta=args.audio_delta,
        audio_wait_sec=args.audio_wait_sec,
    )
    report = args.out_dir / "event_eval_report.md"
    write_report(report, summary)
    print(f"summary: {args.out_dir / 'event_eval_summary.json'}")
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
