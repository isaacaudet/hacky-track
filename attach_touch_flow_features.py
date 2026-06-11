#!/usr/bin/env python3
"""Attach local optical-flow features around the tracked ball.

This is a candidate-level feature step. It does not change OWLv2 detections,
labels, or trajectory fitting. It adds detector-independent local motion cues
around the ball so the touch classifier can learn visual impulse patterns that
are not just coordinate jitter from the detector.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = ROOT / "runs/release-27-public/touch_corpus_v1/touch_training_dataset_v1"
DEFAULT_REVIEW_MANIFEST = ROOT / "runs/release-27-public/touch_corpus_v1/touch_review_manifest.json"
DEFAULT_THRESHOLD = 0.2
DEFAULT_BALL_TOLERANCE_SEC = 0.08


@dataclass(frozen=True)
class BallPoint:
    time_sec: float
    frame_index: int | None
    x: float
    y: float
    score: float


class FrameReader:
    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open video {video_path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def read_gray(self, frame_index: int) -> np.ndarray | None:
        if frame_index < 0:
            return None
        if self.frame_count and frame_index >= self.frame_count:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def release(self) -> None:
        self.cap.release()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def source_video_name(row: dict[str, Any]) -> str | None:
    value = row.get("source_video") or row.get("video_name")
    if value is None:
        return None
    return Path(str(value)).name


def top_ball_detection(row: dict[str, Any], threshold: float) -> BallPoint | None:
    detections = row.get("detections")
    if isinstance(detections, list):
        above = [det for det in detections if float(det.get("score") or 0.0) >= threshold]
        if not above:
            return None
        det = max(above, key=lambda item: float(item.get("score") or 0.0))
        return BallPoint(
            time_sec=float(row["time_sec"]),
            frame_index=None if row.get("frame_index") is None else int(row["frame_index"]),
            x=float(det["x"]),
            y=float(det["y"]),
            score=float(det["score"]),
        )
    if row.get("x") is not None and row.get("y") is not None:
        score = float(row.get("confidence") or row.get("score") or 1.0)
        if score < threshold:
            return None
        return BallPoint(
            time_sec=float(row["time_sec"]),
            frame_index=None if row.get("frame_index") is None else int(row["frame_index"]),
            x=float(row["x"]),
            y=float(row["y"]),
            score=score,
        )
    return None


def detection_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for item in args.detections_jsonl or []:
        paths.append(item)
    for directory in args.detections_dir or []:
        paths.extend(sorted(directory.glob("*.jsonl")))
        paths.extend(sorted(directory.glob("**/*.jsonl")))
    unique: list[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def load_ball_tracks(paths: list[Path], threshold: float) -> dict[str, list[BallPoint]]:
    tracks: dict[str, list[BallPoint]] = {}
    for path in paths:
        for row in read_jsonl(path):
            video_name = source_video_name(row)
            if not video_name:
                continue
            ball = top_ball_detection(row, threshold)
            if ball is None:
                continue
            tracks.setdefault(video_name, []).append(ball)
    for video_name, points in tracks.items():
        deduped: dict[float, BallPoint] = {}
        for point in points:
            key = round(point.time_sec, 6)
            prior = deduped.get(key)
            if prior is None or point.score > prior.score:
                deduped[key] = point
        tracks[video_name] = sorted(deduped.values(), key=lambda item: item.time_sec)
    return tracks


def video_paths_by_name(review_manifest: Path) -> dict[str, Path]:
    doc = read_json(review_manifest)
    out: dict[str, Path] = {}
    for item in doc.get("items", []):
        out[Path(str(item["video_name"])).name] = Path(str(item["video_path"]))
    return out


def cache_key(
    *,
    video_name: str,
    frame_index: int,
    frame_step: int,
    patch_radius_px: int,
    grid_step_px: int,
    ball: BallPoint | None,
) -> str:
    if ball is None:
        ball_part = "noball"
    else:
        ball_part = f"{ball.x:.1f},{ball.y:.1f},{ball.score:.3f}"
    return f"{video_name}|{frame_index}|step={frame_step}|r={patch_radius_px}|grid={grid_step_px}|{ball_part}"


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    return {str(row["cache_key"]): row for row in rows if row.get("cache_key")}


def write_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    write_jsonl(path, sorted(cache.values(), key=lambda row: str(row["cache_key"])))


def nearest_ball(points: list[BallPoint], time_sec: float, tolerance_sec: float) -> BallPoint | None:
    if not points:
        return None
    best = min(points, key=lambda point: abs(point.time_sec - time_sec))
    if abs(best.time_sec - time_sec) > tolerance_sec:
        return None
    return best


def nearest_frame_ball(points: list[BallPoint], frame_index: int, max_delta_frames: int) -> BallPoint | None:
    with_frames = [point for point in points if point.frame_index is not None]
    if not with_frames:
        return None
    best = min(with_frames, key=lambda point: abs(int(point.frame_index or 0) - frame_index))
    if abs(int(best.frame_index or 0) - frame_index) > max_delta_frames:
        return None
    return best


def grid_points(
    *,
    center_x: float,
    center_y: float,
    shape: tuple[int, int],
    radius_px: int,
    grid_step_px: int,
) -> np.ndarray:
    height, width = shape
    points = []
    for y in range(int(round(center_y - radius_px)), int(round(center_y + radius_px)) + 1, grid_step_px):
        for x in range(int(round(center_x - radius_px)), int(round(center_x + radius_px)) + 1, grid_step_px):
            if x < 1 or y < 1 or x >= width - 1 or y >= height - 1:
                continue
            if math.hypot(x - center_x, y - center_y) <= radius_px:
                points.append([float(x), float(y)])
    if not points:
        return np.empty((0, 1, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)


def patch_std(gray: np.ndarray, *, center_x: float, center_y: float, radius_px: int) -> float | None:
    height, width = gray.shape[:2]
    x0 = max(0, int(round(center_x - radius_px)))
    x1 = min(width, int(round(center_x + radius_px + 1)))
    y0 = max(0, int(round(center_y - radius_px)))
    y1 = min(height, int(round(center_y + radius_px + 1)))
    if x1 <= x0 or y1 <= y0:
        return None
    return float(np.std(gray[y0:y1, x0:x1]))


def lk_flow_summary(
    from_gray: np.ndarray,
    to_gray: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    radius_px: int,
    grid_step_px: int,
) -> dict[str, Any]:
    points = grid_points(center_x=center_x, center_y=center_y, shape=from_gray.shape[:2], radius_px=radius_px, grid_step_px=grid_step_px)
    if len(points) == 0:
        return {
            "dx": None,
            "dy": None,
            "mag": None,
            "valid_frac": 0.0,
            "points": 0,
        }
    next_points, status, _err = cv2.calcOpticalFlowPyrLK(
        from_gray,
        to_gray,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_points is None or status is None:
        return {
            "dx": None,
            "dy": None,
            "mag": None,
            "valid_frac": 0.0,
            "points": int(len(points)),
        }
    valid = status.reshape(-1).astype(bool)
    valid_frac = float(valid.sum() / max(len(points), 1))
    if int(valid.sum()) < 2:
        return {
            "dx": None,
            "dy": None,
            "mag": None,
            "valid_frac": round(valid_frac, 6),
            "points": int(len(points)),
        }
    displacement = next_points.reshape(-1, 2)[valid] - points.reshape(-1, 2)[valid]
    dx = float(np.median(displacement[:, 0]))
    dy = float(np.median(displacement[:, 1]))
    return {
        "dx": dx,
        "dy": dy,
        "mag": float(math.hypot(dx, dy)),
        "valid_frac": round(valid_frac, 6),
        "points": int(len(points)),
    }


def missing_features(status: str, ball: BallPoint | None = None, frame_index: int | None = None) -> dict[str, Any]:
    return {
        "flow_feature_status": status,
        "flow_missing": True,
        "flow_ball_missing": ball is None,
        "flow_frame_index": frame_index,
        "flow_ball_x": None if ball is None else round(ball.x, 3),
        "flow_ball_y": None if ball is None else round(ball.y, 3),
        "flow_ball_score": None if ball is None else round(ball.score, 6),
        "flow_patch_radius_px": None,
        "flow_texture_std": None,
        "flow_before_points": 0,
        "flow_after_points": 0,
        "flow_before_valid_frac": 0.0,
        "flow_after_valid_frac": 0.0,
        "flow_before_dx": None,
        "flow_before_dy": None,
        "flow_before_mag": None,
        "flow_after_dx": None,
        "flow_after_dy": None,
        "flow_after_mag": None,
        "flow_dx_delta": None,
        "flow_dy_delta": None,
        "flow_delta_mag": None,
        "flow_impulse_mag": None,
    }


def round_or_none(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def compute_flow_features(
    *,
    reader: FrameReader,
    track: list[BallPoint],
    candidate_time_sec: float,
    ball: BallPoint | None,
    frame_step: int,
    patch_radius_px: int,
    grid_step_px: int,
) -> dict[str, Any]:
    if ball is None:
        frame_index = int(round(candidate_time_sec * reader.fps))
        return missing_features("missing_ball_detection", None, frame_index)
    frame_index = int(ball.frame_index) if ball.frame_index is not None else int(round(candidate_time_sec * reader.fps))
    prev_index = frame_index - frame_step
    next_index = frame_index + frame_step
    prev_gray = reader.read_gray(prev_index)
    center_gray = reader.read_gray(frame_index)
    next_gray = reader.read_gray(next_index)
    if prev_gray is None or center_gray is None or next_gray is None:
        return missing_features("missing_neighbor_frame", ball, frame_index)

    prev_ball = nearest_frame_ball(track, prev_index, max_delta_frames=max(2, frame_step + 1)) or ball
    before = lk_flow_summary(
        prev_gray,
        center_gray,
        center_x=prev_ball.x,
        center_y=prev_ball.y,
        radius_px=patch_radius_px,
        grid_step_px=grid_step_px,
    )
    after = lk_flow_summary(
        center_gray,
        next_gray,
        center_x=ball.x,
        center_y=ball.y,
        radius_px=patch_radius_px,
        grid_step_px=grid_step_px,
    )
    texture = patch_std(center_gray, center_x=ball.x, center_y=ball.y, radius_px=patch_radius_px)
    dx_delta = None
    dy_delta = None
    delta_mag = None
    impulse = None
    if before["dx"] is not None and after["dx"] is not None:
        dx_delta = float(after["dx"] - before["dx"])
    if before["dy"] is not None and after["dy"] is not None:
        dy_delta = float(after["dy"] - before["dy"])
    if before["mag"] is not None and after["mag"] is not None:
        delta_mag = float(after["mag"] - before["mag"])
    if dx_delta is not None and dy_delta is not None:
        impulse = float(math.hypot(dx_delta, dy_delta))

    enough_flow = before["mag"] is not None or after["mag"] is not None
    return {
        "flow_feature_status": "ok" if enough_flow else "insufficient_tracked_points",
        "flow_missing": not enough_flow,
        "flow_ball_missing": False,
        "flow_frame_index": frame_index,
        "flow_ball_x": round(ball.x, 3),
        "flow_ball_y": round(ball.y, 3),
        "flow_ball_score": round(ball.score, 6),
        "flow_patch_radius_px": int(patch_radius_px),
        "flow_texture_std": round_or_none(texture),
        "flow_before_points": int(before["points"]),
        "flow_after_points": int(after["points"]),
        "flow_before_valid_frac": before["valid_frac"],
        "flow_after_valid_frac": after["valid_frac"],
        "flow_before_dx": round_or_none(before["dx"]),
        "flow_before_dy": round_or_none(before["dy"]),
        "flow_before_mag": round_or_none(before["mag"]),
        "flow_after_dx": round_or_none(after["dx"]),
        "flow_after_dy": round_or_none(after["dy"]),
        "flow_after_mag": round_or_none(after["mag"]),
        "flow_dx_delta": round_or_none(dx_delta),
        "flow_dy_delta": round_or_none(dy_delta),
        "flow_delta_mag": round_or_none(delta_mag),
        "flow_impulse_mag": round_or_none(impulse),
    }


def attach_flow_to_rows(
    rows: list[dict[str, Any]],
    *,
    video_paths: dict[str, Path],
    ball_tracks: dict[str, list[BallPoint]],
    cache: dict[str, dict[str, Any]],
    ball_tolerance_sec: float,
    frame_step: int,
    patch_radius_px: int,
    grid_step_px: int,
    force: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    readers: dict[str, FrameReader | None] = {}
    out_rows: list[dict[str, Any]] = []
    per_video: dict[str, dict[str, Any]] = {}
    try:
        for row in rows:
            out = dict(row)
            video_name = str(row["video_name"])
            video_path = video_paths.get(video_name)
            track = ball_tracks.get(video_name, [])
            ball = nearest_ball(track, float(row["candidate_time_sec"]), ball_tolerance_sec)
            frame_index = None
            if ball and ball.frame_index is not None:
                frame_index = int(ball.frame_index)
            if ball is None:
                features = missing_features("missing_ball_detection", None, None)
                out.update(features)
                out_rows.append(out)
                stats = per_video.setdefault(
                    video_name,
                    {
                        "video_name": video_name,
                        "video_id": row.get("video_id"),
                        "split": row.get("split"),
                        "rows": 0,
                        "positive_rows": 0,
                        "ok_rows": 0,
                        "ball_present_rows": 0,
                        "missing_flow_rows": 0,
                    },
                )
                stats["rows"] += 1
                stats["positive_rows"] += int(bool(row.get("label_is_touch")))
                stats["missing_flow_rows"] += 1
                continue
            if frame_index is not None:
                key = cache_key(
                    video_name=video_name,
                    frame_index=frame_index,
                    frame_step=frame_step,
                    patch_radius_px=patch_radius_px,
                    grid_step_px=grid_step_px,
                    ball=ball,
                )
                if not force and key in cache:
                    features = {k: v for k, v in cache[key].items() if k != "cache_key"}
                    out.update(features)
                    out_rows.append(out)
                    stats = per_video.setdefault(
                        video_name,
                        {
                            "video_name": video_name,
                            "video_id": row.get("video_id"),
                            "split": row.get("split"),
                            "rows": 0,
                            "positive_rows": 0,
                            "ok_rows": 0,
                            "ball_present_rows": 0,
                            "missing_flow_rows": 0,
                        },
                    )
                    stats["rows"] += 1
                    stats["positive_rows"] += int(bool(row.get("label_is_touch")))
                    stats["ok_rows"] += int(features.get("flow_feature_status") == "ok")
                    stats["ball_present_rows"] += int(not bool(features.get("flow_ball_missing")))
                    stats["missing_flow_rows"] += int(bool(features.get("flow_missing")))
                    continue
            if video_path is None:
                features = missing_features("missing_video_path", ball, frame_index)
            else:
                if video_name not in readers:
                    try:
                        readers[video_name] = FrameReader(video_path)
                    except RuntimeError:
                        readers[video_name] = None
                reader = readers[video_name]
                if reader is None:
                    features = missing_features("video_open_failed", ball, frame_index)
                else:
                    if frame_index is None:
                        frame_index = int(round(float(row["candidate_time_sec"]) * reader.fps))
                    key = cache_key(
                        video_name=video_name,
                        frame_index=frame_index,
                        frame_step=frame_step,
                        patch_radius_px=patch_radius_px,
                        grid_step_px=grid_step_px,
                        ball=ball,
                    )
                    if not force and key in cache:
                        features = {k: v for k, v in cache[key].items() if k != "cache_key"}
                    else:
                        features = compute_flow_features(
                            reader=reader,
                            track=track,
                            candidate_time_sec=float(row["candidate_time_sec"]),
                            ball=ball,
                            frame_step=frame_step,
                            patch_radius_px=patch_radius_px,
                            grid_step_px=grid_step_px,
                        )
                        cache[key] = {"cache_key": key, **features}
            out.update(features)
            out_rows.append(out)
            stats = per_video.setdefault(
                video_name,
                {
                    "video_name": video_name,
                    "video_id": row.get("video_id"),
                    "split": row.get("split"),
                    "rows": 0,
                    "positive_rows": 0,
                    "ok_rows": 0,
                    "ball_present_rows": 0,
                    "missing_flow_rows": 0,
                },
            )
            stats["rows"] += 1
            stats["positive_rows"] += int(bool(row.get("label_is_touch")))
            stats["ok_rows"] += int(features.get("flow_feature_status") == "ok")
            stats["ball_present_rows"] += int(not bool(features.get("flow_ball_missing")))
            stats["missing_flow_rows"] += int(bool(features.get("flow_missing")))
    finally:
        for reader in readers.values():
            if reader is not None:
                reader.release()
    summary = {
        "rows": len(out_rows),
        "ok_rows": sum(1 for row in out_rows if row.get("flow_feature_status") == "ok"),
        "ball_present_rows": sum(1 for row in out_rows if not row.get("flow_ball_missing")),
        "missing_flow_rows": sum(1 for row in out_rows if row.get("flow_missing")),
        "videos": list(per_video.values()),
    }
    return out_rows, summary


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Touch Optical-Flow Feature Attachment",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Dataset dir: `{manifest['dataset_dir']}`",
        f"- Out dir: `{manifest['out_dir']}`",
        f"- Frame step: `{manifest['frame_step']}`",
        f"- Patch radius: `{manifest['patch_radius_px']}` px",
        f"- Grid step: `{manifest['grid_step_px']}` px",
        f"- Cache: `{manifest['cache_path']}`",
        f"- Train/val rows: `{manifest['train_val']['rows']}`",
        f"- Train/val flow rows: `{manifest['train_val']['ok_rows']}`",
        f"- Frozen-test rows: `{manifest['test_frozen']['rows']}`",
        f"- Frozen-test flow rows: `{manifest['test_frozen']['ok_rows']}`",
        "",
        "## Per-Video",
        "",
        "| video | split | rows | positives | ball | flow ok | missing flow |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in manifest["videos"]:
        lines.append(
            f"| `{row['video_name']}` | {row.get('split')} | {row['rows']} | {row['positive_rows']} | "
            f"{row['ball_present_rows']} | {row['ok_rows']} | {row['missing_flow_rows']} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Optical flow is a soft classifier feature, not a hard touch rule.",
            "- It measures local pre/post motion around the ball patch and keeps missing/weak-flow flags.",
            "- It is intended to help separate actual contact impulse from audio-only footsteps and detector jitter.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def attach_dataset(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else dataset_dir
    cache_path = args.cache_path.resolve() if args.cache_path else out_dir / "touch_flow_frame_features_cache.jsonl"
    paths = detection_paths(args)
    ball_tracks = load_ball_tracks(paths, args.threshold)
    video_paths = video_paths_by_name(args.review_manifest.resolve())
    cache = load_cache(cache_path)
    train_rows = read_jsonl(dataset_dir / "touch_training_candidates.jsonl")
    test_rows = read_jsonl(dataset_dir / "touch_training_test_frozen.jsonl")
    train_out, train_summary = attach_flow_to_rows(
        train_rows,
        video_paths=video_paths,
        ball_tracks=ball_tracks,
        cache=cache,
        ball_tolerance_sec=args.ball_tolerance_sec,
        frame_step=args.frame_step,
        patch_radius_px=args.patch_radius_px,
        grid_step_px=args.grid_step_px,
        force=args.force,
    )
    test_out, test_summary = attach_flow_to_rows(
        test_rows,
        video_paths=video_paths,
        ball_tracks=ball_tracks,
        cache=cache,
        ball_tolerance_sec=args.ball_tolerance_sec,
        frame_step=args.frame_step,
        patch_radius_px=args.patch_radius_px,
        grid_step_px=args.grid_step_px,
        force=args.force,
    )
    if any(row.get("split") == "test_frozen" for row in train_out):
        raise AssertionError("test_frozen row leaked into train/validation output")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "touch_training_candidates.jsonl", train_out)
    write_jsonl(out_dir / "touch_training_test_frozen.jsonl", test_out)
    write_cache(cache_path, cache)
    videos = train_summary["videos"] + test_summary["videos"]
    status = "features_attached" if train_summary["ok_rows"] or test_summary["ok_rows"] else "no_flow_features"
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "review_manifest": str(args.review_manifest),
        "detection_files": [str(path) for path in paths],
        "cache_path": str(cache_path),
        "cache_rows": len(cache),
        "track_videos": sorted(ball_tracks.keys()),
        "threshold": args.threshold,
        "ball_tolerance_sec": args.ball_tolerance_sec,
        "frame_step": args.frame_step,
        "patch_radius_px": args.patch_radius_px,
        "grid_step_px": args.grid_step_px,
        "train_val": train_summary,
        "test_frozen": test_summary,
        "videos": videos,
    }
    write_json(out_dir / "touch_flow_feature_manifest.json", manifest)
    write_report(out_dir / "touch_flow_feature_report.md", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach local optical-flow features around the tracked ball")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW_MANIFEST)
    parser.add_argument("--detections-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--detections-dir", type=Path, action="append", default=[])
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--ball-tolerance-sec", type=float, default=DEFAULT_BALL_TOLERANCE_SEC)
    parser.add_argument("--frame-step", type=int, default=2)
    parser.add_argument("--patch-radius-px", type=int, default=24)
    parser.add_argument("--grid-step-px", type=int, default=6)
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--force", action="store_true", help="recompute cached flow rows")
    return parser.parse_args()


def main() -> None:
    manifest = attach_dataset(parse_args())
    out_dir = Path(manifest["out_dir"])
    print(f"manifest: {out_dir / 'touch_flow_feature_manifest.json'}")
    print(f"report:   {out_dir / 'touch_flow_feature_report.md'}")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "train_val_rows": manifest["train_val"]["rows"],
                "train_val_ok_rows": manifest["train_val"]["ok_rows"],
                "test_frozen_rows": manifest["test_frozen"]["rows"],
                "test_frozen_ok_rows": manifest["test_frozen"]["ok_rows"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
