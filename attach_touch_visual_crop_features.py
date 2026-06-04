#!/usr/bin/env python3
"""Attach fixed ball-centered visual crop features to touch candidates.

The contact side/surface classifier needs some representation of what the
reviewer sees: the local ball + nearby foot patch. Pose keypoints are useful but
sparse and sometimes semantically unreliable in egocentric footage, so this
step adds cached, detector-anchored visual descriptors without changing the
fixed OWLv2 detector or any labels.
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
VISUAL_CROP_FEATURE_VERSION = 1


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
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    def read_bgr(self, frame_index: int) -> np.ndarray | None:
        if frame_index < 0:
            return None
        if self.frame_count and frame_index >= self.frame_count:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return frame

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


def nearest_ball(points: list[BallPoint], time_sec: float, tolerance_sec: float) -> BallPoint | None:
    if not points:
        return None
    best = min(points, key=lambda point: abs(point.time_sec - time_sec))
    if abs(best.time_sec - time_sec) > tolerance_sec:
        return None
    return best


def fallback_row_ball(row: dict[str, Any]) -> BallPoint | None:
    x = row.get("pose_ball_x") or row.get("flow_ball_x")
    y = row.get("pose_ball_y") or row.get("flow_ball_y")
    if x is None or y is None:
        return None
    try:
        return BallPoint(
            time_sec=float(row.get("candidate_time_sec") or 0.0),
            frame_index=None if row.get("pose_frame_index") is None else int(row.get("pose_frame_index")),
            x=float(x),
            y=float(y),
            score=float(row.get("pose_ball_score") or row.get("flow_ball_score") or row.get("detector_confidence_near_candidate") or 1.0),
        )
    except (TypeError, ValueError):
        return None


def video_paths_by_name(review_manifest: Path) -> dict[str, Path]:
    doc = read_json(review_manifest)
    out: dict[str, Path] = {}
    for item in doc.get("items", []):
        out[Path(str(item["video_name"])).name] = Path(str(item["video_path"]))
    return out


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    return {str(row["cache_key"]): row for row in rows if row.get("cache_key")}


def write_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    write_jsonl(path, sorted(cache.values(), key=lambda row: str(row["cache_key"])))


def cache_key(
    *,
    video_name: str,
    frame_index: int,
    crop_size_px: int,
    grid_size: int,
    ball: BallPoint | None,
) -> str:
    if ball is None:
        ball_part = "noball"
    else:
        ball_part = f"{ball.x:.1f},{ball.y:.1f},{ball.score:.3f}"
    return f"v{VISUAL_CROP_FEATURE_VERSION}|{video_name}|{frame_index}|crop={crop_size_px}|grid={grid_size}|{ball_part}"


def default_visual_crop_features(status: str, ball: BallPoint | None = None, frame_index: int | None = None) -> dict[str, Any]:
    return {
        "visual_crop_feature_status": status,
        "visual_crop_present": False,
        "visual_crop_frame_index": frame_index,
        "visual_crop_ball_x": None if ball is None else round(ball.x, 3),
        "visual_crop_ball_y": None if ball is None else round(ball.y, 3),
        "visual_crop_ball_score": None if ball is None else round(ball.score, 6),
        "visual_crop_ball_missing": ball is None,
        "visual_ball_x_norm": None,
        "visual_ball_y_norm": None,
        "visual_ball_dist_from_center_norm": None,
        "visual_frame_width_px": None,
        "visual_frame_height_px": None,
        "visual_crop_width_px": None,
        "visual_crop_height_px": None,
        "visual_crop_area_px": None,
        "visual_crop_coverage_frac": None,
    }


def extract_visual_crop_features(
    frame: np.ndarray,
    *,
    ball: BallPoint,
    frame_index: int,
    crop_size_px: int,
    grid_size: int,
) -> dict[str, Any]:
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return default_visual_crop_features("invalid_frame", ball, frame_index)
    half = crop_size_px / 2.0
    x1 = max(0, int(math.floor(ball.x - half)))
    y1 = max(0, int(math.floor(ball.y - half)))
    x2 = min(width, int(math.ceil(ball.x + half)))
    y2 = min(height, int(math.ceil(ball.y + half)))
    if x2 <= x1 or y2 <= y1:
        return default_visual_crop_features("empty_crop", ball, frame_index)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return default_visual_crop_features("empty_crop", ball, frame_index)

    resized = cv2.resize(crop, (grid_size * 8, grid_size * 8), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    features = {
        "visual_crop_feature_status": "ok",
        "visual_crop_present": True,
        "visual_crop_frame_index": int(frame_index),
        "visual_crop_ball_x": round(ball.x, 3),
        "visual_crop_ball_y": round(ball.y, 3),
        "visual_crop_ball_score": round(ball.score, 6),
        "visual_crop_ball_missing": False,
        "visual_ball_x_norm": float(ball.x / width),
        "visual_ball_y_norm": float(ball.y / height),
        "visual_ball_dist_from_center_norm": float(abs(ball.x / width - 0.5)),
        "visual_frame_width_px": float(width),
        "visual_frame_height_px": float(height),
        "visual_crop_width_px": float(x2 - x1),
        "visual_crop_height_px": float(y2 - y1),
        "visual_crop_area_px": float((x2 - x1) * (y2 - y1)),
        "visual_crop_coverage_frac": float(((x2 - x1) * (y2 - y1)) / max(1, crop_size_px * crop_size_px)),
    }
    for gy in range(grid_size):
        for gx in range(grid_size):
            y_slice = slice(gy * 8, (gy + 1) * 8)
            x_slice = slice(gx * 8, (gx + 1) * 8)
            cell_hsv = hsv[y_slice, x_slice]
            cell_gray = gray[y_slice, x_slice]
            cell_edges = edges[y_slice, x_slice]
            prefix = f"visual_crop_grid_{gy}_{gx}"
            features[f"{prefix}_hue_mean"] = float(np.mean(cell_hsv[:, :, 0]) / 180.0)
            features[f"{prefix}_sat_mean"] = float(np.mean(cell_hsv[:, :, 1]) / 255.0)
            features[f"{prefix}_val_mean"] = float(np.mean(cell_hsv[:, :, 2]) / 255.0)
            features[f"{prefix}_gray_mean"] = float(np.mean(cell_gray) / 255.0)
            features[f"{prefix}_edge_density"] = float(np.count_nonzero(cell_edges) / cell_edges.size)
    return features


def frame_index_for_row(row: dict[str, Any], ball: BallPoint | None, reader: FrameReader) -> int | None:
    if ball is not None and ball.frame_index is not None:
        return int(ball.frame_index)
    for key in ("pose_frame_index", "flow_frame_index", "frame_index"):
        value = row.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    try:
        return int(round(float(row.get("candidate_time_sec") or 0.0) * reader.fps))
    except (TypeError, ValueError):
        return None


def attach_visual_crops_to_rows(
    rows: list[dict[str, Any]],
    *,
    video_paths: dict[str, Path],
    ball_tracks: dict[str, list[BallPoint]],
    cache: dict[str, dict[str, Any]],
    crop_size_px: int,
    grid_size: int,
    ball_tolerance_sec: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    readers: dict[str, FrameReader] = {}
    per_video: dict[str, dict[str, Any]] = {}
    try:
        for row in rows:
            out = dict(row)
            video_name = str(row["video_name"])
            time_sec = float(row["candidate_time_sec"])
            video_path = video_paths.get(video_name)
            ball = nearest_ball(ball_tracks.get(video_name, []), time_sec, ball_tolerance_sec) or fallback_row_ball(row)
            if video_path is None:
                features = default_visual_crop_features("missing_video_path", ball)
            else:
                reader = readers.get(video_name)
                if reader is None:
                    reader = FrameReader(video_path)
                    readers[video_name] = reader
                frame_index = frame_index_for_row(row, ball, reader)
                if ball is None:
                    features = default_visual_crop_features("missing_ball", ball, frame_index)
                elif frame_index is None:
                    features = default_visual_crop_features("missing_frame_index", ball, frame_index)
                else:
                    key = cache_key(video_name=video_name, frame_index=frame_index, crop_size_px=crop_size_px, grid_size=grid_size, ball=ball)
                    if key in cache:
                        features = {k: v for k, v in cache[key].items() if k != "cache_key"}
                    else:
                        frame = reader.read_bgr(frame_index)
                        if frame is None:
                            features = default_visual_crop_features("missing_frame", ball, frame_index)
                        else:
                            features = extract_visual_crop_features(
                                frame,
                                ball=ball,
                                frame_index=frame_index,
                                crop_size_px=crop_size_px,
                                grid_size=grid_size,
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
                },
            )
            stats["rows"] += 1
            stats["positive_rows"] += int(bool(row.get("label_is_touch")))
            stats["ok_rows"] += int(features.get("visual_crop_feature_status") == "ok")
            stats["ball_present_rows"] += int(not bool(features.get("visual_crop_ball_missing")))
    finally:
        for reader in readers.values():
            reader.release()
    return out_rows, {
        "rows": len(out_rows),
        "ok_rows": sum(1 for row in out_rows if row.get("visual_crop_feature_status") == "ok"),
        "ball_present_rows": sum(1 for row in out_rows if not row.get("visual_crop_ball_missing")),
        "videos": list(per_video.values()),
    }


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Touch Visual Crop Feature Attachment",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Dataset dir: `{manifest['dataset_dir']}`",
        f"- Out dir: `{manifest['out_dir']}`",
        f"- Crop size: `{manifest['crop_size_px']}` px",
        f"- Grid size: `{manifest['grid_size']}x{manifest['grid_size']}`",
        f"- Train/val rows: `{manifest['train_val']['rows']}`",
        f"- Train/val ok crop rows: `{manifest['train_val']['ok_rows']}`",
        f"- Frozen-test rows: `{manifest['test_frozen']['rows']}`",
        f"- Frozen-test ok crop rows: `{manifest['test_frozen']['ok_rows']}`",
        f"- Cache: `{manifest['cache_path']}`",
        "",
        "## Per-Video",
        "",
        "| video | split | rows | positives | ok crops | ball present |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in manifest["videos"]:
        lines.append(
            f"| `{row['video_name']}` | {row.get('split')} | {row['rows']} | {row['positive_rows']} | "
            f"{row['ok_rows']} | {row['ball_present_rows']} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- These are soft visual descriptors for contact classification, not detector changes.",
            "- The fixed OWLv2 threshold remains unchanged; missing crops are preserved as status features.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def attach_dataset(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else dataset_dir
    cache_path = args.cache_path.resolve() if args.cache_path else out_dir / "touch_visual_crop_feature_cache.jsonl"
    paths = detection_paths(args)
    ball_tracks = load_ball_tracks(paths, args.threshold)
    video_paths = video_paths_by_name(args.review_manifest.resolve())
    cache = load_cache(cache_path)
    train_rows = read_jsonl(dataset_dir / "touch_training_candidates.jsonl")
    test_rows = read_jsonl(dataset_dir / "touch_training_test_frozen.jsonl")
    train_out, train_summary = attach_visual_crops_to_rows(
        train_rows,
        video_paths=video_paths,
        ball_tracks=ball_tracks,
        cache=cache,
        crop_size_px=args.crop_size_px,
        grid_size=args.grid_size,
        ball_tolerance_sec=args.ball_tolerance_sec,
    )
    test_out, test_summary = attach_visual_crops_to_rows(
        test_rows,
        video_paths=video_paths,
        ball_tracks=ball_tracks,
        cache=cache,
        crop_size_px=args.crop_size_px,
        grid_size=args.grid_size,
        ball_tolerance_sec=args.ball_tolerance_sec,
    )
    if any(row.get("split") == "test_frozen" for row in train_out):
        raise AssertionError("test_frozen row leaked into train/validation output")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "touch_training_candidates.jsonl", train_out)
    write_jsonl(out_dir / "touch_training_test_frozen.jsonl", test_out)
    write_cache(cache_path, cache)
    videos = train_summary["videos"] + test_summary["videos"]
    status = "features_attached" if train_summary["ok_rows"] or test_summary["ok_rows"] else "no_visual_crop_features"
    manifest = {
        "schema_version": 1,
        "visual_crop_feature_version": VISUAL_CROP_FEATURE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "review_manifest": str(args.review_manifest),
        "detection_files": [str(path) for path in paths],
        "cache_path": str(cache_path),
        "threshold": args.threshold,
        "ball_tolerance_sec": args.ball_tolerance_sec,
        "crop_size_px": args.crop_size_px,
        "grid_size": args.grid_size,
        "train_val": train_summary,
        "test_frozen": test_summary,
        "videos": videos,
    }
    write_json(out_dir / "touch_visual_crop_feature_manifest.json", manifest)
    write_report(out_dir / "touch_visual_crop_feature_report.md", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach fixed ball-centered visual crop features to touch training candidates")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW_MANIFEST)
    parser.add_argument("--detections-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--detections-dir", type=Path, action="append", default=[])
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--ball-tolerance-sec", type=float, default=DEFAULT_BALL_TOLERANCE_SEC)
    parser.add_argument("--crop-size-px", type=int, default=224)
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--cache-path", type=Path)
    return parser.parse_args()


def main() -> None:
    manifest = attach_dataset(parse_args())
    out_dir = Path(manifest["out_dir"])
    print(f"manifest: {out_dir / 'touch_visual_crop_feature_manifest.json'}")
    print(f"report:   {out_dir / 'touch_visual_crop_feature_report.md'}")
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
