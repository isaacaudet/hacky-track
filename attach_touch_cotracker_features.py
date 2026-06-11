#!/usr/bin/env python3
"""Attach short-window CoTracker foot-region features to touch candidates.

CoTracker does not know "left kick" or "inner foot" semantics. This stage uses
RTMW only to seed foot landmarks on the candidate frame, then asks CoTracker to
propagate those foot points through a short local video window. The resulting
``cotracker_*`` fields are label-free automatic features. They are separate from
``manual_*`` calibration fields and can be disabled by the release classifier if
they do not improve clip-disjoint metrics.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

import attach_touch_pose_features as pose
import train_release_contact_classifier as contact


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "runs/release-27-public/touch_corpus_v1"
DEFAULT_DATASET_DIR = DEFAULT_CORPUS / "touch_training_dataset_v1"
DEFAULT_LABELS_DIR = DEFAULT_CORPUS / "visual_touch_labels"
DEFAULT_REVIEW_MANIFEST = DEFAULT_CORPUS / "touch_review_manifest.json"
COTRACKER_FEATURE_VERSION = 1
FOOT_PARTS = ("ankle", "big_toe", "small_toe", "heel")


@dataclass(frozen=True)
class SeedPoint:
    side: str
    part: str
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class TrackResult:
    tracks: np.ndarray
    visibility: np.ndarray


class CoTrackerRunner(Protocol):
    def track(
        self,
        frames_rgb: np.ndarray,
        queries: list[SeedPoint],
        query_frame: int,
        *,
        scale_x: float,
        scale_y: float,
    ) -> TrackResult:
        ...


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


def finite_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def ball_xy(row: dict[str, Any]) -> tuple[float, float] | None:
    for prefix in ("visual_crop_ball", "pose_ball", "vision_embedding_ball"):
        x = finite_float(row.get(f"{prefix}_x"))
        y = finite_float(row.get(f"{prefix}_y"))
        if x is not None and y is not None:
            return x, y
    x_norm = finite_float(row.get("visual_ball_x_norm"))
    y_norm = finite_float(row.get("visual_ball_y_norm"))
    if x_norm is not None and y_norm is not None:
        width = finite_float(row.get("video_width")) or finite_float(row.get("frame_width"))
        height = finite_float(row.get("video_height")) or finite_float(row.get("frame_height"))
        if width and height:
            return x_norm * width, y_norm * height
    return None


def frame_index_for_row(row: dict[str, Any], fps: float) -> int:
    for key in ("visual_crop_frame_index", "pose_frame_index", "vision_embedding_frame_index", "frame_index"):
        value = finite_float(row.get(key))
        if value is not None:
            return max(0, int(round(value)))
    return max(0, int(round(float(row.get("candidate_time_sec") or 0.0) * fps)))


def video_paths_by_name(review_manifest: Path) -> dict[str, Path]:
    doc = read_json(review_manifest)
    return {Path(str(item["video_name"])).name: Path(str(item["video_path"])) for item in doc.get("items", [])}


def default_features(status: str) -> dict[str, Any]:
    return {
        "cotracker_feature_version": COTRACKER_FEATURE_VERSION,
        "cotracker_feature_status": status,
        "cotracker_model": None,
        "cotracker_window_frames": 0,
        "cotracker_query_count": 0,
        "cotracker_seed_left_count": 0,
        "cotracker_seed_right_count": 0,
        "cotracker_nearest_track_side": None,
        "cotracker_nearest_track_side_is_left": False,
        "cotracker_nearest_track_side_is_right": False,
        "cotracker_nearest_track_distance_px": None,
        "cotracker_side_distance_margin_px": None,
        "cotracker_side_confidence": None,
        "cotracker_left_min_dist_px": None,
        "cotracker_right_min_dist_px": None,
        "cotracker_left_visibility_fraction": None,
        "cotracker_right_visibility_fraction": None,
        "cotracker_left_center_visible_points": 0,
        "cotracker_right_center_visible_points": 0,
        "cotracker_left_track_stability_px": None,
        "cotracker_right_track_stability_px": None,
        "cotracker_nearest_surface_guess": None,
        "cotracker_nearest_surface_margin_px": None,
        "cotracker_nearest_ball_axis_projection": None,
        "cotracker_nearest_ball_axis_lateral_px": None,
        "cotracker_left_motion_px": None,
        "cotracker_right_motion_px": None,
    }


def point_line_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float | None:
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    denom = math.hypot(dx, dy)
    if denom <= 1e-6:
        return None
    return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / denom


def axis_projection_features(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> dict[str, float | None]:
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    axis_len = math.hypot(dx, dy)
    if axis_len <= 1e-6:
        return {"axis_projection": None, "axis_lateral_px": None}
    vx = px - x1
    vy = py - y1
    projection = (vx * dx + vy * dy) / (axis_len * axis_len)
    lateral = abs(vx * dy - vy * dx) / axis_len
    return {"axis_projection": projection, "axis_lateral_px": lateral}


def load_clip_rgb(
    video_path: Path,
    *,
    center_frame: int,
    radius_frames: int,
    process_width: int,
) -> tuple[np.ndarray, int, float, float, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    first = max(0, center_frame - radius_frames)
    last = center_frame + radius_frames
    if frame_count:
        last = min(frame_count - 1, last)
    frames: list[np.ndarray] = []
    original_width = None
    original_height = None
    try:
        for frame_index in range(first, last + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                continue
            original_height, original_width = frame_bgr.shape[:2]
            scale = min(1.0, process_width / max(1, original_width)) if process_width > 0 else 1.0
            if scale < 1.0:
                frame_bgr = cv2.resize(
                    frame_bgr,
                    (int(round(original_width * scale)), int(round(original_height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames or original_width is None or original_height is None:
        raise RuntimeError(f"could not read local clip around frame {center_frame} from {video_path}")
    query_frame = center_frame - first
    scale_x = frames[0].shape[1] / original_width
    scale_y = frames[0].shape[0] / original_height
    return np.stack(frames), query_frame, fps, scale_x, scale_y


def seed_points_from_pose(
    *,
    pose_model: Any,
    frame_rgb: np.ndarray,
    ball: tuple[float, float],
    scale_x: float,
    scale_y: float,
    keypoint_threshold: float,
) -> list[SeedPoint]:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    bboxes = pose_model.det_model(frame_bgr)
    if len(bboxes) == 0:
        return []
    keypoints, scores = pose_model.pose_model(frame_bgr, bboxes=bboxes)
    scaled_ball = pose.BallPoint(time_sec=0.0, frame_index=None, x=ball[0] * scale_x, y=ball[1] * scale_y, score=1.0)
    person_index = pose.choose_person(keypoints, scores, scaled_ball, keypoint_threshold)
    if person_index is None:
        return []
    kp = keypoints[person_index]
    sc = scores[person_index]
    seeds: list[SeedPoint] = []
    for side, mapping in pose.FOOT_SIDE_KPTS.items():
        for part in FOOT_PARTS:
            index = mapping[part]
            conf = float(sc[index])
            if not math.isfinite(conf) or conf < keypoint_threshold:
                continue
            x, y = map(float, kp[index])
            seeds.append(SeedPoint(side=side, part=part, x=x / scale_x, y=y / scale_y, confidence=conf))
    return seeds


class TorchHubCoTrackerRunner:
    def __init__(self, *, model_name: str, device: str) -> None:
        import torch

        self.torch = torch
        self.model_name = model_name
        self.device = device
        self.model = torch.hub.load("facebookresearch/co-tracker", model_name, trust_repo=True).to(device)
        self.model.eval()

    def track(
        self,
        frames_rgb: np.ndarray,
        queries: list[SeedPoint],
        query_frame: int,
        *,
        scale_x: float,
        scale_y: float,
    ) -> TrackResult:
        torch = self.torch
        video = torch.tensor(frames_rgb).permute(0, 3, 1, 2)[None].float().to(self.device)
        query_rows = [
            [float(query_frame), seed.x * scale_x, seed.y * scale_y]
            for seed in queries
        ]
        query_tensor = torch.tensor([query_rows], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred_tracks, pred_visibility = self.model(video, queries=query_tensor)
        tracks = pred_tracks.detach().cpu().numpy()[0]
        visibility = pred_visibility.detach().cpu().numpy()[0]
        tracks[:, :, 0] = tracks[:, :, 0] / scale_x
        tracks[:, :, 1] = tracks[:, :, 1] / scale_y
        if visibility.ndim == 3:
            visibility = visibility[:, :, 0]
        return TrackResult(tracks=tracks, visibility=visibility)


def side_track_summary(
    *,
    seeds: list[SeedPoint],
    result: TrackResult,
    side: str,
    center_index: int,
    ball: tuple[float, float],
    visibility_threshold: float,
) -> dict[str, Any]:
    indices = [i for i, seed in enumerate(seeds) if seed.side == side]
    if not indices:
        return {
            "min_dist": None,
            "visibility_fraction": None,
            "center_visible_points": 0,
            "stability": None,
            "motion": None,
            "part_points": {},
        }
    visibility = result.visibility[:, indices]
    tracks = result.tracks[:, indices, :]
    visible_mask = visibility >= visibility_threshold
    center_visible = visible_mask[center_index]
    center_points = tracks[center_index]
    distances = [
        math.hypot(float(point[0]) - ball[0], float(point[1]) - ball[1])
        for point, visible in zip(center_points, center_visible)
        if bool(visible)
    ]
    first_visible = tracks[0][visible_mask[0]] if np.any(visible_mask[0]) else None
    last_visible = tracks[-1][visible_mask[-1]] if np.any(visible_mask[-1]) else None
    motion = None
    if first_visible is not None and last_visible is not None and len(first_visible) and len(last_visible):
        motion = float(np.linalg.norm(np.mean(last_visible, axis=0) - np.mean(first_visible, axis=0)))
    stability = None
    if np.any(visible_mask):
        side_tracks = tracks[visible_mask]
        if len(side_tracks):
            stability = float(np.median(np.linalg.norm(side_tracks - np.median(side_tracks, axis=0), axis=1)))
    part_points: dict[str, tuple[float, float]] = {}
    for local_index, seed_index in enumerate(indices):
        if bool(center_visible[local_index]):
            seed = seeds[seed_index]
            point = center_points[local_index]
            part_points[seed.part] = (float(point[0]), float(point[1]))
    return {
        "min_dist": None if not distances else min(distances),
        "visibility_fraction": float(np.mean(visible_mask)),
        "center_visible_points": int(np.count_nonzero(center_visible)),
        "stability": stability,
        "motion": motion,
        "part_points": part_points,
    }


def surface_from_tracked_points(side_summary: dict[str, Any], ball: tuple[float, float]) -> dict[str, Any]:
    points = side_summary.get("part_points") or {}
    heel = points.get("heel")
    big_toe = points.get("big_toe")
    small_toe = points.get("small_toe")
    if heel is None or big_toe is None or small_toe is None:
        return {
            "surface_guess": None,
            "surface_margin_px": None,
            "axis_projection": None,
            "axis_lateral_px": None,
        }
    toe_mid = ((big_toe[0] + small_toe[0]) / 2.0, (big_toe[1] + small_toe[1]) / 2.0)
    axis = axis_projection_features(ball, heel, toe_mid)
    inner = point_line_distance(ball, heel, big_toe)
    outer = point_line_distance(ball, heel, small_toe)
    margin = None if inner is None or outer is None else inner - outer
    return {
        "surface_guess": None if margin is None else ("inner" if margin < 0 else "outer"),
        "surface_margin_px": None if margin is None else abs(margin),
        "axis_projection": axis["axis_projection"],
        "axis_lateral_px": axis["axis_lateral_px"],
    }


def cotracker_features_for_row(
    *,
    row: dict[str, Any],
    video_path: Path,
    pose_model: Any,
    runner: CoTrackerRunner,
    model_name: str,
    device: str,
    window_sec: float,
    process_width: int,
    keypoint_threshold: float,
    visibility_threshold: float,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    center_frame = frame_index_for_row(row, fps)
    ball = ball_xy(row)
    if ball is None:
        return default_features("missing_ball")
    radius_frames = max(2, int(round(window_sec * fps / 2.0)))
    frames, query_frame, _fps, scale_x, scale_y = load_clip_rgb(
        video_path,
        center_frame=center_frame,
        radius_frames=radius_frames,
        process_width=process_width,
    )
    seeds = seed_points_from_pose(
        pose_model=pose_model,
        frame_rgb=frames[query_frame],
        ball=ball,
        scale_x=scale_x,
        scale_y=scale_y,
        keypoint_threshold=keypoint_threshold,
    )
    if not seeds:
        features = default_features("missing_pose_seed")
        features["cotracker_window_frames"] = int(frames.shape[0])
        return features
    result = runner.track(frames, seeds, query_frame, scale_x=scale_x, scale_y=scale_y)
    left = side_track_summary(
        seeds=seeds,
        result=result,
        side="left",
        center_index=query_frame,
        ball=ball,
        visibility_threshold=visibility_threshold,
    )
    right = side_track_summary(
        seeds=seeds,
        result=result,
        side="right",
        center_index=query_frame,
        ball=ball,
        visibility_threshold=visibility_threshold,
    )
    left_dist = left["min_dist"]
    right_dist = right["min_dist"]
    nearest_side = None
    nearest_dist = None
    other_dist = None
    if left_dist is not None and right_dist is not None:
        nearest_side = "left" if left_dist <= right_dist else "right"
        nearest_dist = min(left_dist, right_dist)
        other_dist = max(left_dist, right_dist)
    elif left_dist is not None:
        nearest_side = "left"
        nearest_dist = left_dist
    elif right_dist is not None:
        nearest_side = "right"
        nearest_dist = right_dist
    if nearest_side is None:
        features = default_features("no_visible_tracked_foot_at_center")
        features["cotracker_window_frames"] = int(frames.shape[0])
        features["cotracker_query_count"] = len(seeds)
        return features
    surface = surface_from_tracked_points(left if nearest_side == "left" else right, ball)
    margin = None if left_dist is None or right_dist is None else abs(left_dist - right_dist)
    confidence = None
    if nearest_dist is not None and other_dist is not None:
        confidence = abs(other_dist - nearest_dist) / max(1.0, other_dist + nearest_dist)
    features = default_features("ok")
    features.update(
        {
            "cotracker_model": model_name,
            "cotracker_window_frames": int(frames.shape[0]),
            "cotracker_query_count": len(seeds),
            "cotracker_seed_left_count": sum(1 for seed in seeds if seed.side == "left"),
            "cotracker_seed_right_count": sum(1 for seed in seeds if seed.side == "right"),
            "cotracker_nearest_track_side": nearest_side,
            "cotracker_nearest_track_side_is_left": nearest_side == "left",
            "cotracker_nearest_track_side_is_right": nearest_side == "right",
            "cotracker_nearest_track_distance_px": nearest_dist,
            "cotracker_side_distance_margin_px": margin,
            "cotracker_side_confidence": confidence,
            "cotracker_left_min_dist_px": left_dist,
            "cotracker_right_min_dist_px": right_dist,
            "cotracker_left_visibility_fraction": left["visibility_fraction"],
            "cotracker_right_visibility_fraction": right["visibility_fraction"],
            "cotracker_left_center_visible_points": left["center_visible_points"],
            "cotracker_right_center_visible_points": right["center_visible_points"],
            "cotracker_left_track_stability_px": left["stability"],
            "cotracker_right_track_stability_px": right["stability"],
            "cotracker_nearest_surface_guess": surface["surface_guess"],
            "cotracker_nearest_surface_margin_px": surface["surface_margin_px"],
            "cotracker_nearest_ball_axis_projection": surface["axis_projection"],
            "cotracker_nearest_ball_axis_lateral_px": surface["axis_lateral_px"],
            "cotracker_left_motion_px": left["motion"],
            "cotracker_right_motion_px": right["motion"],
        }
    )
    return features


def row_key(row: dict[str, Any]) -> tuple[str, float]:
    return str(row.get("video_id") or row.get("video_name") or "unknown"), round(float(row.get("candidate_time_sec") or 0.0), 6)


def contact_labeled_keys(rows: list[dict[str, Any]], labels_dir: Path, tolerance_sec: float) -> set[tuple[str, float]]:
    labels = contact.load_label_contact_examples(labels_dir)
    matched, _summary = contact.attach_event_file_contact_labels(rows, labels, tolerance_sec=tolerance_sec)
    labeled = contact.rows_with_contact_labels(matched)
    return {row_key(row) for row in labeled}


def attach_rows(
    rows: list[dict[str, Any]],
    *,
    video_paths: dict[str, Path],
    labels_dir: Path,
    label_match_tolerance_sec: float,
    contact_labeled_only: bool,
    row_limit: int | None,
    pose_model: Any | None,
    runner: CoTrackerRunner | None,
    model_name: str,
    device: str,
    window_sec: float,
    process_width: int,
    keypoint_threshold: float,
    visibility_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []
    labeled = contact_labeled_keys(rows, labels_dir, label_match_tolerance_sec) if contact_labeled_only else set()
    processed = 0
    per_video: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        new_row = dict(row)
        video_name = Path(str(row.get("video_name") or row.get("source_video") or "")).name
        should_process = not contact_labeled_only or row_key(row) in labeled
        if row_limit is not None and processed >= row_limit:
            should_process = False
        if not should_process:
            features = default_features("skipped_not_requested")
        else:
            processed += 1
            if pose_model is None or runner is None:
                features = default_features("runtime_unavailable")
            elif video_name not in video_paths:
                features = default_features("missing_video_path")
            else:
                try:
                    features = cotracker_features_for_row(
                        row=row,
                        video_path=video_paths[video_name],
                        pose_model=pose_model,
                        runner=runner,
                        model_name=model_name,
                        device=device,
                        window_sec=window_sec,
                        process_width=process_width,
                        keypoint_threshold=keypoint_threshold,
                        visibility_threshold=visibility_threshold,
                    )
                except Exception as exc:  # pragma: no cover - runtime failures are environment-specific
                    features = default_features(f"runtime_error: {type(exc).__name__}: {exc}")
        new_row.update(features)
        out.append(new_row)
        status = str(features.get("cotracker_feature_status") or "unknown")
        per_video[str(row.get("video_id") or video_name)]["rows"] += 1
        per_video[str(row.get("video_id") or video_name)][status] += 1
    summary = {
        "rows": len(out),
        "processed_rows": processed,
        "ok_rows": sum(1 for row in out if row.get("cotracker_feature_status") == "ok"),
        "status_counts": dict(Counter(str(row.get("cotracker_feature_status") or "unknown") for row in out)),
        "videos": [
            {"video_id": video_id, **dict(counter)}
            for video_id, counter in sorted(per_video.items())
        ],
    }
    return out, summary


def load_pose_model(args: argparse.Namespace) -> Any | None:
    if args.dry_run:
        return None
    try:
        from rtmlib import Wholebody

        return Wholebody(mode=args.pose_mode, backend="onnxruntime", device=args.pose_device)
    except Exception:
        return None


def attach_dataset(args: argparse.Namespace, *, runner_factory: Any = TorchHubCoTrackerRunner) -> dict[str, Any]:
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else dataset_dir
    video_paths = video_paths_by_name(args.review_manifest.resolve())
    pose_model = load_pose_model(args)
    runner = None
    if not args.dry_run:
        try:
            runner = runner_factory(model_name=args.cotracker_model, device=args.cotracker_device)
        except Exception:
            runner = None
    train_rows = read_jsonl(dataset_dir / "touch_training_candidates.jsonl")
    test_rows = read_jsonl(dataset_dir / "touch_training_test_frozen.jsonl")
    train_out, train_summary = attach_rows(
        train_rows,
        video_paths=video_paths,
        labels_dir=args.labels_dir.resolve(),
        label_match_tolerance_sec=args.label_match_tolerance_sec,
        contact_labeled_only=args.contact_labeled_only,
        row_limit=args.row_limit,
        pose_model=pose_model,
        runner=runner,
        model_name=args.cotracker_model,
        device=args.cotracker_device,
        window_sec=args.window_sec,
        process_width=args.process_width,
        keypoint_threshold=args.keypoint_threshold,
        visibility_threshold=args.visibility_threshold,
    )
    test_limit = None if args.row_limit is None else max(0, args.row_limit - train_summary["processed_rows"])
    test_out, test_summary = attach_rows(
        test_rows,
        video_paths=video_paths,
        labels_dir=args.labels_dir.resolve(),
        label_match_tolerance_sec=args.label_match_tolerance_sec,
        contact_labeled_only=args.contact_labeled_only,
        row_limit=test_limit,
        pose_model=pose_model,
        runner=runner,
        model_name=args.cotracker_model,
        device=args.cotracker_device,
        window_sec=args.window_sec,
        process_width=args.process_width,
        keypoint_threshold=args.keypoint_threshold,
        visibility_threshold=args.visibility_threshold,
    )
    if any(row.get("split") == "test_frozen" for row in train_out):
        raise AssertionError("test_frozen row leaked into train/validation cotracker output")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "touch_training_candidates.jsonl", train_out)
    write_jsonl(out_dir / "touch_training_test_frozen.jsonl", test_out)
    status = "features_attached" if train_summary["ok_rows"] or test_summary["ok_rows"] else "no_cotracker_features"
    manifest = {
        "schema_version": 1,
        "cotracker_feature_version": COTRACKER_FEATURE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "review_manifest": str(args.review_manifest.resolve()),
        "labels_dir": str(args.labels_dir.resolve()),
        "contact_labeled_only": args.contact_labeled_only,
        "row_limit": args.row_limit,
        "dry_run": args.dry_run,
        "pose_mode": args.pose_mode,
        "pose_device": args.pose_device,
        "cotracker_model": args.cotracker_model,
        "cotracker_device": args.cotracker_device,
        "window_sec": args.window_sec,
        "process_width": args.process_width,
        "keypoint_threshold": args.keypoint_threshold,
        "visibility_threshold": args.visibility_threshold,
        "train_val": train_summary,
        "test_frozen": test_summary,
        "notes": [
            "cotracker_* fields are label-free automatic features.",
            "CoTracker is seeded from RTMW foot landmarks on short local windows.",
            "This stage is optional and must be evaluated clip-disjoint before promotion.",
        ],
    }
    write_json(out_dir / "touch_cotracker_feature_manifest.json", manifest)
    write_report(out_dir / "touch_cotracker_feature_report.md", manifest)
    return manifest


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Touch CoTracker Feature Attachment",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Dataset dir: `{manifest['dataset_dir']}`",
        f"- Out dir: `{manifest['out_dir']}`",
        f"- Model: `{manifest['cotracker_model']}` on `{manifest['cotracker_device']}`",
        f"- Pose seed mode: `{manifest['pose_mode']}` on `{manifest['pose_device']}`",
        f"- Contact-labeled only: `{manifest['contact_labeled_only']}`",
        f"- Row limit: `{manifest['row_limit']}`",
        f"- Window: `{manifest['window_sec']}` sec",
        f"- Process width: `{manifest['process_width']}`",
        f"- Train/val rows: `{manifest['train_val']['rows']}`",
        f"- Train/val processed rows: `{manifest['train_val']['processed_rows']}`",
        f"- Train/val ok rows: `{manifest['train_val']['ok_rows']}`",
        f"- Frozen-test rows: `{manifest['test_frozen']['rows']}`",
        f"- Frozen-test processed rows: `{manifest['test_frozen']['processed_rows']}`",
        f"- Frozen-test ok rows: `{manifest['test_frozen']['ok_rows']}`",
        "",
        "## Status Counts",
        "",
        "| split | counts |",
        "| --- | --- |",
        f"| train + validation | `{manifest['train_val']['status_counts']}` |",
        f"| frozen test | `{manifest['test_frozen']['status_counts']}` |",
        "",
        "## Notes",
        "",
    ]
    for note in manifest.get("notes", []):
        lines.append(f"- {note}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach CoTracker foot-continuity features to touch candidates")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW_MANIFEST)
    parser.add_argument("--label-match-tolerance-sec", type=float, default=0.08)
    parser.add_argument("--contact-labeled-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pose-mode", choices=["lightweight", "balanced", "performance"], default="performance")
    parser.add_argument("--pose-device", default="cpu")
    parser.add_argument("--keypoint-threshold", type=float, default=0.25)
    parser.add_argument("--cotracker-model", default="cotracker3_offline")
    parser.add_argument("--cotracker-device", default="cpu")
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    parser.add_argument("--window-sec", type=float, default=0.8)
    parser.add_argument("--process-width", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    manifest = attach_dataset(parse_args())
    out_dir = Path(manifest["out_dir"])
    print(f"manifest: {out_dir / 'touch_cotracker_feature_manifest.json'}")
    print(f"report:   {out_dir / 'touch_cotracker_feature_report.md'}")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "train_val_processed_rows": manifest["train_val"]["processed_rows"],
                "train_val_ok_rows": manifest["train_val"]["ok_rows"],
                "test_frozen_processed_rows": manifest["test_frozen"]["processed_rows"],
                "test_frozen_ok_rows": manifest["test_frozen"]["ok_rows"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
