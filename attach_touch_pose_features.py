#!/usr/bin/env python3
"""Attach RTMW lower-body pose proximity features to touch candidates.

This step is deliberately candidate-level and cached. It does not change the
fixed OWLv2 detector, labels, or L2 track fitting. It adds soft evidence such
as ball-to-toe/ankle/knee distance and pose-missing flags so the classifier can
learn when body proximity is useful without hard-vetoing pose misses.
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

LOWER_KPTS = {
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
    17: "left_big_toe",
    18: "left_small_toe",
    19: "left_heel",
    20: "right_big_toe",
    21: "right_small_toe",
    22: "right_heel",
}
FOOT_KPTS = {
    15: "left_ankle",
    16: "right_ankle",
    17: "left_big_toe",
    18: "left_small_toe",
    19: "left_heel",
    20: "right_big_toe",
    21: "right_small_toe",
    22: "right_heel",
}


@dataclass(frozen=True)
class BallPoint:
    time_sec: float
    frame_index: int | None
    x: float
    y: float
    score: float


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


def video_paths_by_name(review_manifest: Path) -> dict[str, Path]:
    doc = read_json(review_manifest)
    out: dict[str, Path] = {}
    for item in doc.get("items", []):
        name = Path(str(item["video_name"])).name
        out[name] = Path(str(item["video_path"]))
    return out


def cache_key(
    *,
    video_name: str,
    frame_index: int,
    mode: str,
    kpt_thr: float,
    ball: BallPoint | None,
) -> str:
    if ball is None:
        ball_part = "noball"
    else:
        ball_part = f"{ball.x:.1f},{ball.y:.1f},{ball.score:.3f}"
    return f"{video_name}|{frame_index}|{mode}|{kpt_thr:.3f}|{ball_part}"


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    return {str(row["cache_key"]): row for row in rows if row.get("cache_key")}


def write_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    write_jsonl(path, sorted(cache.values(), key=lambda row: str(row["cache_key"])))


def seek_frame(video_path: Path, frame_index: int) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"could not read {video_path} frame {frame_index}")
    return frame, fps


def nearest_distance(
    keypoints: np.ndarray,
    scores: np.ndarray,
    ball: BallPoint,
    keypoint_names: dict[int, str],
    threshold: float,
) -> dict[str, Any]:
    best: dict[str, Any] = {
        "keypoint_index": None,
        "keypoint_name": None,
        "distance_px": None,
        "confidence": None,
    }
    for index, name in keypoint_names.items():
        conf = float(scores[index])
        if not np.isfinite(conf) or conf < threshold:
            continue
        x, y = map(float, keypoints[index])
        dist = math.hypot(x - ball.x, y - ball.y)
        if best["distance_px"] is None or dist < float(best["distance_px"]):
            best = {
                "keypoint_index": int(index),
                "keypoint_name": name,
                "distance_px": float(dist),
                "confidence": conf,
            }
    return best


def shank_length(keypoints: np.ndarray, scores: np.ndarray, threshold: float) -> float | None:
    lengths = []
    for knee, ankle in [(13, 15), (14, 16)]:
        if float(scores[knee]) < threshold or float(scores[ankle]) < threshold:
            continue
        x1, y1 = map(float, keypoints[knee])
        x2, y2 = map(float, keypoints[ankle])
        length = math.hypot(x2 - x1, y2 - y1)
        if length > 8.0:
            lengths.append(length)
    return float(np.median(lengths)) if lengths else None


def choose_person(keypoints: np.ndarray, scores: np.ndarray, ball: BallPoint | None, threshold: float) -> int | None:
    if len(keypoints) == 0:
        return None
    best_i = None
    best_score = -1e18
    for i in range(len(keypoints)):
        lower_score_sum = float(sum(float(scores[i, k]) for k in LOWER_KPTS if np.isfinite(float(scores[i, k]))))
        if ball is None:
            score = lower_score_sum
        else:
            nearest = nearest_distance(keypoints[i], scores[i], ball, FOOT_KPTS, threshold)
            distance = nearest.get("distance_px")
            score = lower_score_sum - (float(distance) if distance is not None else 5000.0) * 0.01
        if score > best_score:
            best_score = score
            best_i = i
    return best_i


def missing_pose_features(status: str, ball: BallPoint | None = None, frame_index: int | None = None) -> dict[str, Any]:
    return {
        "pose_feature_status": status,
        "pose_frame_index": frame_index,
        "pose_ball_x": None if ball is None else round(ball.x, 3),
        "pose_ball_y": None if ball is None else round(ball.y, 3),
        "pose_ball_score": None if ball is None else round(ball.score, 6),
        "pose_ball_missing": ball is None,
        "pose_missing": True,
        "pose_present": False,
        "pose_person_boxes": 0,
        "pose_lower_body_present": False,
        "pose_foot_present": False,
        "pose_shank_length_px": None,
        "pose_nearest_foot_part": None,
        "pose_nearest_foot_conf": None,
        "pose_nearest_foot_dist_px": None,
        "pose_nearest_foot_dist_norm_shank": None,
        "pose_nearest_lower_part": None,
        "pose_nearest_lower_conf": None,
        "pose_nearest_lower_dist_px": None,
        "pose_nearest_lower_dist_norm_shank": None,
    }


def compute_pose_features(
    *,
    model: Any,
    video_path: Path,
    frame_index: int,
    ball: BallPoint | None,
    kpt_thr: float,
) -> dict[str, Any]:
    frame, _fps = seek_frame(video_path, frame_index)
    bboxes = model.det_model(frame)
    box_count = int(len(bboxes))
    if box_count == 0:
        return missing_pose_features("missing_person_box", ball, frame_index)
    keypoints, scores = model.pose_model(frame, bboxes=bboxes)
    person_index = choose_person(keypoints, scores, ball, kpt_thr)
    if person_index is None:
        out = missing_pose_features("missing_pose_keypoints", ball, frame_index)
        out["pose_person_boxes"] = box_count
        return out

    kp = keypoints[person_index]
    sc = scores[person_index]
    lower_present = any(float(sc[k]) >= kpt_thr for k in LOWER_KPTS)
    foot_present = any(float(sc[k]) >= kpt_thr for k in FOOT_KPTS)
    shank = shank_length(kp, sc, kpt_thr)
    nearest_foot = None
    nearest_lower = None
    if ball is not None:
        nearest_foot = nearest_distance(kp, sc, ball, FOOT_KPTS, kpt_thr)
        nearest_lower = nearest_distance(kp, sc, ball, LOWER_KPTS, kpt_thr)
    foot_dist = None if not nearest_foot else nearest_foot.get("distance_px")
    lower_dist = None if not nearest_lower else nearest_lower.get("distance_px")
    return {
        "pose_feature_status": "ok",
        "pose_frame_index": frame_index,
        "pose_ball_x": None if ball is None else round(ball.x, 3),
        "pose_ball_y": None if ball is None else round(ball.y, 3),
        "pose_ball_score": None if ball is None else round(ball.score, 6),
        "pose_ball_missing": ball is None,
        "pose_missing": False,
        "pose_present": True,
        "pose_person_boxes": box_count,
        "pose_lower_body_present": bool(lower_present),
        "pose_foot_present": bool(foot_present),
        "pose_shank_length_px": None if shank is None else round(shank, 6),
        "pose_nearest_foot_part": None if not nearest_foot else nearest_foot.get("keypoint_name"),
        "pose_nearest_foot_conf": None if not nearest_foot else round(float(nearest_foot.get("confidence") or 0.0), 6),
        "pose_nearest_foot_dist_px": None if foot_dist is None else round(float(foot_dist), 6),
        "pose_nearest_foot_dist_norm_shank": None
        if foot_dist is None or not shank
        else round(float(foot_dist) / shank, 6),
        "pose_nearest_lower_part": None if not nearest_lower else nearest_lower.get("keypoint_name"),
        "pose_nearest_lower_conf": None if not nearest_lower else round(float(nearest_lower.get("confidence") or 0.0), 6),
        "pose_nearest_lower_dist_px": None if lower_dist is None else round(float(lower_dist), 6),
        "pose_nearest_lower_dist_norm_shank": None
        if lower_dist is None or not shank
        else round(float(lower_dist) / shank, 6),
    }


def attach_pose_to_rows(
    rows: list[dict[str, Any]],
    *,
    video_paths: dict[str, Path],
    ball_tracks: dict[str, list[BallPoint]],
    cache: dict[str, dict[str, Any]],
    model: Any | None,
    mode: str,
    kpt_thr: float,
    ball_tolerance_sec: float,
    cache_only: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    per_video: dict[str, dict[str, Any]] = {}
    for row in rows:
        out = dict(row)
        video_name = str(row["video_name"])
        time_sec = float(row["candidate_time_sec"])
        video_path = video_paths.get(video_name)
        ball = nearest_ball(ball_tracks.get(video_name, []), time_sec, ball_tolerance_sec)
        if ball and ball.frame_index is not None:
            frame_index = int(ball.frame_index)
        else:
            frame_index = None
        if frame_index is None and video_path is not None:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            frame_index = int(round(time_sec * fps))

        if video_path is None:
            features = missing_pose_features("missing_video_path", ball, frame_index)
        elif frame_index is None:
            features = missing_pose_features("missing_frame_index", ball, frame_index)
        else:
            key = cache_key(video_name=video_name, frame_index=frame_index, mode=mode, kpt_thr=kpt_thr, ball=ball)
            if key in cache:
                features = {k: v for k, v in cache[key].items() if k != "cache_key"}
            elif cache_only:
                features = missing_pose_features("missing_pose_cache", ball, frame_index)
            elif model is None:
                features = missing_pose_features("pose_runtime_unavailable", ball, frame_index)
            else:
                features = compute_pose_features(model=model, video_path=video_path, frame_index=frame_index, ball=ball, kpt_thr=kpt_thr)
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
                "pose_present_rows": 0,
                "foot_present_rows": 0,
                "ball_present_rows": 0,
            },
        )
        stats["rows"] += 1
        stats["positive_rows"] += int(bool(row.get("label_is_touch")))
        stats["ok_rows"] += int(features.get("pose_feature_status") == "ok")
        stats["pose_present_rows"] += int(bool(features.get("pose_present")))
        stats["foot_present_rows"] += int(bool(features.get("pose_foot_present")))
        stats["ball_present_rows"] += int(not bool(features.get("pose_ball_missing")))
    summary = {
        "rows": len(out_rows),
        "ok_rows": sum(1 for row in out_rows if row.get("pose_feature_status") == "ok"),
        "pose_present_rows": sum(1 for row in out_rows if row.get("pose_present")),
        "foot_present_rows": sum(1 for row in out_rows if row.get("pose_foot_present")),
        "ball_present_rows": sum(1 for row in out_rows if not row.get("pose_ball_missing")),
        "videos": list(per_video.values()),
    }
    return out_rows, summary


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Touch Pose Feature Attachment",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Dataset dir: `{manifest['dataset_dir']}`",
        f"- Out dir: `{manifest['out_dir']}`",
        f"- Mode: `{manifest['pose_mode']}`",
        f"- Device: `{manifest['pose_device']}`",
        f"- Keypoint threshold: `{manifest['keypoint_threshold']}`",
        f"- Train/val rows: `{manifest['train_val']['rows']}`",
        f"- Train/val pose-present rows: `{manifest['train_val']['pose_present_rows']}`",
        f"- Train/val foot-present rows: `{manifest['train_val']['foot_present_rows']}`",
        f"- Frozen-test rows: `{manifest['test_frozen']['rows']}`",
        f"- Frozen-test pose-present rows: `{manifest['test_frozen']['pose_present_rows']}`",
        f"- Frozen-test foot-present rows: `{manifest['test_frozen']['foot_present_rows']}`",
        f"- Cache: `{manifest['cache_path']}`",
        "",
        "## Per-Video",
        "",
        "| video | split | rows | positives | pose | foot | ball |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in manifest["videos"]:
        lines.append(
            f"| `{row['video_name']}` | {row.get('split')} | {row['rows']} | {row['positive_rows']} | "
            f"{row['pose_present_rows']} | {row['foot_present_rows']} | {row['ball_present_rows']} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Pose proximity is a soft classifier feature, not a hard rule.",
            "- Rows with missing person boxes or missing pose are preserved with `pose_missing=true`.",
            "- Distance features are normalized by shank length when both knee and ankle are visible.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def attach_dataset(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else dataset_dir
    cache_path = args.cache_path.resolve() if args.cache_path else out_dir / "touch_pose_frame_features_cache.jsonl"
    paths = detection_paths(args)
    ball_tracks = load_ball_tracks(paths, args.threshold)
    video_paths = video_paths_by_name(args.review_manifest.resolve())
    cache = load_cache(cache_path)

    model = None
    runtime_status = "cache_only" if args.cache_only else "loaded"
    if not args.cache_only:
        try:
            from rtmlib import Wholebody

            model = Wholebody(mode=args.pose_mode, backend="onnxruntime", device=args.pose_device)
        except Exception as exc:  # pragma: no cover - exercised only when runtime is unavailable
            runtime_status = f"unavailable: {exc}"
            model = None

    train_rows = read_jsonl(dataset_dir / "touch_training_candidates.jsonl")
    test_rows = read_jsonl(dataset_dir / "touch_training_test_frozen.jsonl")
    train_out, train_summary = attach_pose_to_rows(
        train_rows,
        video_paths=video_paths,
        ball_tracks=ball_tracks,
        cache=cache,
        model=model,
        mode=args.pose_mode,
        kpt_thr=args.keypoint_threshold,
        ball_tolerance_sec=args.ball_tolerance_sec,
        cache_only=args.cache_only,
    )
    test_out, test_summary = attach_pose_to_rows(
        test_rows,
        video_paths=video_paths,
        ball_tracks=ball_tracks,
        cache=cache,
        model=model,
        mode=args.pose_mode,
        kpt_thr=args.keypoint_threshold,
        ball_tolerance_sec=args.ball_tolerance_sec,
        cache_only=args.cache_only,
    )
    if any(row.get("split") == "test_frozen" for row in train_out):
        raise AssertionError("test_frozen row leaked into train/validation output")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "touch_training_candidates.jsonl", train_out)
    write_jsonl(out_dir / "touch_training_test_frozen.jsonl", test_out)
    write_cache(cache_path, cache)
    videos = train_summary["videos"] + test_summary["videos"]
    status = "features_attached" if train_summary["pose_present_rows"] or test_summary["pose_present_rows"] else "no_pose_features"
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "runtime_status": runtime_status,
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "review_manifest": str(args.review_manifest),
        "detection_files": [str(path) for path in paths],
        "cache_path": str(cache_path),
        "pose_mode": args.pose_mode,
        "pose_device": args.pose_device,
        "keypoint_threshold": args.keypoint_threshold,
        "ball_tolerance_sec": args.ball_tolerance_sec,
        "threshold": args.threshold,
        "train_val": train_summary,
        "test_frozen": test_summary,
        "videos": videos,
    }
    write_json(out_dir / "touch_pose_feature_manifest.json", manifest)
    write_report(out_dir / "touch_pose_feature_report.md", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach RTMW pose proximity features to touch training candidates")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW_MANIFEST)
    parser.add_argument("--detections-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--detections-dir", type=Path, action="append", default=[])
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--ball-tolerance-sec", type=float, default=DEFAULT_BALL_TOLERANCE_SEC)
    parser.add_argument("--pose-mode", choices=["lightweight", "balanced", "performance"], default="performance")
    parser.add_argument("--pose-device", default="cpu")
    parser.add_argument("--keypoint-threshold", type=float, default=0.25)
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--cache-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    manifest = attach_dataset(parse_args())
    out_dir = Path(manifest["out_dir"])
    print(f"manifest: {out_dir / 'touch_pose_feature_manifest.json'}")
    print(f"report:   {out_dir / 'touch_pose_feature_report.md'}")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "runtime_status": manifest["runtime_status"],
                "train_val_rows": manifest["train_val"]["rows"],
                "train_val_pose_present_rows": manifest["train_val"]["pose_present_rows"],
                "test_frozen_rows": manifest["test_frozen"]["rows"],
                "test_frozen_pose_present_rows": manifest["test_frozen"]["pose_present_rows"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
