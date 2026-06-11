#!/usr/bin/env python3
"""Attach local ball+shoe visual mask features to touch candidates.

SAM2 is not available in the local environment, so this stage implements a
dependency-light proxy for the same hypothesis: the contact classifier needs
more local visual geometry around the ball and shoe than scalar pose distances
provide. It builds ball-centered polar/texture features and a nearest connected
edge/texture component from the actual video frame.

``local_mask_*`` fields are label-free automatic features. They are optional and
must prove lift in clip-disjoint contact metrics before HUD badge promotion.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import train_release_contact_classifier as contact


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "runs/release-27-public/touch_corpus_v1"
DEFAULT_DATASET_DIR = DEFAULT_CORPUS / "touch_training_dataset_v1"
DEFAULT_LABELS_DIR = DEFAULT_CORPUS / "visual_touch_labels"
DEFAULT_REVIEW_MANIFEST = DEFAULT_CORPUS / "touch_review_manifest.json"
LOCAL_MASK_FEATURE_VERSION = 1
SECTOR_NAMES = ("right", "up_right", "up", "up_left", "left", "down_left", "down", "down_right")


@dataclass(frozen=True)
class BallPoint:
    x: float
    y: float
    frame_index: int | None
    score: float | None = None


class FrameReader:
    def __init__(self, path: Path):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open video {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def read(self, frame_index: int) -> np.ndarray | None:
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


def ball_from_row(row: dict[str, Any], fps: float) -> BallPoint | None:
    for prefix in ("visual_crop_ball", "pose_ball", "vision_embedding_ball"):
        x = finite_float(row.get(f"{prefix}_x"))
        y = finite_float(row.get(f"{prefix}_y"))
        if x is None or y is None:
            continue
        frame_index = None
        for frame_key in (f"{prefix.replace('_ball', '')}_frame_index", "visual_crop_frame_index", "pose_frame_index"):
            value = finite_float(row.get(frame_key))
            if value is not None:
                frame_index = int(round(value))
                break
        if frame_index is None:
            frame_index = int(round(float(row.get("candidate_time_sec") or 0.0) * fps))
        return BallPoint(x=x, y=y, frame_index=max(0, frame_index), score=finite_float(row.get(f"{prefix}_score")))
    return None


def video_paths_by_name(review_manifest: Path) -> dict[str, Path]:
    doc = read_json(review_manifest)
    return {Path(str(item["video_name"])).name: Path(str(item["video_path"])) for item in doc.get("items", [])}


def default_features(status: str, ball: BallPoint | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "local_mask_feature_version": LOCAL_MASK_FEATURE_VERSION,
        "local_mask_feature_status": status,
        "local_mask_frame_index": None if ball is None else ball.frame_index,
        "local_mask_ball_x": None if ball is None else round(ball.x, 3),
        "local_mask_ball_y": None if ball is None else round(ball.y, 3),
        "local_mask_crop_size_px": None,
        "local_mask_ball_radius_px": None,
        "local_mask_ring_edge_density": None,
        "local_mask_ring_texture_frac": None,
        "local_mask_ring_dark_frac": None,
        "local_mask_ring_high_sat_frac": None,
        "local_mask_ring_skin_like_frac": None,
        "local_mask_left_right_texture_delta": None,
        "local_mask_up_down_texture_delta": None,
        "local_mask_diagonal_texture_delta": None,
        "local_mask_left_right_edge_delta": None,
        "local_mask_up_down_edge_delta": None,
        "local_mask_component_present": False,
        "local_mask_component_area_frac": None,
        "local_mask_component_centroid_dx_norm": None,
        "local_mask_component_centroid_dy_norm": None,
        "local_mask_component_dist_px": None,
        "local_mask_component_angle_sin": None,
        "local_mask_component_angle_cos": None,
        "local_mask_component_eccentricity": None,
        "local_mask_component_orientation_sin": None,
        "local_mask_component_orientation_cos": None,
        "local_mask_component_bbox_w_norm": None,
        "local_mask_component_bbox_h_norm": None,
        "local_mask_component_major_len_norm": None,
        "local_mask_component_minor_len_norm": None,
        "local_mask_component_edge_density": None,
        "local_mask_component_sat_mean": None,
        "local_mask_component_val_mean": None,
        "local_mask_component_dark_frac": None,
        "local_mask_component_high_sat_frac": None,
        "local_mask_component_skin_like_frac": None,
    }
    for sector in SECTOR_NAMES:
        out[f"local_mask_sector_{sector}_edge_density"] = None
        out[f"local_mask_sector_{sector}_sat_mean"] = None
        out[f"local_mask_sector_{sector}_val_mean"] = None
        out[f"local_mask_sector_{sector}_texture_frac"] = None
        out[f"local_mask_sector_{sector}_dark_frac"] = None
        out[f"local_mask_sector_{sector}_high_sat_frac"] = None
        out[f"local_mask_sector_{sector}_skin_like_frac"] = None
    return out


def crop_around_ball(frame: np.ndarray, ball: BallPoint, crop_size_px: int) -> tuple[np.ndarray, float, float] | None:
    h, w = frame.shape[:2]
    half = crop_size_px / 2.0
    x1 = max(0, int(math.floor(ball.x - half)))
    y1 = max(0, int(math.floor(ball.y - half)))
    x2 = min(w, int(math.ceil(ball.x + half)))
    y2 = min(h, int(math.ceil(ball.y + half)))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop, ball.x - x1, ball.y - y1


def sector_index(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    angles = np.arctan2(-dy, dx)
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    return np.floor(((angles + np.pi / 8) % (2 * np.pi)) / (np.pi / 4)).astype(np.int32)


def component_features(
    *,
    texture_mask: np.ndarray,
    edge_mask: np.ndarray,
    dark_mask: np.ndarray,
    high_sat_mask: np.ndarray,
    skin_like_mask: np.ndarray,
    hsv: np.ndarray,
    bx: float,
    by: float,
    ball_radius_px: float,
) -> dict[str, Any]:
    component_area_threshold = max(6, int(round(ball_radius_px * 1.2)))
    component_input = cv2.morphologyEx(texture_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _count, labels, stats, centroids = cv2.connectedComponentsWithStats(component_input, connectivity=8)
    best = None
    best_score = None
    for label in range(1, _count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < component_area_threshold:
            continue
        cx, cy = map(float, centroids[label])
        dist = math.hypot(cx - bx, cy - by)
        if dist < ball_radius_px * 0.8:
            continue
        score = dist / max(1.0, math.sqrt(area))
        if best_score is None or score < best_score:
            best = label
            best_score = score
    if best is None:
        return {}
    mask = labels == best
    ys, xs = np.nonzero(mask)
    cx, cy = map(float, centroids[best])
    dist = math.hypot(cx - bx, cy - by)
    angle = math.atan2(cy - by, cx - bx)
    cov = np.cov(np.vstack([xs, ys])) if len(xs) > 2 else np.eye(2)
    eigvals, eigvecs = np.linalg.eigh(cov)
    largest = float(max(eigvals)) if len(eigvals) else 0.0
    smallest = float(min(eigvals)) if len(eigvals) else 0.0
    eccentricity = None if largest <= 1e-6 else 1.0 - smallest / largest
    major_index = int(np.argmax(eigvals)) if len(eigvals) else 0
    major_vec = eigvecs[:, major_index] if getattr(eigvecs, "size", 0) else np.array([1.0, 0.0])
    orientation = math.atan2(float(major_vec[1]), float(major_vec[0]))
    x, y, w, h, area = (int(stats[best, cv2.CC_STAT_LEFT]), int(stats[best, cv2.CC_STAT_TOP]), int(stats[best, cv2.CC_STAT_WIDTH]), int(stats[best, cv2.CC_STAT_HEIGHT]), int(stats[best, cv2.CC_STAT_AREA]))
    return {
        "local_mask_component_present": True,
        "local_mask_component_area_frac": float(np.count_nonzero(mask) / max(1, texture_mask.size)),
        "local_mask_component_centroid_dx_norm": float((cx - bx) / max(1.0, ball_radius_px)),
        "local_mask_component_centroid_dy_norm": float((cy - by) / max(1.0, ball_radius_px)),
        "local_mask_component_dist_px": float(dist),
        "local_mask_component_angle_sin": float(math.sin(angle)),
        "local_mask_component_angle_cos": float(math.cos(angle)),
        "local_mask_component_eccentricity": eccentricity,
        "local_mask_component_orientation_sin": float(math.sin(orientation)),
        "local_mask_component_orientation_cos": float(math.cos(orientation)),
        "local_mask_component_bbox_w_norm": float(w / max(1.0, ball_radius_px)),
        "local_mask_component_bbox_h_norm": float(h / max(1.0, ball_radius_px)),
        "local_mask_component_major_len_norm": float(math.sqrt(max(largest, 0.0)) / max(1.0, ball_radius_px)),
        "local_mask_component_minor_len_norm": float(math.sqrt(max(smallest, 0.0)) / max(1.0, ball_radius_px)),
        "local_mask_component_edge_density": float(np.mean(edge_mask[mask] > 0)) if np.any(mask) else None,
        "local_mask_component_sat_mean": float(np.mean(hsv[:, :, 1][mask])) if np.any(mask) else None,
        "local_mask_component_val_mean": float(np.mean(hsv[:, :, 2][mask])) if np.any(mask) else None,
        "local_mask_component_dark_frac": float(np.mean(dark_mask[mask])) if np.any(mask) else None,
        "local_mask_component_high_sat_frac": float(np.mean(high_sat_mask[mask])) if np.any(mask) else None,
        "local_mask_component_skin_like_frac": float(np.mean(skin_like_mask[mask])) if np.any(mask) else None,
    }


def extract_features(frame: np.ndarray, ball: BallPoint, *, crop_size_px: int, ball_radius_px: float) -> dict[str, Any]:
    item = crop_around_ball(frame, ball, crop_size_px)
    if item is None:
        return default_features("empty_crop", ball)
    crop, bx, by = item
    h, w = crop.shape[:2]
    if h <= 4 or w <= 4:
        return default_features("small_crop", ball)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 45, 135)
    yy, xx = np.indices((h, w))
    dx = xx.astype(float) - bx
    dy = yy.astype(float) - by
    radius = np.sqrt(dx * dx + dy * dy)
    ring = (radius >= ball_radius_px * 1.2) & (radius <= crop_size_px * 0.48)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    hue = hsv[:, :, 0]
    dark = val < 95
    high_sat = sat > 70
    skin_like = (((hue <= 25) | (hue >= 150)) & (sat < 90) & (val > 80))
    texture = ring & ((edges > 0) | high_sat | dark)
    features = default_features("ok", ball)
    features.update(
        {
            "local_mask_crop_size_px": crop_size_px,
            "local_mask_ball_radius_px": ball_radius_px,
        }
    )
    if np.any(ring):
        left = ring & (dx < 0)
        right = ring & (dx >= 0)
        up = ring & (dy < 0)
        down = ring & (dy >= 0)
        slash_a = ring & ((dx * dy) < 0)
        slash_b = ring & ((dx * dy) >= 0)
        features.update(
            {
                "local_mask_ring_edge_density": float(np.mean(edges[ring] > 0)),
                "local_mask_ring_texture_frac": float(np.mean(texture[ring])),
                "local_mask_ring_dark_frac": float(np.mean(dark[ring])),
                "local_mask_ring_high_sat_frac": float(np.mean(high_sat[ring])),
                "local_mask_ring_skin_like_frac": float(np.mean(skin_like[ring])),
                "local_mask_left_right_texture_delta": float(np.mean(texture[left]) - np.mean(texture[right])) if np.any(left) and np.any(right) else None,
                "local_mask_up_down_texture_delta": float(np.mean(texture[up]) - np.mean(texture[down])) if np.any(up) and np.any(down) else None,
                "local_mask_diagonal_texture_delta": float(np.mean(texture[slash_a]) - np.mean(texture[slash_b])) if np.any(slash_a) and np.any(slash_b) else None,
                "local_mask_left_right_edge_delta": float(np.mean(edges[left] > 0) - np.mean(edges[right] > 0)) if np.any(left) and np.any(right) else None,
                "local_mask_up_down_edge_delta": float(np.mean(edges[up] > 0) - np.mean(edges[down] > 0)) if np.any(up) and np.any(down) else None,
            }
        )
    sectors = sector_index(dx, dy)
    for idx, name in enumerate(SECTOR_NAMES):
        mask = ring & (sectors == idx)
        if not np.any(mask):
            continue
        features[f"local_mask_sector_{name}_edge_density"] = float(np.mean(edges[mask] > 0))
        features[f"local_mask_sector_{name}_sat_mean"] = float(np.mean(hsv[:, :, 1][mask]))
        features[f"local_mask_sector_{name}_val_mean"] = float(np.mean(hsv[:, :, 2][mask]))
        features[f"local_mask_sector_{name}_texture_frac"] = float(np.mean(texture[mask]))
        features[f"local_mask_sector_{name}_dark_frac"] = float(np.mean(dark[mask]))
        features[f"local_mask_sector_{name}_high_sat_frac"] = float(np.mean(high_sat[mask]))
        features[f"local_mask_sector_{name}_skin_like_frac"] = float(np.mean(skin_like[mask]))
    features.update(
        component_features(
            texture_mask=texture,
            edge_mask=edges > 0,
            dark_mask=dark,
            high_sat_mask=high_sat,
            skin_like_mask=skin_like,
            hsv=hsv,
            bx=bx,
            by=by,
            ball_radius_px=ball_radius_px,
        )
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
    crop_size_px: int,
    ball_radius_px: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = contact_labeled_keys(rows, labels_dir, label_match_tolerance_sec) if contact_labeled_only else set()
    readers: dict[str, FrameReader] = {}
    out: list[dict[str, Any]] = []
    per_video: dict[str, Counter[str]] = defaultdict(Counter)
    try:
        for row in rows:
            new_row = dict(row)
            video_name = Path(str(row.get("video_name") or row.get("source_video") or "")).name
            requested = not contact_labeled_only or row_key(row) in selected
            if not requested:
                features = default_features("skipped_not_requested")
            elif video_name not in video_paths:
                features = default_features("missing_video_path")
            else:
                reader = readers.get(video_name)
                if reader is None:
                    reader = FrameReader(video_paths[video_name])
                    readers[video_name] = reader
                ball = ball_from_row(row, reader.fps)
                if ball is None:
                    features = default_features("missing_ball")
                elif ball.frame_index is None:
                    features = default_features("missing_frame_index", ball)
                else:
                    frame = reader.read(ball.frame_index)
                    if frame is None:
                        features = default_features("missing_frame", ball)
                    else:
                        features = extract_features(frame, ball, crop_size_px=crop_size_px, ball_radius_px=ball_radius_px)
            new_row.update(features)
            out.append(new_row)
            status = str(features.get("local_mask_feature_status") or "unknown")
            stats = per_video[str(row.get("video_id") or video_name)]
            stats["rows"] += 1
            stats[status] += 1
    finally:
        for reader in readers.values():
            reader.release()
    summary = {
        "rows": len(out),
        "requested_rows": sum(1 for row in out if row.get("local_mask_feature_status") != "skipped_not_requested"),
        "ok_rows": sum(1 for row in out if row.get("local_mask_feature_status") == "ok"),
        "component_rows": sum(1 for row in out if row.get("local_mask_component_present")),
        "status_counts": dict(Counter(str(row.get("local_mask_feature_status") or "unknown") for row in out)),
        "videos": [{"video_id": video_id, **dict(counter)} for video_id, counter in sorted(per_video.items())],
    }
    return out, summary


def attach_dataset(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else dataset_dir
    video_paths = video_paths_by_name(args.review_manifest.resolve())
    train_rows = read_jsonl(dataset_dir / "touch_training_candidates.jsonl")
    test_rows = read_jsonl(dataset_dir / "touch_training_test_frozen.jsonl")
    train_out, train_summary = attach_rows(
        train_rows,
        video_paths=video_paths,
        labels_dir=args.labels_dir.resolve(),
        label_match_tolerance_sec=args.label_match_tolerance_sec,
        contact_labeled_only=args.contact_labeled_only,
        crop_size_px=args.crop_size_px,
        ball_radius_px=args.ball_radius_px,
    )
    test_out, test_summary = attach_rows(
        test_rows,
        video_paths=video_paths,
        labels_dir=args.labels_dir.resolve(),
        label_match_tolerance_sec=args.label_match_tolerance_sec,
        contact_labeled_only=args.contact_labeled_only,
        crop_size_px=args.crop_size_px,
        ball_radius_px=args.ball_radius_px,
    )
    if any(row.get("split") == "test_frozen" for row in train_out):
        raise AssertionError("test_frozen row leaked into train/validation local-mask output")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "touch_training_candidates.jsonl", train_out)
    write_jsonl(out_dir / "touch_training_test_frozen.jsonl", test_out)
    status = "features_attached" if train_summary["ok_rows"] or test_summary["ok_rows"] else "no_local_mask_features"
    manifest = {
        "schema_version": 1,
        "local_mask_feature_version": LOCAL_MASK_FEATURE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "review_manifest": str(args.review_manifest.resolve()),
        "labels_dir": str(args.labels_dir.resolve()),
        "contact_labeled_only": args.contact_labeled_only,
        "crop_size_px": args.crop_size_px,
        "ball_radius_px": args.ball_radius_px,
        "train_val": train_summary,
        "test_frozen": test_summary,
        "notes": [
            "local_mask_* fields are label-free automatic crop/mask proxy features.",
            "This stage is a dependency-light substitute for SAM2-style local shoe/foot masks.",
            "Feature promotion requires clip-disjoint lift over the existing contact classifier.",
        ],
    }
    write_json(out_dir / "touch_local_mask_feature_manifest.json", manifest)
    write_report(out_dir / "touch_local_mask_feature_report.md", manifest)
    return manifest


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Touch Local Mask Feature Attachment",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Dataset dir: `{manifest['dataset_dir']}`",
        f"- Out dir: `{manifest['out_dir']}`",
        f"- Contact-labeled only: `{manifest['contact_labeled_only']}`",
        f"- Crop size: `{manifest['crop_size_px']}` px",
        f"- Ball radius proxy: `{manifest['ball_radius_px']}` px",
        f"- Train/val rows: `{manifest['train_val']['rows']}`",
        f"- Train/val requested rows: `{manifest['train_val']['requested_rows']}`",
        f"- Train/val ok rows: `{manifest['train_val']['ok_rows']}`",
        f"- Train/val component rows: `{manifest['train_val']['component_rows']}`",
        f"- Frozen-test rows: `{manifest['test_frozen']['rows']}`",
        f"- Frozen-test requested rows: `{manifest['test_frozen']['requested_rows']}`",
        f"- Frozen-test ok rows: `{manifest['test_frozen']['ok_rows']}`",
        f"- Frozen-test component rows: `{manifest['test_frozen']['component_rows']}`",
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
    parser = argparse.ArgumentParser(description="Attach local ball+shoe crop/mask proxy features")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW_MANIFEST)
    parser.add_argument("--label-match-tolerance-sec", type=float, default=0.08)
    parser.add_argument("--contact-labeled-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--crop-size-px", type=int, default=192)
    parser.add_argument("--ball-radius-px", type=float, default=12.0)
    return parser.parse_args()


def main() -> None:
    manifest = attach_dataset(parse_args())
    out_dir = Path(manifest["out_dir"])
    print(f"manifest: {out_dir / 'touch_local_mask_feature_manifest.json'}")
    print(f"report:   {out_dir / 'touch_local_mask_feature_report.md'}")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "train_val_requested_rows": manifest["train_val"]["requested_rows"],
                "train_val_ok_rows": manifest["train_val"]["ok_rows"],
                "test_frozen_requested_rows": manifest["test_frozen"]["requested_rows"],
                "test_frozen_ok_rows": manifest["test_frozen"]["ok_rows"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
