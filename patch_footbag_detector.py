#!/usr/bin/env python3
"""Train and evaluate a reviewed-label patch detector for footbag objectness."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class LabelBox:
    class_id: int
    center: tuple[float, float]
    bbox: tuple[float, float, float, float]
    width: float
    height: float


@dataclass(frozen=True)
class ImageRecord:
    split: str
    image_path: Path
    label_path: Path
    labels: tuple[LabelBox, ...]


@dataclass(frozen=True)
class HardNegativePoint:
    split: str
    image_path: Path
    source_path: str
    item_id: str | None
    source_video: str | None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def portable(path: Path | None, base: Path = ROOT) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except (OSError, ValueError):
        return path.name if path.is_absolute() else str(path)


def image_key(path: Path, dataset: Path) -> str:
    try:
        return str(path.resolve().relative_to(dataset.resolve()))
    except (OSError, ValueError):
        return path.name


def label_path_for_image(dataset: Path, image: Path) -> Path:
    return dataset / "labels" / image.relative_to(dataset / "images").with_suffix(".txt")


def read_yolo_labels(label_path: Path, *, width: int, height: int) -> tuple[LabelBox, ...]:
    if not label_path.exists():
        return tuple()
    labels: list[LabelBox] = []
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"{label_path}:{line_no} needs class x y w h")
        class_id = int(float(parts[0]))
        x_c = float(parts[1]) * width
        y_c = float(parts[2]) * height
        box_w = float(parts[3]) * width
        box_h = float(parts[4]) * height
        labels.append(
            LabelBox(
                class_id=class_id,
                center=(x_c, y_c),
                bbox=(x_c - box_w / 2.0, y_c - box_h / 2.0, x_c + box_w / 2.0, y_c + box_h / 2.0),
                width=box_w,
                height=box_h,
            )
        )
    return tuple(labels)


def load_image_records(dataset: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for image_path in sorted((dataset / "images").glob("*/*")):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        label_path = label_path_for_image(dataset, image_path)
        records.append(
            ImageRecord(
                split=image_path.parent.name,
                image_path=image_path,
                label_path=label_path,
                labels=read_yolo_labels(label_path, width=width, height=height),
            )
        )
    return records


def load_hard_negative_points(dataset: Path) -> list[HardNegativePoint]:
    jsonl_path = dataset / "hard_negatives" / "points.jsonl"
    if not jsonl_path.exists():
        return []
    points: list[HardNegativePoint] = []
    for line_no, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{jsonl_path}:{line_no} is not valid JSON") from exc
        source_path = str(record.get("crop") or record.get("yolo_empty_label_image") or "")
        if not source_path:
            continue
        image_path = Path(source_path)
        if not image_path.is_absolute():
            image_path = dataset / image_path
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS or not image_path.exists():
            continue
        points.append(
            HardNegativePoint(
                split=str(record.get("split") or "train"),
                image_path=image_path,
                source_path=source_path,
                item_id=None if record.get("item_id") is None else str(record.get("item_id")),
                source_video=None if record.get("source_video") is None else str(record.get("source_video")),
            )
        )
    return points


def crop_square(image: np.ndarray, center: tuple[float, float], size: int) -> np.ndarray:
    height, width = image.shape[:2]
    half = size // 2
    x = int(round(center[0]))
    y = int(round(center[1]))
    left = x - half
    top = y - half
    right = left + size
    bottom = top + size
    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - width)
    pad_bottom = max(0, bottom - height)
    if pad_left or pad_top or pad_right or pad_bottom:
        image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
        left += pad_left
        right += pad_left
        top += pad_top
        bottom += pad_top
    return image[top:bottom, left:right]


def feature_vector(crop: np.ndarray, feature_size: int = 32) -> np.ndarray:
    resized = cv2.resize(crop, (feature_size, feature_size), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    hsv_small = cv2.resize(hsv, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1) / 255.0
    lab_stats = np.concatenate(
        [
            lab.reshape(-1, 3).mean(axis=0),
            lab.reshape(-1, 3).std(axis=0),
            hsv.reshape(-1, 3).mean(axis=0),
            hsv.reshape(-1, 3).std(axis=0),
        ]
    ).astype(np.float32) / 255.0
    hist_h = cv2.calcHist([hsv], [0], None, [18], [0, 180]).reshape(-1)
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).reshape(-1)
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).reshape(-1)
    hist = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
    hist = hist / max(1.0, float(hist.sum()))
    hog = cv2.HOGDescriptor((32, 32), (16, 16), (8, 8), (8, 8), 9).compute(gray).reshape(-1).astype(np.float32)
    return np.concatenate([hsv_small, lab_stats, hist, hog]).astype(np.float32)


def center_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def too_near_label(center: tuple[float, float], labels: tuple[LabelBox, ...], min_distance: float) -> bool:
    return any(center_distance(center, label.center) < min_distance for label in labels)


def red_object_proposals(image: np.ndarray, *, max_candidates: int = 80) -> list[tuple[float, float]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    red = (((hue < 14) | (hue > 162)) & (sat > 45) & (val > 35)).astype(np.uint8) * 255
    magenta = ((hue > 135) & (sat > 35) & (val > 55)).astype(np.uint8) * 255
    dark = ((val < 92) & (sat > 24)).astype(np.uint8) * 255
    color_mask = cv2.bitwise_or(red, magenta)
    object_mask = cv2.bitwise_or(color_mask, dark)
    object_mask = cv2.medianBlur(object_mask, 3)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, float, float]] = []
    height, width = image.shape[:2]
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 3.0 or area > 3200.0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w > 150 or h > 150:
            continue
        if x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1:
            pass
        moments = cv2.moments(contour)
        if moments["m00"]:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            cx = x + w / 2.0
            cy = y + h / 2.0
        compactness = area / max(1.0, float(w * h))
        red_pixels = float(cv2.countNonZero(color_mask[y : y + h, x : x + w]))
        source_bonus = 2.0 if red_pixels else 0.75
        candidates.append((cx, cy, area * compactness * source_bonus))

    corner_mask = cv2.dilate(object_mask, np.ones((9, 9), np.uint8))
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_candidates, qualityLevel=0.015, minDistance=12, mask=corner_mask)
    if corners is not None:
        for corner in corners.reshape(-1, 2):
            cx, cy = float(corner[0]), float(corner[1])
            if cx < 2 or cy < 2 or cx > width - 3 or cy > height - 3:
                continue
            local = object_mask[max(0, int(cy) - 6) : min(height, int(cy) + 7), max(0, int(cx) - 6) : min(width, int(cx) + 7)]
            density = float(cv2.countNonZero(local)) / max(1.0, float(local.size))
            if density <= 0.02:
                continue
            candidates.append((cx, cy, 25.0 + density * 100.0))
    candidates.sort(key=lambda item: item[2], reverse=True)
    centers: list[tuple[float, float]] = []
    for cx, cy, _score in candidates:
        if all(center_distance((cx, cy), existing) > 10.0 for existing in centers):
            centers.append((cx, cy))
        if len(centers) >= max_candidates:
            break
    return centers


def random_negative_centers(
    image: np.ndarray,
    labels: tuple[LabelBox, ...],
    *,
    rng: random.Random,
    count: int,
    crop_size: int,
) -> list[tuple[float, float]]:
    height, width = image.shape[:2]
    centers: list[tuple[float, float]] = []
    attempts = 0
    min_distance = crop_size * 0.8
    while len(centers) < count and attempts < count * 60:
        attempts += 1
        center = (rng.uniform(0, width - 1), rng.uniform(0, height - 1))
        if too_near_label(center, labels, min_distance):
            continue
        centers.append(center)
    return centers


def split_records(records: list[ImageRecord], split: str) -> list[ImageRecord]:
    return [record for record in records if record.split == split]


def split_hard_negative_points(points: list[HardNegativePoint], split: str) -> list[HardNegativePoint]:
    return [point for point in points if point.split == split]


def build_samples(
    records: list[ImageRecord],
    *,
    hard_negative_points: list[HardNegativePoint] | None = None,
    crop_size: int,
    jitter: int,
    negatives_per_image: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rng = random.Random(seed)
    features: list[np.ndarray] = []
    targets: list[int] = []
    rows: list[dict[str, Any]] = []
    for record in records:
        image = cv2.imread(str(record.image_path))
        if image is None:
            continue
        positive_centers: list[tuple[float, float]] = []
        for label_index, label in enumerate(record.labels):
            positive_centers.append(label.center)
            jitter_radius = max(2.0, min(10.0, max(label.width, label.height) * 0.18))
            for _ in range(jitter):
                positive_centers.append(
                    (
                        label.center[0] + rng.uniform(-jitter_radius, jitter_radius),
                        label.center[1] + rng.uniform(-jitter_radius, jitter_radius),
                    )
                )
            for center in positive_centers[-(jitter + 1) :]:
                features.append(feature_vector(crop_square(image, center, crop_size)))
                targets.append(1)
                rows.append(
                    {
                        "split": record.split,
                        "image": record.image_path.name,
                        "label_index": label_index,
                        "sample": "positive",
                        "x": round(center[0], 3),
                        "y": round(center[1], 3),
                        "source_video": None,
                        "item_id": None,
                    }
                )

        proposal_negatives = [
            center
            for center in red_object_proposals(image, max_candidates=negatives_per_image * 2)
            if not too_near_label(center, record.labels, crop_size * 0.8)
        ][:negatives_per_image]
        random_negatives = random_negative_centers(
            image,
            record.labels,
            rng=rng,
            count=max(negatives_per_image, len(record.labels) * negatives_per_image),
            crop_size=crop_size,
        )
        for center in (proposal_negatives + random_negatives)[: max(negatives_per_image, len(record.labels) * negatives_per_image)]:
            features.append(feature_vector(crop_square(image, center, crop_size)))
            targets.append(0)
            rows.append(
                {
                    "split": record.split,
                    "image": record.image_path.name,
                    "label_index": None,
                    "sample": "negative",
                    "x": round(center[0], 3),
                    "y": round(center[1], 3),
                    "source_video": None,
                    "item_id": None,
                }
            )
    for point in hard_negative_points or []:
        image = cv2.imread(str(point.image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        center = ((width - 1) / 2.0, (height - 1) / 2.0)
        features.append(feature_vector(crop_square(image, center, crop_size)))
        targets.append(0)
        rows.append(
            {
                "split": point.split,
                "image": point.source_path,
                "label_index": None,
                "sample": "hard_negative_point",
                "x": round(center[0], 3),
                "y": round(center[1], 3),
                "source_video": point.source_video,
                "item_id": point.item_id,
            }
        )
    if not features:
        return np.empty((0, 1), dtype=np.float32), np.empty((0,), dtype=np.int64), rows
    return np.vstack(features).astype(np.float32), np.array(targets, dtype=np.int64), rows


def choose_threshold(scores: np.ndarray, targets: np.ndarray) -> tuple[float, dict[str, Any]]:
    if len(scores) == 0:
        return 0.5, {"precision": None, "recall": None, "f1": None}
    thresholds = sorted(set(float(value) for value in np.linspace(0.05, 0.95, 91)) | set(float(value) for value in scores))
    best = (0.5, -1.0, 0.0, 0.0)
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(np.int64)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets,
            predictions,
            average="binary",
            zero_division=0,
        )
        if f1 > best[1] or (f1 == best[1] and precision > best[2]):
            best = (threshold, float(f1), float(precision), float(recall))
    threshold, f1, precision, recall = best
    return float(threshold), {"precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}


def sample_metrics(model: Pipeline, features: np.ndarray, targets: np.ndarray, threshold: float) -> dict[str, Any]:
    if len(targets) == 0:
        return {
            "samples": 0,
            "positives": 0,
            "negatives": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
        }
    scores = model.predict_proba(features)[:, 1]
    predictions = (scores >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(targets, scores) if len(set(targets.tolist())) > 1 else None
    except ValueError:
        auc = None
    return {
        "samples": int(len(targets)),
        "positives": int(targets.sum()),
        "negatives": int(len(targets) - targets.sum()),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "roc_auc": None if auc is None else round(float(auc), 6),
        "mean_positive_score": None if not targets.sum() else round(float(scores[targets == 1].mean()), 6),
        "mean_negative_score": None if int(len(targets) - targets.sum()) == 0 else round(float(scores[targets == 0].mean()), 6),
    }


def build_model(kind: str, seed: int) -> Pipeline:
    if kind == "logistic":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if kind == "extra-trees":
        return Pipeline(
            [
                (
                    "classifier",
                    ExtraTreesClassifier(
                        n_estimators=400,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                )
            ]
        )
    raise ValueError(f"Unsupported patch detector model kind: {kind}")


def train_patch_detector(
    *,
    dataset: Path,
    out_dir: Path,
    crop_size: int = 96,
    jitter: int = 4,
    negatives_per_image: int = 8,
    seed: int = 1337,
    model_kind: str = "logistic",
    dry_run: bool = False,
) -> dict[str, Any]:
    dataset = dataset.expanduser().resolve()
    records = load_image_records(dataset)
    hard_negative_points = load_hard_negative_points(dataset)
    train_records = split_records(records, "train")
    validation_records = split_records(records, "validation")
    test_records = split_records(records, "test")
    train_hard_negative_points = split_hard_negative_points(hard_negative_points, "train")
    validation_hard_negative_points = split_hard_negative_points(hard_negative_points, "validation")
    test_hard_negative_points = split_hard_negative_points(hard_negative_points, "test")
    summary: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trainer": f"sklearn-{model_kind}-patch-objectness",
        "dataset": portable(dataset),
        "out_dir": portable(out_dir),
        "parameters": {
            "model_kind": model_kind,
            "crop_size": crop_size,
            "feature_size": 32,
            "jitter": jitter,
            "negatives_per_image": negatives_per_image,
            "seed": seed,
        },
        "records": {
            "train": len(train_records),
            "validation": len(validation_records),
            "test": len(test_records),
        },
        "hard_negative_points": {
            "total": len(hard_negative_points),
            "train": len(train_hard_negative_points),
            "validation": len(validation_hard_negative_points),
            "test": len(test_hard_negative_points),
        },
        "model_path": None,
        "threshold": None,
        "sample_metrics": {},
    }
    if dry_run:
        return summary

    train_x, train_y, train_rows = build_samples(
        train_records,
        hard_negative_points=train_hard_negative_points,
        crop_size=crop_size,
        jitter=jitter,
        negatives_per_image=negatives_per_image,
        seed=seed,
    )
    if len(set(train_y.tolist())) < 2:
        raise RuntimeError("Patch detector training needs both positive and negative samples.")
    validation_x, validation_y, _validation_rows = build_samples(
        validation_records,
        hard_negative_points=validation_hard_negative_points,
        crop_size=crop_size,
        jitter=0,
        negatives_per_image=negatives_per_image,
        seed=seed + 1,
    )
    test_x, test_y, _test_rows = build_samples(
        test_records,
        hard_negative_points=test_hard_negative_points,
        crop_size=crop_size,
        jitter=0,
        negatives_per_image=negatives_per_image,
        seed=seed + 2,
    )

    model = build_model(model_kind, seed)
    model.fit(train_x, train_y)
    threshold_source = "validation" if len(validation_y) and len(set(validation_y.tolist())) > 1 else "train"
    threshold_x = validation_x if threshold_source == "validation" else train_x
    threshold_y = validation_y if threshold_source == "validation" else train_y
    threshold_scores = model.predict_proba(threshold_x)[:, 1]
    threshold, threshold_metrics = choose_threshold(threshold_scores, threshold_y)
    artifact = {
        "schema_version": 1,
        "model": model,
        "threshold": threshold,
        "crop_size": crop_size,
        "feature_size": 32,
        "model_kind": model_kind,
        "class_names": ["background", "footbag"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "patch_footbag_detector.joblib"
    joblib.dump(artifact, model_path)

    samples_csv = out_dir / "patch_training_samples.csv"
    with samples_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "image", "label_index", "sample", "x", "y", "source_video", "item_id"])
        writer.writeheader()
        writer.writerows(train_rows)

    summary["model_path"] = portable(model_path)
    summary["threshold"] = round(float(threshold), 6)
    summary["threshold_source"] = threshold_source
    summary["threshold_metrics"] = threshold_metrics
    summary["sample_metrics"] = {
        "train": sample_metrics(model, train_x, train_y, threshold),
        "validation": sample_metrics(model, validation_x, validation_y, threshold),
        "test": sample_metrics(model, test_x, test_y, threshold),
    }
    summary["outputs"] = {
        "model": portable(model_path),
        "manifest": portable(out_dir / "patch_training_manifest.json"),
        "training_samples": portable(samples_csv),
    }
    write_json(out_dir / "patch_training_manifest.json", summary)
    return summary


def load_patch_artifact(path: Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise RuntimeError(f"Invalid patch detector artifact: {path}")
    return artifact


def nms_points(detections: list[dict[str, Any]], min_distance: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for detection in sorted(detections, key=lambda item: float(item["confidence"]), reverse=True):
        center = (float(detection["center"][0]), float(detection["center"][1]))
        if all(center_distance(center, (float(item["center"][0]), float(item["center"][1]))) >= min_distance for item in kept):
            kept.append(detection)
    return kept


def detect_patches_in_image(
    image: np.ndarray,
    artifact: dict[str, Any],
    *,
    threshold: float | None = None,
    max_candidates: int = 120,
    max_detections: int = 8,
) -> list[dict[str, Any]]:
    crop_size = int(artifact.get("crop_size") or 96)
    model: Pipeline = artifact["model"]
    threshold_value = float(artifact.get("threshold") if threshold is None else threshold)
    centers = red_object_proposals(image, max_candidates=max_candidates)
    if not centers:
        return []
    features = np.vstack([feature_vector(crop_square(image, center, crop_size)) for center in centers])
    scores = model.predict_proba(features)[:, 1]
    detections: list[dict[str, Any]] = []
    for center, score in zip(centers, scores, strict=False):
        if float(score) < threshold_value:
            continue
        x, y = center
        half = crop_size / 2.0
        detections.append(
            {
                "center": [round(float(x), 3), round(float(y), 3)],
                "bbox": [round(float(x - half), 3), round(float(y - half), 3), round(float(x + half), 3), round(float(y + half), 3)],
                "confidence": round(float(score), 6),
                "source": "patch_detector",
            }
        )
    return nms_points(detections, min_distance=max(12.0, crop_size * 0.35))[:max_detections]


def match_label(label: LabelBox, detections: list[dict[str, Any]], *, tolerance_px: float, box_tolerance_multiplier: float) -> dict[str, Any]:
    threshold = max(tolerance_px, max(label.width, label.height) * box_tolerance_multiplier)
    if not detections:
        return {
            "result": "missing_prediction",
            "center_error_px": None,
            "threshold_px": round(float(threshold), 3),
            "confidence": None,
            "prediction_x": None,
            "prediction_y": None,
        }
    expected = label.center
    error, detection = min(
        (
            center_distance(expected, (float(item["center"][0]), float(item["center"][1]))),
            item,
        )
        for item in detections
    )
    center = detection["center"]
    return {
        "result": "pass" if error <= threshold else "fail_center",
        "center_error_px": round(float(error), 3),
        "threshold_px": round(float(threshold), 3),
        "confidence": detection["confidence"],
        "prediction_x": center[0],
        "prediction_y": center[1],
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if row["kind"] == "positive_label"]
    hard_negatives = [row for row in rows if row["kind"] == "hard_negative_image"]
    errors = [float(row["center_error_px"]) for row in positives if row.get("center_error_px") is not None]
    confidences = [float(row["confidence"]) for row in positives if row.get("confidence") is not None]
    return {
        "positive_labels": len(positives),
        "passed_positive_labels": sum(1 for row in positives if row["result"] == "pass"),
        "missing_predictions": sum(1 for row in positives if row["result"] == "missing_prediction"),
        "failed_center": sum(1 for row in positives if row["result"] == "fail_center"),
        "pass_rate": None if not positives else round(sum(1 for row in positives if row["result"] == "pass") / len(positives), 6),
        "mean_center_error_px": None if not errors else round(mean(errors), 3),
        "mean_confidence": None if not confidences else round(mean(confidences), 6),
        "hard_negative_images": len(hard_negatives),
        "false_positive_hard_negative_images": sum(1 for row in hard_negatives if row["result"] == "fail_false_positive"),
        "hard_negative_false_positive_rate": None
        if not hard_negatives
        else round(sum(1 for row in hard_negatives if row["result"] == "fail_false_positive") / len(hard_negatives), 6),
    }


def write_eval_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "kind",
        "split",
        "image",
        "label_index",
        "expected_x",
        "expected_y",
        "prediction_x",
        "prediction_y",
        "center_error_px",
        "threshold_px",
        "confidence",
        "detections",
        "result",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_eval_md(path: Path, summary: dict[str, Any]) -> None:
    overall = summary["overall"]
    lines = [
        "# Patch Detector Evaluation",
        "",
        f"- Positive labels: `{overall['positive_labels']}`",
        f"- Pass rate: `{overall['pass_rate']}`",
        f"- Missing predictions: `{overall['missing_predictions']}`",
        f"- Mean center error px: `{overall['mean_center_error_px']}`",
        f"- Mean confidence: `{overall['mean_confidence']}`",
        f"- Hard-negative false-positive rate: `{overall['hard_negative_false_positive_rate']}`",
        "",
        "## By Split",
        "",
        "| Split | Labels | Pass Rate | Missing | Mean Error Px | Mean Confidence | Hard-Neg FP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, metrics in sorted(summary["by_split"].items()):
        lines.append(
            f"| {split} | {metrics['positive_labels']} | {metrics['pass_rate']} | "
            f"{metrics['missing_predictions']} | {metrics['mean_center_error_px']} | "
            f"{metrics['mean_confidence']} | {metrics['hard_negative_false_positive_rate']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_patch_detector(
    *,
    dataset: Path,
    model_path: Path,
    out_dir: Path,
    threshold: float | None = None,
    tolerance_px: float = 24.0,
    box_tolerance_multiplier: float = 0.75,
    max_candidates: int = 120,
    max_detections: int = 8,
) -> dict[str, Any]:
    dataset = dataset.expanduser().resolve()
    artifact = load_patch_artifact(model_path)
    records = load_image_records(dataset)
    rows: list[dict[str, Any]] = []
    for record in records:
        image = cv2.imread(str(record.image_path))
        if image is None:
            continue
        detections = detect_patches_in_image(
            image,
            artifact,
            threshold=threshold,
            max_candidates=max_candidates,
            max_detections=max_detections,
        )
        key = image_key(record.image_path, dataset)
        if not record.labels:
            rows.append(
                {
                    "kind": "hard_negative_image",
                    "split": record.split,
                    "image": key,
                    "label_index": None,
                    "expected_x": None,
                    "expected_y": None,
                    "prediction_x": None if not detections else detections[0]["center"][0],
                    "prediction_y": None if not detections else detections[0]["center"][1],
                    "center_error_px": None,
                    "threshold_px": None,
                    "confidence": None if not detections else detections[0]["confidence"],
                    "detections": len(detections),
                    "result": "fail_false_positive" if detections else "pass_no_prediction",
                }
            )
            continue
        for label_index, label in enumerate(record.labels):
            match = match_label(
                label,
                detections,
                tolerance_px=tolerance_px,
                box_tolerance_multiplier=box_tolerance_multiplier,
            )
            rows.append(
                {
                    "kind": "positive_label",
                    "split": record.split,
                    "image": key,
                    "label_index": label_index,
                    "expected_x": round(label.center[0], 3),
                    "expected_y": round(label.center[1], 3),
                    "detections": len(detections),
                    **match,
                }
            )
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[str(row.get("split") or "unknown")].append(row)
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "detector": "sklearn-logistic-patch-objectness",
        "dataset": portable(dataset),
        "model": portable(model_path),
        "parameters": {
            "threshold": threshold,
            "artifact_threshold": artifact.get("threshold"),
            "tolerance_px": tolerance_px,
            "box_tolerance_multiplier": box_tolerance_multiplier,
            "max_candidates": max_candidates,
            "max_detections": max_detections,
        },
        "overall": summarize_rows(rows),
        "by_split": {split: summarize_rows(split_rows) for split, split_rows in by_split.items()},
        "rows": rows,
        "outputs": {
            "metrics_json": portable(out_dir / "patch_detector_metrics.json"),
            "metrics_csv": portable(out_dir / "patch_detector_metrics.csv"),
            "metrics_md": portable(out_dir / "patch_detector_metrics.md"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "patch_detector_metrics.json", summary)
    write_eval_csv(out_dir / "patch_detector_metrics.csv", rows)
    write_eval_md(out_dir / "patch_detector_metrics.md", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate a patch footbag detector")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Train a patch detector from exported YOLO labels")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--out-dir", type=Path, required=True)
    train.add_argument("--crop-size", type=int, default=96)
    train.add_argument("--jitter", type=int, default=4)
    train.add_argument("--negatives-per-image", type=int, default=8)
    train.add_argument("--seed", type=int, default=1337)
    train.add_argument("--model-kind", choices=["logistic", "extra-trees"], default="logistic")
    train.add_argument("--dry-run", action="store_true")

    evaluate = sub.add_parser("evaluate", help="Evaluate patch detector proposals on exported YOLO labels")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--out-dir", type=Path, required=True)
    evaluate.add_argument("--threshold", type=float)
    evaluate.add_argument("--tolerance-px", type=float, default=24.0)
    evaluate.add_argument("--box-tolerance-multiplier", type=float, default=0.75)
    evaluate.add_argument("--max-candidates", type=int, default=120)
    evaluate.add_argument("--max-detections", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        summary = train_patch_detector(
            dataset=args.dataset,
            out_dir=args.out_dir,
            crop_size=args.crop_size,
            jitter=args.jitter,
            negatives_per_image=args.negatives_per_image,
            seed=args.seed,
            model_kind=args.model_kind,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            print(json.dumps(summary, indent=2))
        else:
            print(f"training manifest: {args.out_dir / 'patch_training_manifest.json'}")
            print(f"model: {summary['model_path']}")
            print(json.dumps(summary["sample_metrics"], indent=2))
        return
    if args.command == "evaluate":
        summary = evaluate_patch_detector(
            dataset=args.dataset,
            model_path=args.model,
            out_dir=args.out_dir,
            threshold=args.threshold,
            tolerance_px=args.tolerance_px,
            box_tolerance_multiplier=args.box_tolerance_multiplier,
            max_candidates=args.max_candidates,
            max_detections=args.max_detections,
        )
        print(f"metrics: {args.out_dir / 'patch_detector_metrics.json'}")
        print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
