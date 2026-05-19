#!/usr/bin/env python3
"""Evaluate a trained footbag detector directly on exported YOLO labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


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


def center_for_bbox(bbox: list[float]) -> tuple[float, float]:
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return ordered[idx]


def read_image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    height, width = image.shape[:2]
    return int(width), int(height)


def label_path_for_image(dataset: Path, image: Path) -> Path:
    relative = image.relative_to(dataset / "images")
    return dataset / "labels" / relative.with_suffix(".txt")


def iter_dataset_images(dataset: Path) -> list[tuple[str, Path, Path]]:
    images_root = dataset / "images"
    rows: list[tuple[str, Path, Path]] = []
    for image in sorted(images_root.glob("*/*")):
        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        split = image.parent.name
        rows.append((split, image, label_path_for_image(dataset, image)))
    return rows


def read_yolo_labels(label_path: Path, *, width: int, height: int) -> list[dict[str, Any]]:
    if not label_path.exists():
        return []
    labels: list[dict[str, Any]] = []
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
            {
                "class_id": class_id,
                "center": [x_c, y_c],
                "bbox": [x_c - box_w / 2.0, y_c - box_h / 2.0, x_c + box_w / 2.0, y_c + box_h / 2.0],
                "width": box_w,
                "height": box_h,
            }
        )
    return labels


def normalize_detection(raw: dict[str, Any]) -> dict[str, Any]:
    bbox = raw.get("bbox")
    if bbox is None and {"x1", "y1", "x2", "y2"}.issubset(raw):
        bbox = [raw["x1"], raw["y1"], raw["x2"], raw["y2"]]
    if bbox is None:
        center = raw.get("center") or [raw.get("center_x"), raw.get("center_y")]
        width = float(raw.get("width", raw.get("w", 24.0)) or 24.0)
        height = float(raw.get("height", raw.get("h", width)) or width)
        x, y = float(center[0]), float(center[1])
        bbox = [x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0]
    bbox_f = [float(value) for value in bbox]
    center = raw.get("center")
    if center is None:
        center = center_for_bbox(bbox_f)
    return {
        "bbox": bbox_f,
        "center": [float(center[0]), float(center[1])],
        "confidence": float(raw.get("confidence", raw.get("conf", 1.0)) or 0.0),
        "class_id": int(raw.get("class_id", raw.get("cls", 0)) or 0),
    }


def load_predictions_jsonl(path: Path, dataset: Path) -> dict[str, list[dict[str, Any]]]:
    predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        key = str(row.get("image") or row.get("image_path") or row.get("path") or "")
        if not key:
            continue
        keys = {key, Path(key).name}
        try:
            keys.add(str(Path(key).resolve().relative_to(dataset.resolve())))
        except (OSError, ValueError):
            pass
        detections = row.get("detections")
        if isinstance(detections, list):
            normalized = [normalize_detection(item) for item in detections]
        else:
            normalized = [normalize_detection(row)]
        for candidate_key in keys:
            predictions[candidate_key].extend(normalized)
    return dict(predictions)


def run_model_predictions(
    *,
    images: list[Path],
    model_path: Path,
    confidence_threshold: float,
    imgsz: int,
    max_det: int,
    device: str | None,
) -> dict[str, list[dict[str, Any]]]:
    try:
        from ultralytics import YOLO  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install detector dependencies with `pip install -r requirements-detector.txt`.") from exc

    model = YOLO(str(model_path))
    predictions: dict[str, list[dict[str, Any]]] = {}
    for image in images:
        kwargs: dict[str, Any] = {
            "conf": confidence_threshold,
            "imgsz": imgsz,
            "max_det": max_det,
            "verbose": False,
        }
        if device:
            kwargs["device"] = device
        result = model.predict(str(image), **kwargs)[0]
        boxes = getattr(result, "boxes", None)
        detections: list[dict[str, Any]] = []
        if boxes is not None:
            xyxy = getattr(boxes, "xyxy", [])
            confs = getattr(boxes, "conf", [])
            classes = getattr(boxes, "cls", [])
            for idx, box in enumerate(xyxy):
                confidence = float(confs[idx].item() if hasattr(confs[idx], "item") else confs[idx])
                class_id = int(classes[idx].item() if hasattr(classes[idx], "item") else classes[idx]) if len(classes) else 0
                bbox = [float(value) for value in box.tolist()]
                detections.append(
                    {
                        "bbox": bbox,
                        "center": list(center_for_bbox(bbox)),
                        "confidence": confidence,
                        "class_id": class_id,
                    }
                )
        predictions[str(image)] = detections
        predictions[image.name] = detections
    return predictions


def match_label_to_predictions(
    *,
    label: dict[str, Any],
    detections: list[dict[str, Any]],
    tolerance_px: float,
    box_tolerance_multiplier: float,
    release_confidence_threshold: float,
) -> dict[str, Any]:
    expected = (float(label["center"][0]), float(label["center"][1]))
    threshold = max(tolerance_px, max(float(label["width"]), float(label["height"])) * box_tolerance_multiplier)
    if not detections:
        return {
            "result": "missing_prediction",
            "center_error_px": None,
            "threshold_px": round(threshold, 3),
            "confidence": None,
            "prediction_x": None,
            "prediction_y": None,
            "candidate_center_pass": False,
        }
    ranked = sorted(
        (
            distance(expected, (float(det["center"][0]), float(det["center"][1]))),
            det,
        )
        for det in detections
    )
    error, detection = ranked[0]
    candidate_pass = error <= threshold
    confidence = float(detection.get("confidence") or 0.0)
    if candidate_pass and confidence >= release_confidence_threshold:
        result = "pass"
    elif candidate_pass:
        result = "low_confidence_near_label"
    else:
        result = "fail_center"
    center = detection["center"]
    return {
        "result": result,
        "center_error_px": round(error, 3),
        "threshold_px": round(threshold, 3),
        "confidence": round(confidence, 6),
        "prediction_x": round(float(center[0]), 3),
        "prediction_y": round(float(center[1]), 3),
        "candidate_center_pass": candidate_pass,
    }


def evaluate_image(
    *,
    dataset: Path,
    split: str,
    image: Path,
    label_path: Path,
    detections: list[dict[str, Any]],
    tolerance_px: float,
    box_tolerance_multiplier: float,
    release_confidence_threshold: float,
) -> list[dict[str, Any]]:
    width, height = read_image_size(image)
    labels = read_yolo_labels(label_path, width=width, height=height)
    key = image_key(image, dataset)
    release_detections = [item for item in detections if float(item.get("confidence") or 0.0) >= release_confidence_threshold]
    if not labels:
        return [
            {
                "kind": "hard_negative_image",
                "split": split,
                "image": key,
                "label": portable(label_path, dataset),
                "expected_x": None,
                "expected_y": None,
                "prediction_x": None,
                "prediction_y": None,
                "center_error_px": None,
                "threshold_px": None,
                "confidence": None,
                "result": "fail_false_positive" if release_detections else "pass_no_prediction",
                "candidate_center_pass": None,
                "detections": len(detections),
                "release_confidence_detections": len(release_detections),
                "top_confidence": None if not detections else round(max(float(item.get("confidence") or 0.0) for item in detections), 6),
            }
        ]

    rows: list[dict[str, Any]] = []
    for idx, label in enumerate(labels):
        match = match_label_to_predictions(
            label=label,
            detections=detections,
            tolerance_px=tolerance_px,
            box_tolerance_multiplier=box_tolerance_multiplier,
            release_confidence_threshold=release_confidence_threshold,
        )
        rows.append(
            {
                "kind": "positive_label",
                "split": split,
                "image": key,
                "label": portable(label_path, dataset),
                "label_index": idx,
                "expected_x": round(float(label["center"][0]), 3),
                "expected_y": round(float(label["center"][1]), 3),
                "detections": len(detections),
                "release_confidence_detections": len(release_detections),
                "top_confidence": None if not detections else round(max(float(item.get("confidence") or 0.0) for item in detections), 6),
                **match,
            }
        )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if row["kind"] == "positive_label"]
    hard_negatives = [row for row in rows if row["kind"] == "hard_negative_image"]
    errors = [float(row["center_error_px"]) for row in positives if row.get("center_error_px") is not None]
    top_confidences = [float(row["top_confidence"]) for row in rows if row.get("top_confidence") is not None]
    return {
        "positive_labels": len(positives),
        "passed_positive_labels": sum(1 for row in positives if row["result"] == "pass"),
        "candidate_center_passed_labels": sum(1 for row in positives if row.get("candidate_center_pass") is True),
        "low_confidence_near_label": sum(1 for row in positives if row["result"] == "low_confidence_near_label"),
        "missing_predictions": sum(1 for row in positives if row["result"] == "missing_prediction"),
        "failed_center": sum(1 for row in positives if row["result"] == "fail_center"),
        "pass_rate": None if not positives else round(sum(1 for row in positives if row["result"] == "pass") / len(positives), 6),
        "candidate_center_pass_rate": None
        if not positives
        else round(sum(1 for row in positives if row.get("candidate_center_pass") is True) / len(positives), 6),
        "mean_center_error_px": None if not errors else round(mean(errors), 3),
        "p95_center_error_px": None if not errors else round(float(percentile(errors, 0.95)), 3),
        "mean_top_confidence": None if not top_confidences else round(mean(top_confidences), 6),
        "hard_negative_images": len(hard_negatives),
        "false_positive_hard_negative_images": sum(1 for row in hard_negatives if row["result"] == "fail_false_positive"),
        "hard_negative_false_positive_rate": None
        if not hard_negatives
        else round(sum(1 for row in hard_negatives if row["result"] == "fail_false_positive") / len(hard_negatives), 6),
    }


def numeric(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if math.isfinite(value_f) else None


def threshold_candidates(rows: list[dict[str, Any]], base_thresholds: list[float] | None = None) -> list[float]:
    values = set(base_thresholds or [0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25])
    for row in rows:
        for key in ("confidence", "top_confidence"):
            value = numeric(row.get(key))
            if value is None:
                continue
            values.add(value)
            values.add(value + 1e-6)
    return sorted(round(value, 6) for value in values if value >= 0.0)


def threshold_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    positives = [row for row in rows if row["kind"] == "positive_label"]
    hard_negatives = [row for row in rows if row["kind"] == "hard_negative_image"]
    true_positives = 0
    wrong_center_false_positives = 0
    for row in positives:
        confidence = numeric(row.get("confidence"))
        top_confidence = numeric(row.get("top_confidence"))
        if row.get("candidate_center_pass") is True and confidence is not None and confidence >= threshold:
            true_positives += 1
        elif row.get("candidate_center_pass") is False and top_confidence is not None and top_confidence >= threshold:
            wrong_center_false_positives += 1
    hard_negative_false_positives = sum(
        1 for row in hard_negatives if (numeric(row.get("top_confidence")) or 0.0) >= threshold
    )
    false_positives = wrong_center_false_positives + hard_negative_false_positives
    false_negatives = len(positives) - true_positives
    precision = None if true_positives + false_positives == 0 else true_positives / (true_positives + false_positives)
    recall = None if not positives else true_positives / len(positives)
    f1 = None if not precision or not recall else 2 * precision * recall / (precision + recall)
    hard_negative_fp_rate = None if not hard_negatives else hard_negative_false_positives / len(hard_negatives)
    return {
        "threshold": round(threshold, 6),
        "positive_labels": len(positives),
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "wrong_center_false_positives": wrong_center_false_positives,
        "hard_negative_images": len(hard_negatives),
        "hard_negative_false_positives": hard_negative_false_positives,
        "precision": None if precision is None else round(precision, 6),
        "recall": None if recall is None else round(recall, 6),
        "f1": None if f1 is None else round(f1, 6),
        "hard_negative_false_positive_rate": None if hard_negative_fp_rate is None else round(hard_negative_fp_rate, 6),
    }


def threshold_sweep(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [threshold_metrics(rows, threshold) for threshold in threshold_candidates(rows)]


def recommended_threshold(
    rows: list[dict[str, Any]],
    *,
    calibration_split: str = "validation",
    max_hard_negative_false_positive_rate: float = 0.0,
) -> dict[str, Any]:
    split_rows = [row for row in rows if str(row.get("split") or "") == calibration_split]
    scoped_rows = split_rows if split_rows else rows
    scope = calibration_split if split_rows else "all"
    sweep = threshold_sweep(scoped_rows)
    eligible = [
        item
        for item in sweep
        if item["precision"] is not None
        and (
            item["hard_negative_false_positive_rate"] is None
            or item["hard_negative_false_positive_rate"] <= max_hard_negative_false_positive_rate
        )
    ]
    if not eligible:
        eligible = [item for item in sweep if item["precision"] is not None]
    chosen = max(
        eligible,
        key=lambda item: (
            item["f1"] if item["f1"] is not None else -1.0,
            item["recall"] if item["recall"] is not None else -1.0,
            item["precision"] if item["precision"] is not None else -1.0,
            -item["threshold"],
        ),
    ) if eligible else None
    return {
        "scope": scope,
        "calibration_split": calibration_split,
        "max_hard_negative_false_positive_rate": max_hard_negative_false_positive_rate,
        "recommended_threshold": None if chosen is None else chosen["threshold"],
        "metrics": chosen,
        "eligible_thresholds": len(eligible),
        "sweep": sweep,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        "top_confidence",
        "detections",
        "release_confidence_detections",
        "candidate_center_pass",
        "result",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    overall = summary["overall"]
    recommendation = summary.get("threshold_recommendation") or {}
    recommendation_metrics = recommendation.get("metrics") or {}
    lines = [
        "# Detector Model Evaluation",
        "",
        f"- Positive labels: `{overall['positive_labels']}`",
        f"- Release pass rate: `{overall['pass_rate']}`",
        f"- Candidate center pass rate: `{overall['candidate_center_pass_rate']}`",
        f"- Low-confidence near-label count: `{overall['low_confidence_near_label']}`",
        f"- Mean center error px: `{overall['mean_center_error_px']}`",
        f"- Mean top confidence: `{overall['mean_top_confidence']}`",
        f"- Hard-negative false-positive rate: `{overall['hard_negative_false_positive_rate']}`",
        f"- Recommended threshold: `{recommendation.get('recommended_threshold')}` on `{recommendation.get('scope')}` split",
        f"- Recommended threshold precision/recall/F1: `{recommendation_metrics.get('precision')}` / `{recommendation_metrics.get('recall')}` / `{recommendation_metrics.get('f1')}`",
        "",
        "## By Split",
        "",
        "| Split | Labels | Release Pass | Candidate Center Pass | Low Conf Near Label | Mean Error Px | Mean Top Conf | Hard-Neg FP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, metrics in sorted(summary["by_split"].items()):
        lines.append(
            f"| {split} | {metrics['positive_labels']} | {metrics['pass_rate']} | "
            f"{metrics['candidate_center_pass_rate']} | {metrics['low_confidence_near_label']} | "
            f"{metrics['mean_center_error_px']} | {metrics['mean_top_confidence']} | "
            f"{metrics['hard_negative_false_positive_rate']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_detector_model(
    *,
    dataset: Path,
    model: Path | None,
    out_dir: Path,
    predictions_jsonl: Path | None = None,
    confidence_threshold: float = 0.001,
    release_confidence_threshold: float = 0.25,
    tolerance_px: float = 24.0,
    box_tolerance_multiplier: float = 0.75,
    imgsz: int = 640,
    max_det: int = 300,
    device: str | None = None,
    calibration_split: str = "validation",
    max_hard_negative_false_positive_rate: float = 0.0,
) -> dict[str, Any]:
    dataset = dataset.expanduser().resolve()
    image_rows = iter_dataset_images(dataset)
    images = [image for _, image, _ in image_rows]
    if predictions_jsonl is not None:
        predictions_by_key = load_predictions_jsonl(predictions_jsonl, dataset)
        source = "predictions_jsonl"
    else:
        if model is None:
            raise RuntimeError("Provide --model or --predictions-jsonl.")
        raw_predictions = run_model_predictions(
            images=images,
            model_path=model,
            confidence_threshold=confidence_threshold,
            imgsz=imgsz,
            max_det=max_det,
            device=device,
        )
        predictions_by_key = {}
        for image in images:
            detections = raw_predictions.get(str(image), raw_predictions.get(image.name, []))
            predictions_by_key[image_key(image, dataset)] = detections
            predictions_by_key[image.name] = detections
        source = "model"

    rows: list[dict[str, Any]] = []
    for split, image, label_path in image_rows:
        detections = (
            predictions_by_key.get(image_key(image, dataset))
            or predictions_by_key.get(str(image))
            or predictions_by_key.get(image.name)
            or []
        )
        rows.extend(
            evaluate_image(
                dataset=dataset,
                split=split,
                image=image,
                label_path=label_path,
                detections=detections,
                tolerance_px=tolerance_px,
                box_tolerance_multiplier=box_tolerance_multiplier,
                release_confidence_threshold=release_confidence_threshold,
            )
        )

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[str(row.get("split") or "unknown")].append(row)
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "dataset": portable(dataset),
        "model": portable(model),
        "predictions_jsonl": portable(predictions_jsonl),
        "parameters": {
            "confidence_threshold": confidence_threshold,
            "release_confidence_threshold": release_confidence_threshold,
            "calibration_split": calibration_split,
            "max_hard_negative_false_positive_rate": max_hard_negative_false_positive_rate,
            "tolerance_px": tolerance_px,
            "box_tolerance_multiplier": box_tolerance_multiplier,
            "imgsz": imgsz,
            "max_det": max_det,
            "device": device,
        },
        "overall": summarize_rows(rows),
        "by_split": {split: summarize_rows(split_rows) for split, split_rows in by_split.items()},
        "threshold_recommendation": recommended_threshold(
            rows,
            calibration_split=calibration_split,
            max_hard_negative_false_positive_rate=max_hard_negative_false_positive_rate,
        ),
        "rows": rows,
        "outputs": {
            "metrics_json": portable(out_dir / "detector_model_metrics.json"),
            "metrics_csv": portable(out_dir / "detector_model_metrics.csv"),
            "metrics_md": portable(out_dir / "detector_model_metrics.md"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "detector_model_metrics.json", summary)
    write_csv(out_dir / "detector_model_metrics.csv", rows)
    write_markdown(out_dir / "detector_model_metrics.md", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a detector model directly on exported YOLO images/labels")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--predictions-jsonl", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.001)
    parser.add_argument("--release-confidence-threshold", type=float, default=0.25)
    parser.add_argument("--tolerance-px", type=float, default=24.0)
    parser.add_argument("--box-tolerance-multiplier", type=float, default=0.75)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device")
    parser.add_argument("--calibration-split", default="validation")
    parser.add_argument("--max-hard-negative-false-positive-rate", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_detector_model(
        dataset=args.dataset,
        model=args.model,
        predictions_jsonl=args.predictions_jsonl,
        out_dir=args.out_dir,
        confidence_threshold=args.confidence_threshold,
        release_confidence_threshold=args.release_confidence_threshold,
        tolerance_px=args.tolerance_px,
        box_tolerance_multiplier=args.box_tolerance_multiplier,
        imgsz=args.imgsz,
        max_det=args.max_det,
        device=args.device,
        calibration_split=args.calibration_split,
        max_hard_negative_false_positive_rate=args.max_hard_negative_false_positive_rate,
    )
    print(f"metrics: {args.out_dir / 'detector_model_metrics.json'}")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
