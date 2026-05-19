#!/usr/bin/env python3
"""Run model-backed footbag detection and write a smoothed ball track."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parent


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


def center_for_bbox(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_for_center(center: tuple[float, float], width: float, height: float) -> list[float]:
    x, y = center
    return [x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0]


def bbox_filter_reason(
    bbox: list[float],
    *,
    frame_width: int,
    frame_height: int,
    min_box_size: float = 4.0,
    max_box_side_frac: float = 0.16,
    max_aspect_ratio: float = 3.5,
) -> str | None:
    width = max(0.0, float(bbox[2]) - float(bbox[0]))
    height = max(0.0, float(bbox[3]) - float(bbox[1]))
    if width < min_box_size or height < min_box_size:
        return "box_too_small"
    max_side = min(frame_width, frame_height) * max_box_side_frac
    if max(width, height) > max_side:
        return "box_too_large_for_footbag"
    aspect = max(width / max(height, 1e-6), height / max(width, 1e-6))
    if aspect > max_aspect_ratio:
        return "box_aspect_ratio_unlikely"
    return None


def normalize_detection(raw: dict[str, Any]) -> dict[str, Any]:
    bbox_raw = raw.get("bbox")
    center_raw = raw.get("center")
    if bbox_raw is None and {"x1", "y1", "x2", "y2"}.issubset(raw):
        bbox_raw = [raw["x1"], raw["y1"], raw["x2"], raw["y2"]]
    if center_raw is None and {"center_x", "center_y"}.issubset(raw):
        center_raw = [raw["center_x"], raw["center_y"]]
    if bbox_raw is None and center_raw is not None:
        width = float(raw.get("width", raw.get("w", 24.0)) or 24.0)
        height = float(raw.get("height", raw.get("h", width)) or width)
        bbox_raw = bbox_for_center((float(center_raw[0]), float(center_raw[1])), width, height)
    if bbox_raw is None:
        raise ValueError(f"Detection needs bbox or center: {raw}")

    bbox = [float(value) for value in bbox_raw]
    center = center_for_bbox(bbox) if center_raw is None else (float(center_raw[0]), float(center_raw[1]))
    confidence = float(raw.get("confidence", raw.get("conf", 1.0)) or 0.0)
    return {
        "frame_index": int(raw["frame_index"]),
        "time_sec": None if raw.get("time_sec") is None else float(raw["time_sec"]),
        "bbox": bbox,
        "center": [center[0], center[1]],
        "confidence": confidence,
        "class_id": raw.get("class_id", raw.get("cls", 0)),
        "class_name": raw.get("class_name", raw.get("name", "footbag")),
        "source": raw.get("source", "model"),
    }


def load_detections_jsonl(path: Path) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if "detections" in item:
            frame_index = int(item["frame_index"])
            time_sec = item.get("time_sec")
            for raw in item.get("detections", []):
                merged = {"frame_index": frame_index, "time_sec": time_sec, **raw}
                detections.append(normalize_detection(merged))
        else:
            if "frame_index" not in item:
                raise ValueError(f"{path}:{line_no} is missing frame_index")
            detections.append(normalize_detection(item))
    return sorted(detections, key=lambda item: (int(item["frame_index"]), -float(item.get("confidence", 0.0))))


def write_detections_jsonl(path: Path, detections: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for detection in detections:
            handle.write(json.dumps(detection, sort_keys=True) + "\n")


def video_info(video: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    info = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    cap.release()
    return info


def processed_frame(frame: Any, process_width: int | None, process_height: int | None) -> Any:
    if process_width is None or process_height is None:
        return frame
    return cv2.resize(frame, (int(process_width), int(process_height)), interpolation=cv2.INTER_AREA)


def processed_video_info(
    *,
    fps: float,
    frames: int,
    original_width: int,
    original_height: int,
    process_width: int | None,
    process_height: int | None,
    scanned: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    width = int(process_width or original_width)
    height = int(process_height or original_height)
    info = {
        "fps": fps,
        "frames": frames,
        "width": width,
        "height": height,
        "original_width": original_width,
        "original_height": original_height,
        "coordinate_space": "processed_frame" if process_width and process_height else "original_frame",
        "process_width": process_width,
        "process_height": process_height,
        "scanned_frames": scanned,
    }
    if extra:
        info.update(extra)
    return info


def run_yolo_detections(
    *,
    video: Path,
    model_path: Path,
    confidence_threshold: float,
    every_nth_frame: int,
    max_frames: int | None,
    imgsz: int,
    device: str | None,
    min_box_size: float,
    max_box_side_frac: float,
    max_aspect_ratio: float,
    process_width: int | None = None,
    process_height: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        from ultralytics import YOLO  # type: ignore
        import ultralytics  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install detector dependencies with `pip install -r requirements-detector.txt`.") from exc

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    model = YOLO(str(model_path))
    names = getattr(model, "names", {}) or {}
    detections: list[dict[str, Any]] = []
    filtered_counts: dict[str, int] = {}
    frame_index = 0
    scanned = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % every_nth_frame != 0:
            frame_index += 1
            continue
        if max_frames is not None and scanned >= max_frames:
            break
        scanned += 1
        model_frame = processed_frame(frame, process_width, process_height)
        frame_width = int(model_frame.shape[1])
        frame_height = int(model_frame.shape[0])
        predict_kwargs: dict[str, Any] = {
            "conf": confidence_threshold,
            "imgsz": imgsz,
            "verbose": False,
        }
        if device:
            predict_kwargs["device"] = device
        results = model.predict(model_frame, **predict_kwargs)
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = getattr(boxes, "xyxy", [])
            confs = getattr(boxes, "conf", [])
            classes = getattr(boxes, "cls", [])
            for idx, box in enumerate(xyxy):
                values = [float(value) for value in box.tolist()]
                confidence = float(confs[idx].item() if hasattr(confs[idx], "item") else confs[idx])
                class_id = int(classes[idx].item() if hasattr(classes[idx], "item") else classes[idx]) if len(classes) else 0
                if confidence < confidence_threshold:
                    continue
                reason = bbox_filter_reason(
                    values,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    min_box_size=min_box_size,
                    max_box_side_frac=max_box_side_frac,
                    max_aspect_ratio=max_aspect_ratio,
                )
                if reason is not None:
                    filtered_counts[reason] = filtered_counts.get(reason, 0) + 1
                    continue
                detections.append(
                    normalize_detection(
                        {
                            "frame_index": frame_index,
                            "time_sec": None if fps <= 0 else frame_index / fps,
                            "bbox": values,
                            "confidence": confidence,
                            "class_id": class_id,
                            "class_name": names.get(class_id, "footbag") if isinstance(names, dict) else "footbag",
                            "source": "model",
                        }
                    )
                )
        frame_index += 1
    cap.release()
    return detections, processed_video_info(
        fps=fps,
        frames=total_frames,
        original_width=width,
        original_height=height,
        process_width=process_width,
        process_height=process_height,
        scanned=scanned,
        extra={
            "filtered_detections": filtered_counts,
            "ultralytics_version": getattr(ultralytics, "__version__", None),
        },
    )


def run_patch_detections(
    *,
    video: Path,
    patch_model_path: Path,
    every_nth_frame: int,
    max_frames: int | None,
    process_width: int | None,
    process_height: int | None,
    patch_threshold: float | None,
    patch_max_candidates: int,
    patch_max_detections: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import patch_footbag_detector
    except ModuleNotFoundError as exc:
        raise RuntimeError("Patch detector support requires patch_footbag_detector.py and detector dependencies.") from exc

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    artifact = patch_footbag_detector.load_patch_artifact(patch_model_path)
    detections: list[dict[str, Any]] = []
    frame_index = 0
    scanned = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % every_nth_frame != 0:
            frame_index += 1
            continue
        if max_frames is not None and scanned >= max_frames:
            break
        scanned += 1
        model_frame = processed_frame(frame, process_width, process_height)
        frame_detections = patch_footbag_detector.detect_patches_in_image(
            model_frame,
            artifact,
            threshold=patch_threshold,
            max_candidates=patch_max_candidates,
            max_detections=patch_max_detections,
        )
        for detection in frame_detections:
            detections.append(
                normalize_detection(
                    {
                        **detection,
                        "frame_index": frame_index,
                        "time_sec": None if fps <= 0 else frame_index / fps,
                        "class_name": "footbag",
                        "source": "patch_detector",
                    }
                )
            )
        frame_index += 1
    cap.release()
    return detections, processed_video_info(
        fps=fps,
        frames=total_frames,
        original_width=width,
        original_height=height,
        process_width=process_width,
        process_height=process_height,
        scanned=scanned,
        extra={"patch_threshold": patch_threshold if patch_threshold is not None else artifact.get("threshold")},
    )


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def track_detections(
    detections: list[dict[str, Any]],
    *,
    fps: float | None = None,
    smoothing_alpha: float = 0.65,
    max_gap_frames: int = 6,
    max_jump_px: float = 150.0,
    low_confidence_threshold: float = 0.35,
    tracker_mode: str = "greedy",
) -> list[dict[str, Any]]:
    if tracker_mode == "temporal":
        return track_detections_temporal_path(
            detections,
            fps=fps,
            smoothing_alpha=smoothing_alpha,
            max_gap_frames=max_gap_frames,
            max_jump_px=max_jump_px,
            low_confidence_threshold=low_confidence_threshold,
        )
    if tracker_mode != "greedy":
        raise ValueError(f"Unsupported tracker mode: {tracker_mode}")
    if not detections:
        return []

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for detection in detections:
        grouped[int(detection["frame_index"])].append(detection)
    for frame_detections in grouped.values():
        frame_detections.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)

    frames = sorted(grouped)
    min_frame = frames[0]
    max_frame = frames[-1]
    active_track_id = 1
    last_frame: int | None = None
    last_center: tuple[float, float] | None = None
    last_bbox: list[float] | None = None
    last_confidence = 0.0
    velocity = (0.0, 0.0)
    track: list[dict[str, Any]] = []

    for frame_index in range(min_frame, max_frame + 1):
        candidates = grouped.get(frame_index, [])
        if last_frame is None or last_center is None:
            if not candidates:
                continue
            chosen = candidates[0]
            center = (float(chosen["center"][0]), float(chosen["center"][1]))
            bbox = [float(value) for value in chosen["bbox"]]
            reasons = ["low_confidence"] if float(chosen["confidence"]) < low_confidence_threshold else []
            record = build_track_record(
                frame_index=frame_index,
                fps=fps,
                track_id=active_track_id,
                source="model_detection",
                center=center,
                bbox=bbox,
                confidence=float(chosen["confidence"]),
                raw_detection=chosen,
                distance_from_prediction=None,
                uncertainty_reasons=reasons,
            )
            track.append(record)
            last_frame = frame_index
            last_center = center
            last_bbox = bbox
            last_confidence = float(chosen["confidence"])
            continue

        gap = frame_index - last_frame
        predicted = (last_center[0] + velocity[0] * gap, last_center[1] + velocity[1] * gap)
        chosen: dict[str, Any] | None = None
        chosen_distance: float | None = None
        if candidates:
            ranked = []
            for candidate in candidates:
                candidate_center = (float(candidate["center"][0]), float(candidate["center"][1]))
                candidate_distance = distance(predicted, candidate_center)
                rank = candidate_distance - float(candidate.get("confidence", 0.0)) * 25.0
                ranked.append((rank, candidate_distance, candidate))
            ranked.sort(key=lambda item: item[0])
            _, chosen_distance, chosen = ranked[0]
            if chosen_distance > max_jump_px * max(1, gap):
                active_track_id += 1
                velocity = (0.0, 0.0)
                last_center = None
                last_frame = None
                last_bbox = None
                last_confidence = 0.0
                chosen = None
                chosen_distance = None
                best = candidates[0]
                center = (float(best["center"][0]), float(best["center"][1]))
                bbox = [float(value) for value in best["bbox"]]
                reasons = ["new_track_large_jump"]
                if float(best["confidence"]) < low_confidence_threshold:
                    reasons.append("low_confidence")
                track.append(
                    build_track_record(
                        frame_index=frame_index,
                        fps=fps,
                        track_id=active_track_id,
                        source="model_detection",
                        center=center,
                        bbox=bbox,
                        confidence=float(best["confidence"]),
                        raw_detection=best,
                        distance_from_prediction=None,
                        uncertainty_reasons=reasons,
                    )
                )
                last_frame = frame_index
                last_center = center
                last_bbox = bbox
                last_confidence = float(best["confidence"])
                continue

        if chosen is not None and chosen_distance is not None:
            raw_center = (float(chosen["center"][0]), float(chosen["center"][1]))
            center = (
                smoothing_alpha * raw_center[0] + (1.0 - smoothing_alpha) * predicted[0],
                smoothing_alpha * raw_center[1] + (1.0 - smoothing_alpha) * predicted[1],
            )
            raw_bbox = [float(value) for value in chosen["bbox"]]
            width = raw_bbox[2] - raw_bbox[0]
            height = raw_bbox[3] - raw_bbox[1]
            bbox = bbox_for_center(center, width, height)
            reasons: list[str] = []
            if chosen_distance > max_jump_px * 0.6:
                reasons.append("large_motion")
            if float(chosen["confidence"]) < low_confidence_threshold:
                reasons.append("low_confidence")
            track.append(
                build_track_record(
                    frame_index=frame_index,
                    fps=fps,
                    track_id=active_track_id,
                    source="model_detection",
                    center=center,
                    bbox=bbox,
                    confidence=float(chosen["confidence"]),
                    raw_detection=chosen,
                    distance_from_prediction=chosen_distance,
                    uncertainty_reasons=reasons,
                )
            )
            previous_center = last_center
            previous_frame = last_frame
            last_frame = frame_index
            last_center = center
            last_bbox = bbox
            last_confidence = float(chosen["confidence"])
            frame_delta = max(1, frame_index - previous_frame)
            velocity = ((center[0] - previous_center[0]) / frame_delta, (center[1] - previous_center[1]) / frame_delta)
            continue

        if gap <= max_gap_frames and last_bbox is not None:
            width = last_bbox[2] - last_bbox[0]
            height = last_bbox[3] - last_bbox[1]
            bbox = bbox_for_center(predicted, width, height)
            confidence = max(0.01, last_confidence * (0.65**gap))
            track.append(
                build_track_record(
                    frame_index=frame_index,
                    fps=fps,
                    track_id=active_track_id,
                    source="track_predicted",
                    center=predicted,
                    bbox=bbox,
                    confidence=confidence,
                    raw_detection=None,
                    distance_from_prediction=None,
                    uncertainty_reasons=["missing_detection", "tracker_prediction"],
                )
            )
            continue

        active_track_id += 1
        last_frame = None
        last_center = None
        last_bbox = None
        last_confidence = 0.0
        velocity = (0.0, 0.0)

    return track


def track_detections_temporal_path(
    detections: list[dict[str, Any]],
    *,
    fps: float | None = None,
    smoothing_alpha: float = 0.65,
    max_gap_frames: int = 6,
    max_jump_px: float = 150.0,
    low_confidence_threshold: float = 0.35,
) -> list[dict[str, Any]]:
    if not detections:
        return []

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for detection in detections:
        grouped[int(detection["frame_index"])].append(detection)
    for frame_detections in grouped.values():
        frame_detections.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)

    previous_nodes: list[dict[str, Any]] = []
    all_nodes: list[dict[str, Any]] = []
    best_node_by_frame: dict[int, dict[str, Any]] = {}
    confidence_weight = 100.0
    motion_penalty = 90.0
    gap_penalty = 12.0
    for frame_index in sorted(grouped):
        nodes: list[dict[str, Any]] = []
        for candidate in grouped[frame_index]:
            confidence = max(0.0, min(1.0, float(candidate.get("confidence", 0.0))))
            candidate_score = confidence * confidence_weight
            center = (float(candidate["center"][0]), float(candidate["center"][1]))
            best_score = candidate_score
            best_previous: dict[str, Any] | None = None
            best_distance: float | None = None
            for previous in previous_nodes:
                gap = frame_index - int(previous["frame_index"])
                if gap < 1 or gap > max_gap_frames + 1:
                    continue
                previous_center = previous["center"]
                motion = distance(center, previous_center)
                allowed_motion = max_jump_px * max(1, gap)
                if motion > allowed_motion:
                    continue
                normalized_motion = motion / max(1.0, allowed_motion)
                score = (
                    float(previous["score"])
                    + candidate_score
                    - normalized_motion * normalized_motion * motion_penalty
                    - max(0, gap - 1) * gap_penalty
                )
                if score > best_score:
                    best_score = score
                    best_previous = previous
                    best_distance = motion
            node = {
                "frame_index": frame_index,
                "candidate": candidate,
                "center": center,
                "score": best_score,
                "previous": best_previous,
                "distance_from_previous": best_distance,
            }
            nodes.append(node)
            all_nodes.append(node)
        best_node_by_frame[frame_index] = max(nodes, key=lambda item: float(item["score"]))
        previous_nodes = nodes

    if not all_nodes:
        return []
    best = max(all_nodes, key=lambda item: (float(item["score"]), int(item["frame_index"])))
    best_chain_by_frame: dict[int, dict[str, Any]] = {}
    node: dict[str, Any] | None = best
    while node is not None:
        best_chain_by_frame[int(node["frame_index"])] = node
        node = node.get("previous")
    path_nodes = [best_chain_by_frame.get(frame_index, best_node_by_frame[frame_index]) for frame_index in sorted(best_node_by_frame)]

    track: list[dict[str, Any]] = []
    track_id = 1
    last_frame: int | None = None
    last_center: tuple[float, float] | None = None
    last_bbox: list[float] | None = None
    last_confidence = 0.0
    velocity = (0.0, 0.0)
    previous_selected_node: dict[str, Any] | None = None
    for node in path_nodes:
        frame_index = int(node["frame_index"])
        candidate = node["candidate"]
        raw_center = (float(candidate["center"][0]), float(candidate["center"][1]))
        raw_bbox = [float(value) for value in candidate["bbox"]]
        raw_confidence = float(candidate.get("confidence", 0.0))
        chosen_distance: float | None = None
        path_switched = previous_selected_node is not None and node.get("previous") is not previous_selected_node
        if path_switched:
            track_id += 1
            last_frame = None
            last_center = None
            last_bbox = None
            last_confidence = 0.0
            velocity = (0.0, 0.0)
        if last_frame is None or last_center is None:
            center = raw_center
            bbox = raw_bbox
        else:
            gap = frame_index - last_frame
            if gap > 1 and gap <= max_gap_frames + 1 and last_bbox is not None:
                width = last_bbox[2] - last_bbox[0]
                height = last_bbox[3] - last_bbox[1]
                for missing_frame in range(last_frame + 1, frame_index):
                    missing_gap = missing_frame - last_frame
                    predicted = (last_center[0] + velocity[0] * missing_gap, last_center[1] + velocity[1] * missing_gap)
                    track.append(
                        build_track_record(
                            frame_index=missing_frame,
                            fps=fps,
                            track_id=track_id,
                            source="track_predicted",
                            center=predicted,
                            bbox=bbox_for_center(predicted, width, height),
                            confidence=max(0.01, last_confidence * (0.65**missing_gap)),
                            raw_detection=None,
                            distance_from_prediction=None,
                            uncertainty_reasons=["missing_detection", "tracker_prediction", "temporal_path"],
                        )
                    )
            predicted = (last_center[0] + velocity[0] * max(1, frame_index - last_frame), last_center[1] + velocity[1] * max(1, frame_index - last_frame))
            chosen_distance = distance(predicted, raw_center)
            center = (
                smoothing_alpha * raw_center[0] + (1.0 - smoothing_alpha) * predicted[0],
                smoothing_alpha * raw_center[1] + (1.0 - smoothing_alpha) * predicted[1],
            )
            width = raw_bbox[2] - raw_bbox[0]
            height = raw_bbox[3] - raw_bbox[1]
            bbox = bbox_for_center(center, width, height)
        reasons = ["temporal_path"]
        if path_switched:
            reasons.append("temporal_path_switch")
        if chosen_distance is not None and chosen_distance > max_jump_px * 0.6:
            reasons.append("large_motion")
        if raw_confidence < low_confidence_threshold:
            reasons.append("low_confidence")
        track.append(
            build_track_record(
                frame_index=frame_index,
                fps=fps,
                track_id=track_id,
                source="model_detection",
                center=center,
                bbox=bbox,
                confidence=raw_confidence,
                raw_detection=candidate,
                distance_from_prediction=chosen_distance,
                uncertainty_reasons=reasons,
            )
        )
        if last_frame is not None and last_center is not None:
            frame_delta = max(1, frame_index - last_frame)
            velocity = ((center[0] - last_center[0]) / frame_delta, (center[1] - last_center[1]) / frame_delta)
        last_frame = frame_index
        last_center = center
        last_bbox = bbox
        last_confidence = raw_confidence
        previous_selected_node = node
    return track


def build_track_record(
    *,
    frame_index: int,
    fps: float | None,
    track_id: int,
    source: str,
    center: tuple[float, float],
    bbox: list[float],
    confidence: float,
    raw_detection: dict[str, Any] | None,
    distance_from_prediction: float | None,
    uncertainty_reasons: list[str],
) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "time_sec": None if not fps or fps <= 0 else round(frame_index / fps, 6),
        "track_id": track_id,
        "source": source,
        "center": [round(center[0], 3), round(center[1], 3)],
        "bbox": [round(value, 3) for value in bbox],
        "confidence": round(confidence, 6),
        "raw_center": None if raw_detection is None else [round(float(raw_detection["center"][0]), 3), round(float(raw_detection["center"][1]), 3)],
        "raw_bbox": None if raw_detection is None else [round(float(value), 3) for value in raw_detection["bbox"]],
        "raw_confidence": None if raw_detection is None else round(float(raw_detection["confidence"]), 6),
        "class_name": None if raw_detection is None else raw_detection.get("class_name", "footbag"),
        "distance_from_prediction_px": None if distance_from_prediction is None else round(distance_from_prediction, 3),
        "uncertainty_reasons": uncertainty_reasons,
    }


def write_track_csv(path: Path, track: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame_index",
        "time_sec",
        "track_id",
        "source",
        "center_x",
        "center_y",
        "confidence",
        "raw_center_x",
        "raw_center_y",
        "raw_confidence",
        "uncertainty_reasons",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in track:
            raw_center = item.get("raw_center") or [None, None]
            writer.writerow(
                {
                    "frame_index": item["frame_index"],
                    "time_sec": item.get("time_sec"),
                    "track_id": item["track_id"],
                    "source": item["source"],
                    "center_x": item["center"][0],
                    "center_y": item["center"][1],
                    "confidence": item["confidence"],
                    "raw_center_x": raw_center[0],
                    "raw_center_y": raw_center[1],
                    "raw_confidence": item.get("raw_confidence"),
                    "uncertainty_reasons": ";".join(item.get("uncertainty_reasons", [])),
                }
            )


def build_summary(
    *,
    video: Path | None,
    model: Path | None,
    patch_model: Path | None,
    detections_jsonl: Path | None,
    out_dir: Path,
    dry_run: bool,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "detector": "ultralytics-yolo-footbag",
        "dry_run": dry_run,
        "video": portable(video),
        "model": portable(model),
        "patch_model": portable(patch_model),
        "detections_jsonl": portable(detections_jsonl),
        "out_dir": portable(out_dir),
        "parameters": parameters,
        "video_info": None,
        "counts": {
            "raw_detections": 0,
            "track_points": 0,
            "model_detection_points": 0,
            "predicted_track_points": 0,
        },
        "outputs": {},
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_calibrated_threshold(calibration_metrics: Path | None, fallback: float) -> tuple[float, str, Path | None]:
    if calibration_metrics is None:
        return fallback, "explicit_or_default", None
    doc = read_json(calibration_metrics)
    threshold = (doc.get("threshold_recommendation") or {}).get("recommended_threshold")
    if threshold is None:
        raise ValueError(f"No threshold_recommendation.recommended_threshold in {calibration_metrics}")
    value = float(threshold)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid calibrated threshold in {calibration_metrics}: {threshold}")
    return value, "calibration_metrics", calibration_metrics


def run_detector_inference(
    *,
    video: Path | None,
    model: Path | None,
    out_dir: Path,
    patch_model: Path | None = None,
    detections_jsonl: Path | None = None,
    confidence_threshold: float = 0.25,
    smoothing_alpha: float = 0.65,
    max_gap_frames: int = 6,
    max_jump_px: float = 150.0,
    tracker_mode: str = "greedy",
    every_nth_frame: int = 1,
    max_frames: int | None = None,
    imgsz: int = 640,
    device: str | None = None,
    fps_override: float | None = None,
    min_box_size: float = 4.0,
    max_box_side_frac: float = 0.16,
    max_aspect_ratio: float = 3.5,
    process_width: int | None = None,
    process_height: int | None = None,
    patch_threshold: float | None = None,
    patch_max_candidates: int = 120,
    patch_max_detections: int = 8,
    calibration_metrics: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if every_nth_frame < 1:
        raise ValueError("--every-nth-frame must be >= 1")
    if not (0.0 < smoothing_alpha <= 1.0):
        raise ValueError("--smoothing-alpha must be > 0 and <= 1")
    if (process_width is None) != (process_height is None):
        raise ValueError("--process-width and --process-height must be provided together")
    confidence_threshold, confidence_threshold_source, calibration_metrics_path = resolve_calibrated_threshold(
        calibration_metrics,
        confidence_threshold,
    )
    parameters = {
        "confidence_threshold": confidence_threshold,
        "confidence_threshold_source": confidence_threshold_source,
        "calibration_metrics": portable(calibration_metrics_path),
        "smoothing_alpha": smoothing_alpha,
        "max_gap_frames": max_gap_frames,
        "max_jump_px": max_jump_px,
        "tracker_mode": tracker_mode,
        "every_nth_frame": every_nth_frame,
        "max_frames": max_frames,
        "imgsz": imgsz,
        "device": device,
        "fps_override": fps_override,
        "min_box_size": min_box_size,
        "max_box_side_frac": max_box_side_frac,
        "max_aspect_ratio": max_aspect_ratio,
        "process_width": process_width,
        "process_height": process_height,
        "patch_threshold": patch_threshold,
        "patch_max_candidates": patch_max_candidates,
        "patch_max_detections": patch_max_detections,
    }
    summary = build_summary(
        video=video,
        model=model,
        patch_model=patch_model,
        detections_jsonl=detections_jsonl,
        out_dir=out_dir,
        dry_run=dry_run,
        parameters=parameters,
    )
    if dry_run:
        return summary

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_detections_path = out_dir / "raw_model_detections.jsonl"
    if detections_jsonl is not None:
        detections = load_detections_jsonl(detections_jsonl)
        summary["source"] = "detections_jsonl"
    else:
        if video is None:
            raise RuntimeError("Provide --video when --detections-jsonl is not used.")
        if model is None and patch_model is None:
            raise RuntimeError("Provide --model, --patch-model, or --detections-jsonl.")
        try:
            if patch_model is not None:
                detections, info = run_patch_detections(
                    video=video,
                    patch_model_path=patch_model,
                    every_nth_frame=every_nth_frame,
                    max_frames=max_frames,
                    process_width=process_width,
                    process_height=process_height,
                    patch_threshold=patch_threshold,
                    patch_max_candidates=patch_max_candidates,
                    patch_max_detections=patch_max_detections,
                )
            else:
                detections, info = run_yolo_detections(
                    video=video,
                    model_path=model,
                    confidence_threshold=confidence_threshold,
                    every_nth_frame=every_nth_frame,
                    max_frames=max_frames,
                    imgsz=imgsz,
                    device=device,
                    min_box_size=min_box_size,
                    max_box_side_frac=max_box_side_frac,
                    max_aspect_ratio=max_aspect_ratio,
                    process_width=process_width,
                    process_height=process_height,
                )
        except RuntimeError as exc:
            summary["error"] = {
                "type": "inference_failed",
                "message": str(exc),
            }
            write_json(out_dir / "detector_inference_manifest.json", summary)
            raise
        summary["source"] = "patch_model" if patch_model is not None else "model"
        if patch_model is not None:
            summary["detector"] = "patch-footbag-objectness"
        summary["video_info"] = info
        write_detections_jsonl(raw_detections_path, detections)

    if summary["video_info"] is None and fps_override is not None:
        summary["video_info"] = {
            "fps": fps_override,
            "frames": None,
            "width": None,
            "height": None,
            "source": "fps_override",
        }
    if summary["video_info"] is None and video is not None and video.exists():
        summary["video_info"] = video_info(video)
    fps = None
    if summary["video_info"]:
        fps = float(summary["video_info"].get("fps") or 0.0)
    track = track_detections(
        detections,
        fps=fps,
        smoothing_alpha=smoothing_alpha,
        max_gap_frames=max_gap_frames,
        max_jump_px=max_jump_px,
        tracker_mode=tracker_mode,
    )

    track_json = out_dir / "detector_track.json"
    track_csv = out_dir / "detector_track.csv"
    write_json(
        track_json,
        {
            "schema_version": 1,
            "video": portable(video),
            "model": portable(model),
            "patch_model": portable(patch_model),
            "track": track,
        },
    )
    write_track_csv(track_csv, track)

    summary["counts"] = {
        "raw_detections": len(detections),
        "track_points": len(track),
        "model_detection_points": sum(1 for item in track if item["source"] == "model_detection"),
        "predicted_track_points": sum(1 for item in track if item["source"] == "track_predicted"),
    }
    summary["outputs"] = {
        "track_json": portable(track_json),
        "track_csv": portable(track_csv),
    }
    if raw_detections_path.exists():
        summary["outputs"]["raw_detections_jsonl"] = portable(raw_detections_path)
    write_json(out_dir / "detector_inference_manifest.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trained footbag detector inference and tracker smoothing")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--patch-model", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--detections-jsonl", type=Path, help="Precomputed model detections for offline/testable tracking")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--calibration-metrics", type=Path, help="Detector model evaluation JSON containing threshold_recommendation.recommended_threshold")
    parser.add_argument("--smoothing-alpha", type=float, default=0.65)
    parser.add_argument("--max-gap-frames", type=int, default=6)
    parser.add_argument("--max-jump-px", type=float, default=150.0)
    parser.add_argument("--tracker-mode", choices=["greedy", "temporal"], default="greedy")
    parser.add_argument("--every-nth-frame", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device")
    parser.add_argument("--fps", type=float, help="FPS override for precomputed detections without a source video")
    parser.add_argument("--min-box-size", type=float, default=4.0)
    parser.add_argument("--max-box-side-frac", type=float, default=0.16)
    parser.add_argument("--max-aspect-ratio", type=float, default=3.5)
    parser.add_argument("--process-width", type=int)
    parser.add_argument("--process-height", type=int)
    parser.add_argument("--patch-threshold", type=float)
    parser.add_argument("--patch-max-candidates", type=int, default=120)
    parser.add_argument("--patch-max-detections", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        summary = run_detector_inference(
            video=args.video,
            model=args.model,
            patch_model=args.patch_model,
            out_dir=args.out_dir,
            detections_jsonl=args.detections_jsonl,
            confidence_threshold=args.confidence_threshold,
            smoothing_alpha=args.smoothing_alpha,
            max_gap_frames=args.max_gap_frames,
            max_jump_px=args.max_jump_px,
            tracker_mode=args.tracker_mode,
            every_nth_frame=args.every_nth_frame,
            max_frames=args.max_frames,
            imgsz=args.imgsz,
            device=args.device,
            fps_override=args.fps,
            min_box_size=args.min_box_size,
            max_box_side_frac=args.max_box_side_frac,
            max_aspect_ratio=args.max_aspect_ratio,
            process_width=args.process_width,
            process_height=args.process_height,
            patch_threshold=args.patch_threshold,
            patch_max_candidates=args.patch_max_candidates,
            patch_max_detections=args.patch_max_detections,
            calibration_metrics=args.calibration_metrics,
            dry_run=args.dry_run,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if args.dry_run:
        print(json.dumps(summary, indent=2))
    else:
        print(f"inference manifest: {args.out_dir / 'detector_inference_manifest.json'}")
        print(f"track: {args.out_dir / 'detector_track.json'}")


if __name__ == "__main__":
    main()
