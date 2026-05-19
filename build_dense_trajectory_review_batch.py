from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_SIZE = (688, 912)
VISIBLE_STATES = ["visible", "partially_occluded", "fully_occluded", "out_of_frame", "uncertain", "unlabeled"]
TRAINING_USE_BY_SPLIT = {"test": "audit_only"}


@dataclass
class DenseClipCandidate:
    video: str
    video_path: Path
    center_time_sec: float
    split: str = "unknown"
    priority: float = 0.0
    reasons: list[str] = field(default_factory=list)
    source_rows: list[dict[str, Any]] = field(default_factory=list)
    training_use: str = "train_or_calibration"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_stem(text: str) -> str:
    stem = Path(text).stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")
    return stem or "clip"


def portable(path: Path | None, base: Path = ROOT) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return path.name


def numeric(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if math.isfinite(value_f) else None


def resolve_video(raw: str, *, video_root: Path | None = None) -> Path:
    path = Path(raw).expanduser()
    if path.exists():
        return path.resolve()
    if video_root is not None:
        candidate = video_root / path.name
        if candidate.exists():
            return candidate.resolve()
    return path


def qa_video_paths(qa_manifest: Path, video_root: Path | None = None) -> dict[str, Path]:
    manifest = read_json(qa_manifest)
    paths: dict[str, Path] = {}
    for run in manifest.get("runs", []):
        raw = run.get("video")
        if not raw:
            continue
        resolved = resolve_video(str(raw), video_root=video_root)
        paths[resolved.name] = resolved
        paths[resolved.stem] = resolved
    return paths


def split_map_from_dataset(dataset: Path | None) -> dict[str, str]:
    if dataset is None:
        return {}
    manifest_path = dataset
    if dataset.is_dir():
        manifest_path = dataset / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = read_json(manifest_path)
    return {str(video): str(split) for video, split in manifest.get("splits", {}).items()}


def split_for_video(video: str, row_split: Any, split_map: dict[str, str]) -> str:
    if row_split:
        return str(row_split)
    if video in split_map:
        return split_map[video]
    stem = Path(video).stem
    for key, split in split_map.items():
        if Path(key).stem == stem:
            return split
    return "unknown"


def training_use_for_split(split: str) -> str:
    return TRAINING_USE_BY_SPLIT.get(split, "train_or_calibration")


def target_match(video: str, target_videos: set[str]) -> bool:
    if not target_videos:
        return True
    names = {video, Path(video).name, Path(video).stem}
    return bool(names & target_videos)


def candidate_priority(row: dict[str, Any], reason: str) -> float:
    priority = 0.0
    if reason == "flagged_batch_video":
        priority += 700.0
        priority += float(row.get("prediction_share") or 0.0) * 200.0
        priority -= float(row.get("model_coverage") or 0.0) * 50.0
    elif reason == "hard_negative_false_positive":
        priority += 650.0
    elif reason == "track_center_fail":
        priority += 500.0
        priority += min(300.0, numeric(row.get("center_error_px")) or 0.0)
    elif reason == "missing_track_point":
        priority += 420.0
    else:
        priority += 100.0
    if str(row.get("split") or "") == "validation":
        priority += 30.0
    return priority


def merge_candidate(
    candidates: list[DenseClipCandidate],
    candidate: DenseClipCandidate,
    *,
    merge_within_sec: float,
) -> None:
    for existing in candidates:
        if existing.video != candidate.video:
            continue
        if abs(existing.center_time_sec - candidate.center_time_sec) > merge_within_sec:
            continue
        if candidate.priority > existing.priority:
            existing.center_time_sec = candidate.center_time_sec
            existing.priority = candidate.priority
            existing.split = candidate.split
            existing.training_use = candidate.training_use
        for reason in candidate.reasons:
            if reason not in existing.reasons:
                existing.reasons.append(reason)
        existing.source_rows.extend(candidate.source_rows)
        return
    candidates.append(candidate)


def collect_track_metric_candidates(
    *,
    track_metrics: list[Path],
    qa_videos: dict[str, Path],
    split_map: dict[str, str],
    target_videos: set[str],
    min_time_sec: float | None,
    max_time_sec: float | None,
) -> list[DenseClipCandidate]:
    candidates: list[DenseClipCandidate] = []
    for metrics_path in track_metrics:
        metrics = read_json(metrics_path)
        for row in metrics.get("rows", []):
            video = str(row.get("video") or "")
            if not video or not target_match(video, target_videos):
                continue
            time_sec = numeric(row.get("time_sec"))
            if time_sec is None:
                continue
            if min_time_sec is not None and time_sec < min_time_sec:
                continue
            if max_time_sec is not None and time_sec > max_time_sec:
                continue
            reason: str | None = None
            if row.get("kind") == "positive" and row.get("result") == "fail":
                reason = "track_center_fail"
            elif row.get("kind") == "positive" and row.get("result") == "missing_track_point":
                reason = "missing_track_point"
            elif row.get("kind") == "hard_negative" and row.get("hard_negative_result") == "false_positive_near_bad_point":
                reason = "hard_negative_false_positive"
            if reason is None:
                continue
            video_path = qa_videos.get(video) or qa_videos.get(Path(video).stem)
            if video_path is None:
                continue
            split = split_for_video(video, row.get("split"), split_map)
            candidates.append(
                DenseClipCandidate(
                    video=video_path.name,
                    video_path=video_path,
                    center_time_sec=float(time_sec),
                    split=split,
                    priority=candidate_priority(row, reason),
                    reasons=[reason],
                    source_rows=[{"metrics": portable(metrics_path), "row": row}],
                    training_use=training_use_for_split(split),
                )
            )
    return candidates


def clip_center_from_manifest(manifest_path: Path | None, fallback: float) -> float:
    if manifest_path is None or not manifest_path.exists():
        return fallback
    manifest = read_json(manifest_path)
    video_info = manifest.get("video_info", {})
    fps = numeric(video_info.get("fps")) or 30.0
    scanned = numeric(video_info.get("scanned_frames"))
    if scanned is None:
        counts = manifest.get("counts", {})
        scanned = numeric(counts.get("scanned_frames"))
    if scanned is None:
        return fallback
    return max(0.0, float(scanned) / fps / 2.0)


def collect_batch_flag_candidates(
    *,
    batch_summaries: list[Path],
    qa_videos: dict[str, Path],
    split_map: dict[str, str],
    target_videos: set[str],
    default_center_sec: float,
) -> list[DenseClipCandidate]:
    candidates: list[DenseClipCandidate] = []
    for summary_path in batch_summaries:
        summary = read_json(summary_path)
        for flag in summary.get("flags", []):
            video = str(flag.get("video") or "")
            if not video or not target_match(video, target_videos):
                continue
            video_path = qa_videos.get(video) or qa_videos.get(Path(video).stem)
            if video_path is None:
                continue
            manifest_ref = flag.get("manifest")
            manifest_path = (ROOT / manifest_ref) if manifest_ref else None
            center_time = clip_center_from_manifest(manifest_path, default_center_sec)
            split = split_for_video(video, None, split_map)
            row = {**flag, "split": split}
            candidates.append(
                DenseClipCandidate(
                    video=video_path.name,
                    video_path=video_path,
                    center_time_sec=center_time,
                    split=split,
                    priority=candidate_priority(row, "flagged_batch_video"),
                    reasons=["flagged_batch_video", *[f"flag:{item}" for item in flag.get("reasons", [])]],
                    source_rows=[{"summary": portable(summary_path), "flag": flag}],
                    training_use=training_use_for_split(split),
                )
            )
    return candidates


def select_candidates(
    candidates: list[DenseClipCandidate],
    *,
    max_clips: int,
    per_video: int,
    merge_within_sec: float,
) -> list[DenseClipCandidate]:
    merged: list[DenseClipCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.priority, reverse=True):
        merge_candidate(merged, candidate, merge_within_sec=merge_within_sec)
    selected: list[DenseClipCandidate] = []
    counts: dict[str, int] = {}
    for candidate in sorted(merged, key=lambda item: item.priority, reverse=True):
        count = counts.get(candidate.video, 0)
        if count >= per_video:
            continue
        selected.append(candidate)
        counts[candidate.video] = count + 1
        if len(selected) >= max_clips:
            break
    return selected


def track_points_for(video: str, named_track_roots: list[tuple[str, Path]]) -> dict[str, dict[int, dict[str, Any]]]:
    results: dict[str, dict[int, dict[str, Any]]] = {}
    for name, root in named_track_roots:
        track_path = root / Path(video).stem / "detector_track.json"
        if not track_path.exists():
            results[name] = {}
            continue
        track_doc = read_json(track_path)
        points: dict[int, dict[str, Any]] = {}
        for point in track_doc.get("track", []):
            frame_index = point.get("frame_index")
            if frame_index is None:
                continue
            points[int(frame_index)] = point
        results[name] = points
    return results


def draw_track_hints(tile: np.ndarray, hints: dict[str, dict[str, Any]], *, scale_x: float, scale_y: float) -> None:
    colors = [(0, 0, 255), (255, 0, 255), (255, 128, 0), (0, 255, 255)]
    for idx, (name, point) in enumerate(sorted(hints.items())):
        center = point.get("center") or [None, None]
        if center[0] is None or center[1] is None:
            continue
        x = int(round(float(center[0]) * scale_x))
        y = int(round(float(center[1]) * scale_y))
        color = colors[idx % len(colors)]
        cv2.circle(tile, (x, y), 6, color, 2)
        cv2.putText(tile, name[:12], (max(0, x + 8), max(12, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def render_contact_sheet(samples: list[np.ndarray], out_path: Path, *, cols: int) -> None:
    if not samples:
        return
    cols = max(1, cols)
    rows = int(math.ceil(len(samples) / cols))
    height, width = samples[0].shape[:2]
    sheet = np.full((rows * height, cols * width, 3), 245, dtype=np.uint8)
    for idx, sample in enumerate(samples):
        row = idx // cols
        col = idx % cols
        sheet[row * height : (row + 1) * height, col * width : (col + 1) * width] = sample
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def clip_id_for(candidate: DenseClipCandidate) -> str:
    digest = hashlib.sha1(f"{candidate.video}:{candidate.center_time_sec:.3f}:{','.join(candidate.reasons)}".encode("utf-8")).hexdigest()[:8]
    return f"{safe_stem(candidate.video)}__t{int(round(candidate.center_time_sec * 1000)):07d}__{digest}"


def render_clip(
    candidate: DenseClipCandidate,
    *,
    out_dir: Path,
    seconds_before: float,
    seconds_after: float,
    frame_stride: int,
    process_width: int,
    process_height: int,
    default_radius: float,
    named_track_roots: list[tuple[str, Path]],
    contact_sheet_cols: int,
    contact_sheet_every: int,
    max_contact_sheet_frames: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    clip_id = clip_id_for(candidate)
    clip_dir = out_dir / "clips" / clip_id
    frames_dir = clip_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(candidate.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {candidate.video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_sec = max(0.0, candidate.center_time_sec - seconds_before)
    end_sec = max(start_sec, candidate.center_time_sec + seconds_after)
    start_frame = max(0, int(math.floor(start_sec * fps)))
    end_frame = int(math.ceil(end_sec * fps))
    if total_frames > 0:
        end_frame = min(end_frame, max(0, total_frames - 1))
    track_hints = track_points_for(candidate.video, named_track_roots)

    label_rows: list[dict[str, Any]] = []
    samples: list[np.ndarray] = []
    frame_count = 0
    for frame_index in range(start_frame, end_frame + 1, max(1, frame_stride)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        processed = cv2.resize(frame, (process_width, process_height), interpolation=cv2.INTER_AREA)
        frame_path = frames_dir / f"frame_{frame_index:06d}.jpg"
        cv2.imwrite(str(frame_path), processed)
        time_sec = frame_index / fps
        hints_for_frame = {
            name: points[frame_index]
            for name, points in track_hints.items()
            if frame_index in points
        }
        label_rows.append(
            {
                "schema_version": 1,
                "clip_id": clip_id,
                "source_video": candidate.video,
                "video_path": portable(candidate.video_path),
                "split": candidate.split,
                "training_use": candidate.training_use,
                "frame_index": frame_index,
                "time_sec": round(time_sec, 6),
                "frame_image": portable(frame_path, out_dir),
                "x": None,
                "y": None,
                "radius": default_radius,
                "visibility": "unlabeled",
                "occlusion": "unknown",
                "quality": "pending",
                "label_source": "human_dense_review",
                "model_hints": {
                    name: {
                        "x": point.get("center", [None, None])[0],
                        "y": point.get("center", [None, None])[1],
                        "confidence": point.get("confidence"),
                        "source": point.get("source"),
                    }
                    for name, point in hints_for_frame.items()
                },
            }
        )
        if frame_count % max(1, contact_sheet_every) == 0 and len(samples) < max_contact_sheet_frames:
            sample = cv2.resize(processed, (220, 292), interpolation=cv2.INTER_AREA)
            draw_track_hints(sample, hints_for_frame, scale_x=220 / process_width, scale_y=292 / process_height)
            cv2.putText(sample, f"{candidate.video[:24]}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
            cv2.putText(sample, f"f={frame_index} t={time_sec:.2f}s", (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
            cv2.putText(sample, f"{candidate.split} {candidate.training_use}", (8, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
            samples.append(sample)
        frame_count += 1
    cap.release()

    label_template = clip_dir / "trajectory_labels_template.jsonl"
    label_template.write_text("\n".join(json.dumps(row, sort_keys=True) for row in label_rows) + ("\n" if label_rows else ""), encoding="utf-8")
    contact_sheet = clip_dir / "contact_sheet.jpg"
    render_contact_sheet(samples, contact_sheet, cols=contact_sheet_cols)
    clip_manifest = {
        "clip_id": clip_id,
        "source_video": candidate.video,
        "video_path": portable(candidate.video_path),
        "split": candidate.split,
        "training_use": candidate.training_use,
        "center_time_sec": round(candidate.center_time_sec, 6),
        "start_time_sec": round(start_sec, 6),
        "end_time_sec": round(end_sec, 6),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "fps": round(float(fps), 6),
        "frames_exported": len(label_rows),
        "frames_dir": portable(frames_dir, out_dir),
        "label_template": portable(label_template, out_dir),
        "contact_sheet": portable(contact_sheet, out_dir),
        "selection_reasons": candidate.reasons,
        "priority": round(candidate.priority, 3),
        "source_rows": candidate.source_rows,
    }
    write_json(clip_dir / "clip_manifest.json", clip_manifest)
    return clip_manifest, label_rows


def write_schema(out_dir: Path) -> Path:
    schema_path = out_dir / "dense_trajectory_schema.json"
    write_json(
        schema_path,
        {
            "schema_version": 1,
            "required_fields": [
                "clip_id",
                "source_video",
                "split",
                "training_use",
                "frame_index",
                "time_sec",
                "x",
                "y",
                "radius",
                "visibility",
                "quality",
            ],
            "visibility_values": VISIBLE_STATES,
            "training_use_values": ["train_or_calibration", "audit_only"],
            "split_discipline": "Rows from the held-out test split must remain audit_only and must not be used for training or calibration.",
            "positive_target_visibility": ["visible", "partially_occluded"],
            "no_object_visibility": ["fully_occluded", "out_of_frame"],
        },
    )
    return schema_path


def write_instructions(out_dir: Path) -> Path:
    path = out_dir / "README_dense_review.md"
    path.write_text(
        "\n".join(
            [
                "# Dense Trajectory Review",
                "",
                "Fill `trajectory_labels_template.jsonl` rows with reviewed per-frame bag centers.",
                "",
                "- Use `visibility=visible` when the bag center is directly visible.",
                "- Use `visibility=partially_occluded` when the center is inferable from a visible partial bag.",
                "- Use `visibility=fully_occluded` or `out_of_frame` when there is no detector target.",
                "- Leave `visibility=uncertain` and `quality=pending` for frames that should not train the model.",
                "- Test split rows marked `audit_only` are for failure analysis only and must not be used for training.",
                "- Model hints are overlays only; they are not ground truth.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def parse_named_path(raw: str) -> tuple[str, Path]:
    if ":" not in raw:
        path = Path(raw)
        return path.name, path
    name, path = raw.split(":", 1)
    return name, Path(path)


def build_dense_trajectory_review_batch(
    *,
    qa_manifest: Path,
    out_dir: Path,
    track_metrics: list[Path] | None = None,
    batch_summaries: list[Path] | None = None,
    dataset: Path | None = None,
    video_root: Path | None = None,
    target_videos: list[str] | None = None,
    max_clips: int = 12,
    per_video: int = 3,
    seconds_before: float = 1.0,
    seconds_after: float = 1.0,
    frame_stride: int = 1,
    process_width: int = OUT_SIZE[0],
    process_height: int = OUT_SIZE[1],
    default_radius: float = 10.0,
    min_time_sec: float | None = None,
    max_time_sec: float | None = None,
    default_clip_center_sec: float = 5.0,
    track_roots: list[tuple[str, Path]] | None = None,
    contact_sheet_cols: int = 4,
    contact_sheet_every: int = 10,
    max_contact_sheet_frames: int = 48,
    dry_run: bool = False,
) -> dict[str, Any]:
    qa_videos = qa_video_paths(qa_manifest, video_root)
    split_map = split_map_from_dataset(dataset)
    target_set = set(target_videos or [])
    candidates: list[DenseClipCandidate] = []
    candidates.extend(
        collect_track_metric_candidates(
            track_metrics=list(track_metrics or []),
            qa_videos=qa_videos,
            split_map=split_map,
            target_videos=target_set,
            min_time_sec=min_time_sec,
            max_time_sec=max_time_sec,
        )
    )
    candidates.extend(
        collect_batch_flag_candidates(
            batch_summaries=list(batch_summaries or []),
            qa_videos=qa_videos,
            split_map=split_map,
            target_videos=target_set,
            default_center_sec=default_clip_center_sec,
        )
    )
    selected = select_candidates(
        candidates,
        max_clips=max_clips,
        per_video=per_video,
        merge_within_sec=seconds_before + seconds_after,
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qa_manifest": portable(qa_manifest),
        "dataset": portable(dataset),
        "out_dir": portable(out_dir),
        "parameters": {
            "max_clips": max_clips,
            "per_video": per_video,
            "seconds_before": seconds_before,
            "seconds_after": seconds_after,
            "frame_stride": frame_stride,
            "process_width": process_width,
            "process_height": process_height,
            "default_radius": default_radius,
            "target_videos": list(target_videos or []),
            "min_time_sec": min_time_sec,
            "max_time_sec": max_time_sec,
        },
        "candidate_count": len(candidates),
        "selected_clips": len(selected),
        "clips": [],
        "counts": {
            "train_or_calibration_clips": sum(1 for item in selected if item.training_use == "train_or_calibration"),
            "audit_only_clips": sum(1 for item in selected if item.training_use == "audit_only"),
            "frames_exported": 0,
        },
        "outputs": {},
    }
    if dry_run:
        summary["clips"] = [
            {
                "clip_id": clip_id_for(candidate),
                "source_video": candidate.video,
                "split": candidate.split,
                "training_use": candidate.training_use,
                "center_time_sec": round(candidate.center_time_sec, 6),
                "selection_reasons": candidate.reasons,
                "priority": round(candidate.priority, 3),
            }
            for candidate in selected
        ]
        return summary

    out_dir.mkdir(parents=True, exist_ok=True)
    schema_path = write_schema(out_dir)
    instructions_path = write_instructions(out_dir)
    all_rows: list[dict[str, Any]] = []
    clip_manifests: list[dict[str, Any]] = []
    for candidate in selected:
        clip_manifest, rows = render_clip(
            candidate,
            out_dir=out_dir,
            seconds_before=seconds_before,
            seconds_after=seconds_after,
            frame_stride=frame_stride,
            process_width=process_width,
            process_height=process_height,
            default_radius=default_radius,
            named_track_roots=list(track_roots or []),
            contact_sheet_cols=contact_sheet_cols,
            contact_sheet_every=contact_sheet_every,
            max_contact_sheet_frames=max_contact_sheet_frames,
        )
        clip_manifests.append(clip_manifest)
        all_rows.extend(rows)

    combined_template = out_dir / "dense_trajectory_labels_template.jsonl"
    combined_template.write_text("\n".join(json.dumps(row, sort_keys=True) for row in all_rows) + ("\n" if all_rows else ""), encoding="utf-8")
    summary["clips"] = clip_manifests
    summary["counts"]["frames_exported"] = len(all_rows)
    summary["outputs"] = {
        "manifest": portable(out_dir / "dense_trajectory_review_manifest.json"),
        "combined_label_template": portable(combined_template),
        "schema": portable(schema_path),
        "instructions": portable(instructions_path),
    }
    write_json(out_dir / "dense_trajectory_review_manifest.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a dense trajectory review batch from detector failure evidence")
    parser.add_argument("--qa-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--track-metrics", type=Path, action="append", default=[])
    parser.add_argument("--batch-summary", type=Path, action="append", default=[])
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--target-video", action="append", default=[])
    parser.add_argument("--max-clips", type=int, default=12)
    parser.add_argument("--per-video", type=int, default=3)
    parser.add_argument("--seconds-before", type=float, default=1.0)
    parser.add_argument("--seconds-after", type=float, default=1.0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--process-width", type=int, default=OUT_SIZE[0])
    parser.add_argument("--process-height", type=int, default=OUT_SIZE[1])
    parser.add_argument("--default-radius", type=float, default=10.0)
    parser.add_argument("--min-time-sec", type=float)
    parser.add_argument("--max-time-sec", type=float)
    parser.add_argument("--default-clip-center-sec", type=float, default=5.0)
    parser.add_argument("--track-root", action="append", default=[], help="Optional NAME:TRACKS_ROOT overlay source. May be repeated.")
    parser.add_argument("--contact-sheet-cols", type=int, default=4)
    parser.add_argument("--contact-sheet-every", type=int, default=10)
    parser.add_argument("--max-contact-sheet-frames", type=int, default=48)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_dense_trajectory_review_batch(
        qa_manifest=args.qa_manifest,
        out_dir=args.out_dir,
        track_metrics=args.track_metrics,
        batch_summaries=args.batch_summary,
        dataset=args.dataset,
        video_root=args.video_root,
        target_videos=args.target_video,
        max_clips=args.max_clips,
        per_video=args.per_video,
        seconds_before=args.seconds_before,
        seconds_after=args.seconds_after,
        frame_stride=args.frame_stride,
        process_width=args.process_width,
        process_height=args.process_height,
        default_radius=args.default_radius,
        min_time_sec=args.min_time_sec,
        max_time_sec=args.max_time_sec,
        default_clip_center_sec=args.default_clip_center_sec,
        track_roots=[parse_named_path(raw) for raw in args.track_root],
        contact_sheet_cols=args.contact_sheet_cols,
        contact_sheet_every=args.contact_sheet_every,
        max_contact_sheet_frames=args.max_contact_sheet_frames,
        dry_run=args.dry_run,
    )
    print(f"manifest: {args.out_dir / 'dense_trajectory_review_manifest.json'}")
    print(json.dumps({"candidate_count": summary["candidate_count"], "selected_clips": summary["selected_clips"], **summary["counts"]}, indent=2))


if __name__ == "__main__":
    main()
