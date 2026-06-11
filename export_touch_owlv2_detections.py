#!/usr/bin/env python3
"""Export fixed-OWLv2 detections for touch-training candidates.

This is the L1 export stage for the touch pipeline. It does not train or tune
the detector. It runs a fixed OWLv2 checkpoint and writes detection rows in the
same JSONL shape consumed by `attach_touch_l2_features.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from owlv2_event_eval import DEFAULT_PROMPTS, make_owlv2, model_checkpoint


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "runs/release-27-public/touch_corpus_v1"
DEFAULT_DATASET_DIR = DEFAULT_CORPUS / "touch_training_dataset_v1"
DEFAULT_OUT_DIR = DEFAULT_CORPUS / "owlv2_touch_detections_v1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_training_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    return read_jsonl(dataset_dir / "touch_training_candidates.jsonl") + read_jsonl(dataset_dir / "touch_training_test_frozen.jsonl")


def load_manifest_videos(review_manifest: Path) -> dict[str, dict[str, Any]]:
    if not review_manifest.exists():
        return {}
    doc = read_json(review_manifest)
    return {str(item["video_name"]): item for item in doc.get("items", [])}


def frame_indexes_for_candidates(
    rows: list[dict[str, Any]],
    videos_by_name: dict[str, dict[str, Any]],
    *,
    seconds_before: float,
    seconds_after: float,
    frame_stride: int,
) -> dict[str, dict[str, Any]]:
    by_video: dict[str, list[float]] = defaultdict(list)
    row_meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("candidate_time_sec") is not None:
            video_name = str(row["video_name"])
            by_video[video_name].append(float(row["candidate_time_sec"]))
            row_meta.setdefault(
                video_name,
                {
                    "video_id": row.get("video_id"),
                    "split": row.get("split"),
                },
            )

    plan: dict[str, dict[str, Any]] = {}
    for video_name, times in sorted(by_video.items()):
        item = videos_by_name.get(video_name)
        if not item:
            raise ValueError(f"training row references video missing from review manifest: {video_name}")
        video_path = Path(item["video_path"])
        if not video_path.exists():
            raise FileNotFoundError(f"missing video: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        frames: set[int] = set()
        for time_sec in times:
            start = max(0, int(math.floor((time_sec - seconds_before) * fps)))
            end = min(max(0, frame_count - 1), int(math.ceil((time_sec + seconds_after) * fps)))
            frames.update(range(start, end + 1, max(1, frame_stride)))
        plan[video_name] = {
            "video_name": video_name,
            "video_id": row_meta.get(video_name, {}).get("video_id") or item.get("video_id"),
            "split": row_meta.get(video_name, {}).get("split") or item.get("split"),
            "video_path": str(video_path),
            "fps": fps,
            "frame_count": frame_count,
            "candidate_times": sorted(set(round(time_sec, 6) for time_sec in times)),
            "frame_indexes": sorted(frames),
        }
    return plan


def make_detector(model_name: str, prompts: list[str], device: str) -> Callable[[np.ndarray], list[dict[str, float]]]:
    return make_owlv2(prompts, device, model_checkpoint(model_name))


def detect_frames(
    plan: dict[str, dict[str, Any]],
    *,
    detector: Callable[[np.ndarray], list[dict[str, float]]],
    threshold: float,
    max_frames: int | None = None,
    progress_every: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    per_video = []
    remaining = max_frames
    processed = 0
    total_planned = sum(len(item["frame_indexes"]) for item in plan.values())
    if max_frames is not None:
        total_planned = min(total_planned, max_frames)
    for video_name, item in sorted(plan.items()):
        frame_indexes = list(item["frame_indexes"])
        if remaining is not None:
            frame_indexes = frame_indexes[: max(0, remaining)]
        if not frame_indexes:
            per_video.append({**item, "frames_exported": 0})
            continue
        cap = cv2.VideoCapture(str(item["video_path"]))
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {item['video_path']}")
        for frame_index in frame_indexes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detector(rgb)
            processed += 1
            if progress_every and processed % progress_every == 0:
                print(
                    f"owlv2 export progress: {processed}/{total_planned} frames "
                    f"({video_name} frame {frame_index})",
                    file=sys.stderr,
                    flush=True,
                )
            records.append(
                {
                    "source_video": video_name,
                    "video_id": item.get("video_id"),
                    "split": item.get("split"),
                    "frame_index": int(frame_index),
                    "time_sec": float(frame_index) / float(item["fps"]),
                    "detections": detections,
                    "top_score": None if not detections else float(detections[0]["score"]),
                    "fires": bool(detections and float(detections[0]["score"]) >= threshold),
                }
            )
        cap.release()
        per_video.append({**{k: v for k, v in item.items() if k != "frame_indexes"}, "frames_planned": len(item["frame_indexes"]), "frames_exported": len(frame_indexes)})
        if remaining is not None:
            remaining -= len(frame_indexes)
            if remaining <= 0:
                break
    return records, {"videos": per_video, "frames_exported": len(records)}


def export_detections(args: argparse.Namespace, detector: Callable[[np.ndarray], list[dict[str, float]]] | None = None) -> dict[str, Any]:
    out_dir = args.out_dir.resolve()
    output_jsonl = out_dir / "detections.jsonl"
    manifest_path = out_dir / "owlv2_detection_manifest.json"
    if float(args.threshold) != 0.2:
        raise ValueError("fixed OWLv2 touch export requires threshold == 0.2")
    if output_jsonl.exists() and manifest_path.exists() and not args.force:
        manifest = read_json(manifest_path)
        if float(manifest.get("threshold") or 0.0) != 0.2:
            raise ValueError(f"cached OWLv2 export was not produced with threshold 0.2: {manifest_path}")
        manifest["cached"] = True
        return manifest

    rows = load_training_rows(args.dataset_dir)
    videos_by_name = load_manifest_videos(args.review_manifest)
    plan = frame_indexes_for_candidates(
        rows,
        videos_by_name,
        seconds_before=args.seconds_before,
        seconds_after=args.seconds_after,
        frame_stride=args.frame_stride,
    )
    if detector is None and not args.dry_run and rows:
        detector = make_detector(args.model, args.prompts, args.device)
    if args.dry_run or not rows:
        records: list[dict[str, Any]] = []
        detect_summary = {"videos": [{**{k: v for k, v in item.items() if k != "frame_indexes"}, "frames_planned": len(item["frame_indexes"]), "frames_exported": 0} for item in plan.values()], "frames_exported": 0}
    else:
        assert detector is not None
        records, detect_summary = detect_frames(
            plan,
            detector=detector,
            threshold=args.threshold,
            max_frames=args.max_frames,
            progress_every=getattr(args, "progress_every", 0),
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_jsonl, records)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cached": False,
        "status": "dry_run" if args.dry_run else ("waiting_for_training_rows" if not rows else "exported"),
        "model": args.model,
        "threshold": args.threshold,
        "device": args.device,
        "prompts": args.prompts,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "review_manifest": str(args.review_manifest.resolve()),
        "output_jsonl": str(output_jsonl),
        "seconds_before": args.seconds_before,
        "seconds_after": args.seconds_after,
        "frame_stride": args.frame_stride,
        "candidate_rows": len(rows),
        "videos_planned": len(plan),
        "frames_planned": sum(len(item["frame_indexes"]) for item in plan.values()),
        **detect_summary,
    }
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fixed-OWLv2 detections for touch candidate windows")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_CORPUS / "touch_review_manifest.json")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", choices=["owlv2", "owlv2-large"], default="owlv2")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    parser.add_argument("--seconds-before", type=float, default=1.0)
    parser.add_argument("--seconds-after", type=float, default=1.0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--progress-every", type=int, default=100, help="print detector progress every N exported frames; 0 disables")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    manifest = export_detections(parse_args())
    print(f"manifest: {Path(manifest['output_jsonl']).parent / 'owlv2_detection_manifest.json'}")
    print(f"detections: {manifest['output_jsonl']}")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "candidate_rows": manifest["candidate_rows"],
                "videos_planned": manifest["videos_planned"],
                "frames_planned": manifest["frames_planned"],
                "frames_exported": manifest["frames_exported"],
                "cached": manifest.get("cached", False),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
