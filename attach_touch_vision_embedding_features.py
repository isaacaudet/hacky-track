#!/usr/bin/env python3
"""Attach frozen OWLv2 crop embeddings to touch candidates.

This is the heavier visual branch for v1.0 contact intelligence. It reuses the
fixed OWLv2 model already trusted for ball detection, but only as a frozen image
encoder over ball-centered crops. The L1 detector threshold and detections are
unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

import train_release_contact_classifier as contact
from attach_touch_visual_crop_features import (
    DEFAULT_BALL_TOLERANCE_SEC,
    DEFAULT_DATASET_DIR,
    DEFAULT_REVIEW_MANIFEST,
    DEFAULT_THRESHOLD,
    BallPoint,
    FrameReader,
    cache_key as visual_cache_key,
    detection_paths,
    fallback_row_ball,
    frame_index_for_row,
    load_ball_tracks,
    load_cache,
    nearest_ball,
    video_paths_by_name,
    write_cache,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "runs/release-27-public/touch_corpus_v1"
DEFAULT_LABELS_DIR = DEFAULT_CORPUS / "visual_touch_labels"
VISION_EMBEDDING_FEATURE_VERSION = 1
MODEL_ID_BY_NAME = {
    "owlv2": "google/owlv2-base-patch16-ensemble",
    "owlv2-large": "google/owlv2-large-patch14-ensemble",
}


@dataclass(frozen=True)
class CandidateRef:
    video_id: str
    time_sec: float


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def row_ref(row: dict[str, Any]) -> CandidateRef:
    return CandidateRef(str(row.get("video_id")), round(float(row.get("candidate_time_sec") or 0.0), 6))


def contact_labeled_refs(rows: list[dict[str, Any]], labels_dir: Path, tolerance_sec: float) -> set[CandidateRef]:
    labels = contact.load_label_contact_examples(labels_dir)
    matched, _summary = contact.attach_event_file_contact_labels(rows, labels, tolerance_sec=tolerance_sec)
    refs: set[CandidateRef] = set()
    for row in contact.rows_with_contact_labels(matched):
        refs.add(row_ref(row))
    return refs


def default_embedding_features(status: str, ball: BallPoint | None = None, frame_index: int | None = None, dims: int = 0) -> dict[str, Any]:
    out: dict[str, Any] = {
        "vision_embedding_feature_status": status,
        "vision_embedding_present": False,
        "vision_embedding_frame_index": frame_index,
        "vision_embedding_ball_x": None if ball is None else round(ball.x, 3),
        "vision_embedding_ball_y": None if ball is None else round(ball.y, 3),
        "vision_embedding_ball_score": None if ball is None else round(ball.score, 6),
        "vision_embedding_ball_missing": ball is None,
        "vision_embedding_model": None,
        "vision_embedding_crop_size_px": None,
        "vision_embedding_dim": dims,
    }
    for index in range(dims):
        out[f"vision_embedding_{index:03d}"] = None
    return out


def embedding_cache_key(
    *,
    video_name: str,
    frame_index: int,
    model_name: str,
    crop_size_px: int,
    ball: BallPoint | None,
) -> str:
    base = visual_cache_key(video_name=video_name, frame_index=frame_index, crop_size_px=crop_size_px, grid_size=0, ball=ball)
    return f"embedv{VISION_EMBEDDING_FEATURE_VERSION}|model={model_name}|{base}"


def crop_image(frame: np.ndarray, *, ball: BallPoint, crop_size_px: int) -> Image.Image | None:
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return None
    half = crop_size_px / 2.0
    x1 = max(0, int(math.floor(ball.x - half)))
    y1 = max(0, int(math.floor(ball.y - half)))
    x2 = min(width, int(math.ceil(ball.x + half)))
    y2 = min(height, int(math.ceil(ball.y + half)))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


class Owlv2ImageEncoder:
    def __init__(self, model_name: str, device: str):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.model_name = model_name
        self.model_id = MODEL_ID_BY_NAME[model_name]
        self.device = device
        self.processor = Owlv2Processor.from_pretrained(self.model_id, local_files_only=True)
        self.model = Owlv2ForObjectDetection.from_pretrained(self.model_id, local_files_only=True).eval()
        self.model.to(device)

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, 0), dtype=np.float32)
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            features = self.model.owlv2.get_image_features(pixel_values=pixel_values)
            features = torch.nn.functional.normalize(features, dim=-1)
        return features.detach().cpu().numpy().astype(np.float32)


def encode_pending(
    *,
    encoder: Owlv2ImageEncoder,
    pending: list[tuple[str, dict[str, Any], Image.Image]],
    cache: dict[str, dict[str, Any]],
    crop_size_px: int,
) -> None:
    if not pending:
        return
    embeddings = encoder.encode([item[2] for item in pending])
    for (key, meta, _image), embedding in zip(pending, embeddings):
        features = {
            "vision_embedding_feature_status": "ok",
            "vision_embedding_present": True,
            "vision_embedding_frame_index": meta["frame_index"],
            "vision_embedding_ball_x": round(meta["ball"].x, 3),
            "vision_embedding_ball_y": round(meta["ball"].y, 3),
            "vision_embedding_ball_score": round(meta["ball"].score, 6),
            "vision_embedding_ball_missing": False,
            "vision_embedding_model": encoder.model_name,
            "vision_embedding_crop_size_px": crop_size_px,
            "vision_embedding_dim": int(len(embedding)),
        }
        for index, value in enumerate(embedding):
            features[f"vision_embedding_{index:03d}"] = round(float(value), 7)
        cache[key] = {"cache_key": key, **features}


def attach_embeddings_to_rows(
    rows: list[dict[str, Any]],
    *,
    video_paths: dict[str, Path],
    ball_tracks: dict[str, list[BallPoint]],
    cache: dict[str, dict[str, Any]],
    encoder: Owlv2ImageEncoder,
    crop_size_px: int,
    ball_tolerance_sec: float,
    batch_size: int,
    selected_refs: set[CandidateRef] | None,
    embedding_dim_hint: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    readers: dict[str, FrameReader] = {}
    pending: list[tuple[str, dict[str, Any], Image.Image]] = []
    out_rows: list[dict[str, Any]] = []
    per_video: dict[str, dict[str, Any]] = {}
    try:
        for row in rows:
            out = dict(row)
            ref = row_ref(row)
            video_name = str(row["video_name"])
            time_sec = float(row["candidate_time_sec"])
            video_path = video_paths.get(video_name)
            features: dict[str, Any] | None = None
            if selected_refs is not None and ref not in selected_refs:
                features = default_embedding_features("not_requested", dims=embedding_dim_hint)
            elif video_path is None:
                features = default_embedding_features("missing_video_path", dims=embedding_dim_hint)
            else:
                ball = nearest_ball(ball_tracks.get(video_name, []), time_sec, ball_tolerance_sec) or fallback_row_ball(row)
                reader = readers.get(video_name)
                if reader is None:
                    reader = FrameReader(video_path)
                    readers[video_name] = reader
                frame_index = frame_index_for_row(row, ball, reader)
                if ball is None:
                    features = default_embedding_features("missing_ball", ball, frame_index, embedding_dim_hint)
                elif frame_index is None:
                    features = default_embedding_features("missing_frame_index", ball, frame_index, embedding_dim_hint)
                else:
                    key = embedding_cache_key(
                        video_name=video_name,
                        frame_index=frame_index,
                        model_name=encoder.model_name,
                        crop_size_px=crop_size_px,
                        ball=ball,
                    )
                    if key in cache:
                        features = {k: v for k, v in cache[key].items() if k != "cache_key"}
                    else:
                        frame = reader.read_bgr(frame_index)
                        image = None if frame is None else crop_image(frame, ball=ball, crop_size_px=crop_size_px)
                        if image is None:
                            features = default_embedding_features("missing_crop", ball, frame_index, embedding_dim_hint)
                        else:
                            pending.append((key, {"ball": ball, "frame_index": frame_index}, image))
                            if len(pending) >= batch_size:
                                encode_pending(encoder=encoder, pending=pending, cache=cache, crop_size_px=crop_size_px)
                                pending.clear()
                            features = {"__pending_cache_key": key}
            out.update(features)
            out_rows.append(out)
            stats = per_video.setdefault(
                video_name,
                {
                    "video_name": video_name,
                    "video_id": row.get("video_id"),
                    "split": row.get("split"),
                    "rows": 0,
                    "requested_rows": 0,
                    "ok_rows": 0,
                },
            )
            stats["rows"] += 1
            stats["requested_rows"] += int(selected_refs is None or ref in selected_refs)
        if pending:
            encode_pending(encoder=encoder, pending=pending, cache=cache, crop_size_px=crop_size_px)
            pending.clear()
        resolved_rows: list[dict[str, Any]] = []
        for row in out_rows:
            key = row.pop("__pending_cache_key", None)
            if key is not None:
                row.update({k: v for k, v in cache[key].items() if k != "cache_key"})
            resolved_rows.append(row)
        out_rows = resolved_rows
    finally:
        for reader in readers.values():
            reader.release()

    for row in out_rows:
        stats = per_video[str(row["video_name"])]
        stats["ok_rows"] += int(row.get("vision_embedding_feature_status") == "ok")

    return out_rows, {
        "rows": len(out_rows),
        "requested_rows": sum(1 for row in out_rows if row.get("vision_embedding_feature_status") != "not_requested"),
        "ok_rows": sum(1 for row in out_rows if row.get("vision_embedding_feature_status") == "ok"),
        "videos": list(per_video.values()),
    }


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Touch Vision Embedding Feature Attachment",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Model: `{manifest['model_name']}`",
        f"- Device: `{manifest['device']}`",
        f"- Crop size: `{manifest['crop_size_px']}` px",
        f"- Labeled-only: `{manifest['contact_labeled_only']}`",
        f"- Train/val requested/ok rows: `{manifest['train_val']['requested_rows']}` / `{manifest['train_val']['ok_rows']}`",
        f"- Frozen-test requested/ok rows: `{manifest['test_frozen']['requested_rows']}` / `{manifest['test_frozen']['ok_rows']}`",
        f"- Cache: `{manifest['cache_path']}`",
        "",
        "## Per-Video",
        "",
        "| video | split | rows | requested | ok embeddings |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in manifest["videos"]:
        lines.append(f"| `{row['video_name']}` | {row.get('split')} | {row['rows']} | {row['requested_rows']} | {row['ok_rows']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def attach_dataset(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else dataset_dir
    cache_path = args.cache_path.resolve() if args.cache_path else out_dir / "touch_vision_embedding_feature_cache.jsonl"
    train_rows = read_jsonl(dataset_dir / "touch_training_candidates.jsonl")
    test_rows = read_jsonl(dataset_dir / "touch_training_test_frozen.jsonl")
    selected_refs: set[CandidateRef] | None = None
    if args.contact_labeled_only:
        all_rows = train_rows + test_rows
        selected_refs = contact_labeled_refs(all_rows, args.labels_dir.resolve(), args.label_match_tolerance_sec)
    paths = detection_paths(args)
    ball_tracks = load_ball_tracks(paths, args.threshold)
    video_paths = video_paths_by_name(args.review_manifest.resolve())
    cache = load_cache(cache_path)
    encoder = Owlv2ImageEncoder(args.model_name, args.device)
    embedding_dim_hint = int(encoder.model.config.projection_dim)
    train_out, train_summary = attach_embeddings_to_rows(
        train_rows,
        video_paths=video_paths,
        ball_tracks=ball_tracks,
        cache=cache,
        encoder=encoder,
        crop_size_px=args.crop_size_px,
        ball_tolerance_sec=args.ball_tolerance_sec,
        batch_size=args.batch_size,
        selected_refs=selected_refs,
        embedding_dim_hint=embedding_dim_hint,
    )
    test_out, test_summary = attach_embeddings_to_rows(
        test_rows,
        video_paths=video_paths,
        ball_tracks=ball_tracks,
        cache=cache,
        encoder=encoder,
        crop_size_px=args.crop_size_px,
        ball_tolerance_sec=args.ball_tolerance_sec,
        batch_size=args.batch_size,
        selected_refs=selected_refs,
        embedding_dim_hint=embedding_dim_hint,
    )
    if any(row.get("split") == "test_frozen" for row in train_out):
        raise AssertionError("test_frozen row leaked into train/validation output")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "touch_training_candidates.jsonl", train_out)
    write_jsonl(out_dir / "touch_training_test_frozen.jsonl", test_out)
    write_cache(cache_path, cache)
    videos = train_summary["videos"] + test_summary["videos"]
    status = "features_attached" if train_summary["ok_rows"] or test_summary["ok_rows"] else "no_vision_embedding_features"
    manifest = {
        "schema_version": 1,
        "vision_embedding_feature_version": VISION_EMBEDDING_FEATURE_VERSION,
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
        "batch_size": args.batch_size,
        "model_name": args.model_name,
        "model_id": MODEL_ID_BY_NAME[args.model_name],
        "device": args.device,
        "contact_labeled_only": bool(args.contact_labeled_only),
        "selected_contact_label_refs": None if selected_refs is None else len(selected_refs),
        "train_val": train_summary,
        "test_frozen": test_summary,
        "videos": videos,
    }
    write_json(out_dir / "touch_vision_embedding_feature_manifest.json", manifest)
    write_report(out_dir / "touch_vision_embedding_feature_report.md", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach frozen OWLv2 crop embeddings to touch training candidates")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW_MANIFEST)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--detections-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--detections-dir", type=Path, action="append", default=[])
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--ball-tolerance-sec", type=float, default=DEFAULT_BALL_TOLERANCE_SEC)
    parser.add_argument("--crop-size-px", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--model-name", choices=sorted(MODEL_ID_BY_NAME), default="owlv2")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--contact-labeled-only", action="store_true", help="compute embeddings only for rows that have reviewed contact labels")
    parser.add_argument("--label-match-tolerance-sec", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    manifest = attach_dataset(parse_args())
    out_dir = Path(manifest["out_dir"])
    print(f"manifest: {out_dir / 'touch_vision_embedding_feature_manifest.json'}")
    print(f"report:   {out_dir / 'touch_vision_embedding_feature_report.md'}")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "contact_labeled_only": manifest["contact_labeled_only"],
                "train_val_requested_ok": [manifest["train_val"]["requested_rows"], manifest["train_val"]["ok_rows"]],
                "test_frozen_requested_ok": [manifest["test_frozen"]["requested_rows"], manifest["test_frozen"]["ok_rows"]],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
