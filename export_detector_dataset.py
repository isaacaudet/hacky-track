#!/usr/bin/env python3
"""Export reviewed Hacky Track labels for a trained footbag detector.

The existing QA pipeline still has heuristic ball finding. This exporter turns
reviewed labels into a detector-ready dataset so the heuristic can become a
fallback instead of the primary source of truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parent
OUT_SIZE = (688, 912)
DECIDED = {"approved", "rejected", "missing"}
POSITIVE_BALL_ACCURACY = {"reviewed", "high", "medium"}
NEGATIVE_BALL_ACCURACY = {"bad", "low"}
NEGATIVE_CENTER_PHRASES = (
    "not ball",
    "not the ball",
    "no visible footbag",
    "marker on the leg",
    "marker is on the leg",
    "visible bag away",
    "bag away from",
    "wrong ball",
    "wrong center",
)


@dataclass(frozen=True)
class VideoSource:
    source_video: str
    path: Path


@dataclass(frozen=True)
class ExportExample:
    source_video: str
    review_file: str
    item_id: str
    kind: str
    status: str
    time_sec: float
    x: float
    y: float
    radius: float
    split: str
    ball_label_source: str
    event_role: str
    note: str


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def portable(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except (OSError, ValueError):
        try:
            return str(path.resolve().relative_to(ROOT.resolve()))
        except (OSError, ValueError):
            return path.name if path.is_absolute() else str(path)


def safe_stem(text: str) -> str:
    stem = Path(text).stem or text
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", stem).strip("-") or "item"


def review_text(item: dict[str, Any]) -> str:
    return " ".join(
        [
            str(item.get("note") or ""),
            str(item.get("review_evidence") or ""),
            " ".join(str(tag) for tag in item.get("review_tags", []) or []),
        ]
    ).lower()


def numeric(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_f):
        return None
    return value_f


def item_time(item: dict[str, Any]) -> float | None:
    return numeric(item.get("qa_frame_time_sec")) or numeric(item.get("time_sec")) or numeric(item.get("start_sec"))


def item_center(item: dict[str, Any]) -> tuple[float, float] | None:
    x = numeric(item.get("qa_ball_x")) or numeric(item.get("x"))
    y = numeric(item.get("qa_ball_y")) or numeric(item.get("y"))
    if x is None or y is None:
        return None
    return x, y


def item_radius(item: dict[str, Any], default_radius: float) -> float:
    radius = numeric(item.get("qa_ball_radius")) or numeric(item.get("radius")) or default_radius
    return float(max(8.0, min(48.0, radius)))


def is_bad_ball_center(item: dict[str, Any]) -> bool:
    if str(item.get("status") or "") != "rejected":
        return False
    accuracy = str(item.get("ball_accuracy") or item.get("qa_ball_accuracy") or "").lower()
    if accuracy in NEGATIVE_BALL_ACCURACY:
        return True
    text = review_text(item)
    return any(phrase in text for phrase in NEGATIVE_CENTER_PHRASES)


def is_ball_positive(item: dict[str, Any]) -> bool:
    if str(item.get("status") or "") not in DECIDED:
        return False
    if is_bad_ball_center(item):
        return False
    accuracy = str(item.get("ball_accuracy") or item.get("qa_ball_accuracy") or "").lower()
    status = str(item.get("status") or "")
    if accuracy in POSITIVE_BALL_ACCURACY:
        return True
    return status in {"approved", "missing"} and accuracy not in NEGATIVE_BALL_ACCURACY


def event_role(item: dict[str, Any]) -> str:
    status = str(item.get("status") or "")
    if status == "rejected":
        return "invalid_event"
    if str(item.get("source") or "") == "manual" or status == "missing":
        return "missed_event"
    return "accepted_event"


def resolve_video_path(raw: Any) -> Path:
    path = Path(str(raw or "")).expanduser()
    if path.exists():
        return path
    root_path = ROOT / path
    if root_path.exists():
        return root_path
    downloads = Path.home() / "Downloads" / path.name
    if downloads.exists():
        return downloads
    return path


def video_sources_from_manifest(qa_manifest: Path) -> dict[str, VideoSource]:
    manifest = read_json(qa_manifest)
    sources: dict[str, VideoSource] = {}
    for run in manifest.get("runs", []):
        raw_video = run.get("video") or run.get("source_video")
        if not raw_video:
            qa_path = run.get("qa_events_path")
            if qa_path:
                path = Path(str(qa_path))
                if not path.exists():
                    path = ROOT / path
                if path.exists():
                    raw_video = read_json(path).get("source_video")
        if not raw_video:
            continue
        path = resolve_video_path(raw_video)
        name = Path(str(raw_video)).name
        source = VideoSource(name, path)
        sources[name] = source
        sources[Path(name).stem] = source
        sources[safe_stem(name)] = source
    return sources


def deterministic_video_splits(videos: list[str], seed: int) -> dict[str, str]:
    ordered = sorted(set(videos))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    count = len(ordered)
    if count == 0:
        return {}
    if count == 1:
        return {ordered[0]: "test"}
    if count == 2:
        return {ordered[0]: "train", ordered[1]: "test"}
    test_count = max(1, round(count * 0.15))
    val_count = max(1, round(count * 0.15))
    if test_count + val_count >= count:
        test_count = 1
        val_count = 1
    train_count = count - test_count - val_count
    return {
        video: "train" if idx < train_count else "validation" if idx < train_count + val_count else "test"
        for idx, video in enumerate(ordered)
    }


def read_resized_frame(video: Path, time_sec: float) -> tuple[bool, Any]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return False, None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(time_sec * fps))))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False, None
    return True, cv2.resize(frame, OUT_SIZE, interpolation=cv2.INTER_AREA)


def yolo_line(x: float, y: float, radius: float, width: int, height: int) -> str:
    box = max(18.0, min(96.0, radius * 2.4))
    x_c = min(1.0, max(0.0, x / width))
    y_c = min(1.0, max(0.0, y / height))
    w = min(1.0, max(0.001, box / width))
    h = min(1.0, max(0.001, box / height))
    return f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"


def crop_around(frame: Any, x: float, y: float, size: int) -> Any:
    height, width = frame.shape[:2]
    half = size // 2
    left = max(0, int(round(x)) - half)
    right = min(width, int(round(x)) + half)
    top = max(0, int(round(y)) - half)
    bottom = min(height, int(round(y)) + half)
    return frame[top:bottom, left:right]


def iter_review_examples(
    reviews_dir: Path,
    qa_manifest: Path,
    *,
    default_radius: float,
    seed: int,
) -> tuple[list[tuple[ExportExample, Path]], list[tuple[ExportExample, Path]], dict[str, Any]]:
    videos = video_sources_from_manifest(qa_manifest)
    review_paths = sorted(reviews_dir.glob("*.review.json"))
    review_docs = [read_json(path) | {"_path": path} for path in review_paths]
    split_videos = sorted(
        {
            str(doc.get("source_video") or Path(str(doc["_path"])).name.removesuffix(".review.json"))
            for doc in review_docs
        }
    )
    splits = deterministic_video_splits(split_videos, seed)
    positives: list[tuple[ExportExample, Path]] = []
    hard_negatives: list[tuple[ExportExample, Path]] = []
    stats: dict[str, Any] = {
        "review_files": len(review_paths),
        "reviewed_items": 0,
        "skipped_items": 0,
        "missing_videos": [],
        "splits": splits,
    }
    missing_videos: set[str] = set()

    for doc in review_docs:
        source_video = str(doc.get("source_video") or Path(str(doc["_path"])).stem)
        source = videos.get(source_video) or videos.get(Path(source_video).stem) or videos.get(safe_stem(source_video))
        if source is None or not source.path.exists():
            missing_videos.add(source_video)
            continue
        split = splits.get(source_video, "train")
        for item in doc.get("items", []):
            if str(item.get("status") or "") not in DECIDED:
                continue
            stats["reviewed_items"] += 1
            center = item_center(item)
            time_sec = item_time(item)
            item_id = str(item.get("id") or "")
            if not item_id or center is None or time_sec is None:
                stats["skipped_items"] += 1
                continue
            x, y = center
            example = ExportExample(
                source_video=source_video,
                review_file=portable(Path(str(doc["_path"])), ROOT),
                item_id=item_id,
                kind=str(item.get("kind") or "touch"),
                status=str(item.get("status") or ""),
                time_sec=float(time_sec),
                x=float(x),
                y=float(y),
                radius=item_radius(item, default_radius),
                split=split,
                ball_label_source=str(item.get("ball_accuracy") or item.get("qa_ball_accuracy") or "review_decision"),
                event_role=event_role(item),
                note=str(item.get("note") or item.get("review_evidence") or ""),
            )
            if is_ball_positive(item):
                positives.append((example, source.path))
            if is_bad_ball_center(item):
                hard_negatives.append((example, source.path))

    stats["missing_videos"] = sorted(missing_videos)
    return positives, hard_negatives, stats


def iter_detector_label_decisions(
    *,
    review_manifest: Path,
    decisions_path: Path,
    qa_manifest: Path,
    default_radius: float,
) -> tuple[list[tuple[ExportExample, Path]], list[tuple[ExportExample, Path]], dict[str, Any]]:
    videos = video_sources_from_manifest(qa_manifest)
    review_doc = read_json(review_manifest)
    decisions_doc = read_json(decisions_path)
    items = {str(item.get("detector_label_id") or ""): item for item in review_doc.get("items", [])}
    positives: list[tuple[ExportExample, Path]] = []
    hard_negatives: list[tuple[ExportExample, Path]] = []
    stats = {
        "detector_label_review_manifest": portable(review_manifest, ROOT),
        "detector_label_decisions": portable(decisions_path, ROOT),
        "detector_label_decisions_seen": 0,
        "detector_label_decisions_used": 0,
        "detector_label_decisions_skipped": 0,
        "detector_label_missing_videos": [],
        "detector_label_missing_items": [],
    }
    missing_videos: set[str] = set()
    missing_items: list[str] = []
    for decision in decisions_doc.get("decisions", []):
        stats["detector_label_decisions_seen"] += 1
        status = str(decision.get("detector_status") or "").lower()
        if status in {"", "pending", "skip"}:
            stats["detector_label_decisions_skipped"] += 1
            continue
        item_id = str(decision.get("detector_label_id") or "")
        item = items.get(item_id)
        if item is None:
            missing_items.append(item_id)
            stats["detector_label_decisions_skipped"] += 1
            continue
        source_video = str(item.get("source_video") or decision.get("source_video") or "")
        source = videos.get(source_video) or videos.get(Path(source_video).stem) or videos.get(safe_stem(source_video))
        if source is None or not source.path.exists():
            missing_videos.add(source_video)
            stats["detector_label_decisions_skipped"] += 1
            continue
        x = numeric(decision.get("corrected_x")) if status == "corrected" else None
        y = numeric(decision.get("corrected_y")) if status == "corrected" else None
        if x is None or y is None:
            center = item_center(item)
            if center is None:
                stats["detector_label_decisions_skipped"] += 1
                continue
            x, y = center
        time_sec = item_time(item)
        if time_sec is None:
            stats["detector_label_decisions_skipped"] += 1
            continue
        radius = numeric(decision.get("radius")) or item_radius(item, default_radius)
        split = str(item.get("split") or "train")
        note = str(decision.get("evidence") or item.get("note") or "")
        example = ExportExample(
            source_video=source_video,
            review_file=portable(decisions_path, ROOT),
            item_id=item_id,
            kind=str(item.get("event_type") or "detector_label"),
            status=status,
            time_sec=float(time_sec),
            x=float(x),
            y=float(y),
            radius=float(max(8.0, min(48.0, radius))),
            split=split,
            ball_label_source="detector_label_review",
            event_role="detector_object_review",
            note=note,
        )
        if status in {"footbag", "corrected"}:
            positives.append((example, source.path))
            stats["detector_label_decisions_used"] += 1
        elif status in {"not_footbag", "not_visible"}:
            hard_negatives.append((example, source.path))
            stats["detector_label_decisions_used"] += 1
        else:
            stats["detector_label_decisions_skipped"] += 1
    stats["detector_label_missing_videos"] = sorted(missing_videos)
    stats["detector_label_missing_items"] = sorted(missing_items)
    return positives, hard_negatives, stats


def parse_detector_label_pair(raw: str) -> tuple[Path, Path]:
    if ":" not in raw:
        raise ValueError("Detector label review pairs must use REVIEW_MANIFEST:DECISIONS_JSON")
    review_raw, decisions_raw = raw.split(":", 1)
    if not review_raw or not decisions_raw:
        raise ValueError("Detector label review pairs must include both paths")
    return Path(review_raw), Path(decisions_raw)


def detector_label_pair_stats(pairs: list[tuple[Path, Path]]) -> dict[str, Any]:
    return {
        "detector_label_review_sets": [
            {
                "detector_label_review_manifest": portable(review_manifest, ROOT),
                "detector_label_decisions": portable(decisions_path, ROOT),
            }
            for review_manifest, decisions_path in pairs
        ],
        "detector_label_decisions_seen": 0,
        "detector_label_decisions_used": 0,
        "detector_label_decisions_skipped": 0,
        "detector_label_missing_videos": [],
        "detector_label_missing_items": [],
    }


def export_dataset(
    qa_manifest: Path,
    reviews_dir: Path,
    out_dir: Path,
    *,
    seed: int = 1337,
    default_radius: float = 22.0,
    crop_size: int = 160,
    include_hard_negative_yolo: bool = True,
    detector_label_review_manifest: Path | None = None,
    detector_label_decisions: Path | None = None,
    detector_label_review_pairs: list[tuple[Path, Path]] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    positives, hard_negatives, stats = iter_review_examples(
        reviews_dir,
        qa_manifest,
        default_radius=default_radius,
        seed=seed,
    )
    review_pairs = list(detector_label_review_pairs or [])
    if (detector_label_review_manifest is None) != (detector_label_decisions is None):
        raise ValueError("Provide both --detector-label-review-manifest and --detector-label-decisions, or use --detector-label-review-pair")
    if detector_label_review_manifest is not None and detector_label_decisions is not None:
        review_pairs.insert(0, (detector_label_review_manifest, detector_label_decisions))
    detector_label_stats: dict[str, Any] = {}
    if review_pairs:
        detector_label_stats = detector_label_pair_stats(review_pairs)
        missing_videos: set[str] = set()
        missing_items: set[str] = set()
        for review_manifest, decisions_path in review_pairs:
            detector_positives, detector_hard_negatives, pair_stats = iter_detector_label_decisions(
                review_manifest=review_manifest,
                decisions_path=decisions_path,
                qa_manifest=qa_manifest,
                default_radius=default_radius,
            )
            positives.extend(detector_positives)
            hard_negatives.extend(detector_hard_negatives)
            detector_label_stats["detector_label_decisions_seen"] += pair_stats["detector_label_decisions_seen"]
            detector_label_stats["detector_label_decisions_used"] += pair_stats["detector_label_decisions_used"]
            detector_label_stats["detector_label_decisions_skipped"] += pair_stats["detector_label_decisions_skipped"]
            missing_videos.update(pair_stats.get("detector_label_missing_videos", []))
            missing_items.update(pair_stats.get("detector_label_missing_items", []))
        detector_label_stats["detector_label_missing_videos"] = sorted(missing_videos)
        detector_label_stats["detector_label_missing_items"] = sorted(missing_items)
        if len(review_pairs) == 1:
            detector_label_stats["detector_label_review_manifest"] = portable(review_pairs[0][0], ROOT)
            detector_label_stats["detector_label_decisions"] = portable(review_pairs[0][1], ROOT)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "qa_manifest": portable(qa_manifest, ROOT),
        "reviews_dir": portable(reviews_dir, ROOT),
        "out_dir": portable(out_dir, ROOT),
        "seed": seed,
        "class_names": ["footbag"],
        "positive_labels": len(positives),
        "hard_negative_points": len(hard_negatives),
        **stats,
        **detector_label_stats,
        "written_images": 0,
        "written_label_files": 0,
        "written_hard_negative_crops": 0,
        "written_hard_negative_yolo_images": 0,
    }
    if dry_run:
        return summary

    out_dir.mkdir(parents=True, exist_ok=True)
    labels_jsonl = out_dir / "reviewed_detector_labels.jsonl"
    hard_negative_jsonl = out_dir / "hard_negatives" / "points.jsonl"
    labels_jsonl.parent.mkdir(parents=True, exist_ok=True)
    hard_negative_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with labels_jsonl.open("w", encoding="utf-8") as label_handle:
        for example, video in positives:
            ok, frame = read_resized_frame(video, example.time_sec)
            if not ok:
                summary["skipped_items"] += 1
                continue
            height, width = frame.shape[:2]
            digest = hashlib.sha1(f"{example.source_video}:{example.item_id}:{example.time_sec:.4f}".encode("utf-8")).hexdigest()[:10]
            name = f"{safe_stem(example.source_video)}__{safe_stem(example.item_id)}__{digest}.jpg"
            image_path = out_dir / "images" / example.split / name
            label_path = out_dir / "labels" / example.split / name.replace(".jpg", ".txt")
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), frame)
            label_path.write_text(yolo_line(example.x, example.y, example.radius, width, height) + "\n", encoding="utf-8")
            record = {
                **example.__dict__,
                "image": portable(image_path, out_dir),
                "label": portable(label_path, out_dir),
                "video": video.name,
            }
            label_handle.write(json.dumps(record, sort_keys=True) + "\n")
            summary["written_images"] += 1
            summary["written_label_files"] += 1

    with hard_negative_jsonl.open("w", encoding="utf-8") as neg_handle:
        for example, video in hard_negatives:
            ok, frame = read_resized_frame(video, example.time_sec)
            if not ok:
                summary["skipped_items"] += 1
                continue
            digest = hashlib.sha1(f"neg:{example.source_video}:{example.item_id}:{example.time_sec:.4f}".encode("utf-8")).hexdigest()[:10]
            crop_path = out_dir / "hard_negatives" / "crops" / f"{safe_stem(example.source_video)}__{safe_stem(example.item_id)}__{digest}.jpg"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            crop = crop_around(frame, example.x, example.y, crop_size)
            cv2.imwrite(str(crop_path), crop)
            record = {
                **example.__dict__,
                "crop": portable(crop_path, out_dir),
                "video": video.name,
                "purpose": "point-level not-footbag center for hard-negative mining",
            }
            if include_hard_negative_yolo:
                yolo_name = f"{safe_stem(example.source_video)}__hard-negative__{safe_stem(example.item_id)}__{digest}.jpg"
                image_path = out_dir / "images" / example.split / yolo_name
                label_path = out_dir / "labels" / example.split / yolo_name.replace(".jpg", ".txt")
                image_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(image_path), crop)
                label_path.write_text("", encoding="utf-8")
                record["yolo_empty_label_image"] = portable(image_path, out_dir)
                record["yolo_empty_label"] = portable(label_path, out_dir)
                summary["written_images"] += 1
                summary["written_label_files"] += 1
                summary["written_hard_negative_yolo_images"] += 1
            neg_handle.write(json.dumps(record, sort_keys=True) + "\n")
            summary["written_hard_negative_crops"] += 1

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/validation",
                "test: images/test",
                "names:",
                "  0: footbag",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_json(out_dir / "manifest.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export reviewed footbag detector labels and hard negatives")
    parser.add_argument("--qa-manifest", type=Path, required=True)
    parser.add_argument("--reviews-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--default-radius", type=float, default=22.0)
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--no-hard-negative-yolo", action="store_true", help="Do not add hard-negative crops as empty-label YOLO images")
    parser.add_argument("--detector-label-review-manifest", type=Path)
    parser.add_argument("--detector-label-decisions", type=Path)
    parser.add_argument(
        "--detector-label-review-pair",
        action="append",
        default=[],
        help="Additional detector object-label review/decision pair as REVIEW_MANIFEST:DECISIONS_JSON. May be repeated.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_dataset(
        args.qa_manifest,
        args.reviews_dir,
        args.out_dir,
        seed=args.seed,
        default_radius=args.default_radius,
        crop_size=args.crop_size,
        include_hard_negative_yolo=not args.no_hard_negative_yolo,
        detector_label_review_manifest=args.detector_label_review_manifest,
        detector_label_decisions=args.detector_label_decisions,
        detector_label_review_pairs=[parse_detector_label_pair(raw) for raw in args.detector_label_review_pair],
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(f"dataset: {args.out_dir}")
        print(f"data: {args.out_dir / 'data.yaml'}")
        print(f"manifest: {args.out_dir / 'manifest.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
