#!/usr/bin/env python3
"""Manage the end-to-end touch corpus scaffold.

This is intentionally not a detector or model trainer. It handles the part that
must be stable before touch training can be honest:

- discover the local video corpus;
- summarize which clips already have touch/event labels;
- freeze clip-disjoint train/validation/test assignments;
- write a concrete labeling manifest so visual, muted touch review can proceed
  without changing the split later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO_ROOT = Path.home() / "Downloads"
DEFAULT_OUT_DIR = ROOT / "runs/release-27-public/touch_corpus_v1"
DEFAULT_RELEASE_TEST = {
    "video-439_singular_display.MOV",
    "video-478_singular_display.MOV",
    "video-482_singular_display.MOV",
    "video-486_singular_display.MOV",
}
DEFAULT_EXPANDED_TEST = {
    "video-63_singular_display.MOV",
    "video-68_singular_display.MOV",
}
DEFAULT_VALIDATION = {
    "video-332_singular_display 2.MOV",
    "video-431_singular_display.MOV",
    "video-445_singular_display.MOV",
    "video-490_singular_display.MOV",
}
VISUAL_LABEL_REQUIRED_NOTE = (
    "Touch GT must be visually verified with audio muted. Audio-derived labels are "
    "allowed only as candidate hints, never as held-out truth for audio-using models."
)


@dataclass(frozen=True)
class EventSummary:
    path: Path | None
    annotation_method: str | None
    touches: int
    drops: int
    stalls: int
    rallies: int
    circular_audio_gt_risk: bool


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def slug_for_video(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", path.stem).strip("-")


def canonical_data_stem(video_path: Path) -> str:
    stem = video_path.stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    return stem


def candidate_event_paths(video_path: Path) -> list[Path]:
    canonical = canonical_data_stem(video_path)
    slug = slug_for_video(video_path)
    return [
        ROOT / "data" / f"{canonical}.events.json",
        ROOT / "data" / f"{slug}.events.json",
    ]


def discover_videos(video_root: Path) -> list[Path]:
    patterns = [
        "video-*_singular_display*.MOV",
        "video-*_singular_display*.mov",
        "video-*_singular_display*.MP4",
        "video-*_singular_display*.mp4",
    ]
    videos: set[Path] = set()
    for pattern in patterns:
        videos.update(video_root.glob(pattern))
    return sorted(path.resolve() for path in videos)


def video_metadata(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"readable": False}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "readable": True,
        "fps": round(fps, 6),
        "frames": frames,
        "duration_sec": None if fps <= 0 else round(frames / fps, 3),
        "width": width,
        "height": height,
    }


def summarize_events(video_path: Path) -> EventSummary:
    event_path: Path | None = None
    for path in candidate_event_paths(video_path):
        if not path.exists():
            continue
        doc = read_json(path)
        source_video = Path(str(doc.get("source_video") or "")).name
        if source_video == video_path.name:
            event_path = path
            break
    if event_path is None:
        return EventSummary(None, None, 0, 0, 0, 0, False)
    doc = read_json(event_path)
    events = [event for rally in doc.get("rallies", []) for event in rally.get("events", [])]
    method = str(doc.get("annotation_method") or "")
    return EventSummary(
        path=event_path,
        annotation_method=method or None,
        touches=sum(1 for event in events if event.get("type") == "touch"),
        drops=sum(1 for event in events if event.get("type") == "drop_floor"),
        stalls=sum(1 for event in events if event.get("type") == "stall"),
        rallies=len(doc.get("rallies", [])),
        circular_audio_gt_risk="audio" in method.lower(),
    )


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def assign_splits(videos: list[Path]) -> dict[str, str]:
    names = {path.name for path in videos}
    test = set(DEFAULT_RELEASE_TEST & names) | set(DEFAULT_EXPANDED_TEST & names)
    validation = set(DEFAULT_VALIDATION & names) - test
    remaining = sorted(names - test - validation, key=stable_hash)
    # Keep a validation set even when the default names are unavailable.
    while len(validation) < 4 and remaining:
        validation.add(remaining.pop(0))
    assignments: dict[str, str] = {}
    for path in videos:
        if path.name in test:
            assignments[path.name] = "test_frozen"
        elif path.name in validation:
            assignments[path.name] = "validation"
        else:
            assignments[path.name] = "train"
    return assignments


def build_inventory(video_root: Path) -> dict[str, Any]:
    videos = discover_videos(video_root)
    assignments = assign_splits(videos)
    rows: list[dict[str, Any]] = []
    for index, video in enumerate(videos, start=1):
        events = summarize_events(video)
        rows.append(
            {
                "video_index": index,
                "video_id": slug_for_video(video),
                "video_name": video.name,
                "video_path": str(video),
                "split": assignments[video.name],
                "metadata": video_metadata(video),
                "events_path": None if events.path is None else str(events.path),
                "annotation_method": events.annotation_method,
                "touch_labels": events.touches,
                "drop_labels": events.drops,
                "stall_labels": events.stalls,
                "rallies_labeled": events.rallies,
                "has_touch_gt": events.touches > 0,
                "circular_audio_gt_risk": events.circular_audio_gt_risk,
                "label_status": "labeled" if events.touches > 0 else "needs_visual_touch_review",
            }
        )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "video_root": str(video_root),
        "video_count": len(rows),
        "split_policy": {
            "test_frozen": sorted(DEFAULT_RELEASE_TEST | DEFAULT_EXPANDED_TEST),
            "validation": sorted(DEFAULT_VALIDATION),
            "note": (
                "Preserves the prior four-video release test set and adds video-63/video-68 "
                "as a six-video frozen test expansion before new touch labels are created."
            ),
        },
        "visual_label_required_note": VISUAL_LABEL_REQUIRED_NOTE,
        "videos": rows,
        "summary": summarize_inventory_rows(rows),
    }


def summarize_inventory_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, dict[str, int]] = {}
    for row in rows:
        split = row["split"]
        item = by_split.setdefault(split, {"videos": 0, "labeled_videos": 0, "touches": 0})
        item["videos"] += 1
        if row["has_touch_gt"]:
            item["labeled_videos"] += 1
        item["touches"] += int(row["touch_labels"])
    return {
        "videos": len(rows),
        "labeled_videos": sum(1 for row in rows if row["has_touch_gt"]),
        "unlabeled_videos": sum(1 for row in rows if not row["has_touch_gt"]),
        "touch_labels": sum(int(row["touch_labels"]) for row in rows),
        "audio_derived_labeled_videos": sum(1 for row in rows if row["circular_audio_gt_risk"]),
        "by_split": by_split,
    }


def build_touch_review_manifest(inventory: dict[str, Any]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for row in inventory["videos"]:
        items.append(
            {
                "video_id": row["video_id"],
                "video_name": row["video_name"],
                "video_path": row["video_path"],
                "split": row["split"],
                "status": "needs_visual_verification" if row["has_touch_gt"] else "needs_initial_visual_labels",
                "existing_events_path": row["events_path"],
                "existing_annotation_method": row["annotation_method"],
                "existing_touch_labels": row["touch_labels"],
                "circular_audio_gt_risk": row["circular_audio_gt_risk"],
                "review_requirements": {
                    "audio_muted": True,
                    "confirm_touches_from_video_only": True,
                    "mark_drops": True,
                    "mark_stalls": True,
                    "do_not_change_split": True,
                },
            }
        )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "visual_label_required_note": VISUAL_LABEL_REQUIRED_NOTE,
        "items": items,
        "summary": {
            "items": len(items),
            "needs_initial_visual_labels": sum(1 for item in items if item["status"] == "needs_initial_visual_labels"),
            "needs_visual_verification": sum(1 for item in items if item["status"] == "needs_visual_verification"),
            "frozen_test_items": sum(1 for item in items if item["split"] == "test_frozen"),
        },
    }


def write_markdown_report(path: Path, inventory: dict[str, Any], review_manifest_path: Path) -> None:
    summary = inventory["summary"]
    lines = [
        "# Touch Corpus Scaffold",
        "",
        "This is the stable corpus/split scaffold for end-to-end touch training.",
        "",
        f"- Videos discovered: `{summary['videos']}`",
        f"- Videos with any touch GT: `{summary['labeled_videos']}`",
        f"- Videos needing initial visual touch labels: `{summary['unlabeled_videos']}`",
        f"- Existing touch labels: `{summary['touch_labels']}`",
        f"- Audio-derived labeled videos: `{summary['audio_derived_labeled_videos']}`",
        f"- Review manifest: `{review_manifest_path}`",
        "",
        "## Split Summary",
        "",
        "| split | videos | labeled videos | touch labels |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split, row in sorted(summary["by_split"].items()):
        lines.append(f"| {split} | {row['videos']} | {row['labeled_videos']} | {row['touches']} |")
    lines.extend(
        [
            "",
            "## Frozen Test Set",
            "",
            "The frozen test set preserves the prior release-test videos and adds two unlabeled expansion clips before new touch labels are created.",
            "",
        ]
    )
    for row in inventory["videos"]:
        if row["split"] == "test_frozen":
            lines.append(f"- `{row['video_name']}` — labels: `{row['touch_labels']}`")
    lines.extend(
        [
            "",
            "## Labeling Rule",
            "",
            VISUAL_LABEL_REQUIRED_NOTE,
            "",
            "Use audio only to propose candidates. During ground-truth acceptance, the reviewer should verify contact from muted video.",
            "",
            "## Per-Video Inventory",
            "",
            "| video | split | touch labels | annotation | status |",
            "| --- | --- | ---: | --- | --- |",
        ]
    )
    for row in inventory["videos"]:
        annotation = row["annotation_method"] or "-"
        if row["circular_audio_gt_risk"]:
            annotation += " (audio-circular risk)"
        lines.append(
            f"| `{row['video_name']}` | {row['split']} | {row['touch_labels']} | "
            f"{annotation} | {row['label_status']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_scaffold(args: argparse.Namespace) -> None:
    video_root = args.video_root.expanduser().resolve()
    out_dir = args.out_dir.resolve()
    inventory = build_inventory(video_root)
    review_manifest = build_touch_review_manifest(inventory)
    inventory_path = out_dir / "touch_corpus_inventory.json"
    review_manifest_path = out_dir / "touch_review_manifest.json"
    write_json(inventory_path, inventory)
    write_json(review_manifest_path, review_manifest)
    write_jsonl(out_dir / "touch_review_items.jsonl", review_manifest["items"])
    write_markdown_report(out_dir / "touch_corpus_report.md", inventory, review_manifest_path)
    print(f"inventory: {inventory_path}")
    print(f"review:    {review_manifest_path}")
    print(f"items:     {out_dir / 'touch_review_items.jsonl'}")
    print(f"report:    {out_dir / 'touch_corpus_report.md'}")
    print(json.dumps(inventory["summary"], indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the touch corpus scaffold and frozen splits")
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    run_scaffold(parse_args())


if __name__ == "__main__":
    main()
