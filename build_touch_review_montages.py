#!/usr/bin/env python3
"""Render muted motion montages for touch-review candidates.

This is a reviewer aid only. It does not write labels. It turns unchecked
candidate moments into short silent MP4 windows so a human can review motion
quickly, then record decisions in touch_review_app.py.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from build_touch_review_contact_sheets import FILTER_CHOICES, hint_matches_filter, load_reviews, review_for_hint
from build_touch_training_table import (
    DEFAULT_CANDIDATES_DIR,
    DEFAULT_CORPUS,
    DEFAULT_LABELS_DIR,
    DEFAULT_REVIEW_MANIFEST,
    candidate_path,
    read_json,
    safe_slug,
)
from touch_label_readiness import classify_video, clustered_review_hints, likely_touch_hint, load_items


DEFAULT_OUT_DIR = DEFAULT_CORPUS / "touch_review_montages"
COLORS_RGB = {
    "likely_unchecked": (74, 215, 255),
    "audio_only": (248, 193, 74),
    "unchecked": (236, 242, 248),
}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def color_bgr(filter_name: str) -> tuple[int, int, int]:
    r, g, b = COLORS_RGB[filter_name]
    return (b, g, r)


def read_frame(cap: cv2.VideoCapture, *, time_sec: float, fps: float, frame_count: int) -> Any:
    frame_index = max(0, min(frame_count - 1, int(round(time_sec * fps)))) if frame_count > 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if ok and frame is not None:
        return frame
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    return cv2.UMat(height, width, cv2.CV_8UC3).get()


def draw_overlay(
    frame: Any,
    *,
    hint: dict[str, Any],
    filter_name: str,
    candidate_index: int,
    total_candidates: int,
    sample_time_sec: float,
    output_width: int,
    label_height: int,
) -> Any:
    height, width = frame.shape[:2]
    scale = output_width / max(1, width)
    output_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (output_width, output_height))
    canvas = cv2.copyMakeBorder(resized, label_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(12, 17, 24))
    color = color_bgr(filter_name)
    cv2.rectangle(canvas, (0, 0), (output_width - 1, label_height - 1), (12, 17, 24), thickness=-1)
    cv2.rectangle(canvas, (2, label_height + 2), (output_width - 3, label_height + output_height - 3), color, thickness=4)
    kind = "likely/model" if likely_touch_hint(hint) else "audio-tail"
    source = str(hint.get("source") or "")
    if len(source) > 64:
        source = source[:61] + "..."
    title = f"{candidate_index}/{total_candidates}  candidate {float(hint['time_sec']):.3f}s  {kind}"
    subtitle = f"clip time {sample_time_sec:.3f}s  {source}  MUTED"
    cv2.putText(canvas, title, (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (236, 242, 248), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (14, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
    return canvas


def render_montage(
    *,
    video_path: Path,
    hints: list[dict[str, Any]],
    filter_name: str,
    out_path: Path,
    seconds_before: float,
    seconds_after: float,
    output_fps: float,
    output_width: int,
    label_height: int,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or output_width)
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    output_height = int(round(source_height * (output_width / max(1, source_width)))) + label_height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(output_fps),
        (int(output_width), int(output_height)),
    )
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"cannot create montage video: {out_path}")
    frames_per_hint = max(1, int(math.ceil((seconds_before + seconds_after) * output_fps)))
    try:
        for candidate_index, hint in enumerate(hints, start=1):
            center = float(hint["time_sec"])
            start = max(0.0, center - seconds_before)
            for frame_offset in range(frames_per_hint):
                sample_time = start + frame_offset / output_fps
                frame = read_frame(cap, time_sec=sample_time, fps=fps, frame_count=frame_count)
                writer.write(
                    draw_overlay(
                        frame,
                        hint=hint,
                        filter_name=filter_name,
                        candidate_index=candidate_index,
                        total_candidates=len(hints),
                        sample_time_sec=sample_time,
                        output_width=output_width,
                        label_height=label_height,
                    )
                )
    finally:
        writer.release()
        cap.release()
    return {
        "path": str(out_path),
        "filter": filter_name,
        "candidate_count": len(hints),
        "seconds_before": seconds_before,
        "seconds_after": seconds_after,
        "output_fps": output_fps,
        "output_width": output_width,
        "output_height": output_height,
        "duration_sec": round(len(hints) * frames_per_hint / output_fps, 3),
        "times_sec": [round(float(hint["time_sec"]), 3) for hint in hints],
    }


def write_markdown(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Touch Review Candidate Montages",
        "",
        "Reviewer aid only. These silent MP4s do not create labels; use `touch_review_app.py` to approve touches and mark no-touch decisions.",
        "",
        f"- Output dir: `{manifest['out_dir']}`",
        f"- Videos considered: `{manifest['summary']['videos_considered']}`",
        f"- Montages rendered: `{manifest['summary']['montages_rendered']}`",
        f"- Candidates rendered: `{manifest['summary']['candidates_rendered']}`",
        "",
        "## Montages",
        "",
        "| video | split | status | filter | candidates | duration | montage |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in manifest["videos"]:
        if not row["montages"]:
            lines.append(
                f"| `{row['video_name']}` | {row['split']} | {row['status']} |  | 0 |  | no matching unchecked hints |"
            )
            continue
        for montage in row["montages"]:
            lines.append(
                f"| `{row['video_name']}` | {row['split']} | {row['status']} | {montage['filter']} | "
                f"{montage['candidate_count']} | {montage['duration_sec']:.1f}s | `{montage['path']}` |"
            )
    lines.extend(
        [
            "",
            "## Suggested Use",
            "",
            "```bash",
            "python3 touch_review_app.py",
            "```",
            "",
            "- Watch likely/model montages first; those are the highest-priority possible touches.",
            "- Use the app's `Likely unchecked` and `Audio only` filters to record the decisions.",
            "- The montage is silent by construction, preserving the visual-only label requirement.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_montages(args: argparse.Namespace) -> dict[str, Any]:
    review_manifest = args.review_manifest.resolve()
    candidates_dir = args.candidates_dir.resolve()
    labels_dir = args.labels_dir.resolve()
    out_dir = args.out_dir.resolve()
    items = load_items(review_manifest)
    requested_ids = set(args.video_id or [])
    filters = list(args.filters)
    videos: list[dict[str, Any]] = []
    for item in items:
        if requested_ids and str(item["video_id"]) not in requested_ids:
            continue
        if args.split and item.get("split") != args.split:
            continue
        video_id = str(item["video_id"])
        candidates_file = candidate_path(candidates_dir, video_id)
        if not candidates_file.exists():
            continue
        readiness = classify_video(
            item=item,
            labels_dir=labels_dir,
            candidates_dir=candidates_dir,
            review_match_tolerance_sec=args.review_match_tolerance_sec,
            candidate_cluster_gap_sec=args.candidate_cluster_gap_sec,
        )
        if args.only_incomplete and readiness["status"] == "complete_ready":
            continue
        candidate_payload = read_json(candidates_file)
        labels_file = labels_dir / f"{safe_slug(video_id)}.events.json"
        include_generated = True
        if labels_file.exists():
            include_generated = not bool(read_json(labels_file).get("legacy_import"))
        hints = clustered_review_hints(
            candidate_payload,
            cluster_gap_sec=args.candidate_cluster_gap_sec,
            include_generated_event_hints=include_generated,
        )
        reviews = load_reviews(labels_dir, item)
        video_path = Path(str(item["video_path"]))
        video_montages: list[dict[str, Any]] = []
        for filter_name in filters:
            matching = [
                hint
                for hint in hints
                if hint_matches_filter(
                    hint,
                    review_for_hint(hint, reviews, tolerance_sec=args.review_match_tolerance_sec),
                    filter_name,
                )
            ]
            if not matching:
                continue
            for page_index, page_hints in enumerate(chunked(matching, args.max_candidates_per_montage), start=1):
                suffix = f"_{filter_name}_p{page_index:03d}.mp4"
                montage_path = out_dir / video_id / f"{safe_slug(video_id)}{suffix}"
                video_montages.append(
                    render_montage(
                        video_path=video_path,
                        hints=page_hints,
                        filter_name=filter_name,
                        out_path=montage_path,
                        seconds_before=args.seconds_before,
                        seconds_after=args.seconds_after,
                        output_fps=args.output_fps,
                        output_width=args.output_width,
                        label_height=args.label_height,
                    )
                )
        videos.append(
            {
                "video_id": video_id,
                "video_name": item["video_name"],
                "split": item["split"],
                "status": readiness["status"],
                "total_hint_count": readiness["total_hint_count"],
                "unchecked_hint_count": readiness["unchecked_hint_count"],
                "likely_unchecked_hint_count": readiness["likely_unchecked_hint_count"],
                "audio_only_unchecked_hint_count": readiness["audio_only_unchecked_hint_count"],
                "montages": video_montages,
            }
        )
        if args.max_videos and len(videos) >= args.max_videos:
            break
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "review_manifest": str(review_manifest),
        "candidates_dir": str(candidates_dir),
        "labels_dir": str(labels_dir),
        "out_dir": str(out_dir),
        "split": args.split,
        "filters": filters,
        "only_incomplete": args.only_incomplete,
        "seconds_before": args.seconds_before,
        "seconds_after": args.seconds_after,
        "output_fps": args.output_fps,
        "summary": {
            "videos_considered": len(videos),
            "montages_rendered": sum(len(row["montages"]) for row in videos),
            "candidates_rendered": sum(int(montage["candidate_count"]) for row in videos for montage in row["montages"]),
        },
        "videos": videos,
    }
    write_json(out_dir / "touch_review_montages.json", manifest)
    write_markdown(out_dir / "touch_review_montages.md", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render silent touch-review candidate montage videos")
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW_MANIFEST)
    parser.add_argument("--candidates-dir", type=Path, default=DEFAULT_CANDIDATES_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split", default="test_frozen")
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--filters", nargs="+", choices=FILTER_CHOICES, default=["likely_unchecked", "audio_only"])
    parser.add_argument("--only-incomplete", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--review-match-tolerance-sec", type=float, default=0.05)
    parser.add_argument("--candidate-cluster-gap-sec", type=float, default=0.04)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--max-candidates-per-montage", type=int, default=24)
    parser.add_argument("--seconds-before", type=float, default=0.35)
    parser.add_argument("--seconds-after", type=float, default=0.35)
    parser.add_argument("--output-fps", type=float, default=15.0)
    parser.add_argument("--output-width", type=int, default=720)
    parser.add_argument("--label-height", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    manifest = build_montages(parse_args())
    print(f"json:   {manifest['out_dir']}/touch_review_montages.json")
    print(f"report: {manifest['out_dir']}/touch_review_montages.md")
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
