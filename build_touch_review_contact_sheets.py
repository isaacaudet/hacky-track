#!/usr/bin/env python3
"""Render contact sheets for muted visual touch-review candidates.

This is reviewer aid only. It never writes labels. It renders unchecked
candidate moments so the human reviewer can scan likely/model hints and the
audio-only tail before making decisions in touch_review_app.py.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageDraw

from build_touch_training_table import (
    DEFAULT_CANDIDATES_DIR,
    DEFAULT_CORPUS,
    DEFAULT_LABELS_DIR,
    DEFAULT_REVIEW_MANIFEST,
    candidate_path,
    candidate_reviews,
    label_path,
    read_json,
    safe_slug,
)
from touch_label_readiness import classify_video, clustered_review_hints, likely_touch_hint, load_items


DEFAULT_OUT_DIR = DEFAULT_CORPUS / "touch_review_contact_sheets"
FILTER_CHOICES = ("likely_unchecked", "audio_only", "unchecked")
COLORS = {
    "likely_unchecked": (74, 215, 255),
    "audio_only": (248, 193, 74),
    "unchecked": (236, 242, 248),
}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def review_for_hint(hint: dict[str, Any], reviews: list[dict[str, Any]], *, tolerance_sec: float) -> dict[str, Any] | None:
    hint_time = float(hint["time_sec"])
    for review in reviews:
        if abs(float(review["time_sec"]) - hint_time) <= tolerance_sec:
            return review
    return None


def load_reviews(labels_dir: Path, item: dict[str, Any]) -> list[dict[str, Any]]:
    path = label_path(labels_dir, str(item["video_id"]))
    if not path.exists():
        return []
    return candidate_reviews(read_json(path))


def hint_matches_filter(hint: dict[str, Any], review: dict[str, Any] | None, filter_name: str) -> bool:
    if review is not None:
        return False
    likely = likely_touch_hint(hint)
    if filter_name == "likely_unchecked":
        return likely
    if filter_name == "audio_only":
        return not likely
    if filter_name == "unchecked":
        return True
    raise ValueError(f"unsupported filter: {filter_name}")


def frame_at_time(cap: cv2.VideoCapture, *, time_sec: float, fps: float, frame_count: int) -> Image.Image:
    frame_index = max(0, min(frame_count - 1, int(round(time_sec * fps)))) if frame_count > 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok or frame is None:
        return Image.new("RGB", (320, 240), (15, 18, 24))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def make_tile(
    image: Image.Image,
    *,
    hint: dict[str, Any],
    index: int,
    total: int,
    filter_name: str,
    thumb_width: int,
    label_height: int,
) -> Image.Image:
    color = COLORS[filter_name]
    scale = thumb_width / max(1, image.width)
    thumb_height = max(1, int(round(image.height * scale)))
    resized = image.resize((thumb_width, thumb_height))
    tile = Image.new("RGB", (thumb_width, thumb_height + label_height), (12, 17, 24))
    tile.paste(resized, (0, 0))
    draw = ImageDraw.Draw(tile)
    border = max(3, thumb_width // 80)
    for offset in range(border):
        draw.rectangle(
            [offset, offset, thumb_width - 1 - offset, thumb_height - 1 - offset],
            outline=color,
        )
    time_label = f"{index}/{total}  {float(hint['time_sec']):.3f}s"
    kind_label = "likely/model" if likely_touch_hint(hint) else "audio-tail"
    source_label = str(hint.get("source") or "")
    if len(source_label) > 36:
        source_label = source_label[:33] + "..."
    draw.rectangle([0, thumb_height, thumb_width, thumb_height + label_height], fill=(12, 17, 24))
    draw.text((8, thumb_height + 6), time_label, fill=(236, 242, 248))
    draw.text((8, thumb_height + 24), kind_label, fill=color)
    draw.text((8, thumb_height + 42), source_label, fill=(152, 166, 184))
    return tile


def render_sheet(
    *,
    video_path: Path,
    hints: list[dict[str, Any]],
    filter_name: str,
    out_path: Path,
    cols: int,
    thumb_width: int,
    label_height: int,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    rows = max(1, int(math.ceil(len(hints) / float(cols))))
    tiles: list[Image.Image] = []
    try:
        for index, hint in enumerate(hints, start=1):
            image = frame_at_time(cap, time_sec=float(hint["time_sec"]), fps=fps, frame_count=frame_count)
            tiles.append(
                make_tile(
                    image,
                    hint=hint,
                    index=index,
                    total=len(hints),
                    filter_name=filter_name,
                    thumb_width=thumb_width,
                    label_height=label_height,
                )
            )
    finally:
        cap.release()
    if not tiles:
        raise ValueError("render_sheet called with no hints")
    tile_width, tile_height = tiles[0].size
    sheet = Image.new("RGB", (cols * tile_width, rows * tile_height), (9, 13, 19))
    for index, tile in enumerate(tiles):
        x = (index % cols) * tile_width
        y = (index // cols) * tile_height
        sheet.paste(tile, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return {
        "path": str(out_path),
        "filter": filter_name,
        "candidate_count": len(hints),
        "cols": cols,
        "rows": rows,
        "thumb_width": thumb_width,
        "image_width": sheet.width,
        "image_height": sheet.height,
        "times_sec": [round(float(hint["time_sec"]), 3) for hint in hints],
    }


def write_markdown(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Touch Review Contact Sheets",
        "",
        "Reviewer aid only. These sheets do not create labels; use `touch_review_app.py` to approve touches and mark no-touch decisions.",
        "",
        f"- Status source: `{manifest['review_manifest']}`",
        f"- Output dir: `{manifest['out_dir']}`",
        f"- Videos considered: `{manifest['summary']['videos_considered']}`",
        f"- Sheets rendered: `{manifest['summary']['sheets_rendered']}`",
        f"- Candidates rendered: `{manifest['summary']['candidates_rendered']}`",
        "",
        "## Sheets",
        "",
        "| video | split | status | filter | candidates | sheet |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in manifest["videos"]:
        if not row["sheets"]:
            lines.append(
                f"| `{row['video_name']}` | {row['split']} | {row['status']} |  | 0 | no matching unchecked hints |"
            )
            continue
        for sheet in row["sheets"]:
            lines.append(
                f"| `{row['video_name']}` | {row['split']} | {row['status']} | {sheet['filter']} | "
                f"{sheet['candidate_count']} | `{sheet['path']}` |"
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
            "- Scan the likely/model sheet first; those are the only high-priority possible touches.",
            "- Use the app's `Likely unchecked` filter to approve/reject those moments.",
            "- If the audio-tail sheet is visually all non-touch, use `Shift+A` in the app to mark that tail no-touch and save the clip complete.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def build_contact_sheets(args: argparse.Namespace) -> dict[str, Any]:
    review_manifest = args.review_manifest.resolve()
    candidates_dir = args.candidates_dir.resolve()
    labels_dir = args.labels_dir.resolve()
    out_dir = args.out_dir.resolve()
    items = load_items(review_manifest)
    filters = list(args.filters)
    videos: list[dict[str, Any]] = []
    for item in items:
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
        include_generated = True
        labels_file = label_path(labels_dir, video_id)
        if labels_file.exists():
            labels_doc = read_json(labels_file)
            include_generated = not bool(labels_doc.get("legacy_import"))
        hints = clustered_review_hints(
            candidate_payload,
            cluster_gap_sec=args.candidate_cluster_gap_sec,
            include_generated_event_hints=include_generated,
        )
        reviews = load_reviews(labels_dir, item)
        video_path = Path(str(item["video_path"]))
        video_sheets: list[dict[str, Any]] = []
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
            for page_index, page_hints in enumerate(chunked(matching, args.max_candidates_per_sheet), start=1):
                suffix = f"_{filter_name}_p{page_index:03d}.png"
                sheet_path = out_dir / video_id / f"{safe_slug(video_id)}{suffix}"
                video_sheets.append(
                    render_sheet(
                        video_path=video_path,
                        hints=page_hints,
                        filter_name=filter_name,
                        out_path=sheet_path,
                        cols=args.cols,
                        thumb_width=args.thumb_width,
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
                "sheets": video_sheets,
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
        "summary": {
            "videos_considered": len(videos),
            "sheets_rendered": sum(len(row["sheets"]) for row in videos),
            "candidates_rendered": sum(
                int(sheet["candidate_count"]) for row in videos for sheet in row["sheets"]
            ),
        },
        "videos": videos,
    }
    write_json(out_dir / "touch_review_contact_sheets.json", manifest)
    write_markdown(out_dir / "touch_review_contact_sheets.md", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render touch-review candidate contact sheets")
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW_MANIFEST)
    parser.add_argument("--candidates-dir", type=Path, default=DEFAULT_CANDIDATES_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split", default="test_frozen")
    parser.add_argument("--filters", nargs="+", choices=FILTER_CHOICES, default=["likely_unchecked", "audio_only"])
    parser.add_argument("--only-incomplete", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--review-match-tolerance-sec", type=float, default=0.05)
    parser.add_argument("--candidate-cluster-gap-sec", type=float, default=0.04)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--max-candidates-per-sheet", type=int, default=24)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--thumb-width", type=int, default=220)
    parser.add_argument("--label-height", type=int, default=62)
    return parser.parse_args()


def main() -> None:
    manifest = build_contact_sheets(parse_args())
    print(f"json:   {manifest['out_dir']}/touch_review_contact_sheets.json")
    print(f"report: {manifest['out_dir']}/touch_review_contact_sheets.md")
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
