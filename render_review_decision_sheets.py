#!/usr/bin/env python3
"""Render current review-batch decision sheets from assisted suggestions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from qa_rally_enrichment import OUT_SIZE, read_resized_frame


ROOT = Path(__file__).resolve().parent
DEFAULT_BATCH = ROOT / "outputs" / "review_batches" / "latest_review_batch.json"
DEFAULT_SUGGESTIONS = ROOT / "outputs" / "review_batches" / "assisted_review_suggestions.json"
DEFAULT_OUT_DIR = ROOT / "outputs" / "review_batches" / "decision_sheets"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def labeled_tile(item: dict[str, Any]) -> np.ndarray | None:
    tile_path = rel_path(item["tile_path"])
    tile = cv2.imread(str(tile_path), cv2.IMREAD_COLOR)
    if tile is None:
        return None
    header = f"{item['kind']} {float(item['time_sec']):.3f}s {item['bucket']}"
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 26), (20, 20, 20), -1)
    cv2.putText(tile, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def write_sheet(items: list[dict[str, Any]], out_path: Path, *, cols: int = 4) -> None:
    tiles = [tile for item in items if (tile := labeled_tile(item)) is not None]
    if not tiles:
        return
    tile_h = max(tile.shape[0] for tile in tiles)
    tile_w = max(tile.shape[1] for tile in tiles)
    rows = math.ceil(len(tiles) / cols)
    sheet = np.full((rows * tile_h, cols * tile_w, 3), 22, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        y = row * tile_h
        x = col * tile_w
        sheet[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def full_frame_drop_tile(batch_item: dict[str, Any], suggestion: dict[str, Any]) -> np.ndarray | None:
    video = rel_path(batch_item["video_path"])
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame = read_resized_frame(cap, fps, float(batch_item.get("frame_time_sec") or batch_item["time_sec"]))
    cap.release()
    if frame is None:
        return None
    if batch_item.get("qa_ball_x") is not None and batch_item.get("qa_ball_y") is not None:
        x = int(round(float(batch_item["qa_ball_x"])))
        y = int(round(float(batch_item["qa_ball_y"])))
        r = max(18, int(round(float(batch_item.get("qa_ball_radius") or 18) + 8)))
        cv2.circle(frame, (x, y), r, (0, 230, 255), 4, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, (0, 230, 255), -1, cv2.LINE_AA)
    if batch_item.get("x") is not None and batch_item.get("y") is not None:
        cv2.drawMarker(
            frame,
            (int(round(float(batch_item["x"]))), int(round(float(batch_item["y"])))),
            (255, 220, 80),
            markerType=cv2.MARKER_CROSS,
            markerSize=28,
            thickness=3,
            line_type=cv2.LINE_AA,
        )
    label = (
        f"{suggestion['bucket']} {batch_item['source_video']} "
        f"{float(batch_item['time_sec']):.2f}s score={float(suggestion['evidence_score']):.2f}"
    )
    cv2.rectangle(frame, (0, 0), (OUT_SIZE[0], 36), (20, 20, 20), -1)
    cv2.putText(frame, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.resize(frame, (344, 456), interpolation=cv2.INTER_AREA)


def write_drop_full_frame_sheet(
    batch_items: dict[str, dict[str, Any]],
    suggestions: list[dict[str, Any]],
    out_path: Path,
    *,
    cols: int = 4,
) -> None:
    tiles: list[np.ndarray] = []
    for suggestion in suggestions:
        if suggestion["kind"] != "drop_floor":
            continue
        batch_item = batch_items.get(suggestion["batch_item_id"])
        if batch_item is None:
            continue
        tile = full_frame_drop_tile(batch_item, suggestion)
        if tile is not None:
            tiles.append(tile)
    if not tiles:
        return
    tile_h, tile_w = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / cols)
    sheet = np.full((rows * tile_h, cols * tile_w, 3), 22, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        y = row * tile_h
        x = col * tile_w
        sheet[y : y + tile_h, x : x + tile_w] = tile
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render current review decision sheets")
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--suggestions", type=Path, default=DEFAULT_SUGGESTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    batch = load_json(args.batch)
    suggestion_doc = load_json(args.suggestions)
    items = suggestion_doc["items"]
    by_bucket: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_bucket.setdefault(str(item["bucket"]), []).append(item)

    for bucket, bucket_items in sorted(by_bucket.items()):
        out_path = args.out_dir / f"{bucket}_sheet.jpg"
        write_sheet(bucket_items, out_path)
        print(f"{bucket}: {out_path} ({len(bucket_items)} items)")

    batch_items = {item["batch_item_id"]: item for item in batch["items"]}
    drop_path = args.out_dir / "drop_floor_fullframe_sheet.jpg"
    write_drop_full_frame_sheet(batch_items, items, drop_path)
    print(f"drop full frame: {drop_path}")


if __name__ == "__main__":
    main()
