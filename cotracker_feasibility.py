#!/usr/bin/env python3
"""Run a narrow CoTracker feasibility spike on dense-review clips.

This is intentionally not a labeling UI. It answers one question: after a
single human seed point, does CoTracker hold the footbag point across a short
POV clip, or does it drift onto shoes/hands/background?
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
DEFAULT_BATCH = ROOT / "runs" / "release-27-public" / "dense_trajectory_review_v2_with_sources"
DEFAULT_OUT = ROOT / "outputs" / "cotracker_feasibility"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def resolve_clip(batch_dir: Path, clip_id: str | None, clip_dir: Path | None) -> Path:
    if clip_dir is not None:
        path = clip_dir.expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"clip dir not found: {path}")
        return path
    if not clip_id:
        raise SystemExit("Provide --clip-id or --clip-dir.")
    matches = sorted((batch_dir / "clips").glob(f"{clip_id}*"))
    if len(matches) != 1:
        raise SystemExit(f"Expected one clip for {clip_id!r}, found {len(matches)}")
    return matches[0].resolve()


def load_frames(batch_dir: Path, rows: list[dict[str, Any]], resize_scale: float) -> tuple[np.ndarray, np.ndarray]:
    frames_bgr: list[np.ndarray] = []
    frames_rgb: list[np.ndarray] = []
    for row in rows:
        frame_path = batch_dir / str(row["frame_image"])
        image = cv2.imread(str(frame_path))
        if image is None:
            raise SystemExit(f"could not read frame: {frame_path}")
        frames_bgr.append(image)
        if resize_scale != 1.0:
            image_for_model = cv2.resize(
                image,
                (max(1, int(round(image.shape[1] * resize_scale))), max(1, int(round(image.shape[0] * resize_scale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            image_for_model = image
        frames_rgb.append(cv2.cvtColor(image_for_model, cv2.COLOR_BGR2RGB))
    return np.stack(frames_bgr, axis=0), np.stack(frames_rgb, axis=0)


def choose_device(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_cotracker(device: torch.device):
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    model = model.to(device)
    model.eval()
    return model


def run_cotracker(
    frames_rgb: np.ndarray,
    *,
    seed_local_index: int,
    seed_x: float,
    seed_y: float,
    resize_scale: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, str]:
    video = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2)[None].float().to(device)
    query = torch.tensor(
        [[[float(seed_local_index), float(seed_x * resize_scale), float(seed_y * resize_scale)]]],
        dtype=torch.float32,
        device=device,
    )
    model = load_cotracker(device)
    try:
        with torch.no_grad():
            tracks, vis = model(video, queries=query, backward_tracking=True)
    except Exception:
        if device.type == "cpu":
            raise
        cpu = torch.device("cpu")
        model = model.to(cpu)
        with torch.no_grad():
            tracks, vis = model(video.to(cpu), queries=query.to(cpu), backward_tracking=True)
        device = cpu
    xy = tracks[0, :, 0, :].detach().cpu().numpy().astype(np.float64)
    xy /= max(resize_scale, 1e-9)
    visible = vis[0, :, 0].detach().cpu().numpy().astype(np.float64)
    return xy, visible, str(device)


def hint_distances(row: dict[str, Any], x: float, y: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, hint in sorted((row.get("model_hints") or {}).items()):
        hx = hint.get("x")
        hy = hint.get("y")
        if hx is None or hy is None:
            continue
        out[name] = round(float(math.hypot(float(x) - float(hx), float(y) - float(hy))), 3)
    return out


def summarize(rows: list[dict[str, Any]], points: np.ndarray, visible: np.ndarray, seed_local_index: int) -> dict[str, Any]:
    by_source: dict[str, list[float]] = defaultdict(list)
    for row, point in zip(rows, points):
        for name, dist in hint_distances(row, float(point[0]), float(point[1])).items():
            by_source[name].append(float(dist))
    source_summary = {}
    for name, vals in by_source.items():
        arr = np.array(vals, dtype=np.float64)
        source_summary[name] = {
            "frames": int(len(vals)),
            "median_dist_px": round(float(np.median(arr)), 3),
            "p90_dist_px": round(float(np.quantile(arr, 0.9)), 3),
            "within_30px_rate": round(float(np.mean(arr <= 30.0)), 4),
            "within_60px_rate": round(float(np.mean(arr <= 60.0)), 4),
        }
    jumps = np.linalg.norm(np.diff(points, axis=0), axis=1) if len(points) > 1 else np.array([])
    return {
        "frames": len(rows),
        "seed_local_index": seed_local_index,
        "seed_frame_index": rows[seed_local_index]["frame_index"],
        "visibility_mean": round(float(np.mean(visible)), 4),
        "visibility_min": round(float(np.min(visible)), 4),
        "visibility_below_0_5_frames": int(np.sum(visible < 0.5)),
        "jump_median_px": None if len(jumps) == 0 else round(float(np.median(jumps)), 3),
        "jump_p90_px": None if len(jumps) == 0 else round(float(np.quantile(jumps, 0.9)), 3),
        "hint_agreement": source_summary,
    }


def render_overlay(
    out_path: Path,
    frames_bgr: np.ndarray,
    rows: list[dict[str, Any]],
    points: np.ndarray,
    visible: np.ndarray,
    *,
    seed_local_index: int,
    label: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_bgr[0].shape[:2]
    fps = 12.0
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    trail: list[tuple[int, int]] = []
    colors = {
        "v10_greedy": (0, 0, 255),
        "v10_temporal": (255, 0, 255),
        "v11_greedy": (255, 128, 0),
    }
    for idx, (frame, row, point, vis) in enumerate(zip(frames_bgr, rows, points, visible)):
        image = frame.copy()
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        trail.append((x, y))
        for a, b in zip(trail[-20:], trail[-19:]):
            cv2.line(image, a, b, (0, 255, 255), 2)
        for name, hint in sorted((row.get("model_hints") or {}).items()):
            hx = hint.get("x")
            hy = hint.get("y")
            if hx is None or hy is None:
                continue
            color = colors.get(name, (180, 180, 180))
            cv2.circle(image, (int(round(float(hx))), int(round(float(hy)))), 8, color, 2)
            cv2.putText(image, name, (int(round(float(hx))) + 10, int(round(float(hy))) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 9, (0, 255, 255), -1 if vis >= 0.5 else 2)
        cv2.circle(image, (x, y), 14, (0, 0, 0), 2)
        if idx == seed_local_index:
            cv2.circle(image, (x, y), 22, (255, 255, 255), 3)
        text = f"{label} f={row['frame_index']} t={row['time_sec']:.2f}s vis={vis:.2f}"
        cv2.rectangle(image, (0, 0), (w, 34), (0, 0, 0), -1)
        cv2.putText(image, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(image)
    writer.release()


def run_clip(args: argparse.Namespace) -> dict[str, Any]:
    batch_dir = args.batch_dir.expanduser().resolve()
    clip_dir = resolve_clip(batch_dir, args.clip_id, args.clip_dir)
    label_path = clip_dir / "trajectory_labels_template.jsonl"
    rows = read_jsonl(label_path)
    frame_to_local = {int(row["frame_index"]): idx for idx, row in enumerate(rows)}
    if args.seed_frame not in frame_to_local:
        raise SystemExit(f"seed frame {args.seed_frame} is not in {label_path}")
    seed_local = frame_to_local[args.seed_frame]
    frames_bgr, frames_rgb = load_frames(batch_dir, rows, args.resize_scale)
    device = choose_device(args.device)
    points, visible, used_device = run_cotracker(
        frames_rgb,
        seed_local_index=seed_local,
        seed_x=args.x,
        seed_y=args.y,
        resize_scale=args.resize_scale,
        device=device,
    )
    out_dir = args.out_dir.expanduser().resolve() / (args.name or clip_dir.name)
    point_rows: list[dict[str, Any]] = []
    for row, point, vis in zip(rows, points, visible):
        point_rows.append(
            {
                "clip_id": row["clip_id"],
                "source_video": row["source_video"],
                "frame_index": row["frame_index"],
                "time_sec": row["time_sec"],
                "x": round(float(point[0]), 3),
                "y": round(float(point[1]), 3),
                "visibility_score": round(float(vis), 5),
                "seed": int(row["frame_index"]) == args.seed_frame,
                "hint_distances_px": hint_distances(row, float(point[0]), float(point[1])),
            }
        )
    summary = summarize(rows, points, visible, seed_local)
    summary.update(
        {
            "clip_dir": str(clip_dir),
            "clip_id": rows[0]["clip_id"],
            "label": args.name or clip_dir.name,
            "device": used_device,
            "resize_scale": args.resize_scale,
            "seed": {"frame_index": args.seed_frame, "x": args.x, "y": args.y},
            "outputs": {
                "points_jsonl": str(out_dir / "propagated_points.jsonl"),
                "overlay_mp4": str(out_dir / "overlay.mp4"),
                "summary_json": str(out_dir / "summary.json"),
            },
        }
    )
    write_jsonl(out_dir / "propagated_points.jsonl", point_rows)
    write_json(out_dir / "summary.json", summary)
    render_overlay(out_dir / "overlay.mp4", frames_bgr, rows, points, visible, seed_local_index=seed_local, label=summary["label"])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CoTracker3 on one dense-review clip from a single seed point.")
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--clip-id")
    parser.add_argument("--clip-dir", type=Path)
    parser.add_argument("--seed-frame", type=int, required=True)
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--name")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resize-scale", type=float, default=0.5)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    return parser.parse_args()


def main() -> None:
    summary = run_clip(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
