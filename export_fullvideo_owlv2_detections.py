#!/usr/bin/env python3
"""Run OWLv2 over EVERY frame of one video (not just touch-candidate windows).

The HUD ball trail draws from the detection cache, which only covers windows
around touch candidates -- the trail's discontinuity between rallies is mostly
missing coverage, not tracker failure. This produces a full-coverage
detections.jsonl in the exact schema the HUD renderer already consumes
(pass it via --detections-jsonl).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2

from export_touch_owlv2_detections import make_detector, write_json, write_jsonl
from render_touch_release_hud import DEFAULT_CORPUS, video_lookup, DEFAULT_INVENTORY

DEFAULT_PROMPTS = ["a footbag", "a hacky sack", "a small ball", "a small round bean bag", "a ball"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--model", choices=["owlv2", "owlv2-large"], default="owlv2")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()

    video = video_lookup(args.inventory).get(args.video_id)
    if video is None:
        raise SystemExit(f"unknown video_id: {args.video_id}")
    out_dir = args.out_dir or (DEFAULT_CORPUS / f"owlv2_fullvideo_{args.video_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = make_detector(args.model, DEFAULT_PROMPTS, args.device)
    cap = cv2.VideoCapture(str(video["video_path"]))
    if not cap.isOpened():
        raise SystemExit(f"could not open {video['video_path']}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    records = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % args.frame_stride == 0:
            detections = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            records.append({
                "source_video": video["video_name"],
                "video_id": args.video_id,
                "split": video.get("split"),
                "frame_index": frame_index,
                "time_sec": frame_index / fps,
                "detections": detections,
                "top_score": None if not detections else float(detections[0]["score"]),
                "fires": bool(detections and float(detections[0]["score"]) >= args.threshold),
            })
            if args.progress_every and len(records) % args.progress_every == 0:
                print(f"full-video owlv2: {len(records)}/{total // args.frame_stride} frames", file=sys.stderr, flush=True)
        frame_index += 1
    cap.release()

    write_jsonl(out_dir / "detections.jsonl", records)
    write_json(out_dir / "owlv2_detection_manifest.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "full_video",
        "video_id": args.video_id,
        "video_path": str(video["video_path"]),
        "model": args.model,
        "device": args.device,
        "threshold": args.threshold,
        "frame_stride": args.frame_stride,
        "frames_exported": len(records),
        "frames_total": total,
        "prompts": DEFAULT_PROMPTS,
    })
    print(f"wrote {len(records)} frames -> {out_dir / 'detections.jsonl'}")


if __name__ == "__main__":
    main()
