#!/usr/bin/env python3
"""Head-stabilized ball-trajectory proxy for contact SIDE (left vs right foot).

The old screen-ball proxy (0.553, below the 0.671 majority baseline) read raw
screen coordinates, which a head-mounted camera corrupts: when the head turns,
the ball moves on screen without moving in the world. This experiment removes
that confound: estimate global camera motion frame-to-frame (phase correlation
on downscaled grayscale background) over the 0.5s before each contact, subtract
it from the cached OWLv2 ball track, and read where the stabilized incoming
trajectory actually lands relative to the body midline.

Features per contact (all in stabilized, contact-frame coordinates):
  - landing x offset of the ball at contact vs frame center
  - mean incoming horizontal velocity over the approach
  - raw screen x offset (the old failed proxy, kept as a control)

Zero-fit sign rules + leave-one-video-out logistic regression vs 0.671.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import audio_type_ablation as ab
import train_release_contact_classifier as tc
from stereo_side_ablation import video_file_for, lovo_eval

DEFAULT_OUT_DIR = tc.DEFAULT_CORPUS / "trajectory_side_ablation_v1"
DETECTION_DIRS = [
    "owlv2_touch_detections_v1",
    "owlv2_touch_detections_assistant_v1",
    "owlv2_touch_detections_contact_missing_v1",
    "owlv2_stall_drop_missing_detections_v1",
]
PRE_SEC = 0.5
STAB_SCALE = 0.25
MAX_TRACK_JUMP_PX = 220.0
BASELINE = 0.671


def load_detections() -> dict[str, list[dict[str, Any]]]:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    for name in DETECTION_DIRS:
        path = tc.DEFAULT_CORPUS / name / "detections.jsonl"
        if not path.exists():
            continue
        for row in tc.read_jsonl(path):
            key = (row["video_id"], row["frame_index"])
            if key in seen:
                continue
            seen.add(key)
            by_video[str(row["video_id"])].append(row)
    for rows in by_video.values():
        rows.sort(key=lambda r: float(r["time_sec"]))
    return by_video


def ball_track(frames: list[dict[str, Any]], contact_t: float) -> list[dict[str, Any]]:
    """Greedy nearest-to-previous track over the pre-contact window, walking
    backward from the contact frame (where the top detection is most reliable)."""
    window = [f for f in frames if contact_t - PRE_SEC <= float(f["time_sec"]) <= contact_t + 0.05]
    window.sort(key=lambda f: float(f["time_sec"]), reverse=True)
    track: list[dict[str, Any]] = []
    prev_xy: tuple[float, float] | None = None
    for frame in window:
        dets = frame.get("detections") or []
        if not dets:
            continue
        if prev_xy is None:
            best = max(dets, key=lambda d: float(d["score"]))
        else:
            best = min(dets, key=lambda d: (d["x"] - prev_xy[0]) ** 2 + (d["y"] - prev_xy[1]) ** 2)
            if float(np.hypot(best["x"] - prev_xy[0], best["y"] - prev_xy[1])) > MAX_TRACK_JUMP_PX:
                continue
        prev_xy = (float(best["x"]), float(best["y"]))
        track.append({"time_sec": float(frame["time_sec"]), "frame_index": int(frame["frame_index"]),
                      "x": prev_xy[0], "y": prev_xy[1]})
    track.reverse()
    return track


def camera_shifts(video_path: Path, fps: float, t0: float, t1: float) -> dict[int, tuple[float, float]] | None:
    """Cumulative camera translation per frame index over [t0, t1], relative to
    the LAST frame (the contact frame): shift[i] = camera offset of frame i."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    f_start, f_end = int(t0 * fps), int(np.ceil(t1 * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
    grays: list[tuple[int, np.ndarray]] = []
    for fi in range(f_start, f_end + 1):
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, None, fx=STAB_SCALE, fy=STAB_SCALE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        grays.append((fi, cv2.GaussianBlur(gray, (5, 5), 0)))
    cap.release()
    if len(grays) < 2:
        return None
    win = cv2.createHanningWindow(grays[0][1].shape[::-1], cv2.CV_32F)
    # Pairwise shift of frame i relative to i+1, accumulated back from the end.
    shifts: dict[int, tuple[float, float]] = {grays[-1][0]: (0.0, 0.0)}
    acc = np.zeros(2)
    for (fi, g0), (_, g1) in zip(reversed(grays[:-1]), reversed(grays[1:])):
        (dx, dy), _resp = cv2.phaseCorrelate(g0, g1, win)
        # g1 = g0 shifted by (dx, dy): camera moved by (-dx, -dy) from i to i+1.
        acc += np.array([dx, dy]) / STAB_SCALE
        shifts[fi] = (float(acc[0]), float(acc[1]))
    return shifts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--search-dir", type=Path, action="append", default=[Path.home() / "Downloads"])
    args = parser.parse_args()

    rows = [r for r in ab.load_labeled_rows() if r.get("contact_side") in ("left", "right")]
    detections = load_detections()
    meta_cache: dict[str, tuple[Path, float, float]] = {}

    feat_rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for r in rows:
        vid = str(r["video_id"])
        contact_t = float(r.get("contact_label_time_sec") or r.get("candidate_time_sec"))
        if vid not in meta_cache:
            path = video_file_for(vid, args.search_dir)
            if path is None:
                skipped.append(f"{vid}: no video file")
                continue
            cap = cv2.VideoCapture(str(path))
            meta_cache[vid] = (path, cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
        path, fps, width = meta_cache[vid]
        track = ball_track(detections.get(vid, []), contact_t)
        if len(track) < 6:
            skipped.append(f"{vid}@{contact_t:.2f}: track too short ({len(track)})")
            continue
        shifts = camera_shifts(path, fps, track[0]["time_sec"], track[-1]["time_sec"])
        if shifts is None:
            skipped.append(f"{vid}@{contact_t:.2f}: stabilization failed")
            continue
        center = width / 2.0
        pts = []
        for p in track:
            shift = shifts.get(p["frame_index"])
            if shift is None:
                continue
            pts.append({"t": p["time_sec"], "raw_x": p["x"], "stab_x": p["x"] + shift[0]})
        if len(pts) < 6:
            skipped.append(f"{vid}@{contact_t:.2f}: too few stabilized points")
            continue
        ts = np.array([p["t"] for p in pts])
        stab_x = np.array([p["stab_x"] for p in pts])
        vel = float(np.polyfit(ts - ts[-1], stab_x, 1)[0]) if len(pts) >= 2 else 0.0
        feat_rows.append({
            "video_id": vid,
            "candidate_time_sec": float(r["candidate_time_sec"]),
            "contact_side": str(r["contact_side"]),
            "traj_landing_dx": float(stab_x[-1] - center),
            "traj_incoming_vx": vel,
            "traj_raw_dx": float(pts[-1]["raw_x"] - center),
            "n_points": len(pts),
        })

    report: dict[str, Any] = {"n": len(feat_rows), "skipped": skipped, "baseline_majority": BASELINE}
    print(f"contacts with stabilized trajectories: {len(feat_rows)}/{len(rows)} (skipped {len(skipped)})")

    for key in ("traj_landing_dx", "traj_raw_dx"):
        vals = np.array([fr[key] for fr in feat_rows])
        side = np.array([fr["contact_side"] for fr in feat_rows])
        pred = np.where(vals < 0, "left", "right")
        acc = float(np.mean(pred == side))
        means = {s: float(np.mean(vals[side == s])) for s in ("left", "right")}
        report[f"sign_rule_{key}"] = {"accuracy": round(acc, 4),
                                      "class_means_px": {k: round(v, 1) for k, v in means.items()}}
        print(f"sign rule {key}: acc={acc:.3f}  mean px L/R = {means['left']:.0f} / {means['right']:.0f}")

    keys = ["traj_landing_dx", "traj_incoming_vx"]
    preds = lovo_eval(feat_rows, keys)
    acc = float(np.mean([p["correct"] for p in preds])) if preds else 0.0
    per_class = {s: float(np.mean([p["correct"] for p in preds if p["label"] == s]) if any(p["label"] == s for p in preds) else 0.0)
                 for s in ("left", "right")}
    report["lovo"] = {"n": len(preds), "accuracy": round(acc, 4),
                      "per_class_recall": {k: round(v, 4) for k, v in per_class.items()},
                      "beats_majority_baseline": acc > BASELINE}
    print(f"LOVO stabilized trajectory: acc={acc:.3f} (baseline {BASELINE}) "
          f"recall L/R = {per_class['left']:.3f}/{per_class['right']:.3f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "trajectory_side_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.out_dir / "contact_side_oof_scores.jsonl").open("w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(p, sort_keys=True) + "\n")
    with (args.out_dir / "trajectory_features.jsonl").open("w", encoding="utf-8") as fh:
        for fr in feat_rows:
            fh.write(json.dumps(fr, sort_keys=True) + "\n")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
