#!/usr/bin/env python3
"""Reproducible Hacky Track pipeline runner."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_MANIFEST = ROOT / "outputs" / "full_training_27" / "full_training_manifest.json"
DEFAULT_QA_MANIFEST = ROOT / "outputs" / "full_training_27_qa" / "qa_manifest.json"
DEFAULT_HUD_VIDEO = ROOT / "outputs" / "best_rally_hud" / "best_rally_sprite_hud_overlay.mp4"
DEFAULT_REVIEW_BATCH = ROOT / "outputs" / "review_batches" / "latest_review_batch.json"
DEFAULT_BALL_AUDIT = ROOT / "outputs" / "ball_tracking_audit" / "audit_metrics.json"
DEFAULT_VALIDATION = ROOT / "outputs" / "review_validation" / "validation_metrics.json"
DEFAULT_HUD_SUMMARY = ROOT / "outputs" / "best_rally_hud" / "best_rally_sprite_hud_summary.json"
DEFAULT_HUD_VERIFY = ROOT / "outputs" / "best_rally_hud" / "hud_verification.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def videos_from_manifest(path: Path) -> list[Path]:
    manifest = read_json(path)
    return [Path(run["video"]) for run in manifest.get("runs", []) if Path(run["video"]).exists()]


def run_step(name: str, cmd: list[str], *, cwd: Path) -> None:
    print(f"\n== {name} ==")
    print(" ".join(str(part) for part in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def verify_hud(video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open HUD video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"HUD video has no readable first frame: {video_path}")
    first_frame_mean = float(frame.mean())
    if first_frame_mean <= 1.0:
        raise RuntimeError(f"HUD video appears blank: {video_path}")
    streams: list[str] = []
    if shutil.which("ffprobe"):
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,codec_name,duration",
                "-of",
                "compact=p=0:nk=1",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        streams = [line for line in result.stdout.splitlines() if line.strip()]
        if not any("|video|" in line for line in streams):
            raise RuntimeError("HUD video has no video stream")
        if not any("|audio|" in line for line in streams):
            raise RuntimeError("HUD video has no audio stream")
    return {
        "path": str(video_path),
        "fps": round(float(fps), 3),
        "frames": frames,
        "duration_sec": None if fps <= 0 else round(frames / fps, 3),
        "width": width,
        "height": height,
        "first_frame_mean": round(first_frame_mean, 2),
        "streams": streams,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Hacky Track full pipeline")
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--qa-manifest", type=Path, default=DEFAULT_QA_MANIFEST)
    parser.add_argument("--skip-training", action="store_true", help="Reuse existing full_training_manifest.json")
    parser.add_argument("--skip-qa", action="store_true", help="Reuse existing QA manifest")
    parser.add_argument("--skip-review-batch", action="store_true")
    parser.add_argument("--review-batch-size", type=int, default=96)
    parser.add_argument("--skip-ball-audit", action="store_true")
    parser.add_argument("--ball-audit-max-events", type=int, default=240)
    parser.add_argument("--skip-assisted-review", action="store_true")
    parser.add_argument("--seed-review-batch", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-hud", action="store_true")
    parser.add_argument("--skip-goal-status", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = sys.executable
    if not args.skip_training:
        videos = videos_from_manifest(args.train_manifest) if args.train_manifest.exists() else []
        if not videos:
            raise RuntimeError(f"No videos found in {args.train_manifest}")
        run_step(
            "full training and base event generation",
            [
                python,
                "full_training_run.py",
                *[str(video) for video in videos],
                "--out-root",
                "outputs/full_training_27",
                "--model-dir",
                "models/full_training_27",
            ],
            cwd=ROOT,
        )
    if not args.skip_qa:
        run_step(
            "QA enrichment with hidden drop detection",
            [
                python,
                "qa_rally_enrichment.py",
                "--manifest",
                str(args.train_manifest),
                "--out-root",
                "outputs/full_training_27_qa",
            ],
            cwd=ROOT,
        )
    if not args.skip_review_batch:
        run_step(
            "build balanced visual review batch",
            [
                python,
                "build_review_batch.py",
                "--manifest",
                str(args.qa_manifest),
                "--max-items",
                str(args.review_batch_size),
            ],
            cwd=ROOT,
        )
    if not args.skip_ball_audit:
        run_step(
            "independent ball tracking audit",
            [
                python,
                "audit_ball_tracking.py",
                "--manifest",
                str(args.qa_manifest),
                "--max-events",
                str(args.ball_audit_max_events),
            ],
            cwd=ROOT,
        )
    if not args.skip_assisted_review:
        run_step(
            "assisted review triage suggestions",
            [
                python,
                "assist_review_batch.py",
                "--batch",
                str(DEFAULT_REVIEW_BATCH),
                "--audit-results",
                str(ROOT / "outputs" / "ball_tracking_audit" / "audit_results.jsonl"),
            ],
            cwd=ROOT,
        )
    if args.seed_review_batch:
        run_step(
            "seed priority review batch as pending",
            [python, "seed_review_batch.py", "--batch", str(DEFAULT_REVIEW_BATCH)],
            cwd=ROOT,
        )
    if not args.skip_validation:
        run_step(
            "review validation export",
            [python, "review_validation.py", "--batch", str(DEFAULT_REVIEW_BATCH), "--ball-audit", str(DEFAULT_BALL_AUDIT)],
            cwd=ROOT,
        )
    if not args.skip_hud:
        run_step(
            "strict complete best-rally HUD render",
            [python, "render_best_rally_hud.py", "--manifest", str(args.qa_manifest)],
            cwd=ROOT,
        )
        verification = verify_hud(DEFAULT_HUD_VIDEO)
        out_path = DEFAULT_HUD_VERIFY
        out_path.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
        print(f"hud verification: {out_path}")
    if not args.skip_goal_status:
        run_step(
            "goal stop-gate status report",
            [
                python,
                "goal_status_report.py",
                "--qa-manifest",
                str(args.qa_manifest),
                "--ball-audit",
                str(DEFAULT_BALL_AUDIT),
                "--validation",
                str(DEFAULT_VALIDATION),
                "--hud-summary",
                str(DEFAULT_HUD_SUMMARY),
                "--hud-verification",
                str(DEFAULT_HUD_VERIFY),
            ],
            cwd=ROOT,
        )


if __name__ == "__main__":
    main()
