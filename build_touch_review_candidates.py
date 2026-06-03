#!/usr/bin/env python3
"""Build loose audio touch-candidate hints for visual review.

These candidates are not labels. They are intentionally high-recall hints for a
human reviewer watching muted video. The output is consumed by the event-level
touch review step and by later feature extraction.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from owlv2_event_eval import AUDIO_DELTA, AUDIO_WAIT_SEC


ROOT = Path(__file__).resolve().parent
DEFAULT_REVIEW_MANIFEST = ROOT / "runs/release-27-public/touch_corpus_v1/touch_review_manifest.json"
DEFAULT_OUT_DIR = ROOT / "runs/release-27-public/touch_corpus_v1/audio_candidates"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def extract_audio(video_path: Path, wav_path: Path, force: bool) -> None:
    if wav_path.exists() and not force:
        return
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-ac", "1", "-ar", "48000", str(wav_path)],
        check=True,
        capture_output=True,
    )


def audio_onsets(video_path: Path, out_dir: Path, *, delta: float, wait_sec: float, force: bool) -> list[dict[str, Any]]:
    import librosa

    safe_name = video_path.stem.replace("/", "_")
    wav_path = out_dir / "wav" / f"{safe_name}.wav"
    extract_audio(video_path, wav_path, force)
    audio, sr = librosa.load(str(wav_path), sr=48000, mono=True)
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    env_times = librosa.times_like(onset_env, sr=sr)
    wait_frames = max(0, int(round(wait_sec * sr / 512.0)))
    onset_times = librosa.onset.onset_detect(
        y=audio,
        sr=sr,
        units="time",
        onset_envelope=onset_env,
        backtrack=False,
        delta=delta,
        wait=wait_frames,
    )
    rows: list[dict[str, Any]] = []
    for onset_time in onset_times:
        idx = int(np.argmin(np.abs(env_times - onset_time)))
        rows.append(
            {
                "time_sec": round(float(onset_time), 6),
                "strength": round(float(onset_env[idx]), 6),
                "source": "loose_audio_onset",
            }
        )
    return rows


def existing_event_hints(events_path: str | None) -> list[dict[str, Any]]:
    if not events_path:
        return []
    path = Path(events_path)
    if not path.exists():
        return []
    doc = read_json(path)
    hints: list[dict[str, Any]] = []
    for rally in doc.get("rallies", []):
        for event in rally.get("events", []):
            if event.get("time_sec") is None:
                continue
            event_type = event.get("type")
            if event_type not in {"touch", "drop_floor", "stall"}:
                continue
            hints.append(
                {
                    "time_sec": round(float(event["time_sec"]), 6),
                    "source": f"existing_{event_type}",
                    "event_type": event_type,
                    "rally_id": rally.get("id"),
                    "annotation_is_truth": False,
                }
            )
    return sorted(hints, key=lambda item: float(item["time_sec"]))


def generated_hint_paths(video_id: str) -> list[Path]:
    candidates = [
        ROOT / "outputs" / video_id / "trained_events.json",
        ROOT / "outputs" / video_id / "grounded_events.json",
        ROOT / "outputs" / video_id / "candidate_events.json",
        ROOT / "outputs" / "full_training_improved" / video_id / "trained_events.json",
        ROOT / "outputs" / "full_training_baseline" / video_id / "trained_events.json",
        ROOT / "outputs" / "full_training_27_qa" / video_id / "qa_events.json",
    ]
    out: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        out.append(path)
    return out


def generated_event_hints(video_id: str, *, enabled: bool) -> list[dict[str, Any]]:
    if not enabled:
        return []
    hints: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for path in generated_hint_paths(video_id):
        doc = read_json(path)
        for rally in doc.get("rallies", []):
            for event in rally.get("events", []):
                event_type = event.get("type")
                if event_type not in {"touch", "drop_floor", "stall"} or event.get("time_sec") is None:
                    continue
                time_sec = round(float(event["time_sec"]), 6)
                key = (str(event_type), int(round(time_sec * 20.0)))
                if key in seen:
                    continue
                seen.add(key)
                row = {
                    "time_sec": time_sec,
                    "source": f"generated_{event_type}",
                    "event_type": event_type,
                    "rally_id": rally.get("id"),
                    "generated_source_path": str(path),
                    "annotation_is_truth": False,
                }
                for field in ("confidence", "label", "audio_z", "visual_score", "motion_score", "x", "y"):
                    if event.get(field) is not None:
                        row[field] = event[field]
                hints.append(row)
    return sorted(hints, key=lambda item: float(item["time_sec"]))


def build_candidates(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_json(args.review_manifest)
    out_dir = args.out_dir.resolve()
    rows: list[dict[str, Any]] = []
    items = manifest.get("items", [])
    if args.limit is not None:
        items = items[: args.limit]
    for index, item in enumerate(items, start=1):
        video_path = Path(item["video_path"])
        if not video_path.exists():
            raise FileNotFoundError(f"missing video: {video_path}")
        candidates = audio_onsets(
            video_path,
            out_dir,
            delta=args.audio_delta,
            wait_sec=args.audio_wait_sec,
            force=args.force,
        )
        existing_hints = existing_event_hints(item.get("existing_events_path"))
        generated_hints = generated_event_hints(item["video_id"], enabled=args.include_generated_event_hints)
        payload = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "video_id": item["video_id"],
            "video_name": item["video_name"],
            "video_path": item["video_path"],
            "split": item["split"],
            "audio_parameters": {
                "delta": args.audio_delta,
                "wait_sec": args.audio_wait_sec,
            },
            "review_note": (
                "Candidates are hints only. Human reviewer must confirm contact from muted video "
                "before writing ground truth."
            ),
            "audio_candidates": candidates,
            "existing_event_hints": existing_hints,
            "generated_event_hints": generated_hints,
        }
        candidate_path = out_dir / f"{item['video_id']}.touch_candidates.json"
        write_json(candidate_path, payload)
        rows.append(
            {
                "video_id": item["video_id"],
                "video_name": item["video_name"],
                "split": item["split"],
                "candidate_path": str(candidate_path),
                "audio_candidates": len(candidates),
                "existing_event_hints": len(existing_hints),
                "generated_event_hints": len(generated_hints),
            }
        )
        print(
            f"{index:>2}/{len(items)} {item['video_name']}: "
            f"audio={len(candidates)} existing={len(existing_hints)} generated={len(generated_hints)}"
        )
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "review_manifest": str(args.review_manifest),
        "out_dir": str(out_dir),
        "videos": len(rows),
        "audio_parameters": {
            "delta": args.audio_delta,
            "wait_sec": args.audio_wait_sec,
        },
        "total_audio_candidates": sum(int(row["audio_candidates"]) for row in rows),
        "total_existing_event_hints": sum(int(row["existing_event_hints"]) for row in rows),
        "total_generated_event_hints": sum(int(row["generated_event_hints"]) for row in rows),
        "rows": rows,
    }
    write_json(out_dir / "touch_candidate_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build loose audio touch-review candidate hints")
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--audio-delta", type=float, default=AUDIO_DELTA)
    parser.add_argument("--audio-wait-sec", type=float, default=AUDIO_WAIT_SEC)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include-generated-event-hints", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    summary = build_candidates(parse_args())
    print(f"summary: {summary['out_dir']}/touch_candidate_summary.json")
    print(
        json.dumps(
            {
                "videos": summary["videos"],
                "total_audio_candidates": summary["total_audio_candidates"],
                "total_existing_event_hints": summary["total_existing_event_hints"],
                "total_generated_event_hints": summary["total_generated_event_hints"],
                "audio_parameters": summary["audio_parameters"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
