#!/usr/bin/env python3
"""Public Hacky Track release CLI.

This wrapper keeps the internal research scripts usable while exposing a
versioned run directory that a new user can understand.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS = ROOT / "runs"
STRICT_AMBIGUOUS_CONTACT_TYPES = {"unknown_contact", "foot_candidate", "knee_candidate", "ground_candidate"}
DEFAULT_TOUCH_CORPUS = ROOT / "runs/release-27-public/touch_corpus_v1"
DEFAULT_TOUCH_OVERRIDES = ROOT / "release_overrides/touch_visual_overrides_v1.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)


def portable_path_ref(path: Path | None, base: Path = ROOT) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except (OSError, ValueError):
        return path.name if path.is_absolute() else str(path)


def display_command(command: list[str]) -> str:
    cleaned: list[str] = []
    home = str(Path.home())
    for part in command:
        text = str(part)
        if text.startswith(home + "/Downloads/"):
            text = Path(text).name
        elif text.startswith(str(ROOT) + "/"):
            text = text.replace(str(ROOT) + "/", "", 1)
        elif text == str(ROOT):
            text = "."
        elif text.startswith(home + "/"):
            text = text.replace(home + "/", "~/", 1)
        cleaned.append(shlex.quote(text))
    return " ".join(cleaned)


def run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ordered_jsonl_video_ids(path: Path) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in read_jsonl(path):
        video_id = str(row.get("video_id") or "")
        if video_id and video_id not in seen:
            seen.add(video_id)
            out.append(video_id)
    return out


def fmt_release_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def release_gate_passed(summary: dict[str, Any]) -> bool:
    touch = summary.get("touch_classifier") or {}
    hud = summary.get("hud_model_only") or {}
    corrected = summary.get("hud_visual_corrected") or {}
    return bool(
        touch.get("cv_gate_passed")
        and touch.get("frozen_gate_passed")
        and hud.get("status") == "passed"
        and (not corrected or corrected.get("status") == "passed")
    )


def ensure_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Missing required tool `{name}`. Install it first, e.g. `brew install ffmpeg`.")


def free_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run_step(name: str, cmd: list[str], *, cwd: Path, dry_run: bool = False) -> None:
    print(f"\n== {name} ==")
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=cwd, check=True)


def verify_hud(video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open HUD video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_means: list[tuple[int, float]] = []
    for frame_no in [0, max(0, frames // 4), max(0, frames // 2), max(0, (frames * 3) // 4), max(0, frames - 1)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if ok:
            sample_means.append((frame_no, round(float(frame.mean()), 2)))
    cap.release()
    if not sample_means or max(mean for _, mean in sample_means) <= 1.0:
        raise RuntimeError(f"HUD video appears blank: {video_path}")

    streams: list[str] = []
    if shutil.which("ffprobe"):
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,codec_name,duration,width,height",
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
        "path": rel(video_path.resolve(), ROOT),
        "fps": round(float(fps), 3),
        "frames": frames,
        "duration_sec": None if fps <= 0 else round(frames / fps, 3),
        "width": width,
        "height": height,
        "sampled_frame_means": sample_means,
        "streams": streams,
    }


def qa_totals(manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    runs = manifest.get("runs", [])
    return {
        "videos": len(runs),
        "touches": sum(int(item.get("touch_candidates") or 0) for item in runs),
        "drops": sum(int(item.get("ground_hit_candidates") or 0) for item in runs),
        "stalls": sum(int(item.get("stall_candidates") or 0) for item in runs),
        "atw": sum(int(item.get("around_the_world_candidates") or 0) for item in runs),
        "rallies": sum(int(item.get("rallies") or 0) for item in runs),
        "suppressed_touches": sum(int(item.get("suppressed_touch_candidates") or 0) for item in runs),
        "suppressed_drops": sum(int(item.get("suppressed_drop_candidates") or 0) for item in runs),
        "large_ball_corrections": sum(int(item.get("large_ball_corrections") or 0) for item in runs),
    }


def event_time(event: dict[str, Any]) -> float:
    return float(event.get("time_sec", event.get("start_sec", 0.0)) or 0.0)


def strict_complete_rally_count(manifest_path: Path) -> int:
    if not manifest_path.exists():
        return 0
    manifest = read_json(manifest_path)
    count = 0
    for run in manifest.get("runs", []):
        qa_path = Path(str(run.get("qa_events_path") or ""))
        if not qa_path.exists():
            qa_path = ROOT / qa_path
        if not qa_path.exists():
            continue
        doc = read_json(qa_path)
        for rally in doc.get("rallies", []):
            events = [
                item
                for item in doc.get("events", [])
                if int(item.get("qa_rally_id") or -1) == int(rally.get("id") or -2)
            ]
            touches = [item for item in events if item.get("type") == "touch"]
            touch_times = [event_time(item) for item in touches]
            hidden_drop_gaps = [
                touch_times[idx] - touch_times[idx - 1]
                for idx in range(1, len(touch_times))
                if touch_times[idx] - touch_times[idx - 1] > 1.65
            ]
            explicit_drops = [item for item in events if item.get("type") == "drop_floor"]
            contact_events = [item for item in events if item.get("type") in {"touch", "stall", "drop_floor"}]
            terminal_drop = bool(contact_events and contact_events[-1].get("type") == "drop_floor")
            nonterminal_drops = explicit_drops[:-1] if terminal_drop else explicit_drops
            low_ball = [item for item in events if item.get("qa_ball_accuracy") == "low"]
            ambiguous_contacts = [
                item
                for item in touches
                if item.get("contact_type") in STRICT_AMBIGUOUS_CONTACT_TYPES
            ]
            if (
                len(touches) >= 6
                and not nonterminal_drops
                and not hidden_drop_gaps
                and not low_ball
                and not ambiguous_contacts
                and not rally.get("ended_by_gap_without_floor_reset")
            ):
                count += 1
    return count


def export_run_data(run_dir: Path, qa_manifest: Path) -> dict[str, Path]:
    manifest = read_json(qa_manifest)
    all_events: list[dict[str, Any]] = []
    all_rallies: list[dict[str, Any]] = []
    for video_index, run in enumerate(manifest.get("runs", [])):
        qa_path = Path(str(run.get("qa_events_path") or ""))
        if not qa_path.exists():
            qa_path = ROOT / qa_path
        if not qa_path.exists():
            continue
        doc = read_json(qa_path)
        source_video = Path(str(doc.get("source_video") or run.get("video") or f"video_{video_index}")).name
        for event in doc.get("events", []):
            item = dict(event)
            item["source_video"] = source_video
            item["video_index"] = video_index
            all_events.append(item)
        for rally in doc.get("rallies", []):
            item = dict(rally)
            item["source_video"] = source_video
            item["video_index"] = video_index
            all_rallies.append(item)

    events_json = run_dir / "events.json"
    events_csv = run_dir / "events.csv"
    rallies_json = run_dir / "rallies.json"
    write_json(events_json, {"schema_version": 1, "events": all_events})
    write_json(rallies_json, {"schema_version": 1, "rallies": all_rallies})

    fieldnames = [
        "source_video",
        "video_index",
        "type",
        "time_sec",
        "start_sec",
        "end_sec",
        "duration_sec",
        "qa_rally_id",
        "touch_number",
        "confidence",
        "contact_side",
        "contact_type",
        "trick_label",
        "qa_ball_x",
        "qa_ball_y",
        "qa_ball_accuracy",
        "note",
    ]
    with events_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for event in all_events:
            writer.writerow(event)
    return {"events_json": events_json, "events_csv": events_csv, "rallies_json": rallies_json}


def write_summary(run_dir: Path, command: list[str], artifacts: dict[str, Path]) -> None:
    qa_manifest = artifacts.get("qa_manifest")
    totals = qa_totals(qa_manifest) if qa_manifest and qa_manifest.exists() else {}
    audit = read_json(artifacts["ball_audit"]) if artifacts.get("ball_audit") and artifacts["ball_audit"].exists() else {}
    strict_audit = read_json(artifacts["strict_rally_audit"]) if artifacts.get("strict_rally_audit") and artifacts["strict_rally_audit"].exists() else {}
    validation = read_json(artifacts["validation"]) if artifacts.get("validation") and artifacts["validation"].exists() else {}
    release = read_json(artifacts["release_metrics"]) if artifacts.get("release_metrics") and artifacts["release_metrics"].exists() else {}
    hud = read_json(artifacts["hud_verify"]) if artifacts.get("hud_verify") and artifacts["hud_verify"].exists() else {}

    lines = [
        "# Hacky Track Run Summary",
        "",
        f"- Run directory: `{rel(run_dir, ROOT)}`",
        f"- Command: `{display_command(command)}`",
        "",
        "## Totals",
        "",
    ]
    if totals:
        lines.extend(
            [
                f"- Videos: {totals['videos']}",
                f"- Touches: {totals['touches']}",
                f"- Floor resets: {totals['drops']}",
                f"- Stalls: {totals['stalls']}",
                f"- ATW candidates: {totals['atw']}",
                f"- Rallies: {totals['rallies']}",
                f"- Suppressed touches: {totals['suppressed_touches']}",
                f"- Suppressed floor resets: {totals['suppressed_drops']}",
                f"- Large ball corrections: {totals['large_ball_corrections']}",
            ]
        )
    if audit:
        lines.extend(["", "## Ball Audit", "", f"- Pass rate: {float(audit.get('pass_rate') or 0.0) * 100:.1f}%", f"- Sampled events: {audit.get('sampled_events')}"])
    if strict_audit:
        summary = strict_audit.get("summary", {})
        lines.extend(
            [
                "",
                "## Strict Rally Audit",
                "",
                f"- Strict-complete rallies: {summary.get('strict_complete_rallies')}/{summary.get('rallies')}",
                f"- Has strict best rally: {summary.get('has_strict_best_rally')}",
            ]
        )
        rejection_counts = summary.get("rejection_counts") or {}
        if rejection_counts:
            top_reasons = ", ".join(f"{key}={value}" for key, value in sorted(rejection_counts.items())[:5])
            lines.append(f"- Top rejection counts: {top_reasons}")
    batch = validation.get("review_batch") if validation else None
    if batch:
        lines.extend(
            [
                "",
                "## Review",
                "",
                f"- Batch decisions: {batch.get('decided_items')}/{batch.get('total_items')}",
                f"- Pending: {batch.get('pending_items')}",
                f"- Unreviewed: {batch.get('unreviewed_items')}",
            ]
        )
    if hud:
        lines.extend(["", "## HUD Verification", "", f"- Video: `{rel(Path(str(hud.get('path'))), run_dir)}`", f"- Frames: {hud.get('frames')}", f"- Duration: {hud.get('duration_sec')}s"])
    if release:
        blockers = release.get("release_blockers", [])
        lines.extend(
            [
                "",
                "## Release Evaluation",
                "",
                f"- Evaluation ID: `{release.get('evaluation_id')}`",
                f"- Clean current-batch labels: {release.get('label_inventory', {}).get('label_tiers', {}).get('clean_current_batch', 0)}",
                f"- Release blockers: {len(blockers)}",
            ]
        )

    lines.extend(["", "## Artifacts", ""])
    for name, path in artifacts.items():
        if path.exists():
            lines.append(f"- {name}: `{rel(path, run_dir)}`")
    lines.append("")
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def make_run_outputs_portable(run_dir: Path, input_videos: list[Path]) -> dict[str, Any]:
    replacements: list[tuple[str, str]] = []
    root = str(ROOT.resolve())
    home = str(Path.home())
    for video in input_videos:
        replacements.append((str(video), video.name))
        try:
            replacements.append((str(video.resolve()), video.name))
        except OSError:
            pass
    replacements.extend(
        [
            (root + "/", ""),
            (root, "."),
            (home + "/", "~/"),
        ]
    )
    text_suffixes = {".json", ".jsonl", ".md", ".csv", ".txt"}
    changed: list[str] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = text
        for old, new in replacements:
            updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            changed.append(rel(path, run_dir))
    remaining: list[str] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "/Users/" in text:
            remaining.append(rel(path, run_dir))
    audit = {
        "schema_version": 1,
        "policy": "release text artifacts should not contain hardcoded /Users paths",
        "sanitized_files": changed,
        "remaining_files_with_users_paths": remaining,
        "passed": not remaining,
    }
    audit_path = run_dir / "portable_paths_audit.json"
    write_json(audit_path, audit)
    return audit


def create_run_dir(out_root: Path, name: str | None, overwrite: bool) -> Path:
    run_dir = out_root / (name or run_id())
    if run_dir.exists() and not overwrite:
        raise RuntimeError(f"Run directory already exists: {run_dir}. Use --overwrite or choose --run-name.")
    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def process(args: argparse.Namespace) -> None:
    ensure_tool("ffmpeg")
    ensure_tool("ffprobe")
    videos = [path.expanduser().resolve() for path in args.videos]
    missing = [str(path) for path in videos if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing input video(s): {', '.join(missing)}")
    if not videos and not args.skip_training:
        raise RuntimeError("Provide at least one video path, or use --skip-training with an existing run directory.")

    run_dir = create_run_dir(args.out_root.expanduser().resolve(), args.run_name, args.overwrite)
    training_dir = run_dir / "training"
    model_dir = run_dir / "models"
    qa_dir = run_dir / "qa"
    review_dir = run_dir / "review_batches"
    audit_dir = run_dir / "ball_tracking_audit"
    strict_audit_dir = run_dir / "strict_rally_audit"
    validation_dir = run_dir / "validation"
    release_eval_dir = run_dir / "release_evaluation"
    hud_dir = run_dir / "hud"
    reviews_dir = args.reviews_dir.expanduser().resolve() if args.reviews_dir else run_dir / "reviews"

    python = sys.executable
    training_manifest = training_dir / "full_training_manifest.json"
    qa_manifest = qa_dir / "qa_manifest.json"
    review_batch = review_dir / "latest_review_batch.json"
    audit_metrics = audit_dir / "audit_metrics.json"
    audit_results = audit_dir / "audit_results.jsonl"
    strict_audit_json = strict_audit_dir / "strict_rally_audit.json"
    strict_audit_report = strict_audit_dir / "strict_rally_audit.md"
    validation_metrics = validation_dir / "validation_metrics.json"
    release_metrics = release_eval_dir / "release_metrics.json"
    hud_video = hud_dir / "best_rally_sprite_hud_overlay.mp4"
    hud_verify = hud_dir / "hud_verification.json"

    if not args.skip_training:
        run_step(
            "training and base event generation",
            [
                python,
                "full_training_run.py",
                *[str(video) for video in videos],
                "--out-root",
                str(training_dir),
                "--model-dir",
                str(model_dir),
            ],
            cwd=ROOT,
            dry_run=args.dry_run,
        )
    if not args.skip_qa:
        run_step(
            "QA enrichment and rally analytics",
            [python, "qa_rally_enrichment.py", "--manifest", str(training_manifest), "--out-root", str(qa_dir)],
            cwd=ROOT,
            dry_run=args.dry_run,
        )
    if not args.skip_strict_audit:
        if args.dry_run or qa_manifest.exists():
            run_step(
                "strict rally audit",
                [python, "strict_rally_audit.py", "--manifest", str(qa_manifest), "--out-dir", str(strict_audit_dir)],
                cwd=ROOT,
                dry_run=args.dry_run,
            )
        else:
            print(f"warning: strict rally audit skipped because QA manifest is missing: {qa_manifest}", file=sys.stderr)
    if not args.skip_review_batch:
        run_step(
            "review batch",
            [python, "build_review_batch.py", "--manifest", str(qa_manifest), "--out-dir", str(review_dir), "--max-items", str(args.review_batch_size)],
            cwd=ROOT,
            dry_run=args.dry_run,
        )
        run_step(
            "ball tracking audit",
            [python, "audit_ball_tracking.py", "--manifest", str(qa_manifest), "--out-dir", str(audit_dir), "--max-events", str(args.ball_audit_max_events)],
            cwd=ROOT,
            dry_run=args.dry_run,
        )
        run_step(
            "assisted review suggestions",
            [python, "assist_review_batch.py", "--batch", str(review_batch), "--audit-results", str(audit_results), "--out-dir", str(review_dir)],
            cwd=ROOT,
            dry_run=args.dry_run,
        )
    if not args.skip_validation:
        run_step(
            "validation export",
            [
                python,
                "review_validation.py",
                "--reviews-dir",
                str(reviews_dir),
                "--out-dir",
                str(validation_dir),
                "--batch",
                str(review_batch),
                "--ball-audit",
                str(audit_metrics),
            ],
            cwd=ROOT,
            dry_run=args.dry_run,
        )
    if not args.skip_release_evaluation:
        run_step(
            "release evaluation",
            [
                python,
                "release_evaluation.py",
                "--reviews-dir",
                str(reviews_dir),
                "--batch",
                str(review_batch),
                "--training-manifest",
                str(training_manifest),
                "--model-dir",
                str(model_dir),
                "--ball-audit",
                str(audit_metrics),
                "--out-dir",
                str(release_eval_dir),
            ],
            cwd=ROOT,
            dry_run=args.dry_run,
        )
    hud_error = hud_dir / "hud_error.json"
    if not args.skip_hud:
        hud_cmd = [python, "render_best_rally_hud.py", "--manifest", str(qa_manifest), "--out-dir", str(hud_dir)]
        if args.allow_incomplete_hud:
            hud_cmd.append("--allow-incomplete")
        strict_count = 1 if args.dry_run else strict_complete_rally_count(qa_manifest)
        if strict_count == 0 and not args.allow_incomplete_hud:
            message = (
                "HUD render skipped because no strict-complete best rally was available. "
                "Rerun with --allow-incomplete-hud to render the best available rally, "
                "or review/reset events until a strict-complete rally exists."
            )
            strict_audit = read_json(strict_audit_json) if strict_audit_json.exists() else {}
            if args.require_hud:
                raise RuntimeError(message)
            print(f"warning: {message}", file=sys.stderr)
            if not args.dry_run:
                write_json(
                    hud_error,
                    {
                        "error": message,
                        "strict_complete_rallies": strict_count,
                        "require_hud": bool(args.require_hud),
                        "allow_incomplete_hud": bool(args.allow_incomplete_hud),
                        "strict_rally_audit": rel(strict_audit_json, run_dir),
                        "top_rejected_candidates": strict_audit.get("top_rejected_candidates", [])[:5],
                    },
                )
        else:
            try:
                run_step("best-rally HUD render", hud_cmd, cwd=ROOT, dry_run=args.dry_run)
                if not args.dry_run:
                    write_json(hud_verify, verify_hud(hud_video))
            except subprocess.CalledProcessError as exc:
                if args.require_hud:
                    raise
                message = f"HUD render failed with exit code {exc.returncode}."
                print(f"warning: {message}", file=sys.stderr)
                if not args.dry_run:
                    write_json(
                        hud_error,
                        {
                            "error": message,
                            "returncode": exc.returncode,
                            "require_hud": bool(args.require_hud),
                            "allow_incomplete_hud": bool(args.allow_incomplete_hud),
                        },
                    )

    export_artifacts = export_run_data(run_dir, qa_manifest) if qa_manifest.exists() else {}
    artifacts = {
        **export_artifacts,
        "training_manifest": training_manifest,
        "qa_manifest": qa_manifest,
        "qa_report": qa_dir / "qa_report.md",
        "review_batch": review_batch,
        "ball_audit": audit_metrics,
        "strict_rally_audit": strict_audit_json,
        "strict_rally_audit_report": strict_audit_report,
        "validation": validation_metrics,
        "reviewed_labels": validation_dir / "reviewed_labels.jsonl",
        "release_metrics": release_metrics,
        "release_report": release_eval_dir / "release_evaluation_report.md",
        "dataset_splits": release_eval_dir / "dataset_splits.json",
        "portable_paths_audit": run_dir / "portable_paths_audit.json",
        "hud_video": hud_video,
        "hud_preview": hud_dir / "best_rally_sprite_hud_preview.jpg",
        "hud_verify": hud_verify,
        "hud_error": hud_error,
    }
    if not args.dry_run:
        write_json(
            run_dir / "run_manifest.json",
            {
                "schema_version": 1,
                "run_dir": rel(run_dir, ROOT),
                "videos": [video.name for video in videos],
                "reviews_dir": rel(reviews_dir, run_dir),
                "artifacts": {name: rel(path, run_dir) for name, path in artifacts.items()},
            },
        )
        write_summary(run_dir, sys.argv, artifacts)
        make_run_outputs_portable(run_dir, videos)
        print(f"\nrun summary: {run_dir / 'summary.md'}")


def review(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve()
    qa_manifest = args.qa_manifest or run_dir / "qa" / "qa_manifest.json"
    review_batch = args.review_batch or run_dir / "review_batches" / "latest_review_batch.json"
    assisted_review = args.assisted_review or run_dir / "review_batches" / "assisted_review_suggestions.json"
    reviews_dir = args.reviews_dir or run_dir / "reviews"
    port = free_port(args.port)
    cmd = [
        sys.executable,
        "review_app.py",
        "--host",
        args.host,
        "--port",
        str(port),
        "--qa-manifest",
        str(qa_manifest),
        "--review-batch",
        str(review_batch),
        "--assisted-review",
        str(assisted_review),
        "--reviews-dir",
        str(reviews_dir),
    ]
    run_step("review app", cmd, cwd=ROOT, dry_run=args.dry_run)


def review_evidence(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve()
    run_manifest = read_json(run_dir / "run_manifest.json") if (run_dir / "run_manifest.json").exists() else {}
    artifacts = run_manifest.get("artifacts", {}) if isinstance(run_manifest.get("artifacts"), dict) else {}
    manifest_reviews = resolve_run_manifest_path(run_dir, run_manifest.get("reviews_dir"))
    batch = args.batch or resolve_run_manifest_path(run_dir, artifacts.get("review_batch")) or run_dir / "review_batches" / "latest_review_batch.json"
    reviews_dir = args.reviews_dir or manifest_reviews or run_dir / "reviews"
    out_dir = args.out_dir or run_dir / "review_evidence"
    suggestions_dir = args.suggestions_dir or batch.parent
    suggestions = args.suggestions or suggestions_dir / "assisted_review_suggestions.json"
    audit_results = args.audit_results or run_dir / "ball_tracking_audit" / "audit_results.jsonl"
    if not args.skip_suggestions:
        run_step(
            "assisted review suggestions",
            [
                sys.executable,
                "assist_review_batch.py",
                "--batch",
                str(batch),
                "--audit-results",
                str(audit_results),
                "--out-dir",
                str(suggestions_dir),
            ],
            cwd=ROOT,
            dry_run=args.dry_run,
        )
    cmd = [
        sys.executable,
        "prepare_review_evidence.py",
        "--batch",
        str(batch),
        "--suggestions",
        str(suggestions),
        "--reviews-dir",
        str(reviews_dir),
        "--out-dir",
        str(out_dir),
        "--max-items-per-bucket",
        str(args.max_items_per_bucket),
        "--seconds",
        str(args.seconds),
        "--cols",
        str(args.cols),
    ]
    run_step("review evidence pack", cmd, cwd=ROOT, dry_run=args.dry_run)


def seed_reviews(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve()
    run_manifest = read_json(run_dir / "run_manifest.json") if (run_dir / "run_manifest.json").exists() else {}
    artifacts = run_manifest.get("artifacts", {}) if isinstance(run_manifest.get("artifacts"), dict) else {}
    manifest_reviews = resolve_run_manifest_path(run_dir, run_manifest.get("reviews_dir"))
    qa_manifest = args.qa_manifest or resolve_run_manifest_path(run_dir, artifacts.get("qa_manifest")) or run_dir / "qa" / "qa_manifest.json"
    batch = args.batch or resolve_run_manifest_path(run_dir, artifacts.get("review_batch")) or run_dir / "review_batches" / "latest_review_batch.json"
    assisted_review = args.assisted_review or batch.parent / "assisted_review_suggestions.json"
    reviews_dir = args.reviews_dir or manifest_reviews or run_dir / "reviews"
    out_path = args.out or reviews_dir / "seed_review_batch_summary.json"
    cmd = [
        sys.executable,
        "seed_review_batch.py",
        "--batch",
        str(batch),
        "--qa-manifest",
        str(qa_manifest),
        "--reviews-dir",
        str(reviews_dir),
        "--assisted-review",
        str(assisted_review),
        "--out",
        str(out_path),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    run_step("seed review batch", cmd, cwd=ROOT, dry_run=args.dry_run)


def apply_review_decisions(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve()
    run_manifest = read_json(run_dir / "run_manifest.json") if (run_dir / "run_manifest.json").exists() else {}
    manifest_reviews = resolve_run_manifest_path(run_dir, run_manifest.get("reviews_dir"))
    reviews_dir = args.reviews_dir or manifest_reviews or run_dir / "reviews"
    decisions = args.decisions or reviews_dir / "codex_visual_review_decisions.json"
    out_path = args.out or reviews_dir / "apply_review_decisions_summary.json"
    cmd = [
        sys.executable,
        "apply_review_decisions.py",
        "--decisions",
        str(decisions),
        "--reviews-dir",
        str(reviews_dir),
        "--out",
        str(out_path),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    run_step("apply review decisions", cmd, cwd=ROOT, dry_run=args.dry_run)


def verify(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve()
    hud_video = args.hud_video or run_dir / "hud" / "best_rally_sprite_hud_overlay.mp4"
    result = verify_hud(hud_video)
    out_path = run_dir / "hud" / "hud_verification.json"
    write_json(out_path, result)
    print(json.dumps(result, indent=2))
    print(f"verification: {out_path}")


def resolve_run_manifest_path(run_dir: Path, raw: Any) -> Path | None:
    if raw in (None, ""):
        return None
    path = Path(str(raw))
    if path.is_absolute():
        return path
    run_path = run_dir / path
    if run_path.exists() or str(raw).startswith(("training/", "models/", "qa/", "review_batches/", "ball_tracking_audit/", "validation/", "release_evaluation/")):
        return run_path
    return ROOT / path


def evaluate(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else None
    run_manifest = read_json(run_dir / "run_manifest.json") if run_dir and (run_dir / "run_manifest.json").exists() else {}
    artifacts = run_manifest.get("artifacts", {}) if isinstance(run_manifest.get("artifacts"), dict) else {}
    manifest_reviews = resolve_run_manifest_path(run_dir, run_manifest.get("reviews_dir")) if run_dir else None
    reviews_dir = args.reviews_dir or manifest_reviews or (run_dir / "reviews" if run_dir else ROOT / "reviews")
    batch = args.batch or (resolve_run_manifest_path(run_dir, artifacts.get("review_batch")) if run_dir else None) or (ROOT / "outputs" / "review_batches" / "latest_review_batch.json")
    training_manifest = args.training_manifest or (resolve_run_manifest_path(run_dir, artifacts.get("training_manifest")) if run_dir else None)
    model_dir = args.model_dir or (run_dir / "models" if run_dir else None)
    ball_audit = args.ball_audit or (resolve_run_manifest_path(run_dir, artifacts.get("ball_audit")) if run_dir else None)
    out_dir = args.out_dir or (resolve_run_manifest_path(run_dir, "release_evaluation") if run_dir else ROOT / "outputs" / "release_evaluation")
    cmd = [
        sys.executable,
        "release_evaluation.py",
        "--reviews-dir",
        str(reviews_dir),
        "--batch",
        str(batch),
        "--out-dir",
        str(out_dir),
        "--seed",
        str(args.seed),
    ]
    if training_manifest is not None:
        cmd.extend(["--training-manifest", str(training_manifest)])
    if model_dir is not None:
        cmd.extend(["--model-dir", str(model_dir)])
    if ball_audit is not None:
        cmd.extend(["--ball-audit", str(ball_audit)])
    run_step("release evaluation", cmd, cwd=ROOT, dry_run=args.dry_run)


def apply_reviews(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve()
    run_manifest = read_json(run_dir / "run_manifest.json") if (run_dir / "run_manifest.json").exists() else {}
    artifacts = run_manifest.get("artifacts", {}) if isinstance(run_manifest.get("artifacts"), dict) else {}
    manifest_reviews = resolve_run_manifest_path(run_dir, run_manifest.get("reviews_dir"))
    qa_manifest = args.qa_manifest or resolve_run_manifest_path(run_dir, artifacts.get("qa_manifest")) or run_dir / "qa" / "qa_manifest.json"
    reviews_dir = args.reviews_dir or manifest_reviews or run_dir / "reviews"
    out_root = args.out_root or run_dir / "qa_reviewed"
    strict_out = args.strict_audit_out or run_dir / "strict_rally_audit_reviewed"
    run_step(
        "apply review decisions to QA",
        [
            sys.executable,
            "apply_reviews_to_qa.py",
            "--manifest",
            str(qa_manifest),
            "--reviews-dir",
            str(reviews_dir),
            "--out-root",
            str(out_root),
        ],
        cwd=ROOT,
        dry_run=args.dry_run,
    )
    run_step(
        "strict rally audit for reviewed QA",
        [
            sys.executable,
            "strict_rally_audit.py",
            "--manifest",
            str(out_root / "qa_manifest.json"),
            "--out-dir",
            str(strict_out),
        ],
        cwd=ROOT,
        dry_run=args.dry_run,
    )


def export_detector_dataset(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else None
    run_manifest = read_json(run_dir / "run_manifest.json") if run_dir and (run_dir / "run_manifest.json").exists() else {}
    artifacts = run_manifest.get("artifacts", {}) if isinstance(run_manifest.get("artifacts"), dict) else {}
    manifest_reviews = resolve_run_manifest_path(run_dir, run_manifest.get("reviews_dir")) if run_dir else None
    qa_manifest = args.qa_manifest or (resolve_run_manifest_path(run_dir, artifacts.get("qa_manifest")) if run_dir else None)
    reviews_dir = args.reviews_dir or manifest_reviews or (run_dir / "reviews" if run_dir else None)
    out_dir = args.out_dir or (run_dir / "detector_dataset" if run_dir else ROOT / "outputs" / "detector_dataset")
    if qa_manifest is None or reviews_dir is None:
        raise RuntimeError("Provide --run-dir or both --qa-manifest and --reviews-dir.")
    cmd = [
        sys.executable,
        "export_detector_dataset.py",
        "--qa-manifest",
        str(qa_manifest),
        "--reviews-dir",
        str(reviews_dir),
        "--out-dir",
        str(out_dir),
        "--seed",
        str(args.seed),
        "--default-radius",
        str(args.default_radius),
        "--crop-size",
        str(args.crop_size),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.detector_label_review_manifest:
        cmd.extend(["--detector-label-review-manifest", str(args.detector_label_review_manifest)])
    if args.detector_label_decisions:
        cmd.extend(["--detector-label-decisions", str(args.detector_label_decisions)])
    for pair in args.detector_label_review_pair or []:
        cmd.extend(["--detector-label-review-pair", pair])
    run_step("detector dataset export", cmd, cwd=ROOT, dry_run=args.dry_run)


def detector_label_review(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else None
    run_manifest = read_json(run_dir / "run_manifest.json") if run_dir and (run_dir / "run_manifest.json").exists() else {}
    artifacts = run_manifest.get("artifacts", {}) if isinstance(run_manifest.get("artifacts"), dict) else {}
    qa_manifest = args.qa_manifest or (resolve_run_manifest_path(run_dir, artifacts.get("qa_manifest")) if run_dir else None)
    reviews_dir = args.reviews_dir or (resolve_run_manifest_path(run_dir, run_manifest.get("reviews_dir")) if run_dir else None)
    dataset_manifest = args.dataset_manifest or (run_dir / "detector_dataset" / "manifest.json" if run_dir else None)
    if dataset_manifest is not None and not dataset_manifest.exists():
        dataset_manifest = None
    out_dir = args.out_dir or (run_dir / "detector_label_review" if run_dir else ROOT / "outputs" / "detector_label_review")
    if qa_manifest is None:
        raise RuntimeError("Provide --run-dir or --qa-manifest.")
    cmd = [
        sys.executable,
        "build_detector_label_review_batch.py",
        "--qa-manifest",
        str(qa_manifest),
        "--out-dir",
        str(out_dir),
        "--max-items",
        str(args.max_items),
        "--per-video",
        str(args.per_video),
        "--default-radius",
        str(args.default_radius),
        "--crop-size",
        str(args.crop_size),
        "--cols",
        str(args.cols),
    ]
    if reviews_dir:
        cmd.extend(["--reviews-dir", str(reviews_dir)])
    if dataset_manifest:
        cmd.extend(["--dataset-manifest", str(dataset_manifest)])
    if args.video_root:
        cmd.extend(["--video-root", str(args.video_root)])
    if args.dry_run:
        cmd.append("--dry-run")
    run_step("detector label review batch", cmd, cwd=ROOT, dry_run=args.dry_run)


def assist_detector_labels(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else None
    review_manifest = args.review_manifest or (run_dir / "detector_label_review" / "detector_label_review_manifest.json" if run_dir else None)
    out_path = args.out or (review_manifest.parent / "detector_label_assisted_decisions.json" if review_manifest else None)
    if review_manifest is None or out_path is None:
        raise RuntimeError("Provide --run-dir or --review-manifest and --out.")
    cmd = [
        sys.executable,
        "assist_detector_label_decisions.py",
        "--review-manifest",
        str(review_manifest),
        "--out",
        str(out_path),
        "--min-confidence",
        str(args.min_confidence),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    run_step("assist detector label decisions", cmd, cwd=ROOT, dry_run=args.dry_run)


def detector_false_positive_review(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else None
    qa_manifest = args.qa_manifest or qa_manifest_for_run(run_dir)
    inference_root = args.inference_root or (run_dir / "detector_inference" if run_dir else None)
    dataset_manifest = args.dataset_manifest or (run_dir / "detector_dataset" / "manifest.json" if run_dir else None)
    if dataset_manifest is not None and not dataset_manifest.exists():
        dataset_manifest = None
    out_dir = args.out_dir or (run_dir / "detector_false_positive_review" if run_dir else ROOT / "outputs" / "detector_false_positive_review")
    if qa_manifest is None:
        raise RuntimeError("Provide --run-dir or --qa-manifest.")
    cmd = [
        sys.executable,
        "build_detector_false_positive_review_batch.py",
        "--qa-manifest",
        str(qa_manifest),
        "--out-dir",
        str(out_dir),
        "--max-items",
        str(args.max_items),
        "--per-video",
        str(args.per_video),
        "--crop-size",
        str(args.crop_size),
        "--cols",
        str(args.cols),
        "--min-confidence",
        str(args.min_confidence),
        "--max-confidence",
        str(args.max_confidence),
    ]
    if inference_root:
        cmd.extend(["--inference-root", str(inference_root)])
    if args.batch_manifest:
        cmd.extend(["--batch-manifest", str(args.batch_manifest)])
    if dataset_manifest:
        cmd.extend(["--dataset-manifest", str(dataset_manifest)])
    if args.video_root:
        cmd.extend(["--video-root", str(args.video_root)])
    if args.dry_run:
        cmd.append("--dry-run")
    run_step("detector false-positive review batch", cmd, cwd=ROOT, dry_run=args.dry_run)


def detector_error_review(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else None
    qa_manifest = args.qa_manifest or qa_manifest_for_run(run_dir)
    track_metrics = args.track_metrics or (run_dir / "detector_evaluation" / "detector_track_metrics.json" if run_dir else None)
    dataset = args.dataset or (run_dir / "detector_dataset" if run_dir else None)
    model_metrics = args.model_metrics or (run_dir / "detector_model_evaluation" / "detector_model_metrics.json" if run_dir else None)
    if dataset is not None and not dataset.exists():
        dataset = None
    if model_metrics is not None and not model_metrics.exists():
        model_metrics = None
    out_dir = args.out_dir or (run_dir / "detector_error_review" if run_dir else ROOT / "outputs" / "detector_error_review")
    if qa_manifest is None or track_metrics is None:
        raise RuntimeError("Provide --run-dir or both --qa-manifest and --track-metrics.")
    cmd = [
        sys.executable,
        "build_detector_error_review_batch.py",
        "--qa-manifest",
        str(qa_manifest),
        "--track-metrics",
        str(track_metrics),
        "--out-dir",
        str(out_dir),
        "--max-items",
        str(args.max_items),
        "--per-video",
        str(args.per_video),
        "--crop-size",
        str(args.crop_size),
        "--cols",
        str(args.cols),
        "--low-confidence-per-split",
        str(args.low_confidence_per_split),
    ]
    if dataset:
        cmd.extend(["--dataset", str(dataset)])
    if model_metrics:
        cmd.extend(["--model-metrics", str(model_metrics)])
    if args.video_root:
        cmd.extend(["--video-root", str(args.video_root)])
    if args.exclude_audit_only:
        cmd.append("--exclude-audit-only")
    if args.dry_run:
        cmd.append("--dry-run")
    run_step("detector error review batch", cmd, cwd=ROOT, dry_run=args.dry_run)


def dense_trajectory_review(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else None
    qa_manifest = args.qa_manifest or qa_manifest_for_run(run_dir)
    dataset = args.dataset or (run_dir / "detector_dataset" if run_dir else None)
    out_dir = args.out_dir or (run_dir / "dense_trajectory_review" if run_dir else ROOT / "outputs" / "dense_trajectory_review")
    if qa_manifest is None:
        raise RuntimeError("Provide --run-dir or --qa-manifest.")
    cmd = [
        sys.executable,
        "build_dense_trajectory_review_batch.py",
        "--qa-manifest",
        str(qa_manifest),
        "--out-dir",
        str(out_dir),
        "--max-clips",
        str(args.max_clips),
        "--per-video",
        str(args.per_video),
        "--seconds-before",
        str(args.seconds_before),
        "--seconds-after",
        str(args.seconds_after),
        "--frame-stride",
        str(args.frame_stride),
        "--process-width",
        str(args.process_width),
        "--process-height",
        str(args.process_height),
        "--default-radius",
        str(args.default_radius),
        "--default-clip-center-sec",
        str(args.default_clip_center_sec),
        "--contact-sheet-cols",
        str(args.contact_sheet_cols),
        "--contact-sheet-every",
        str(args.contact_sheet_every),
        "--max-contact-sheet-frames",
        str(args.max_contact_sheet_frames),
    ]
    for track_metric in args.track_metrics:
        cmd.extend(["--track-metrics", str(track_metric)])
    for batch_summary in args.batch_summary:
        cmd.extend(["--batch-summary", str(batch_summary)])
    for target_video in args.target_video:
        cmd.extend(["--target-video", str(target_video)])
    for track_root in args.track_root:
        cmd.extend(["--track-root", str(track_root)])
    if dataset is not None and dataset.exists():
        cmd.extend(["--dataset", str(dataset)])
    if args.video_root:
        cmd.extend(["--video-root", str(args.video_root)])
    if args.min_time_sec is not None:
        cmd.extend(["--min-time-sec", str(args.min_time_sec)])
    if args.max_time_sec is not None:
        cmd.extend(["--max-time-sec", str(args.max_time_sec)])
    if args.dry_run:
        cmd.append("--dry-run")
    run_step("dense trajectory review batch", cmd, cwd=ROOT, dry_run=args.dry_run)


def evaluate_dense_trajectory(args: argparse.Namespace) -> None:
    out_dir = args.out_dir or ROOT / "outputs" / "dense_trajectory_evaluation"
    cmd = [
        sys.executable,
        "evaluate_dense_trajectory.py",
        "--labels-jsonl",
        str(args.labels_jsonl),
        "--out-dir",
        str(out_dir),
        "--tolerance-px",
        str(args.tolerance_px),
        "--radius-multiplier",
        str(args.radius_multiplier),
        "--confidence-threshold",
        str(args.confidence_threshold),
        "--max-time-delta-sec",
        str(args.max_time_delta_sec),
    ]
    if args.predictions_jsonl:
        cmd.extend(["--predictions-jsonl", str(args.predictions_jsonl)])
    if args.tracks_root:
        cmd.extend(["--tracks-root", str(args.tracks_root)])
    if args.dry_run:
        cmd.append("--dry-run")
    run_step("dense trajectory evaluation", cmd, cwd=ROOT, dry_run=args.dry_run)


def train_detector(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else None
    dataset = args.dataset or (run_dir / "detector_dataset" if run_dir else None)
    out_dir = args.out_dir or (run_dir / "detector_models" if run_dir else ROOT / "models" / "footbag_detector")
    if dataset is None:
        raise RuntimeError("Provide --run-dir or --dataset.")
    cmd = [
        sys.executable,
        "train_footbag_detector.py",
        "--dataset",
        str(dataset),
        "--out-dir",
        str(out_dir),
        "--base-model",
        str(args.base_model),
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--run-name",
        str(args.run_name),
    ]
    if args.device:
        cmd.extend(["--device", str(args.device)])
    if args.recover_existing:
        cmd.append("--recover-existing")
    if args.dry_run:
        cmd.append("--dry-run")
    run_step("footbag detector training", cmd, cwd=ROOT, dry_run=args.dry_run)


def train_patch_detector(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else None
    dataset = args.dataset or (run_dir / "detector_dataset" if run_dir else None)
    out_dir = args.out_dir or (run_dir / "patch_detector" if run_dir else ROOT / "models" / "patch_detector")
    if dataset is None:
        raise RuntimeError("Provide --run-dir or --dataset.")
    cmd = [
        sys.executable,
        "patch_footbag_detector.py",
        "train",
        "--dataset",
        str(dataset),
        "--out-dir",
        str(out_dir),
        "--crop-size",
        str(args.crop_size),
        "--jitter",
        str(args.jitter),
        "--negatives-per-image",
        str(args.negatives_per_image),
        "--seed",
        str(args.seed),
        "--model-kind",
        str(args.model_kind),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    run_step("patch footbag detector training", cmd, cwd=ROOT, dry_run=args.dry_run)


def evaluate_patch_detector(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else None
    dataset = args.dataset or (run_dir / "detector_dataset" if run_dir else None)
    model = args.model or (run_dir / "patch_detector" / "patch_footbag_detector.joblib" if run_dir else None)
    out_dir = args.out_dir or (run_dir / "patch_detector_evaluation" if run_dir else ROOT / "outputs" / "patch_detector_evaluation")
    if dataset is None:
        raise RuntimeError("Provide --run-dir or --dataset.")
    if model is None:
        raise RuntimeError("Provide --run-dir or --model.")
    cmd = [
        sys.executable,
        "patch_footbag_detector.py",
        "evaluate",
        "--dataset",
        str(dataset),
        "--model",
        str(model),
        "--out-dir",
        str(out_dir),
        "--tolerance-px",
        str(args.tolerance_px),
        "--box-tolerance-multiplier",
        str(args.box_tolerance_multiplier),
        "--max-candidates",
        str(args.max_candidates),
        "--max-detections",
        str(args.max_detections),
    ]
    if args.threshold is not None:
        cmd.extend(["--threshold", str(args.threshold)])
    run_step("patch footbag detector evaluation", cmd, cwd=ROOT, dry_run=args.dry_run)


def resolve_video_input(raw: str | Path, *, video_root: Path | None = None, run_dir: Path | None = None) -> Path:
    path = Path(raw).expanduser()
    candidates = [path]
    if not path.is_absolute():
        if video_root is not None:
            candidates.append(video_root.expanduser() / path)
            candidates.append(video_root.expanduser() / path.name)
        if run_dir is not None:
            candidates.append(run_dir / path)
            candidates.append(run_dir / path.name)
        candidates.append(ROOT / path)
    elif video_root is not None:
        candidates.append(video_root.expanduser() / path.name)
    downloads_candidate = Path.home() / "Downloads" / path.name
    candidates.append(downloads_candidate)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return path.resolve() if path.is_absolute() else path


def qa_manifest_for_run(run_dir: Path | None, explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()
    if run_dir is None:
        return None
    run_manifest = read_json(run_dir / "run_manifest.json") if (run_dir / "run_manifest.json").exists() else {}
    artifacts = run_manifest.get("artifacts", {}) if isinstance(run_manifest.get("artifacts"), dict) else {}
    return resolve_run_manifest_path(run_dir, artifacts.get("qa_manifest")) or run_dir / "qa" / "qa_manifest.json"


def detector_batch_videos(
    *,
    explicit_videos: list[Path],
    qa_manifest: Path | None,
    video_root: Path | None,
    run_dir: Path | None,
) -> list[Path]:
    if explicit_videos:
        return [resolve_video_input(video, video_root=video_root, run_dir=run_dir) for video in explicit_videos]
    if qa_manifest is None or not qa_manifest.exists():
        return []
    manifest = read_json(qa_manifest)
    videos: list[Path] = []
    seen: set[str] = set()
    for run in manifest.get("runs", []):
        raw = run.get("video")
        if not raw:
            continue
        resolved = resolve_video_input(str(raw), video_root=video_root, run_dir=run_dir)
        key = str(resolved)
        if key not in seen:
            videos.append(resolved)
            seen.add(key)
    return videos


def detect_footbag(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else None
    video = args.video.expanduser().resolve() if args.video else None
    model = args.model or (run_dir / "detector_models" / "footbag_detector_best.pt" if run_dir else None)
    patch_model = args.patch_model or (run_dir / "patch_detector" / "patch_footbag_detector.joblib" if run_dir else None)
    if args.model is None and model is not None and not model.exists():
        model = None
    if args.patch_model is None and patch_model is not None and not patch_model.exists():
        patch_model = None
    if args.detections_jsonl is None and video is None:
        raise RuntimeError("Provide --video unless --detections-jsonl is used.")
    if args.detections_jsonl is None and model is None and patch_model is None:
        raise RuntimeError("Provide --model, --patch-model, --run-dir with a detector artifact, or --detections-jsonl.")
    if args.out_dir:
        out_dir = args.out_dir
    elif run_dir and video:
        out_dir = run_dir / "detector_inference" / video.stem
    elif run_dir:
        out_dir = run_dir / "detector_inference"
    else:
        out_dir = ROOT / "outputs" / "detector_inference"
    cmd = [
        sys.executable,
        "footbag_detector_inference.py",
        "--out-dir",
        str(out_dir),
        "--confidence-threshold",
        str(args.confidence_threshold),
        "--smoothing-alpha",
        str(args.smoothing_alpha),
        "--max-gap-frames",
        str(args.max_gap_frames),
        "--max-jump-px",
        str(args.max_jump_px),
        "--tracker-mode",
        str(args.tracker_mode),
        "--every-nth-frame",
        str(args.every_nth_frame),
        "--imgsz",
        str(args.imgsz),
        "--min-box-size",
        str(args.min_box_size),
        "--max-box-side-frac",
        str(args.max_box_side_frac),
        "--max-aspect-ratio",
        str(args.max_aspect_ratio),
    ]
    if video:
        cmd.extend(["--video", str(video)])
    if args.calibration_metrics:
        cmd.extend(["--calibration-metrics", str(args.calibration_metrics)])
    if model:
        cmd.extend(["--model", str(model)])
    if patch_model and (args.patch_model is not None or model is None):
        cmd.extend(["--patch-model", str(patch_model)])
    if args.detections_jsonl:
        cmd.extend(["--detections-jsonl", str(args.detections_jsonl)])
    if args.max_frames is not None:
        cmd.extend(["--max-frames", str(args.max_frames)])
    if args.device:
        cmd.extend(["--device", str(args.device)])
    if args.fps is not None:
        cmd.extend(["--fps", str(args.fps)])
    if args.process_width is not None:
        cmd.extend(["--process-width", str(args.process_width)])
    if args.process_height is not None:
        cmd.extend(["--process-height", str(args.process_height)])
    if args.patch_threshold is not None:
        cmd.extend(["--patch-threshold", str(args.patch_threshold)])
    if args.patch_max_candidates is not None:
        cmd.extend(["--patch-max-candidates", str(args.patch_max_candidates)])
    if args.patch_max_detections is not None:
        cmd.extend(["--patch-max-detections", str(args.patch_max_detections)])
    if args.dry_run:
        cmd.append("--dry-run")
    run_step("footbag detector inference", cmd, cwd=ROOT, dry_run=args.dry_run)


def detect_footbag_batch(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else None
    qa_manifest = qa_manifest_for_run(run_dir, args.qa_manifest)
    video_root = args.video_root.expanduser().resolve() if args.video_root else None
    videos = detector_batch_videos(
        explicit_videos=[video.expanduser() for video in args.videos],
        qa_manifest=qa_manifest,
        video_root=video_root,
        run_dir=run_dir,
    )
    if args.max_videos is not None:
        videos = videos[: args.max_videos]
    if not videos:
        raise RuntimeError("No videos found. Provide videos, --qa-manifest, or --run-dir with a QA manifest.")

    out_root = args.out_root or (run_dir / "detector_inference" if run_dir else ROOT / "outputs" / "detector_inference")
    detections_root = args.detections_root.expanduser().resolve() if args.detections_root else None
    model = args.model or (run_dir / "detector_models" / "footbag_detector_best.pt" if run_dir else None)
    patch_model = args.patch_model or (run_dir / "patch_detector" / "patch_footbag_detector.joblib" if run_dir else None)
    if args.patch_model is None and patch_model is not None and not patch_model.exists():
        patch_model = None
    if detections_root is None and model is None and patch_model is None:
        raise RuntimeError("Provide --model, --patch-model, --run-dir with a detector artifact, or --detections-root.")
    if detections_root is None and not args.dry_run and model is not None and not model.exists():
        raise RuntimeError(f"Detector model not found: {model}. Train one with `hackytrack.py train-detector` first.")
    if detections_root is None and not args.dry_run and patch_model is not None and not patch_model.exists():
        raise RuntimeError(f"Patch detector model not found: {patch_model}. Train one with `hackytrack.py train-patch-detector` first.")
    missing_videos = [video for video in videos if not video.exists()]
    if missing_videos and detections_root is None:
        sample = ", ".join(str(video) for video in missing_videos[:3])
        raise RuntimeError(f"Missing source videos for model inference: {sample}. Pass --video-root or explicit video paths.")

    runs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, video in enumerate(videos, start=1):
        out_dir = out_root / video.stem
        cmd = [
            sys.executable,
            "footbag_detector_inference.py",
            "--out-dir",
            str(out_dir),
            "--confidence-threshold",
            str(args.confidence_threshold),
            "--smoothing-alpha",
            str(args.smoothing_alpha),
            "--max-gap-frames",
            str(args.max_gap_frames),
            "--max-jump-px",
            str(args.max_jump_px),
            "--tracker-mode",
            str(args.tracker_mode),
            "--every-nth-frame",
            str(args.every_nth_frame),
            "--imgsz",
            str(args.imgsz),
            "--min-box-size",
            str(args.min_box_size),
            "--max-box-side-frac",
            str(args.max_box_side_frac),
            "--max-aspect-ratio",
            str(args.max_aspect_ratio),
        ]
        if video.exists():
            cmd.extend(["--video", str(video)])
        elif detections_root is not None:
            cmd.extend(["--video", str(video)])
        if args.calibration_metrics:
            cmd.extend(["--calibration-metrics", str(args.calibration_metrics)])
        if model is not None:
            cmd.extend(["--model", str(model)])
        if patch_model is not None and (args.patch_model is not None or model is None):
            cmd.extend(["--patch-model", str(patch_model)])
        if detections_root is not None:
            detections_jsonl = detections_root / f"{video.stem}.jsonl"
            if not args.dry_run and not detections_jsonl.exists():
                failure = {
                    "video": portable_path_ref(video),
                    "out_dir": portable_path_ref(out_dir, out_root.parent),
                    "error": f"missing detections JSONL: {portable_path_ref(detections_jsonl)}",
                }
                failures.append(failure)
                runs.append({**failure, "status": "failed"})
                if not args.continue_on_error:
                    raise RuntimeError(failure["error"])
                continue
            cmd.extend(["--detections-jsonl", str(detections_jsonl)])
        if args.max_frames is not None:
            cmd.extend(["--max-frames", str(args.max_frames)])
        if args.device:
            cmd.extend(["--device", str(args.device)])
        if args.fps is not None:
            cmd.extend(["--fps", str(args.fps)])
        if args.process_width is not None:
            cmd.extend(["--process-width", str(args.process_width)])
        if args.process_height is not None:
            cmd.extend(["--process-height", str(args.process_height)])
        if args.patch_threshold is not None:
            cmd.extend(["--patch-threshold", str(args.patch_threshold)])
        if args.patch_max_candidates is not None:
            cmd.extend(["--patch-max-candidates", str(args.patch_max_candidates)])
        if args.patch_max_detections is not None:
            cmd.extend(["--patch-max-detections", str(args.patch_max_detections)])
        if args.dry_run:
            cmd.append("--dry-run")
        try:
            run_step(f"footbag detector inference {index}/{len(videos)}", cmd, cwd=ROOT, dry_run=args.dry_run)
        except subprocess.CalledProcessError as exc:
            failure = {
                "video": portable_path_ref(video),
                "out_dir": portable_path_ref(out_dir, out_root.parent),
                "error": f"command failed with exit code {exc.returncode}",
            }
            failures.append(failure)
            runs.append({**failure, "status": "failed"})
            if not args.continue_on_error:
                raise
            continue
        runs.append(
            {
                "video": portable_path_ref(video),
                "out_dir": portable_path_ref(out_dir, out_root.parent),
                "manifest": portable_path_ref(out_dir / "detector_inference_manifest.json", out_root.parent),
                "track_json": portable_path_ref(out_dir / "detector_track.json", out_root.parent),
                "status": "dry_run" if args.dry_run else "completed",
            }
        )

    summary = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(),
        "command": "detect-footbag-batch",
        "dry_run": args.dry_run,
        "qa_manifest": portable_path_ref(qa_manifest),
        "video_root": portable_path_ref(video_root),
        "model": portable_path_ref(model),
        "calibration_metrics": portable_path_ref(args.calibration_metrics),
        "patch_model": portable_path_ref(patch_model if args.patch_model is not None or model is None else None),
        "detections_root": portable_path_ref(detections_root),
        "out_root": portable_path_ref(out_root, out_root.parent),
        "counts": {
            "videos": len(videos),
            "completed": sum(1 for run in runs if run.get("status") == "completed"),
            "failed": len(failures),
            "dry_run": sum(1 for run in runs if run.get("status") == "dry_run"),
        },
        "runs": runs,
        "failures": failures,
    }
    if not args.dry_run:
        write_json(out_root / "detector_batch_manifest.json", summary)
    print(json.dumps(summary["counts"], indent=2))


def apply_detector_track(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else None
    qa_manifest = args.qa_manifest or (
        resolve_run_manifest_path(run_dir, read_json(run_dir / "run_manifest.json").get("artifacts", {}).get("qa_manifest"))
        if run_dir and (run_dir / "run_manifest.json").exists()
        else None
    )
    tracks_root = args.tracks_root or (run_dir / "detector_inference" if run_dir else None)
    out_root = args.out_root or (run_dir / "qa_model_track" if run_dir else None)
    cmd = [
        sys.executable,
        "apply_detector_track_to_qa.py",
        "--max-time-delta-sec",
        str(args.max_time_delta_sec),
        "--max-center-delta-px",
        str(args.max_center_delta_px),
    ]
    if args.qa_events or args.track_json or args.out_events:
        if not (args.qa_events and args.track_json and args.out_events):
            raise RuntimeError("Single-file mode requires --qa-events, --track-json, and --out-events.")
        cmd.extend(["--qa-events", str(args.qa_events), "--track-json", str(args.track_json), "--out-events", str(args.out_events)])
    else:
        if qa_manifest is None or tracks_root is None or out_root is None:
            raise RuntimeError("Provide --run-dir, --qa-manifest/--tracks-root/--out-root, or single-file arguments.")
        cmd.extend(["--qa-manifest", str(qa_manifest), "--tracks-root", str(tracks_root), "--out-root", str(out_root)])
    if args.promote_model:
        cmd.append("--promote-model")
    if args.allow_predicted:
        cmd.append("--allow-predicted")
    run_step("apply detector track to QA", cmd, cwd=ROOT, dry_run=args.dry_run)


def evaluate_detector(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else None
    dataset = args.dataset or (run_dir / "detector_dataset" if run_dir else None)
    tracks_root = args.tracks_root or (run_dir / "detector_inference" if run_dir else None)
    out_dir = args.out_dir or (run_dir / "detector_evaluation" if run_dir else ROOT / "outputs" / "detector_evaluation")
    if args.labels_jsonl is None and dataset is None:
        raise RuntimeError("Provide --run-dir, --dataset, or --labels-jsonl.")
    if tracks_root is None:
        raise RuntimeError("Provide --run-dir or --tracks-root.")
    cmd = [
        sys.executable,
        "evaluate_detector_tracks.py",
        "--tracks-root",
        str(tracks_root),
        "--out-dir",
        str(out_dir),
        "--max-time-delta-sec",
        str(args.max_time_delta_sec),
        "--tolerance-px",
        str(args.tolerance_px),
        "--radius-multiplier",
        str(args.radius_multiplier),
    ]
    if dataset is not None:
        cmd.extend(["--dataset", str(dataset)])
    if args.labels_jsonl is not None:
        cmd.extend(["--labels-jsonl", str(args.labels_jsonl)])
    if args.hard_negatives_jsonl is not None:
        cmd.extend(["--hard-negatives-jsonl", str(args.hard_negatives_jsonl)])
    run_step("detector track evaluation", cmd, cwd=ROOT, dry_run=args.dry_run)


def summarize_detector_batch(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else None
    tracks_root = args.tracks_root or (run_dir / "detector_inference" if run_dir else None)
    batch_manifest = args.batch_manifest or (tracks_root / "detector_batch_manifest.json" if tracks_root else None)
    out_dir = args.out_dir or (run_dir / "detector_batch_summary" if run_dir else ROOT / "outputs" / "detector_batch_summary")
    if tracks_root is None and batch_manifest is None:
        raise RuntimeError("Provide --run-dir, --tracks-root, or --batch-manifest.")
    cmd = [
        sys.executable,
        "summarize_detector_batch.py",
        "--out-dir",
        str(out_dir),
        "--high-prediction-share",
        str(args.high_prediction_share),
        "--low-model-coverage",
        str(args.low_model_coverage),
    ]
    if batch_manifest is not None and batch_manifest.exists():
        cmd.extend(["--batch-manifest", str(batch_manifest)])
    elif tracks_root is not None:
        cmd.extend(["--tracks-root", str(tracks_root)])
    if args.dry_run:
        cmd.append("--dry-run")
    run_step("detector batch summary", cmd, cwd=ROOT, dry_run=args.dry_run)


def evaluate_detector_model(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else None
    dataset = args.dataset or (run_dir / "detector_dataset" if run_dir else None)
    out_dir = args.out_dir or (run_dir / "detector_model_evaluation" if run_dir else ROOT / "outputs" / "detector_model_evaluation")
    if dataset is None:
        raise RuntimeError("Provide --run-dir or --dataset.")
    if args.model is None and args.predictions_jsonl is None:
        raise RuntimeError("Provide --model or --predictions-jsonl.")
    cmd = [
        sys.executable,
        "evaluate_detector_model.py",
        "--dataset",
        str(dataset),
        "--out-dir",
        str(out_dir),
        "--confidence-threshold",
        str(args.confidence_threshold),
        "--release-confidence-threshold",
        str(args.release_confidence_threshold),
        "--tolerance-px",
        str(args.tolerance_px),
        "--box-tolerance-multiplier",
        str(args.box_tolerance_multiplier),
        "--imgsz",
        str(args.imgsz),
        "--max-det",
        str(args.max_det),
        "--calibration-split",
        str(args.calibration_split),
        "--max-hard-negative-false-positive-rate",
        str(args.max_hard_negative_false_positive_rate),
    ]
    if args.model is not None:
        cmd.extend(["--model", str(args.model)])
    if args.predictions_jsonl is not None:
        cmd.extend(["--predictions-jsonl", str(args.predictions_jsonl)])
    if args.device:
        cmd.extend(["--device", str(args.device)])
    run_step("detector model evaluation", cmd, cwd=ROOT, dry_run=args.dry_run)


def report(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve()
    cmd = [
        sys.executable,
        "release_candidate_report.py",
        "--run-dir",
        str(run_dir),
    ]
    if args.out_dir:
        cmd.extend(["--out-dir", str(args.out_dir)])
    override_args = {
        "--qa-manifest": args.qa_manifest,
        "--review-batch": args.review_batch,
        "--strict-rally-audit": args.strict_rally_audit,
        "--qa-reviewed-manifest": args.qa_reviewed_manifest,
        "--strict-rally-audit-reviewed": args.strict_rally_audit_reviewed,
        "--hud-verify": args.hud_verify,
        "--hud-error": args.hud_error,
    }
    for flag, value in override_args.items():
        if value is not None:
            cmd.extend([flag, str(value)])
    if args.tests_passed:
        cmd.append("--tests-passed")
    if args.test_evidence:
        cmd.extend(["--test-evidence", args.test_evidence])
    run_step("release candidate report", cmd, cwd=ROOT, dry_run=args.dry_run)


def release_touch_paths(corpus_dir: Path) -> dict[str, Path]:
    classifier_dir = corpus_dir / "touch_classifier_v1"
    return {
        "classifier_dir": classifier_dir,
        "metrics": classifier_dir / "touch_classifier_metrics.json",
        "frozen_events": classifier_dir / "touch_classifier_frozen_events.jsonl",
        "oof_events": classifier_dir / "touch_classifier_oof_events.jsonl",
        "detections": corpus_dir / "owlv2_touch_detections_v1/detections.jsonl",
        "inventory": corpus_dir / "touch_corpus_inventory.json",
        "visual_labels": corpus_dir / "visual_touch_labels",
        "dataset": corpus_dir / "touch_training_dataset_v1",
    }


def release_summary_from_artifacts(
    *,
    run_dir: Path,
    corpus_dir: Path,
    command_log: list[dict[str, str]],
    model_hud_dir: Path,
    corrected_hud_dir: Path | None,
    contact_dir: Path,
    touch_overrides: Path | None,
) -> dict[str, Any]:
    paths = release_touch_paths(corpus_dir)
    metrics = read_json(paths["metrics"]) if paths["metrics"].exists() else {}
    cv_event = metrics.get("event_level_leave_one_video_out") or {}
    frozen_event = metrics.get("event_level_frozen_test") or {}
    model_manifest = read_json(model_hud_dir / "release_touch_hud_manifest.json") if (model_hud_dir / "release_touch_hud_manifest.json").exists() else {}
    model_analytics = read_json(model_hud_dir / "analytics_frozen_only/release_rally_analytics.json") if (model_hud_dir / "analytics_frozen_only/release_rally_analytics.json").exists() else {}
    corrected_manifest = (
        read_json(corrected_hud_dir / "release_touch_hud_manifest.json")
        if corrected_hud_dir is not None and (corrected_hud_dir / "release_touch_hud_manifest.json").exists()
        else {}
    )
    corrected_analytics = (
        read_json(corrected_hud_dir / "analytics_frozen_only/release_rally_analytics.json")
        if corrected_hud_dir is not None and (corrected_hud_dir / "analytics_frozen_only/release_rally_analytics.json").exists()
        else {}
    )
    contact_summary = read_json(contact_dir / "release_contact_classifier_summary.json") if (contact_dir / "release_contact_classifier_summary.json").exists() else {}
    summary = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(),
        "scope": "v0.1_touch_hud_release",
        "run_dir": portable_path_ref(run_dir),
        "corpus_dir": portable_path_ref(corpus_dir),
        "commands": command_log,
        "touch_classifier": {
            "status": metrics.get("status"),
            "feature_mode": metrics.get("feature_mode"),
            "cv_gate_passed": bool((metrics.get("event_level_cv_gate") or {}).get("passes")),
            "frozen_gate_passed": bool((metrics.get("event_level_frozen_test_gate") or {}).get("passes")),
            "cv_event": {
                "precision": cv_event.get("precision"),
                "recall": cv_event.get("recall"),
                "f1": cv_event.get("f1"),
                "false_positive": cv_event.get("false_positive"),
                "false_negative": cv_event.get("false_negative"),
            },
            "frozen_event": {
                "precision": frozen_event.get("precision"),
                "recall": frozen_event.get("recall"),
                "f1": frozen_event.get("f1"),
                "false_positive": frozen_event.get("false_positive"),
                "false_negative": frozen_event.get("false_negative"),
            },
        },
        "hud_model_only": {
            "status": model_manifest.get("status"),
            "videos": len(model_manifest.get("videos") or []),
            "preview_sheet": portable_path_ref(Path(model_manifest["preview_sheet"])) if model_manifest.get("preview_sheet") else None,
            "analytics": (model_analytics.get("aggregate") or {}).get("touch_metrics", {}),
            "report": portable_path_ref(model_hud_dir / "release_touch_hud_report.md"),
            "analytics_report": portable_path_ref(model_hud_dir / "analytics_frozen_only/release_rally_analytics.md"),
        },
        "hud_visual_corrected": (
            {
                "status": corrected_manifest.get("status"),
                "videos": len(corrected_manifest.get("videos") or []),
                "touch_overrides": portable_path_ref(touch_overrides) if touch_overrides else None,
                "preview_sheet": portable_path_ref(Path(corrected_manifest["preview_sheet"])) if corrected_manifest.get("preview_sheet") else None,
                "analytics": (corrected_analytics.get("aggregate") or {}).get("touch_metrics", {}),
                "report": portable_path_ref(corrected_hud_dir / "release_touch_hud_report.md") if corrected_hud_dir else None,
                "analytics_report": portable_path_ref(corrected_hud_dir / "analytics_frozen_only/release_rally_analytics.md") if corrected_hud_dir else None,
            }
            if corrected_hud_dir is not None
            else {}
        ),
        "contact_classifier": {
            "status": contact_summary.get("status"),
            "reasons": contact_summary.get("reasons") or [],
            "rows_with_pose_features": contact_summary.get("rows_with_pose_features"),
            "rows_with_contact_labels": contact_summary.get("rows_with_contact_labels"),
            "report": portable_path_ref(contact_dir / "release_contact_classifier_report.md"),
        },
        "limitations": [
            "v0.1 releases generic touch timing and HUD output only.",
            "Side/contact-type/knee classification is not promoted until pose features and reviewed contact labels pass their own gate.",
            "Stall/drop badges are rendered from reviewed labels when available; the OWLv2 touch release path does not infer them yet.",
            "Visual overrides are output corrections, not classifier training labels or gate evidence.",
        ],
    }
    summary["status"] = "release_ready_v0_1" if release_gate_passed(summary) else "blocked"
    return summary


def write_touch_release_report(path: Path, summary: dict[str, Any]) -> None:
    touch = summary["touch_classifier"]
    model = summary["hud_model_only"]
    corrected = summary.get("hud_visual_corrected") or {}
    contact = summary["contact_classifier"]

    def metrics_row(label: str, metrics: dict[str, Any]) -> str:
        return (
            f"| {label} | {fmt_release_metric(metrics.get('precision'))} | "
            f"{fmt_release_metric(metrics.get('recall'))} | {fmt_release_metric(metrics.get('f1'))} | "
            f"{fmt_release_metric(metrics.get('false_positive'))} | {fmt_release_metric(metrics.get('false_negative'))} |"
        )

    lines = [
        "# Touch HUD Release Readiness",
        "",
        f"- Status: `{summary['status']}`",
        f"- Scope: `{summary['scope']}`",
        f"- Created: `{summary['created_at']}`",
        f"- Run dir: `{summary['run_dir']}`",
        "",
        "## Commands",
        "",
    ]
    for command in summary.get("commands") or []:
        lines.append(f"- `{command['name']}`: `{command['command']}`")
    lines.extend(
        [
            "",
            "## Touch Gates",
            "",
            f"- Classifier status: `{touch.get('status')}`",
            f"- Feature mode: `{touch.get('feature_mode')}`",
            f"- Leave-clips-out CV gate: `{touch.get('cv_gate_passed')}`",
            f"- Frozen-test gate: `{touch.get('frozen_gate_passed')}`",
            "",
            "| split | precision | recall | F1 | FP | FN |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            metrics_row("leave-clips-out CV", touch.get("cv_event") or {}),
            metrics_row("frozen test", touch.get("frozen_event") or {}),
            "",
            "## HUD Output",
            "",
            f"- Model-only HUD status: `{model.get('status')}`",
            f"- Model-only videos: `{model.get('videos')}`",
            f"- Model-only preview: `{model.get('preview_sheet')}`",
            f"- Model-only report: `{model.get('report')}`",
            f"- Model-only analytics: `{model.get('analytics_report')}`",
            "",
            "| HUD set | precision | recall | F1 | FP | FN |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            metrics_row("model-only frozen HUD", model.get("analytics") or {}),
        ]
    )
    if corrected:
        lines.extend(
            [
                metrics_row("visual-corrected frozen HUD", corrected.get("analytics") or {}),
                "",
                f"- Visual-corrected HUD status: `{corrected.get('status')}`",
                f"- Visual-corrected videos: `{corrected.get('videos')}`",
                f"- Visual override file: `{corrected.get('touch_overrides')}`",
                f"- Visual-corrected preview: `{corrected.get('preview_sheet')}`",
                f"- Visual-corrected report: `{corrected.get('report')}`",
                f"- Visual-corrected analytics: `{corrected.get('analytics_report')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Rally Intelligence Status",
            "",
            f"- Contact classifier status: `{contact.get('status')}`",
            f"- Rows with pose features: `{contact.get('rows_with_pose_features')}`",
            f"- Rows with reviewed contact labels: `{contact.get('rows_with_contact_labels')}`",
            f"- Contact report: `{contact.get('report')}`",
            "",
            "Contact/type blockers:",
            "",
        ]
    )
    for reason in contact.get("reasons") or ["none"]:
        lines.append(f"- {reason}")
    lines.extend(["", "## Limitations", ""])
    for limitation in summary.get("limitations") or []:
        lines.append(f"- {limitation}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_video_id_args(cmd: list[str], video_ids: list[str]) -> None:
    for video_id in video_ids:
        cmd.extend(["--video-id", video_id])


def touch_release(args: argparse.Namespace) -> None:
    corpus_dir = args.corpus_dir.resolve()
    paths = release_touch_paths(corpus_dir)
    detections = (args.detections_jsonl or paths["detections"]).resolve()
    out_dir = (args.out_dir or (DEFAULT_RUNS / f"touch-release-{run_id()}")).resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory already exists and is not empty: {out_dir}. Pass --overwrite to reuse it.")
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    command_log: list[dict[str, str]] = []

    def run_release_step(name: str, cmd: list[str]) -> None:
        command_log.append({"name": name, "command": display_command(cmd)})
        run_step(name, cmd, cwd=ROOT, dry_run=args.dry_run)

    if not args.skip_pipeline:
        pipeline_cmd = [
            python,
            "run_touch_pipeline.py",
            "--corpus-dir",
            str(corpus_dir),
            "--detections-jsonl",
            str(detections),
            "--attach-audio-features",
        ]
        if args.import_existing_labels:
            pipeline_cmd.append("--import-existing-labels")
        run_release_step("touch pipeline", pipeline_cmd)

    frozen_video_ids = ordered_jsonl_video_ids(paths["frozen_events"])
    selected_video_ids = args.video_id or []
    analytics_video_ids = selected_video_ids or frozen_video_ids

    model_hud_dir = out_dir / "hud_model_only"
    base_render_cmd = [
        python,
        "render_touch_release_hud.py",
        "--frozen-events",
        str(paths["frozen_events"]),
        "--oof-events",
        str(paths["oof_events"]),
        "--detections-jsonl",
        str(detections),
        "--inventory",
        str(paths["inventory"]),
        "--visual-labels-dir",
        str(paths["visual_labels"]),
        "--out-dir",
        str(model_hud_dir),
        "--extra-non-frozen",
        str(args.extra_non_frozen),
    ]
    append_video_id_args(base_render_cmd, selected_video_ids)
    if args.max_seconds is not None:
        base_render_cmd.extend(["--max-seconds", str(args.max_seconds)])
    if args.allow_missing_audio:
        base_render_cmd.append("--allow-missing-audio")
    if args.allow_missing_centers:
        base_render_cmd.append("--allow-missing-centers")
    run_release_step("model-only release HUD", base_render_cmd)

    model_analytics_cmd = [
        python,
        "release_rally_analytics.py",
        "--hud-dir",
        str(model_hud_dir),
        "--out-dir",
        str(model_hud_dir / "analytics_frozen_only"),
    ]
    append_video_id_args(model_analytics_cmd, analytics_video_ids)
    run_release_step("model-only frozen analytics", model_analytics_cmd)

    corrected_hud_dir: Path | None = None
    touch_overrides = args.touch_overrides
    if touch_overrides is not None and touch_overrides.exists():
        corrected_hud_dir = out_dir / "hud_visual_corrected"
        corrected_render_cmd = [*base_render_cmd]
        corrected_render_cmd[corrected_render_cmd.index(str(model_hud_dir))] = str(corrected_hud_dir)
        corrected_render_cmd.extend(["--touch-overrides", str(touch_overrides)])
        run_release_step("visual-corrected release HUD", corrected_render_cmd)
        corrected_analytics_cmd = [
            python,
            "release_rally_analytics.py",
            "--hud-dir",
            str(corrected_hud_dir),
            "--out-dir",
            str(corrected_hud_dir / "analytics_frozen_only"),
        ]
        append_video_id_args(corrected_analytics_cmd, analytics_video_ids)
        run_release_step("visual-corrected frozen analytics", corrected_analytics_cmd)
    elif touch_overrides is not None:
        print(f"warning: touch override file not found, skipping visual-corrected HUD: {touch_overrides}", file=sys.stderr)

    contact_dir = out_dir / "contact_classifier"
    contact_cmd = [
        python,
        "train_release_contact_classifier.py",
        "--dataset-dir",
        str(paths["dataset"]),
        "--labels-dir",
        str(paths["visual_labels"]),
        "--out-dir",
        str(contact_dir),
    ]
    run_release_step("contact classifier readiness", contact_cmd)

    if args.dry_run:
        return

    summary = release_summary_from_artifacts(
        run_dir=out_dir,
        corpus_dir=corpus_dir,
        command_log=command_log,
        model_hud_dir=model_hud_dir,
        corrected_hud_dir=corrected_hud_dir,
        contact_dir=contact_dir,
        touch_overrides=touch_overrides if touch_overrides and touch_overrides.exists() else None,
    )
    write_json(out_dir / "touch_release_readiness.json", summary)
    write_touch_release_report(out_dir / "touch_release_readiness.md", summary)
    print(f"status: {summary['status']}")
    print(f"report: {out_dir / 'touch_release_readiness.md'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hacky Track release CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    process_parser = sub.add_parser("process", help="Process videos into a versioned run directory")
    process_parser.add_argument("videos", nargs="*", type=Path)
    process_parser.add_argument("--out-root", type=Path, default=DEFAULT_RUNS)
    process_parser.add_argument("--run-name")
    process_parser.add_argument("--overwrite", action="store_true")
    process_parser.add_argument("--reviews-dir", type=Path, default=None, help="Defaults to <run>/reviews")
    process_parser.add_argument("--review-batch-size", type=int, default=96)
    process_parser.add_argument("--ball-audit-max-events", type=int, default=240)
    process_parser.add_argument("--skip-training", action="store_true")
    process_parser.add_argument("--skip-qa", action="store_true")
    process_parser.add_argument("--skip-strict-audit", action="store_true")
    process_parser.add_argument("--skip-review-batch", action="store_true")
    process_parser.add_argument("--skip-validation", action="store_true")
    process_parser.add_argument("--skip-release-evaluation", action="store_true")
    process_parser.add_argument("--skip-hud", action="store_true")
    process_parser.add_argument("--allow-incomplete-hud", action="store_true", help="Render the best available rally if no strict-complete rally exists")
    process_parser.add_argument("--require-hud", action="store_true", help="Fail the command if HUD rendering fails")
    process_parser.add_argument("--dry-run", action="store_true")
    process_parser.set_defaults(func=process)

    review_parser = sub.add_parser("review", help="Open the review app for a run directory")
    review_parser.add_argument("--run-dir", type=Path, required=True)
    review_parser.add_argument("--host", default="127.0.0.1")
    review_parser.add_argument("--port", type=int, default=8765)
    review_parser.add_argument("--qa-manifest", type=Path)
    review_parser.add_argument("--review-batch", type=Path)
    review_parser.add_argument("--assisted-review", type=Path)
    review_parser.add_argument("--reviews-dir", type=Path)
    review_parser.add_argument("--dry-run", action="store_true")
    review_parser.set_defaults(func=review)

    evidence_parser = sub.add_parser("review-evidence", help="Build grouped visual evidence sheets for review")
    evidence_parser.add_argument("--run-dir", type=Path, required=True)
    evidence_parser.add_argument("--batch", type=Path)
    evidence_parser.add_argument("--reviews-dir", type=Path)
    evidence_parser.add_argument("--out-dir", type=Path)
    evidence_parser.add_argument("--audit-results", type=Path)
    evidence_parser.add_argument("--suggestions", type=Path)
    evidence_parser.add_argument("--suggestions-dir", type=Path)
    evidence_parser.add_argument("--skip-suggestions", action="store_true")
    evidence_parser.add_argument("--max-items-per-bucket", type=int, default=36)
    evidence_parser.add_argument("--seconds", type=float, default=0.56)
    evidence_parser.add_argument("--cols", type=int, default=1)
    evidence_parser.add_argument("--dry-run", action="store_true")
    evidence_parser.set_defaults(func=review_evidence)

    seed_parser = sub.add_parser("seed-reviews", help="Seed review-batch items as pending records in a run reviews directory")
    seed_parser.add_argument("--run-dir", type=Path, required=True)
    seed_parser.add_argument("--qa-manifest", type=Path)
    seed_parser.add_argument("--batch", type=Path)
    seed_parser.add_argument("--assisted-review", type=Path)
    seed_parser.add_argument("--reviews-dir", type=Path)
    seed_parser.add_argument("--out", type=Path)
    seed_parser.add_argument("--dry-run", action="store_true")
    seed_parser.set_defaults(func=seed_reviews)

    decisions_parser = sub.add_parser("apply-decisions", help="Apply traceable review decisions to a run reviews directory")
    decisions_parser.add_argument("--run-dir", type=Path, required=True)
    decisions_parser.add_argument("--decisions", type=Path)
    decisions_parser.add_argument("--reviews-dir", type=Path)
    decisions_parser.add_argument("--out", type=Path)
    decisions_parser.add_argument("--dry-run", action="store_true")
    decisions_parser.set_defaults(func=apply_review_decisions)

    verify_parser = sub.add_parser("verify", help="Verify a rendered HUD video")
    verify_parser.add_argument("--run-dir", type=Path, required=True)
    verify_parser.add_argument("--hud-video", type=Path)
    verify_parser.set_defaults(func=verify)

    evaluate_parser = sub.add_parser("evaluate", help="Recompute release metrics from reviewed labels")
    evaluate_parser.add_argument("--run-dir", type=Path)
    evaluate_parser.add_argument("--reviews-dir", type=Path)
    evaluate_parser.add_argument("--batch", type=Path)
    evaluate_parser.add_argument("--training-manifest", type=Path)
    evaluate_parser.add_argument("--model-dir", type=Path)
    evaluate_parser.add_argument("--ball-audit", type=Path)
    evaluate_parser.add_argument("--out-dir", type=Path)
    evaluate_parser.add_argument("--seed", type=int, default=1337)
    evaluate_parser.add_argument("--dry-run", action="store_true")
    evaluate_parser.set_defaults(func=evaluate)

    apply_parser = sub.add_parser("apply-reviews", help="Apply review decisions back into QA events and rerun strict rally audit")
    apply_parser.add_argument("--run-dir", type=Path, required=True)
    apply_parser.add_argument("--qa-manifest", type=Path)
    apply_parser.add_argument("--reviews-dir", type=Path)
    apply_parser.add_argument("--out-root", type=Path)
    apply_parser.add_argument("--strict-audit-out", type=Path)
    apply_parser.add_argument("--dry-run", action="store_true")
    apply_parser.set_defaults(func=apply_reviews)

    detector_parser = sub.add_parser("export-detector-dataset", help="Export reviewed labels for trained footbag detection")
    detector_parser.add_argument("--run-dir", type=Path)
    detector_parser.add_argument("--qa-manifest", type=Path)
    detector_parser.add_argument("--reviews-dir", type=Path)
    detector_parser.add_argument("--out-dir", type=Path)
    detector_parser.add_argument("--seed", type=int, default=1337)
    detector_parser.add_argument("--default-radius", type=float, default=22.0)
    detector_parser.add_argument("--crop-size", type=int, default=160)
    detector_parser.add_argument("--detector-label-review-manifest", type=Path)
    detector_parser.add_argument("--detector-label-decisions", type=Path)
    detector_parser.add_argument("--detector-label-review-pair", action="append", default=[], help="Additional REVIEW_MANIFEST:DECISIONS_JSON pair. May be repeated.")
    detector_parser.add_argument("--dry-run", action="store_true")
    detector_parser.set_defaults(func=export_detector_dataset)

    detector_review_parser = sub.add_parser("detector-label-review", help="Build a review batch for detector ball labels")
    detector_review_parser.add_argument("--run-dir", type=Path)
    detector_review_parser.add_argument("--qa-manifest", type=Path)
    detector_review_parser.add_argument("--reviews-dir", type=Path)
    detector_review_parser.add_argument("--dataset-manifest", type=Path)
    detector_review_parser.add_argument("--video-root", type=Path)
    detector_review_parser.add_argument("--out-dir", type=Path)
    detector_review_parser.add_argument("--max-items", type=int, default=160)
    detector_review_parser.add_argument("--per-video", type=int, default=8)
    detector_review_parser.add_argument("--default-radius", type=float, default=22.0)
    detector_review_parser.add_argument("--crop-size", type=int, default=192)
    detector_review_parser.add_argument("--cols", type=int, default=4)
    detector_review_parser.add_argument("--dry-run", action="store_true")
    detector_review_parser.set_defaults(func=detector_label_review)

    assist_detector_parser = sub.add_parser("assist-detector-labels", help="Create conservative CV-assisted detector-label decisions")
    assist_detector_parser.add_argument("--run-dir", type=Path)
    assist_detector_parser.add_argument("--review-manifest", type=Path)
    assist_detector_parser.add_argument("--out", type=Path)
    assist_detector_parser.add_argument("--min-confidence", type=float, default=0.70)
    assist_detector_parser.add_argument("--dry-run", action="store_true")
    assist_detector_parser.set_defaults(func=assist_detector_labels)

    false_positive_parser = sub.add_parser("detector-false-positive-review", help="Mine model false positives into hard-negative review items")
    false_positive_parser.add_argument("--run-dir", type=Path)
    false_positive_parser.add_argument("--qa-manifest", type=Path)
    false_positive_parser.add_argument("--inference-root", type=Path)
    false_positive_parser.add_argument("--batch-manifest", type=Path)
    false_positive_parser.add_argument("--dataset-manifest", type=Path)
    false_positive_parser.add_argument("--video-root", type=Path)
    false_positive_parser.add_argument("--out-dir", type=Path)
    false_positive_parser.add_argument("--max-items", type=int, default=120)
    false_positive_parser.add_argument("--per-video", type=int, default=24)
    false_positive_parser.add_argument("--crop-size", type=int, default=192)
    false_positive_parser.add_argument("--cols", type=int, default=4)
    false_positive_parser.add_argument("--min-confidence", type=float, default=0.001)
    false_positive_parser.add_argument("--max-confidence", type=float, default=0.08)
    false_positive_parser.add_argument("--dry-run", action="store_true")
    false_positive_parser.set_defaults(func=detector_false_positive_review)

    error_review_parser = sub.add_parser("detector-error-review", help="Mine detector track/model failures into v11 review items")
    error_review_parser.add_argument("--run-dir", type=Path)
    error_review_parser.add_argument("--qa-manifest", type=Path)
    error_review_parser.add_argument("--track-metrics", type=Path)
    error_review_parser.add_argument("--dataset", type=Path)
    error_review_parser.add_argument("--model-metrics", type=Path)
    error_review_parser.add_argument("--video-root", type=Path)
    error_review_parser.add_argument("--out-dir", type=Path)
    error_review_parser.add_argument("--max-items", type=int, default=120)
    error_review_parser.add_argument("--per-video", type=int, default=12)
    error_review_parser.add_argument("--crop-size", type=int, default=224)
    error_review_parser.add_argument("--cols", type=int, default=4)
    error_review_parser.add_argument("--low-confidence-per-split", type=int, default=8)
    error_review_parser.add_argument("--exclude-audit-only", action="store_true")
    error_review_parser.add_argument("--dry-run", action="store_true")
    error_review_parser.set_defaults(func=detector_error_review)

    dense_review_parser = sub.add_parser("dense-trajectory-review", help="Build dense frame-level trajectory review clips from detector failures")
    dense_review_parser.add_argument("--run-dir", type=Path)
    dense_review_parser.add_argument("--qa-manifest", type=Path)
    dense_review_parser.add_argument("--track-metrics", type=Path, action="append", default=[])
    dense_review_parser.add_argument("--batch-summary", type=Path, action="append", default=[])
    dense_review_parser.add_argument("--dataset", type=Path)
    dense_review_parser.add_argument("--video-root", type=Path)
    dense_review_parser.add_argument("--out-dir", type=Path)
    dense_review_parser.add_argument("--target-video", action="append", default=[])
    dense_review_parser.add_argument("--max-clips", type=int, default=12)
    dense_review_parser.add_argument("--per-video", type=int, default=3)
    dense_review_parser.add_argument("--seconds-before", type=float, default=1.0)
    dense_review_parser.add_argument("--seconds-after", type=float, default=1.0)
    dense_review_parser.add_argument("--frame-stride", type=int, default=1)
    dense_review_parser.add_argument("--process-width", type=int, default=688)
    dense_review_parser.add_argument("--process-height", type=int, default=912)
    dense_review_parser.add_argument("--default-radius", type=float, default=10.0)
    dense_review_parser.add_argument("--min-time-sec", type=float)
    dense_review_parser.add_argument("--max-time-sec", type=float)
    dense_review_parser.add_argument("--default-clip-center-sec", type=float, default=5.0)
    dense_review_parser.add_argument("--track-root", action="append", default=[], help="Optional NAME:TRACKS_ROOT overlay source. May be repeated.")
    dense_review_parser.add_argument("--contact-sheet-cols", type=int, default=4)
    dense_review_parser.add_argument("--contact-sheet-every", type=int, default=10)
    dense_review_parser.add_argument("--max-contact-sheet-frames", type=int, default=48)
    dense_review_parser.add_argument("--dry-run", action="store_true")
    dense_review_parser.set_defaults(func=dense_trajectory_review)

    train_detector_parser = sub.add_parser("train-detector", help="Train a custom footbag detector from exported labels")
    train_detector_parser.add_argument("--run-dir", type=Path)
    train_detector_parser.add_argument("--dataset", type=Path)
    train_detector_parser.add_argument("--out-dir", type=Path)
    train_detector_parser.add_argument("--base-model", default="yolo11n.pt")
    train_detector_parser.add_argument("--epochs", type=int, default=80)
    train_detector_parser.add_argument("--imgsz", type=int, default=640)
    train_detector_parser.add_argument("--batch", default="-1")
    train_detector_parser.add_argument("--device")
    train_detector_parser.add_argument("--run-name", default="footbag-detector")
    train_detector_parser.add_argument("--recover-existing", action="store_true", help="Write manifest/model copy from an existing Ultralytics run")
    train_detector_parser.add_argument("--dry-run", action="store_true")
    train_detector_parser.set_defaults(func=train_detector)

    train_patch_parser = sub.add_parser("train-patch-detector", help="Train a reviewed-label patch/objectness footbag detector")
    train_patch_parser.add_argument("--run-dir", type=Path)
    train_patch_parser.add_argument("--dataset", type=Path)
    train_patch_parser.add_argument("--out-dir", type=Path)
    train_patch_parser.add_argument("--crop-size", type=int, default=96)
    train_patch_parser.add_argument("--jitter", type=int, default=4)
    train_patch_parser.add_argument("--negatives-per-image", type=int, default=10)
    train_patch_parser.add_argument("--seed", type=int, default=1337)
    train_patch_parser.add_argument("--model-kind", choices=["logistic", "extra-trees"], default="logistic")
    train_patch_parser.add_argument("--dry-run", action="store_true")
    train_patch_parser.set_defaults(func=train_patch_detector)

    eval_patch_parser = sub.add_parser("evaluate-patch-detector", help="Evaluate patch detector proposals against exported labels")
    eval_patch_parser.add_argument("--run-dir", type=Path)
    eval_patch_parser.add_argument("--dataset", type=Path)
    eval_patch_parser.add_argument("--model", type=Path)
    eval_patch_parser.add_argument("--out-dir", type=Path)
    eval_patch_parser.add_argument("--threshold", type=float)
    eval_patch_parser.add_argument("--tolerance-px", type=float, default=24.0)
    eval_patch_parser.add_argument("--box-tolerance-multiplier", type=float, default=0.75)
    eval_patch_parser.add_argument("--max-candidates", type=int, default=120)
    eval_patch_parser.add_argument("--max-detections", type=int, default=8)
    eval_patch_parser.add_argument("--dry-run", action="store_true")
    eval_patch_parser.set_defaults(func=evaluate_patch_detector)

    detect_parser = sub.add_parser("detect-footbag", help="Run trained footbag detector inference and tracker smoothing")
    detect_parser.add_argument("--run-dir", type=Path)
    detect_parser.add_argument("--video", type=Path)
    detect_parser.add_argument("--model", type=Path)
    detect_parser.add_argument("--patch-model", type=Path)
    detect_parser.add_argument("--detections-jsonl", type=Path)
    detect_parser.add_argument("--out-dir", type=Path)
    detect_parser.add_argument("--confidence-threshold", type=float, default=0.25)
    detect_parser.add_argument("--calibration-metrics", type=Path)
    detect_parser.add_argument("--smoothing-alpha", type=float, default=0.65)
    detect_parser.add_argument("--max-gap-frames", type=int, default=6)
    detect_parser.add_argument("--max-jump-px", type=float, default=150.0)
    detect_parser.add_argument("--tracker-mode", choices=["greedy", "temporal"], default="greedy")
    detect_parser.add_argument("--every-nth-frame", type=int, default=1)
    detect_parser.add_argument("--max-frames", type=int)
    detect_parser.add_argument("--imgsz", type=int, default=640)
    detect_parser.add_argument("--device")
    detect_parser.add_argument("--fps", type=float, help="FPS override for precomputed detections without a source video")
    detect_parser.add_argument("--min-box-size", type=float, default=4.0)
    detect_parser.add_argument("--max-box-side-frac", type=float, default=0.16)
    detect_parser.add_argument("--max-aspect-ratio", type=float, default=3.5)
    detect_parser.add_argument("--process-width", type=int)
    detect_parser.add_argument("--process-height", type=int)
    detect_parser.add_argument("--patch-threshold", type=float)
    detect_parser.add_argument("--patch-max-candidates", type=int, default=120)
    detect_parser.add_argument("--patch-max-detections", type=int, default=8)
    detect_parser.add_argument("--dry-run", action="store_true")
    detect_parser.set_defaults(func=detect_footbag)

    detect_batch_parser = sub.add_parser("detect-footbag-batch", help="Run detector inference for every video in a run or QA manifest")
    detect_batch_parser.add_argument("videos", nargs="*", type=Path)
    detect_batch_parser.add_argument("--run-dir", type=Path)
    detect_batch_parser.add_argument("--qa-manifest", type=Path)
    detect_batch_parser.add_argument("--video-root", type=Path, help="Directory containing source videos named by the QA manifest")
    detect_batch_parser.add_argument("--model", type=Path)
    detect_batch_parser.add_argument("--patch-model", type=Path)
    detect_batch_parser.add_argument("--detections-root", type=Path, help="Directory containing <video-stem>.jsonl precomputed detections")
    detect_batch_parser.add_argument("--out-root", type=Path)
    detect_batch_parser.add_argument("--confidence-threshold", type=float, default=0.25)
    detect_batch_parser.add_argument("--calibration-metrics", type=Path)
    detect_batch_parser.add_argument("--smoothing-alpha", type=float, default=0.65)
    detect_batch_parser.add_argument("--max-gap-frames", type=int, default=6)
    detect_batch_parser.add_argument("--max-jump-px", type=float, default=150.0)
    detect_batch_parser.add_argument("--tracker-mode", choices=["greedy", "temporal"], default="greedy")
    detect_batch_parser.add_argument("--every-nth-frame", type=int, default=1)
    detect_batch_parser.add_argument("--max-frames", type=int)
    detect_batch_parser.add_argument("--max-videos", type=int)
    detect_batch_parser.add_argument("--imgsz", type=int, default=640)
    detect_batch_parser.add_argument("--device")
    detect_batch_parser.add_argument("--fps", type=float, help="FPS override for precomputed detections without a source video")
    detect_batch_parser.add_argument("--min-box-size", type=float, default=4.0)
    detect_batch_parser.add_argument("--max-box-side-frac", type=float, default=0.16)
    detect_batch_parser.add_argument("--max-aspect-ratio", type=float, default=3.5)
    detect_batch_parser.add_argument("--process-width", type=int)
    detect_batch_parser.add_argument("--process-height", type=int)
    detect_batch_parser.add_argument("--patch-threshold", type=float)
    detect_batch_parser.add_argument("--patch-max-candidates", type=int, default=120)
    detect_batch_parser.add_argument("--patch-max-detections", type=int, default=8)
    detect_batch_parser.add_argument("--continue-on-error", action="store_true")
    detect_batch_parser.add_argument("--dry-run", action="store_true")
    detect_batch_parser.set_defaults(func=detect_footbag_batch)

    apply_track_parser = sub.add_parser("apply-detector-track", help="Attach detector-track ball evidence to QA events")
    apply_track_parser.add_argument("--run-dir", type=Path)
    apply_track_parser.add_argument("--qa-manifest", type=Path)
    apply_track_parser.add_argument("--tracks-root", type=Path)
    apply_track_parser.add_argument("--out-root", type=Path)
    apply_track_parser.add_argument("--qa-events", type=Path)
    apply_track_parser.add_argument("--track-json", type=Path)
    apply_track_parser.add_argument("--out-events", type=Path)
    apply_track_parser.add_argument("--max-time-delta-sec", type=float, default=0.08)
    apply_track_parser.add_argument("--max-center-delta-px", type=float, default=36.0)
    apply_track_parser.add_argument("--promote-model", action="store_true")
    apply_track_parser.add_argument("--allow-predicted", action="store_true")
    apply_track_parser.add_argument("--dry-run", action="store_true")
    apply_track_parser.set_defaults(func=apply_detector_track)

    eval_detector_parser = sub.add_parser("evaluate-detector", help="Evaluate detector tracks against reviewed labels")
    eval_detector_parser.add_argument("--run-dir", type=Path)
    eval_detector_parser.add_argument("--dataset", type=Path)
    eval_detector_parser.add_argument("--labels-jsonl", type=Path)
    eval_detector_parser.add_argument("--hard-negatives-jsonl", type=Path)
    eval_detector_parser.add_argument("--tracks-root", type=Path)
    eval_detector_parser.add_argument("--out-dir", type=Path)
    eval_detector_parser.add_argument("--max-time-delta-sec", type=float, default=0.08)
    eval_detector_parser.add_argument("--tolerance-px", type=float, default=24.0)
    eval_detector_parser.add_argument("--radius-multiplier", type=float, default=1.5)
    eval_detector_parser.add_argument("--dry-run", action="store_true")
    eval_detector_parser.set_defaults(func=evaluate_detector)

    summarize_detector_parser = sub.add_parser("summarize-detector-batch", help="Summarize detector batch inference coverage")
    summarize_detector_parser.add_argument("--run-dir", type=Path)
    summarize_detector_parser.add_argument("--tracks-root", type=Path)
    summarize_detector_parser.add_argument("--batch-manifest", type=Path)
    summarize_detector_parser.add_argument("--out-dir", type=Path)
    summarize_detector_parser.add_argument("--high-prediction-share", type=float, default=0.35)
    summarize_detector_parser.add_argument("--low-model-coverage", type=float, default=0.20)
    summarize_detector_parser.add_argument("--dry-run", action="store_true")
    summarize_detector_parser.set_defaults(func=summarize_detector_batch)

    eval_detector_model_parser = sub.add_parser("evaluate-detector-model", help="Evaluate a detector model directly on exported YOLO labels")
    eval_detector_model_parser.add_argument("--run-dir", type=Path)
    eval_detector_model_parser.add_argument("--dataset", type=Path)
    eval_detector_model_parser.add_argument("--model", type=Path)
    eval_detector_model_parser.add_argument("--predictions-jsonl", type=Path)
    eval_detector_model_parser.add_argument("--out-dir", type=Path)
    eval_detector_model_parser.add_argument("--confidence-threshold", type=float, default=0.001)
    eval_detector_model_parser.add_argument("--release-confidence-threshold", type=float, default=0.25)
    eval_detector_model_parser.add_argument("--tolerance-px", type=float, default=24.0)
    eval_detector_model_parser.add_argument("--box-tolerance-multiplier", type=float, default=0.75)
    eval_detector_model_parser.add_argument("--imgsz", type=int, default=640)
    eval_detector_model_parser.add_argument("--max-det", type=int, default=300)
    eval_detector_model_parser.add_argument("--device")
    eval_detector_model_parser.add_argument("--calibration-split", default="validation")
    eval_detector_model_parser.add_argument("--max-hard-negative-false-positive-rate", type=float, default=0.0)
    eval_detector_model_parser.add_argument("--dry-run", action="store_true")
    eval_detector_model_parser.set_defaults(func=evaluate_detector_model)

    dense_eval_parser = sub.add_parser("evaluate-dense-trajectory", help="Evaluate dense frame-level trajectory labels")
    dense_eval_parser.add_argument("--labels-jsonl", type=Path, required=True)
    dense_eval_parser.add_argument("--predictions-jsonl", type=Path)
    dense_eval_parser.add_argument("--tracks-root", type=Path)
    dense_eval_parser.add_argument("--out-dir", type=Path)
    dense_eval_parser.add_argument("--tolerance-px", type=float, default=12.0)
    dense_eval_parser.add_argument("--radius-multiplier", type=float, default=1.5)
    dense_eval_parser.add_argument("--confidence-threshold", type=float, default=0.01)
    dense_eval_parser.add_argument("--max-time-delta-sec", type=float, default=0.04)
    dense_eval_parser.add_argument("--dry-run", action="store_true")
    dense_eval_parser.set_defaults(func=evaluate_dense_trajectory)

    report_parser = sub.add_parser("report", help="Write a release-candidate audit report for a run")
    report_parser.add_argument("--run-dir", type=Path, required=True)
    report_parser.add_argument("--out-dir", type=Path)
    report_parser.add_argument("--qa-manifest", type=Path)
    report_parser.add_argument("--review-batch", type=Path)
    report_parser.add_argument("--strict-rally-audit", type=Path)
    report_parser.add_argument("--qa-reviewed-manifest", type=Path)
    report_parser.add_argument("--strict-rally-audit-reviewed", type=Path)
    report_parser.add_argument("--hud-verify", type=Path)
    report_parser.add_argument("--hud-error", type=Path)
    report_parser.add_argument("--tests-passed", action="store_true")
    report_parser.add_argument("--test-evidence", default="")
    report_parser.add_argument("--dry-run", action="store_true")
    report_parser.set_defaults(func=report)

    touch_release_parser = sub.add_parser("touch-release", help="Run the v0.1 touch/HUD release readiness workflow")
    touch_release_parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_TOUCH_CORPUS)
    touch_release_parser.add_argument("--detections-jsonl", type=Path)
    touch_release_parser.add_argument("--out-dir", type=Path)
    touch_release_parser.add_argument("--touch-overrides", type=Path, default=DEFAULT_TOUCH_OVERRIDES)
    touch_release_parser.add_argument("--video-id", action="append", default=[])
    touch_release_parser.add_argument("--extra-non-frozen", type=int, default=1)
    touch_release_parser.add_argument("--max-seconds", type=float)
    touch_release_parser.add_argument("--skip-pipeline", action="store_true", help="Reuse current classifier artifacts and only render/report")
    touch_release_parser.add_argument("--import-existing-labels", action="store_true")
    touch_release_parser.add_argument("--allow-missing-audio", action="store_true")
    touch_release_parser.add_argument("--allow-missing-centers", action="store_true")
    touch_release_parser.add_argument("--overwrite", action="store_true")
    touch_release_parser.add_argument("--dry-run", action="store_true")
    touch_release_parser.set_defaults(func=touch_release)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
