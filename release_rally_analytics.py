#!/usr/bin/env python3
"""Summarize release HUD rally output and audit misses/fake touches.

This is intentionally an output-layer analysis: it reads already-rendered HUD
event documents, compares predicted touch events to muted visual labels when
available, and reports rally-level stats without retraining or retuning the
classifier.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "runs/release-27-public/touch_corpus_v1"
DEFAULT_HUD_DIR = DEFAULT_CORPUS / "release_touch_hud_v1"
DEFAULT_VISUAL_LABELS = DEFAULT_CORPUS / "visual_touch_labels"
DEFAULT_LEGACY_EVENTS_DIR = ROOT / "data"
DEFAULT_OUT_DIR = DEFAULT_HUD_DIR / "analytics"
DEFAULT_MATCH_TOL_SEC = 0.2


@dataclass(frozen=True)
class TimedEvent:
    time_sec: float
    type: str
    source: str = ""
    label: str = ""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def safe_slug(value: str) -> str:
    return str(value).replace(" ", "-").replace("/", "-").replace("\\", "-").strip("-")


def flatten_events(doc: dict[str, Any], *, event_type: str | None = None) -> list[TimedEvent]:
    events: list[TimedEvent] = []
    for rally in doc.get("rallies", []):
        for event in rally.get("events", []):
            etype = str(event.get("type") or "")
            if event_type is not None and etype != event_type:
                continue
            if event.get("review_status") not in {None, "", "approved", "reviewed"}:
                continue
            if event.get("time_sec") is None:
                continue
            events.append(
                TimedEvent(
                    time_sec=float(event["time_sec"]),
                    type=etype,
                    source=str(event.get("source") or doc.get("annotation_method") or ""),
                    label=str(event.get("label") or ""),
                )
            )
    return sorted(events, key=lambda item: item.time_sec)


def hud_event_doc_paths(hud_dir: Path, video_ids: list[str] | None = None) -> list[Path]:
    wanted = set(video_ids or [])
    paths: list[Path] = []
    for path in sorted(hud_dir.glob("*/release_touch_hud_events.json")):
        video_id = path.parent.name
        if wanted and video_id not in wanted:
            continue
        paths.append(path)
    return paths


def label_paths_for_hud_doc(doc: dict[str, Any], video_id: str, labels_dir: Path, legacy_events_dir: Path) -> list[Path]:
    source_video = str(doc.get("source_video") or "")
    stem = Path(source_video).stem
    candidates = [
        labels_dir / f"{video_id}.events.json",
        labels_dir / f"{safe_slug(video_id)}.events.json",
    ]
    if stem:
        candidates.extend(
            [
                legacy_events_dir / f"{stem}.events.json",
                legacy_events_dir / f"{safe_slug(stem)}.events.json",
            ]
        )
    out: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists():
            out.append(path)
    return out


def label_events_for_video(doc: dict[str, Any], video_id: str, labels_dir: Path, legacy_events_dir: Path) -> tuple[list[TimedEvent], list[Path]]:
    events: list[TimedEvent] = []
    seen: set[tuple[str, int]] = set()
    paths = label_paths_for_hud_doc(doc, video_id, labels_dir, legacy_events_dir)
    for path in paths:
        label_doc = read_json(path)
        for event in flatten_events(label_doc):
            key = (event.type, round(event.time_sec * 20))
            if key in seen:
                continue
            seen.add(key)
            events.append(event)
    return sorted(events, key=lambda item: item.time_sec), paths


def match_times(predicted: list[TimedEvent], truth: list[TimedEvent], *, tol_sec: float) -> tuple[dict[int, int], set[int]]:
    candidates: list[tuple[float, int, int]] = []
    for pred_index, pred in enumerate(predicted):
        for truth_index, item in enumerate(truth):
            delta = abs(pred.time_sec - item.time_sec)
            if delta <= tol_sec:
                candidates.append((delta, pred_index, truth_index))
    matches: dict[int, int] = {}
    used_truth: set[int] = set()
    for _, pred_index, truth_index in sorted(candidates):
        if pred_index in matches or truth_index in used_truth:
            continue
        matches[pred_index] = truth_index
        used_truth.add(truth_index)
    return matches, used_truth


def precision_recall(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = None
    if precision is not None and recall is not None and precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def rally_rows(doc: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rally in doc.get("rallies", []):
        events = sorted(rally.get("events", []), key=lambda item: float(item.get("time_sec") or 0.0))
        touches = [event for event in events if event.get("type") == "touch"]
        stalls = [event for event in events if event.get("type") == "stall"]
        drops = [event for event in events if event.get("type") == "drop_floor"]
        manual_contact_touches = [event for event in touches if event.get("manual_contact_label")]
        touch_times = [float(event["time_sec"]) for event in touches]
        start = float(rally.get("start_sec") or (touch_times[0] if touch_times else 0.0))
        end = float(rally.get("end_sec") or (touch_times[-1] if touch_times else start))
        gaps = [b - a for a, b in zip(touch_times, touch_times[1:])]
        duration = max(0.0, end - start)
        rows.append(
            {
                "rally_id": rally.get("id"),
                "label": rally.get("label"),
                "start_sec": round(start, 6),
                "end_sec": round(end, 6),
                "duration_sec": round(duration, 6),
                "touches": len(touches),
                "manual_contact_labels": len(manual_contact_touches),
                "manual_contact_side_labels": sum(1 for event in manual_contact_touches if event.get("contact_side") not in {None, "", "unknown"}),
                "manual_contact_surface_labels": sum(1 for event in manual_contact_touches if event.get("contact_surface") not in {None, "", "unknown"}),
                "manual_contact_type_labels": sum(1 for event in manual_contact_touches if event.get("contact_type") not in {None, "", "unknown"}),
                "stalls": len(stalls),
                "drops": len(drops),
                "touch_rate_per_sec": None if duration <= 0 else round(len(touches) / duration, 6),
                "longest_gap_sec": None if not gaps else round(max(gaps), 6),
                "first_touch_sec": None if not touch_times else round(touch_times[0], 6),
                "last_touch_sec": None if not touch_times else round(touch_times[-1], 6),
            }
        )
    return rows


def contact_badge_summary(doc: dict[str, Any]) -> dict[str, Any]:
    touches = [event for rally in doc.get("rallies", []) for event in rally.get("events", []) if event.get("type") == "touch"]
    manual = [event for event in touches if event.get("manual_contact_label")]
    return {
        "manual_contact_labels": len(manual),
        "manual_contact_side_labels": sum(1 for event in manual if event.get("contact_side") not in {None, "", "unknown"}),
        "manual_contact_surface_labels": sum(1 for event in manual if event.get("contact_surface") not in {None, "", "unknown"}),
        "manual_contact_type_labels": sum(1 for event in manual if event.get("contact_type") not in {None, "", "unknown"}),
        "manual_contact_label_coverage": len(manual) / len(touches) if touches else None,
    }


def analyze_video(events_path: Path, labels_dir: Path, legacy_events_dir: Path, tol_sec: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    doc = read_json(events_path)
    video_id = events_path.parent.name
    predicted_touches = flatten_events(doc, event_type="touch")
    label_events, label_paths = label_events_for_video(doc, video_id, labels_dir, legacy_events_dir)
    truth_touches = [event for event in label_events if event.type == "touch"]
    matches, used_truth = match_times(predicted_touches, truth_touches, tol_sec=tol_sec)
    false_positive_rows: list[dict[str, Any]] = []
    false_negative_rows: list[dict[str, Any]] = []
    for index, pred in enumerate(predicted_touches):
        if index in matches:
            continue
        nearest = min((abs(pred.time_sec - truth.time_sec) for truth in truth_touches), default=None)
        false_positive_rows.append(
            {
                "video_id": video_id,
                "error_type": "false_positive",
                "time_sec": round(pred.time_sec, 6),
                "nearest_truth_delta_sec": None if nearest is None else round(float(nearest), 6),
                "source_video": doc.get("source_video"),
            }
        )
    for index, truth in enumerate(truth_touches):
        if index in used_truth:
            continue
        nearest = min((abs(pred.time_sec - truth.time_sec) for pred in predicted_touches), default=None)
        false_negative_rows.append(
            {
                "video_id": video_id,
                "error_type": "false_negative",
                "time_sec": round(truth.time_sec, 6),
                "nearest_prediction_delta_sec": None if nearest is None else round(float(nearest), 6),
                "source_video": doc.get("source_video"),
            }
        )
    rallies = rally_rows(doc)
    contact_badges = contact_badge_summary(doc)
    best_rally = max(rallies, key=lambda row: (int(row["touches"]), float(row["duration_sec"] or 0.0)), default=None)
    metrics = precision_recall(len(matches), len(false_positive_rows), len(false_negative_rows))
    summary = {
        "video_id": video_id,
        "source_video": doc.get("source_video"),
        "events_path": str(events_path),
        "label_paths": [str(path) for path in label_paths],
        "match_tol_sec": tol_sec,
        "predicted_touches": len(predicted_touches),
        "truth_touches": len(truth_touches),
        "truth_stalls": sum(1 for event in label_events if event.type == "stall"),
        "truth_drop_floor": sum(1 for event in label_events if event.type == "drop_floor"),
        **contact_badges,
        "rallies": len(rallies),
        "best_rally": best_rally,
        "rally_rows": rallies,
        "touch_metrics": metrics,
        "false_positive_times_sec": [row["time_sec"] for row in false_positive_rows],
        "false_negative_times_sec": [row["time_sec"] for row in false_negative_rows],
    }
    return summary, false_positive_rows + false_negative_rows


def aggregate_video_summaries(videos: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int((video.get("touch_metrics") or {}).get("true_positive") or 0) for video in videos)
    fp = sum(int((video.get("touch_metrics") or {}).get("false_positive") or 0) for video in videos)
    fn = sum(int((video.get("touch_metrics") or {}).get("false_negative") or 0) for video in videos)
    return {
        "videos": len(videos),
        "rallies": sum(int(video.get("rallies") or 0) for video in videos),
        "predicted_touches": sum(int(video.get("predicted_touches") or 0) for video in videos),
        "truth_touches": sum(int(video.get("truth_touches") or 0) for video in videos),
        "truth_stalls": sum(int(video.get("truth_stalls") or 0) for video in videos),
        "truth_drop_floor": sum(int(video.get("truth_drop_floor") or 0) for video in videos),
        "manual_contact_labels": sum(int(video.get("manual_contact_labels") or 0) for video in videos),
        "manual_contact_side_labels": sum(int(video.get("manual_contact_side_labels") or 0) for video in videos),
        "manual_contact_surface_labels": sum(int(video.get("manual_contact_surface_labels") or 0) for video in videos),
        "manual_contact_type_labels": sum(int(video.get("manual_contact_type_labels") or 0) for video in videos),
        "touch_metrics": precision_recall(tp, fp, fn),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Release Rally Analytics",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Created: `{manifest['created_at']}`",
        f"- Match tolerance: `{manifest['match_tol_sec']}` sec",
        "",
        "## Aggregate",
        "",
    ]
    agg = manifest["aggregate"]
    m = agg["touch_metrics"]
    lines.extend(
        [
            f"- Videos: `{agg['videos']}`",
            f"- Rallies: `{agg['rallies']}`",
            f"- Touch precision/recall/F1: `{fmt(m['precision'])}` / `{fmt(m['recall'])}` / `{fmt(m['f1'])}`",
            f"- FP/FN: `{m['false_positive']}` / `{m['false_negative']}`",
            f"- Reviewed stalls/drop_floor available: `{agg['truth_stalls']}` / `{agg['truth_drop_floor']}`",
            f"- Manual reviewed contact badges: `{agg['manual_contact_labels']}` "
            f"(side `{agg['manual_contact_side_labels']}`, surface `{agg['manual_contact_surface_labels']}`, type `{agg['manual_contact_type_labels']}`)",
            "",
            "## Per Video",
            "",
            "| video | split/source | pred | truth | P | R | FP | FN | manual badges | side/surface | rallies | best rally |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
        ]
    )
    for video in manifest["videos"]:
        m = video["touch_metrics"]
        best = video.get("best_rally") or {}
        best_text = "-" if not best else f"R{best.get('rally_id')} {best.get('touches')} touches"
        lines.append(
            f"| `{video['video_id']}` | `{video.get('source_video')}` | {video['predicted_touches']} | {video['truth_touches']} | "
            f"{fmt(m['precision'])} | {fmt(m['recall'])} | {m['false_positive']} | {m['false_negative']} | "
            f"{video.get('manual_contact_labels', 0)} | {video.get('manual_contact_side_labels', 0)}/{video.get('manual_contact_surface_labels', 0)} | "
            f"{video['rallies']} | {best_text} |"
        )
    lines.extend(["", "## Error Times", ""])
    for video in manifest["videos"]:
        if not video["false_positive_times_sec"] and not video["false_negative_times_sec"]:
            continue
        lines.append(f"### {video['video_id']}")
        lines.append("")
        lines.append(f"- False positives: `{video['false_positive_times_sec']}`")
        lines.append(f"- Missed touches: `{video['false_negative_times_sec']}`")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_analytics(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir.resolve()
    videos: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for events_path in hud_event_doc_paths(args.hud_dir, args.video_id or None):
        summary, rows = analyze_video(events_path, args.visual_labels_dir, args.legacy_events_dir, args.match_tol_sec)
        videos.append(summary)
        errors.extend(rows)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if videos else "no_hud_event_docs",
        "hud_dir": str(args.hud_dir),
        "visual_labels_dir": str(args.visual_labels_dir),
        "legacy_events_dir": str(args.legacy_events_dir),
        "match_tol_sec": args.match_tol_sec,
        "aggregate": aggregate_video_summaries(videos),
        "videos": videos,
        "errors_jsonl": str(out_dir / "release_rally_event_errors.jsonl"),
        "report": str(out_dir / "release_rally_analytics.md"),
    }
    write_json(out_dir / "release_rally_analytics.json", manifest)
    write_jsonl(out_dir / "release_rally_event_errors.jsonl", errors)
    write_report(out_dir / "release_rally_analytics.md", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize release HUD rallies and audit touch misses/fakes")
    parser.add_argument("--hud-dir", type=Path, default=DEFAULT_HUD_DIR)
    parser.add_argument("--visual-labels-dir", type=Path, default=DEFAULT_VISUAL_LABELS)
    parser.add_argument("--legacy-events-dir", type=Path, default=DEFAULT_LEGACY_EVENTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--match-tol-sec", type=float, default=DEFAULT_MATCH_TOL_SEC)
    parser.add_argument("--video-id", action="append", default=[])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build_analytics(args)
    agg = manifest["aggregate"]["touch_metrics"]
    print(f"status: {manifest['status']}")
    print(f"report: {manifest['report']}")
    print(f"errors: {manifest['errors_jsonl']}")
    print(f"touch P/R/F1: {fmt(agg['precision'])} / {fmt(agg['recall'])} / {fmt(agg['f1'])}")


if __name__ == "__main__":
    main()
