#!/usr/bin/env python3
"""Attach trained-detector ball-track evidence to QA event files."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def portable(path: Path, base: Path = ROOT) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except (OSError, ValueError):
        return path.name if path.is_absolute() else str(path)


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.exists() or path.is_absolute():
        return path
    return ROOT / path


def event_time(event: dict[str, Any]) -> float | None:
    value = event.get("qa_frame_time_sec", event.get("time_sec", event.get("start_sec")))
    return None if value is None else float(value)


def center_distance(event: dict[str, Any], track_point: dict[str, Any]) -> float | None:
    if event.get("qa_ball_x") is None or event.get("qa_ball_y") is None:
        return None
    center = track_point.get("center") or []
    if len(center) < 2:
        return None
    return math.hypot(float(event["qa_ball_x"]) - float(center[0]), float(event["qa_ball_y"]) - float(center[1]))


def track_time(track_point: dict[str, Any]) -> float | None:
    value = track_point.get("time_sec")
    return None if value is None else float(value)


def nearest_track_point(
    track: list[dict[str, Any]],
    time_sec: float,
    *,
    max_time_delta_sec: float,
) -> tuple[dict[str, Any] | None, float | None]:
    timed = [(item, track_time(item)) for item in track]
    timed = [(item, time_value) for item, time_value in timed if time_value is not None]
    if not timed:
        return None, None
    item, nearest_time = min(timed, key=lambda pair: abs(float(pair[1]) - time_sec))
    delta = abs(float(nearest_time) - time_sec)
    if delta > max_time_delta_sec:
        return None, delta
    return item, delta


def model_source_for(track_point: dict[str, Any]) -> str:
    if track_point.get("source") == "track_predicted":
        return "model_track_predicted"
    return "model_detector"


def annotate_event(
    event: dict[str, Any],
    track_point: dict[str, Any] | None,
    *,
    time_delta_sec: float | None,
    max_center_delta_px: float,
    promote_model: bool,
    allow_predicted: bool,
) -> tuple[dict[str, Any], dict[str, int]]:
    out = dict(event)
    counts = {
        "matched": 0,
        "promoted": 0,
        "mismatched": 0,
        "no_track": 0,
        "predicted_matches": 0,
    }
    if track_point is None:
        out["model_ball_agreement"] = "no_model_track"
        out["model_ball_review_reason"] = "no_track_point_near_event_time"
        counts["no_track"] = 1
        return out, counts

    center = track_point.get("center") or [None, None]
    distance_px = center_distance(out, track_point)
    is_predicted = track_point.get("source") == "track_predicted"
    if is_predicted:
        counts["predicted_matches"] = 1
    counts["matched"] = 1
    out.update(
        {
            "model_ball_x": center[0],
            "model_ball_y": center[1],
            "model_ball_confidence": track_point.get("confidence"),
            "model_ball_source": model_source_for(track_point),
            "model_ball_frame_index": track_point.get("frame_index"),
            "model_ball_time_sec": track_point.get("time_sec"),
            "model_ball_time_delta_sec": None if time_delta_sec is None else round(float(time_delta_sec), 6),
            "model_ball_distance_px": None if distance_px is None else round(distance_px, 3),
            "model_ball_uncertainty_reasons": track_point.get("uncertainty_reasons", []),
        }
    )

    if distance_px is None:
        agreement = "no_existing_qa_ball"
    elif distance_px <= max_center_delta_px:
        agreement = "match"
    else:
        agreement = "mismatch"
        counts["mismatched"] = 1
    out["model_ball_agreement"] = agreement
    if agreement == "mismatch":
        out["model_ball_review_reason"] = "model_heuristic_center_mismatch"
    elif is_predicted:
        out["model_ball_review_reason"] = "model_track_prediction"
    else:
        out["model_ball_review_reason"] = None

    can_promote = promote_model and (allow_predicted or not is_predicted) and agreement != "mismatch"
    if can_promote:
        counts["promoted"] = 1
        if "heuristic_ball_evidence" not in out:
            out["heuristic_ball_evidence"] = {
                "qa_ball_x": out.get("qa_ball_x"),
                "qa_ball_y": out.get("qa_ball_y"),
                "qa_ball_radius": out.get("qa_ball_radius"),
                "qa_ball_confidence": out.get("qa_ball_confidence"),
                "qa_ball_source": out.get("qa_ball_source"),
                "qa_frame_time_sec": out.get("qa_frame_time_sec"),
                "qa_ball_accuracy": out.get("qa_ball_accuracy"),
            }
        out["qa_ball_x"] = center[0]
        out["qa_ball_y"] = center[1]
        out["qa_ball_confidence"] = track_point.get("confidence")
        out["qa_ball_source"] = model_source_for(track_point)
        out["qa_frame_time_sec"] = track_point.get("time_sec")
        out["qa_ball_accuracy"] = "model_high" if float(track_point.get("confidence") or 0.0) >= 0.75 else "model_review"
        if out.get("x") is not None and out.get("y") is not None:
            out["qa_ball_correction_px"] = round(math.hypot(float(out["x"]) - float(center[0]), float(out["y"]) - float(center[1])), 3)
    return out, counts


def apply_track_to_doc(
    qa_doc: dict[str, Any],
    track_doc: dict[str, Any],
    *,
    max_time_delta_sec: float = 0.08,
    max_center_delta_px: float = 36.0,
    promote_model: bool = False,
    allow_predicted: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    track = list(track_doc.get("track", []))
    out_doc = dict(qa_doc)
    out_events: list[dict[str, Any]] = []
    totals = {
        "events": 0,
        "matched": 0,
        "promoted": 0,
        "mismatched": 0,
        "no_track": 0,
        "predicted_matches": 0,
    }
    for event in qa_doc.get("events", []):
        totals["events"] += 1
        time_sec = event_time(event)
        if time_sec is None:
            annotated, counts = annotate_event(
                event,
                None,
                time_delta_sec=None,
                max_center_delta_px=max_center_delta_px,
                promote_model=promote_model,
                allow_predicted=allow_predicted,
            )
        else:
            track_point, delta = nearest_track_point(track, time_sec, max_time_delta_sec=max_time_delta_sec)
            annotated, counts = annotate_event(
                event,
                track_point,
                time_delta_sec=delta,
                max_center_delta_px=max_center_delta_px,
                promote_model=promote_model,
                allow_predicted=allow_predicted,
            )
        for key, value in counts.items():
            totals[key] += value
        out_events.append(annotated)

    out_doc["events"] = out_events
    out_doc["detector_track_applied"] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "track_video": track_doc.get("video"),
        "track_model": track_doc.get("model"),
        "max_time_delta_sec": max_time_delta_sec,
        "max_center_delta_px": max_center_delta_px,
        "promote_model": promote_model,
        "allow_predicted": allow_predicted,
        "counts": totals,
    }
    return out_doc, totals


def track_path_for_doc(doc: dict[str, Any], qa_path: Path, tracks_root: Path) -> Path:
    source_video = doc.get("source_video")
    if source_video:
        stem = Path(str(source_video)).stem
    else:
        stem = qa_path.parent.name
    return tracks_root / stem / "detector_track.json"


def apply_one(
    *,
    qa_events: Path,
    track_json: Path,
    out_events: Path,
    max_time_delta_sec: float,
    max_center_delta_px: float,
    promote_model: bool,
    allow_predicted: bool,
) -> dict[str, Any]:
    qa_doc = read_json(qa_events)
    track_doc = read_json(track_json)
    out_doc, counts = apply_track_to_doc(
        qa_doc,
        track_doc,
        max_time_delta_sec=max_time_delta_sec,
        max_center_delta_px=max_center_delta_px,
        promote_model=promote_model,
        allow_predicted=allow_predicted,
    )
    write_json(out_events, out_doc)
    return {
        "qa_events": portable(qa_events),
        "track_json": portable(track_json),
        "out_events": portable(out_events),
        **counts,
    }


def apply_manifest(
    *,
    qa_manifest: Path,
    tracks_root: Path,
    out_root: Path,
    max_time_delta_sec: float,
    max_center_delta_px: float,
    promote_model: bool,
    allow_predicted: bool,
) -> dict[str, Any]:
    manifest = read_json(qa_manifest)
    runs: list[dict[str, Any]] = []
    totals = {
        "videos": 0,
        "missing_tracks": 0,
        "events": 0,
        "matched": 0,
        "promoted": 0,
        "mismatched": 0,
        "no_track": 0,
        "predicted_matches": 0,
    }
    for run in manifest.get("runs", []):
        qa_path = resolve_path(run["qa_events_path"])
        qa_doc = read_json(qa_path)
        track_json = track_path_for_doc(qa_doc, qa_path, tracks_root)
        out_dir = out_root / qa_path.parent.name
        out_events = out_dir / "qa_events.json"
        next_run = dict(run)
        totals["videos"] += 1
        if not track_json.exists():
            totals["missing_tracks"] += 1
            next_run["detector_track_status"] = "missing"
            next_run["expected_detector_track"] = portable(track_json)
            runs.append(next_run)
            continue
        result = apply_one(
            qa_events=qa_path,
            track_json=track_json,
            out_events=out_events,
            max_time_delta_sec=max_time_delta_sec,
            max_center_delta_px=max_center_delta_px,
            promote_model=promote_model,
            allow_predicted=allow_predicted,
        )
        for key in ["events", "matched", "promoted", "mismatched", "no_track", "predicted_matches"]:
            totals[key] += int(result.get(key, 0))
        next_run["qa_events_path"] = portable(out_events)
        next_run["detector_track_status"] = "applied"
        next_run["detector_track_json"] = portable(track_json)
        next_run["model_ball_matches"] = result["matched"]
        next_run["model_ball_mismatches"] = result["mismatched"]
        next_run["model_ball_promoted"] = result["promoted"]
        runs.append(next_run)

    out_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": portable(qa_manifest),
        "tracks_root": portable(tracks_root),
        "annotation_method": "qa_events_with_detector_track_evidence",
        "counts": totals,
        "runs": runs,
    }
    write_json(out_root / "qa_manifest.json", out_manifest)
    write_json(out_root / "detector_track_application_summary.json", out_manifest)
    return out_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach detector-track ball evidence to QA events")
    one = parser.add_argument_group("single QA file")
    one.add_argument("--qa-events", type=Path)
    one.add_argument("--track-json", type=Path)
    one.add_argument("--out-events", type=Path)
    batch = parser.add_argument_group("QA manifest")
    batch.add_argument("--qa-manifest", type=Path)
    batch.add_argument("--tracks-root", type=Path)
    batch.add_argument("--out-root", type=Path)
    parser.add_argument("--max-time-delta-sec", type=float, default=0.08)
    parser.add_argument("--max-center-delta-px", type=float, default=36.0)
    parser.add_argument("--promote-model", action="store_true")
    parser.add_argument("--allow-predicted", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.qa_manifest:
        if not args.tracks_root or not args.out_root:
            raise SystemExit("--qa-manifest requires --tracks-root and --out-root")
        summary = apply_manifest(
            qa_manifest=args.qa_manifest,
            tracks_root=args.tracks_root,
            out_root=args.out_root,
            max_time_delta_sec=args.max_time_delta_sec,
            max_center_delta_px=args.max_center_delta_px,
            promote_model=args.promote_model,
            allow_predicted=args.allow_predicted,
        )
        print(f"manifest: {args.out_root / 'qa_manifest.json'}")
        print(json.dumps(summary["counts"], indent=2))
        return
    if not args.qa_events or not args.track_json or not args.out_events:
        raise SystemExit("Provide either --qa-manifest/--tracks-root/--out-root or --qa-events/--track-json/--out-events")
    result = apply_one(
        qa_events=args.qa_events,
        track_json=args.track_json,
        out_events=args.out_events,
        max_time_delta_sec=args.max_time_delta_sec,
        max_center_delta_px=args.max_center_delta_px,
        promote_model=args.promote_model,
        allow_predicted=args.allow_predicted,
    )
    print(f"qa events: {args.out_events}")
    print(json.dumps({key: result[key] for key in ["events", "matched", "promoted", "mismatched", "no_track", "predicted_matches"]}, indent=2))


if __name__ == "__main__":
    main()
