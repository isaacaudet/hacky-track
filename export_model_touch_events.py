#!/usr/bin/env python3
"""Export model-only touch event streams for clips without release event rows.

The release touch classifier writes OOF/frozen-test events only for visually
reviewed clips. Stall/drop sequence features also need touch-stream context on
reset-review clips that do not yet have touch labels. This exporter fills that
gap with a separate model-only artifact. It never overwrites release metrics or
the OOF/frozen event files.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from attach_touch_audio_features import attach_dataset as attach_audio_dataset
from attach_touch_l2_features import attach_dataset as attach_l2_dataset
from build_touch_training_table import cluster_candidates, load_manifest, read_json
from train_touch_classifier import event_rows_from_predictions, model_predictions


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "runs/release-27-public/touch_corpus_v1"
DEFAULT_REVIEW_MANIFEST = DEFAULT_CORPUS / "touch_review_manifest.json"
DEFAULT_CANDIDATES_DIR = DEFAULT_CORPUS / "audio_candidates"
DEFAULT_CLASSIFIER_DIR = DEFAULT_CORPUS / "touch_classifier_v1"
DEFAULT_DATASET_DIR = DEFAULT_CLASSIFIER_DIR / "model_only_event_dataset"
DEFAULT_STALL_DROP_ROWS = DEFAULT_CORPUS / "release_stall_drop_classifier_v1/stall_drop_training_rows.jsonl"
DEFAULT_DETECTIONS_JSONL = [
    DEFAULT_CORPUS / "owlv2_touch_detections_v1/detections.jsonl",
    DEFAULT_CORPUS / "owlv2_touch_detections_contact_missing_v1/detections.jsonl",
    DEFAULT_CORPUS / "owlv2_stall_drop_missing_detections_v1/detections.jsonl",
]
DEFAULT_EXISTING_EVENTS = [
    DEFAULT_CLASSIFIER_DIR / "touch_classifier_frozen_events.jsonl",
    DEFAULT_CLASSIFIER_DIR / "touch_classifier_oof_events.jsonl",
]
DEFAULT_AUDIO_WAV_DIR = DEFAULT_CORPUS / "audio_candidates/wav"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def normalize_video_id(value: Any) -> str:
    text = Path(str(value or "")).name
    text = text.removesuffix(".MOV").removesuffix(".mov").removesuffix(".review.json")
    text = text.replace("_singular_display 2", "_singular_display-2")
    return text


def event_stream_video_ids(paths: list[Path]) -> set[str]:
    out: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            for key in (row.get("video_id"), row.get("video_name")):
                if key:
                    out.add(normalize_video_id(key))
    return out


def reset_review_video_ids(path: Path) -> set[str]:
    out: set[str] = set()
    for row in read_jsonl(path):
        if row.get("kind") in {"drop_floor", "stall"}:
            out.add(normalize_video_id(row.get("video_id") or row.get("video_name")))
    return out


def candidate_path(candidates_dir: Path, video_id: str) -> Path:
    return candidates_dir / f"{safe_slug(video_id)}.touch_candidates.json"


def build_unreviewed_candidate_rows(
    *,
    manifest_items: list[dict[str, Any]],
    selected_video_ids: set[str],
    candidates_dir: Path,
    cluster_gap_sec: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    for item in manifest_items:
        video_id = str(item["video_id"])
        if video_id not in selected_video_ids:
            continue
        path = candidate_path(candidates_dir, video_id)
        if not path.exists():
            video_rows.append({"video_id": video_id, "video_name": item.get("video_name"), "status": "missing_candidates", "rows": 0})
            continue
        payload = read_json(path)
        clusters = cluster_candidates(payload, cluster_gap_sec=cluster_gap_sec, include_generated_event_hints=True)
        rows: list[dict[str, Any]] = []
        for index, cluster in enumerate(clusters):
            prev_time = None if index == 0 else clusters[index - 1].time_sec
            next_time = None if index + 1 >= len(clusters) else clusters[index + 1].time_sec
            rows.append(
                {
                    "schema_version": 1,
                    "video_id": video_id,
                    "video_name": item["video_name"],
                    "split": item["split"],
                    "candidate_time_sec": round(float(cluster.time_sec), 6),
                    "label_is_touch": False,
                    "label_touch_time_sec": None,
                    "label_touch_delta_sec": None,
                    "candidate_reviewed": False,
                    "candidate_review_decision": None,
                    "candidate_review_source": "model_only_unreviewed",
                    "candidate_review_delta_sec": None,
                    "has_audio": bool(cluster.has_audio),
                    "audio_strength": None if cluster.audio_strength is None else round(float(cluster.audio_strength), 6),
                    "has_existing_hint": bool(cluster.has_existing_hint),
                    "existing_hint_types": list(cluster.existing_hint_types),
                    "raw_sources": list(cluster.raw_sources),
                    "time_since_prev_candidate_sec": None if prev_time is None else round(float(cluster.time_sec - prev_time), 6),
                    "time_to_next_candidate_sec": None if next_time is None else round(float(next_time - cluster.time_sec), 6),
                    "nearest_drop_delta_sec": None,
                    "nearest_stall_delta_sec": None,
                    "in_stall_window": False,
                    "trajectory_feature_status": "missing_l2_features",
                    "trajectory_break_support": None,
                    "trajectory_nearest_break_delta_sec": None,
                    "trajectory_max_positive_dvy": None,
                    "height_reversal": None,
                    "detector_confidence_near_candidate": None,
                    "model_only_stream": True,
                }
            )
        target = test_rows if item.get("split") == "test_frozen" else train_rows
        target.extend(rows)
        video_rows.append(
            {
                "video_id": video_id,
                "video_name": item.get("video_name"),
                "split": item.get("split"),
                "status": "selected",
                "rows": len(rows),
            }
        )
    return train_rows, test_rows, video_rows


def retag_model_only_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["event_match_type"] = "model_only_unreviewed"
        item["matched_truth_time_sec"] = None
        item["matched_truth_delta_sec"] = None
        item["model_only_stream"] = True
        item["truth_source"] = "none_unreviewed_model_stream"
        item["source"] = "touch_classifier_model_only"
        out.append(item)
    return out


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Model-Only Touch Event Stream",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Model: `{manifest['model_path']}`",
        f"- Existing event streams respected: `{manifest['existing_event_streams']}`",
        f"- Selected videos: `{manifest['selected_videos']}`",
        f"- Candidate rows: `{manifest['candidate_rows']}`",
        f"- Prediction rows: `{manifest['prediction_rows']}`",
        f"- Event rows: `{manifest['event_rows']}`",
        f"- Events JSONL: `{manifest['events_jsonl']}`",
        "",
        "This file is model-only context for downstream sequence features. It is not release touch evidence and does not overwrite OOF/frozen event outputs.",
        "",
        "## Videos",
        "",
        "| video | split | rows | events | status |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    events_by_video = Counter(str(row.get("video_id")) for row in read_jsonl(Path(manifest["events_jsonl"])))
    for row in manifest["videos"]:
        lines.append(
            f"| `{row.get('video_id')}` | `{row.get('split')}` | {row.get('rows', 0)} | "
            f"{events_by_video.get(str(row.get('video_id')), 0)} | `{row.get('status')}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_model_touch_events(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()
    review_items, _ = load_manifest(args.review_manifest)
    existing_stream_videos = event_stream_video_ids(args.existing_events_jsonl)
    if args.video_id:
        selected = {normalize_video_id(video_id) for video_id in args.video_id}
    else:
        selected = reset_review_video_ids(args.stall_drop_rows) - existing_stream_videos

    train_rows, test_rows, video_rows = build_unreviewed_candidate_rows(
        manifest_items=review_items,
        selected_video_ids=selected,
        candidates_dir=args.candidates_dir.resolve(),
        cluster_gap_sec=args.cluster_gap_sec,
    )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(dataset_dir / "touch_training_candidates.jsonl", train_rows)
    write_jsonl(dataset_dir / "touch_training_test_frozen.jsonl", test_rows)

    l2_manifest = attach_l2_dataset(
        argparse.Namespace(
            dataset_dir=dataset_dir,
            out_dir=dataset_dir,
            detections_jsonl=args.detections_jsonl,
            detections_dir=[],
            threshold=args.trajectory_threshold,
            break_tolerance_sec=args.break_tolerance_sec,
            max_track_gap_sec=args.max_track_gap_sec,
            min_points=args.min_points,
        )
    )
    audio_manifest = attach_audio_dataset(
        argparse.Namespace(
            dataset_dir=dataset_dir,
            out_dir=dataset_dir,
            audio_wav_dir=args.audio_wav_dir,
            window_sec=args.audio_feature_window_sec,
            n_fft=args.audio_feature_n_fft,
            hop_length=args.audio_feature_hop_length,
        )
    )

    rows = read_jsonl(dataset_dir / "touch_training_candidates.jsonl") + read_jsonl(dataset_dir / "touch_training_test_frozen.jsonl")
    model_doc = joblib.load(args.model_path)
    predictions = model_predictions(
        model_doc["model"],
        rows,
        list(model_doc["feature_names"]),
        float(model_doc["threshold"]),
        apply_precision_gate=True,
    )
    events = retag_model_only_events(event_rows_from_predictions(predictions, labels_dir=None, include_false_negatives=False))

    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "touch_classifier_model_only_predictions.jsonl"
    events_path = out_dir / "touch_classifier_model_only_events.jsonl"
    manifest_path = out_dir / "touch_classifier_model_only_events_manifest.json"
    report_path = out_dir / "touch_classifier_model_only_events_report.md"
    write_jsonl(predictions_path, predictions)
    write_jsonl(events_path, events)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "model_path": str(args.model_path),
        "review_manifest": str(args.review_manifest),
        "candidates_dir": str(args.candidates_dir),
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "existing_event_streams": [str(path) for path in args.existing_events_jsonl],
        "existing_stream_video_count": len(existing_stream_videos),
        "selected_videos": sorted(selected),
        "videos": video_rows,
        "candidate_rows": len(rows),
        "prediction_rows": len(predictions),
        "event_rows": len(events),
        "predictions_jsonl": str(predictions_path),
        "events_jsonl": str(events_path),
        "l2_manifest": l2_manifest,
        "audio_manifest": audio_manifest,
        "report": str(report_path),
    }
    write_json(manifest_path, manifest)
    write_report(report_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export model-only touch event streams for clips without release event rows")
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW_MANIFEST)
    parser.add_argument("--candidates-dir", type=Path, default=DEFAULT_CANDIDATES_DIR)
    parser.add_argument("--stall-drop-rows", type=Path, default=DEFAULT_STALL_DROP_ROWS)
    parser.add_argument("--existing-events-jsonl", type=Path, action="append", default=list(DEFAULT_EXISTING_EVENTS))
    parser.add_argument("--detections-jsonl", type=Path, action="append", default=list(DEFAULT_DETECTIONS_JSONL))
    parser.add_argument("--model-path", type=Path, default=DEFAULT_CLASSIFIER_DIR / "touch_classifier.joblib")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_CLASSIFIER_DIR)
    parser.add_argument("--video-id", action="append")
    parser.add_argument("--cluster-gap-sec", type=float, default=0.18)
    parser.add_argument("--trajectory-threshold", type=float, default=0.2)
    parser.add_argument("--break-tolerance-sec", type=float, default=0.08)
    parser.add_argument("--max-track-gap-sec", type=float, default=0.25)
    parser.add_argument("--min-points", type=int, default=12)
    parser.add_argument("--audio-wav-dir", type=Path, default=DEFAULT_AUDIO_WAV_DIR)
    parser.add_argument("--audio-feature-window-sec", type=float, default=0.18)
    parser.add_argument("--audio-feature-n-fft", type=int, default=1024)
    parser.add_argument("--audio-feature-hop-length", type=int, default=128)
    return parser


def main() -> None:
    manifest = export_model_touch_events(build_parser().parse_args())
    print(f"status: {manifest['status']}")
    print(f"events: {manifest['events_jsonl']}")
    print(f"report: {manifest['report']}")
    print(f"videos: {len(manifest['selected_videos'])} rows: {manifest['candidate_rows']} events: {manifest['event_rows']}")


if __name__ == "__main__":
    main()
