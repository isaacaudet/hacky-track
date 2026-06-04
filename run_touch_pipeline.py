#!/usr/bin/env python3
"""Run the fixed-OWLv2 touch pipeline scaffold end to end.

This command assembles the current pipeline stages around the human labeling
boundary. It can rebuild the corpus/candidate/table artifacts, optionally
attach cached OWLv2 L2 features, run the guarded classifier trainer, and write a
single status report showing exactly what is still missing.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from attach_touch_l2_features import attach_dataset
from attach_touch_audio_features import attach_dataset as attach_audio_dataset
from attach_touch_flow_features import attach_dataset as attach_flow_dataset
from attach_touch_foot_track_features import attach_dataset as attach_foot_track_dataset
from attach_touch_pose_features import attach_dataset as attach_pose_dataset
from attach_touch_visual_crop_features import attach_dataset as attach_visual_crop_dataset
from attach_touch_vision_embedding_features import attach_dataset as attach_vision_embedding_dataset
from build_touch_review_candidates import AUDIO_DELTA, AUDIO_WAIT_SEC, build_candidates
from build_touch_training_table import build_training_table
from export_touch_owlv2_detections import export_detections
from import_existing_touch_labels import import_existing_labels
from prefill_touch_labels_from_reviews import prefill_labels
from touch_corpus_manager import DEFAULT_VIDEO_ROOT, build_inventory, build_touch_review_manifest, write_json, write_jsonl, write_markdown_report
from touch_label_readiness import build_readiness_report
from train_touch_classifier import train_classifier


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "runs/release-27-public/touch_corpus_v1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def release_gate_status(classifier_summary: dict[str, Any]) -> dict[str, Any]:
    classifier_status = classifier_summary.get("status")
    cv_gate = classifier_summary.get("event_level_cv_gate") or classifier_summary.get("cv_gate") or {}
    frozen_gate = classifier_summary.get("event_level_frozen_test_gate") or classifier_summary.get("frozen_test_gate") or {}
    strict_trained = classifier_status == "trained"
    cv_passes = bool(cv_gate.get("passes"))
    frozen_passes = bool(frozen_gate.get("passes"))
    passes = bool(strict_trained and cv_passes and frozen_passes)
    if passes:
        status = "release_gate_passed"
        reason = "strict classifier trained and both leave-clips-out CV and frozen-test merged event gates pass"
    elif strict_trained:
        status = "trained_gate_failed"
        reason = "strict classifier trained, but merged event touch gates did not both pass"
    else:
        status = "not_ready"
        reason = "strict classifier is not trained"
    return {
        "status": status,
        "passes": passes,
        "strict_trained": strict_trained,
        "cv_gate_passes": cv_passes,
        "frozen_test_gate_passes": frozen_passes,
        "reason": reason,
        "cv_gate": cv_gate or None,
        "frozen_test_gate": frozen_gate or None,
        "gate_level": "merged_event",
    }


def training_table_effective_status(training_manifest: dict[str, Any], l2_summary: dict[str, Any]) -> str:
    status = str(training_manifest.get("status") or "unknown")
    candidate_rows = int(training_manifest.get("candidate_rows") or 0)
    ok_rows = int(l2_summary.get("train_val_ok_rows") or 0) + int(l2_summary.get("test_frozen_ok_rows") or 0)
    if candidate_rows and ok_rows >= candidate_rows:
        return "l2_features_attached"
    if ok_rows:
        return "partial_l2_features_attached"
    return status


def pose_feature_summary_for_status(manifest: dict[str, Any] | None) -> dict[str, Any]:
    if not manifest:
        return {
            "status": "skipped",
            "train_val_rows": 0,
            "train_val_pose_present_rows": 0,
            "train_val_foot_present_rows": 0,
            "test_frozen_rows": 0,
            "test_frozen_pose_present_rows": 0,
            "test_frozen_foot_present_rows": 0,
        }
    return {
        "status": manifest.get("status"),
        "runtime_status": manifest.get("runtime_status"),
        "train_val_rows": (manifest.get("train_val") or {}).get("rows", 0),
        "train_val_pose_present_rows": (manifest.get("train_val") or {}).get("pose_present_rows", 0),
        "train_val_foot_present_rows": (manifest.get("train_val") or {}).get("foot_present_rows", 0),
        "test_frozen_rows": (manifest.get("test_frozen") or {}).get("rows", 0),
        "test_frozen_pose_present_rows": (manifest.get("test_frozen") or {}).get("pose_present_rows", 0),
        "test_frozen_foot_present_rows": (manifest.get("test_frozen") or {}).get("foot_present_rows", 0),
    }


def audio_feature_summary_for_status(manifest: dict[str, Any] | None) -> dict[str, Any]:
    if not manifest:
        return {
            "status": "skipped",
            "train_val_rows": 0,
            "train_val_ok_rows": 0,
            "test_frozen_rows": 0,
            "test_frozen_ok_rows": 0,
        }
    return {
        "status": manifest.get("status"),
        "train_val_rows": (manifest.get("train_val") or {}).get("rows", 0),
        "train_val_ok_rows": (manifest.get("train_val") or {}).get("ok_rows", 0),
        "test_frozen_rows": (manifest.get("test_frozen") or {}).get("rows", 0),
        "test_frozen_ok_rows": (manifest.get("test_frozen") or {}).get("ok_rows", 0),
    }


def flow_feature_summary_for_status(manifest: dict[str, Any] | None) -> dict[str, Any]:
    if not manifest:
        return {
            "status": "skipped",
            "train_val_rows": 0,
            "train_val_ok_rows": 0,
            "test_frozen_rows": 0,
            "test_frozen_ok_rows": 0,
        }
    return {
        "status": manifest.get("status"),
        "train_val_rows": (manifest.get("train_val") or {}).get("rows", 0),
        "train_val_ok_rows": (manifest.get("train_val") or {}).get("ok_rows", 0),
        "test_frozen_rows": (manifest.get("test_frozen") or {}).get("rows", 0),
        "test_frozen_ok_rows": (manifest.get("test_frozen") or {}).get("ok_rows", 0),
    }


def visual_crop_feature_summary_for_status(manifest: dict[str, Any] | None) -> dict[str, Any]:
    if not manifest:
        return {
            "status": "skipped",
            "train_val_rows": 0,
            "train_val_ok_rows": 0,
            "test_frozen_rows": 0,
            "test_frozen_ok_rows": 0,
        }
    return {
        "status": manifest.get("status"),
        "train_val_rows": (manifest.get("train_val") or {}).get("rows", 0),
        "train_val_ok_rows": (manifest.get("train_val") or {}).get("ok_rows", 0),
        "test_frozen_rows": (manifest.get("test_frozen") or {}).get("rows", 0),
        "test_frozen_ok_rows": (manifest.get("test_frozen") or {}).get("ok_rows", 0),
    }


def vision_embedding_feature_summary_for_status(manifest: dict[str, Any] | None) -> dict[str, Any]:
    if not manifest:
        return {
            "status": "skipped",
            "train_val_rows": 0,
            "train_val_requested_rows": 0,
            "train_val_ok_rows": 0,
            "test_frozen_rows": 0,
            "test_frozen_requested_rows": 0,
            "test_frozen_ok_rows": 0,
        }
    return {
        "status": manifest.get("status"),
        "model_name": manifest.get("model_name"),
        "contact_labeled_only": manifest.get("contact_labeled_only"),
        "train_val_rows": (manifest.get("train_val") or {}).get("rows", 0),
        "train_val_requested_rows": (manifest.get("train_val") or {}).get("requested_rows", 0),
        "train_val_ok_rows": (manifest.get("train_val") or {}).get("ok_rows", 0),
        "test_frozen_rows": (manifest.get("test_frozen") or {}).get("rows", 0),
        "test_frozen_requested_rows": (manifest.get("test_frozen") or {}).get("requested_rows", 0),
        "test_frozen_ok_rows": (manifest.get("test_frozen") or {}).get("ok_rows", 0),
    }


def foot_track_feature_summary_for_status(manifest: dict[str, Any] | None) -> dict[str, Any]:
    if not manifest:
        return {
            "status": "skipped",
            "train_val_rows": 0,
            "train_val_ok_rows": 0,
            "train_val_manual_calibrated_rows": 0,
            "test_frozen_rows": 0,
            "test_frozen_ok_rows": 0,
            "test_frozen_manual_calibrated_rows": 0,
        }
    return {
        "status": manifest.get("status"),
        "train_val_rows": (manifest.get("train_val") or {}).get("rows", 0),
        "train_val_ok_rows": (manifest.get("train_val") or {}).get("ok_rows", 0),
        "train_val_manual_calibrated_rows": (manifest.get("train_val") or {}).get("manual_calibrated_rows", 0),
        "test_frozen_rows": (manifest.get("test_frozen") or {}).get("rows", 0),
        "test_frozen_ok_rows": (manifest.get("test_frozen") or {}).get("ok_rows", 0),
        "test_frozen_manual_calibrated_rows": (manifest.get("test_frozen") or {}).get("manual_calibrated_rows", 0),
    }


def should_run_diagnostic_classifier(classifier_summary: dict[str, Any], *, audio_only: bool) -> bool:
    """Train a non-release diagnostic model when frozen-test labels are the only hard blocker."""
    if audio_only or classifier_summary.get("status") != "not_ready":
        return False
    reasons = [str(reason) for reason in classifier_summary.get("reasons", [])]
    if not reasons:
        return False
    allowed = {
        "no frozen-test candidate rows; release gate evaluation requires visually reviewed frozen-test labels",
        "frozen-test rows must contain both positive and negative candidates for touch precision/recall gate evaluation",
    }
    return all(reason in allowed for reason in reasons)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    train_status = summary.get("classifier", {}).get("status")
    reasons = summary.get("classifier", {}).get("reasons") or []
    readiness = summary.get("label_readiness", {})
    next_clips = readiness.get("next_clips", [])
    release_gate = summary.get("release_gate", {})
    diagnostic = summary.get("diagnostic_classifier") or {}
    diagnostic_cv = diagnostic.get("cv_gate") or {}
    diagnostic_agg = diagnostic.get("leave_one_video_out_aggregate") or {}
    diagnostic_errors = diagnostic.get("out_of_fold_errors") or {}
    detection_inputs = summary.get("detection_inputs") or {}
    detection_jsonls = [str(item) for item in detection_inputs.get("jsonl", [])]
    detection_dirs = [str(item) for item in detection_inputs.get("dirs", [])]
    release_command_parts = ["python3 run_touch_pipeline.py"]
    for detection_jsonl in detection_jsonls:
        release_command_parts.extend(["--detections-jsonl", detection_jsonl])
    for detections_dir in detection_dirs:
        release_command_parts.extend(["--detections-dir", detections_dir])
    release_command_parts.append("--attach-audio-features")
    lines = [
        "# Fixed-OWLv2 Touch Pipeline Status",
        "",
        f"- Status: `{summary['status']}`",
        f"- Release gate: `{'PASS' if release_gate.get('passes') else 'NOT PASSED'}`",
        f"- Release gate reason: {release_gate.get('reason')}",
        f"- Release gate level: `{release_gate.get('gate_level')}`",
        f"- Corpus dir: `{summary['corpus_dir']}`",
        f"- Videos: `{summary['corpus'].get('videos')}`",
        f"- Frozen-test videos: `{summary['review_manifest'].get('frozen_test_items')}`",
        f"- Audio candidates: `{summary['candidates'].get('total_audio_candidates')}`",
        f"- Existing event hints: `{summary['candidates'].get('total_existing_event_hints')}`",
        f"- Generated model hints: `{summary['candidates'].get('total_generated_event_hints')}`",
        f"- Legacy labels imported/skipped: `{summary['legacy_import'].get('imported')}` / `{summary['legacy_import'].get('skipped_existing_label')}`",
        f"- Review subset drafts prefilled/skipped: `{summary['review_subset_prefill'].get('prefilled')}` / `{summary['review_subset_prefill'].get('skipped_existing_label')}`",
        f"- Label readiness: `{readiness.get('status')}`",
        f"- Complete-ready visual labels: `{readiness.get('complete_ready_videos')}` / `{summary['corpus'].get('videos')}`",
        f"- Non-test / frozen-test ready: `{readiness.get('non_test_complete_ready_videos')}` / `{readiness.get('frozen_test_complete_ready_videos')}`",
        f"- Unchecked visual-review hints: `{readiness.get('unchecked_hints')}`",
        f"- Likely/model unchecked hints: `{readiness.get('likely_unchecked_hints')}`",
        f"- Audio-tail unchecked hints: `{readiness.get('audio_only_unchecked_hints')}`",
        f"- Complete labeled videos used: `{summary['training_table'].get('labeled_videos')}`",
        f"- Draft label files skipped: `{summary['training_table'].get('skipped_incomplete_video_count')}`",
        f"- Candidate rows: `{summary['training_table'].get('candidate_rows')}`",
        f"- Reviewed candidate rows: `{summary['training_table'].get('reviewed_candidate_rows')}`",
        f"- Negative candidate rows without review: `{summary['training_table'].get('negative_candidate_rows_without_review')}`",
        f"- OWLv2 export status: `{summary['owlv2_export'].get('status')}`",
        f"- OWLv2 frames planned/exported: `{summary['owlv2_export'].get('frames_planned')}` / `{summary['owlv2_export'].get('frames_exported')}`",
        f"- Detection JSONL inputs: `{len(detection_jsonls)}`",
        f"- Detection directory inputs: `{len(detection_dirs)}`",
        f"- L2 rows with features: `{summary['l2_features'].get('train_val_ok_rows')}`",
        f"- Audio timbre status: `{summary['audio_features'].get('status')}`",
        f"- Audio timbre train/test rows: `{summary['audio_features'].get('train_val_ok_rows')}` / `{summary['audio_features'].get('test_frozen_ok_rows')}`",
        f"- Optical-flow status: `{summary['flow_features'].get('status')}`",
        f"- Optical-flow train/test rows: `{summary['flow_features'].get('train_val_ok_rows')}` / `{summary['flow_features'].get('test_frozen_ok_rows')}`",
        f"- Pose feature status: `{summary['pose_features'].get('status')}`",
        f"- Pose train/test foot-present rows: `{summary['pose_features'].get('train_val_foot_present_rows')}` / `{summary['pose_features'].get('test_frozen_foot_present_rows')}`",
        f"- Foot-track status: `{summary['foot_track_features'].get('status')}`",
        f"- Foot-track train/test rows: `{summary['foot_track_features'].get('train_val_ok_rows')}` / `{summary['foot_track_features'].get('test_frozen_ok_rows')}`",
        f"- Visual-crop status: `{summary['visual_crop_features'].get('status')}`",
        f"- Visual-crop train/test rows: `{summary['visual_crop_features'].get('train_val_ok_rows')}` / `{summary['visual_crop_features'].get('test_frozen_ok_rows')}`",
        f"- Vision-embedding status: `{summary['vision_embedding_features'].get('status')}`",
        f"- Vision-embedding train/test requested/ok rows: "
        f"`{summary['vision_embedding_features'].get('train_val_requested_rows')}` / `{summary['vision_embedding_features'].get('train_val_ok_rows')}` and "
        f"`{summary['vision_embedding_features'].get('test_frozen_requested_rows')}` / `{summary['vision_embedding_features'].get('test_frozen_ok_rows')}`",
        f"- Classifier status: `{train_status}`",
        f"- Diagnostic classifier status: `{diagnostic.get('status')}`",
        "",
        "## Label Readiness",
        "",
        "| priority | video | split | status | hints | unchecked | likely left | audio-tail left | reason |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in next_clips[:8]:
        lines.append(
            f"| {row['priority']} | `{row['video_name']}` | {row['split']} | {row['status']} | "
            f"{row['total_hint_count']} | {row['unchecked_hint_count']} | "
            f"{row.get('likely_unchecked_hint_count', 0)} | {row.get('audio_only_unchecked_hint_count', 0)} | {row['reason']} |"
        )
    if not next_clips:
        lines.append("|  |  |  |  |  |  |  |  | No missing review work reported. |")
    lines.extend(
        [
            "",
            f"Full checklist: `{summary['paths'].get('label_readiness_report')}`",
            "",
            "## Release Gate",
            "",
            f"- Strict trained: `{release_gate.get('strict_trained')}`",
            f"- Leave-clips-out CV gate: `{release_gate.get('cv_gate_passes')}`",
            f"- Frozen-test gate: `{release_gate.get('frozen_test_gate_passes')}`",
            f"- Gate level: `{release_gate.get('gate_level')}`",
            f"- Status: `{release_gate.get('status')}`",
            "",
        "## Classifier Readiness",
        "",
        ]
    )
    if reasons:
        for reason in reasons:
            lines.append(f"- {reason}")
    else:
        lines.append("- No readiness blockers reported by the trainer.")
    lines.extend(["", "## Diagnostic Classifier", ""])
    if diagnostic.get("status") in {"trained_smoke", "trained"}:
        lines.extend(
            [
                "- Non-release diagnostic only: trained without frozen-test rows.",
                f"- Report: `{diagnostic.get('report_path')}`",
                f"- Rows: `{diagnostic.get('train_val_rows')}` train/validation, `{diagnostic.get('test_frozen_rows')}` frozen test",
                f"- Leave-clips-out precision: `{float(diagnostic_agg.get('precision') or 0.0):.3f}`",
                f"- Leave-clips-out recall: `{float(diagnostic_agg.get('recall') or 0.0):.3f}`",
                f"- Leave-clips-out F1: `{float(diagnostic_agg.get('f1') or 0.0):.3f}`",
                f"- Candidate CV gate: `{'PASS' if diagnostic_cv.get('passes') else 'FAIL'}` "
                "(still not release evidence without frozen-test labels)",
                f"- Out-of-fold false negatives / false positives: `{diagnostic_errors.get('false_negative', 0)}` / `{diagnostic_errors.get('false_positive', 0)}`",
                f"- Error JSONL: `{diagnostic.get('error_jsonl')}`",
                f"- Error CSV: `{diagnostic.get('error_csv')}`",
            ]
        )
    elif diagnostic.get("status") == "skipped":
        lines.append(f"- Skipped: {diagnostic.get('reason')}")
    else:
        lines.append("- Not run.")
    lines.extend(
        [
            "",
            "## Commands",
            "",
            "```bash",
            "python3 run_touch_pipeline.py --import-existing-labels",
            "python3 prefill_touch_labels_from_reviews.py",
            "python3 run_touch_pipeline.py --prefill-review-subsets",
            "python3 run_touch_pipeline.py",
            "python3 import_existing_touch_labels.py",
            "python3 touch_review_app.py",
            "python3 run_touch_pipeline.py --export-owlv2-detections",
            " ".join(release_command_parts),
            "```",
            "",
            "Notes:",
            "- Audio candidates are hints only; labels must come from muted visual review.",
            "- Frozen-test rows are exported separately and are not used for training or cross-validation.",
            "- Final release evidence requires visual labels, matching fixed-OWLv2 tracks, L2 features, and trained classifier metrics.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def release_feature_disable_flags(args: argparse.Namespace) -> dict[str, bool]:
    """Release/diagnostic classifiers exclude optical-flow and pose columns by
    default. On the held-out frozen-test clips both lower precision (the binding
    release gate) relative to the audio+trajectory feature set: optical-flow is a
    net negative and pose has sparse coverage. Flow/pose are still attached and
    shown in the ablation table; opt them into the release model with
    --include-flow-features / --include-pose-features."""
    return {
        "disable_flow_features": not getattr(args, "include_flow_features", False),
        "disable_pose_features": not getattr(args, "include_pose_features", False),
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    corpus_dir = args.corpus_dir.resolve()
    corpus_dir.mkdir(parents=True, exist_ok=True)

    inventory = build_inventory(args.video_root.expanduser().resolve())
    review_manifest = build_touch_review_manifest(inventory)
    inventory_path = corpus_dir / "touch_corpus_inventory.json"
    review_manifest_path = corpus_dir / "touch_review_manifest.json"
    write_json(inventory_path, inventory)
    write_json(review_manifest_path, review_manifest)
    write_jsonl(corpus_dir / "touch_review_items.jsonl", review_manifest["items"])
    write_markdown_report(corpus_dir / "touch_corpus_report.md", inventory, review_manifest_path)

    candidates_dir = args.candidates_dir.resolve() if args.candidates_dir else corpus_dir / "audio_candidates"
    candidate_summary = build_candidates(
        argparse.Namespace(
            review_manifest=review_manifest_path,
            out_dir=candidates_dir,
            audio_delta=args.audio_delta,
            audio_wait_sec=args.audio_wait_sec,
            limit=args.candidate_limit,
            force=args.force_audio,
            include_generated_event_hints=args.include_generated_event_hints,
        )
    )

    labels_dir = args.labels_dir.resolve() if args.labels_dir else corpus_dir / "visual_touch_labels"
    dataset_dir = args.dataset_dir.resolve() if args.dataset_dir else corpus_dir / "touch_training_dataset_v1"
    legacy_import_summary: dict[str, Any] = {
        "status": "skipped",
        "videos_with_legacy_events": 0,
        "imported": 0,
        "skipped_existing_label": 0,
        "approved_touches": 0,
        "candidate_reviews": 0,
        "audio_assisted_imports": [],
    }
    if args.import_existing_labels:
        legacy_import_summary = import_existing_labels(
            argparse.Namespace(
                review_manifest=review_manifest_path,
                candidates_dir=candidates_dir,
                labels_dir=labels_dir,
                touch_tolerance_sec=args.touch_tolerance_sec,
                cluster_gap_sec=args.cluster_gap_sec,
                force=args.force_import_existing_labels,
                dry_run=False,
            )
        )
        legacy_import_summary["status"] = "ran"
    review_subset_prefill_summary: dict[str, Any] = {
        "status": "skipped",
        "prefilled": 0,
        "would_prefill": 0,
        "skipped_existing_label": 0,
        "missing_review_file": 0,
        "prefilled_events": 0,
        "prefilled_touches": 0,
        "candidate_reviews": 0,
        "candidate_touch_reviews": 0,
        "conflicted_review_items": 0,
    }
    if args.prefill_review_subsets:
        review_subset_prefill_summary = prefill_labels(
            argparse.Namespace(
                review_manifest=review_manifest_path,
                reviews_dir=args.reviews_dir,
                candidates_dir=candidates_dir,
                labels_dir=labels_dir,
                split=args.prefill_review_split,
                candidate_match_tolerance_sec=args.touch_tolerance_sec,
                conflict_tolerance_sec=args.prefill_conflict_tolerance_sec,
                cluster_gap_sec=args.cluster_gap_sec,
                write_empty=False,
                force=args.force_prefill_review_subsets,
                dry_run=False,
            )
        )
        review_subset_prefill_summary["status"] = "ran"
    elif (labels_dir / "review_subset_prefill_summary.json").exists():
        review_subset_prefill_summary = read_json(labels_dir / "review_subset_prefill_summary.json")
        review_subset_prefill_summary["status"] = "existing"
    training_manifest = build_training_table(
        argparse.Namespace(
            review_manifest=review_manifest_path,
            candidates_dir=candidates_dir,
            labels_dir=labels_dir,
            out_dir=dataset_dir,
            touch_tolerance_sec=args.touch_tolerance_sec,
            cluster_gap_sec=args.cluster_gap_sec,
            allow_non_visual_labels=args.allow_non_visual_labels,
            allow_incomplete_labels=args.allow_incomplete_labels,
            require_labels=False,
        )
    )
    readiness_manifest = build_readiness_report(
        argparse.Namespace(
            review_manifest=review_manifest_path,
            candidates_dir=candidates_dir,
            labels_dir=labels_dir,
            out_dir=corpus_dir,
            review_match_tolerance_sec=max(args.cluster_gap_sec, 0.05),
            candidate_cluster_gap_sec=args.cluster_gap_sec,
            min_non_test_videos=args.min_videos,
            min_frozen_test_videos=1,
        )
    )

    detection_jsonls = list(args.detections_jsonl or [])
    owlv2_export_summary: dict[str, Any] = {
        "status": "skipped",
        "output_jsonl": None,
        "candidate_rows": training_manifest["candidate_rows"],
        "videos_planned": 0,
        "frames_planned": 0,
        "frames_exported": 0,
        "cached": False,
    }
    if args.export_owlv2_detections:
        owlv2_out_dir = args.owlv2_out_dir.resolve() if args.owlv2_out_dir else corpus_dir / "owlv2_touch_detections_v1"
        owlv2_manifest = export_detections(
            argparse.Namespace(
                dataset_dir=dataset_dir,
                review_manifest=review_manifest_path,
                out_dir=owlv2_out_dir,
                model=args.owlv2_model,
                threshold=0.2,
                device=args.owlv2_device,
                prompts=args.owlv2_prompts,
                seconds_before=args.owlv2_seconds_before,
                seconds_after=args.owlv2_seconds_after,
                frame_stride=args.owlv2_frame_stride,
                max_frames=args.owlv2_max_frames,
                dry_run=args.owlv2_dry_run,
                force=args.owlv2_force,
            )
        )
        owlv2_export_summary = {
            "status": owlv2_manifest["status"],
            "output_jsonl": owlv2_manifest["output_jsonl"],
            "candidate_rows": owlv2_manifest["candidate_rows"],
            "videos_planned": owlv2_manifest["videos_planned"],
            "frames_planned": owlv2_manifest["frames_planned"],
            "frames_exported": owlv2_manifest["frames_exported"],
            "cached": owlv2_manifest.get("cached", False),
            "model": owlv2_manifest["model"],
            "threshold": owlv2_manifest["threshold"],
        }
        if not args.owlv2_dry_run:
            detection_jsonls.append(Path(owlv2_manifest["output_jsonl"]))

    l2_out_dir = dataset_dir
    l2_summary = {
        "status": "skipped_no_detections",
        "train_val_rows": training_manifest["train_val_candidate_rows"],
        "train_val_ok_rows": 0,
        "test_frozen_rows": training_manifest["test_frozen_candidate_rows"],
        "test_frozen_ok_rows": 0,
    }
    if detection_jsonls or args.detections_dir:
        l2_manifest = attach_dataset(
            argparse.Namespace(
                dataset_dir=dataset_dir,
                out_dir=l2_out_dir,
                detections_jsonl=detection_jsonls,
                detections_dir=args.detections_dir,
                threshold=args.trajectory_threshold,
                break_tolerance_sec=args.break_tolerance_sec,
                max_track_gap_sec=args.max_track_gap_sec,
                min_points=args.min_points,
            )
        )
        l2_summary = {
            "status": l2_manifest["status"],
            "train_val_rows": l2_manifest["train_val"]["rows"],
            "train_val_ok_rows": l2_manifest["train_val"]["ok_rows"],
            "test_frozen_rows": l2_manifest["test_frozen"]["rows"],
            "test_frozen_ok_rows": l2_manifest["test_frozen"]["ok_rows"],
        }

    audio_manifest: dict[str, Any] | None = None
    if args.attach_audio_features:
        audio_manifest = attach_audio_dataset(
            argparse.Namespace(
                dataset_dir=l2_out_dir,
                out_dir=l2_out_dir,
                audio_wav_dir=args.audio_wav_dir if args.audio_wav_dir else candidates_dir / "wav",
                window_sec=args.audio_feature_window_sec,
                n_fft=args.audio_feature_n_fft,
                hop_length=args.audio_feature_hop_length,
            )
        )
    audio_summary = audio_feature_summary_for_status(audio_manifest)

    flow_manifest: dict[str, Any] | None = None
    if args.attach_flow_features:
        flow_manifest = attach_flow_dataset(
            argparse.Namespace(
                dataset_dir=l2_out_dir,
                out_dir=l2_out_dir,
                review_manifest=review_manifest_path,
                detections_jsonl=detection_jsonls,
                detections_dir=args.detections_dir,
                threshold=args.trajectory_threshold,
                ball_tolerance_sec=args.flow_ball_tolerance_sec,
                frame_step=args.flow_frame_step,
                patch_radius_px=args.flow_patch_radius_px,
                grid_step_px=args.flow_grid_step_px,
                cache_path=args.flow_cache_path,
                force=args.flow_force,
            )
        )
    flow_summary = flow_feature_summary_for_status(flow_manifest)

    pose_manifest: dict[str, Any] | None = None
    if args.attach_pose_features:
        pose_manifest = attach_pose_dataset(
            argparse.Namespace(
                dataset_dir=l2_out_dir,
                out_dir=l2_out_dir,
                review_manifest=review_manifest_path,
                detections_jsonl=detection_jsonls,
                detections_dir=args.detections_dir,
                threshold=args.trajectory_threshold,
                ball_tolerance_sec=args.pose_ball_tolerance_sec,
                pose_mode=args.pose_mode,
                pose_device=args.pose_device,
                keypoint_threshold=args.pose_keypoint_threshold,
                cache_path=args.pose_cache_path,
                cache_only=args.pose_cache_only,
            )
        )
    pose_summary = pose_feature_summary_for_status(pose_manifest)

    foot_track_manifest: dict[str, Any] | None = None
    if args.attach_foot_track_features:
        foot_track_manifest = attach_foot_track_dataset(
            argparse.Namespace(
                dataset_dir=l2_out_dir,
                out_dir=l2_out_dir,
                labels_dir=labels_dir,
                label_match_tolerance_sec=args.touch_tolerance_sec,
                window_sec=args.foot_track_window_sec,
            )
        )
    foot_track_summary = foot_track_feature_summary_for_status(foot_track_manifest)

    visual_crop_manifest: dict[str, Any] | None = None
    if args.attach_visual_crop_features:
        visual_crop_manifest = attach_visual_crop_dataset(
            argparse.Namespace(
                dataset_dir=l2_out_dir,
                out_dir=l2_out_dir,
                review_manifest=review_manifest_path,
                detections_jsonl=detection_jsonls,
                detections_dir=args.detections_dir,
                threshold=args.trajectory_threshold,
                ball_tolerance_sec=args.visual_crop_ball_tolerance_sec,
                crop_size_px=args.visual_crop_size_px,
                grid_size=args.visual_crop_grid_size,
                cache_path=args.visual_crop_cache_path,
            )
        )
    visual_crop_summary = visual_crop_feature_summary_for_status(visual_crop_manifest)

    vision_embedding_manifest: dict[str, Any] | None = None
    if args.attach_vision_embedding_features:
        vision_embedding_manifest = attach_vision_embedding_dataset(
            argparse.Namespace(
                dataset_dir=l2_out_dir,
                out_dir=l2_out_dir,
                review_manifest=review_manifest_path,
                labels_dir=labels_dir,
                detections_jsonl=detection_jsonls,
                detections_dir=args.detections_dir,
                threshold=args.trajectory_threshold,
                ball_tolerance_sec=args.vision_embedding_ball_tolerance_sec,
                crop_size_px=args.vision_embedding_crop_size_px,
                batch_size=args.vision_embedding_batch_size,
                model_name=args.vision_embedding_model,
                device=args.vision_embedding_device,
                cache_path=args.vision_embedding_cache_path,
                contact_labeled_only=args.vision_embedding_contact_labeled_only,
                label_match_tolerance_sec=args.touch_tolerance_sec,
            )
        )
    vision_embedding_summary = vision_embedding_feature_summary_for_status(vision_embedding_manifest)

    release_disable = release_feature_disable_flags(args)
    classifier_summary = train_classifier(
        argparse.Namespace(
            dataset_dir=l2_out_dir,
            out_dir=corpus_dir / "touch_classifier_v1",
            threshold=args.classifier_threshold,
            min_videos=args.min_videos,
            allow_small=args.allow_small_train,
            allow_missing_trajectory=args.allow_missing_trajectory,
            allow_missing_frozen_test=args.allow_missing_frozen_test,
            audio_only=args.audio_only,
            disable_pose_features=release_disable["disable_pose_features"],
            disable_flow_features=release_disable["disable_flow_features"],
            disable_audio_timbre_features=args.disable_audio_timbre_features,
            labels_dir=labels_dir,
        )
    )
    diagnostic_classifier_summary: dict[str, Any] = {
        "status": "skipped",
        "reason": "strict classifier trained or blocker is not limited to frozen-test labels",
    }
    if should_run_diagnostic_classifier(classifier_summary, audio_only=args.audio_only):
        diagnostic_classifier_summary = train_classifier(
            argparse.Namespace(
                dataset_dir=l2_out_dir,
                out_dir=corpus_dir / "touch_classifier_diagnostic_v1",
                threshold=args.classifier_threshold,
                min_videos=args.min_videos,
                allow_small=args.allow_small_train,
                allow_missing_trajectory=args.allow_missing_trajectory,
                allow_missing_frozen_test=True,
                audio_only=False,
                disable_pose_features=release_disable["disable_pose_features"],
                disable_flow_features=release_disable["disable_flow_features"],
                disable_audio_timbre_features=args.disable_audio_timbre_features,
                labels_dir=labels_dir,
            )
        )

    release_gate = release_gate_status(classifier_summary)
    status = release_gate["status"]
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "corpus_dir": str(corpus_dir),
        "paths": {
            "inventory": str(inventory_path),
            "review_manifest": str(review_manifest_path),
            "candidate_summary": str(candidates_dir / "touch_candidate_summary.json"),
            "label_readiness": str(corpus_dir / "touch_label_readiness.json"),
            "label_readiness_report": str(corpus_dir / "touch_label_readiness.md"),
            "training_dataset_manifest": str(dataset_dir / "touch_training_dataset_manifest.json"),
            "owlv2_detection_manifest": None
            if owlv2_export_summary.get("output_jsonl") is None
            else str(Path(str(owlv2_export_summary["output_jsonl"])).parent / "owlv2_detection_manifest.json"),
            "classifier_metrics": str(corpus_dir / "touch_classifier_v1" / "touch_classifier_metrics.json"),
            "diagnostic_classifier_metrics": str(corpus_dir / "touch_classifier_diagnostic_v1" / "touch_classifier_metrics.json"),
            "audio_feature_manifest": str(dataset_dir / "touch_audio_feature_manifest.json"),
            "flow_feature_manifest": str(dataset_dir / "touch_flow_feature_manifest.json"),
            "pose_feature_manifest": str(dataset_dir / "touch_pose_feature_manifest.json"),
            "foot_track_feature_manifest": str(dataset_dir / "touch_foot_track_feature_manifest.json"),
            "visual_crop_feature_manifest": str(dataset_dir / "touch_visual_crop_feature_manifest.json"),
            "vision_embedding_feature_manifest": str(dataset_dir / "touch_vision_embedding_feature_manifest.json"),
        },
        "corpus": inventory["summary"],
        "review_manifest": review_manifest["summary"],
        "candidates": {
            "videos": candidate_summary["videos"],
            "total_audio_candidates": candidate_summary["total_audio_candidates"],
            "total_existing_event_hints": candidate_summary["total_existing_event_hints"],
            "total_generated_event_hints": candidate_summary.get("total_generated_event_hints", 0),
        },
        "legacy_import": legacy_import_summary,
        "review_subset_prefill": review_subset_prefill_summary,
        "label_readiness": {
            "status": readiness_manifest["status"],
            "complete_ready_videos": readiness_manifest["summary"]["complete_ready_videos"],
            "non_test_complete_ready_videos": readiness_manifest["summary"]["non_test_complete_ready_videos"],
            "frozen_test_complete_ready_videos": readiness_manifest["summary"]["frozen_test_complete_ready_videos"],
            "total_hints": readiness_manifest["summary"]["total_hints"],
            "unchecked_hints": readiness_manifest["summary"]["unchecked_hints"],
            "likely_unchecked_hints": readiness_manifest["summary"]["likely_unchecked_hints"],
            "audio_only_unchecked_hints": readiness_manifest["summary"]["audio_only_unchecked_hints"],
            "candidate_reviews": readiness_manifest["summary"]["candidate_reviews"],
            "minimum_strict_prerequisites_met": readiness_manifest["summary"]["minimum_strict_prerequisites_met"],
            "next_clips": readiness_manifest["next_clips"][:8],
        },
        "training_table": {
            "status": training_table_effective_status(training_manifest, l2_summary),
            "pre_l2_status": training_manifest["status"],
            "labeled_videos": training_manifest["labeled_videos"],
            "skipped_incomplete_video_count": training_manifest["skipped_incomplete_video_count"],
            "candidate_rows": training_manifest["candidate_rows"],
            "positive_candidate_rows": training_manifest["positive_candidate_rows"],
            "candidate_reviews": training_manifest["candidate_reviews"],
            "reviewed_candidate_rows": training_manifest["reviewed_candidate_rows"],
            "negative_candidate_rows_without_review": training_manifest["negative_candidate_rows_without_review"],
            "touches_without_candidate": training_manifest["touches_without_candidate"],
        },
        "owlv2_export": owlv2_export_summary,
        "detection_inputs": {
            "jsonl": [str(path) for path in detection_jsonls],
            "dirs": [str(path) for path in args.detections_dir],
        },
        "l2_features": l2_summary,
        "audio_features": audio_summary,
        "flow_features": flow_summary,
        "pose_features": pose_summary,
        "foot_track_features": foot_track_summary,
        "visual_crop_features": visual_crop_summary,
        "vision_embedding_features": vision_embedding_summary,
        "release_gate": release_gate,
        "classifier": {
            "status": classifier_summary["status"],
            "reasons": classifier_summary.get("reasons", []),
            "feature_mode": classifier_summary["feature_mode"],
            "train_val_rows": classifier_summary["train_val_rows"],
            "test_frozen_rows": classifier_summary["test_frozen_rows"],
            "cv_gate": classifier_summary.get("cv_gate"),
            "frozen_test_gate": classifier_summary.get("frozen_test_gate"),
            "event_level_cv_gate": classifier_summary.get("event_level_cv_gate"),
            "event_level_frozen_test_gate": classifier_summary.get("event_level_frozen_test_gate"),
            "event_level_leave_one_video_out": classifier_summary.get("event_level_leave_one_video_out"),
            "event_level_frozen_test": classifier_summary.get("event_level_frozen_test"),
        },
        "diagnostic_classifier": {
            "status": diagnostic_classifier_summary.get("status"),
            "reason": diagnostic_classifier_summary.get("reason"),
            "feature_mode": diagnostic_classifier_summary.get("feature_mode"),
            "train_val_rows": diagnostic_classifier_summary.get("train_val_rows"),
            "test_frozen_rows": diagnostic_classifier_summary.get("test_frozen_rows"),
            "cv_gate": diagnostic_classifier_summary.get("cv_gate"),
            "frozen_test_gate": diagnostic_classifier_summary.get("frozen_test_gate"),
            "leave_one_video_out_aggregate": (diagnostic_classifier_summary.get("leave_one_video_out") or {}).get("aggregate"),
            "out_of_fold_errors": diagnostic_classifier_summary.get("out_of_fold_errors"),
            "error_jsonl": diagnostic_classifier_summary.get("error_jsonl"),
            "error_csv": diagnostic_classifier_summary.get("error_csv"),
            "report_path": str(corpus_dir / "touch_classifier_diagnostic_v1" / "touch_classifier_report.md")
            if diagnostic_classifier_summary.get("status") not in {None, "skipped"}
            else None,
        },
    }
    write_json(corpus_dir / "touch_pipeline_status.json", summary)
    write_report(corpus_dir / "touch_pipeline_status.md", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-OWLv2 touch pipeline scaffold")
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--candidates-dir", type=Path)
    parser.add_argument("--labels-dir", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--detections-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--detections-dir", type=Path, action="append", default=[])
    parser.add_argument("--export-owlv2-detections", action="store_true", help="run fixed OWLv2 over windows around labeled candidates")
    parser.add_argument("--owlv2-out-dir", type=Path)
    parser.add_argument("--owlv2-model", choices=["owlv2", "owlv2-large"], default="owlv2")
    parser.add_argument("--owlv2-device", default="mps")
    parser.add_argument(
        "--owlv2-prompts",
        nargs="*",
        default=["a footbag", "a hacky sack", "a small ball", "a small round bean bag", "a ball"],
    )
    parser.add_argument("--owlv2-seconds-before", type=float, default=1.0)
    parser.add_argument("--owlv2-seconds-after", type=float, default=1.0)
    parser.add_argument("--owlv2-frame-stride", type=int, default=1)
    parser.add_argument("--owlv2-max-frames", type=int)
    parser.add_argument("--owlv2-dry-run", action="store_true")
    parser.add_argument("--owlv2-force", action="store_true")
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--force-audio", action="store_true")
    parser.add_argument("--include-generated-event-hints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--audio-delta", type=float, default=AUDIO_DELTA)
    parser.add_argument("--audio-wait-sec", type=float, default=AUDIO_WAIT_SEC)
    parser.add_argument("--touch-tolerance-sec", type=float, default=0.20)
    parser.add_argument("--cluster-gap-sec", type=float, default=0.04)
    parser.add_argument("--allow-non-visual-labels", action="store_true")
    parser.add_argument("--allow-incomplete-labels", action="store_true", help="consume draft label files; smoke/debug only")
    parser.add_argument("--import-existing-labels", action="store_true", help="convert legacy data/*.events.json files into completed review labels before table build")
    parser.add_argument("--force-import-existing-labels", action="store_true", help="overwrite existing imported review labels when --import-existing-labels is used")
    parser.add_argument("--prefill-review-subsets", action="store_true", help="prefill incomplete draft labels from older partial reviews/*.review.json files")
    parser.add_argument("--force-prefill-review-subsets", action="store_true", help="overwrite existing draft labels when --prefill-review-subsets is used")
    parser.add_argument("--prefill-review-split", default="test_frozen", help="manifest split to prefill from reviews; use empty string for all splits")
    parser.add_argument("--prefill-conflict-tolerance-sec", type=float, default=0.05)
    parser.add_argument("--reviews-dir", type=Path, default=ROOT / "reviews")
    parser.add_argument("--trajectory-threshold", type=float, default=0.2)
    parser.add_argument("--break-tolerance-sec", type=float, default=0.11)
    parser.add_argument("--max-track-gap-sec", type=float, default=0.25)
    parser.add_argument("--min-points", type=int, default=12)
    parser.add_argument("--attach-audio-features", action="store_true", help="attach short-window audio timbre features from cached WAV files")
    parser.add_argument("--audio-wav-dir", type=Path)
    parser.add_argument("--audio-feature-window-sec", type=float, default=0.16)
    parser.add_argument("--audio-feature-n-fft", type=int, default=1024)
    parser.add_argument("--audio-feature-hop-length", type=int, default=128)
    parser.add_argument("--attach-flow-features", action="store_true", help="attach local optical-flow motion features around the tracked ball")
    parser.add_argument("--flow-ball-tolerance-sec", type=float, default=0.08)
    parser.add_argument("--flow-frame-step", type=int, default=2)
    parser.add_argument("--flow-patch-radius-px", type=int, default=24)
    parser.add_argument("--flow-grid-step-px", type=int, default=6)
    parser.add_argument("--flow-cache-path", type=Path)
    parser.add_argument("--flow-force", action="store_true", help="recompute cached optical-flow feature rows")
    parser.add_argument("--attach-pose-features", action="store_true", help="run RTMW wholebody pose on candidate frames and attach soft body-proximity features")
    parser.add_argument("--pose-mode", choices=["lightweight", "balanced", "performance"], default="performance")
    parser.add_argument("--pose-device", default="cpu")
    parser.add_argument("--pose-keypoint-threshold", type=float, default=0.25)
    parser.add_argument("--pose-ball-tolerance-sec", type=float, default=0.08)
    parser.add_argument("--pose-cache-path", type=Path)
    parser.add_argument("--pose-cache-only", action="store_true", help="attach only pose rows already in the pose cache")
    parser.add_argument("--attach-foot-track-features", action="store_true", help="attach temporal foot identity/continuity features from existing RTMW pose geometry")
    parser.add_argument("--foot-track-window-sec", type=float, default=1.0)
    parser.add_argument("--attach-visual-crop-features", action="store_true", help="attach cached ball-centered visual crop descriptors for contact side/surface classification")
    parser.add_argument("--visual-crop-ball-tolerance-sec", type=float, default=0.08)
    parser.add_argument("--visual-crop-size-px", type=int, default=224)
    parser.add_argument("--visual-crop-grid-size", type=int, default=6)
    parser.add_argument("--visual-crop-cache-path", type=Path)
    parser.add_argument("--attach-vision-embedding-features", action="store_true", help="attach frozen OWLv2 crop embeddings for v1.0 contact classification")
    parser.add_argument("--vision-embedding-ball-tolerance-sec", type=float, default=0.08)
    parser.add_argument("--vision-embedding-crop-size-px", type=int, default=224)
    parser.add_argument("--vision-embedding-batch-size", type=int, default=4)
    parser.add_argument("--vision-embedding-model", choices=["owlv2", "owlv2-large"], default="owlv2")
    parser.add_argument("--vision-embedding-device", default="mps")
    parser.add_argument("--vision-embedding-cache-path", type=Path)
    parser.add_argument("--vision-embedding-contact-labeled-only", action="store_true", help="compute embeddings only for rows with reviewed contact labels")
    parser.add_argument("--classifier-threshold", type=float, default=0.5)
    parser.add_argument("--min-videos", type=int, default=3)
    parser.add_argument("--allow-small-train", action="store_true")
    parser.add_argument("--allow-missing-trajectory", action="store_true")
    parser.add_argument("--allow-missing-frozen-test", action="store_true", help="train without frozen-test rows; smoke only")
    parser.add_argument("--audio-only", action="store_true")
    parser.add_argument("--include-pose-features", action="store_true", help="include pose columns in the release classifier (default: excluded; sparse coverage lowers held-out precision)")
    parser.add_argument("--include-flow-features", action="store_true", help="include optical-flow columns in the release classifier (default: excluded; net-negative on held-out precision)")
    parser.add_argument("--disable-audio-timbre-features", action="store_true", help="when --audio-only is used, use only basic audio columns")
    return parser.parse_args()


def main() -> None:
    summary = run_pipeline(parse_args())
    print(f"status: {summary['status']}")
    print(f"report: {Path(summary['corpus_dir']) / 'touch_pipeline_status.md'}")
    print(
        json.dumps(
            {
                "classifier": summary["classifier"],
                "diagnostic_classifier": summary.get("diagnostic_classifier"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
