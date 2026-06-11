#!/usr/bin/env python3
"""Generate non-authoritative review suggestions for the priority batch."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_BATCH = ROOT / "outputs" / "review_batches" / "latest_review_batch.json"
DEFAULT_AUDIT_RESULTS = ROOT / "outputs" / "ball_tracking_audit" / "audit_results.jsonl"
DEFAULT_OUT_DIR = ROOT / "outputs" / "review_batches"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def maybe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def audit_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        source = str(row.get("source_video") or "")
        event_index = row.get("event_index")
        if source and event_index is not None:
            out[(source, int(event_index))] = row
    return out


def item_tags(item: dict[str, Any]) -> set[str]:
    return {str(tag) for tag in item.get("review_tags", [])}


def classify(item: dict[str, Any], audit: dict[str, Any] | None) -> tuple[str, str, float, list[str]]:
    kind = str(item.get("kind") or "")
    tags = item_tags(item)
    confidence = maybe_float(item.get("confidence")) or 0.0
    ball_conf = maybe_float(item.get("qa_ball_confidence")) or 0.0
    foot_distance = maybe_float(item.get("foot_distance"))
    y = maybe_float(item.get("qa_ball_y"))
    duration = maybe_float(item.get("duration_sec")) or 0.0
    contact_type = str(item.get("contact_type") or "")
    audit_status = str((audit or {}).get("audit_status") or "not_sampled")
    score = 0.0
    reasons: list[str] = []

    if audit_status == "pass":
        score += 1.4
        reasons.append("independent_ball_audit_pass")
    elif audit_status == "uncertain":
        score -= 0.8
        reasons.append("independent_ball_audit_uncertain")
    elif audit_status == "fail":
        score -= 2.0
        reasons.append("independent_ball_audit_fail")
    else:
        reasons.append("not_in_high_confidence_ball_audit")

    if item.get("qa_ball_accuracy") == "high":
        score += 0.7
        reasons.append("high_ball_accuracy")
    elif item.get("qa_ball_accuracy") == "medium":
        score += 0.25
        reasons.append("medium_ball_accuracy")
    else:
        score -= 0.6
        reasons.append("low_or_missing_ball_accuracy")

    if confidence >= 0.85:
        score += 0.5
        reasons.append("high_detector_confidence")
    elif confidence < 0.58:
        score -= 0.45
        reasons.append("low_detector_confidence")

    if ball_conf >= 0.82:
        score += 0.25

    if kind == "touch":
        if contact_type == "foot":
            score += 0.65
            reasons.append("foot_contact_label")
        elif contact_type in {"foot_candidate", "knee_candidate", "unknown_contact", "unknown"}:
            score -= 0.55
            reasons.append("ambiguous_touch_contact")
        if foot_distance is not None and foot_distance <= 110:
            score += 0.35
            reasons.append("close_to_foot")
        elif foot_distance is not None and foot_distance >= 155:
            score -= 0.55
            reasons.append("far_from_foot")
        if "audio_track_context" in tags and (maybe_float(item.get("visual_score")) or 0.0) <= 0.05:
            score -= 0.35
            reasons.append("audio_track_context_without_visual_score")
        if "floor_risk_touch" in tags and y is not None and y >= 860:
            score -= 0.35
            reasons.append("near_floor_touch_risk")
        if score >= 2.5:
            return "likely_approve", "approve after quick video check", round(score, 3), reasons
        if score <= 0.25:
            return "likely_reject", "reject if video confirms no foot contact", round(score, 3), reasons
        return "needs_review", "inspect touch timing and foot contact", round(score, 3), reasons

    if kind == "drop_floor":
        if contact_type == "ground":
            score += 0.8
            reasons.append("ground_contact_label")
        if y is not None and y >= 815:
            score += 0.55
            reasons.append("low_frame_position")
        if str(item.get("drop_source") or "").startswith("visual_floor"):
            score += 0.35
            reasons.append("visual_floor_drop_source")
        if score >= 2.45:
            return "likely_approve", "approve if sack is visibly on floor/reset", round(score, 3), reasons
        return "needs_review", "inspect reset boundary and floor contact", round(score, 3), reasons

    if kind == "stall":
        if contact_type == "stall":
            score += 0.9
            reasons.append("stall_contact_label")
        if duration >= 0.18:
            score += 0.35
            reasons.append("nonzero_stall_window")
        else:
            score -= 0.45
            reasons.append("short_stall_window")
        if score >= 2.55:
            return "likely_approve", "approve if held contact is visible across window", round(score, 3), reasons
        return "needs_review", "inspect start/end of stall window", round(score, 3), reasons

    if kind == "around_the_world":
        if confidence >= 0.86:
            score += 0.5
            reasons.append("high_trick_candidate_confidence")
        return "needs_review", "verify leg circles around sack and assign exact trick label", round(score, 3), reasons

    return "needs_review", "unknown event kind", round(score, 3), reasons


def build_suggestions(batch_path: Path, audit_results_path: Path) -> dict[str, Any]:
    batch = read_json(batch_path)
    audit_by_event = audit_lookup(read_jsonl(audit_results_path))
    suggestions: list[dict[str, Any]] = []
    for item in batch.get("items", []):
        event_index = maybe_int(item.get("event_index"))
        audit = None if event_index is None else audit_by_event.get((str(item.get("source_video") or ""), event_index))
        bucket, recommended_action, score, reasons = classify(item, audit)
        suggestions.append(
            {
                "batch_item_id": item.get("batch_item_id"),
                "review_item_id": item.get("review_item_id"),
                "source_video": item.get("source_video"),
                "review_stem": item.get("review_stem"),
                "kind": item.get("kind"),
                "time_sec": item.get("time_sec"),
                "bucket": bucket,
                "recommended_action": recommended_action,
                "evidence_score": score,
                "reasons": reasons,
                "audit_status": None if audit is None else audit.get("audit_status"),
                "contact_side": item.get("contact_side"),
                "contact_type": item.get("contact_type"),
                "confidence": item.get("confidence"),
                "qa_ball_accuracy": item.get("qa_ball_accuracy"),
                "crop_path": item.get("crop_path"),
                "tile_path": item.get("tile_path"),
                "review_tags": item.get("review_tags", []),
            }
        )
    return {
        "batch_path": str(batch_path),
        "audit_results_path": str(audit_results_path),
        "batch_id": batch.get("batch_id"),
        "items": suggestions,
        "summary": {
            "total": len(suggestions),
            "bucket_counts": dict(Counter(item["bucket"] for item in suggestions)),
            "kind_bucket_counts": {
                kind: dict(Counter(item["bucket"] for item in suggestions if item["kind"] == kind))
                for kind in sorted({str(item["kind"]) for item in suggestions})
            },
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Assisted Review Suggestions",
        "",
        "These are non-authoritative suggestions. They are meant to speed review, not to replace approve/reject/missing decisions.",
        "",
        f"- Batch: `{payload['batch_id']}`",
        f"- Items: {summary['total']}",
        f"- Bucket counts: {', '.join(f'{key}={value}' for key, value in sorted(summary['bucket_counts'].items()))}",
        "",
        "## Kind Buckets",
        "",
        "| Kind | likely_approve | likely_reject | needs_review |",
        "| --- | ---: | ---: | ---: |",
    ]
    for kind, counts in summary["kind_bucket_counts"].items():
        lines.append(
            f"| `{kind}` | {counts.get('likely_approve', 0)} | {counts.get('likely_reject', 0)} | {counts.get('needs_review', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Suggested Review Order",
            "",
            "| Item | Kind | Time | Bucket | Action | Evidence | Crop |",
            "| --- | --- | ---: | --- | --- | ---: | --- |",
        ]
    )
    ordered = sorted(
        payload["items"],
        key=lambda item: (
            {"likely_reject": 0, "likely_approve": 1, "needs_review": 2}.get(str(item["bucket"]), 9),
            -float(item.get("evidence_score") or 0.0),
            str(item.get("source_video") or ""),
            float(item.get("time_sec") or 0.0),
        ),
    )
    for item in ordered:
        lines.append(
            f"| `{item['batch_item_id']}` | `{item['kind']}` | {float(item.get('time_sec') or 0):.3f} | "
            f"{item['bucket']} | {item['recommended_action']} | {item['evidence_score']:.3f} | `{item.get('crop_path')}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate assisted review suggestions for the priority batch")
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--audit-results", type=Path, default=DEFAULT_AUDIT_RESULTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_suggestions(args.batch, args.audit_results)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "assisted_review_suggestions.json"
    jsonl_path = args.out_dir / "assisted_review_suggestions.jsonl"
    report_path = args.out_dir / "assisted_review_suggestions.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_jsonl(jsonl_path, payload["items"])
    write_report(report_path, payload)
    print(f"suggestions: {json_path}")
    print(f"jsonl: {jsonl_path}")
    print(f"report: {report_path}")
    print(f"bucket_counts: {payload['summary']['bucket_counts']}")


if __name__ == "__main__":
    main()
