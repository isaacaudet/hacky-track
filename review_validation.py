#!/usr/bin/env python3
"""Export reviewed labels and validation counts for Hacky Track."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_REVIEWS = ROOT / "reviews"
DEFAULT_OUT = ROOT / "outputs" / "review_validation"
DEFAULT_BATCH = ROOT / "outputs" / "review_batches" / "latest_review_batch.json"
DEFAULT_BALL_AUDIT = ROOT / "outputs" / "ball_tracking_audit" / "audit_metrics.json"
KINDS = ("touch", "drop_floor", "stall", "around_the_world")


@dataclass
class KindStats:
    approved_candidates: int = 0
    rejected_candidates: int = 0
    manual_missing: int = 0
    pending_candidates: int = 0
    wrong_side: int = 0
    wrong_contact_type: int = 0
    duplicate_rejections: int = 0

    @property
    def precision(self) -> float | None:
        denom = self.approved_candidates + self.rejected_candidates
        if denom == 0:
            return None
        return self.approved_candidates / denom

    @property
    def recall(self) -> float | None:
        denom = self.approved_candidates + self.manual_missing
        if denom == 0:
            return None
        return self.approved_candidates / denom


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def norm_kind(item: dict[str, Any]) -> str:
    kind = str(item.get("kind") or "touch")
    return kind if kind in KINDS else "touch"


def compatible_contact_type(detector_type: Any, reviewed_type: Any) -> bool:
    detector = str(detector_type or "").strip()
    reviewed = str(reviewed_type or "").strip()
    if not detector or not reviewed:
        return True
    if detector == reviewed:
        return True
    compatible = {
        "foot_candidate": "foot",
        "knee_candidate": "knee",
        "ground_candidate": "ground",
    }
    return compatible.get(detector) == reviewed


def reviewed_label_rows(review_path: Path, doc: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in doc.get("items", []):
        rows.append(
            {
                "review_file": str(review_path.relative_to(ROOT)) if review_path.is_relative_to(ROOT) else str(review_path),
                "source_video": doc.get("source_video"),
                "item_id": item.get("id"),
                "source": item.get("source", "candidate"),
                "kind": norm_kind(item),
                "status": item.get("status", "pending"),
                "time_sec": item.get("time_sec"),
                "start_sec": item.get("start_sec"),
                "end_sec": item.get("end_sec"),
                "duration_sec": item.get("duration_sec"),
                "rally_id": item.get("rally_id"),
                "touch_number": item.get("touch_number"),
                "contact_side": item.get("contact_side", "unknown"),
                "detector_contact_side": item.get("detector_contact_side"),
                "side_confidence": item.get("side_confidence"),
                "side_source": item.get("side_source"),
                "side_uncertainty_reason": item.get("side_uncertainty_reason"),
                "contact_type": item.get("contact_type", "unknown"),
                "detector_contact_type": item.get("detector_contact_type"),
                "trick_label": item.get("trick_label", ""),
                "x": item.get("x"),
                "y": item.get("y"),
                "confidence": item.get("confidence"),
                "review_tags": item.get("review_tags", []),
                "note": item.get("note", ""),
            }
        )
    return rows


def review_key(source_video: Any, item_id: Any) -> str:
    return f"{source_video or ''}::{item_id or ''}"


def review_batch_coverage(batch_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not batch_path.exists():
        return None
    batch = read_json(batch_path)
    lookup = {review_key(row.get("source_video"), row.get("item_id")): row for row in rows if row.get("item_id")}
    status_counts: dict[str, int] = defaultdict(int)
    kind_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tag_counts: dict[str, int] = defaultdict(int)
    video_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    items_out: list[dict[str, Any]] = []
    for item in batch.get("items", []):
        key = review_key(item.get("source_video"), item.get("review_item_id"))
        row = lookup.get(key)
        if row is None:
            status = "unreviewed"
        else:
            status = str(row.get("status") or "pending")
        kind = str(item.get("kind") or "touch")
        status_counts[status] += 1
        kind_counts[kind][status] += 1
        video_counts[str(item.get("source_video") or "")][status] += 1
        for tag in item.get("review_tags", []):
            tag_counts[str(tag)] += 1
        items_out.append(
            {
                "batch_item_id": item.get("batch_item_id"),
                "review_item_id": item.get("review_item_id"),
                "source_video": item.get("source_video"),
                "kind": kind,
                "time_sec": item.get("time_sec"),
                "status": status,
                "review_tags": item.get("review_tags", []),
            }
        )

    total = len(batch.get("items", []))
    decided = sum(status_counts.get(status, 0) for status in ("approved", "rejected", "missing"))
    present = total - status_counts.get("unreviewed", 0)
    kind_metrics: dict[str, dict[str, Any]] = {}
    for kind, counts in sorted(kind_counts.items()):
        approved = int(counts.get("approved", 0))
        rejected = int(counts.get("rejected", 0))
        missing = int(counts.get("missing", 0))
        denom = approved + rejected
        recall_denom = approved + missing
        kind_metrics[kind] = {
            "approved": approved,
            "rejected": rejected,
            "missing": missing,
            "pending": int(counts.get("pending", 0)),
            "unreviewed": int(counts.get("unreviewed", 0)),
            "precision_on_current_batch": None if denom == 0 else round(approved / denom, 4),
            "recall_on_current_batch": None if recall_denom == 0 else round(approved / recall_denom, 4),
        }

    return {
        "batch_id": batch.get("batch_id"),
        "batch_path": str(batch_path.relative_to(ROOT)) if batch_path.is_relative_to(ROOT) else str(batch_path),
        "contact_sheet_path": batch.get("contact_sheet_path"),
        "total_items": total,
        "decided_items": decided,
        "present_in_review_files": present,
        "pending_items": status_counts.get("pending", 0),
        "unreviewed_items": status_counts.get("unreviewed", 0),
        "decision_coverage": None if total == 0 else round(decided / total, 4),
        "status_counts": dict(sorted(status_counts.items())),
        "kind_status_counts": {kind: dict(counts) for kind, counts in sorted(kind_counts.items())},
        "kind_metrics": kind_metrics,
        "top_tags": dict(sorted(tag_counts.items(), key=lambda pair: (-pair[1], pair[0]))[:20]),
        "video_status_counts": {video: dict(counts) for video, counts in sorted(video_counts.items())},
        "items": items_out,
    }


def ball_audit_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    audit = read_json(path)
    return {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "report_path": audit.get("report_path"),
        "sampled_events": audit.get("sampled_events"),
        "total_high_confidence_events": audit.get("total_high_confidence_events"),
        "pass_rate": audit.get("pass_rate"),
        "target_pass_rate": audit.get("target_pass_rate"),
        "target_met": audit.get("target_met"),
        "unsupported_confident_events": audit.get("unsupported_confident_events"),
        "status_counts": audit.get("status_counts", {}),
        "correction_counts": audit.get("correction_counts", {}),
        "kind_status_counts": audit.get("kind_status_counts", {}),
    }


def analyze_reviews(review_paths: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_kind: dict[str, KindStats] = {kind: KindStats() for kind in KINDS}
    by_video: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    rows: list[dict[str, Any]] = []
    confusion = {
        "missed_touch": 0,
        "false_touch": 0,
        "missed_drop": 0,
        "false_drop": 0,
        "duplicate_stall_or_touch": 0,
        "wrong_side": 0,
        "wrong_contact_type": 0,
    }

    for review_path in review_paths:
        doc = read_json(review_path)
        video = str(doc.get("source_video") or review_path.stem)
        label_rows = reviewed_label_rows(review_path, doc)
        rows.extend(label_rows)
        for item in doc.get("items", []):
            kind = norm_kind(item)
            source = str(item.get("source") or "candidate")
            status = str(item.get("status") or "pending")
            stats = by_kind[kind]
            by_video[video]["items"] += 1
            by_video[video][f"{kind}_{status}"] += 1
            if source == "candidate" and status == "approved":
                stats.approved_candidates += 1
                by_video[video]["approved_candidates"] += 1
            elif source == "candidate" and status == "rejected":
                stats.rejected_candidates += 1
                by_video[video]["rejected_candidates"] += 1
                if kind == "touch":
                    confusion["false_touch"] += 1
                if kind == "drop_floor":
                    confusion["false_drop"] += 1
                note = str(item.get("note") or "").lower()
                if kind in {"touch", "stall"} and "duplicate" in note:
                    stats.duplicate_rejections += 1
                    confusion["duplicate_stall_or_touch"] += 1
            elif source == "candidate" and status == "pending":
                stats.pending_candidates += 1
                by_video[video]["pending_candidates"] += 1
            elif source == "manual" and status == "missing":
                stats.manual_missing += 1
                by_video[video]["manual_missing"] += 1
                if kind == "touch":
                    confusion["missed_touch"] += 1
                if kind == "drop_floor":
                    confusion["missed_drop"] += 1

            detector_side = item.get("detector_contact_side")
            if status in {"approved", "missing"} and detector_side and item.get("contact_side") and detector_side != item.get("contact_side"):
                stats.wrong_side += 1
                confusion["wrong_side"] += 1
            detector_type = item.get("detector_contact_type")
            if (
                status in {"approved", "missing"}
                and detector_type
                and item.get("contact_type")
                and not compatible_contact_type(detector_type, item.get("contact_type"))
            ):
                stats.wrong_contact_type += 1
                confusion["wrong_contact_type"] += 1

    metrics = {
        "review_files": len(review_paths),
        "reviewed_items": len(rows),
        "kinds": {
            kind: {
                "approved_candidates": stats.approved_candidates,
                "rejected_candidates": stats.rejected_candidates,
                "manual_missing": stats.manual_missing,
                "pending_candidates": stats.pending_candidates,
                "precision_on_reviewed_candidates": None if stats.precision is None else round(stats.precision, 4),
                "recall_against_reviewed_labels": None if stats.recall is None else round(stats.recall, 4),
                "wrong_side": stats.wrong_side,
                "wrong_contact_type": stats.wrong_contact_type,
                "duplicate_rejections": stats.duplicate_rejections,
            }
            for kind, stats in by_kind.items()
        },
        "confusion": confusion,
        "videos": {video: dict(counts) for video, counts in sorted(by_video.items())},
    }
    return metrics, rows


def write_report(metrics: dict[str, Any], path: Path) -> None:
    lines = [
        "# Review Validation Report",
        "",
        f"- Review files: {metrics['review_files']}",
        f"- Reviewed items: {metrics['reviewed_items']}",
        "",
        "## Kind Metrics",
        "",
        "| Kind | Approved candidates | Rejected candidates | Manual missing | Pending | Precision | Recall | Wrong side | Wrong type | Duplicate rejects |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kind, stats in metrics["kinds"].items():
        lines.append(
            f"| `{kind}` | {stats['approved_candidates']} | {stats['rejected_candidates']} | "
            f"{stats['manual_missing']} | {stats['pending_candidates']} | "
            f"{pct(stats['precision_on_reviewed_candidates'])} | {pct(stats['recall_against_reviewed_labels'])} | "
            f"{stats['wrong_side']} | {stats['wrong_contact_type']} | {stats['duplicate_rejections']} |"
        )
    lines.extend(["", "## Confusion Counts", ""])
    for key, value in metrics["confusion"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Reviewed Videos", ""])
    lines.extend(["| Video | Items | Approved cand | Rejected cand | Manual missing | Pending cand |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for video, counts in metrics["videos"].items():
        lines.append(
            f"| `{video}` | {counts.get('items', 0)} | {counts.get('approved_candidates', 0)} | "
            f"{counts.get('rejected_candidates', 0)} | {counts.get('manual_missing', 0)} | {counts.get('pending_candidates', 0)} |"
        )
    batch = metrics.get("review_batch")
    if batch:
        lines.extend(
            [
                "",
                "## Review Batch Coverage",
                "",
                f"- Batch: `{batch['batch_id']}`",
                f"- Contact sheet: `{batch.get('contact_sheet_path')}`",
                f"- Items: {batch['total_items']}",
                f"- Present in review files: {batch['present_in_review_files']}",
                f"- Decided: {batch['decided_items']} ({pct(batch['decision_coverage'])})",
                f"- Pending: {batch['pending_items']}",
                f"- Unreviewed: {batch['unreviewed_items']}",
                "",
                "| Kind | Approved | Rejected | Missing | Pending | Unreviewed | Current precision | Current recall |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for kind, counts in batch["kind_metrics"].items():
            lines.append(
                f"| `{kind}` | {counts.get('approved', 0)} | {counts.get('rejected', 0)} | "
                f"{counts.get('missing', 0)} | {counts.get('pending', 0)} | {counts.get('unreviewed', 0)} | "
                f"{pct(counts.get('precision_on_current_batch'))} | {pct(counts.get('recall_on_current_batch'))} |"
            )
        lines.extend(["", "Top batch tags:"])
        for tag, count in list(batch.get("top_tags", {}).items())[:12]:
            lines.append(f"- {tag}: {count}")
    ball_audit = metrics.get("ball_tracking_audit")
    if ball_audit:
        lines.extend(
            [
                "",
                "## Ball Tracking Audit",
                "",
                f"- Audit metrics: `{ball_audit.get('path')}`",
                f"- Audit report: `{ball_audit.get('report_path')}`",
                f"- Sampled events: {ball_audit.get('sampled_events')} of {ball_audit.get('total_high_confidence_events')}",
                f"- Pass rate: {pct(ball_audit.get('pass_rate'))}",
                f"- Target pass rate: {pct(ball_audit.get('target_pass_rate'))}",
                f"- Target met: {ball_audit.get('target_met')}",
                f"- Unsupported confident events: {ball_audit.get('unsupported_confident_events')}",
                "",
                "| Kind | Pass | Uncertain | Fail |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for kind, counts in ball_audit.get("kind_status_counts", {}).items():
            lines.append(f"| `{kind}` | {counts.get('pass', 0)} | {counts.get('uncertain', 0)} | {counts.get('fail', 0)} |")
        lines.extend(["", "Correction counts:"])
        for key, value in ball_audit.get("correction_counts", {}).items():
            lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Precision is approved candidate / reviewed candidate decisions for that kind.",
            "- Recall is approved candidate / (approved candidate + manual missing labels) for that kind.",
            "- Pending candidates are intentionally excluded from precision/recall denominators.",
            "- This report is only as strong as the manually reviewed subset currently present under `reviews/`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Hacky Track review labels and validation metrics")
    parser.add_argument("--reviews-dir", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--ball-audit", type=Path, default=DEFAULT_BALL_AUDIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review_paths = sorted(args.reviews_dir.glob("*.review.json"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics, rows = analyze_reviews(review_paths)
    batch = review_batch_coverage(args.batch, rows)
    if batch is not None:
        metrics["review_batch"] = batch
    audit = ball_audit_summary(args.ball_audit)
    if audit is not None:
        metrics["ball_tracking_audit"] = audit
    metrics_path = args.out_dir / "validation_metrics.json"
    labels_path = args.out_dir / "reviewed_labels.jsonl"
    report_path = args.out_dir / "validation_report.md"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with labels_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    write_report(metrics, report_path)
    print(f"metrics: {metrics_path}")
    print(f"labels: {labels_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
