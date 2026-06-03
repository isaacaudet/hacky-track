#!/usr/bin/env python3
"""Evaluate precision-gate shadow impact on all reviewed release labels.

The gate is not applied to release outputs here. This script answers the
counterfactual: if a gate-vetoed candidate were suppressed, what would happen to
precision and recall by split, video, and kind?
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from release_evaluation import DECIDED_STATUSES, KINDS, safe_div


ROOT = Path(__file__).resolve().parent
DEFAULT_LABELS = ROOT / "outputs/release_evaluation_27/reviewed_labels.release.jsonl"
DEFAULT_BENCHMARK = ROOT / "outputs/candidate_benchmark/candidate_benchmark.jsonl"
DEFAULT_OUT_DIR = ROOT / "outputs/precision_gate_shadow_eval"
MATCH_TOL_SEC = 0.20
PREFERRED_GATE_SOURCES = ("qa", "qa_suppressed", "trained", "calibrated")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def norm_kind(value: Any) -> str:
    kind = str(value or "touch")
    if kind in {"drop", "floor", "floor_reset"}:
        return "drop_floor"
    return kind


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("source_video") or row.get("video") or ""), str(row.get("item_id") or ""), norm_kind(row.get("kind")))


def gate_source_rank(source: Any) -> int:
    try:
        return PREFERRED_GATE_SOURCES.index(str(source))
    except ValueError:
        return len(PREFERRED_GATE_SOURCES)


def build_gate_index(benchmark_rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, str, str], list[dict[str, Any]]], dict[tuple[str, str], list[dict[str, Any]]]]:
    by_item: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_video_kind: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in benchmark_rows:
        video = str(row.get("video") or row.get("source_video") or "")
        kind = norm_kind(row.get("kind"))
        closest_id = str(row.get("closest_review_item_id") or "")
        if closest_id:
            by_item[(video, closest_id, kind)].append(row)
        by_video_kind[(video, kind)].append(row)
    for rows in by_item.values():
        rows.sort(
            key=lambda row: (
                gate_source_rank(row.get("source")),
                not bool(row.get("precision_gate_hard_veto", row.get("precision_gate_veto"))),
                not bool(row.get("precision_gate_soft_flag")),
                -float(row.get("precision_gate_score") or 0),
            )
        )
    for rows in by_video_kind.values():
        rows.sort(
            key=lambda row: (
                gate_source_rank(row.get("source")),
                not bool(row.get("precision_gate_hard_veto", row.get("precision_gate_veto"))),
                not bool(row.get("precision_gate_soft_flag")),
                -float(row.get("precision_gate_score") or 0),
            )
        )
    return by_item, by_video_kind


def match_gate_row(
    label: dict[str, Any],
    by_item: dict[tuple[str, str, str], list[dict[str, Any]]],
    by_video_kind: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[dict[str, Any] | None, str]:
    video, item_id, kind = row_key(label)
    exact = by_item.get((video, item_id, kind), [])
    if exact:
        return exact[0], "item_id"

    label_time = as_float(label.get("time_sec"))
    if label_time is None:
        return None, ""
    candidates: list[tuple[float, dict[str, Any]]] = []
    for row in by_video_kind.get((video, kind), []):
        row_time = as_float(row.get("time_sec"))
        if row_time is None:
            continue
        dt = abs(row_time - label_time)
        if dt <= MATCH_TOL_SEC:
            candidates.append((dt, row))
    if not candidates:
        return None, ""
    candidates.sort(
        key=lambda item: (
            item[0],
            gate_source_rank(item[1].get("source")),
            not bool(item[1].get("precision_gate_hard_veto", item[1].get("precision_gate_veto"))),
            not bool(item[1].get("precision_gate_soft_flag")),
        )
    )
    return candidates[0][1], "time"


def enrich_labels(labels: list[dict[str, Any]], benchmark_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_item, by_video_kind = build_gate_index(benchmark_rows)
    enriched: list[dict[str, Any]] = []
    for label in labels:
        row = dict(label)
        gate_row, match_method = match_gate_row(row, by_item, by_video_kind)
        gate_veto = bool(gate_row and gate_row.get("precision_gate_hard_veto", gate_row.get("precision_gate_veto")))
        row["precision_gate_match_method"] = match_method
        row["precision_gate_candidate_source"] = None if gate_row is None else gate_row.get("source")
        row["precision_gate_candidate_time_sec"] = None if gate_row is None else gate_row.get("time_sec")
        row["precision_gate_veto"] = gate_veto
        row["precision_gate_hard_veto"] = gate_veto
        row["precision_gate_soft_flag"] = bool(gate_row and gate_row.get("precision_gate_soft_flag"))
        row["precision_gate_score"] = None if gate_row is None else gate_row.get("precision_gate_score")
        row["precision_gate_reasons"] = "" if gate_row is None else str(gate_row.get("precision_gate_reasons") or "")
        row["precision_gate_hard_reasons"] = "" if gate_row is None else str(gate_row.get("precision_gate_hard_reasons") or "")
        row["precision_gate_soft_reasons"] = "" if gate_row is None else str(gate_row.get("precision_gate_soft_reasons") or "")
        row["precision_gate_tracker_disagreement_px"] = None if gate_row is None else gate_row.get("precision_gate_tracker_disagreement_px")
        row["precision_gate_y_ratio"] = None if gate_row is None else gate_row.get("precision_gate_y_ratio")
        row["precision_gate_effective_status"] = effective_status(row)
        row["precision_gate_changed"] = row["precision_gate_effective_status"] != row.get("status")
        enriched.append(row)
    return enriched


def effective_status(row: dict[str, Any]) -> str:
    if row.get("source") != "candidate":
        return str(row.get("status") or "")
    if not row.get("precision_gate_veto"):
        return str(row.get("status") or "")
    status = str(row.get("status") or "")
    if status == "approved":
        return "vetoed_approved"
    if status == "rejected":
        return "vetoed_rejected"
    return status


def selected_rows(rows: list[dict[str, Any]], *, split: str | None = None) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("label_tier") == "clean_current_batch"
        and row.get("status") in DECIDED_STATUSES
        and (split is None or row.get("split") == split)
    ]


def mark_conflict_labels(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["near_time_conflict"] = False
        row["near_time_conflict_statuses"] = ""
        row["near_time_conflict_item_ids"] = ""
    for group in conflict_groups(rows):
        ids = set(str(group["item_ids"]).split(";"))
        for row in rows:
            if str(row.get("source_video") or "") != group["video"] or norm_kind(row.get("kind")) != group["kind"]:
                continue
            if str(row.get("item_id") or "") in ids:
                row["near_time_conflict"] = True
                row["near_time_conflict_statuses"] = group["statuses"]
                row["near_time_conflict_item_ids"] = group["item_ids"]


def kind_metrics(rows: list[dict[str, Any]], *, shadow: bool) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for kind in KINDS:
        kind_rows = [row for row in rows if norm_kind(row.get("kind")) == kind]
        approved = 0
        rejected = 0
        manual_missing = 0
        approved_vetoed = 0
        rejected_vetoed = 0
        for row in kind_rows:
            source = str(row.get("source") or "")
            status = str(row.get("status") or "")
            effective = str(row.get("precision_gate_effective_status") or status) if shadow else status
            if source == "manual" and status == "missing":
                manual_missing += 1
                continue
            if source != "candidate":
                continue
            if not shadow:
                if status == "approved":
                    approved += 1
                elif status == "rejected":
                    rejected += 1
                continue
            if effective == "approved":
                approved += 1
            elif effective == "rejected":
                rejected += 1
            elif effective == "vetoed_approved":
                approved_vetoed += 1
            elif effective == "vetoed_rejected":
                rejected_vetoed += 1

        recall_denominator = approved + approved_vetoed + manual_missing
        metrics[kind] = {
            "approved_candidates": approved,
            "rejected_candidates": rejected,
            "manual_missing": manual_missing,
            "approved_vetoed": approved_vetoed,
            "rejected_vetoed": rejected_vetoed,
            "reviewed_decisions": len(kind_rows),
            "precision": safe_div(approved, approved + rejected),
            "recall": safe_div(approved, recall_denominator),
        }
    return metrics


def metric_block(rows: list[dict[str, Any]], *, split: str | None = None, exclude_conflicts: bool = False) -> dict[str, Any]:
    base_rows = selected_rows(rows, split=split)
    if exclude_conflicts:
        base_rows = [row for row in base_rows if not row.get("near_time_conflict")]
    before = kind_metrics(base_rows, shadow=False)
    after = kind_metrics(base_rows, shadow=True)
    return {
        "items": len(base_rows),
        "before": before,
        "after_shadow": after,
        "changed": {
            "total": sum(1 for row in base_rows if row.get("precision_gate_changed")),
            "approved_to_veto": sum(1 for row in base_rows if row.get("precision_gate_effective_status") == "vetoed_approved"),
            "rejected_to_veto": sum(1 for row in base_rows if row.get("precision_gate_effective_status") == "vetoed_rejected"),
        },
    }


def per_video_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows(rows):
        by_video[str(row.get("source_video") or "")].append(row)
    out: dict[str, Any] = {}
    for video, group in sorted(by_video.items()):
        split = str(group[0].get("split") or "") if group else ""
        out[video] = {
            "split": split,
            "items": len(group),
            "before": kind_metrics(group, shadow=False),
            "after_shadow": kind_metrics(group, shadow=True),
            "changed": {
                "total": sum(1 for row in group if row.get("precision_gate_changed")),
                "approved_to_veto": sum(1 for row in group if row.get("precision_gate_effective_status") == "vetoed_approved"),
                "rejected_to_veto": sum(1 for row in group if row.get("precision_gate_effective_status") == "vetoed_rejected"),
            },
        }
    return out


def conflict_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean = [row for row in rows if row.get("label_tier") == "clean_current_batch" and row.get("status") in DECIDED_STATUSES]
    groups: list[dict[str, Any]] = []
    by_video_kind: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in clean:
        by_video_kind[(str(row.get("source_video") or ""), norm_kind(row.get("kind")))].append(row)

    for (video, kind), items in by_video_kind.items():
        items = sorted(items, key=lambda row: as_float(row.get("time_sec")) or -1.0)
        idx = 0
        while idx < len(items):
            row = items[idx]
            t = as_float(row.get("time_sec"))
            if t is None:
                idx += 1
                continue
            cluster = [row]
            idx += 1
            while idx < len(items):
                other_time = as_float(items[idx].get("time_sec"))
                if other_time is None or abs(float(other_time) - t) > MATCH_TOL_SEC:
                    break
                cluster.append(items[idx])
                idx += 1
            if len(cluster) == 1:
                continue
            statuses = sorted({str(item.get("status") or "") for item in cluster})
            item_ids = sorted({str(item.get("item_id") or "") for item in cluster})
            if len(statuses) <= 1 and len(item_ids) == len(cluster):
                continue
            groups.append(
                {
                    "video": video,
                    "kind": kind,
                    "time_sec": round(t, 3),
                    "statuses": ";".join(statuses),
                    "item_ids": ";".join(item_ids),
                    "notes": " | ".join(str(item.get("note") or "")[:160] for item in cluster),
                }
            )
    return groups


def changed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "source_video",
        "split",
        "time_sec",
        "kind",
        "item_id",
        "source",
        "status",
        "precision_gate_effective_status",
        "precision_gate_candidate_source",
        "precision_gate_match_method",
        "precision_gate_score",
        "precision_gate_reasons",
        "precision_gate_hard_reasons",
        "precision_gate_soft_reasons",
        "near_time_conflict",
        "near_time_conflict_statuses",
        "near_time_conflict_item_ids",
        "note",
    ]
    changed = [row for row in selected_rows(rows) if row.get("precision_gate_changed")]
    changed.sort(key=lambda row: (str(row.get("split")), str(row.get("source_video")), float(row.get("time_sec") or 0.0), str(row.get("item_id"))))
    return [{field: row.get(field) for field in fields} for row in changed]


def write_gated_reviews(labels: list[dict[str, Any]], reviews_dir: Path, out_dir: Path, *, exclude_conflicts: bool = False) -> dict[str, Any]:
    by_item = {
        (
            str(row.get("source_video") or ""),
            str(row.get("item_id") or ""),
            norm_kind(row.get("kind")),
        ): row
        for row in labels
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    changed = 0
    conflict_excluded = 0
    copied = 0
    for review_path in sorted(reviews_dir.glob("*.review.json")):
        doc = json.loads(review_path.read_text(encoding="utf-8"))
        source_video = str(doc.get("source_video") or review_path.stem)
        out_doc = dict(doc)
        out_items: list[dict[str, Any]] = []
        for item in doc.get("items", []):
            out_item = dict(item)
            key = (source_video, str(item.get("id") or ""), norm_kind(item.get("kind")))
            gate_row = by_item.get(key)
            if gate_row:
                out_item["precision_gate_hard_veto"] = bool(gate_row.get("precision_gate_hard_veto"))
                out_item["precision_gate_soft_flag"] = bool(gate_row.get("precision_gate_soft_flag"))
                out_item["precision_gate_score"] = gate_row.get("precision_gate_score")
                out_item["precision_gate_reasons"] = gate_row.get("precision_gate_reasons")
                out_item["precision_gate_hard_reasons"] = gate_row.get("precision_gate_hard_reasons")
                out_item["precision_gate_soft_reasons"] = gate_row.get("precision_gate_soft_reasons")
                out_item["near_time_conflict"] = bool(gate_row.get("near_time_conflict"))
                out_item["near_time_conflict_statuses"] = gate_row.get("near_time_conflict_statuses")
                out_item["near_time_conflict_item_ids"] = gate_row.get("near_time_conflict_item_ids")
                is_conflict_excluded = bool(exclude_conflicts and gate_row.get("near_time_conflict"))
                if is_conflict_excluded:
                    out_item["source"] = "label_conflict_excluded"
                    out_item["label_conflict_excluded"] = True
                    out_item["pre_conflict_source"] = item.get("source", "candidate")
                    conflict_excluded += 1
                elif gate_row.get("precision_gate_effective_status") == "vetoed_rejected":
                    out_item["source"] = "precision_gate_suppressed"
                    out_item["precision_gate_suppressed"] = True
                    out_item["pre_gate_source"] = item.get("source", "candidate")
                    changed += 1
            out_items.append(out_item)
        out_doc["items"] = out_items
        out_doc["precision_gate_applied"] = True
        out_doc["precision_gate_policy"] = "hard-vetoed rejected candidates are audit-only"
        (out_dir / review_path.name).write_text(json.dumps(out_doc, indent=2) + "\n", encoding="utf-8")
        copied += 1
    return {
        "review_files": copied,
        "precision_gate_suppressed_items": changed,
        "label_conflict_excluded_items": conflict_excluded,
        "path": str(out_dir),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def write_report(path: Path, doc: dict[str, Any]) -> None:
    lines = [
        "# Precision Gate Shadow Evaluation",
        "",
        "This is a counterfactual report. The precision gate did not modify release labels or QA output.",
        "",
        "## Split Metrics",
        "",
        "| Split | Items | Kind | Before P | After P | Before R | After R | Approved vetoed | Rejected vetoed |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("train", "validation", "test", "all"):
        block = doc["splits"][split]
        for kind in ("touch", "drop_floor", "stall", "around_the_world"):
            before = block["before"][kind]
            after = block["after_shadow"][kind]
            lines.append(
                f"| {split} | {block['items']} | `{kind}` | {percent(before['precision'])} | {percent(after['precision'])} | "
                f"{percent(before['recall'])} | {percent(after['recall'])} | {after['approved_vetoed']} | {after['rejected_vetoed']} |"
            )
    lines.extend(
        [
            "",
            "## Split Metrics Excluding Near-Time Conflicts",
            "",
            "| Split | Items | Kind | Before P | After P | Before R | After R | Approved vetoed | Rejected vetoed |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split in ("train", "validation", "test", "all"):
        block = doc["splits_excluding_conflicts"][split]
        for kind in ("touch", "drop_floor"):
            before = block["before"][kind]
            after = block["after_shadow"][kind]
            lines.append(
                f"| {split} | {block['items']} | `{kind}` | {percent(before['precision'])} | {percent(after['precision'])} | "
                f"{percent(before['recall'])} | {percent(after['recall'])} | {after['approved_vetoed']} | {after['rejected_vetoed']} |"
            )
    lines.extend(
        [
            "",
            "## Changed Candidates",
            "",
            f"- Total changed clean labels: {doc['changed_counts']['total']}",
            f"- Approved candidates vetoed: {doc['changed_counts']['approved_to_veto']}",
            f"- Approved candidates vetoed inside near-time conflicts: {doc['changed_counts']['approved_to_veto_conflicted']}",
            f"- Rejected candidates vetoed: {doc['changed_counts']['rejected_to_veto']}",
            "",
            "## Promotion Checks",
            "",
            f"- Approved vetoes: {doc['promotion_checks']['approved_vetoes']}",
            f"- Approved vetoes excluding near-time conflicts: {doc['promotion_checks']['approved_vetoes_excluding_conflicts']}",
            f"- Per-video touch recall floor: {percent(doc['promotion_checks']['touch_recall_floor'])}",
            f"- Per-video touch recall violations: {len(doc['promotion_checks']['per_video_touch_recall_floor_violations'])}",
            f"- Drop approved vetoes: {doc['promotion_checks']['drop_approved_vetoes']}",
            f"- Drop rejected vetoes: {doc['promotion_checks']['drop_rejected_vetoes']}",
            f"- Gated review files: {doc.get('gated_reviews', {}).get('review_files', 0)}",
            f"- Gated suppressed review items: {doc.get('gated_reviews', {}).get('precision_gate_suppressed_items', 0)}",
            f"- Conflict-clean excluded review items: {doc.get('gated_conflict_clean_reviews', {}).get('label_conflict_excluded_items', 0)}",
            "",
            "## Drop Changed Videos",
            "",
            "| Video | Split | Before P | After P | Approved vetoed | Rejected vetoed |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in doc["promotion_checks"]["drop_changed_videos"]:
        lines.append(
            f"| {row['video']} | {row['split']} | {percent(row['before_precision'])} | {percent(row['after_precision'])} | "
            f"{row['approved_vetoed']} | {row['rejected_vetoed']} |"
        )
    if doc["promotion_checks"]["per_video_touch_recall_floor_violations"]:
        lines.extend(["", "## Per-Video Touch Recall Violations", ""])
        for row in doc["promotion_checks"]["per_video_touch_recall_floor_violations"]:
            lines.append(f"- {row['video']} ({row['split']}): {percent(row['touch_recall'])}")
    lines.extend(
        [
            "",
            "## Conflict Audit",
            "",
            f"- Near-time duplicate/conflict groups: {len(doc['conflicts'])}",
            "- See `conflicts.csv` and `shadow_changed_candidates.csv` for inspection rows.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_doc(
    labels: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    *,
    labels_path: Path,
    benchmark_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    enriched = enrich_labels(labels, benchmark_rows)
    mark_conflict_labels(enriched)
    splits = {split: metric_block(enriched, split=split) for split in ("train", "validation", "test")}
    splits["all"] = metric_block(enriched, split=None)
    splits_excluding_conflicts = {split: metric_block(enriched, split=split, exclude_conflicts=True) for split in ("train", "validation", "test")}
    splits_excluding_conflicts["all"] = metric_block(enriched, split=None, exclude_conflicts=True)
    per_video = per_video_metrics(enriched)
    changed = changed_rows(enriched)
    conflicts = conflict_groups(enriched)
    touch_recall_floor = 0.85
    per_video_touch_recall_violations = []
    drop_changed_videos = []
    for video, data in per_video.items():
        touch_after = data["after_shadow"]["touch"]
        touch_recall = touch_after["recall"]
        if touch_recall is not None and touch_recall < touch_recall_floor:
            per_video_touch_recall_violations.append(
                {
                    "video": video,
                    "split": data["split"],
                    "touch_recall": touch_recall,
                    "approved_vetoed": touch_after["approved_vetoed"],
                }
            )
        drop_after = data["after_shadow"]["drop_floor"]
        if drop_after["approved_vetoed"] or drop_after["rejected_vetoed"]:
            drop_changed_videos.append(
                {
                    "video": video,
                    "split": data["split"],
                    "before_precision": data["before"]["drop_floor"]["precision"],
                    "after_precision": drop_after["precision"],
                    "approved_vetoed": drop_after["approved_vetoed"],
                    "rejected_vetoed": drop_after["rejected_vetoed"],
                }
            )
    return (
        {
            "schema_version": 1,
            "inputs": {
                "labels": str(labels_path),
                "benchmark": str(benchmark_path),
            },
            "splits": splits,
            "splits_excluding_conflicts": splits_excluding_conflicts,
            "per_video": per_video,
            "promotion_checks": {
                "touch_recall_floor": touch_recall_floor,
                "approved_vetoes": sum(1 for row in changed if row.get("precision_gate_effective_status") == "vetoed_approved"),
                "approved_vetoes_excluding_conflicts": sum(
                    1
                    for row in changed
                    if row.get("precision_gate_effective_status") == "vetoed_approved" and not row.get("near_time_conflict")
                ),
                "per_video_touch_recall_floor_violations": per_video_touch_recall_violations,
                "drop_changed_videos": drop_changed_videos,
                "drop_approved_vetoes": sum(
                    1
                    for row in changed
                    if row.get("kind") == "drop_floor" and row.get("precision_gate_effective_status") == "vetoed_approved"
                ),
                "drop_rejected_vetoes": sum(
                    1
                    for row in changed
                    if row.get("kind") == "drop_floor" and row.get("precision_gate_effective_status") == "vetoed_rejected"
                ),
            },
            "changed_counts": {
                "total": len(changed),
                "approved_to_veto": sum(1 for row in changed if row.get("precision_gate_effective_status") == "vetoed_approved"),
                "rejected_to_veto": sum(1 for row in changed if row.get("precision_gate_effective_status") == "vetoed_rejected"),
                "approved_to_veto_conflicted": sum(
                    1
                    for row in changed
                    if row.get("precision_gate_effective_status") == "vetoed_approved" and row.get("near_time_conflict")
                ),
            },
            "changed_by_reason": dict(Counter(row.get("precision_gate_reasons") or "" for row in changed)),
            "conflicts": conflicts,
        },
        changed,
        conflicts,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score precision-gate shadow impact against all reviewed labels.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--reviews-dir", type=Path, default=ROOT / "reviews")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = read_jsonl(args.labels)
    benchmark_rows = read_jsonl(args.benchmark)
    doc, changed, conflicts = build_doc(labels, benchmark_rows, labels_path=args.labels, benchmark_path=args.benchmark)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "shadow_metrics.json"
    changed_path = args.out_dir / "shadow_changed_candidates.csv"
    conflicts_path = args.out_dir / "conflicts.csv"
    report_path = args.out_dir / "shadow_report.md"
    gated_reviews_dir = args.out_dir / "gated_reviews"
    enriched_labels = enrich_labels(labels, benchmark_rows)
    mark_conflict_labels(enriched_labels)
    gated_reviews = write_gated_reviews(
        enriched_labels,
        args.reviews_dir,
        gated_reviews_dir,
    )
    conflict_clean_reviews_dir = args.out_dir / "gated_conflict_clean_reviews"
    gated_conflict_clean_reviews = write_gated_reviews(
        enriched_labels,
        args.reviews_dir,
        conflict_clean_reviews_dir,
        exclude_conflicts=True,
    )
    doc["gated_reviews"] = gated_reviews
    doc["gated_conflict_clean_reviews"] = gated_conflict_clean_reviews
    write_json(metrics_path, doc)
    write_csv(changed_path, changed, list(changed[0].keys()) if changed else ["source_video"])
    write_csv(conflicts_path, conflicts, list(conflicts[0].keys()) if conflicts else ["video", "kind", "time_sec", "statuses", "item_ids", "notes"])
    write_report(report_path, doc)
    print(f"metrics:   {metrics_path}")
    print(f"changed:   {changed_path}")
    print(f"conflicts: {conflicts_path}")
    print(f"report:    {report_path}")
    print(f"labels:    {len(labels)}")
    print(f"changed:   {len(changed)}")
    print(f"conflicts: {len(conflicts)}")


if __name__ == "__main__":
    main()
