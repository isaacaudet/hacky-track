#!/usr/bin/env python3
"""Apply traceable review decisions to seeded review files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from review_app import ROOT, utc_now, write_json


DEFAULT_DECISIONS = ROOT / "reviews" / "codex_visual_review_decisions.json"
DEFAULT_REVIEWS_DIR = ROOT / "reviews"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def review_path_for_stem(stem: str, reviews_dir: Path) -> Path:
    return reviews_dir / f"{stem}.review.json"


def apply_decisions(decisions_path: Path, *, reviews_dir: Path = DEFAULT_REVIEWS_DIR, dry_run: bool = False) -> dict[str, Any]:
    decisions_doc = read_json(decisions_path)
    decisions = decisions_doc.get("decisions", decisions_doc if isinstance(decisions_doc, list) else [])
    by_stem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for decision in decisions:
        stem = str(decision.get("review_stem") or "")
        if not stem:
            raise ValueError(f"Decision missing review_stem: {decision}")
        by_stem[stem].append(decision)

    summary = {
        "decisions_path": str(decisions_path),
        "reviews_dir": str(reviews_dir),
        "dry_run": dry_run,
        "review_files_seen": 0,
        "applied": 0,
        "missing_files": [],
        "missing_items": [],
        "status_counts": defaultdict(int),
        "kind_counts": defaultdict(lambda: defaultdict(int)),
    }

    for stem, stem_decisions in sorted(by_stem.items()):
        review_path = review_path_for_stem(stem, reviews_dir)
        if not review_path.exists():
            summary["missing_files"].append(str(review_path))
            continue
        doc = read_json(review_path)
        items = doc.get("items", [])
        by_id = {str(item.get("id") or ""): item for item in items if item.get("id")}
        summary["review_files_seen"] += 1
        changed = False
        for decision in stem_decisions:
            item_id = str(decision.get("item_id") or "")
            item = by_id.get(item_id)
            if item is None:
                summary["missing_items"].append({"review_stem": stem, "item_id": item_id})
                continue
            status = str(decision.get("status") or "")
            if status not in {"approved", "rejected", "missing", "pending"}:
                raise ValueError(f"Unsupported status {status!r} for {stem}/{item_id}")
            item["status"] = status
            if decision.get("kind"):
                item["kind"] = decision["kind"]
            for field in (
                "time_sec",
                "start_sec",
                "end_sec",
                "duration_sec",
                "x",
                "y",
                "ball_confidence",
                "ball_accuracy",
                "foot_x",
                "foot_y",
                "foot_confidence",
                "foot_distance",
                "contact_side",
                "contact_type",
                "contact_confidence",
                "side_confidence",
                "side_source",
                "side_uncertainty_reason",
                "trick_label",
                "drop_source",
                "drop_score",
            ):
                if field in decision:
                    item[field] = decision[field]
            tags = list(item.get("review_tags") or [])
            for tag in ["codex_visual_review", str(decision.get("review_tag") or "")]:
                if tag and tag not in tags:
                    tags.append(tag)
            item["review_tags"] = tags
            evidence = str(decision.get("evidence") or "").strip()
            previous_note = str(item.get("note") or "").strip()
            note = f"codex_visual_review: {evidence}" if evidence else "codex_visual_review"
            item["note"] = note if not previous_note else f"{previous_note}; {note}"
            item["reviewed_by"] = "codex_visual_review"
            item["reviewed_at"] = decisions_doc.get("reviewed_at") or utc_now()
            item["review_evidence"] = decision.get("evidence")
            item["review_decision_source"] = str(decisions_path.relative_to(ROOT)) if decisions_path.is_relative_to(ROOT) else str(decisions_path)
            summary["applied"] += 1
            summary["status_counts"][status] += 1
            summary["kind_counts"][str(item.get("kind") or "touch")][status] += 1
            changed = True
        if changed:
            doc["updated_at"] = utc_now()
            if not dry_run:
                write_json(review_path, doc)

    summary["status_counts"] = dict(summary["status_counts"])
    summary["kind_counts"] = {kind: dict(counts) for kind, counts in summary["kind_counts"].items()}
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply review decisions to seeded review files")
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--reviews-dir", type=Path, default=DEFAULT_REVIEWS_DIR)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = apply_decisions(args.decisions, reviews_dir=args.reviews_dir, dry_run=args.dry_run)
    out_path = args.out or args.reviews_dir / "apply_review_decisions_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"summary: {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
