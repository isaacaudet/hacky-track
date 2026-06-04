#!/usr/bin/env python3
"""Promote side-specific trick labels to explicit wearer-limb side basis.

This is a data migration for reviewed touch labels created before
`contact_side_basis` existed. If a reviewed event already has a side-specific
trick label such as `left_inner_kick` or `right_stall`, that label was made
from the contacting limb. It should not be sent back through manual side-basis
review as a legacy screen/pose ambiguity.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_LABELS_DIR = ROOT / "runs/release-27-public/touch_corpus_v1/visual_touch_labels"
DEFAULT_REPORT = ROOT / "runs/release-27-public/touch_corpus_v1/release_contact_classifier_v1/side_semantics_audit/contact_side_basis_migration_report.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def side_from_trick_label(trick_label: str | None) -> str | None:
    raw = str(trick_label or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in {"l", "left"} or raw.startswith("left_") or raw.endswith("_left") or "_left_" in raw:
        return "left"
    if raw in {"r", "right"} or raw.startswith("right_") or raw.endswith("_right") or "_right_" in raw:
        return "right"
    return None


def reviewed_event(event: dict[str, Any]) -> bool:
    return event.get("review_status") in {None, "", "approved", "reviewed"}


def migrate_doc(doc: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    migrated = json.loads(json.dumps(doc))
    changes: list[dict[str, Any]] = []
    for rally_index, rally in enumerate(migrated.get("rallies", [])):
        for event_index, event in enumerate(rally.get("events", [])):
            if not reviewed_event(event):
                continue
            side = str(event.get("contact_side") or "")
            trick_label = str(event.get("trick_label") or "")
            basis = str(event.get("contact_side_basis") or "")
            trick_side = side_from_trick_label(trick_label)
            if basis or side not in {"left", "right"} or trick_side != side:
                continue
            event["contact_side_basis"] = "wearer_limb"
            changes.append(
                {
                    "rally_index": rally_index,
                    "event_index": event_index,
                    "time_sec": event.get("time_sec"),
                    "contact_side": side,
                    "trick_label": trick_label,
                    "new_contact_side_basis": "wearer_limb",
                }
            )
    return migrated, changes


def run_migration(labels_dir: Path, report_path: Path, *, dry_run: bool) -> dict[str, Any]:
    files = sorted(labels_dir.glob("*.events.json"))
    file_reports: list[dict[str, Any]] = []
    basis_before: Counter[str] = Counter()
    basis_after: Counter[str] = Counter()
    changed_files = 0
    changed_events = 0
    for path in files:
        doc = read_json(path)
        for rally in doc.get("rallies", []):
            for event in rally.get("events", []):
                if event.get("contact_side") in {"left", "right"} or side_from_trick_label(event.get("trick_label")):
                    basis_before[str(event.get("contact_side_basis") or "missing")] += 1
        migrated, changes = migrate_doc(doc)
        for rally in migrated.get("rallies", []):
            for event in rally.get("events", []):
                if event.get("contact_side") in {"left", "right"} or side_from_trick_label(event.get("trick_label")):
                    basis_after[str(event.get("contact_side_basis") or "missing")] += 1
        if changes:
            changed_files += 1
            changed_events += len(changes)
            file_reports.append({"path": str(path), "changes": changes})
            if not dry_run:
                write_json(path, migrated)
    report = {
        "schema_version": 1,
        "status": "dry_run" if dry_run else "complete",
        "labels_dir": str(labels_dir),
        "files_scanned": len(files),
        "files_changed": changed_files,
        "events_changed": changed_events,
        "basis_before": dict(basis_before),
        "basis_after": dict(basis_after),
        "files": file_reports,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Promote side-specific trick labels to contact_side_basis=wearer_limb")
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_migration(args.labels_dir.resolve(), args.report.resolve(), dry_run=args.dry_run)
    print(f"status: {report['status']}")
    print(f"report: {args.report}")
    print(json.dumps({key: report[key] for key in ("files_scanned", "files_changed", "events_changed", "basis_before", "basis_after")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
