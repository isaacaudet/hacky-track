#!/usr/bin/env python3
"""Audit which QA rallies are eligible for strict best-rally HUD selection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from render_best_rally_hud import rally_selection_notes


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "outputs" / "full_training_27_qa" / "qa_manifest.json"
DEFAULT_OUT_DIR = ROOT / "outputs" / "strict_rally_audit"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_event_path(raw: Any) -> Path:
    path = Path(str(raw or ""))
    if path.exists():
        return path
    root_path = ROOT / path
    if root_path.exists():
        return root_path
    return path


def rejection_reasons(notes: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if int(notes.get("touches") or 0) < 6:
        reasons.append("touches_below_6")
    drop_reject_count = notes.get("nonterminal_drop_floor_events")
    if drop_reject_count is None:
        drop_reject_count = notes.get("explicit_drop_floor_events")
    if int(drop_reject_count or 0) > 0:
        reasons.append("explicit_drop_floor")
    if int(notes.get("hidden_drop_gap_count") or 0) > 0:
        reasons.append("hidden_airtime_gap")
    if int(notes.get("low_ball_accuracy_events") or 0) > 0:
        reasons.append("low_ball_accuracy")
    if int(notes.get("ambiguous_contact_events") or 0) > 0:
        reasons.append("ambiguous_contact")
    if notes.get("ended_by_gap_without_floor_reset"):
        reasons.append("ended_by_gap_without_floor_reset")
    return reasons


def audit_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    rows: list[dict[str, Any]] = []
    for video_index, run in enumerate(manifest.get("runs", [])):
        qa_path = resolve_event_path(run.get("qa_events_path"))
        if not qa_path.exists():
            continue
        doc = read_json(qa_path)
        source_video = Path(str(doc.get("source_video") or run.get("video") or qa_path.parent.name)).name
        for rally in doc.get("rallies", []):
            rally_id = int(rally.get("id") or 0)
            events = [
                item
                for item in doc.get("events", [])
                if int(item.get("qa_rally_id") or -1) == rally_id
            ]
            notes = rally_selection_notes(events, rally)
            reasons = rejection_reasons(notes)
            rows.append(
                {
                    "source_video": source_video,
                    "video_index": video_index,
                    "qa_events_path": rel(qa_path),
                    "rally_id": rally_id,
                    "touches": int(rally.get("touches") or notes.get("touches") or 0),
                    "stalls": int(rally.get("stalls") or 0),
                    "around_the_world": int(rally.get("around_the_world") or 0),
                    "duration_sec": rally.get("duration_sec"),
                    "quality_score": rally.get("quality_score"),
                    "quality_grade": rally.get("quality_grade"),
                    "strict_complete": bool(notes.get("strict_complete")),
                    "rejection_reasons": reasons,
                    "selection_notes": notes,
                    "end_reason": rally.get("end_reason"),
                    "next_contact_gap_sec": rally.get("next_contact_gap_sec"),
                }
            )
    rejection_counts = Counter(reason for row in rows for reason in row["rejection_reasons"])
    strict_rows = [row for row in rows if row["strict_complete"]]
    rejected_rows = [row for row in rows if not row["strict_complete"]]
    top_rejected = sorted(
        rejected_rows,
        key=lambda item: (
            -float(item.get("quality_score") or 0.0),
            -int(item.get("touches") or 0),
            int(item.get("video_index") or 0),
        ),
    )[:20]
    top_strict = sorted(
        strict_rows,
        key=lambda item: (
            -float(item.get("quality_score") or 0.0),
            -int(item.get("touches") or 0),
            int(item.get("video_index") or 0),
        ),
    )[:10]
    return {
        "schema_version": 1,
        "manifest_path": rel(manifest_path),
        "summary": {
            "rallies": len(rows),
            "strict_complete_rallies": len(strict_rows),
            "rejected_rallies": len(rejected_rows),
            "videos": len(manifest.get("runs", [])),
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "has_strict_best_rally": bool(strict_rows),
        },
        "top_strict_candidates": top_strict,
        "top_rejected_candidates": top_rejected,
        "rallies": rows,
    }


def write_report(path: Path, audit: dict[str, Any]) -> None:
    summary = audit["summary"]
    lines = [
        "# Strict Rally Audit",
        "",
        f"- Manifest: `{audit['manifest_path']}`",
        f"- Rallies: `{summary['rallies']}`",
        f"- Strict-complete rallies: `{summary['strict_complete_rallies']}`",
        f"- Rejected rallies: `{summary['rejected_rallies']}`",
        f"- Has strict best rally: `{summary['has_strict_best_rally']}`",
        "",
        "## Rejection Counts",
        "",
    ]
    for reason, count in sorted(summary.get("rejection_counts", {}).items()):
        lines.append(f"- {reason}: {count}")
    if audit.get("top_strict_candidates"):
        lines.extend(["", "## Top Strict Candidates", ""])
        for row in audit["top_strict_candidates"]:
            lines.append(
                f"- `{row['source_video']}` rally `{row['rally_id']}`: "
                f"{row['touches']} touches, score `{row.get('quality_score')}`"
            )
    lines.extend(["", "## Top Rejected Candidates", ""])
    for row in audit.get("top_rejected_candidates", []):
        reasons = ", ".join(row.get("rejection_reasons") or [])
        lines.append(
            f"- `{row['source_video']}` rally `{row['rally_id']}`: "
            f"{row['touches']} touches, score `{row.get('quality_score')}`, rejected by {reasons or 'unknown'}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit strict-complete best-rally eligibility")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_manifest(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.out_dir / "strict_rally_audit.json"
    report_path = args.out_dir / "strict_rally_audit.md"
    write_json(audit_path, audit)
    write_report(report_path, audit)
    print(f"strict rally audit: {audit_path}")
    print(f"strict-complete rallies: {audit['summary']['strict_complete_rallies']}")


if __name__ == "__main__":
    main()
