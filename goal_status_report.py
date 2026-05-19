#!/usr/bin/env python3
"""Summarize current evidence against the Hacky Track goal stop gates."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_QA_MANIFEST = ROOT / "outputs" / "full_training_27_qa" / "qa_manifest.json"
DEFAULT_BALL_AUDIT = ROOT / "outputs" / "ball_tracking_audit" / "audit_metrics.json"
DEFAULT_VALIDATION = ROOT / "outputs" / "review_validation" / "validation_metrics.json"
DEFAULT_HUD_SUMMARY = ROOT / "outputs" / "best_rally_hud" / "best_rally_sprite_hud_summary.json"
DEFAULT_HUD_VERIFY = ROOT / "outputs" / "best_rally_hud" / "hud_verification.json"
DEFAULT_HUD_VISUAL_REVIEW = ROOT / "outputs" / "best_rally_hud" / "hud_visual_review.json"
DEFAULT_OUT_DIR = ROOT / "outputs" / "goal_status"

PRECISION_TARGETS = {
    "touch": 0.85,
    "drop_floor": 0.85,
    "stall": 0.90,
    "around_the_world": 0.90,
}
MIN_APPROVED_STALL_WINDOWS = 3


@dataclass(frozen=True)
class Gate:
    name: str
    status: str
    evidence: str
    next_step: str = ""


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def count_label(kind: str, stats: dict[str, Any]) -> str:
    return (
        f"{kind}: precision={pct(stats.get('precision_on_reviewed_candidates'))}, "
        f"approved={stats.get('approved_candidates', 0)}, rejected={stats.get('rejected_candidates', 0)}"
    )


def batch_count_label(kind: str, stats: dict[str, Any]) -> str:
    return (
        f"{kind}: current_precision={pct(stats.get('precision_on_current_batch'))}, "
        f"approved={stats.get('approved', 0)}, rejected={stats.get('rejected', 0)}"
    )


def rel(path: Path | str | None) -> str:
    if path is None:
        return "missing"
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def load_qa_docs(manifest: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not manifest:
        return []
    docs: list[dict[str, Any]] = []
    for run in manifest.get("runs", []):
        qa_path = ROOT / str(run.get("qa_events_path") or "")
        if not qa_path.exists():
            qa_path = Path(str(run.get("qa_events_path") or ""))
        doc = read_json(qa_path)
        if doc:
            docs.append(doc)
    return docs


def qa_totals(docs: list[dict[str, Any]]) -> dict[str, int]:
    totals = {
        "touches": 0,
        "drop_floor": 0,
        "stalls": 0,
        "around_the_world": 0,
        "rallies": 0,
        "low_ball_accuracy_events": 0,
        "large_ball_corrections": 0,
    }
    for doc in docs:
        events = doc.get("events", [])
        totals["touches"] += sum(1 for item in events if item.get("type") == "touch")
        totals["drop_floor"] += sum(1 for item in events if item.get("type") == "drop_floor")
        totals["stalls"] += sum(1 for item in events if item.get("type") == "stall")
        totals["around_the_world"] += sum(1 for item in events if item.get("type") == "around_the_world")
        totals["rallies"] += len(doc.get("rallies", []))
        summary = doc.get("summary", {})
        totals["low_ball_accuracy_events"] += int(summary.get("low_ball_accuracy_events") or 0)
        totals["large_ball_corrections"] += int(summary.get("large_ball_corrections") or 0)
    return totals


def video_352_split_gate(docs: list[dict[str, Any]]) -> Gate:
    target = next((doc for doc in docs if "video-352_singular_display 2.MOV" in str(doc.get("source_video"))), None)
    if target is None:
        return Gate(
            "Known video-352 internal drop split",
            "FAIL",
            "No QA document found for video-352_singular_display 2.MOV.",
            "Regenerate QA for the 27-video manifest and verify the source video is included.",
        )
    drops = [
        item
        for item in target.get("events", [])
        if item.get("type") == "drop_floor" and 21.8 <= float(item.get("time_sec") or 0.0) <= 23.5
    ]
    before = [
        rally
        for rally in target.get("rallies", [])
        if float(rally.get("start_sec") or 0.0) < 22.8 <= float(rally.get("end_sec") or 0.0)
    ]
    after = [
        rally
        for rally in target.get("rallies", [])
        if 22.8 < float(rally.get("start_sec") or 0.0) < 25.0
    ]
    if drops and before and after:
        drop_times = ", ".join(f"{float(item.get('time_sec')):.2f}s" for item in drops)
        return Gate(
            "Known video-352 internal drop split",
            "PASS",
            f"Found floor reset at {drop_times}; rallies exist on both sides of the reset.",
        )
    return Gate(
        "Known video-352 internal drop split",
        "FAIL",
        f"Drop candidates near 22.9s: {len(drops)}; before rallies: {len(before)}; after rallies: {len(after)}.",
        "Tighten hidden floor-reset timing before using best-rally selection.",
    )


def build_gates(
    manifest: dict[str, Any] | None,
    docs: list[dict[str, Any]],
    audit: dict[str, Any] | None,
    validation: dict[str, Any] | None,
    hud_summary: dict[str, Any] | None,
    hud_verify: dict[str, Any] | None,
    hud_visual_review: dict[str, Any] | None,
) -> list[Gate]:
    gates: list[Gate] = []
    totals = qa_totals(docs)
    runs = len(manifest.get("runs", [])) if manifest else 0
    if runs == 27 and len(docs) == 27:
        gates.append(
            Gate(
                "Full 27-video QA manifest",
                "PASS",
                f"Loaded 27 QA docs with {totals['touches']} touches, {totals['drop_floor']} ground resets, {totals['stalls']} stalls, and {totals['rallies']} rallies.",
            )
        )
    else:
        gates.append(
            Gate(
                "Full 27-video QA manifest",
                "FAIL",
                f"Manifest runs: {runs}; readable QA docs: {len(docs)}.",
                "Run the pipeline without --skip-training/--skip-qa after confirming the 27 source videos.",
            )
        )

    gates.append(video_352_split_gate(docs))

    if audit and audit.get("target_met") is True:
        gates.append(
            Gate(
                "Ball tracking audit target",
                "PASS",
                f"Pass rate {pct(audit.get('pass_rate'))} on {audit.get('sampled_events')} sampled high-confidence events; target {pct(audit.get('target_pass_rate'))}.",
            )
        )
    else:
        gates.append(
            Gate(
                "Ball tracking audit target",
                "FAIL",
                "No passing audit metrics found.",
                "Rerun audit_ball_tracking.py and fix confident centers that lack visual support.",
            )
        )

    batch = validation.get("review_batch") if validation else None
    decision_coverage = batch.get("decision_coverage") if batch else None
    if batch and float(decision_coverage or 0.0) >= 1.0:
        gates.append(
            Gate(
                "Review batch decision coverage",
                "PASS",
                f"All {batch.get('total_items')} batch items have decisions.",
            )
        )
    elif batch:
        gates.append(
            Gate(
                "Review batch decision coverage",
                "BLOCKED",
                f"{batch.get('decided_items')} of {batch.get('total_items')} review-batch items decided ({pct(decision_coverage)}).",
                "Use review_app.py to approve/reject candidates and add missing touches/drops/stalls, then rerun review_validation.py.",
            )
        )
    else:
        gates.append(
            Gate(
                "Review batch decision coverage",
                "FAIL",
                "No review-batch metrics found in validation report.",
                "Run build_review_batch.py and review_validation.py.",
            )
        )

    reviewed_items = validation.get("reviewed_items") if validation else 0
    kinds = validation.get("kinds", {}) if validation else {}
    measurable_kinds = [
        kind
        for kind, stats in kinds.items()
        if stats.get("precision_on_reviewed_candidates") is not None or stats.get("recall_against_reviewed_labels") is not None
    ]
    if reviewed_items and len(measurable_kinds) >= 3:
        gates.append(
            Gate(
                "Reviewed validation metrics",
                "PASS",
                f"{reviewed_items} reviewed labels produce measurable precision/recall for {', '.join(measurable_kinds)}.",
            )
        )
    else:
        gates.append(
            Gate(
                "Reviewed validation metrics",
                "BLOCKED",
                f"{reviewed_items or 0} reviewed labels; measurable kinds: {', '.join(measurable_kinds) or 'none'}.",
                "Review enough touch, drop, stall, and trick examples to produce real precision/recall instead of empty denominators.",
            )
        )

    quality_failures: list[str] = []
    batch_kind_metrics = batch.get("kind_metrics", {}) if batch else {}
    for kind, target in PRECISION_TARGETS.items():
        stats = batch_kind_metrics.get(kind, {})
        precision = stats.get("precision_on_current_batch")
        if precision is None or float(precision) < target:
            quality_failures.append(f"{batch_count_label(kind, stats)} target={pct(target)}")
    if quality_failures:
        gates.append(
            Gate(
                "Reviewed detector quality targets",
                "BLOCKED",
                "; ".join(quality_failures),
                "Use the reviewed rejects to tighten candidate generation/reclassification, rerun QA, rebuild the review batch, and revalidate.",
            )
        )
    else:
        gates.append(
            Gate(
                "Reviewed detector quality targets",
                "PASS",
                "; ".join(batch_count_label(kind, batch_kind_metrics.get(kind, {})) for kind in PRECISION_TARGETS),
            )
        )

    stall_stats = kinds.get("stall", {})
    approved_stalls = int(stall_stats.get("approved_candidates") or 0)
    if approved_stalls >= MIN_APPROVED_STALL_WINDOWS:
        gates.append(
            Gate(
                "Stall window detection",
                "PASS",
                f"{approved_stalls} reviewed stall windows approved; minimum target is {MIN_APPROVED_STALL_WINDOWS}.",
            )
        )
    else:
        gates.append(
            Gate(
                "Stall window detection",
                "BLOCKED",
                f"{approved_stalls} reviewed stall windows approved; target is at least {MIN_APPROVED_STALL_WINDOWS}.",
                "Replace instant/near-foot stall candidates with duration windows that prove the sack is held on a foot/knee, then revalidate.",
            )
        )

    wrong_type_total = sum(int(stats.get("wrong_contact_type") or 0) for stats in kinds.values())
    wrong_side_total = sum(int(stats.get("wrong_side") or 0) for stats in kinds.values())
    if wrong_type_total == 0 and wrong_side_total == 0:
        gates.append(
            Gate(
                "Contact side/type label quality",
                "PASS",
                "Reviewed accepted/missing labels show no side or contact-type mismatches.",
            )
        )
    else:
        gates.append(
            Gate(
                "Contact side/type label quality",
                "BLOCKED",
                f"Reviewed labels report wrong_side={wrong_side_total}, wrong_contact_type={wrong_type_total}.",
                "Improve side/type assignment and preserve ambiguous labels instead of forcing incorrect foot/knee/type labels.",
            )
        )

    selection = hud_summary.get("selection_notes", {}) if hud_summary else {}
    source_video = Path(str(hud_summary.get("source_video", ""))).name if hud_summary else ""
    if selection.get("strict_complete") is True and "video-352_singular_display 2.MOV" not in source_video:
        gates.append(
            Gate(
                "Strict best-rally selection",
                "PASS",
                f"Selected {source_video} with {selection.get('touches')} touches, no explicit drops, and strict_complete=True.",
            )
        )
    else:
        gates.append(
            Gate(
                "Strict best-rally selection",
                "FAIL",
                f"Selected source={source_video or 'missing'}, selection={selection}.",
                "Require strict_complete=True and exclude the known split video-352 failure case.",
            )
        )

    streams = hud_verify.get("streams", []) if hud_verify else []
    has_video = any("|video|" in stream for stream in streams)
    has_audio = any("|audio|" in stream for stream in streams)
    if hud_verify and has_video and has_audio and float(hud_verify.get("first_frame_mean") or 0.0) > 1.0:
        gates.append(
            Gate(
                "HUD technical verification",
                "PASS",
                f"{hud_verify.get('width')}x{hud_verify.get('height')}, {hud_verify.get('frames')} frames, {hud_verify.get('duration_sec')}s, video+audio streams present.",
            )
        )
    else:
        gates.append(
            Gate(
                "HUD technical verification",
                "FAIL",
                f"HUD verification={hud_verify}.",
                "Rerender HUD and rerun ffprobe/OpenCV verification.",
            )
        )

    if hud_visual_review and hud_visual_review.get("status") == "pass":
        gates.append(
            Gate(
                "HUD visual clipping/occlusion",
                "PASS",
                str(hud_visual_review.get("evidence", "HUD preview was visually reviewed.")),
            )
        )
    else:
        gates.append(
            Gate(
                "HUD visual clipping/occlusion",
                "NEEDS_VISUAL_REVIEW",
                "Automated verification proves the MP4 is playable but does not prove text never clips or effects never cover the sack.",
                "Inspect outputs/best_rally_hud/best_rally_sprite_hud_preview.jpg and the MP4 after each HUD change.",
            )
        )
    return gates


def write_outputs(gates: list[Gate], totals: dict[str, int], paths: dict[str, Path], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    status_counts: dict[str, int] = {}
    for gate in gates:
        status_counts[gate.status] = status_counts.get(gate.status, 0) + 1
    payload = {
        "status_counts": status_counts,
        "qa_totals": totals,
        "paths": {key: rel(path) for key, path in paths.items()},
        "gates": [gate.__dict__ for gate in gates],
        "goal_complete": all(gate.status == "PASS" for gate in gates),
    }
    json_path = out_dir / "goal_status.json"
    md_path = out_dir / "goal_status_report.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Hacky Track Goal Status",
        "",
        f"- Goal complete: {payload['goal_complete']}",
        f"- Gate status counts: {', '.join(f'{key}={value}' for key, value in sorted(status_counts.items()))}",
        f"- QA totals: touches={totals['touches']}, drops={totals['drop_floor']}, stalls={totals['stalls']}, rallies={totals['rallies']}",
        "",
        "## Stop Gates",
        "",
        "| Gate | Status | Evidence | Next step |",
        "| --- | --- | --- | --- |",
    ]
    for gate in gates:
        lines.append(
            f"| {gate.name} | {gate.status} | {gate.evidence.replace('|', '/')} | {gate.next_step.replace('|', '/') or '-'} |"
        )
    lines.extend(["", "## Artifact Paths", ""])
    for key, path in paths.items():
        lines.append(f"- {key}: `{rel(path)}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a Hacky Track goal status report")
    parser.add_argument("--qa-manifest", type=Path, default=DEFAULT_QA_MANIFEST)
    parser.add_argument("--ball-audit", type=Path, default=DEFAULT_BALL_AUDIT)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--hud-summary", type=Path, default=DEFAULT_HUD_SUMMARY)
    parser.add_argument("--hud-verification", type=Path, default=DEFAULT_HUD_VERIFY)
    parser.add_argument("--hud-visual-review", type=Path, default=DEFAULT_HUD_VISUAL_REVIEW)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_json(args.qa_manifest)
    docs = load_qa_docs(manifest)
    audit = read_json(args.ball_audit)
    validation = read_json(args.validation)
    hud_summary = read_json(args.hud_summary)
    hud_verify = read_json(args.hud_verification)
    hud_visual_review = read_json(args.hud_visual_review)
    totals = qa_totals(docs)
    gates = build_gates(manifest, docs, audit, validation, hud_summary, hud_verify, hud_visual_review)
    paths = {
        "qa_manifest": args.qa_manifest,
        "ball_audit": args.ball_audit,
        "validation": args.validation,
        "hud_summary": args.hud_summary,
        "hud_verification": args.hud_verification,
        "hud_visual_review": args.hud_visual_review,
    }
    json_path, md_path = write_outputs(gates, totals, paths, args.out_dir)
    print(f"goal status: {json_path}")
    print(f"goal report: {md_path}")


if __name__ == "__main__":
    main()
