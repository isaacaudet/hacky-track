#!/usr/bin/env python3
"""Write a release-candidate audit report for a Hacky Track run directory."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


@dataclass
class Gate:
    name: str
    status: str
    evidence: str
    next_step: str = ""


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def rel(path: Path | None, base: Path = ROOT) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def resolve_run_path(run_dir: Path, raw: Any) -> Path | None:
    if raw in (None, ""):
        return None
    path = Path(str(raw))
    if path.is_absolute():
        return path
    run_path = run_dir / path
    if run_path.exists() or str(raw).split("/", 1)[0] in {
        "training",
        "models",
        "qa",
        "review_batches",
        "ball_tracking_audit",
        "validation",
        "release_evaluation",
        "hud",
        "strict_rally_audit",
        "strict_rally_audit_reviewed",
        "qa_reviewed",
        "release_report",
    }:
        return run_path
    return ROOT / path


def artifact_paths(run_dir: Path, overrides: dict[str, Any] | None = None) -> dict[str, Path]:
    manifest = read_json(run_dir / "run_manifest.json") or {}
    artifacts = manifest.get("artifacts", {}) if isinstance(manifest.get("artifacts"), dict) else {}
    defaults = {
        "summary": "summary.md",
        "run_manifest": "run_manifest.json",
        "portable_paths_audit": "portable_paths_audit.json",
        "events_json": "events.json",
        "events_csv": "events.csv",
        "rallies_json": "rallies.json",
        "training_manifest": "training/full_training_manifest.json",
        "model_dir": "models",
        "qa_manifest": "qa/qa_manifest.json",
        "qa_report": "qa/qa_report.md",
        "qa_reviewed_manifest": "qa_reviewed/qa_manifest.json",
        "review_batch": "review_batches/latest_review_batch.json",
        "ball_audit": "ball_tracking_audit/audit_metrics.json",
        "strict_rally_audit": "strict_rally_audit/strict_rally_audit.json",
        "strict_rally_audit_report": "strict_rally_audit/strict_rally_audit.md",
        "strict_rally_audit_reviewed": "strict_rally_audit_reviewed/strict_rally_audit.json",
        "validation": "validation/validation_metrics.json",
        "release_metrics": "release_evaluation/release_metrics.json",
        "release_report": "release_evaluation/release_evaluation_report.md",
        "dataset_splits": "release_evaluation/dataset_splits.json",
        "hud_video": "hud/best_rally_sprite_hud_overlay.mp4",
        "hud_summary": "hud/best_rally_sprite_hud_summary.json",
        "hud_verify": "hud/hud_verification.json",
        "hud_error": "hud/hud_error.json",
    }
    effective_overrides = dict(overrides or {})
    if effective_overrides.get("strict_rally_audit") and not effective_overrides.get("strict_rally_audit_report"):
        effective_overrides["strict_rally_audit_report"] = Path(str(effective_overrides["strict_rally_audit"])).with_suffix(".md")
    if effective_overrides.get("release_metrics") and not effective_overrides.get("release_report"):
        effective_overrides["release_report"] = Path(str(effective_overrides["release_metrics"])).with_name("release_evaluation_report.md")
    if effective_overrides.get("release_metrics") and not effective_overrides.get("dataset_splits"):
        effective_overrides["dataset_splits"] = Path(str(effective_overrides["release_metrics"])).with_name("dataset_splits.json")
    merged = {**defaults, **artifacts, **effective_overrides}
    return {name: resolve_run_path(run_dir, raw) or (run_dir / str(raw)) for name, raw in merged.items()}


def pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def qa_totals(qa_manifest: dict[str, Any] | None) -> dict[str, int]:
    runs = qa_manifest.get("runs", []) if qa_manifest else []
    return {
        "videos": len(runs),
        "touches": sum(int(run.get("touch_candidates") or 0) for run in runs),
        "drops": sum(int(run.get("ground_hit_candidates") or 0) for run in runs),
        "stalls": sum(int(run.get("stall_candidates") or 0) for run in runs),
        "atw": sum(int(run.get("around_the_world_candidates") or 0) for run in runs),
        "rallies": sum(int(run.get("rallies") or 0) for run in runs),
        "suppressed_touches": sum(int(run.get("suppressed_touch_candidates") or 0) for run in runs),
        "suppressed_drops": sum(int(run.get("suppressed_drop_candidates") or 0) for run in runs),
    }


def load_qa_doc(path: Path) -> dict[str, Any] | None:
    if path.exists():
        return read_json(path)
    root_path = ROOT / path
    if root_path.exists():
        return read_json(root_path)
    return None


def known_video_352_gate(qa_manifest: dict[str, Any] | None) -> Gate:
    if not qa_manifest:
        return Gate("video-352 drop near 22.9s", "fail", "No QA manifest found.", "Run the full pipeline first.")
    for run in qa_manifest.get("runs", []):
        if Path(str(run.get("video") or "")).name != "video-352_singular_display 2.MOV":
            continue
        qa_path = Path(str(run.get("qa_events_path") or ""))
        doc = load_qa_doc(qa_path)
        if not doc:
            return Gate("video-352 drop near 22.9s", "fail", f"QA doc missing at {qa_path}.", "Regenerate QA.")
        drops = [
            event
            for event in doc.get("events", [])
            if event.get("type") == "drop_floor" and 22.5 <= float(event.get("time_sec") or 0.0) <= 23.3
        ]
        if drops:
            times = ", ".join(f"{float(event.get('time_sec')):.3f}s" for event in drops)
            return Gate("video-352 drop near 22.9s", "pass", f"Detected floor reset at {times}.")
        return Gate("video-352 drop near 22.9s", "fail", "No floor reset found in the 22.5-23.3s window.", "Fix hidden floor reset detection.")
    return Gate("video-352 drop near 22.9s", "fail", "video-352_singular_display 2.MOV not present in QA manifest.", "Use the 27-video source set.")


def unresolved_better_rejected_candidate(strict_audit: dict[str, Any] | None) -> dict[str, Any] | None:
    if not strict_audit:
        return None
    top = strict_audit.get("top_strict_candidates", [])
    if not top:
        return None
    top_strict = top[0]
    strict_score = float(top_strict.get("quality_score") or 0.0)
    strict_touches = int(top_strict.get("touches") or 0)
    for item in strict_audit.get("top_rejected_candidates", []):
        if float(item.get("quality_score") or 0.0) <= strict_score:
            continue
        if int(item.get("touches") or 0) <= strict_touches:
            continue
        if any(
            reason in {"ended_by_gap_without_floor_reset", "hidden_airtime_gap", "ambiguous_contact", "low_ball_accuracy"}
            for reason in item.get("rejection_reasons", [])
        ):
            return item
    return None


def strict_rally_gate(strict_audit: dict[str, Any] | None) -> Gate:
    if not strict_audit:
        return Gate("Strict best-rally eligibility", "fail", "Missing strict rally audit.", "Run strict_rally_audit.py.")
    summary = strict_audit.get("summary", {})
    strict_count = int(summary.get("strict_complete_rallies") or 0)
    total = int(summary.get("rallies") or 0)
    if strict_count > 0:
        top = strict_audit.get("top_strict_candidates", [])
        first = unresolved_better_rejected_candidate(strict_audit)
        if first:
            reasons = ", ".join(first.get("rejection_reasons") or [])
            return Gate(
                "Strict best-rally eligibility",
                "blocked",
                (
                    f"{strict_count}/{total} rallies are strict-complete, but a better review-required candidate outranks the strict pick: "
                    f"{first.get('source_video')} rally {first.get('rally_id')} with {first.get('touches')} touches, "
                    f"score {first.get('quality_score')}, rejected by {reasons or 'unknown'}."
                ),
                "Review/apply the higher-touch candidate's missing drop/contact evidence before declaring a best rally.",
            )
        evidence = f"{strict_count}/{total} rallies are strict-complete."
        if top:
            first = top[0]
            evidence += f" Top: {first.get('source_video')} rally {first.get('rally_id')} with {first.get('touches')} touches."
        return Gate("Strict best-rally eligibility", "pass", evidence)
    rejected = strict_audit.get("top_rejected_candidates", [])[:3]
    reasons = "; ".join(
        f"{item.get('source_video')} rally {item.get('rally_id')}: {', '.join(item.get('rejection_reasons') or [])}"
        for item in rejected
    )
    return Gate(
        "Strict best-rally eligibility",
        "blocked",
        f"0/{total} strict-complete rallies. Top rejected: {reasons or 'none'}.",
        "Review/apply missing resets and ambiguous contacts, then rerun strict_rally_audit.py.",
    )


def hud_gate(hud_verify: dict[str, Any] | None, strict_audit: dict[str, Any] | None, hud_error: dict[str, Any] | None = None) -> Gate:
    strict_count = int((strict_audit or {}).get("summary", {}).get("strict_complete_rallies") or 0)
    if strict_count <= 0:
        error_text = ""
        if hud_error:
            error_text = f" HUD error: {hud_error.get('error')}"
        return Gate(
            "HUD MP4 technical verification",
            "blocked",
            f"No strict-complete rally is available, so a release HUD cannot be verified.{error_text}",
            "Fix review/QA until strict-complete rally count is >0, render HUD, and run hackytrack.py verify.",
        )
    unresolved = unresolved_better_rejected_candidate(strict_audit)
    if unresolved:
        return Gate(
            "HUD MP4 technical verification",
            "blocked",
            (
                "HUD verification is blocked because best-rally selection is unresolved: "
                f"{unresolved.get('source_video')} rally {unresolved.get('rally_id')} has "
                f"{unresolved.get('touches')} touches and outranks the strict HUD candidate."
            ),
            "Review/apply the unresolved higher-touch rally before rendering the release HUD.",
        )
    if not hud_verify:
        return Gate("HUD MP4 technical verification", "fail", "Missing HUD verification JSON.", "Run hackytrack.py verify.")
    streams = hud_verify.get("streams", [])
    has_video = any("|video|" in stream for stream in streams)
    has_audio = any("|audio|" in stream for stream in streams)
    sample_means = [float(item[1]) for item in hud_verify.get("sampled_frame_means", []) if isinstance(item, list) and len(item) == 2]
    nonblank = bool(sample_means) and max(sample_means) > 1.0
    if has_video and has_audio and nonblank:
        return Gate(
            "HUD MP4 technical verification",
            "pass",
            f"{hud_verify.get('width')}x{hud_verify.get('height')}, {hud_verify.get('frames')} frames, {hud_verify.get('duration_sec')}s, video/audio streams, nonblank samples.",
        )
    return Gate("HUD MP4 technical verification", "fail", f"has_video={has_video}, has_audio={has_audio}, sample_means={sample_means}.", "Rerender and verify HUD.")


def model_artifact_gate(model_dir: Path | None) -> Gate:
    if not model_dir or not model_dir.exists():
        return Gate("Model artifacts", "fail", "Missing model directory.", "Run training through hackytrack.py process.")
    artifacts = sorted(path.name for path in model_dir.glob("*") if path.is_file())
    has_model = any(path.endswith((".joblib", ".pkl", ".pt", ".onnx")) for path in artifacts)
    has_report = "training_report.json" in artifacts
    if has_model and has_report:
        return Gate("Model artifacts", "pass", f"Saved artifacts: {', '.join(artifacts)}.")
    return Gate("Model artifacts", "fail", f"Files present: {artifacts}.", "Save a model artifact and training_report.json.")


def release_metric_gates(release_metrics: dict[str, Any] | None) -> list[Gate]:
    if not release_metrics:
        return [Gate("Release metrics", "fail", "Missing release_metrics.json.", "Run hackytrack.py evaluate.")]
    gates: list[Gate] = []
    for item in release_metrics.get("target_gates", []):
        name = str(item.get("name") or "release metric")
        status = str(item.get("status") or "unknown")
        if status == "pass":
            gate_status = "pass"
        elif status == "candidate_only" and "release guard" in name:
            gate_status = "guarded"
        else:
            gate_status = "fail" if status == "fail" else "blocked"
        value = item.get("value")
        if isinstance(value, float):
            value_text = pct(value)
        else:
            value_text = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        gates.append(
            Gate(
                name,
                gate_status,
                f"value={value_text}, target={item.get('target')}",
                item.get("message")
                or ("Candidate-only guard is active; collect more reviewed examples before releasing this classifier." if gate_status == "guarded" else "Review/train more data." if gate_status != "pass" else ""),
            )
        )
    return gates


def review_batch_gate(batch: dict[str, Any] | None) -> Gate:
    if not batch:
        return Gate("Review batch", "fail", "Missing review batch.", "Run build_review_batch.py.")
    source_counts = batch.get("summary", {}).get("source_counts", {})
    active_count = int(source_counts.get("active_learning") or 0)
    candidate_count = int(source_counts.get("candidate") or 0)
    if active_count > 0 and candidate_count > 0:
        tag_counts = batch.get("summary", {}).get("tag_counts", {})
        gap_count = int(tag_counts.get("gap_without_floor_reset") or 0)
        gap_text = f", including {gap_count} gap-reset proposals" if gap_count else ""
        return Gate("Review batch active learning", "pass", f"{candidate_count} detector candidates and {active_count} active-learning proposals{gap_text}.")
    if active_count > 0:
        return Gate("Review batch active learning", "blocked", f"{active_count} active-learning proposals but no detector candidates.", "Rebalance batch selection.")
    return Gate("Review batch active learning", "blocked", "No active-learning proposals in the current batch.", "Rebuild the review batch with suppressed-event proposals.")


def review_application_gate(reviewed_manifest: dict[str, Any] | None, reviewed_strict_audit: dict[str, Any] | None) -> Gate:
    if not reviewed_manifest:
        return Gate("Reviewed decisions affect analytics", "blocked", "Missing reviewed QA manifest.", "Run hackytrack.py apply-reviews after making review decisions.")
    summary = reviewed_manifest.get("summary", {})
    inserted = int(summary.get("manual_missing_inserted") or 0)
    approved = int(summary.get("approved_candidates") or 0)
    rejected = int(summary.get("rejected_candidates") or 0)
    reviewed_audit_text = ""
    if reviewed_strict_audit:
        counts = reviewed_strict_audit.get("summary", {}).get("rejection_counts", {})
        reviewed_audit_text = f" Reviewed strict-audit rejection counts: {counts}."
    if inserted or approved or rejected:
        return Gate(
            "Reviewed decisions affect analytics",
            "pass",
            f"Applied decisions: inserted_missing={inserted}, approved_candidates={approved}, rejected_candidates={rejected}.{reviewed_audit_text}",
        )
    return Gate(
        "Reviewed decisions affect analytics",
        "blocked",
        f"Reviewed QA manifest exists but no decisions changed analytics.{reviewed_audit_text}",
        "Approve/reject candidates or approve active-learning missing events, then rerun apply-reviews.",
    )


def artifact_gate(paths: dict[str, Path]) -> Gate:
    required = [
        "summary",
        "run_manifest",
        "events_json",
        "events_csv",
        "rallies_json",
        "training_manifest",
        "qa_manifest",
        "strict_rally_audit",
        "review_batch",
        "ball_audit",
        "validation",
        "release_metrics",
    ]
    missing = [name for name in required if not paths.get(name) or not paths[name].exists()]
    if not missing:
        return Gate("Required run artifacts", "pass", f"All {len(required)} required artifacts are present.")
    return Gate("Required run artifacts", "fail", f"Missing: {', '.join(missing)}.", "Rerun hackytrack.py process.")


def portable_paths_gate(audit: dict[str, Any] | None) -> Gate:
    if audit and audit.get("passed") is True:
        return Gate("Portable text paths", "pass", f"Sanitized {len(audit.get('sanitized_files', []))} files; no /Users paths remain.")
    remaining = audit.get("remaining_files_with_users_paths") if audit else None
    return Gate("Portable text paths", "fail", f"Remaining files: {remaining or 'unknown'}", "Run path sanitization or fix hardcoded paths.")


def docs_gate() -> Gate:
    required = [ROOT / "README.md", ROOT / "CHANGELOG.md", ROOT / "RELEASE_STATUS.md", ROOT / "RELEASE_GOAL.md"]
    missing = [path.name for path in required if not path.exists()]
    if not missing:
        return Gate("Release docs", "pass", "README, changelog, release status, and release goal files exist.")
    return Gate("Release docs", "blocked", f"Missing docs: {', '.join(missing)}.", "Write missing release documentation.")


def tests_gate(tests_passed: bool, evidence: str) -> Gate:
    if tests_passed:
        return Gate("Tests", "pass", evidence or "Caller reported tests passed.")
    return Gate("Tests", "blocked", "No test-pass evidence supplied to this report.", "Run unit/integration/verifier tests and regenerate the report with --tests-passed.")


def build_report(
    run_dir: Path,
    *,
    tests_passed: bool = False,
    test_evidence: str = "",
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths = artifact_paths(run_dir, overrides)
    qa_manifest = read_json(paths["qa_manifest"])
    release_metrics = read_json(paths["release_metrics"])
    hud_verify = read_json(paths["hud_verify"])
    hud_error = read_json(paths["hud_error"])
    batch = read_json(paths["review_batch"])
    audit = read_json(paths["portable_paths_audit"])
    strict_audit = read_json(paths["strict_rally_audit"])
    reviewed_manifest = read_json(paths["qa_reviewed_manifest"])
    reviewed_strict_audit = read_json(paths["strict_rally_audit_reviewed"])
    selection_strict_audit = reviewed_strict_audit or strict_audit
    totals = qa_totals(qa_manifest)
    gates = [
        artifact_gate(paths),
        portable_paths_gate(audit),
        Gate("Full pipeline video count", "pass" if totals["videos"] >= 27 else "blocked", f"{totals['videos']} videos in QA manifest.", "Run all 27 videos." if totals["videos"] < 27 else ""),
        model_artifact_gate(paths.get("model_dir")),
        known_video_352_gate(qa_manifest),
        review_batch_gate(batch),
        strict_rally_gate(selection_strict_audit),
        review_application_gate(reviewed_manifest, reviewed_strict_audit),
        hud_gate(hud_verify, selection_strict_audit, hud_error),
        docs_gate(),
        tests_gate(tests_passed, test_evidence),
        *release_metric_gates(release_metrics),
    ]
    status_counts: dict[str, int] = {}
    for gate in gates:
        status_counts[gate.status] = status_counts.get(gate.status, 0) + 1
    complete = all(gate.status in {"pass", "guarded"} for gate in gates)
    return {
        "schema_version": 1,
        "run_dir": rel(run_dir),
        "goal_complete": complete,
        "status_counts": status_counts,
        "qa_totals": totals,
        "artifact_paths": {name: rel(path) for name, path in paths.items()},
        "gates": [gate.__dict__ for gate in gates],
        "known_limitations": [
            gate.evidence if not gate.next_step else f"{gate.evidence} Next: {gate.next_step}"
            for gate in gates
            if gate.status == "guarded"
        ],
        "limitations": [
            gate.evidence if not gate.next_step else f"{gate.evidence} Next: {gate.next_step}"
            for gate in gates
            if gate.status not in {"pass", "guarded"}
        ],
    }


def write_report(doc: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "release_candidate_report.json"
    md_path = out_dir / "release_candidate_report.md"
    write_json(json_path, doc)
    lines = [
        "# Hacky Track Release Candidate Report",
        "",
        f"- Run: `{doc['run_dir']}`",
        f"- Goal complete: `{doc['goal_complete']}`",
        f"- Gate counts: {', '.join(f'{key}={value}' for key, value in sorted(doc['status_counts'].items()))}",
        f"- QA totals: {doc['qa_totals']}",
        "",
        "## Gates",
        "",
        "| Gate | Status | Evidence | Next step |",
        "| --- | --- | --- | --- |",
    ]
    for gate in doc["gates"]:
        lines.append(
            f"| {gate['name']} | {gate['status']} | {str(gate['evidence']).replace('|', '/')} | {str(gate.get('next_step') or '-').replace('|', '/')} |"
        )
    lines.extend(["", "## Artifacts", ""])
    for name, path in doc["artifact_paths"].items():
        if path:
            lines.append(f"- {name}: `{path}`")
    if doc["limitations"]:
        lines.extend(["", "## Limitations / Unmet Targets", ""])
        for item in doc["limitations"]:
            lines.append(f"- {item}")
    if doc.get("known_limitations"):
        lines.extend(["", "## Known Guarded Limitations", ""])
        for item in doc["known_limitations"]:
            lines.append(f"- {item}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a release-candidate audit report for a Hacky Track run")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--qa-manifest", type=Path)
    parser.add_argument("--review-batch", type=Path)
    parser.add_argument("--strict-rally-audit", type=Path)
    parser.add_argument("--qa-reviewed-manifest", type=Path)
    parser.add_argument("--strict-rally-audit-reviewed", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--release-metrics", type=Path)
    parser.add_argument("--dataset-splits", type=Path)
    parser.add_argument("--release-evaluation-report", type=Path)
    parser.add_argument("--ball-audit", type=Path)
    parser.add_argument("--hud-video", type=Path)
    parser.add_argument("--hud-summary", type=Path)
    parser.add_argument("--hud-verify", type=Path)
    parser.add_argument("--hud-error", type=Path)
    parser.add_argument("--tests-passed", action="store_true")
    parser.add_argument("--test-evidence", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    out_dir = args.out_dir or run_dir / "release_report"
    overrides = {
        key: value
        for key, value in {
            "qa_manifest": args.qa_manifest,
            "review_batch": args.review_batch,
            "strict_rally_audit": args.strict_rally_audit,
            "qa_reviewed_manifest": args.qa_reviewed_manifest,
            "strict_rally_audit_reviewed": args.strict_rally_audit_reviewed,
            "validation": args.validation,
            "release_metrics": args.release_metrics,
            "dataset_splits": args.dataset_splits,
            "release_report": args.release_evaluation_report,
            "ball_audit": args.ball_audit,
            "hud_video": args.hud_video,
            "hud_summary": args.hud_summary,
            "hud_verify": args.hud_verify,
            "hud_error": args.hud_error,
        }.items()
        if value is not None
    }
    doc = build_report(run_dir, tests_passed=args.tests_passed, test_evidence=args.test_evidence, overrides=overrides)
    json_path, md_path = write_report(doc, out_dir)
    print(f"release report: {json_path}")
    print(f"release report md: {md_path}")


if __name__ == "__main__":
    main()
