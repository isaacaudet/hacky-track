# Hacky Track Release Goal

Turn Hacky Track from a working prototype into a releasable tool that other people can install, run on their own POV/Meta footbag videos, review uncertain events, export analytics, and render the sprite-sheet HUD without hand-editing JSON or relying on one local machine layout.

## Limitations To Solve

- Replace heuristic-only detection with a trained/retrained model stack using reviewed labels, while keeping CV/audio fallback paths for auditability.
- Promote ball detection to a custom `footbag` detector/segmenter trained from reviewed positives and hard negatives. HSV/red-blob snapping must become an auditable fallback, not a release gate.
- Export reviewed detector datasets in a portable YOLO-style layout, with hard-negative point/crop manifests for frames where the current marker is visibly not the bag.
- Keep detector labels separate from event labels: a hand-held bag is a positive `footbag` object but a rejected touch event; a marker on a leg/grass is a hard-negative ball-center point.
- Separate old exploratory labels from clean train/validation/test metrics.
- Improve left/right/contact-type classification with pose/limb evidence and calibrated `unknown` states.
- Keep knee detection candidate-only until enough reviewed knee data exists.
- Improve stall detection from conservative candidates into real start/end/duration/support/release windows.
- Add active-learning review for likely missed touches, drops, stalls, and tricks, not just already-detected candidates.
- Package scripts, docs, tests, and outputs so a new user can install, run, review, and export results from documented commands.

## Required Deliverables

- Public CLI for processing videos into one versioned run directory.
- Browser review app for approve/reject, missing touches, drops, stall windows, side/type/trick labels.
- Model training/evaluation pipeline with saved versioned artifacts.
- Detector dataset export: `data.yaml`, split images/labels, reviewed label JSONL, hard-negative crops/points, and a manifest with counts and split provenance.
- Detector-label review mining: QA ball-center candidates, preview crops/contact sheets, and decision templates so more object labels can be reviewed without polluting clean metrics.
- Detector trainer wrapper: documented optional dependencies, public CLI command, versioned training manifest, and saved best-model artifact when training succeeds.
- Detector model sanity gate: evaluate trained checkpoints directly on exported YOLO labels before allowing them into video inference, including release-confidence pass rate, low-confidence near-label failures, center error, and hard-negative false-positive rate.
- Trained second-stage objectness path: reviewed crop/patch classifier over auditable candidate proposals, used to test whether small-footbag detection improves before full detector promotion.
- Inference pipeline with trained-detector detections, run-level batch inference, tracker smoothing, confidence, and uncertainty reasons on every ball-track point/event.
- QA adapter that attaches model ball-track evidence to each event, preserves heuristic evidence for audit, and only promotes model centers under explicit validation gates.
- Detector-specific evaluation against reviewed positive labels and hard-negative points, with split metrics that must pass before the model replaces heuristic ball centers.
- Portable dataset/run schema with no hardcoded `/Users/...` paths in release-facing outputs.
- Reports: `summary.md`, `events.json`, `events.csv`, `rallies.json`, QA sheets, validation metrics, and HUD MP4.
- Sprite-sheet HUD renderer that works for arbitrary rally counts and durations.
- README with install, quickstart, review workflow, training workflow, troubleshooting, and known limitations.
- Unit, integration, and verifier tests.
- Release candidate changelog.

## Metric Targets

- Ball tracking: at least 95% event-center pass rate on held-out reviewed data.
- Touch detection: at least 90% precision and 85% recall.
- Duplicate touches: under 5% of accepted touches.
- Drop/floor resets: at least 90% precision and 90% recall.
- Stall windows: at least 85% precision and 80% recall.
- Side classification: at least 85% accuracy, with ambiguous cases kept `unknown`.
- Contact type: at least 85% accuracy.
- Knee: released only with at least 20 reviewed knee examples and at least 80% held-out precision; otherwise candidate-only.
- Tricks: released only with reviewed examples and at least 80% precision.

## Stop Gates

- Full release pipeline runs on all 27 existing videos.
- A smoke test runs on one video through the same public command.
- A review UI decision is created or updated through the app and proven to affect analytics.
- Training/evaluation saves model artifacts and clean split metrics.
- A trained detector must pass the model-level sanity gate on reviewed held-out labels before it can be promoted into rally analytics or HUD selection.
- Known failures stay fixed: video-352 drop near 22.9s, dropped rallies excluded from best-rally selection, and duplicate floor resets suppressed.
- HUD MP4 passes ffprobe audio/video checks, OpenCV nonblank-frame checks, clipping checks, and occlusion checks.
- Tests pass.
- Docs are sufficient for a new user.
- Final release report lists artifact paths, commands, model metrics, remaining limitations, and any unmet or waived targets.
