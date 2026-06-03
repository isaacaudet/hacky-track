# Hacky Track Release Readiness Plan

## Release Definition

Hacky Track should be released in two explicit layers.

### v0.1: Touch HUD Release

This is the near-term release target.

It includes:

- Fixed OWLv2 ball centers, L2 trajectory features, audio features, and merged event-level touch output.
- HUD rendering from release touch events, with OWLv2/L2 touch spark anchors.
- Model-only reports and optional visual-corrected HUD reports.
- Rally analytics: touch count, best rally, touch rate, longest gap, exact missed/fake touch times.
- Review/correction workflow for output-only visual overrides.
- Honest limitations for side/contact type, stall/drop automation, and tricks.

It does not claim:

- Reliable left/right/knee/contact-type classification.
- Automatic stall/drop detection as model predictions.
- Trick recognition.
- Fully generalized new-user detector training.

### v1.0: Rally Intelligence Release

This is the next product layer after v0.1.

It requires:

- Pose/body proximity features attached to touch candidates.
- Reviewed contact labels across at least 3 videos and at least 20 examples per promoted class.
- Automatic stall/drop evaluation that clears release gates.
- Contact side/type evaluation that clears release gates.
- HUD badges for left/right/knee/stall/drop/trick only after their gates pass.

## Current Status

v0.1 touch timing is release-ready:

- Leave-clips-out CV merged event gate passes.
- Frozen-test merged event gate passes.
- Visual-corrected frozen HUD set passes video/audio/nonblank checks.
- Frozen-test visual-corrected HUD analytics: precision `1.000`, recall `0.993`, F1 `0.996`, false positives `0`, missed touches `1`.
- One-command release workflow writes `runs/release-27-public/touch_release_v0_1/touch_release_readiness.md` with status `release_ready_v0_1`.
- Full test suite passes: `224/224`.

Current non-goal status:

- Contact classifier is `not_ready`.
- Pose/body proximity rows are missing from the release contact classifier input.
- Reviewed left/right/contact labels are missing.
- Stall/drop badges are label-backed display events only.

## v0.1 Blockers And Evidence

### 1. Public One-Command Release Path

The release path used to work through separate scripts:

- `run_touch_pipeline.py`
- `render_touch_release_hud.py`
- `release_rally_analytics.py`
- `release_event_error_audit.py`
- `train_release_contact_classifier.py`

Release-ready means a user can run one documented command that produces one versioned output directory with status, HUD videos, analytics, and limitations.

Current command:

```bash
python3 hackytrack.py touch-release \
  --out-dir runs/release-27-public/touch_release_v0_1 \
  --touch-overrides release_overrides/touch_visual_overrides_v1.json \
  --overwrite
```

Acceptance:

- One command writes a versioned release directory. DONE.
- It runs the touch pipeline from cached detections or exports detections when requested. DONE.
- It renders HUD overlays. DONE.
- It writes model-only and visual-corrected analytics when overrides are provided. DONE.
- It runs contact-classifier readiness and records blockers. DONE.
- It writes a final release readiness report. DONE.

### 2. Portable Output Paths

Release-facing reports should avoid hardcoded `/Users/...` paths where possible. Internal manifests can keep absolute paths for local reproducibility, but user-facing reports should prefer repo-relative paths.

Acceptance:

- Final release report uses repo-relative artifact paths. DONE.
- README commands are copy-pasteable from repo root. DONE.
- No release-facing Markdown artifact depends on one local machine layout. DONE.

### 3. Final Release Readiness Report

The release needs a single report that says what passed, what did not, and what is intentionally out of scope.

Acceptance:

- Includes touch CV and frozen-test metrics. DONE.
- Includes HUD render verification. DONE.
- Includes visual-corrected frozen metrics when overrides exist. DONE.
- Includes contact/stall/drop/trick readiness status. DONE.
- Includes exact commands used. DONE.
- Includes final limitations and next release work. DONE.

### 4. Fresh Smoke Run

Before tagging release, run the one-command path on at least one representative video or frozen-test set.

Acceptance:

- HUD MP4 has video, audio, and nonblank sampled frames. DONE.
- Preview sheet renders. DONE.
- Analytics report matches expected metrics. DONE.
- Tests pass after the smoke run. DONE.

## v1.0 Blockers

These are not v0.1 blockers unless the release is re-scoped to require full rally intelligence.

### Contact Side/Type

Needs:

- Pose/body proximity features attached to candidates.
- Reviewed labels for left/right/contact type.
- Clip-disjoint evaluation.

Promotion gate:

- At least 20 reviewed examples per promoted class.
- At least 3 labeled videos.
- At least 85% held-out contact-type accuracy.
- At least 85% side accuracy, with ambiguous cases left `unknown`.

### Stall/Drop Automation

Needs:

- Reviewed stall windows and drop/floor-reset events.
- Separate prediction/evaluation path from generic touches.
- HUD badges only after gates pass.

Promotion gate:

- Drop/floor reset precision and recall at least 90%.
- Stall precision at least 85%, recall at least 80%.

### Tricks/Knee

Needs:

- Reviewed examples and candidate-specific labels.

Promotion gate:

- Knee: at least 20 reviewed examples and at least 80% held-out precision.
- Tricks: at least 80% held-out precision before display as product facts.

## Immediate Engineering Order

1. Add a public release command or wrapper for the v0.1 path. Current command:
   `python3 hackytrack.py touch-release`.
2. Generate a final v0.1 release readiness report from actual artifacts.
3. Make report paths release-facing and repo-relative.
4. Run the release command on the frozen-test HUD set and one train sanity clip.
5. Keep side/contact/stall/drop/tricks in the report as explicit not-ready gates, not silent omissions.
6. After v0.1 is stable, attach pose features and label contact types for v1.0.

## Definition Of Done For v0.1

- `python3 -m unittest discover tests` passes.
- Public release command runs from repo root.
- Final report says `release_ready_v0_1` or gives explicit blockers.
- HUD videos render and pass video/audio/nonblank verification.
- Touch timing passes precision/recall gates at merged event level.
- README explains install, run, review/correct, and known limitations.
- Release candidate commit is pushed.
