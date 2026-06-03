# Hacky Track v0.1 Release Readiness Report

## Verdict

`release_ready_v0_1`

Hacky Track is defensible as a **v0.1 touch/HUD release**:

- Generic touch timing is release-ready at merged event level.
- HUD rendering works from release touch events with OWLv2/L2 ball anchors.
- Model-only and visual-corrected output workflows are both verified.
- Rally analytics are available for touch count, best rally, touch rate, longest gap, and exact error times.
- Contact side/type, automatic stall/drop, and trick recognition are explicitly not promoted.

This is **not** a full rally-intelligence release.

## Release Scope

### Included

- Fixed OWLv2 detection cache as the L1 ball source.
- L2 trajectory features.
- Audio features.
- Fused audio + trajectory touch classifier.
- Merged event-level touch output.
- Release HUD overlays.
- Visual correction override workflow for output-only HUD corrections.
- Rally analytics and touch-error reporting.
- Contact/type readiness gate that reports blockers instead of fabricating labels.

### Not Included

- Reliable left/right/knee/contact-type classification.
- Automatic stall/drop prediction.
- Trick recognition.
- New custom detector promotion.
- Claims that reviewed visual overrides are model-training data.

## Canonical Command

Run from repo root:

```bash
python3 hackytrack.py touch-release \
  --out-dir runs/release-27-public/touch_release_v0_1 \
  --touch-overrides release_overrides/touch_visual_overrides_v1.json \
  --overwrite
```

That command refreshes the touch pipeline, renders model-only HUDs, renders
visual-corrected HUDs, runs frozen-test analytics, checks contact/type readiness,
and writes the final run report:

```text
runs/release-27-public/touch_release_v0_1/touch_release_readiness.md
```

## Evidence From Full v0.1 Run

Full run artifact:

```text
runs/release-27-public/touch_release_v0_1/touch_release_readiness.md
```

Status:

```text
release_ready_v0_1
```

### Touch Classifier Gates

| split | precision | recall | F1 | FP | FN | gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| leave-clips-out CV | 0.916 | 0.894 | 0.905 | 7 | 9 | PASS |
| frozen test | 0.986 | 0.993 | 0.989 | 2 | 1 | PASS |

Thresholds:

| metric | threshold |
| --- | ---: |
| touch precision | >= 0.90 |
| touch recall | >= 0.85 |

### HUD Verification

The full default release render produced:

- 6 frozen-test videos.
- 1 train sanity video.
- Model-only HUD set.
- Visual-corrected HUD set.

Both HUD sets passed renderer verification:

- video stream present
- audio stream present
- sampled frames nonblank
- OWLv2/L2 touch anchors present
- no HSV/color fallback for release touch sparks

Preview sheet:

```text
runs/release-27-public/touch_release_v0_1/hud_visual_corrected/release_touch_hud_preview_sheet.jpg
```

### HUD Analytics

| HUD set | frozen videos | precision | recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| model-only frozen HUD | 6 | 0.986 | 0.993 | 0.989 | 2 | 1 |
| visual-corrected frozen HUD | 6 | 1.000 | 0.993 | 0.996 | 0 | 1 |

The visual override file removes only two reviewed false-positive HUD events on
`video-68_singular_display`. It is an output-correction artifact, not model
training data and not classifier gate evidence.

Remaining visual-corrected frozen-test error:

| video | type | time |
| --- | --- | ---: |
| `video-63_singular_display` | missed touch | 10.652s |

### Rally Analytics

The release analytics report lists per-video:

- predicted touches
- reviewed truth touches
- precision/recall
- rally count
- best rally
- exact false-positive times
- exact missed-touch times

Analytics artifact:

```text
runs/release-27-public/touch_release_v0_1/hud_visual_corrected/analytics_frozen_only/release_rally_analytics.md
```

## Contact/Type Status

Contact side/type is not release-ready.

Current contact classifier status:

```text
not_ready
```

Blockers:

- pose/body proximity columns are missing
- 0 rows with reviewed contact labels
- fewer than 3 videos with reviewed contact labels
- fewer than 2 contact or side classes

Contact classifier artifact:

```text
runs/release-27-public/touch_release_v0_1/contact_classifier/release_contact_classifier_report.md
```

## Known Limitations

- v0.1 outputs generic touches only.
- Side/contact-type/knee badges are not promoted.
- Stall/drop badges are label-backed display events only when reviewed labels exist.
- Automatic stall/drop prediction is not promoted.
- Trick recognition is not promoted.
- Current evidence is strong for the reviewed corpus, but additional visual labels would shrink confidence intervals and harden generalization.
- `video-344_singular_display-2` remains the weakest leave-one-video-out clip for touch recall.

## Tests

Current verification:

```bash
python3 -m py_compile hackytrack.py render_touch_release_hud.py release_rally_analytics.py train_release_contact_classifier.py train_touch_classifier.py tests/test_hackytrack_touch_release.py
python3 -m unittest discover tests
```

Result:

```text
224/224 tests pass
```

## Release Decision

Ship as:

```text
Hacky Track v0.1 Touch HUD Release
```

Do not ship as:

```text
Full rally intelligence / contact-type / trick-classification release
```

## Next Release Layer

v1.0 work should focus on:

1. Attach pose/body proximity features to candidate rows.
2. Add reviewed contact side/type labels across at least 3 videos.
3. Promote side/contact classification only after held-out gates pass.
4. Build automatic stall/drop prediction and evaluate separately.
5. Render left/right/knee/stall/drop/trick HUD badges only after those gates are real.
