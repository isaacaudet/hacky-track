# Hacky Track Touch Detection Release Candidate

## Status

`release_gate_passed`

This release candidate promotes the touch pipeline to merged event-level output. Raw cue candidates are still tracked for diagnostics, but the product-facing touch output is the event-level merge/NMS result.

## Reproduction Command

```bash
python3 run_touch_pipeline.py \
  --detections-jsonl runs/release-27-public/touch_corpus_v1/owlv2_touch_detections_v1/detections.jsonl \
  --attach-audio-features
```

Primary status artifact:

```text
runs/release-27-public/touch_corpus_v1/touch_pipeline_status.md
```

## Release Gates

| split | gate level | precision | recall | f1 | fp | fn | result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| leave-clips-out CV | merged event | 0.905 | 0.894 | 0.899 | 8 | 9 | PASS |
| frozen test | merged event | 0.952 | 0.979 | 0.965 | 7 | 3 | PASS |

Gate thresholds:

| metric | threshold |
| --- | ---: |
| touch precision | >= 0.90 |
| touch recall | >= 0.85 |

The top-level pipeline status reports:

```text
Status: release_gate_passed
Release gate level: merged_event
Leave-clips-out CV gate: True
Frozen-test gate: True
```

## HUD Status

`passed`

The release HUD now renders from merged event-level classifier output instead of the legacy QA manifest path.

Reproduction command:

```bash
python3 render_touch_release_hud.py
```

Rendered set:

| group | videos |
| --- | ---: |
| frozen-test clips | 6 |
| train sanity clip | 1 |
| total HUD videos | 7 |

HUD verification:

| check | result |
| --- | --- |
| video stream present | PASS |
| audio stream present | PASS |
| sampled frames nonblank | PASS |
| touch anchors | 156 / 156 from L2-clean interpolated OWLv2 centers |
| HSV/color fallback for touch sparks | not used |

HUD artifacts:

| artifact | path |
| --- | --- |
| HUD render report | `runs/release-27-public/touch_corpus_v1/release_touch_hud_v1/release_touch_hud_report.md` |
| HUD manifest | `runs/release-27-public/touch_corpus_v1/release_touch_hud_v1/release_touch_hud_manifest.json` |
| Preview sheet | `runs/release-27-public/touch_corpus_v1/release_touch_hud_v1/release_touch_hud_preview_sheet.jpg` |

## What Changed

- The classifier still uses the fixed OWLv2 detector cache and does not lower the detector threshold.
- The release model excludes optical-flow and pose features by default; those remain diagnostic until they prove useful.
- Candidate-level decisions are merged into event-level touches using duplicate merge/NMS.
- Precision vetoes suppress weak/no-trajectory candidate fires.
- Recall rescue restores high-audio, high-impulse events that the classifier under-scores.
- Frozen-test rows remain held out from training and cross-validation.

Current output rules over existing features:

| rule | role |
| --- | --- |
| `no_trajectory_corroboration` | precision veto |
| `weak_audio_weak_trajectory` | precision veto |
| `high_impulse_audio_trajectory_rescue` | recall rescue |

## Validation Artifacts

| artifact | path |
| --- | --- |
| Pipeline status | `runs/release-27-public/touch_corpus_v1/touch_pipeline_status.md` |
| Classifier report | `runs/release-27-public/touch_corpus_v1/touch_classifier_v1/touch_classifier_report.md` |
| Classifier metrics | `runs/release-27-public/touch_corpus_v1/touch_classifier_v1/touch_classifier_metrics.json` |
| Frozen event output | `runs/release-27-public/touch_corpus_v1/touch_classifier_v1/touch_classifier_frozen_events.jsonl` |
| Out-of-fold event output | `runs/release-27-public/touch_corpus_v1/touch_classifier_v1/touch_classifier_oof_events.jsonl` |
| Event error audit | `runs/release-27-public/touch_corpus_v1/event_error_audit_v1/event_error_audit_report.md` |
| Trajectory drilldown | `runs/release-27-public/touch_corpus_v1/trajectory_error_drilldown_v1/trajectory_error_drilldown_report.md` |
| Release HUD report | `runs/release-27-public/touch_corpus_v1/release_touch_hud_v1/release_touch_hud_report.md` |

## Verification

Compile check:

```bash
python3 -m py_compile train_touch_classifier.py trajectory_error_drilldown.py event_error_audit.py run_touch_pipeline.py render_touch_release_hud.py tests/test_train_touch_classifier.py
```

Touch pipeline test suite:

```bash
python3 -m unittest \
  tests.test_build_touch_review_montages \
  tests.test_prefill_touch_labels_from_reviews \
  tests.test_touch_review_app \
  tests.test_touch_label_readiness \
  tests.test_attach_touch_l2_features \
  tests.test_export_touch_owlv2_detections \
  tests.test_run_touch_pipeline \
  tests.test_touch_pipeline_end_to_end \
  tests.test_attach_touch_pose_features \
  tests.test_build_touch_review_contact_sheets \
  tests.test_import_existing_touch_labels \
  tests.test_attach_touch_flow_features \
  tests.test_train_touch_classifier \
  tests.test_prefill_train_touch_suggestions \
  tests.test_build_touch_training_table \
  tests.test_attach_touch_audio_features
```

Current result: `86` tests pass.

## Remaining Risks

- This is a touch-detection release candidate, not a full trick/side/contact-type release.
- Candidate-level CV still fails; the release pass depends on merged event-level output, which is the intended product output.
- `video-344_singular_display-2` remains the weakest leave-one-video-out clip. Its remaining misses are mostly weak trajectory impulse, touch/stall overlap, or low classifier score.
- Remaining leave-clips-out errors are concentrated in trajectory artifacts and track gaps, not OWLv2 detection thresholding.
- Label coverage is adequate for this release gate, but more visually reviewed clips would harden the classifier and shrink confidence intervals.
- The cached OWLv2 detections file is large, so full status refreshes are slower than ideal.

## Next Hardening Work

1. Improve the trajectory layer on `video-344_singular_display-2` before adding new classifier complexity.
2. Add a compact indexed detector-track cache so full pipeline refreshes do not rescan the full OWLv2 JSONL each run.
3. Keep pose/body proximity as a soft diagnostic feature until it demonstrates held-out lift.
4. Only after touch timing is stable, resume side/contact-type classification.
