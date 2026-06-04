# Hacky Track Touch Detection Release Candidate

## Status

`release_gate_passed`

This release candidate promotes the touch pipeline to merged event-level output. Raw cue candidates are still tracked for diagnostics, but the product-facing touch output is the event-level merge/NMS result.

Current status is intentionally split:

- Model-only merged events pass both leave-clips-out CV and frozen-test touch gates when the complete OWLv2 detection cache set is supplied.
- Visual-corrected frozen-test HUDs are excellent and remain useful release preview artifacts.
- Side/surface/contact intelligence remains separate from touch timing and is not v1.0-ready.

## Reproduction Command

```bash
python3 run_touch_pipeline.py \
  --detections-jsonl runs/release-27-public/touch_corpus_v1/owlv2_touch_detections_v1/detections.jsonl \
  --detections-jsonl runs/release-27-public/touch_corpus_v1/owlv2_touch_detections_contact_missing_v1/detections.jsonl \
  --attach-audio-features
```

Primary status artifact:

```text
runs/release-27-public/touch_corpus_v1/touch_pipeline_status.md
```

## Release Gates

| split | gate level | precision | recall | f1 | fp | fn | result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| leave-clips-out CV | merged event | 0.925 | 0.896 | 0.910 | 7 | 10 | PASS |
| frozen test | merged event | 0.979 | 0.986 | 0.982 | 3 | 2 | PASS |

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

The prior CV recall blocker was caused by running with only
`owlv2_touch_detections_v1/detections.jsonl`, which omitted
`video-340_singular_display-2`. Including
`owlv2_touch_detections_contact_missing_v1/detections.jsonl` restores L2
trajectory features for that clip and moves event-level CV over gate. The
remaining weak automatic-touch clip is `video-344_singular_display-2`, but the
aggregate merged-event gate now passes.

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

## Rally Analytics + Contact Status

The release HUD adapter now supports reviewed `stall` and `drop_floor` labels in
the HUD event doc. Touches remain classifier predictions; stall/drop labels are
rendered only when reviewed labels exist.

Current smoke render:

```bash
python3 render_touch_release_hud.py \
  --out-dir runs/release-27-public/touch_corpus_v1/release_touch_hud_v5 \
  --video-id video-68_singular_display \
  --video-id video-296_singular_display-2
```

Smoke status:

| video | touches | stalls | drops | anchors | verification |
| --- | ---: | ---: | ---: | ---: | --- |
| `video-68_singular_display` | 89 | 0 | 0 | 89 | video/audio/nonblank PASS |
| `video-296_singular_display-2` | 11 | 4 | 0 | 15 | video/audio/nonblank PASS |

The long-clip audit now reports exact miss/fake times instead of relying on
visual impression:

```bash
python3 release_rally_analytics.py \
  --hud-dir runs/release-27-public/touch_corpus_v1/release_touch_hud_v5 \
  --out-dir runs/release-27-public/touch_corpus_v1/release_touch_hud_v5/analytics
```

| video | precision | recall | false positives | missed touches |
| --- | ---: | ---: | --- | --- |
| `video-68_singular_display` | 0.978 | 1.000 | `[4.330667, 19.786667]` | `[]` |
| `video-296_singular_display-2` | 1.000 | 0.917 | `[]` | `[4.843]` |

The `video-68` errors also have visual strips with local ball track, reviewed
touch markers, predicted touch markers, cue-level audio strength, trajectory
break support, and candidate gate state:

```bash
python3 release_event_error_audit.py \
  --errors-jsonl runs/release-27-public/touch_corpus_v1/release_touch_hud_v5/analytics/release_rally_event_errors.jsonl \
  --out-dir runs/release-27-public/touch_corpus_v1/release_touch_hud_v5/event_error_audit \
  --video-id video-68_singular_display
```

Failure-mode histogram:

| mode | count |
| --- | ---: |
| loud footstep with ball motion nearby | 2 |

Audit contact sheet:

```text
runs/release-27-public/touch_corpus_v1/release_touch_hud_v5/event_error_audit/release_event_error_audit_contact_sheet.jpg
```

Visual-corrected HUD render:

```bash
python3 render_touch_release_hud.py \
  --out-dir runs/release-27-public/touch_corpus_v1/release_touch_hud_v6_visual_corrected \
  --touch-overrides release_overrides/touch_visual_overrides_v1.json \
  --video-id video-68_singular_display \
  --video-id video-296_singular_display-2
```

The override file removes only two visually audited `video-68` false positives
from HUD rendering. It is not used as model-training data or classifier gate
evidence.

| video | precision | recall | false positives | missed touches |
| --- | ---: | ---: | --- | --- |
| `video-68_singular_display` | 1.000 | 1.000 | `[]` | `[]` |
| `video-296_singular_display-2` | 1.000 | 0.917 | `[]` | `[4.843]` |

Full frozen-test visual-corrected HUD set with reviewed manual contact badges:

```bash
python3 render_touch_release_hud.py \
  --out-dir runs/release-27-public/touch_corpus_v1/release_touch_hud_v8_contact_badges_frozen_corrected \
  --touch-overrides release_overrides/touch_visual_overrides_v1.json

python3 release_rally_analytics.py \
  --hud-dir runs/release-27-public/touch_corpus_v1/release_touch_hud_v8_contact_badges_frozen_corrected \
  --out-dir runs/release-27-public/touch_corpus_v1/release_touch_hud_v8_contact_badges_frozen_corrected/analytics_frozen_only \
  --video-id video-439_singular_display \
  --video-id video-478_singular_display \
  --video-id video-482_singular_display \
  --video-id video-486_singular_display \
  --video-id video-63_singular_display \
  --video-id video-68_singular_display
```

Frozen-test corrected HUD status:

| scope | videos | precision | recall | F1 | false positives | missed touches | manual badges | side/surface badges |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| frozen-test corrected HUD | 6 | 1.000 | 0.993 | 0.996 | 0 | 1 | 64 | 64/21 |

The only remaining frozen-test corrected miss is
`video-63_singular_display` at `10.652s`. The default v8 render also includes
one train sanity clip (`video-234_singular_display-2`); that clip is excluded
from the frozen-only release metric above.

Side/surface/contact classification remains separate from touch timing. Current
clip-disjoint contact metrics are:

| target | rows | selected model | raw / balanced accuracy | gate |
| --- | ---: | --- | ---: | --- |
| contact type | 82 | ridge classifier, no visual crop and no tracking features | 0.963 / 0.786 | raw pass; release-scope fail until stall/knee/drop labels reach the floor |
| wearer side | 76 | LinearSVC, no visual/tracking features; diagnostic temporal smoothing | 0.776 / 0.752 raw, 0.816 / 0.791 smoothed | fail; smoothing remains diagnostic-only |
| inner/outer surface | 25 | gradient boosting, pose only | 0.800 / 0.643 | fail |
| drop/floor reset | 35 | ExtraTrees over OWLv2/L2 + floor/context + merged-touch gap features | 0.682 P / 0.714 R | fail |
| stall | 32 | n/a | n/a | not ready: 3 approved stalls |

The contact classifier report now includes the exact label inventory and release
gaps: side has enough left/right count coverage but still fails accuracy;
diagnostic temporal smoothing improves side but still misses the 0.85 gate;
surface needs +13 inner and +2 outer labels; full contact type needs +13 stall,
+20 knee, and +20 drop_floor labels before automatic HUD badges can be promoted.

The HUD now shows reviewed contact labels as manual badges when a merged touch
matches a visual label. These are explicitly label-backed display facts, not
automatic side/surface classifier predictions.

Automatic stall/drop audit artifacts:

```text
runs/release-27-public/touch_corpus_v1/release_stall_drop_classifier_v1/release_stall_drop_classifier_report.md
runs/release-27-public/touch_corpus_v1/release_stall_drop_classifier_v1/error_audit/stall_drop_error_contact_sheet.jpg
runs/release-27-public/touch_corpus_v1/owlv2_stall_drop_missing_detections_v1/detections.jsonl
runs/release-27-public/touch_corpus_v1/touch_classifier_v1/touch_classifier_model_only_events_report.md
runs/release-27-public/touch_corpus_v1/release_stall_drop_classifier_model_only_fulltrack_v1/release_stall_drop_classifier_report.md
```

The model-only touch event stream fills 7 reset/stall clips that lacked
OOF/frozen release touch streams, producing 44 unreviewed model events. It is
not release evidence. In the full-track stall/drop ablation it raises touch
stream coverage from 20 to 27 videos, but automatic drop worsens from
0.682/0.714 to 0.652/0.714 precision/recall. The release path therefore keeps
model-only touch streams out of product decisions until a later held-out
ablation proves lift.

Side sequence-smoothing audit artifacts:

```text
runs/release-27-public/touch_corpus_v1/release_contact_classifier_v1/contact_error_audit/contact_side_sequence_smoothed_error_audit.jsonl
runs/release-27-public/touch_corpus_v1/release_contact_classifier_v1/contact_error_audit/sequence_smoothed_strips/
```

The smoothed-side audit leaves 14 held-out failures: 4 pose-side disagreements,
6 visual-ambiguity cases, and 4 pose-missing cases. This is why the side layer
stays diagnostic-only.

Foot-track feature artifact:

```text
runs/release-27-public/touch_corpus_v1/touch_training_dataset_v1/touch_foot_track_feature_report.md
```

The current implementation uses the prebuilt RTMW/rtmlib foot keypoints as the
foot-region backend and adds temporal continuity features around each candidate
(`foot_track_*`). It also writes label-derived `manual_foot_*` calibration fields
for reviewed/manual display workflows only. Those manual fields are deliberately
excluded from automatic classifier features. In clip-disjoint evaluation the
automatic foot-track features do not improve side or surface enough to promote
HUD badges, and the selected release-safe modes exclude them where they hurt.

CoTracker feature artifact:

```text
runs/release-27-public/touch_corpus_v1/touch_training_dataset_v1/touch_cotracker_feature_report.md
```

CoTracker3 was run on the current contact-labeled corpus with RTMW foot
landmarks as seeds. It produced 63 usable tracked windows out of 82 processed
contact-labeled candidates (11/17 train+validation, 52/65 frozen test). The
feature ablation still selects no-tracking modes for contact type and side;
side remains `0.776 / 0.752` and surface remains `0.800 / 0.643`. So CoTracker
is implemented and measured, but not promoted as an automatic side/surface
badge signal.

Supplemental detection-cache smoke check:

```bash
python3 render_touch_release_hud.py \
  --out-dir runs/release-27-public/touch_corpus_v1/release_touch_hud_v9_detection_cache_complete_smoke \
  --video-id video-340_singular_display-2 \
  --max-seconds 12 \
  --allow-missing-audio
```

This verifies the clip that was missing from the primary detection cache:

| video | touches | anchors | center source | video/audio/nonblank |
| --- | ---: | ---: | --- | --- |
| `video-340_singular_display-2` | 11 | 11 | `l2_clean_interpolated:11` | True/True/True |

## What Changed

- The classifier still uses the fixed OWLv2 detector cache and does not lower the detector threshold.
- The release model excludes optical-flow and pose features by default; those remain diagnostic until they prove useful.
- Candidate-level decisions are merged into event-level touches using duplicate merge/NMS.
- Precision vetoes suppress weak/no-trajectory candidate fires.
- Recall rescue restores high-audio, high-impulse events that the classifier under-scores.
- Final event-artifact vetoes suppress audited merged-event artifacts without removing any approved current-corpus touches.
- Frozen-test rows remain held out from training and cross-validation.
- Release HUD event docs can include reviewed stall/drop events with OWLv2/L2 anchors.
- Release HUD event docs can attach reviewed contact labels to matched touch events as manual badges.
- Rally analytics now reports aggregate and per-video best rally, touch rate, longest gap, rendered reviewed stall/drop counts, and exact FP/FN times.
- Rally analytics now reports manual contact badge coverage separately from automatic touch metrics.
- Contact side/type has a separate readiness/evaluation path instead of being mixed into touch timing.

Current output rules over existing features:

| rule | role |
| --- | --- |
| `no_trajectory_corroboration` | precision veto |
| `weak_audio_weak_trajectory` | precision veto |
| `high_impulse_audio_trajectory_rescue` | recall rescue |
| `soft_audio_strong_impulse_trajectory_rescue` | recall rescue |
| `high_score_audio_no_trajectory_rescue` | recall rescue |
| `very_weak_audio_artifact` | final event veto |
| `flat_control_or_duplicate_artifact` | final event veto |
| `non_ballistic_high_residual_artifact` | final event veto |
| `low_impulse_close_peak_trough_artifact` | final event veto |
| `top_of_arc_no_impulse_artifact` | final event veto |

## Validation Artifacts

| artifact | path |
| --- | --- |
| Pipeline status | `runs/release-27-public/touch_corpus_v1/touch_pipeline_status.md` |
| Classifier report | `runs/release-27-public/touch_corpus_v1/touch_classifier_v1/touch_classifier_report.md` |
| Classifier metrics | `runs/release-27-public/touch_corpus_v1/touch_classifier_v1/touch_classifier_metrics.json` |
| Frozen event output | `runs/release-27-public/touch_corpus_v1/touch_classifier_v1/touch_classifier_frozen_events.jsonl` |
| Out-of-fold event output | `runs/release-27-public/touch_corpus_v1/touch_classifier_v1/touch_classifier_oof_events.jsonl` |
| Frozen event vetoes | `runs/release-27-public/touch_corpus_v1/touch_classifier_v1/touch_classifier_frozen_event_vetoes.jsonl` |
| Out-of-fold event vetoes | `runs/release-27-public/touch_corpus_v1/touch_classifier_v1/touch_classifier_oof_event_vetoes.jsonl` |
| Event error audit | `runs/release-27-public/touch_corpus_v1/event_error_audit_v1/event_error_audit_report.md` |
| Trajectory drilldown | `runs/release-27-public/touch_corpus_v1/trajectory_error_drilldown_v1/trajectory_error_drilldown_report.md` |
| Release HUD report | `runs/release-27-public/touch_corpus_v1/release_touch_hud_v5/release_touch_hud_report.md` |
| Release rally analytics | `runs/release-27-public/touch_corpus_v1/release_touch_hud_v5/analytics/release_rally_analytics.md` |
| Release event-error strips | `runs/release-27-public/touch_corpus_v1/release_touch_hud_v5/event_error_audit/release_event_error_audit_report.md` |
| Visual override file | `release_overrides/touch_visual_overrides_v1.json` |
| Visual-corrected HUD report | `runs/release-27-public/touch_corpus_v1/release_touch_hud_v6_visual_corrected/release_touch_hud_report.md` |
| Visual-corrected analytics | `runs/release-27-public/touch_corpus_v1/release_touch_hud_v6_visual_corrected/analytics/release_rally_analytics.md` |
| Full frozen visual-corrected HUD report | `runs/release-27-public/touch_corpus_v1/release_touch_hud_v8_contact_badges_frozen_corrected/release_touch_hud_report.md` |
| Full frozen visual-corrected analytics | `runs/release-27-public/touch_corpus_v1/release_touch_hud_v8_contact_badges_frozen_corrected/analytics_frozen_only/release_rally_analytics.md` |
| Contact classifier status | `runs/release-27-public/touch_corpus_v1/release_contact_classifier_v1/release_contact_classifier_report.md` |

## Verification

Compile check:

```bash
python3 -m py_compile train_touch_classifier.py trajectory_error_drilldown.py event_error_audit.py release_event_error_audit.py run_touch_pipeline.py render_touch_release_hud.py release_rally_analytics.py train_release_contact_classifier.py tests/test_train_touch_classifier.py
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
  tests.test_attach_touch_audio_features \
  tests.test_release_rally_analytics \
  tests.test_render_touch_release_hud \
  tests.test_release_contact_classifier \
  tests.test_release_event_error_audit
```

Current result: `284` tests pass.

## Remaining Risks

- This is a touch-detection release candidate, not a full automatic trick/side/contact-type release.
- Long-video `video-68_singular_display` still has 2 visually audited fake touches and no misses at the 0.2s match tolerance.
- Stall/drop badges are label-backed display events, not automatic release predictions yet.
- Automatic drop/floor reset has now been evaluated separately with full OWLv2/L2 coverage; sequence-window/floor-context features improve the L2-only baseline from 0.609/0.667 to 0.652/0.714 P/R, and available merged-touch gap context improves it again to 0.682/0.714, but the reset gate still fails.
- A model-only touch-stream ablation increases reset/stall stream coverage to 27 videos, but worsens drop to 0.652/0.714 P/R, so it remains diagnostic-only.
- Drop/floor reset score-threshold diagnostics do not rescue the gate: the best F1 threshold is 0.35 with 0.667 precision / 0.952 recall, so the blocker is feature separability, not the default 0.5 threshold.
- Reviewed contact badges are manual display facts; automatic side/surface badges remain blocked by failed contact gates.
- Diagnostic side sequence smoothing improves held-out side from 0.776/0.752 to 0.816/0.791, but it is still below the 0.85 gate and is not used for automatic HUD badges.
- Candidate-level CV still fails; the release pass depends on merged event-level output, which is the intended product output.
- `video-344_singular_display-2` remains the weakest leave-one-video-out clip. Its remaining misses are mostly weak trajectory impulse, touch/stall overlap, or low classifier score.
- The release command must include both OWLv2 detection JSONLs above; omitting the contact-missing cache removes trajectory features for `video-340_singular_display-2` and reproduces the old recall failure.
- Remaining leave-clips-out errors are concentrated in trajectory artifacts and track gaps, not OWLv2 detection thresholding.
- Label coverage is adequate for this release gate, but more visually reviewed clips would harden the classifier and shrink confidence intervals.
- The cached OWLv2 detections file is large, so full status refreshes are slower than ideal.

## Next Hardening Work

1. Improve the trajectory layer on `video-344_singular_display-2` before adding new classifier complexity.
2. Add a compact indexed detector-track cache so full pipeline refreshes do not rescan the full OWLv2 JSONL each run.
3. Keep pose/body proximity as a soft diagnostic feature; pose-only surface modeling helps, but the gate still needs more inner/outer labels and better foot identity.
4. Only after touch timing is stable, resume side/contact-type classification.
