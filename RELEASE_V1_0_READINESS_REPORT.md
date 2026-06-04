# Hacky Track v1.0 Rally Intelligence Readiness

## Verdict

`not_ready_for_v1_0`

v0.1 generic touch timing and the OWLv2/L2 HUD path are release-shaped. v1.0 rally intelligence is the next layer: left/right, inner/outer, knee, stall/drop, tricks, and calibrated HUD badges. The current code now has the right training/evaluation scaffolding and the first contact classifier, but side and surface are not accurate enough to ship as broad product facts.

## Current Evidence

### Touch Timing / HUD Baseline

Current artifact:

```text
runs/release-27-public/touch_corpus_v1/touch_pipeline_status.md
```

Current merged-event touch status:

| split | precision | recall | F1 | gate |
| --- | ---: | ---: | ---: | --- |
| leave-clips-out CV | 0.925 | 0.896 | 0.910 | pass |
| frozen test | 0.979 | 0.986 | 0.982 | pass |

Interpretation:

- Frozen-test touch timing is release-shaped, especially with reviewed visual HUD overrides.
- Leave-clips-out CV now passes after including both OWLv2 detection caches in the L2 feature attachment path.
- The prior `video-340_singular_display-2` failure was a missing-cache problem, not a classifier weakness: the main cache omitted the clip, while `owlv2_touch_detections_contact_missing_v1/detections.jsonl` contains the needed detections.
- `video-344_singular_display-2` remains the weakest automatic-touch clip, but the aggregate merged-event gate now passes.

### Contact Classifier

Current artifact:

```text
runs/release-27-public/touch_corpus_v1/release_contact_classifier_v1/release_contact_classifier_report.md
```

Current matched reviewed labels:

| target | rows | classes |
| --- | ---: | --- |
| contact type | 82 | kick 75, stall 7 |
| side | 76 | right 51, left 25; all `wearer_limb` side basis |
| surface | 25 | outer 18, inner 7 |

Clip-disjoint leave-one-video-out results:

| target | accuracy | gate | selected feature mode | interpretation |
| --- | ---: | --- | --- | --- |
| contact type | 0.939 | raw pass / release-scope fail | no_visual_crop + gradient boosting | Raw kick/stall accuracy passes, but class coverage blocks full v1.0: stall 7, knee 0, drop_floor 0. |
| side | 0.750 | fail | no_visual_crop + ExtraTrees | Improved from 0.628, but still below the 0.85 side gate. |
| surface | 0.760 | fail | no_visual_crop + ExtraTrees | Still below gate; inner examples are the dominant misses. |

Feature-mode ablation:

| target | current best | prior logistic baseline | result |
| --- | ---: | ---: | --- |
| contact type | 0.939 | 0.902 | Gradient boosting is strongest, but release-scope coverage is still missing stall/knee/drop labels. |
| side | 0.750 | 0.628 | ExtraTrees beats majority baseline (0.671), but still misses many left contacts. |
| surface | 0.760 | 0.760 | Model family does not clear the small, imbalanced inner/outer set. |

Confidence/abstention does not rescue the failing targets:

- Side stays below gate; the new ExtraTrees model reaches 0.750 full coverage but high-confidence abstention does not rescue it.
- All current approved/training side labels are now `wearer_limb`, inferred from the side-specific trick labels you already reviewed.
- Surface stays below gate; selected model is 0.760 at full coverage and remains below 0.85 under confidence filtering.
- Contact type has a raw accuracy pass, but the new release-scope gate correctly keeps it unpromoted until stall/knee/drop_floor class coverage reaches the release floor.

### Pose / Body Proximity

Pose/body geometry is attached and cached:

```text
runs/release-27-public/touch_corpus_v1/touch_training_dataset_v1/touch_pose_feature_report.md
```

Current coverage:

| rows | pose present | usable pose distance |
| ---: | ---: | ---: |
| 624 | 281 | 259 |

Interpretation:

- RTMW/wholebody keypoints are useful as soft features.
- They are not reliable enough as hard gates in Ray-Ban Meta POV footage.
- Side is especially unreliable because model anatomical left/right and user-visible contact side are not stable in egocentric partial-foot frames.

### Visual Crop Features

New artifact:

```text
runs/release-27-public/touch_corpus_v1/touch_training_dataset_v1/touch_visual_crop_feature_report.md
```

Current coverage:

| split | rows | ok crop rows |
| --- | ---: | ---: |
| train + validation | 283 | 249 |
| frozen test | 341 | 277 |

Interpretation:

- Ball-centered visual crops are now attached as cached candidate-level features.
- They are useful only when paired with the stronger frozen OWLv2 embedding for surface classification.
- The classifier now selects feature groups per target so a net-negative visual branch cannot silently ship.

### Frozen Vision Embeddings

New artifact:

```text
runs/release-27-public/touch_corpus_v1/touch_training_dataset_v1/touch_vision_embedding_feature_report.md
```

Current reviewed-contact coverage:

| split | requested | ok embeddings |
| --- | ---: | ---: |
| train + validation | 17 | 17 |
| frozen test | 65 | 65 |

Interpretation:

- Reusing OWLv2 as a frozen crop encoder is technically viable and fully cached for reviewed contact labels.
- It is slow on MPS and should remain an explicit v1.0 prep step, not default release processing.
- Visual features improve inner/outer surface accuracy from 0.640 to 0.760, but still miss the 0.85 gate.
- It does not solve side; side remains a label/semantics/egocentric-foot-identity problem.

### Contact Error Audit

New artifact:

```text
runs/release-27-public/touch_corpus_v1/release_contact_classifier_v1/contact_error_audit/contact_error_audit_report.md
```

Current error buckets:

| bucket | count | meaning |
| --- | ---: | --- |
| pose_missing | 7 | Pose unavailable at contact time. |
| pose_side_disagreement | 6 | Pose side conflicts with the reviewed side. |
| side_visual_ambiguity | 6 | Image crop/embedding still cannot infer side reliably. |
| type_motion_ambiguity | 4 | Mostly stall/kick errors; needs dwell/control features and more stall labels. |
| surface_label_or_geometry_ambiguity | 4 | Inner/outer needs more labels despite embedding gains. |
| pose_surface_disagreement | 2 | Pose foot-edge geometry conflicts with reviewed surface. |
| stall_window_confusion | 1 | Stall context feature is still weak. |

Representative strips are rendered under:

```text
runs/release-27-public/touch_corpus_v1/release_contact_classifier_v1/contact_error_audit/strips/
```

### Side Semantics Audit

New artifact:

```text
runs/release-27-public/touch_corpus_v1/release_contact_classifier_v1/side_semantics_audit/side_semantics_audit_report.md
```

This audit compares reviewed `left/right` labels against four conventions:

- RTMW anatomical side from nearest lower-body/foot keypoints.
- Flipped RTMW anatomical side.
- Screen side of the ball.
- Flipped screen side.

Global mapping scores:

| predictor | covered | coverage | accuracy |
| --- | ---: | ---: | ---: |
| pose anatomical | 50 | 0.658 | 0.540 |
| pose flipped | 50 | 0.658 | 0.460 |
| screen ball | 76 | 1.000 | 0.553 |
| screen ball flipped | 76 | 1.000 | 0.447 |
| screen ball, 0.08 center deadzone | 15 | 0.197 | 0.733 |

Interpretation:

- No global convention is strong enough to use as a side classifier.
- One clip is consistent with flipped screen side, a few clips weakly match screen side, and several are mixed.
- Visual strips show both directions of failure: some true labels align with pose while screen side disagrees, and other clips have pose side inverted relative to the reviewed label.
- Side is therefore a semantic/reviewer-definition problem before it is a model-capacity problem.

Representative strips are rendered under:

```text
runs/release-27-public/touch_corpus_v1/release_contact_classifier_v1/side_semantics_audit/strips/
```

### HUD Contact Badges

New artifact:

```text
runs/release-27-public/touch_corpus_v1/release_touch_hud_v8_contact_badges_frozen_corrected/release_touch_hud_report.md
```

Frozen-test corrected HUD analytics:

```text
runs/release-27-public/touch_corpus_v1/release_touch_hud_v8_contact_badges_frozen_corrected/analytics_frozen_only/release_rally_analytics.md
```

Current frozen-test HUD status:

| scope | touch P/R/F1 | manual badges | side badges | surface badges | interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| frozen-test corrected HUD | 1.000 / 0.993 / 0.996 | 64 | 64 | 21 | Manual reviewed contact labels render correctly; automatic side/surface classifiers remain unpromoted. |

Interpretation:

- The HUD can now show reviewed `left_kick`, `right_outer_kick`, `left_inner_kick`, knee, and stall-style labels when they came from visual labels.
- These badges are label-backed/manual display facts, not model-predicted contact intelligence.
- Generic reviewed touches with no contact detail do not show a fake `reviewed_touch` badge.

## What Changed In This Pass

Code changes:

- `attach_touch_visual_crop_features.py` adds cached ball-centered visual crop descriptors and normalized ball position features.
- `attach_touch_vision_embedding_features.py` adds cached frozen OWLv2 crop embeddings for reviewed contact rows or all rows.
- `run_touch_pipeline.py` can attach visual crop features via `--attach-visual-crop-features` and reports their status.
- `run_touch_pipeline.py` can attach OWLv2 crop embeddings via `--attach-vision-embedding-features` and reports their status.
- `train_release_contact_classifier.py` now evaluates feature modes per target and stores selected target-specific modes in the model artifact.
- `train_release_contact_classifier.py` now evaluates bounded model families per target (`logistic_regression`, `extra_trees`, `gradient_boosting`) and stores the selected family in the model artifact.
- `train_release_contact_classifier.py` now reports selective accuracy by prediction confidence, so abstention claims are measurable.
- `train_release_contact_classifier.py` now separates raw accuracy gates from release-scope gates, so kick/stall accuracy cannot be promoted as full kick/knee/stall/drop intelligence until class coverage exists.
- `contact_error_audit.py` renders visual strips for current contact classifier errors, clears stale strips before rendering, and buckets failures by likely mode.
- `render_touch_release_hud.py` now attaches reviewed contact labels to matched merged touch events as manual HUD badges, with provenance and match deltas.
- `render_touch_release_hud.py` and `hackytrack.py touch-release` now accept multiple OWLv2 detection JSONLs, so supplemental contact-missing caches are not silently dropped.
- `run_touch_pipeline.py` now records the actual detection JSONL inputs in the status report and prints a reproducible release command using those inputs.
- `release_rally_analytics.py` now reports manual contact badge coverage separately from automatic touch metrics.
- `side_semantics_audit.py` audits side labels against pose and screen-side conventions, with visual disagreement strips.
- `touch_review_app.py` and contact-label parsing support the full label vocabulary: left/right kick, inner/outer, knee, stall, and ground/drop.
- `touch_review_app.py` now persists `contact_side_basis` and explicitly defines side as the contacting limb / wearer side, not screen-left/right.
- `touch_review_app.py` and `train_release_contact_classifier.py` now infer `contact_side_basis=wearer_limb` from side-specific trick labels such as `left_inner_kick` and `right_stall`.
- `touch_review_app.py` now surfaces side-basis debt directly: progress shows usable wearer-side labels vs legacy left/right labels, and `Next side-basis` jumps through legacy side labels that need explicit basis review.
- `touch_review_app.py` now includes side-basis debt in the global summary, video list, and review queue, with a `Load next side-basis clip` path separate from touch-hint review.
- `train_release_contact_classifier.py` preserves side-basis metadata and ignores side labels explicitly marked `screen_position`, `pose_anatomical`, `ambiguous`, or `unknown` for wearer-side training.
- `migrate_contact_side_basis.py` promoted 69 previously saved side-specific trick labels to explicit `wearer_limb` basis in 6 reviewed label files.

Browser smoke artifact:

```text
runs/release-27-public/touch_corpus_v1/release_contact_classifier_v1/side_semantics_audit/review_app_side_basis_smoke.png
runs/release-27-public/touch_corpus_v1/release_contact_classifier_v1/side_semantics_audit/review_app_side_basis_progress_smoke.png
runs/release-27-public/touch_corpus_v1/release_contact_classifier_v1/side_semantics_audit/review_app_side_basis_queue_smoke.png
runs/release-27-public/touch_corpus_v1/release_contact_classifier_v1/side_semantics_audit/review_app_side_basis_migrated_smoke.png
```

Migration artifact:

```text
runs/release-27-public/touch_corpus_v1/release_contact_classifier_v1/side_semantics_audit/contact_side_basis_migration_report.json
```

Tests:

```text
python3 -m unittest discover tests
```

Current result:

```text
Ran 263 tests in 7.793s
OK
```

## Library Reality Check

Side/contact-surface detection is not solved by a library. Current libraries solve foot landmarks, not footbag contact semantics.

- RTMW/wholebody pose estimates body, face, hand, and foot keypoints, including foot landmarks, which is the right landmark source.
- COCO-WholeBody-style feet expose keypoints, not semantic “inside/outside kick” labels.
- MediaPipe-style pose/wholebody can provide heel/toe landmarks, but not footbag-specific contact classification.

So the correct v1.0 path is custom classification on top of:

- OWLv2/L2 ball center and trajectory.
- RTMW foot/knee landmarks as soft features.
- Visual crop embeddings/classifier for the local ball+foot patch.
- Reviewed side/type/surface/stall/drop labels.

Sources used for this conclusion:

- [RTMW paper](https://arxiv.org/abs/2407.08634)
- [RTMW model card](https://huggingface.co/akore/rtmw-l-384x288)
- [COCO-WholeBody dataset summary](https://paperswithcode.com/dataset/coco-wholebody)
- [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)

## v1.0 Promotion Gates

Do not render these as product facts until the gate passes.

| signal | minimum labels | release gate | current state |
| --- | ---: | --- | --- |
| left/right side | >=20 explicit wearer-limb labels per promoted class, >=3 videos | >=85% clip-disjoint side accuracy | 76 wearer-limb rows, 0.750 accuracy, fail |
| inner/outer surface | >=20 per promoted class, >=3 videos | >=85% clip-disjoint surface accuracy | 25 rows, 0.760 accuracy, fail |
| contact type: kick/knee/stall/drop | >=20 per promoted class, >=3 videos | >=85% contact-type accuracy | 82 rows, 0.939 raw accuracy, release-scope fail: stall 7, knee 0, drop_floor 0 |
| drop/floor reset | enough reviewed positives and negatives across clips | precision >=90%, recall >=90% | 35 clean reviewed reset rows, 0.682 P / 0.714 R after rally-sequence reset features, fail |
| stall | enough reviewed stall windows and non-stall controls | precision >=85%, recall >=80% | 32 clean reviewed stall candidates, but only 3 approved stalls, not-ready |
| tricks | >=20 examples per promoted trick | >=80% held-out precision | not ready |

## Next Engineering Work

The next high-yield work is not more scalar pose tweaks. The frozen embedding branch helped surface but did not solve side, so the remaining work is better labels plus a side-specific egocentric foot-identity strategy.

1. Improve side semantics:
   - The audit shows current labels are not explained by a single pose-side or screen-side convention.
   - Do not manually redo existing side-specific trick labels; they now count as `wearer_limb`.
   - Future side labels should still use `wearer_limb` only when the contacting limb is visually clear; use unknown/ambiguous otherwise.
   - Keep screen-position or pose-anatomical observations as audit metadata, not wearer-side training labels.
   - Build a side-specific foot identity feature only after the semantic target is explicit.
2. Rework drop/stall as a sequence problem:
   - The new stall/drop audit removes the missing-cache confound: all 67 clean rows now have OWLv2/L2 features.
   - Sequence-window/floor-context features improve the prior L2-only drop baseline from 0.609/0.667 to 0.652/0.714 precision/recall; adding available merged-touch gap context improves it again to 0.682/0.714 on 35 clean reset rows, but still fails the 0.90/0.90 gate.
   - The visual error sheet shows many reset labels are hidden/gap-style decisions, not obvious single-frame grounded-ball facts.
   - Next drop attempt should use full rally sequence state: model touch stream on every clip, long no-touch gaps, ball disappearance, and post-candidate rally-reset state rather than another point-local reset classifier.
3. Add dwell/control features for stall:
   - Ball speed/stationary duration around candidate.
   - Ball-on-foot proximity persistence across multiple frames.
   - Separate stall window positives from kick impulses.
4. Improve the label plan:
   - Inner/outer needs at least 20 examples per class, not 7 inner rows.
   - Side needs more balanced left rows and clips where left/right alternates cleanly.
   - Knee/drop/trick labels need their own minimum counts before HUD badges can be promoted.
   - Ambiguous side/surface should stay unknown; noisy labels will hurt more than missing labels.
5. HUD promotion:
   - Keep v0.1 HUD generic for touches.
   - Render side/surface/knee/stall/drop badges only behind gate-passed model outputs or explicit visual overrides.

## Current Definition Of Done For v1.0

- Contact crop embedding model exists and is evaluated clip-disjoint.
- Side accuracy >=85% on held-out clips.
- Surface accuracy >=85% on held-out clips.
- Stall/drop have separate reviewed labels and pass their gates.
- HUD badges render only for promoted signals.
- Reports show promoted and not-promoted signals separately.
- Tests pass and artifacts include visual audit sheets for remaining misses.
