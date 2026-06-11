# Release Status

Updated: 2026-05-18

## Completed In This Pass

- Added `RELEASE_GOAL.md` with the releasable-MVP stop gates.
- Added public CLI: `python3 hackytrack.py`.
- Added versioned run directories under `runs/<run-name-or-timestamp>/`.
- Added run-level exports:
  - `summary.md`
  - `run_manifest.json`
  - `events.json`
  - `events.csv`
  - `rallies.json`
- Added run-specific review-app arguments:
  - `--qa-manifest`
  - `--review-batch`
  - `--assisted-review`
  - `--reviews-dir`
- Added HUD verification through the public CLI.
- Added release evaluation:
  - deterministic video-level train/validation/test splits
  - clean current-batch labels separated from exploratory labels
  - held-out target gates for touch/drop/stall/side/type/duplicates/knee/tricks
  - saved model-artifact metadata
- Added safe behavior when no strict-complete rally exists:
  - default analytics still finish
  - `hud/hud_error.json` explains why HUD was skipped
  - `--allow-incomplete-hud` renders the best available rally
  - `--require-hud` preserves strict release-gate failure behavior
- Added release CLI tests in `tests/test_release_cli.py`.
- Added release evaluation tests in `tests/test_release_evaluation.py`.
- Added active-learning review proposals for suppressed/uncertain likely missed touches, drops, stalls, and tricks.
- Added active-learning tests in `tests/test_active_learning_review_batch.py`.
- Updated README quickstart for the public CLI and run directory layout.

## Verified Smoke Evidence

Smoke command:

```bash
python3 hackytrack.py process \
  --run-name smoke-video-482-hud \
  --overwrite \
  --review-batch-size 24 \
  --ball-audit-max-events 32 \
  --allow-incomplete-hud \
  /Users/isaacaudet/Downloads/video-482_singular_display.MOV
```

Smoke run directory:

```text
runs/smoke-video-482-hud/
```

Verified outputs:

- `runs/smoke-video-482-hud/summary.md`
- `runs/smoke-video-482-hud/events.json`
- `runs/smoke-video-482-hud/events.csv`
- `runs/smoke-video-482-hud/rallies.json`
- `runs/smoke-video-482-hud/hud/best_rally_sprite_hud_overlay.mp4`
- `runs/smoke-video-482-hud/hud/hud_verification.json`

Current public smoke command with release evaluation and path audit:

```bash
python3 hackytrack.py process \
  --run-name smoke-video-482-release-eval \
  --overwrite \
  --review-batch-size 24 \
  --ball-audit-max-events 32 \
  --allow-incomplete-hud \
  /Users/isaacaudet/Downloads/video-482_singular_display.MOV
```

Current smoke artifacts:

- `runs/smoke-video-482-release-eval/summary.md`
- `runs/smoke-video-482-release-eval/run_manifest.json`
- `runs/smoke-video-482-release-eval/portable_paths_audit.json`
- `runs/smoke-video-482-release-eval/release_evaluation/release_metrics.json`
- `runs/smoke-video-482-release-eval/hud/best_rally_sprite_hud_overlay.mp4`

Path audit:

- `portable_paths_audit.json`: passed
- `rg -n "/Users/" runs/smoke-video-482-release-eval` across text artifacts: no matches

Full 27-video public run:

```bash
python3 hackytrack.py process \
  --run-name release-27-public \
  --overwrite \
  --review-batch-size 96 \
  --ball-audit-max-events 240 \
  --reviews-dir reviews \
  --allow-incomplete-hud \
  <27 Downloads videos>
```

Full run directory:

```text
runs/release-27-public/
```

Original full run results before the side/drop strictness follow-up:

- Videos: `27`
- Touch candidates after QA filtering: `160`
- Floor resets after QA filtering: `36`
- Stall windows: `3`
- Around-the-world candidates: `1`
- Rallies: `62`
- Suppressed touch candidates: `100`
- Suppressed duplicate/noisy floor resets: `25`
- Ball audit pass rate: `99.1%`
- Review batch coverage from existing `reviews/`: `96/96`
- HUD selected strict-complete rally: `video-482_singular_display.MOV`, rally `1`, `8` touches, `1` stall, quality `84.4` (superseded below; this is no longer accepted as strict)
- HUD verification: `1096x1464`, `201` frames, `6.699s`, video and audio streams present
- Portable path audit: passed, no text `/Users/` matches
- Known video-352 drop near `22.896s`: detected as `drop_floor` with `visual_floor_between_contacts`

Full run release target gates:

- Ball event-center pass rate: `99.1%` vs target `95.0%`
- Held-out touch precision/recall: `100.0%` / `100.0%`
- Held-out drop precision/recall: `100.0%` / `100.0%`
- Held-out stall precision/recall: `100.0%` / `100.0%`
- Held-out side/contact-type accuracy: `100.0%` / `100.0%`
- Duplicate touch rate: `0.0%`
- Model artifact saved: `runs/release-27-public/models/footbag_patch_hgb.joblib`
- Remaining release gate: knee stays candidate-only because the current clean batch has no reviewed knee examples

Review-app proof:

- Review app was launched against `runs/smoke-video-482-hud`.
- One touch was approved through `/api/review/video-482_singular_display`.
- Validation changed to `1/6` decided with touch current precision `100%` on that smoke subset.
- Approved-events export wrote `runs/smoke-video-482-hud/reviews/video-482_singular_display.approved_events.json`.

Active-learning proof:

- Full 27-video active-learning batch check: `runs/release-27-public/review_batches_active_balanced_test/latest_review_batch.json`
- Balanced source counts: `72` detector candidates, `24` active-learning missing-event proposals
- Active-learning coverage: `9` likely missed touches, `7` likely missed drops, `4` likely missed stalls, `4` likely missed ATW/trick proposals
- Review app API exposed active-learning proposals as pending manual missing proposals in `runs/release-27-public/reviews_active_test`
- Marking one active-learning ATW proposal as `missing` changed validation and release-evaluation recall denominators in `runs/release-27-public/validation_active_test/validation_metrics.json` and `runs/release-27-public/release_evaluation_active_test/release_metrics.json`

Side/drop strictness follow-up:

- Regenerated QA with contact-side and strict-rally fixes at `runs/release-27-public/qa_sidefix/qa_manifest.json`.
- Reliable foot contacts now use the detected foot side relative to frame midpoint instead of a broad `center` dead zone; uncertain sides stay `unknown`.
- High-frame/background false foot contacts are downgraded away from confident `foot` labels.
- Rallies ending by a long gap without a detected floor reset are marked `ended_by_gap_without_floor_reset` and blocked from strict best-rally selection.
- `video-482_singular_display.MOV` is no longer accepted as a strict complete 8-touch best rally; it is marked as ending by a `2.46s` gap without a floor reset.
- Strict complete rally count on the regenerated manifest is `0`; the renderer correctly refuses to create a verified best-rally HUD from these labels.
- Incomplete review-candidate HUD, not release-verified: `runs/release-27-public/hud_sidefix_review_candidate/best_rally_sprite_hud_overlay.mp4`.
- Strict rally audit: `runs/release-27-public/strict_rally_audit_sidefix/strict_rally_audit.md`.
- Gap/reset review batch: `runs/release-27-public/review_batches_sidefix_gap/latest_review_batch.json`.
- The gap/reset review batch selects `9` `gap_without_floor_reset` active-learning drop proposals, including `video-482_singular_display.MOV` rally `1` and `video-230_singular_display 2.MOV` rally `1`.
- Review application bridge: `apply_reviews_to_qa.py` and `hackytrack.py apply-reviews` write reviewed QA analytics and rerun strict rally audit.
- Demo review application, without mutating real reviews: approving `video-482_singular_display.MOV` gap proposal inserted `1` reviewed `drop_floor` event at `13.62s` in `runs/release-27-public/qa_sidefix_reviewed_gap_demo/qa_manifest.json`.
- After the demo application, `video-482_singular_display.MOV` rally `1` changed from `ended_by_gap_without_floor_reset` to `explicit_drop_floor`, proving review decisions now affect rally analytics.
- Release-candidate gate report command added: `hackytrack.py report`.
- Corrected 27-video gate report: `runs/release-27-public/release_report_sidefix/release_candidate_report.md`.
- Current gate report status: `goal_complete=false`, with blockers for strict best-rally eligibility and HUD MP4 verification; knee is correctly treated as a guarded candidate-only limitation.

Side-calibrated best-rally follow-up:

- Regenerated all 27 videos with a stricter side gate at `runs/release-27-public/qa_sidefix_v2/qa_manifest.json`.
- Centerline foot contacts now stay `unknown` instead of becoming confident left/right labels from screen position alone.
- `video-506_singular_display 2.MOV` now keeps the known left-kick failure at `1.96s` as `unknown` instead of wrongly labeling it `right`.
- Review batching now prioritizes side-risk events through `side_needs_review`; the v2 review batch has `55` `side_unknown` items and `6` `side_needs_review` items.
- Review UI side corrections can be made from the keyboard: `0` unknown, `1` left, `2` right.
- Terminal floor resets are now strict-rally boundaries, not counted drops, while nonterminal drops and hidden gaps still reject a rally.
- Strict v2 audit: `runs/release-27-public/strict_rally_audit_sidefix_v2/strict_rally_audit.md`.
- v2 review batch: `runs/release-27-public/review_batches_sidefix_v2/latest_review_batch.json`.
- v2 reviewed demo: `runs/release-27-public/qa_sidefix_v2_reviewed_gap_demo/qa_manifest.json`.
- v2 release report: `runs/release-27-public/release_report_sidefix_v2/release_candidate_report.md`.
- Current v2 gate report status: `goal_complete=false`, `pass=21`, `blocked=2`, `guarded=1`.
- The release report now blocks best-rally/HUD selection because `video-230_singular_display 2.MOV` rally `1` has `11` touches and outranks the current strict 8-touch candidate, but still needs drop/reset review before it can be declared the best rally.

Side-evidence schema follow-up:

- Regenerated all 27 videos with side-evidence fields at `runs/release-27-public/qa_sidefix_v3/qa_manifest.json`.
- Touch/stall/drop events now carry `side_confidence`, `side_source`, and `side_uncertainty_reason` in JSON/CSV/review records.
- Review validation and release evaluation rows now preserve the side-evidence fields so clean side metrics can exclude uncertainty-driven `unknown` labels.
- v3 review batch: `runs/release-27-public/review_batches_sidefix_v3/latest_review_batch.json`.
- v3 review sheet: `runs/release-27-public/review_batches_sidefix_v3/qa_sidefix_v3_96_review_items/review_batch_sheet.jpg`.
- v3 strict audit: `runs/release-27-public/strict_rally_audit_sidefix_v3/strict_rally_audit.md`.
- v3 release report: `runs/release-27-public/release_report_sidefix_v3/release_candidate_report.md`.
- v3 review batch includes `55` `low_side_confidence` items, `55` `side_unknown` items, `36` `side_limb_near_frame_centerline` items, `6` `side_limb_plausible_not_confirmed` items, and `3` `side_no_reliable_limb_contact` items.
- Current v3 gate report status: `goal_complete=false`, `pass=21`, `blocked=2`, `guarded=1`.
- The remaining v3 blockers are still strict best-rally eligibility and HUD MP4 verification because `video-230_singular_display 2.MOV` rally `1` remains a higher-scoring 11-touch review-required candidate.

Reviewed blocker and HUD follow-up:

- Visual evidence for `video-230_singular_display 2.MOV` rally `1` was captured at `runs/release-27-public/blocker_review/video230/gap_8p8_16p0_sheet.jpg`.
- The active-learning blocker `miss-r001-dropgap-0010190` was accepted through the review app API and saved at `runs/release-27-public/reviews_v3_blocker/video-230_singular_display-2.review.json`.
- The accepted reset was moved to `9.30s`, where the bag is visibly on the grass after the final rally touch at `9.04s`.
- Applying that review produced `runs/release-27-public/qa_sidefix_v3_reviewed_blocker/qa_manifest.json`.
- Strict reviewed audit: `runs/release-27-public/strict_rally_audit_sidefix_v3_reviewed_blocker/strict_rally_audit.md`.
- Best strict rally is now `video-230_singular_display 2.MOV` rally `1`: `11` touches, score `90.5`, no rejection reasons.
- Reviewed best-rally HUD: `runs/release-27-public/hud_sidefix_v3_reviewed_blocker/best_rally_sprite_hud_overlay.mp4`.
- HUD verification: `runs/release-27-public/hud_sidefix_v3_reviewed_blocker/hud_verification.json` with video/audio streams and nonblank sampled frames.
- Reviewed blocker release report: `runs/release-27-public/release_report_v3_reviewed_blocker/release_candidate_report.md`.
- Current reviewed-blocker report status: `goal_complete=false`, `pass=13`, `blocked=8`, `fail=1`, `guarded=2`.
- Strict best-rally and HUD gates now pass, but held-out metric gates are not release-ready because this v3 blocker review set has only one reviewed decision and no meaningful held-out denominator; drop/floor recall is `0.0%` on that narrow split.

Review-evidence follow-up:

- Added `prepare_review_evidence.py` and `hackytrack.py review-evidence` to generate grouped visual sheets from a run review batch.
- Fixed assisted review scoring so active-learning gap proposals with nonnumeric event IDs like `gap-rally-3` no longer crash suggestion generation.
- Current v3 evidence pack: `runs/release-27-public/review_evidence_sidefix_v3_blocker/review_evidence_report.md`.
- Evidence sheets written for `gap_floor_reset`, `ball_accuracy`, `drop_review`, `side_contact`, `stall_trick`, `missed_touch`, and `all_priority`.
- Current evidence inventory: `96` review items, `88` unreviewed, `7` pending, `1` missing; bucket counts include `9` gap floor resets, `58` ball-accuracy risks, `62` side/contact risks, `21` drop-review items, `10` stall/trick items, and `9` likely missed touches.
- Public CLI smoke output: `runs/release-27-public/review_evidence_cli_smoke/review_evidence_report.md`.
- Side classifier follow-up: `qa_rally_enrichment.py` now uses tighter foot/ball geometry for obvious left/right contacts and keeps exact centerline contacts review-needed; review batch records now preserve `foot_confidence` for side audits.

v4 side-fix artifact refresh:

- Regenerated the full 27-video QA pass at `runs/release-27-public/qa_sidefix_v4/qa_manifest.json`.
- Side unknowns improved on the same 174 touch candidates: v3 `unknown=88`, `right=58`, `left=28`; v4 `unknown=35`, `right=94`, `left=45`.
- Low-side-confidence touch count dropped from `88` to `35`.
- v4 strict audit before review: `runs/release-27-public/strict_rally_audit_sidefix_v4/strict_rally_audit.md`.
- v4 review batch: `runs/release-27-public/review_batches_sidefix_v4/latest_review_batch.json`.
- v4 reviewed QA after applying the existing video-230 blocker decision: `runs/release-27-public/qa_sidefix_v4_reviewed_blocker/qa_manifest.json`.
- v4 reviewed strict audit: `runs/release-27-public/strict_rally_audit_sidefix_v4_reviewed_blocker/strict_rally_audit.md`; top strict rally remains `video-230_singular_display 2.MOV` rally `1`, `11` touches, score `90.5`.
- v4 reviewed HUD: `runs/release-27-public/hud_sidefix_v4_reviewed_blocker/best_rally_sprite_hud_overlay.mp4`; verification copied to `runs/release-27-public/hud_sidefix_v4_reviewed_blocker/hud_verification.json`.
- v4 evidence pack: `runs/release-27-public/review_evidence_sidefix_v4_blocker/review_evidence_report.md`.
- v4 release report: `runs/release-27-public/release_report_v4_reviewed_blocker/release_candidate_report.md`; current status is still `goal_complete=false`, `pass=13`, `blocked=8`, `fail=1`, `guarded=2`, because held-out reviewed labels are still insufficient.
- Fixed `release_candidate_report.py` so strict best-rally and HUD gates use the reviewed strict audit when one is supplied, instead of blocking on stale unreviewed audit state.

Run-scoped review decision follow-up:

- Added `hackytrack.py seed-reviews` and run-scoped `seed_review_batch.py` options for `--qa-manifest`, `--reviews-dir`, and `--assisted-review`.
- Added `hackytrack.py apply-decisions` and run-scoped `apply_review_decisions.py --reviews-dir` so review decisions can be applied to a run-local review folder.
- Seeded v4 batch reviews at `runs/release-27-public/reviews_v4_seeded`; batch coverage is now `96/96` present in review files.
- Added auditable visual decisions at `runs/release-27-public/reviews_v4_seeded/codex_visual_review_decisions.json`.
- Applied `18` visual decisions: `8` approved `drop_floor`, `7` missing `drop_floor`, `2` approved `touch`, and `1` rejected `stall`.
- Regenerated reviewed QA: `runs/release-27-public/qa_sidefix_v4_reviewed_seeded/qa_manifest.json` with `approved_candidates=10`, `rejected_candidates=1`, `manual_missing_inserted=8`, `pending_candidates=62`.
- Regenerated strict audit: `runs/release-27-public/strict_rally_audit_sidefix_v4_reviewed_seeded/strict_rally_audit.md`; strict-complete rallies increased to `3`.
- Regenerated validation: `runs/release-27-public/validation_v4_seeded_reviewed/validation_report.md`; current batch decision coverage is `16.67%`.
- Regenerated release evaluation: `runs/release-27-public/release_evaluation_v4_seeded_reviewed/release_evaluation_report.md`.
- Regenerated release report: `runs/release-27-public/release_report_v4_seeded_reviewed/release_candidate_report.md`; status improved to `goal_complete=false`, `pass=19`, `fail=2`, `blocked=1`, `guarded=2`.
- Held-out gates now passing after review decisions: ball event-center pass rate, touch precision, touch recall, duplicate touch rate, drop/floor precision, side accuracy, contact-type accuracy, and model artifact saved.
- Remaining metric blockers are now concrete rather than missing-denominator: drop/floor recall is `50.0%` vs `90.0%` target, stall precision is `0.0%` vs `85.0%` target, and stall recall still has no held-out reviewed denominator.

Detector replacement follow-up:

- Confirmed the remaining obvious frame failures are heuristic ball-center failures, not acceptable release behavior.
- Added a long-term detector-replacement target to `RELEASE_GOAL.md`: a custom reviewed `footbag` detector/segmenter should become primary, with HSV/red-blob snapping retained only as fallback/audit evidence.
- Added `export_detector_dataset.py` and `hackytrack.py export-detector-dataset`.
- Added `build_detector_label_review_batch.py` and `hackytrack.py detector-label-review` to mine QA ball centers into detector-specific object-label review sheets and decision templates; `export_detector_dataset.py` can now ingest completed detector-label decisions.
- Added `requirements-detector.txt`, `train_footbag_detector.py`, and `hackytrack.py train-detector` for the custom detector training path.
- Added `footbag_detector_inference.py` and `hackytrack.py detect-footbag` for model-backed ball tracking with tracker smoothing, confidence, and uncertainty reasons.
- Added `hackytrack.py detect-footbag-batch` for run-level detector inference across every video in a run or QA manifest, with an auditable batch manifest.
- Added `apply_detector_track_to_qa.py` and `hackytrack.py apply-detector-track` so model ball-track evidence can be attached to every QA event while preserving the old heuristic evidence for audit.
- Added `evaluate_detector_tracks.py` and `hackytrack.py evaluate-detector` for detector center-pass metrics against reviewed positives and false-positive checks against hard-negative points.
- The exporter separates object labels from event labels:
  - a hand-held bag can still become a positive `footbag` detector label
  - a marker on skin, grass, or another non-bag patch becomes a hard-negative center crop
  - rejected touches remain invalid events without corrupting the detector labels
- Added unit coverage in `tests/test_export_detector_dataset.py`.
- Added unit coverage in `tests/test_train_footbag_detector.py`.
- Hard negatives now also write empty-label YOLO images, so false ball-center crops can directly train the detector not to fire on those patches.
- Regenerated QA with lower-limb side anchoring and stricter false-touch suppression at `runs/release-27-public/qa_sidefix_v7/qa_manifest.json`.
- Known reviewed false positives are now suppressed in v7:
  - `video-431_singular_display.MOV` touch candidate at `2.10s`
  - `video-474_singular_display.MOV` touch candidate at `4.72s`
- The previously suppressed real touch at `video-474_singular_display.MOV` `4.41s` is now retained as a touch candidate.
- Built a first detector dataset at `runs/release-27-public/detector_dataset_v7_reviewed/`:
  - `data.yaml`
  - `reviewed_detector_labels.jsonl`
  - split `images/` and `labels/`
  - `hard_negatives/points.jsonl`
  - `hard_negatives/crops/`
- Current detector export counts: `24` positive `footbag` labels and `3` hard-negative center crops from `25` review files.
- Current detector training status: real Ultralytics training now runs and saves `runs/release-27-public/detector_models_v7_reviewed/footbag_detector_best.pt`.
- Current detector validation metrics from `runs/release-27-public/detector_models_v7_reviewed/training_manifest.json` are still `0.0` precision, `0.0` recall, `0.0` mAP50, and `0.0` mAP50-95, so the trained artifact is not releasable yet.
- Current detector inference status: public command, batch command, smoothing/export schema, QA-event adapter, and detector-specific evaluation are implemented and unit-tested. A real one-video smoke inference with the trained artifact completed at `runs/release-27-public/detector_inference_v7_smoke/`, but produced `0` raw detections/track points, confirming more reviewed labels/training work is required.
- Detector-label review expansion batch built at `runs/release-27-public/detector_label_review_v1/`:
  - `191` candidate object-label reviews mined from QA events after excluding already reviewed event items
  - `105` selected for review across `24` videos with per-video cap
  - `85` suggested as likely `footbag`
  - `20` marked `verify_or_correct`
  - split coverage: `71` train, `18` validation, `16` test
  - all `105` preview tiles rendered; no `/Users/...` paths in the manifest
- Pending detector-label decisions remain separate from training. A dry-run export with the generated decision template saw `105` pending decisions and used `0`, preserving clean-label discipline until review is complete.
- Added conservative CV-assisted detector-label decisions at `runs/release-27-public/detector_label_review_v1/detector_label_assisted_decisions.json`: `93` obvious `footbag`, `12` pending, `0` automatic hard negatives. These are experimental assisted labels, not final human-reviewed release labels.
- Exported assisted detector dataset `runs/release-27-public/detector_dataset_v8_assisted/`: `117` positive labels, `3` hard-negative points, `120` YOLO images/labels, and `93` detector-label decisions used.
- Trained `runs/release-27-public/detector_models_v8_assisted/footbag_detector_best.pt` for `25` epochs. Final validation metrics improved from zero to precision `0.72207`, recall `0.23717`, mAP50 `0.28271`, and mAP50-95 `0.22295`.
- Added bbox sanity filtering before tracker smoothing. Low-confidence smoke filtering removed `37,864` too-large boxes, `1,839` aspect-ratio failures, and `51` too-small boxes.
- Current video-inference blocker: at normal confidence `0.05`, the v8 assisted model still produced `0` detections on the first 180 frames of `video-352`; at `0.001`, it produces many low-confidence false positives. More hard negatives and video-inference mining are required before promotion.
- Added assisted detector-label decisions at `runs/release-27-public/detector_label_review_v1/detector_label_assisted_decisions.json` as an experimental bootstrap only:
  - `93` likely `footbag` labels accepted by conservative CV assistance
  - `12` left `pending`
  - this file is marked assisted, not final human-reviewed release truth
- Built assisted detector dataset `runs/release-27-public/detector_dataset_v8_assisted/`:
  - `117` positive labels
  - `3` hard-negative empty-label YOLO images
  - `120` written images/label files
- Recovered interrupted v8 Ultralytics training through `hackytrack.py train-detector --recover-existing`:
  - model artifact: `runs/release-27-public/detector_models_v8_assisted/footbag_detector_best.pt`
  - manifest: `runs/release-27-public/detector_models_v8_assisted/training_manifest.json`
  - final recovered epoch: `14`
  - validation precision: `0.00197`
  - validation recall: `0.59091`
  - validation mAP50: `0.00141`
  - validation mAP50-95: `0.00046`
- Added detector model sanity gate `hackytrack.py evaluate-detector-model` and ran it at `runs/release-27-public/detector_model_evaluation_v8_assisted/`:
  - release pass rate: `0.0`
  - candidate center pass rate at low confidence: `1.0`
  - low-confidence near-label failures: `117 / 117`
  - mean center error: `10.751px`
  - p95 center error: `32.293px`
  - mean top confidence: `0.012926`
  - hard-negative false-positive rate at release confidence: `0.0`
- v8 assisted smoke video inference at `runs/release-27-public/detector_inference_v8_assisted_smoke/` completed on three videos but produced `0` raw detections and `0` track points at `0.05` confidence. This confirms the current model can localize reviewed still-frame candidates only at unusably low confidence and must not replace heuristic ball centers.
- v8 assisted smoke track evaluation at `runs/release-27-public/detector_track_evaluation_v8_assisted_smoke/` has `0.0` pass rate: `109` labels have missing track files because only three smoke videos were run, and `8` labels on the smoke videos have missing track points.
- Added trained patch/objectness detector path:
  - script: `patch_footbag_detector.py`
  - public commands: `hackytrack.py train-patch-detector` and `hackytrack.py evaluate-patch-detector`
  - artifact: `runs/release-27-public/patch_detector_v2_assisted/patch_footbag_detector.joblib`
  - training manifest: `runs/release-27-public/patch_detector_v2_assisted/patch_training_manifest.json`
  - evaluator: `runs/release-27-public/patch_detector_eval_v2_assisted_cli/patch_detector_metrics.json`
- Patch detector v2 result with expanded red/dark/high-contrast proposals and top-100 diagnostic matching:
  - overall pass rate: `0.726496`
  - passed labels: `85 / 117`
  - missing predictions: `0`
  - failed centers: `32`
  - mean center error: `37.378px`
  - hard-negative false-positive rate: `0.666667`
  - split pass rates: test `0.789474`, train `0.736842`, validation `0.636364`
- Patch detector conclusion: this is a real improvement over YOLO's `0.0` release pass rate, but it is not promotable yet. The proposal stage now finds candidates for all reviewed images, while ranking/calibration and hard-negative rejection remain blockers.
- Fixed detector inference coordinate-space plumbing:
  - `footbag_detector_inference.py` now supports `--process-width 688 --process-height 912`
  - manifests record original video size and processed-frame size
  - `hackytrack.py detect-footbag` and `detect-footbag-batch` can run `--patch-model`
  - this prevents trained detector tracks from being compared against QA labels in mismatched original-video coordinates
- Patch-model video smoke at `runs/release-27-public/patch_detector_inference_v2_smoke/video-352_singular_display 2/`:
  - source: `patch_model`
  - coordinate space: `processed_frame`
  - scanned frames: `180`
  - raw detections: `1326`
  - track points: `180`
  - track evaluation at `runs/release-27-public/patch_detector_track_eval_v2_smoke_t050/` still has `0.0` pass rate for the available reviewed labels, because high-confidence false proposals dominate the tracker
- Mined patch-model false positives into a hard-negative review batch at `runs/release-27-public/patch_detector_false_positive_review_v2_smoke_highconf/`:
  - total candidates: `1455`
  - selected items: `40`
  - contact sheet: `runs/release-27-public/patch_detector_false_positive_review_v2_smoke_highconf/detector_label_review_sheet.jpg`
  - the sheet shows many high-confidence mistakes on hands, shoes, shadows, statues, and grass; some crops also contain the real bag but with the model center wrong, so these must be reviewed as `not_footbag`, `footbag`, or `corrected` rather than blindly used as negatives
- Ran conservative assisted decisions for that high-confidence false-positive sheet at `runs/release-27-public/patch_detector_false_positive_review_v2_smoke_highconf/detector_false_positive_assisted_decisions.json`: `5` possible `footbag` rows, `35` pending rows, `0` automatic hard negatives. This confirms hard negatives require explicit review rather than automatic rejection.
- Updated `patch_footbag_detector.py` so `hard_negatives/points.jsonl` rows are loaded as centered `hard_negative_point` samples for train/validation/test sample metrics. This closes the feedback-loop plumbing: future reviewed `not_footbag` detector-label decisions will affect patch training directly, not just create empty-label images.
- Trained patch detector v3 with hard-negative point ingestion:
  - artifact: `runs/release-27-public/patch_detector_v3_assisted/patch_footbag_detector.joblib`
  - training manifest: `runs/release-27-public/patch_detector_v3_assisted/patch_training_manifest.json`
  - hard-negative point splits: total `3`, train `0`, validation `0`, test `3`
  - validation sample metrics: precision `0.548387`, recall `0.772727`, f1 `0.641509`
  - test sample metrics: precision `0.197183`, recall `0.736842`, f1 `0.311111`
- Evaluated patch detector v3 at `runs/release-27-public/patch_detector_eval_v3_assisted_cli/patch_detector_metrics.json`:
  - overall pass rate: `0.726496`
  - passed labels: `85 / 117`
  - missing predictions: `0`
  - failed centers: `32`
  - mean center error: `37.378px`
  - hard-negative false-positive rate: `0.666667`
- Patch detector v3 conclusion: this is an infrastructure fix, not an accuracy win yet. The current reviewed hard-negative points are all in the held-out test split, so training behavior is unchanged; the next required data step is to review obvious high-confidence false positives as `not_footbag` in train/validation splits and retrain.
- Ran patch detector v3 across the 27-video QA manifest at `runs/release-27-public/patch_detector_inference_v3_27_smoke/`:
  - completed videos: `27 / 27`
  - failures: `0`
  - frame cap: `180` per video
  - coordinate space: processed `688x912`
- Built a 27-video high-confidence detector-label review sheet at `runs/release-27-public/patch_detector_false_positive_review_v3_27_highconf/`:
  - total candidates: `16257`
  - selected items: `120`
  - split coverage: train `80`, validation `16`, test `12`, unknown `12`
  - contact sheet: `runs/release-27-public/patch_detector_false_positive_review_v3_27_highconf/detector_label_review_sheet.jpg`
- Fixed the false-positive mining language: detections far from QA ball centers now default to `verify_or_correct`, because the sheet shows many real footbags held in a hand or outside the event timeline. Only explicit review decisions should become `not_footbag` hard negatives.
- Assisted triage after that fix marked `15` likely `footbag`, `0` automatic hard negatives, and `105` pending. This is the correct conservative behavior for mixed object-label review sheets.
- Added a conservative Codex visual decision file with `38` obvious `not_footbag` labels at `runs/release-27-public/patch_detector_false_positive_review_v3_27_highconf/detector_false_positive_codex_visual_decisions.json`. Ambiguous tiles were left out.
- Exported v9 detector dataset with those hard negatives at `runs/release-27-public/detector_dataset_v9_codex_hardneg/`:
  - positive labels: `117`
  - hard-negative points: `41`
  - written images/labels: `158`
  - detector-label decisions used: `131 / 143`
  - hard-negative point splits available to patch training: train `22`, validation `12`, test `7`
- Added `--model-kind extra-trees` to the patch detector trainer/public CLI and trained `runs/release-27-public/patch_detector_v5_extra_trees_hardneg/patch_footbag_detector.joblib`.
- Patch detector v4 logistic with v9 hard negatives is not releasable:
  - low-threshold pass rate: `0.700855`
  - low-threshold hard-negative false-positive rate: `0.853659`
  - artifact-threshold pass rate: `0.384615`
  - artifact-threshold hard-negative false-positive rate: `0.268293`
- Patch detector v5 extra-trees improves candidate center pass rate but still has no release-grade threshold:
  - threshold `0.05`: pass rate `0.760684`, hard-negative false-positive rate `0.97561`
  - threshold `0.20`: pass rate `0.709402`, hard-negative false-positive rate `0.634146`
  - threshold `0.35`: pass rate `0.564103`, hard-negative false-positive rate `0.219512`
  - threshold `0.50`: pass rate `0.461538`, hard-negative false-positive rate `0.02439`
  - artifact threshold: pass rate `0.478632`, hard-negative false-positive rate `0.073171`
- Detector conclusion after v9/v5: the weak CV proposal plus patch classifier stack is useful for active learning, but it cannot be the release detector. The next architecture target is a proper reviewed-label detector/segmenter or tracker that sees temporal context and full-frame object geometry, with the patch model retained only as an audit/mining baseline.

Detector false-positive mining update:

- Added `build_detector_false_positive_review_batch.py` and public `hackytrack.py detector-false-positive-review` to mine trained-model detections that are far from reviewed QA ball centers into object-label review sheets. These are review candidates, not automatic hard negatives, because sparse touch labels do not describe the bag location on every frame.
- Built `runs/release-27-public/detector_false_positive_review_v1/` from low-confidence v8 inference on the first 180 frames of `video-352`: `3042` candidate detections, `80` selected review items, `0` hardcoded `/Users/...` paths, and `80/80` rendered crops.
- Tightened `assist_detector_label_decisions.py` so skin/hand blobs are no longer accepted as footbags, blue/yellow panel balls can be accepted, and false-positive candidates only flip back to `footbag` when compact bag-colored evidence is close to the marked center. Assisted output for that batch: `5` footbag, `4` not_footbag, `71` pending.
- Added repeatable `--detector-label-review-pair` ingestion to `export-detector-dataset`, allowing v8 assisted labels plus false-positive review decisions to export together.
- Exported `runs/release-27-public/detector_dataset_v9_false_positive/`: `122` positive labels, `7` hard-negative points, `129` YOLO images/labels, `102` used detector-label decisions, `83` skipped/pending decisions.
- Trained `runs/release-27-public/detector_models_v9_false_positive/footbag_detector_best.pt` for `25` epochs. Best-epoch validation metrics by mAP50: precision `0.80283`, recall `0.14815`, mAP50 `0.29620`, mAP50-95 `0.19880`. Compared to v8, precision and mAP50 improved, but recall regressed and remains far below release targets.
- Ran detector-model sanity gate at `runs/release-27-public/detector_model_evaluation_v9_false_positive_calibrated/`: release-confidence pass rate `0.0`, low-confidence candidate-center pass rate `0.991803`, mean center error `7.661px`, and hard-negative false-positive rate `0.0`. This confirms localization is good when the model fires near labels, but confidence calibration/recall are not release-ready at the default threshold.
- Added validation-split threshold calibration to the detector-model evaluation. For v9, the recommended threshold is `0.011504`, with validation precision `1.0`, recall `0.296296`, F1 `0.457143`, and hard-negative false-positive rate `0.0`.
- Added `--calibration-metrics` to `detect-footbag` / `detect-footbag-batch`, so inference can consume the recommended threshold artifact directly. Artifact-driven v9 smoke inference on first 180 frames of `video-352` wrote `runs/release-27-public/detector_inference_v9_false_positive_smoke_calibrated_artifact/` with threshold source `calibration_metrics`, `22` raw detections, `20` model-detection track points, and `58` interpolated track points.
- v9 smoke inference comparison on first 180 frames of `video-352`: at confidence `0.05`, `0` raw detections and `0` track points; at calibrated confidence `0.011504`, `22` raw detections and `78` track points; at confidence `0.001`, `12087` raw detections and `180` track points, down from v8's `14246` raw detections but still too noisy without calibration/review.
- Ran calibrated v9 detector batch across all `27` videos for the first `300` frames each at `runs/release-27-public/detector_inference_v9_calibrated_batch_300/`. All `27/27` completed with no failures, no zero-raw-detection videos, and no zero-track videos.
- Added `summarize_detector_batch.py` and public `hackytrack.py summarize-detector-batch`. The saved summary at `runs/release-27-public/detector_batch_summary_v9_calibrated_300/` reports `7469` scanned frames, `5397` raw detections, `6168` track points, model coverage `0.569286`, interpolation share `0.310636`, and `10` flagged videos. Worst flagged clip remains `video-352_singular_display 2.MOV` with model coverage `0.1` and interpolation share `0.714286`.
- Mined broad review candidates from the calibrated 27-video batch into `runs/release-27-public/detector_false_positive_review_v9_calibrated_batch_300/`: `2695` candidates, `160` selected, `160/160` rendered crops, `26` source videos, and no hardcoded `/Users/...` paths. CV assist marked `30` as clear footbag and left `130` pending. Visual inspection shows many candidates are real bag-on-shoe positives outside sparse QA touch labels, so this batch should primarily be reviewed as recall-recovery/correction data, not blindly as hard negatives.
- Detector conclusion after v9: the active-learning hard-negative/recall-recovery loop works and improves selectivity, but it is not releasable yet. The next required step is review of `detector_false_positive_review_v9_calibrated_batch_300`, export of those decisions into the detector dataset, and retraining so recall can recover without reintroducing non-ball detections.

Detector v10 recall-recovery follow-up:

- Added visual review decisions for `runs/release-27-public/detector_false_positive_review_v9_calibrated_batch_300/` at `detector_false_positive_codex_visual_decisions.json`: `159` rows marked `footbag`, `1` row marked `not_footbag`. Visual inspection showed the batch was mostly real bag-on-shoe positives outside sparse QA event labels, so it became recall-recovery data rather than a hard-negative-only batch.
- Exported `runs/release-27-public/detector_dataset_v10_codex_recall/`: `281` positive labels, `8` hard-negative points, `289` YOLO images/labels, `8` hard-negative YOLO images, `345` detector decisions seen, `262` used, and `83` skipped/pending.
- Trained `runs/release-27-public/detector_models_v10_codex_recall/footbag_detector_best.pt` for `25` epochs. Best-epoch validation metrics by mAP50 improved to precision `0.80353`, recall `0.60000`, mAP50 `0.62331`, and mAP50-95 `0.41208`; final-epoch metrics were precision `0.81476`, recall `0.61586`, mAP50 `0.56532`, and mAP50-95 `0.38782`.
- Ran detector-model sanity gate at `runs/release-27-public/detector_model_evaluation_v10_codex_recall_calibrated/`: release-confidence pass rate `0.604982`, low-confidence candidate-center pass rate `0.985765`, mean center error `4.484px`, p95 center error `21.195px`, hard-negative false-positive rate `0.0`, and recommended validation threshold `0.029018` with precision `0.942857`, recall `0.66`, and F1 `0.776471`.
- Ran calibrated v10 detector batch across all `27` videos for the first `300` frames in processed QA coordinates at `runs/release-27-public/detector_inference_v10_calibrated_batch_300_processed/`. All `27/27` completed with no failures, no zero-raw-detection videos, and no zero-track videos.
- Saved processed-coordinate batch summary at `runs/release-27-public/detector_batch_summary_v10_calibrated_300_processed/`: `7469` scanned frames, `13432` raw detections, `6924` track points, model coverage `0.845629`, interpolation share `0.087811`, and `1` flagged video. The remaining flagged clip is `video-352_singular_display 2.MOV` with model coverage `0.25` and interpolation share `0.456522`.
- The v10 bounded batch materially improves over v9: model coverage increased from `0.569286` to `0.845629`, interpolation share dropped from `0.310636` to `0.087811`, and flagged videos dropped from `10` to `1`.
- Processed-coordinate track evaluation at `runs/release-27-public/detector_track_evaluation_v10_calibrated_batch_300_processed/` is still not promotable: overall pass rate `0.729537`, mean center error `35.548px`, p95 center error `226.593px`, `38` missing track points, and hard-negative false-positive rate `0.125`. Held-out test split pass rate is `0.666667` with hard-negative false-positive rate `0.25`.
- Coordinate-space note: `runs/release-27-public/detector_track_evaluation_v10_calibrated_batch_300/` is intentionally not used as evidence because that first evaluation compared original-video-coordinate tracks against processed `688x912` QA labels.
- Current detector conclusion after v10: the full-frame YOLO path has moved from infrastructure/prototype to a useful trained detector, but it still fails release promotion gates. The next required work is targeted review/retraining around `video-352`, the hard-negative miss in the held-out split, and the high center-error tail before detector-backed centers can replace heuristic QA centers or drive HUD correctness claims.

Detector v11 error-recovery follow-up:

- Added `build_detector_error_review_batch.py` and public `hackytrack.py detector-error-review` to mine failed model/track evaluations into detector-label review sheets. The builder overlays expected reviewed-label centers against model/track centers and marks held-out test split rows as `audit_only` so they diagnose failures without leaking into training.
- Built `runs/release-27-public/detector_error_review_v11/` from v10 model/track metrics: `105` candidates, `98` selected review items, `98/98` rendered crops, and `5` paged contact sheets. Selected reasons were `38` `track_center_fail`, `4` `model_center_fail`, `35` `missing_track_point`, `1` `hard_negative_false_positive`, and `20` `low_confidence_near_label`. Split/use counts were `53` train, `26` validation, `19` test; `79` train-or-calibration rows and `19` audit-only rows.
- Added `detector_error_codex_decisions.json` for that batch using source-label reuse: `79` train/validation rows marked `footbag`, `19` test rows skipped. No new hard-negative training row was added because the only hard-negative false positive in this batch was held out in the test split.
- Exported `runs/release-27-public/detector_dataset_v11_error_recovery/`: `360` positive labels, `8` hard-negative points, `368` YOLO images/labels, `8` hard-negative YOLO images, `443` detector decisions seen, `341` used, and `102` skipped/pending.
- Trained `runs/release-27-public/detector_models_v11_error_recovery/footbag_detector_best.pt` for `40` epochs. Best-epoch validation metrics by mAP50 were precision `0.74848`, recall `0.46992`, mAP50 `0.47379`, and mAP50-95 `0.29606`; final-epoch metrics were precision `0.69742`, recall `0.44737`, mAP50 `0.45102`, and mAP50-95 `0.28129`. This is weaker than v10's best mAP50 `0.62331`.
- Ran detector-model sanity gate at `runs/release-27-public/detector_model_evaluation_v11_error_recovery_calibrated/`: release-confidence pass rate `0.588889`, low-confidence candidate-center pass rate `0.952778`, mean center error `7.506px`, p95 center error `37.593px`, and hard-negative false-positive rate `0.0`. Recommended validation threshold is `0.052778` with precision `0.9375`, recall `0.394737`, and F1 `0.555556`.
- Ran calibrated v11 detector batch across all `27` videos for the first `300` frames in processed QA coordinates at `runs/release-27-public/detector_inference_v11_calibrated_batch_300_processed/`. All `27/27` completed with no failures, no zero-raw-detection videos, and no zero-track videos.
- Saved processed-coordinate batch summary at `runs/release-27-public/detector_batch_summary_v11_calibrated_300_processed/`: `7469` scanned frames, `5815` raw detections, `6113` track points, model coverage `0.643594`, interpolation share `0.213643`, and `3` flagged videos. Flagged clips were `video-352_singular_display 2.MOV` with model coverage `0.09` and interpolation share `0.674699`, plus `video-230_singular_display 2.MOV` and `video-234_singular_display 2.MOV` for high interpolation share.
- Processed-coordinate track evaluation at `runs/release-27-public/detector_track_evaluation_v11_calibrated_batch_300_processed/` regressed from v10 and is not promotable: overall pass rate `0.597222`, mean center error `57.876px`, p95 center error `329.289px`, `73` missing track points, and hard-negative false-positive rate `0.125`. Validation split pass rate is `0.355263` with mean center error `113.383px`; held-out test split pass rate remains `0.666667` with hard-negative false-positive rate `0.25`.
- Current detector conclusion after v11: v11 is evidence-only and must not replace heuristic QA centers. The error-recovery loop exported more labels, but the retrained detector reduced model coverage, increased interpolation, worsened center-error tails, and failed the same held-out hard-negative gate. Do not run detector-backed release/HUD promotion until a future detector beats v10 on processed-coordinate track accuracy and hard-negative safety.

Dense temporal-tracking tranche:

- Added `DETECTOR_REPLACEMENT_WINNING_PLAN.md`. The plan freezes v10 as the current strongest full-frame YOLO baseline and changes the next architecture target to dense short-clip trajectory labels, a temporal heatmap model, and offline single-object path decoding with an explicit no-object/occluded state.
- Ran v10 calibrated processed-coordinate inference with the existing `--tracker-mode temporal` at `runs/release-27-public/detector_inference_v10_calibrated_batch_300_processed_temporal/`. The batch completed `27/27` videos with `0` failures.
- Saved the v10 temporal batch summary at `runs/release-27-public/detector_batch_summary_v10_calibrated_300_processed_temporal/`: `7469` scanned frames, `13432` raw detections, `6421` track points, model coverage `0.845629`, interpolation share `0.016353`, and `0` flagged videos. Compared with v10 greedy, interpolation share improved from `0.087811`, but track coverage dropped from `0.927032` to `0.859687`.
- Saved the v10 temporal track evaluation at `runs/release-27-public/detector_track_evaluation_v10_calibrated_batch_300_processed_temporal/`: overall pass rate `0.72242`, mean center error `34.397px`, p95 center error `208.082px`, `39` missing track points, and hard-negative false-positive rate `0.125`. Temporal tracking improves the p95 error tail from v10 greedy's `226.593px`, but does not beat v10 greedy's pass rate `0.729537` or hard-negative safety. It is a baseline only, not a promotion path.
- Added `build_dense_trajectory_review_batch.py` and public `hackytrack.py dense-trajectory-review` to export frame-level dense review clips from detector failure evidence. Outputs include a dense trajectory schema, per-clip frame folders, contact sheets with optional model-track overlays, per-clip templates, and a combined `dense_trajectory_labels_template.jsonl`. Split discipline is encoded through `training_use`; held-out test rows are `audit_only`.
- Added `evaluate_dense_trajectory.py` and public `hackytrack.py evaluate-dense-trajectory` to score reviewed dense labels against prediction JSONL files or detector-track roots. The evaluator reports visible-frame pass/missing/error metrics and no-target false-positive rates by split.
- Built the first dense review batch at `runs/release-27-public/dense_trajectory_review_v1/` from v10/v11 failure evidence: `27` candidates, `5` selected clips, `313` exported frame rows, `5` train-or-calibration clips, and `0` audit-only clips. Clip coverage is:
  - `video-352_singular_display 2.MOV`: validation clips centered at `2.18s`, `4.999154s`, and `8.52s` for low model coverage, high interpolation, track-center failure, and missing-track windows.
  - `video-230_singular_display 2.MOV`: train clip centered at `4.999485s` for high interpolation.
  - `video-234_singular_display 2.MOV`: train clip centered at `4.999311s` for high interpolation and track-center failure.
- Dense labels are still pending human review. The generated template evaluation at `runs/release-27-public/dense_trajectory_review_v1/template_eval_v10_temporal/` correctly reports `0` reviewed visible frames because the batch is an unlabeled template. The next blocker is to fill reviewed per-frame centers/visibility states before temporal heatmap training can begin.
- Current detector conclusion after this tranche: v10 temporal tracking is not enough. The next real improvement must come from reviewed dense trajectory labels and a temporal heatmap/path-decoding model, not another sparse-label YOLO retrain.

Tests run:

```bash
python3 -m unittest tests.test_train_footbag_detector tests.test_export_detector_dataset tests.test_assist_detector_label_decisions tests.test_build_detector_false_positive_review_batch tests.test_release_cli
python3 -m py_compile train_footbag_detector.py export_detector_dataset.py assist_detector_label_decisions.py build_detector_false_positive_review_batch.py hackytrack.py
python3 hackytrack.py detector-false-positive-review --qa-manifest runs/release-27-public/qa_sidefix_v7/qa_manifest.json --inference-root runs/release-27-public/detector_inference_v8_assisted_smoke_lowconf_filtered --dataset-manifest runs/release-27-public/detector_dataset_v8_assisted/manifest.json --out-dir runs/release-27-public/detector_false_positive_review_v1 --max-items 80 --per-video 80 --crop-size 192 --cols 5
python3 hackytrack.py assist-detector-labels --review-manifest runs/release-27-public/detector_false_positive_review_v1/detector_false_positive_review_manifest.json --out runs/release-27-public/detector_false_positive_review_v1/detector_false_positive_assisted_decisions.json --min-confidence 0.70
python3 hackytrack.py export-detector-dataset --qa-manifest runs/release-27-public/qa_sidefix_v7/qa_manifest.json --reviews-dir runs/release-27-public/reviews_v6_seeded --out-dir runs/release-27-public/detector_dataset_v9_false_positive --detector-label-review-manifest runs/release-27-public/detector_label_review_v1/detector_label_review_manifest.json --detector-label-decisions runs/release-27-public/detector_label_review_v1/detector_label_assisted_decisions.json --detector-label-review-pair runs/release-27-public/detector_false_positive_review_v1/detector_false_positive_review_manifest.json:runs/release-27-public/detector_false_positive_review_v1/detector_false_positive_assisted_decisions.json
python3 hackytrack.py train-detector --dataset runs/release-27-public/detector_dataset_v9_false_positive --out-dir runs/release-27-public/detector_models_v9_false_positive --base-model yolo11n.pt --epochs 25 --imgsz 640 --batch -1 --run-name footbag-detector-v9-false-positive
python3 hackytrack.py evaluate-detector-model --dataset runs/release-27-public/detector_dataset_v9_false_positive --model runs/release-27-public/detector_models_v9_false_positive/footbag_detector_best.pt --out-dir runs/release-27-public/detector_model_evaluation_v9_false_positive_calibrated --confidence-threshold 0.001 --release-confidence-threshold 0.25 --calibration-split validation --max-hard-negative-false-positive-rate 0 --tolerance-px 24 --box-tolerance-multiplier 0.75 --imgsz 640 --max-det 300
python3 hackytrack.py detect-footbag --video '/Users/isaacaudet/Downloads/video-352_singular_display 2.MOV' --model runs/release-27-public/detector_models_v9_false_positive/footbag_detector_best.pt --out-dir runs/release-27-public/detector_inference_v9_false_positive_smoke --confidence-threshold 0.05 --max-frames 180 --imgsz 640
python3 hackytrack.py detect-footbag --video '/Users/isaacaudet/Downloads/video-352_singular_display 2.MOV' --model runs/release-27-public/detector_models_v9_false_positive/footbag_detector_best.pt --calibration-metrics runs/release-27-public/detector_model_evaluation_v9_false_positive_calibrated/detector_model_metrics.json --out-dir runs/release-27-public/detector_inference_v9_false_positive_smoke_calibrated_artifact --max-frames 180 --imgsz 640
python3 hackytrack.py detect-footbag --video '/Users/isaacaudet/Downloads/video-352_singular_display 2.MOV' --model runs/release-27-public/detector_models_v9_false_positive/footbag_detector_best.pt --out-dir runs/release-27-public/detector_inference_v9_false_positive_smoke_lowconf_filtered --confidence-threshold 0.001 --max-frames 180 --imgsz 640
python3 hackytrack.py detect-footbag-batch --qa-manifest runs/release-27-public/qa_sidefix_v7/qa_manifest.json --model runs/release-27-public/detector_models_v9_false_positive/footbag_detector_best.pt --calibration-metrics runs/release-27-public/detector_model_evaluation_v9_false_positive_calibrated/detector_model_metrics.json --out-root runs/release-27-public/detector_inference_v9_calibrated_batch_300 --max-frames 300 --imgsz 640 --continue-on-error
python3 hackytrack.py summarize-detector-batch --batch-manifest runs/release-27-public/detector_inference_v9_calibrated_batch_300/detector_batch_manifest.json --out-dir runs/release-27-public/detector_batch_summary_v9_calibrated_300 --high-prediction-share 0.35 --low-model-coverage 0.20
python3 hackytrack.py detector-false-positive-review --qa-manifest runs/release-27-public/qa_sidefix_v7/qa_manifest.json --inference-root runs/release-27-public/detector_inference_v9_calibrated_batch_300 --dataset-manifest runs/release-27-public/detector_dataset_v9_false_positive/manifest.json --out-dir runs/release-27-public/detector_false_positive_review_v9_calibrated_batch_300 --max-items 160 --per-video 8 --min-confidence 0.011504 --max-confidence 1.0 --crop-size 192 --cols 5
python3 hackytrack.py assist-detector-labels --review-manifest runs/release-27-public/detector_false_positive_review_v9_calibrated_batch_300/detector_false_positive_review_manifest.json --out runs/release-27-public/detector_false_positive_review_v9_calibrated_batch_300/detector_false_positive_assisted_decisions.json --min-confidence 0.70
python3 hackytrack.py export-detector-dataset --qa-manifest runs/release-27-public/qa_sidefix_v7/qa_manifest.json --reviews-dir runs/release-27-public/reviews_v6_seeded --out-dir runs/release-27-public/detector_dataset_v10_codex_recall --detector-label-review-manifest runs/release-27-public/detector_label_review_v1/detector_label_review_manifest.json --detector-label-decisions runs/release-27-public/detector_label_review_v1/detector_label_assisted_decisions.json --detector-label-review-pair runs/release-27-public/detector_false_positive_review_v1/detector_false_positive_review_manifest.json:runs/release-27-public/detector_false_positive_review_v1/detector_false_positive_assisted_decisions.json --detector-label-review-pair runs/release-27-public/detector_false_positive_review_v9_calibrated_batch_300/detector_false_positive_review_manifest.json:runs/release-27-public/detector_false_positive_review_v9_calibrated_batch_300/detector_false_positive_codex_visual_decisions.json
python3 hackytrack.py train-detector --dataset runs/release-27-public/detector_dataset_v10_codex_recall --out-dir runs/release-27-public/detector_models_v10_codex_recall --base-model yolo11n.pt --epochs 25 --imgsz 640 --batch -1 --run-name footbag-detector-v10-codex-recall
python3 hackytrack.py evaluate-detector-model --dataset runs/release-27-public/detector_dataset_v10_codex_recall --model runs/release-27-public/detector_models_v10_codex_recall/footbag_detector_best.pt --out-dir runs/release-27-public/detector_model_evaluation_v10_codex_recall_calibrated --confidence-threshold 0.001 --release-confidence-threshold 0.25 --calibration-split validation --max-hard-negative-false-positive-rate 0 --tolerance-px 24 --box-tolerance-multiplier 0.75 --imgsz 640 --max-det 300
python3 hackytrack.py detect-footbag-batch --qa-manifest runs/release-27-public/qa_sidefix_v7/qa_manifest.json --model runs/release-27-public/detector_models_v10_codex_recall/footbag_detector_best.pt --calibration-metrics runs/release-27-public/detector_model_evaluation_v10_codex_recall_calibrated/detector_model_metrics.json --out-root runs/release-27-public/detector_inference_v10_calibrated_batch_300_processed --process-width 688 --process-height 912 --max-frames 300 --imgsz 640 --continue-on-error
python3 hackytrack.py summarize-detector-batch --batch-manifest runs/release-27-public/detector_inference_v10_calibrated_batch_300_processed/detector_batch_manifest.json --out-dir runs/release-27-public/detector_batch_summary_v10_calibrated_300_processed --high-prediction-share 0.35 --low-model-coverage 0.20
python3 hackytrack.py evaluate-detector --dataset runs/release-27-public/detector_dataset_v10_codex_recall --tracks-root runs/release-27-public/detector_inference_v10_calibrated_batch_300_processed --out-dir runs/release-27-public/detector_track_evaluation_v10_calibrated_batch_300_processed
python3 -m unittest tests.test_build_detector_error_review_batch tests.test_release_cli
python3 -m py_compile build_detector_error_review_batch.py hackytrack.py
python3 hackytrack.py detector-error-review --qa-manifest runs/release-27-public/qa_sidefix_v7/qa_manifest.json --track-metrics runs/release-27-public/detector_track_evaluation_v10_calibrated_batch_300_processed/detector_track_metrics.json --dataset runs/release-27-public/detector_dataset_v10_codex_recall --model-metrics runs/release-27-public/detector_model_evaluation_v10_codex_recall_calibrated/detector_model_metrics.json --out-dir runs/release-27-public/detector_error_review_v11 --max-items 120 --per-video 12 --crop-size 224 --cols 4 --low-confidence-per-split 8
python3 hackytrack.py export-detector-dataset --qa-manifest runs/release-27-public/qa_sidefix_v7/qa_manifest.json --reviews-dir runs/release-27-public/reviews_v6_seeded --out-dir runs/release-27-public/detector_dataset_v11_error_recovery --detector-label-review-manifest runs/release-27-public/detector_label_review_v1/detector_label_review_manifest.json --detector-label-decisions runs/release-27-public/detector_label_review_v1/detector_label_assisted_decisions.json --detector-label-review-pair runs/release-27-public/detector_false_positive_review_v1/detector_false_positive_review_manifest.json:runs/release-27-public/detector_false_positive_review_v1/detector_false_positive_assisted_decisions.json --detector-label-review-pair runs/release-27-public/detector_false_positive_review_v9_calibrated_batch_300/detector_false_positive_review_manifest.json:runs/release-27-public/detector_false_positive_review_v9_calibrated_batch_300/detector_false_positive_codex_visual_decisions.json --detector-label-review-pair runs/release-27-public/detector_error_review_v11/detector_error_review_manifest.json:runs/release-27-public/detector_error_review_v11/detector_error_codex_decisions.json
python3 hackytrack.py train-detector --dataset runs/release-27-public/detector_dataset_v11_error_recovery --out-dir runs/release-27-public/detector_models_v11_error_recovery --base-model yolo11n.pt --epochs 40 --imgsz 640 --batch -1 --run-name footbag-detector-v11-error-recovery
python3 hackytrack.py evaluate-detector-model --dataset runs/release-27-public/detector_dataset_v11_error_recovery --model runs/release-27-public/detector_models_v11_error_recovery/footbag_detector_best.pt --out-dir runs/release-27-public/detector_model_evaluation_v11_error_recovery_calibrated --confidence-threshold 0.001 --release-confidence-threshold 0.25 --calibration-split validation --max-hard-negative-false-positive-rate 0 --tolerance-px 24 --box-tolerance-multiplier 0.75 --imgsz 640 --max-det 300
python3 hackytrack.py detect-footbag-batch --qa-manifest runs/release-27-public/qa_sidefix_v7/qa_manifest.json --model runs/release-27-public/detector_models_v11_error_recovery/footbag_detector_best.pt --calibration-metrics runs/release-27-public/detector_model_evaluation_v11_error_recovery_calibrated/detector_model_metrics.json --out-root runs/release-27-public/detector_inference_v11_calibrated_batch_300_processed --process-width 688 --process-height 912 --max-frames 300 --imgsz 640 --continue-on-error
python3 hackytrack.py summarize-detector-batch --batch-manifest runs/release-27-public/detector_inference_v11_calibrated_batch_300_processed/detector_batch_manifest.json --out-dir runs/release-27-public/detector_batch_summary_v11_calibrated_300_processed --high-prediction-share 0.35 --low-model-coverage 0.20
python3 hackytrack.py evaluate-detector --dataset runs/release-27-public/detector_dataset_v11_error_recovery --tracks-root runs/release-27-public/detector_inference_v11_calibrated_batch_300_processed --out-dir runs/release-27-public/detector_track_evaluation_v11_calibrated_batch_300_processed
python3 hackytrack.py detect-footbag-batch --qa-manifest runs/release-27-public/qa_sidefix_v7/qa_manifest.json --model runs/release-27-public/detector_models_v10_codex_recall/footbag_detector_best.pt --calibration-metrics runs/release-27-public/detector_model_evaluation_v10_codex_recall_calibrated/detector_model_metrics.json --out-root runs/release-27-public/detector_inference_v10_calibrated_batch_300_processed_temporal --process-width 688 --process-height 912 --max-frames 300 --imgsz 640 --tracker-mode temporal --continue-on-error
python3 hackytrack.py summarize-detector-batch --batch-manifest runs/release-27-public/detector_inference_v10_calibrated_batch_300_processed_temporal/detector_batch_manifest.json --out-dir runs/release-27-public/detector_batch_summary_v10_calibrated_300_processed_temporal --high-prediction-share 0.35 --low-model-coverage 0.20
python3 hackytrack.py evaluate-detector --dataset runs/release-27-public/detector_dataset_v10_codex_recall --tracks-root runs/release-27-public/detector_inference_v10_calibrated_batch_300_processed_temporal --out-dir runs/release-27-public/detector_track_evaluation_v10_calibrated_batch_300_processed_temporal
python3 -m unittest tests.test_build_dense_trajectory_review_batch tests.test_evaluate_dense_trajectory tests.test_release_cli
python3 -m py_compile build_dense_trajectory_review_batch.py evaluate_dense_trajectory.py hackytrack.py
python3 hackytrack.py dense-trajectory-review --qa-manifest runs/release-27-public/qa_sidefix_v7/qa_manifest.json --dataset runs/release-27-public/detector_dataset_v11_error_recovery --track-metrics runs/release-27-public/detector_track_evaluation_v10_calibrated_batch_300_processed/detector_track_metrics.json --track-metrics runs/release-27-public/detector_track_evaluation_v11_calibrated_batch_300_processed/detector_track_metrics.json --batch-summary runs/release-27-public/detector_batch_summary_v10_calibrated_300_processed/detector_batch_summary.json --batch-summary runs/release-27-public/detector_batch_summary_v10_calibrated_300_processed_temporal/detector_batch_summary.json --batch-summary runs/release-27-public/detector_batch_summary_v11_calibrated_300_processed/detector_batch_summary.json --track-root v10_greedy:runs/release-27-public/detector_inference_v10_calibrated_batch_300_processed --track-root v10_temporal:runs/release-27-public/detector_inference_v10_calibrated_batch_300_processed_temporal --track-root v11_greedy:runs/release-27-public/detector_inference_v11_calibrated_batch_300_processed --target-video 'video-352_singular_display 2.MOV' --target-video 'video-230_singular_display 2.MOV' --target-video 'video-234_singular_display 2.MOV' --out-dir runs/release-27-public/dense_trajectory_review_v1 --max-clips 9 --per-video 3 --seconds-before 1.0 --seconds-after 1.0 --frame-stride 1 --max-time-sec 10 --contact-sheet-cols 4 --contact-sheet-every 10 --max-contact-sheet-frames 48
python3 hackytrack.py evaluate-dense-trajectory --labels-jsonl runs/release-27-public/dense_trajectory_review_v1/dense_trajectory_labels_template.jsonl --tracks-root runs/release-27-public/detector_inference_v10_calibrated_batch_300_processed_temporal --out-dir runs/release-27-public/dense_trajectory_review_v1/template_eval_v10_temporal --confidence-threshold 0.01
python3 -m unittest discover tests
python3 -m py_compile hackytrack.py build_detector_error_review_batch.py build_dense_trajectory_review_batch.py evaluate_dense_trajectory.py export_detector_dataset.py build_detector_label_review_batch.py assist_detector_label_decisions.py train_footbag_detector.py footbag_detector_inference.py apply_detector_track_to_qa.py evaluate_detector_tracks.py evaluate_detector_model.py build_detector_false_positive_review_batch.py patch_footbag_detector.py release_candidate_report.py release_evaluation.py review_app.py qa_rally_enrichment.py render_best_rally_hud.py summarize_detector_batch.py
python3 -m unittest tests.test_evaluate_detector_model tests.test_train_footbag_detector tests.test_release_cli
python3 hackytrack.py train-detector --dataset runs/release-27-public/detector_dataset_v8_assisted --out-dir runs/release-27-public/detector_models_v8_assisted --base-model yolo11n.pt --epochs 25 --imgsz 640 --batch -1 --run-name footbag-detector-v8-assisted --recover-existing
python3 hackytrack.py detect-footbag-batch --qa-manifest runs/release-27-public/qa_sidefix_v7/qa_manifest.json --model runs/release-27-public/detector_models_v8_assisted/footbag_detector_best.pt --out-root runs/release-27-public/detector_inference_v8_assisted_smoke --max-videos 3 --max-frames 180 --confidence-threshold 0.05
python3 hackytrack.py evaluate-detector-model --dataset runs/release-27-public/detector_dataset_v8_assisted --model runs/release-27-public/detector_models_v8_assisted/footbag_detector_best.pt --out-dir runs/release-27-public/detector_model_evaluation_v8_assisted --confidence-threshold 0.001 --release-confidence-threshold 0.25 --tolerance-px 24 --box-tolerance-multiplier 0.75 --imgsz 640 --max-det 300
python3 hackytrack.py evaluate-detector --dataset runs/release-27-public/detector_dataset_v8_assisted --tracks-root runs/release-27-public/detector_inference_v8_assisted_smoke --out-dir runs/release-27-public/detector_track_evaluation_v8_assisted_smoke
python3 patch_footbag_detector.py train --dataset runs/release-27-public/detector_dataset_v8_assisted --out-dir runs/release-27-public/patch_detector_v2_assisted --crop-size 96 --jitter 4 --negatives-per-image 18 --seed 1337
python3 patch_footbag_detector.py evaluate --dataset runs/release-27-public/detector_dataset_v8_assisted --model runs/release-27-public/patch_detector_v2_assisted/patch_footbag_detector.joblib --out-dir runs/release-27-public/patch_detector_eval_v2_assisted_top100 --threshold 0.05 --tolerance-px 24 --box-tolerance-multiplier 0.75 --max-candidates 320 --max-detections 100
python3 hackytrack.py evaluate-patch-detector --dataset runs/release-27-public/detector_dataset_v8_assisted --model runs/release-27-public/patch_detector_v2_assisted/patch_footbag_detector.joblib --out-dir runs/release-27-public/patch_detector_eval_v2_assisted_cli --threshold 0.05 --tolerance-px 24 --box-tolerance-multiplier 0.75 --max-candidates 320 --max-detections 100
python3 hackytrack.py detect-footbag --video '/Users/isaacaudet/Downloads/video-352_singular_display 2.MOV' --patch-model runs/release-27-public/patch_detector_v2_assisted/patch_footbag_detector.joblib --out-dir 'runs/release-27-public/patch_detector_inference_v2_smoke/video-352_singular_display 2' --process-width 688 --process-height 912 --patch-threshold 0.50 --patch-max-candidates 320 --patch-max-detections 8 --max-frames 180 --every-nth-frame 1 --max-gap-frames 6 --max-jump-px 150
python3 hackytrack.py evaluate-detector --dataset runs/release-27-public/detector_dataset_v8_assisted --tracks-root runs/release-27-public/patch_detector_inference_v2_smoke --out-dir runs/release-27-public/patch_detector_track_eval_v2_smoke_t050
python3 hackytrack.py detector-false-positive-review --qa-manifest runs/release-27-public/qa_sidefix_v7/qa_manifest.json --inference-root runs/release-27-public/patch_detector_inference_v2_smoke --dataset-manifest runs/release-27-public/detector_dataset_v8_assisted/manifest.json --out-dir runs/release-27-public/patch_detector_false_positive_review_v2_smoke_highconf --max-items 40 --per-video 40 --min-confidence 0.5 --max-confidence 1.0 --crop-size 192 --cols 5
python3 hackytrack.py assist-detector-labels --review-manifest runs/release-27-public/patch_detector_false_positive_review_v2_smoke_highconf/detector_false_positive_review_manifest.json --out runs/release-27-public/patch_detector_false_positive_review_v2_smoke_highconf/detector_false_positive_assisted_decisions.json --min-confidence 0.70
python3 patch_footbag_detector.py train --dataset runs/release-27-public/detector_dataset_v8_assisted --out-dir runs/release-27-public/patch_detector_v3_assisted --crop-size 96 --jitter 4 --negatives-per-image 18 --seed 1337
python3 hackytrack.py evaluate-patch-detector --dataset runs/release-27-public/detector_dataset_v8_assisted --model runs/release-27-public/patch_detector_v3_assisted/patch_footbag_detector.joblib --out-dir runs/release-27-public/patch_detector_eval_v3_assisted_cli --threshold 0.05 --tolerance-px 24 --box-tolerance-multiplier 0.75 --max-candidates 320 --max-detections 100
python3 -m unittest tests.test_patch_footbag_detector tests.test_release_cli tests.test_footbag_detector_inference
python3 -m py_compile patch_footbag_detector.py hackytrack.py footbag_detector_inference.py
python3 -m unittest discover tests
python3 -m py_compile hackytrack.py footbag_detector_inference.py patch_footbag_detector.py build_detector_false_positive_review_batch.py export_detector_dataset.py train_footbag_detector.py evaluate_detector_model.py evaluate_detector_tracks.py apply_detector_track_to_qa.py
python3 hackytrack.py detect-footbag-batch --qa-manifest runs/release-27-public/qa_sidefix_v7/qa_manifest.json --patch-model runs/release-27-public/patch_detector_v3_assisted/patch_footbag_detector.joblib --out-root runs/release-27-public/patch_detector_inference_v3_27_smoke --process-width 688 --process-height 912 --patch-threshold 0.50 --patch-max-candidates 320 --patch-max-detections 8 --max-frames 180 --every-nth-frame 1 --max-gap-frames 6 --max-jump-px 150 --continue-on-error
python3 hackytrack.py detector-false-positive-review --qa-manifest runs/release-27-public/qa_sidefix_v7/qa_manifest.json --batch-manifest runs/release-27-public/patch_detector_inference_v3_27_smoke/detector_batch_manifest.json --dataset-manifest runs/release-27-public/detector_dataset_v8_assisted/manifest.json --out-dir runs/release-27-public/patch_detector_false_positive_review_v3_27_highconf --max-items 120 --per-video 6 --min-confidence 0.5 --max-confidence 1.0 --crop-size 192 --cols 5
python3 hackytrack.py export-detector-dataset --qa-manifest runs/release-27-public/qa_sidefix_v7/qa_manifest.json --reviews-dir runs/release-27-public/reviews_v6_seeded --out-dir runs/release-27-public/detector_dataset_v9_codex_hardneg --detector-label-review-manifest runs/release-27-public/detector_label_review_v1/detector_label_review_manifest.json --detector-label-decisions runs/release-27-public/detector_label_review_v1/detector_label_assisted_decisions.json --detector-label-review-pair runs/release-27-public/patch_detector_false_positive_review_v3_27_highconf/detector_false_positive_review_manifest.json:runs/release-27-public/patch_detector_false_positive_review_v3_27_highconf/detector_false_positive_codex_visual_decisions.json
python3 hackytrack.py train-patch-detector --dataset runs/release-27-public/detector_dataset_v9_codex_hardneg --out-dir runs/release-27-public/patch_detector_v5_extra_trees_hardneg --crop-size 96 --jitter 4 --negatives-per-image 18 --seed 1337 --model-kind extra-trees
python3 hackytrack.py evaluate-patch-detector --dataset runs/release-27-public/detector_dataset_v9_codex_hardneg --model runs/release-27-public/patch_detector_v5_extra_trees_hardneg/patch_footbag_detector.joblib --out-dir runs/release-27-public/patch_detector_eval_v5_extra_trees_hardneg_t035 --threshold 0.35 --tolerance-px 24 --box-tolerance-multiplier 0.75 --max-candidates 320 --max-detections 100
python3 -m unittest tests.test_build_detector_false_positive_review_batch tests.test_assist_detector_label_decisions tests.test_release_cli
python3 -m py_compile build_detector_false_positive_review_batch.py assist_detector_label_decisions.py hackytrack.py
python3 -m unittest tests.test_patch_footbag_detector tests.test_release_cli
python3 -m py_compile patch_footbag_detector.py hackytrack.py
python3 -m unittest discover tests
python3 -m py_compile hackytrack.py footbag_detector_inference.py patch_footbag_detector.py build_detector_false_positive_review_batch.py export_detector_dataset.py train_footbag_detector.py evaluate_detector_model.py evaluate_detector_tracks.py apply_detector_track_to_qa.py
python3 -m unittest tests.test_patch_footbag_detector tests.test_release_cli && python3 -m py_compile patch_footbag_detector.py hackytrack.py
python3 -m unittest discover tests && python3 -m py_compile hackytrack.py footbag_detector_inference.py patch_footbag_detector.py build_detector_false_positive_review_batch.py export_detector_dataset.py train_footbag_detector.py evaluate_detector_model.py evaluate_detector_tracks.py apply_detector_track_to_qa.py
python3 -m unittest tests.test_review_batch_io && python3 -m py_compile seed_review_batch.py apply_review_decisions.py hackytrack.py
python3 -m unittest discover tests
python3 -m py_compile hackytrack.py prepare_review_evidence.py assist_review_batch.py release_evaluation.py review_app.py qa_rally_enrichment.py review_validation.py render_best_rally_hud.py build_review_batch.py strict_rally_audit.py apply_reviews_to_qa.py release_candidate_report.py audit_ball_tracking.py full_training_run.py train_multimodal_detector.py
python3 -m unittest tests.test_release_candidate_report && python3 -m py_compile release_candidate_report.py
python3 -m unittest tests.test_contact_gating tests.test_review_evidence tests.test_release_cli
python3 -m py_compile hackytrack.py prepare_review_evidence.py assist_review_batch.py qa_rally_enrichment.py review_app.py build_review_batch.py apply_reviews_to_qa.py
python3 hackytrack.py review-evidence --run-dir runs/release-27-public --batch runs/release-27-public/review_batches_sidefix_v3/latest_review_batch.json --reviews-dir runs/release-27-public/reviews_v3_blocker --out-dir runs/release-27-public/review_evidence_cli_smoke --max-items-per-bucket 4
python3 -m unittest tests.test_release_evaluation tests.test_release_cli
python3 -m unittest tests.test_active_learning_review_batch tests.test_release_evaluation tests.test_release_cli
python3 -m unittest tests.test_contact_gating tests.test_active_learning_review_batch tests.test_release_evaluation tests.test_release_cli
python3 -m unittest tests.test_active_learning_review_batch tests.test_strict_rally_audit tests.test_contact_gating tests.test_release_cli
python3 -m unittest tests.test_apply_reviews_to_qa tests.test_active_learning_review_batch tests.test_strict_rally_audit
python3 -m unittest tests.test_release_candidate_report tests.test_release_cli
python3 -m unittest discover tests
python3 -m py_compile hackytrack.py release_evaluation.py review_app.py qa_rally_enrichment.py review_validation.py render_best_rally_hud.py build_review_batch.py
python3 -m py_compile strict_rally_audit.py build_review_batch.py hackytrack.py render_best_rally_hud.py qa_rally_enrichment.py review_app.py
python3 -m py_compile apply_reviews_to_qa.py hackytrack.py strict_rally_audit.py build_review_batch.py qa_rally_enrichment.py review_app.py
python3 -m py_compile release_candidate_report.py hackytrack.py
python3 hackytrack.py verify --run-dir runs/smoke-video-482-release-eval
python3 hackytrack.py apply-reviews --dry-run --run-dir runs/release-27-public --qa-manifest runs/release-27-public/qa_sidefix/qa_manifest.json --reviews-dir runs/release-27-public/reviews_gap_apply_demo --out-root runs/release-27-public/qa_sidefix_reviewed_gap_demo_cli --strict-audit-out runs/release-27-public/strict_rally_audit_reviewed_gap_demo_cli
python3 release_candidate_report.py --run-dir runs/release-27-public --out-dir runs/release-27-public/release_report_sidefix --qa-manifest runs/release-27-public/qa_sidefix/qa_manifest.json --review-batch runs/release-27-public/review_batches_sidefix_gap/latest_review_batch.json --strict-rally-audit runs/release-27-public/strict_rally_audit_sidefix/strict_rally_audit.json --qa-reviewed-manifest runs/release-27-public/qa_sidefix_reviewed_gap_demo/qa_manifest.json --strict-rally-audit-reviewed runs/release-27-public/strict_rally_audit_reviewed_gap_demo/strict_rally_audit.json --tests-passed --test-evidence "python3 -m unittest discover tests; py_compile core scripts"
python3 qa_rally_enrichment.py --manifest runs/release-27-public/training/full_training_manifest.json --out-root runs/release-27-public/qa_sidefix_v2
python3 strict_rally_audit.py --manifest runs/release-27-public/qa_sidefix_v2/qa_manifest.json --out-dir runs/release-27-public/strict_rally_audit_sidefix_v2
python3 build_review_batch.py --manifest runs/release-27-public/qa_sidefix_v2/qa_manifest.json --out-dir runs/release-27-public/review_batches_sidefix_v2 --max-items 96
python3 apply_reviews_to_qa.py --manifest runs/release-27-public/qa_sidefix_v2/qa_manifest.json --reviews-dir runs/release-27-public/reviews_gap_apply_demo --out-root runs/release-27-public/qa_sidefix_v2_reviewed_gap_demo
python3 strict_rally_audit.py --manifest runs/release-27-public/qa_sidefix_v2_reviewed_gap_demo/qa_manifest.json --out-dir runs/release-27-public/strict_rally_audit_sidefix_v2_reviewed_gap_demo
python3 release_candidate_report.py --run-dir runs/release-27-public --out-dir runs/release-27-public/release_report_sidefix_v2 --qa-manifest runs/release-27-public/qa_sidefix_v2/qa_manifest.json --review-batch runs/release-27-public/review_batches_sidefix_v2/latest_review_batch.json --strict-rally-audit runs/release-27-public/strict_rally_audit_sidefix_v2/strict_rally_audit.json --qa-reviewed-manifest runs/release-27-public/qa_sidefix_v2_reviewed_gap_demo/qa_manifest.json --strict-rally-audit-reviewed runs/release-27-public/strict_rally_audit_sidefix_v2_reviewed_gap_demo/strict_rally_audit.json --tests-passed --test-evidence "python3 -m unittest discover tests; py_compile core scripts"
python3 qa_rally_enrichment.py --manifest runs/release-27-public/training/full_training_manifest.json --out-root runs/release-27-public/qa_sidefix_v3
python3 strict_rally_audit.py --manifest runs/release-27-public/qa_sidefix_v3/qa_manifest.json --out-dir runs/release-27-public/strict_rally_audit_sidefix_v3
python3 build_review_batch.py --manifest runs/release-27-public/qa_sidefix_v3/qa_manifest.json --out-dir runs/release-27-public/review_batches_sidefix_v3 --max-items 96
python3 apply_reviews_to_qa.py --manifest runs/release-27-public/qa_sidefix_v3/qa_manifest.json --reviews-dir runs/release-27-public/reviews_gap_apply_demo --out-root runs/release-27-public/qa_sidefix_v3_reviewed_gap_demo
python3 strict_rally_audit.py --manifest runs/release-27-public/qa_sidefix_v3_reviewed_gap_demo/qa_manifest.json --out-dir runs/release-27-public/strict_rally_audit_sidefix_v3_reviewed_gap_demo
python3 release_candidate_report.py --run-dir runs/release-27-public --out-dir runs/release-27-public/release_report_sidefix_v3 --qa-manifest runs/release-27-public/qa_sidefix_v3/qa_manifest.json --review-batch runs/release-27-public/review_batches_sidefix_v3/latest_review_batch.json --strict-rally-audit runs/release-27-public/strict_rally_audit_sidefix_v3/strict_rally_audit.json --qa-reviewed-manifest runs/release-27-public/qa_sidefix_v3_reviewed_gap_demo/qa_manifest.json --strict-rally-audit-reviewed runs/release-27-public/strict_rally_audit_sidefix_v3_reviewed_gap_demo/strict_rally_audit.json --tests-passed --test-evidence "python3 -m unittest discover tests; py_compile core scripts"
python3 review_app.py --host 127.0.0.1 --port 8765 --qa-manifest runs/release-27-public/qa_sidefix_v3/qa_manifest.json --review-batch runs/release-27-public/review_batches_sidefix_v3/latest_review_batch.json --reviews-dir runs/release-27-public/reviews_v3_blocker
python3 apply_reviews_to_qa.py --manifest runs/release-27-public/qa_sidefix_v3/qa_manifest.json --reviews-dir runs/release-27-public/reviews_v3_blocker --out-root runs/release-27-public/qa_sidefix_v3_reviewed_blocker
python3 review_validation.py --reviews-dir runs/release-27-public/reviews_v3_blocker --out-dir runs/release-27-public/validation_v3_blocker --batch runs/release-27-public/review_batches_sidefix_v3/latest_review_batch.json
python3 strict_rally_audit.py --manifest runs/release-27-public/qa_sidefix_v3_reviewed_blocker/qa_manifest.json --out-dir runs/release-27-public/strict_rally_audit_sidefix_v3_reviewed_blocker
python3 release_evaluation.py --reviews-dir runs/release-27-public/reviews_v3_blocker --batch runs/release-27-public/review_batches_sidefix_v3/latest_review_batch.json --out-dir runs/release-27-public/release_evaluation_v3_blocker --training-manifest runs/release-27-public/training/full_training_manifest.json --model-dir runs/release-27-public/models --ball-audit runs/release-27-public/ball_tracking_audit/audit_metrics.json
python3 render_best_rally_hud.py --manifest runs/release-27-public/qa_sidefix_v3_reviewed_blocker/qa_manifest.json --out-dir runs/release-27-public/hud_sidefix_v3_reviewed_blocker
python3 release_candidate_report.py --run-dir runs/release-27-public --out-dir runs/release-27-public/release_report_v3_reviewed_blocker --qa-manifest runs/release-27-public/qa_sidefix_v3_reviewed_blocker/qa_manifest.json --review-batch runs/release-27-public/review_batches_sidefix_v3/latest_review_batch.json --strict-rally-audit runs/release-27-public/strict_rally_audit_sidefix_v3_reviewed_blocker/strict_rally_audit.json --qa-reviewed-manifest runs/release-27-public/qa_sidefix_v3_reviewed_blocker/qa_manifest.json --strict-rally-audit-reviewed runs/release-27-public/strict_rally_audit_sidefix_v3_reviewed_blocker/strict_rally_audit.json --validation runs/release-27-public/validation_v3_blocker/validation_metrics.json --release-metrics runs/release-27-public/release_evaluation_v3_blocker/release_metrics.json --hud-video runs/release-27-public/hud_sidefix_v3_reviewed_blocker/best_rally_sprite_hud_overlay.mp4 --hud-summary runs/release-27-public/hud_sidefix_v3_reviewed_blocker/best_rally_sprite_hud_summary.json --hud-verify runs/release-27-public/hud_sidefix_v3_reviewed_blocker/hud_verification.json --tests-passed --test-evidence "python3 -m unittest discover tests; py_compile core scripts; HUD verify audio/video/nonblank"
python3 -m unittest discover tests
python3 -m py_compile hackytrack.py release_evaluation.py review_app.py qa_rally_enrichment.py review_validation.py render_best_rally_hud.py build_review_batch.py strict_rally_audit.py apply_reviews_to_qa.py release_candidate_report.py audit_ball_tracking.py assist_review_batch.py full_training_run.py train_multimodal_detector.py
python3 -m unittest tests.test_export_detector_dataset tests.test_release_cli
python3 hackytrack.py export-detector-dataset --qa-manifest runs/release-27-public/qa_sidefix_v7/qa_manifest.json --reviews-dir runs/release-27-public/reviews_v6_seeded --out-dir runs/release-27-public/detector_dataset_v7_reviewed
python3 -m unittest tests.test_train_footbag_detector
python3 train_footbag_detector.py --dataset runs/release-27-public/detector_dataset_v7_reviewed --out-dir runs/release-27-public/detector_models_v7 --dry-run
python3 hackytrack.py train-detector --dataset runs/release-27-public/detector_dataset_v7_reviewed --out-dir runs/release-27-public/detector_models_v7_reviewed --base-model yolo11n.pt --epochs 3 --imgsz 640 --batch -1 --run-name footbag-detector-v7-reviewed-metrics
python3 -m unittest tests.test_export_detector_dataset tests.test_build_detector_label_review_batch tests.test_release_cli
python3 hackytrack.py detector-label-review --qa-manifest runs/release-27-public/qa_sidefix_v7/qa_manifest.json --reviews-dir runs/release-27-public/reviews_v6_seeded --dataset-manifest runs/release-27-public/detector_dataset_v7_reviewed/manifest.json --out-dir runs/release-27-public/detector_label_review_v1 --max-items 135 --per-video 5 --crop-size 192 --cols 5
python3 -m unittest tests.test_assist_detector_label_decisions tests.test_release_cli
python3 hackytrack.py assist-detector-labels --review-manifest runs/release-27-public/detector_label_review_v1/detector_label_review_manifest.json --out runs/release-27-public/detector_label_review_v1/detector_label_assisted_decisions.json --min-confidence 0.70
python3 export_detector_dataset.py --qa-manifest runs/release-27-public/qa_sidefix_v7/qa_manifest.json --reviews-dir runs/release-27-public/reviews_v6_seeded --out-dir /tmp/hackytrack_detector_dataset_pending_decisions --detector-label-review-manifest runs/release-27-public/detector_label_review_v1/detector_label_review_manifest.json --detector-label-decisions runs/release-27-public/detector_label_review_v1/detector_label_decisions_template.json --dry-run
python3 hackytrack.py export-detector-dataset --qa-manifest runs/release-27-public/qa_sidefix_v7/qa_manifest.json --reviews-dir runs/release-27-public/reviews_v6_seeded --out-dir runs/release-27-public/detector_dataset_v8_assisted --detector-label-review-manifest runs/release-27-public/detector_label_review_v1/detector_label_review_manifest.json --detector-label-decisions runs/release-27-public/detector_label_review_v1/detector_label_assisted_decisions.json
python3 hackytrack.py train-detector --dataset runs/release-27-public/detector_dataset_v8_assisted --out-dir runs/release-27-public/detector_models_v8_assisted --base-model yolo11n.pt --epochs 25 --imgsz 640 --batch -1 --run-name footbag-detector-v8-assisted
python3 -m unittest tests.test_footbag_detector_inference
python3 hackytrack.py detect-footbag-batch --qa-manifest runs/release-27-public/qa_sidefix_v7/qa_manifest.json --model runs/release-27-public/detector_models_v8_assisted/footbag_detector_best.pt --out-root runs/release-27-public/detector_inference_v8_assisted_smoke --max-videos 1 --max-frames 180 --confidence-threshold 0.05
python3 hackytrack.py detect-footbag-batch --qa-manifest runs/release-27-public/qa_sidefix_v7/qa_manifest.json --model runs/release-27-public/detector_models_v8_assisted/footbag_detector_best.pt --out-root runs/release-27-public/detector_inference_v8_assisted_smoke_lowconf_filtered --max-videos 1 --max-frames 180 --confidence-threshold 0.001
python3 -m unittest tests.test_evaluate_detector_tracks tests.test_apply_detector_track_to_qa tests.test_footbag_detector_inference tests.test_release_cli
python3 -m py_compile evaluate_detector_tracks.py apply_detector_track_to_qa.py footbag_detector_inference.py hackytrack.py train_footbag_detector.py export_detector_dataset.py
python3 -m unittest discover tests
python3 hackytrack.py detect-footbag --detections-jsonl /tmp/hackytrack_detector_fixture.jsonl --out-dir /tmp/hackytrack_detector_smoke --fps 30
python3 hackytrack.py detect-footbag-batch --qa-manifest /tmp/hackytrack_detector_batch/qa_manifest.json --detections-root /tmp/hackytrack_detector_batch/detections --out-root /tmp/hackytrack_detector_batch/out --fps 30
python3 hackytrack.py detect-footbag-batch --qa-manifest runs/release-27-public/qa_sidefix_v7/qa_manifest.json --model runs/release-27-public/detector_models_v7_reviewed/footbag_detector_best.pt --out-root runs/release-27-public/detector_inference_v7_smoke --max-videos 1 --max-frames 90 --confidence-threshold 0.05
python3 hackytrack.py apply-detector-track --qa-events /tmp/hackytrack_apply_track_smoke/qa_events.json --track-json /tmp/hackytrack_apply_track_smoke/track.json --out-events /tmp/hackytrack_apply_track_smoke/qa_events_model.json
python3 hackytrack.py evaluate-detector --labels-jsonl /tmp/hackytrack_detector_eval/labels.jsonl --hard-negatives-jsonl /tmp/hackytrack_detector_eval/hard_negatives.jsonl --tracks-root /tmp/hackytrack_detector_eval/tracks --out-dir /tmp/hackytrack_detector_eval/metrics
```

Release evaluation artifacts:

- Smoke run: `runs/smoke-video-482-hud/release_evaluation/release_metrics.json`
- 27-video reviewed set: `outputs/release_evaluation_27/release_metrics.json`
- Public 27-video run: `runs/release-27-public/release_evaluation/release_metrics.json`

Historical existing-batch blockers from `outputs/release_evaluation_27/release_metrics.json` before the public run rebuilt the review batch:

- Touch precision: `84.6%` vs target `90.0%`
- Drop/floor precision: `60.0%` vs target `90.0%`
- Knee: candidate-only, `4` reviewed examples vs required `20`, precision `0.0%`
- Tricks: candidate-only, `25.0%` reviewed precision vs required `80.0%`

## Remaining Release Blockers

- Detector replacement is not complete. v10 remains the strongest full-frame YOLO candidate, but processed-coordinate track evaluation still fails the release-quality center/error and hard-negative gates; v11 error recovery regressed from v10 and is evidence-only. Model-backed centers must not replace heuristic QA centers yet.
- Detector-backed evidence has not yet been propagated through a full reviewed release evaluation to prove improvements in touch precision/recall, drop detection, stall windows, side/contact labels, and HUD correctness on held-out reviewed videos.
- Public 27-video held-out gates now pass except knee, but the denominators are still small; the new active-learning proposals must be reviewed to strengthen recall claims.
- Contact side/type classification is stricter than before, but still needs real reviewed left/right labels before it can be treated as release-calibrated.
- Side classification now has confidence/source/reason fields, but it is still contour-geometry evidence rather than robust pose-trained side recognition; the `unknown` cases need review labels before training can claim the side target.
- The top 11-touch best-rally blocker has been reviewed and the HUD now verifies, but the rest of the `gap_without_floor_reset` active-learning proposals still need human review.
- The v3 reviewed-blocker metric report is intentionally not release-ready: one accepted blocker proves the review/apply/HUD path, but it is too small to satisfy held-out metric targets.
- Knee remains intentionally candidate-only; this is a guarded limitation, not a release blocker, until at least 20 reviewed knee examples exist with at least 80% precision.
- Stall-window recall target is only weakly proven because the held-out denominator is small.
- Active-learning review exists, but the newly proposed missing-event items have not yet been fully reviewed by a human.
- Full docs still need training/evaluation and troubleshooting sections beyond the quickstart.
- Release candidate changelog/tag is not complete.
