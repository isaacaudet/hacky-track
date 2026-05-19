# Detector Replacement Winning Plan

## Status

The detector stack is operational, but the detector is not release-ready.

Current best baseline:

- v10 full-frame YOLO detector remains the strongest model-backed candidate.
- v10 model-level localization is good when the model fires: candidate-center pass rate `0.985765`, mean center error `4.484px`, p95 `21.195px`, hard-negative FP rate `0.0`.
- v10 processed-video tracking still fails release gates: track pass rate `0.729537`, mean center error `35.548px`, p95 `226.593px`, missing track points `38`, hard-negative FP rate `0.125`, held-out test hard-negative FP rate `0.25`.

v11 result:

- v11 completed the error-recovery loop, but regressed from v10.
- Track pass rate fell to `0.597222`, mean center error rose to `57.876px`, p95 rose to `329.289px`, model coverage fell to `0.643594`, and interpolation share rose to `0.213643`.
- v11 is evidence-only and must not replace heuristic QA centers.

## Diagnosis

The core failure is not simply "the model cannot see a footbag." The v10 model can often localize near reviewed labels. The bigger failure is that the current data and runtime problem are mismatched:

- Sparse QA/touch labels do not teach continuous bag location.
- Single-frame full-frame detection is fragile for a tiny, fast, often occluded object.
- Greedy smoothing can turn sparse or wrong detections into long, confident-looking bad tracks.
- v11 added mostly positive recovery labels without adding enough train/validation hard negatives, so it reduced coverage and worsened the tracking tail.
- Current evaluation is event-frame-heavy; we need dense trajectory labels to train and evaluate the actual object-tracking task.

## Research Takeaways

The likely winning direction is a temporal heatmap tracker, not another sparse-label YOLO retrain.

Relevant prior art:

- TrackNet frames sports-ball tracking as heatmap prediction from single or consecutive frames, specifically for tiny, blurry, high-speed balls. Source: https://arxiv.org/abs/1907.03698
- TrackNetV4 improves this family by adding motion attention maps because visual features alone struggle under blur, occlusion, and low visibility. Source: https://arxiv.org/abs/2409.14543
- TTNet uses temporal/spatial video analysis for table tennis and reports strong small-ball coordinate accuracy with event context. Source: https://arxiv.org/abs/2004.09927
- TOTNet targets sports-ball occlusion directly with temporal 3D convolutions, visibility-weighted loss, and occlusion augmentation. Source: https://arxiv.org/abs/2508.09650
- ByteTrack's useful principle is to use low-confidence detections for association instead of discarding them too early. Source: https://arxiv.org/abs/2110.06864
- SAHI can improve small-object detection by slicing, but it still remains per-frame detection and should be treated as a candidate-mining aid, not the main solution. Source: https://arxiv.org/abs/2202.06934
- SAM 2 and CoTracker-style tools are useful for annotation assistance and QA, but should not become opaque release dependencies until they are measured on footbag clips. Sources: https://arxiv.org/abs/2408.00714 and https://arxiv.org/abs/2410.11831
- CVAT supports point-track and shape interpolation, which matches the dense trajectory labeling need. Sources: https://docs.cvat.ai/docs/annotation/manual-annotation/shapes/annotation-with-points/liner-interpolation-with-one-point/ and https://docs.cvat.ai/docs/annotation/manual-annotation/modes/track-mode-basics/

## Method To Build

Build v12 as a dense temporal footbag tracker:

1. Label dense short clips, not just touch frames.
2. Train a multi-frame heatmap model that predicts bag center and visibility.
3. Decode tracks with an offline global trajectory optimizer instead of greedy smoothing.
4. Use YOLO/v10, heuristics, SAM 2, CoTracker, and audio only as label-mining and audit aids.
5. Promote only if dense-frame, event-frame, hard-negative, and downstream release gates all beat v10 and the heuristic baseline.

## Data Strategy

Create a new dense trajectory dataset schema:

```json
{
  "clip_id": "video-352__frames_000000_000180",
  "source_video": "video-352_singular_display 2.MOV",
  "split": "validation",
  "frame_index": 123,
  "time_sec": 4.1,
  "x": 318.4,
  "y": 522.8,
  "radius": 10.0,
  "visibility": "visible",
  "occlusion": "none",
  "quality": "reviewed",
  "label_source": "human_dense"
}
```

Visibility states:

- `visible`: label center is directly visible.
- `partially_occluded`: center is inferred from a visible partial bag.
- `fully_occluded`: no center target for detection loss; may be used only for trajectory/visibility loss if confidently inferred.
- `out_of_frame`: no detection target.
- `uncertain`: audit only, not training.

Minimum first dense-label sprint:

- `video-352`: at least 4 clips around the worst failure windows.
- `video-230` and `video-234`: high-interpolation v11 clips.
- 4 to 6 train-split clips with clean bag visibility and varied shoes/floor/background.
- 2 validation clips from non-test videos.
- Keep existing test split untouched.
- Target `5,000` to `10,000` dense labeled frames before judging the heatmap architecture.
- Add at least `500` reviewed no-bag or not-bag hard-negative frames/points from train/validation split only: shoes, socks, hands, floor marks, shadows, red/blue/yellow non-bag patches, and the exact failure lookalikes from v10/v11.

Important rule:

- Do not train from held-out test failures. Test failures become audit items only.

## Model Architecture

Primary model:

- A small temporal U-Net or encoder-decoder heatmap model.
- Input: `5` or `7` consecutive processed-coordinate frames.
- Optional extra channels: frame differences, motion magnitude, HSV/red-blob evidence as audit/motion hints.
- Output for the center frame:
  - one Gaussian center heatmap
  - one visibility/objectness logit
  - optional radius/scale head

Loss:

- Heatmap focal/BCE or MSE against Gaussian target.
- Soft-argmax or coordinate loss for visible/partially visible frames.
- Visibility loss for visible vs no-target/occluded/out-of-frame frames.
- Hard-negative no-object loss for reviewed false regions and no-bag frames.
- Occlusion augmentation and motion blur augmentation.

Why this should beat YOLO:

- The output is a point heatmap, which matches the actual center-tracking task.
- Consecutive frames let the model learn motion, blur, and partial disappearance.
- Negative frames teach "do not hallucinate the bag" directly.
- The model does not need to learn a stable bounding box around a tiny deformed object.

Fallback/secondary experiments:

- Run v10 with existing `--tracker-mode temporal` as a cheap baseline, but do not treat that as the winning architecture.
- Try SAHI/sliced YOLO only as a candidate miner and comparison baseline.
- Try SAM 2 or CoTracker only to accelerate dense labeling; require local footbag validation before trusting them.

## Tracker Architecture

Replace greedy smoothing with an offline single-object trajectory decoder:

- Generate top-K heatmap peaks per frame, including low-confidence candidates.
- Add an explicit no-object/occluded state.
- Use dynamic programming or Viterbi over the full clip.
- Cost terms:
  - negative heatmap/objectness score
  - velocity and acceleration penalty
  - max-jump guard
  - penalty for switching to no-object
  - penalty for long unsupported interpolation
  - optional contact-window prior from QA/audio only as weak evidence, never as forced truth
- Smooth final coordinates with a Kalman/Rauch-Tung-Striebel pass only after selecting the global path.
- Preserve uncertainty: do not emit release-grade centers when the selected path is unsupported for too long.

This borrows the useful ByteTrack idea, but specialized for one tiny object: low-confidence candidates can maintain a track, but they should not become confident outputs unless the whole trajectory supports them.

## Evaluation Gates

Do not promote any v12 candidate unless it passes all of these.

Dense validation gate:

- Visible-frame center RMSE under `12px`.
- Visible-frame p95 center error under `30px`.
- Missing visible frame rate under `5%`.
- No-object/hard-negative false positive rate under `2%`.
- Occluded-frame visibility classification is calibrated enough that occluded frames do not become hallucinated precise centers.

Existing event-label gate:

- Beat v10 on processed-coordinate detector-track evaluation.
- Minimum target: overall pass rate at least `0.85`, validation pass rate at least `0.80`, held-out test pass rate at least `0.80`.
- Mean center error under `20px`.
- P95 center error under `60px`.
- Hard-negative FP rate `0.0` overall and `0.0` on held-out test.
- Prediction/interpolation share lower than v10's `0.087811`, unless dense labels prove those predicted spans are accurate.

Release analytics gate:

- Re-run full reviewed release evaluation with detector-backed evidence.
- Must improve or preserve:
  - ball-center accuracy
  - touch precision and recall
  - drop detection
  - stall windows
  - side/contact labels
  - HUD correctness
- Heuristic/audio evidence remains audit/fallback only until this gate passes.

## Execution Plan

Phase 0: cheap checks, no new labels

- Re-run v10 with current `--tracker-mode temporal`.
- Compare greedy vs temporal on the same processed-coordinate evaluation.
- Inspect v10/v11 failure CSVs and make a ranked list of the top 20 error windows by p95 contribution, missing points, and hard-negative proximity.

Exit criteria:

- If temporal mode alone materially improves v10, keep it as a baseline.
- If it does not, stop spending time on greedy/DP over YOLO boxes alone.

Phase 1: dense-label tooling

- Add a trajectory review/export format for frame-level point labels.
- Add a dense trajectory contact sheet/video renderer with model overlay and visibility state.
- Add CVAT import/export support or an internal review route for point tracks.
- Add split discipline: train/validation can be used for training; test remains audit-only.

Exit criteria:

- Can round-trip a labeled clip into `trajectory_labels.jsonl`.
- Can render labels over video for QA.
- Can evaluate a predicted dense track against dense labels.

Phase 2: dense data sprint

- Label the minimum first sprint listed above.
- Mine hard negatives from v10/v11 failures, but only from train/validation videos.
- Use v10 detections, heuristic centers, SAM 2, CoTracker, and CVAT interpolation only as prefill suggestions requiring review.

Exit criteria:

- `5,000+` dense labeled frames.
- `500+` reviewed train/validation hard-negative/no-object frames or points.
- At least one validation clip from the known hard class.

Phase 3: heatmap model MVP

- Implement a small PyTorch temporal heatmap trainer.
- Train on dense labels with visible/no-object/occlusion states.
- Export predictions as the same detector-track schema so existing evaluation tools can compare it to v10.

Exit criteria:

- Beats v10 model-level center quality on dense validation.
- Hard-negative FP under the dense validation target.

Phase 4: offline trajectory decoder

- Add top-K heatmap peak extraction.
- Add Viterbi/DP single-object path selection with no-object state.
- Add final smoothing and uncertainty flags.
- Compare against greedy, existing temporal path, ByteTrack-style low-confidence association, and raw heatmap argmax.

Exit criteria:

- Passes the existing event-label gate and dense validation gate.
- Specifically fixes video-352 without causing new held-out false positives.

Phase 5: release integration

- Attach detector-backed evidence into QA events.
- Re-run release metrics and HUD verification.
- Promote only if downstream metrics improve and no held-out gate regresses.

## What Not To Do Next

- Do not train v12 as another sparse QA-label YOLO run.
- Do not add test split failures into training.
- Do not lower confidence thresholds to recover coverage without a hard-negative pass.
- Do not let interpolation hide missing detections.
- Do not use SAM 2, CoTracker, HSV, or audio as ground truth.
- Do not claim HUD/touch/drop improvement from detector metrics alone; prove it in the release evaluation.

## First Concrete Next Step

Start with Phase 0 and Phase 1:

1. Run v10 processed-coordinate inference with `--tracker-mode temporal`.
2. Compare it against the v10 greedy baseline.
3. Build the dense trajectory label schema/export/evaluator.
4. Prepare the first dense-label batch around video-352, video-230, and video-234.

The winning bet is dense temporal supervision plus global single-object path decoding. Everything else should support that loop or be treated as a baseline.

## First Tranche Execution Notes

The first execution tranche has started.

- v10 was rerun with `--tracker-mode temporal` at `runs/release-27-public/detector_inference_v10_calibrated_batch_300_processed_temporal/`.
- Temporal mode completed all `27` videos and reduced interpolation share to `0.016353`, but did not beat v10 greedy on the main promotion gates: pass rate `0.72242` versus greedy `0.729537`, hard-negative FP rate still `0.125`.
- Dense trajectory tooling now exists:
  - `build_dense_trajectory_review_batch.py`
  - `hackytrack.py dense-trajectory-review`
  - `evaluate_dense_trajectory.py`
  - `hackytrack.py evaluate-dense-trajectory`
- First dense review batch: `runs/release-27-public/dense_trajectory_review_v1/`.
- Batch contents: `5` clips, `313` frame rows, `0` test/audit-only clips.
- Clip targets:
  - `video-352_singular_display 2.MOV` validation clips centered at `2.18s`, `4.999154s`, and `8.52s`.
  - `video-230_singular_display 2.MOV` train clip centered at `4.999485s`.
  - `video-234_singular_display 2.MOV` train clip centered at `4.999311s`.

Current blocker:

- The dense label template is still pending human review. Fill reviewed `x`, `y`, `visibility`, `occlusion`, and `quality` fields before temporal heatmap training.
