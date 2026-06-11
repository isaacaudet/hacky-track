# Footbag Tracking — Winning Plan V2 (Research-Backed)

> Supersedes the architecture sections of `DETECTOR_REPLACEMENT_WINNING_PLAN.md`.
> The v1 goals, evaluation discipline, and data-hygiene rules still hold. This
> document replaces *how* the tracker is built, based on a focused literature
> review (Sept 2024 – 2025 sources, see References).

**Goal:** Replace heuristic ball tracking with an explainable, data-efficient
tracker that hits the v1 metric gates without a multi-month from-scratch
labeling sprint.

---

## 1. What V1 Got Wrong

The v1 plan (`## Method To Build`, `## Model Architecture`) commits to:

> "Train a small temporal U-Net or encoder-decoder heatmap model … Target
> 5,000 to 10,000 dense labeled frames before judging the heatmap architecture."

This is the same bet that failed for v7–v11, just relabeled. Three problems:

1. **It reinvents an existing, pretrained model.** A "small temporal U-Net /
   encoder-decoder that outputs a Gaussian center heatmap from N consecutive
   frames" is *exactly* the TrackNet / WASB architecture — which already exists,
   is open-source (MIT), and ships **pretrained weights on tennis and badminton
   balls**. Those are small, fast, motion-blurred balls with flight dynamics
   close to a footbag. We should *fine-tune* that, not rebuild it.

2. **It still assumes the data wall.** From-scratch training needs the 5–10k
   frames. Fine-tuning a pretrained heatmap tracker needs a few hundred to a
   couple thousand — a transfer-learning regime, not a from-scratch regime.

3. **It never mentions camera ego-motion.** v1 plans an "offline trajectory
   decoder" with velocity/acceleration penalties, but in a POV video the ball's
   *image-space* path is not ballistic — camera head-motion is mixed in. Any
   physics prior (velocity smoothness, parabola fitting, gravity) is invalid
   until camera motion is removed. This is a hard prerequisite the v1 plan skips.

v1 also treats CoTracker / SAM 2 as "don't trust until validated." Correct for
*inference* — but wrong for *labeling*. They are the tool that breaks the data
wall (see Phase 2).

---

## 2. The Architecture: A Layered Pipeline

The core insight from the research: **stop asking one model to do everything.**
Separate the job into layers, each with a different strength and a different
failure mode. The detector does *recall*; physics does *precision*; pose does
*semantics*.

```
 raw POV video
     │
 ┌───▼─────────────────────────┐
 │ L0  Ego-motion compensation │  ORB+MAGSAC affine, ball masked, NO smoothing
 │     → stabilized frame      │  every ball position warped into one ref frame
 └───┬─────────────────────────┘
 ┌───▼─────────────────────────┐
 │ L1  Heatmap ball detector   │  fine-tuned WASB; per-frame candidate
 │     → candidate centers     │  centers + confidence. Recall-oriented.
 └───┬─────────────────────────┘
 ┌───▼─────────────────────────┐
 │ L2  Physics arc layer       │  RANSAC multi-arc parabola fit in the
 │     → smooth audited track  │  stabilized frame; factor-graph smoothing;
 │       + touch candidates    │  rejects detector outliers, fills gaps.
 └───┬─────────────────────────┘
 ┌───▼─────────────────────────┐
 │ L3  Touch + contact         │  touches = arc intersections, confirmed by
 │     → events                │  residual test + RTMPose foot proximity.
 └───┬─────────────────────────┘
   events.json / HUD
```

Why this design answers the v1 failures:

- **The detector no longer has to be release-grade.** Its job is to surface a
  few trustworthy points per ballistic arc. A "failing" detector at the v1
  95%-center-pass gate is still a perfectly good candidate miner. **Gate the
  system end-to-end, not L1 in isolation** — that single reframe probably makes
  the existing v10 checkpoint already useful.
- **Physics needs ~3 points per arc, not per-frame labels.** A parabola is
  6 parameters; RANSAC fits it from 3 detections and over-determines it with 5+.
  This is where the data-efficiency comes from.
- **Every output is explainable** — the stated v1 goal. An arc either fits the
  sparse detections or it visibly does not; a touch is a geometric event.

### L0 — Ego-motion compensation (vision-only)

> **Prototype result (`prototype_ego_motion.py`, see `PROTOTYPE_L0_FINDINGS.md`).**
> L0 was prototyped on video-506. Ego-motion estimation is robust (618 RANSAC
> inliers/frame). But the prototype found L0's role is **narrower than this plan
> first assumed**: a clean ballistic arc already fits a parabola at 13–22 px
> *in raw image coordinates*, even with 40–128 px of camera motion during the
> arc — because over a short arc, head motion is itself ≈polynomial and the
> parabola fit absorbs it. So **per-arc fitting does not need global
> stabilisation**, and naive frame-to-frame accumulation drifts badly. L0 is
> needed only *locally* — to chain adjacent arcs into one frame for touch
> detection and physical velocities. Keep L0 small and local; do not gate L2
> arc fitting on it. The dominant residual is ball-detection noise → L1 matters
> more than L0.

Per consecutive frame pair: detect background corners (`goodFeaturesToTrack`),
track them (`calcOpticalFlowPyrLK`) **with the detected ball region masked out**,
fit a 4-DOF similarity/affine transform with `cv2.estimateAffinePartial2D` or
`findHomography(..., cv2.USAC_MAGSAC)`. Accumulate transforms from a reference
frame. **Do not smooth the camera path** — cinematic stabilizers (L1-optimal,
MeshFlow, deep) produce a *fictional* smoothed path; we need the *true* motion
to subtract. Per-frame quality gate: if inlier count is too low, interpolate the
transform (head motion is locally smooth).

> **Critical open question — which camera?** "Meta Glasses" is ambiguous.
> *Consumer Ray-Ban Meta* records a plain MP4 with **no accessible IMU/gyro** →
> vision-only L0 as above. *Project Aria research glasses* expose IMU → camera
> rotation can be removed exactly and cheaply with gyro de-rotation
> (`x' = K[I − [ω]×Δt]K⁻¹ x`), which is far more robust. **Confirm this before
> building L0.** Check a sample file: `ffprobe -show_streams sample.mp4`.

Known limitation to design around: a 2D transform removes camera *rotation* well
but not parallax from camera *translation* (the wearer walking). For mostly
stationary kicking footage this residual is small and L2's RANSAC tolerates it.
If footage involves significant walking, that residual is the accuracy ceiling
of any vision-only approach — flag it, don't hide it.

### L1 — Heatmap ball detector (fine-tuned WASB)

Adopt **WASB** ("Widely Applicable Strong Baseline", BMVC 2023, MIT license).
It is a multi-frame heatmap tracker with a high-resolution backbone and built-in
temporal-consistency tracking; it beats all TrackNet variants across 5 sports.
Heatmap regression — not bounding boxes — is the right paradigm for a ~5–8 cm
ball: it gives sub-pixel centers, tolerates imprecise labels, and degrades
gracefully under blur (a blurred ball → a wider/lower heatmap blob).

- Fine-tune from WASB's pretrained tennis/badminton ball weights on our
  bootstrapped footbag labels (Phase 2). This is few-shot transfer, not
  from-scratch training.
- Upgrade path: **BlurBall** (2025, same lineage) explicitly predicts blur
  length + orientation and uses a center-of-streak labeling convention. Switch
  to it *only if* motion blur remains L1's dominant failure mode after
  fine-tuning WASB. BlurBall's repo bundles WASB weights, so they interoperate.
- Adopt BlurBall's labeling convention now regardless: **label the ball at the
  center of the blur streak**, not the leading edge.

Skip pure bounding-box detectors (YOLO) as the primary tracker — that was the
v7–v11 dead end. Keep the v10 YOLO checkpoint only as a redundant candidate
miner if convenient.

### L2 — Physics arc layer

> **Prototype result (`prototype_arc_touches.py`, see `PROTOTYPE_L2_FINDINGS.md`).**
> L2 was prototyped on video-506's full 19-touch rally. The DP
> piecewise-parabolic segmentation is **provably correct** — F1 = 1.00 on clean
> ballistic rallies, fully robust to Gaussian jitter (σ = 12 px) and to 25 %
> dropout. It breaks on one thing: **gross outliers / wrong-blob detections**
> (8 % outliers → precision 1.00 → 0.34). A robust pre-filter fixes *isolated*
> outliers but not *sustained* ones (burst lock-on). Two consequences: (a) L2's
> robust fitting must reject whole bad *runs*, not just lone points; (b) the
> detector must be **stateless per-frame** — a continuity/nearest-blob tracker
> produces exactly the burst lock-on L2 cannot survive, which is an extra reason
> to adopt WASB (per-frame heatmap, no continuity state). The physics premise
> holds; touch detection is gated on detector quality, not the algorithm.

In the stabilized frame, treat the detector's candidate centers as a
**multi-model fitting problem**:

- **Sequential RANSAC** parabola fitting (x linear in t, y quadratic in t): fit
  an arc, remove inliers, repeat. Each consensus set is one ballistic arc.
  Alternative: J-linkage / T-linkage to recover all arcs at once without
  pre-specifying the touch count.
- **Factor-graph batch smoothing** (offline — this is a labeling/analysis
  pipeline, not real-time): projectile-motion factors between states,
  measurement factors at confident detections, and a *free velocity-jump
  variable* at each touch node (no physics factor across a touch). Solve as one
  nonlinear least-squares. This interpolates every unlabeled frame with
  physics-consistent centers and per-frame uncertainty.
- Optional: per-arc metric scale via the gravity trick `q = g / a_px` (a_px =
  quadratic pixel-acceleration coefficient). Cross-arc scale consistency is a
  free sanity check.

### L3 — Touch detection + contact classification

- **Touch candidates = arc intersections** (velocity kinks). The literature is
  explicit: trajectory-only kink detection is **high recall, low precision** —
  noise creates spurious kinks. So *confirm* each candidate:
  1. **Residual test:** a real touch makes one global parabola fit blow up while
     two separate arcs both fit with low residual.
  2. **Visual confirmation:** RTMPose foot/limb proximity + motion at that
     frame. (Badminton hit-detection: fusing trajectory + visual cue lifted
     precision from ~59% to ~90%.)
- **Contact classification:** run **RTMPose** (COCO-WholeBody / RTMW variant,
  Apache 2.0, real-time) — the only fast, commercially-licensed pose model that
  outputs real **foot keypoints** (toe + heel, both feet), not just ankles. Use
  it top-down with a lower-body person crop (top-down pose tolerates partial
  POV bodies; MediaPipe's holistic detector does not). At each confirmed touch,
  the nearest foot/limb keypoint to the ball center gives **side** (L/R) and
  **limb** (toe/heel/knee/ankle) → contact type. Train a small classifier on
  features `[ball pos/vel relative to each keypoint, keypoint velocities,
  trajectory curvature]` windowed around the kink.

---

## 3. Data Strategy — Breaking the Labeling Wall

The wall is human labeling time. The fix is **click-and-propagate**, not
hand-typing thousands of frames.

**Honest caveat first (EgoPoints benchmark):** point-tracking accuracy
*collapses* on egocentric footage — models at 64–77% on standard video drop to
36–59% on POV, and re-identification after the object leaves frame is 0–15%.
**No tool gives fully hands-free labels here.** But a human-in-the-loop
click-and-propagate loop is still a realistic **10–30× throughput gain**
(≈5–10× on heavily blurred fast-flight clips).

Loop:

1. **Segment clips at natural breaks** — wherever the ball leaves frame or is
   long-occluded — so each segment has continuous ball visibility. This
   sidesteps the broken re-identification.
2. **Click once per segment** (optionally 2–3 points near the ball — CoTracker3
   exploits cross-track attention).
3. **Propagate with CoTracker3 offline** (bidirectional; sample to ~7 fps if
   GPU-memory bound). Yields dense per-frame centers + visibility flags.
4. **Auto-flag drift** where curvature spikes implausibly, visibility drops, or
   velocity exceeds physical limits.
5. **Human corrects only flagged frames** — re-click, re-propagate forward.
   SAM 2 is a good UI for slow/large-ball segments (mask centroid); it is
   *worse* than CoTracker3 for fast flight.
6. **Approve at the arc level, not the frame level.** Fit the L2 parabola to
   the propagated points; the reviewer approves/rejects the *arc* (one decision)
   instead of 180 frames.
7. **Close the loop:** once a few thousand frames exist, fine-tune WASB, then
   use WASB to pre-label new video, with CoTracker3 only filling gaps.

License note: **CoTracker3 is CC-BY-NC** — fine for internal label generation
(the labels are ours), not for shipping inside the product. SAM 2, TAPIR/
BootsTAPIR, RTMPose, WASB are permissively licensed.

Target first sprint (unchanged from v1, now reachable): ~3–5k dense labeled
frames across train/validation + ≥500 reviewed hard-negative frames/points
(shoes, socks, hands, grass, shadows, colored non-bag patches). Test split
stays untouched; test failures are audit-only, never training data.

---

## 4. Phased Roadmap with Gates

Each phase produces something testable and does not start the next until its
gate passes.

**Phase 0 — Foundations (~1 week).** Commit the 48 uncommitted files; pin
dependency versions; add CI running the existing 95 tests. Determine the camera
(Ray-Ban vs Aria) via `ffprobe`. *Gate:* clean repo, green CI, camera known.

**Phase 1 — Ego-motion compensation (~1–2 weeks).** Build the L0 stabilization
module. *Gate:* on ≥5 clips, a hand-marked static background point stays within
a few px of fixed across the clip after warping; ball masking verified.

**Phase 2 — Labeling bootstrap (~2–3 weeks, then ongoing).** CoTracker3
click-and-propagate pipeline + arc-level review UI (extend the existing
`review_app.py`). *Gate:* ≥3k dense train/val frames + ≥500 hard negatives,
arc-reviewed.

**Phase 3 — Heatmap detector (~2 weeks).** Fine-tune WASB from pretrained
weights. *Gate:* v1 dense-validation gate — visible-frame center RMSE < 12 px,
p95 < 30 px, missing-visible-frame rate < 5%, hard-negative FP rate < 2%.

**Phase 4 — Physics arc layer (~2 weeks).** L2 RANSAC multi-arc + factor-graph
smoothing; L3 touch detection (arc intersection + residual + visual confirm).
*Gate:* touch precision ≥ 90%, recall ≥ 85%; drop/floor ≥ 90% / ≥ 90% on
reviewed held-out data.

**Phase 5 — Contact classification (~1–2 weeks).** RTMPose + kink-frame
classifier. *Gate:* side ≥ 85%, contact type ≥ 85%, ambiguous kept `unknown`.

**Phase 6 — Integration + release eval.** Wire L0–L3 into `hackytrack.py`;
run the full 27-video release evaluation. *Gate:* the v1 system-level release
gates, evaluated end-to-end — not on L1 alone.

---

## 5. Risks & Honest Caveats

- **Camera translation parallax** is the accuracy ceiling of vision-only L0. If
  footage involves walking, no 2D method fully fixes it; the honest alternative
  is monocular SLAM — much heavier. Decide based on actual footage.
- **Egocentric point tracking is unreliable** (EgoPoints). Labeling stays
  human-in-the-loop; budget reviewer time, do not assume automation.
- **Trajectory-kink touch detection is noisy.** The residual test + visual
  confirmation are not optional polish — they are required for the precision
  gate.
- **27 videos is still a narrow distribution.** Even with this pipeline, a
  *universal* footbag detector across all shoes/floors/lighting is ambitious.
  A per-user calibration step (the player clicks their bag once) is a far more
  tractable fallback and footbag players will happily do it. Keep this in
  reserve.
- **WASB pretrained weights may transfer poorly** if footbag appearance/motion
  diverges from tennis/badminton more than expected. Mitigation: BlurBall
  upgrade path; aggressive augmentation (synthetic motion blur, color jitter);
  pseudo-labeling.

---

## 6. Immediate Next Steps

1. **Confirm the camera** (Ray-Ban consumer vs Project Aria) — this changes L0
   materially. One `ffprobe` call answers it.
2. **Phase 0 hygiene** — commit, pin deps, CI. Non-destructive, removes
   data-loss risk today.
3. ~~Prototype L0 on one existing clip.~~ **Done** — `prototype_ego_motion.py`,
   findings in `PROTOTYPE_L0_FINDINGS.md`. The physics premise is confirmed:
   clean ballistic arcs fit parabolas at 13–22 px. The result also refined the
   plan — L0 becomes local rather than a global prerequisite (see the L0
   callout above), and L1 (heatmap detector) is the higher priority.
4. ~~Prototype L2.~~ **Done** — `prototype_arc_touches.py`, findings in
   `PROTOTYPE_L2_FINDINGS.md`. The arc-segmentation premise holds: touch
   detection is exact on clean rallies and is gated only on detector quality
   (specifically, no sustained wrong-blob lock-on).
5. **Next experiment — L1 candidate quality.** Three prototypes (L0, L2, the
   velocity-kink test) now all point at L1 as the critical path. Stand up WASB
   with its pretrained tennis/badminton weights and run it zero-shot on a few
   footbag clips: confirm it surfaces a usable footbag heatmap peak *and* does
   not lock onto wrong objects, before committing to the Phase 2 labeling
   sprint.

---

## References

Tracker / heatmap detection:
- WASB — https://arxiv.org/abs/2311.05237 · https://github.com/nttcom/WASB-SBDT
- BlurBall — https://arxiv.org/abs/2509.18387 · https://github.com/cogsys-tuebingen/blurball
- TrackNetV3 — https://github.com/qaz812345/TrackNetV3
- TrackNetV4 (motion attention) — https://arxiv.org/abs/2409.14543

Physics / trajectory:
- Physics-based ball tracking (basketball) — https://www.sciencedirect.com/science/article/abs/pii/S104732030800117X
- Egocentric event-based ping-pong, gyro compensation (closest analogue) — https://arxiv.org/html/2506.07860
- Badminton hit detection (trajectory + visual fusion) — https://pmc.ncbi.nlm.nih.gov/articles/PMC11244353/
- Factor-graph ball localization/prediction — https://arxiv.org/pdf/2401.17185
- GraviCap (gravity-aware monocular 3D) — https://arxiv.org/abs/2108.08844
- Gravity-as-scale math — https://ar5iv.labs.arxiv.org/html/1909.02211

Ego-motion / stabilization:
- OpenCV MAGSAC++ evaluation — https://opencv.org/evaluating-opencvs-new-ransacs/
- Grundmann L1-optimal stabilization (what NOT to use) — https://research.google.com/pubs/archive/37041.pdf
- Project Aria (IMU access) — https://www.projectaria.com/

Labeling / pose:
- CoTracker3 — https://github.com/facebookresearch/co-tracker · https://cotracker3.github.io/
- EgoPoints (egocentric tracking-collapse benchmark) — https://arxiv.org/pdf/2412.04592
- SAM 2 — https://github.com/facebookresearch/sam2
- RTMPose / RTMW whole-body — https://github.com/open-mmlab/mmpose · https://arxiv.org/pdf/2407.08634
