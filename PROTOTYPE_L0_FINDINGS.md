# L0 Ego-Motion Prototype — Findings

Prototype: `prototype_ego_motion.py` · clip: `video-506` · window: frames 30–165
(1.0 s – 5.5 s, 136 frames). Run it with `python3 prototype_ego_motion.py`;
outputs land in `tmp_proto/` (`ego_motion_tracks.png`, `prototype_metrics.json`,
`stabilised_average.png`).

## What was tested

The premise of `DETECTOR_REPLACEMENT_PLAN_V2.md`: after removing camera
head-motion, the footbag's image path between touches becomes
piecewise-parabolic (ballistic), so physics-based arc fitting is valid.

Pipeline: HSV colour detection of the coral footbag → frame-to-frame camera
motion (LK optical flow on background features, ball masked out, 4-DOF
similarity transform) → accumulate transforms → warp ball centres into a
reference frame → fit `y = quadratic(t)`, `x = linear(t)` per inter-touch arc
and compare the raw vs stabilised fit residual.

## Results

| Arc (s)      | type            | camera motion | raw fit RMSE | stab fit RMSE |
|--------------|-----------------|--------------:|-------------:|--------------:|
| −1.0 – 1.15  | pre-rally setup |       157 px  |    41 px     |    46 px      |
| **1.15 – 1.96** | **clean flight** |   128 px  |  **22 px**   |   24 px       |
| 1.96 – 3.0   | flight (noisy detections) | 128 px | 147 px |   153 px      |
| 3.0 – 3.78   | toe stall (ball ~static) | 211 px | 23 px |    19 px      |
| **3.78 – 4.36** | **clean flight** |    40 px  |  **13 px**   |   12 px       |

- Ego-motion estimation: **618 RANSAC inliers per frame** (median); total camera
  path 1017 px; net reference-frame displacement ≈ 370 px over 4.5 s — the
  camera moves a lot.
- Ball detection: 76 % of frames (HSV blob; misses on foot-occlusion / blur).

## Three findings

**1. The physics premise holds.** The two genuinely clean single ballistic
flights fit a parabola at **13 px and 22 px residual** — a footbag arc *is*
parabolic in image space to within detector noise.

**2. Ego-motion compensation does NOT improve per-arc fits — and this is the
important correction.** The clean arcs had 40–128 px of real camera motion
during them, yet removing that motion changed the residual by ≤2 px (and
sometimes slightly *worse*). Reason: over a short (~0.5–0.7 s) arc, head motion
is itself smooth — approximately linear/quadratic — so the parabola fit
*already absorbs it*. Explicitly subtracting it adds nothing, and naive
frame-to-frame transform accumulation injects drift (the chain accumulated an
implausible −730 px translation by frame 130; scale error compounds
exponentially).

**3. The dominant residual is ball-detection noise**, not camera motion. The
~13–22 px residual on clean arcs is HSV-blob-centroid wobble on a motion-blurred
ball. Better detection — the L1 WASB heatmap tracker — is the real lever for arc
quality, not better stabilisation.

## Implications for `DETECTOR_REPLACEMENT_PLAN_V2.md`

The V2 plan made L0 (global ego-motion compensation) a hard prerequisite for the
physics layer. The prototype shows that is **too strong**:

- **Per-arc parabola fitting works in raw image coordinates.** Do not gate arc
  fitting on a globally accurate stabiliser. Fit each arc where it lies.
- **Ego-motion is needed only for *global* consistency** — chaining adjacent
  arcs into one frame so touches can be found as arc intersections, and so
  velocities/gravity-scale are physical. That is a local, between-arc need.
- **Naive frame-to-frame accumulation drifts and must not be used globally.**
  Use *local* ego-motion (only between adjacent arcs, re-anchored each touch),
  or a drift-controlled estimator (keyframe re-anchoring / bundle adjustment).
- **Re-prioritise:** L1 (heatmap detector) matters more than L0. The footbag
  arc is already parabolic; what is missing is a precise, dense ball centre.

Net effect: L0 gets *smaller and simpler* (local, not global), and L1 moves up.
The layered architecture is otherwise intact and the physics premise is
confirmed.

## Caveats

- One clip, one rally. video-506 is a clean outdoor clip with a high-contrast
  ball. Confirm on a harder clip (indoor, cluttered, lower contrast).
- The HSV detector is per-clip and noisy; the 147 px arc is a detection failure,
  not a physics failure. A real detector (L1) is needed before the physics layer
  can be evaluated properly.
- `tmp_proto/frames/` holds throwaway extracted frames and can be deleted.
