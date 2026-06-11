# L2/L3 Physics-Layer Prototype — Findings

Prototype: `prototype_arc_touches.py` · clip: `video-506` (full rally, 19
touches) · run: `python3 prototype_arc_touches.py` → `tmp_proto/`
(`arc_touches.png`, `synthetic_conditions.png`, `arc_touch_metrics.json`).

## Goal

Test the second premise of `DETECTOR_REPLACEMENT_PLAN_V2.md`: a footbag rally is
a chain of ballistic arcs, so **touch detection = segmenting the ball track into
piecewise-parabolic pieces**, each breakpoint a candidate touch. Measure touch
precision/recall against video-506's 19 hand-labelled touches (±0.20 s).

## Method

1. Ball track: HSV colour detection (reused from `prototype_ego_motion.py`),
   in raw image coordinates — the L0 finding showed arcs are parabolic without
   stabilisation.
2. **L2 — optimal piecewise fit by dynamic programming.** Partition the track
   into segments; each scored by a quadratic-in-t fit to `y` and a linear-in-t
   fit to `x`; a per-break penalty `lambda` controls sensitivity. Prefix sums
   make each segment cost O(1), so the DP is O(n²).
3. **L3 — touch time** = intersection of the two adjacent fitted parabolas.
4. **Velocity-kink confirmation** — keep only breaks where ball velocity changes
   sharply (intrinsic to the trajectory, no model).
5. **Synthetic control** — generate exact ballistic rallies with known touches,
   inject realistic corruption, and re-run, to separate algorithm error from
   detector error.

## Results

Real clip, video-506 (19 touches, ±0.20 s):

| Stage | precision | recall | F1 |
|---|--:|--:|--:|
| L2 segmentation only          | 0.49 | 0.90 | 0.63 |
| + velocity-kink confirmation  | 0.59 | 0.68 | 0.63 |
| + robust outlier pre-filter   | —    | —    | 0.65 |

Synthetic control (exact ballistic arcs, known touches):

| Condition | precision | recall | F1 |
|---|--:|--:|--:|
| clean (σ=0)                         | 1.00 | 1.00 | **1.00** |
| Gaussian centroid noise σ=12 px     | 1.00 | 1.00 | **1.00** |
| + 25 % dropout (missing frames)     | 1.00 | 1.00 | **1.00** |
| + 8 % i.i.d. gross outliers         | 0.34 | 1.00 | 0.51 |
| &nbsp;&nbsp;…+ robust pre-filter    | 0.77 | 1.00 | **0.87** |
| + sustained outlier bursts (lock-on)| 0.74 | 1.00 | 0.85 |
| &nbsp;&nbsp;…+ robust pre-filter    | 0.74 | 1.00 | 0.85 |

## Findings

**1. The L2 algorithm is correct.** On clean ballistic rallies it recovers every
touch exactly (F1 = 1.00). It is fully robust to Gaussian centroid jitter (1.00
at σ = 12 px) and to 25 % missing detections. The DP piecewise-parabolic
segmentation is sound — the rally *is* a chain of arcs and the breakpoints *are*
the touches.

**2. Segmentation alone is high-recall, low-precision** (real clip: recall 0.90,
precision 0.49) — exactly the pattern the badminton hit-detection literature
reported. It finds the touches but also fires on noise.

**3. Gross outliers are the killer — not jitter, not gaps.** 8 % wrong-blob
detections crash precision from 1.00 to 0.34. This is the single corruption that
breaks L2.

**4. A robust pre-filter fixes *isolated* outliers (F1 0.51 → 0.87) but not
*sustained* ones.** Real wrong-blob detections come in bursts: the
continuity-tracked HSV detector locks onto a wrong red object and stays there
for 8–16 frames. A local-window filter can't catch a burst (the burst's own
neighbours agree with it). On the real clip the filter dropped only 3 of 424
points → F1 barely moved (0.63 → 0.65).

**5. Velocity-kink confirmation does not rescue precision on a noisy track** —
the same noise that creates false breaks corrupts the velocity estimate, so the
kink magnitude no longer discriminates.

## Implications for `DETECTOR_REPLACEMENT_PLAN_V2.md`

This is the **third independent line of evidence** — with the L0 prototype and
the velocity-kink result — all converging on the same conclusion: **L1, the
detector, is the critical path.** The physics layer is already correct; it is
gated entirely on track quality.

Concrete refinements to the plan:

- **L2 needs robust fitting**, as the plan said — but a *local* outlier filter
  is not enough. It must reject *sustained* false runs (RANSAC over whole arcs,
  or trajectory-consensus across the rally).
- **The detector must be stateless per-frame.** A continuity/nearest-blob
  tracker structurally produces burst lock-on — the exact failure L2 cannot
  survive. This is a strong argument for the plan's L1 choice: **WASB is a
  per-frame heatmap detector** with no continuity state, so it cannot lock onto
  a wrong object for a sustained run. Adopt WASB partly *for this reason*.
- **Touch precision target (≥90 %) is reachable** — the synthetic clean result
  proves the algorithm has the headroom. Closing the gap is a detection problem,
  not an algorithm problem.

## Caveats

- One clip. video-506 is a clean outdoor clip; harder clips will have more
  outliers, more stalls.
- The HSV detector is a stand-in for L1. Its burst lock-on is *the* corruption
  diagnosed here — a real L1 detector is needed before L2 can be scored fairly.
- Stalls (ball held on the foot) are non-ballistic and were not modelled in the
  synthetic control; they add breakpoints and need explicit handling in L3.
