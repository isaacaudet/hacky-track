# Footbag Tracker — Longform Goal & Execution Roadmap

> Execution sequencing for `DETECTOR_REPLACEMENT_PLAN_V2.md`. The V2 plan is the
> *architecture* (what to build). This is the *roadmap* (in what order, what to
> learn between steps, and when to stop). Read the V2 plan first.

---

## The Goal

**Replace heuristic footbag tracking with a trained, explainable detector
pipeline that a new user can run on their own consumer Ray-Ban Meta POV clip —
producing an audited rally with touches, ball track, and contact labels that
meet the `RELEASE_GOAL.md` metric gates — without hand-editing JSON.**

The goal is *done* when, on held-out reviewed footage, all of the following
hold (these are the binding success criteria — not aspirations):

| Metric | Target |
|---|---|
| Ball-track event-center pass rate | ≥ 95 % |
| Touch precision / recall | ≥ 90 % / ≥ 85 % |
| Duplicate touches | < 5 % of accepted touches |
| Drop / floor-reset precision / recall | ≥ 90 % / ≥ 90 % |
| Side classification accuracy | ≥ 85 % (ambiguous kept `unknown`) |
| Contact-type accuracy | ≥ 85 % |
| Detector sustained false-positive lock-on | none (hard requirement) |
| Pipeline | runs end-to-end via one `hackytrack.py` command |

Anything not met at the end is **explicitly waived in writing** with a reason —
never silently dropped.

---

## How This Roadmap Works

The project's failure mode (see `RELEASE_STATUS.md`: v7–v11) was **building
tooling faster than it learned anything** — ten review-batch scripts, no metric
movement. This roadmap is structured to prevent that.

Work proceeds in **rounds**. Every round has the same shape:

1. **Strategize round** — before any code: settle the open questions, state the
   bet (the hypothesis the round tests), decide the approach. Output: a short
   decision note.
2. **Tasks** — the concrete build/measure work. Bounded.
3. **Learning round** — after the work: measure against the gate, and write a
   **findings document** (like `PROTOTYPE_L0_FINDINGS.md` /
   `PROTOTYPE_L2_FINDINGS.md`). A round is not complete without it.
4. **Decision gate** — proceed / pivot / kill, on quantitative criteria stated
   *before* the work, not after.

**Three standing rules, enforced every round:**

- **No round may build tooling beyond what its own tasks require.** If a tool
  would help a *future* round, note it and build it *then*.
- **Every round produces exactly one learning artifact** — a findings doc with
  numbers. No artifact → round not done.
- **A round may end in "pivot" or "kill."** Proceeding is not the default; it is
  earned by passing the gate. Off-ramps are listed and are real.

Effort estimates are in engineer-weeks and assume one focused developer with
occasional GPU access. They are planning figures, not commitments.

---

## Where We Are Now — Round 0 (complete)

Round 0 was research + two de-risking prototypes. Done:

- **Research** — 4-way literature review → `DETECTOR_REPLACEMENT_PLAN_V2.md`.
  Key correction: do not build a temporal heatmap model from scratch; fine-tune
  **WASB** (pretrained, MIT). Architecture is layered: L0 ego-motion → L1
  heatmap detector → L2 physics arcs → L3 touches/pose.
- **L0 prototype** (`prototype_ego_motion.py`, `PROTOTYPE_L0_FINDINGS.md`) —
  ego-motion estimation is robust; but short ballistic arcs are already
  parabolic in *raw* coordinates (camera motion absorbed by the fit). L0 shrinks
  to a *local* role.
- **L2 prototype** (`prototype_arc_touches.py`, `PROTOTYPE_L2_FINDINGS.md`) —
  the arc-segmentation touch detector is provably correct (F1 = 1.00 on clean
  rallies) and gated only on detector quality; **sustained wrong-blob lock-on**
  is the one corruption it cannot survive.

**Round 0's verdict, triangulated three ways:** L1 — a real, stateless,
per-frame detector — is the critical path. Everything downstream already works
given a clean track. Rounds 1–3 therefore target L1; Rounds 4–6 assemble and
ship around it.

---

## Round 1 — Detector Feasibility

**Effort:** ~1.5 weeks  ·  **Entry condition:** Round 0 complete.

**The bet:** WASB's tennis/badminton-pretrained weights transfer to a footbag
well enough to be a *candidate miner* — and, being a per-frame heatmap detector,
it does not produce the sustained lock-on that killed L2 in Round 0.

### Strategize round

Settle before tasking:
- GPU access — local CUDA, a cloud GPU, or Colab? WASB inference and later
  fine-tuning need it. Decide and provision now.
- Which 5 clips form the feasibility set? Pick a deliberate spread: 2 easy
  (clear ball, mild camera), 2 hard (blur, clutter, indoor), 1 with the ball
  leaving frame. Variety matters more than count.
- Measuring stick: reuse existing QA ball centers, or hand-label fresh? Decide
  the ~150–200-frame reference set source.

### Tasks

- **T1.1 — Repo hygiene (do first, blocks nothing else).** Branch; commit the
  ~48 uncommitted files in logical chunks; pin `requirements.txt` to exact
  versions; add a CI workflow running the existing 95 tests. Non-destructive,
  removes data-loss risk.
- **T1.2 — Environment.** `venv`; install `torch` + WASB dependencies; confirm
  GPU path. Add `requirements-tracker.txt` (kept separate from the core deps).
- **T1.3 — Stand up WASB.** Clone `nttcom/WASB-SBDT`; download the model zoo;
  run the tennis/badminton checkpoint on one footbag clip to confirm it executes
  and emits heatmaps.
- **T1.4 — Feasibility harness.** Script that runs WASB zero-shot over the 5
  clips and records, per frame: heatmap peak (x, y), peak confidence, and the
  top-2 peaks (to detect competing responses).
- **T1.5 — Reference labels.** Produce the ~150–200-frame sparse ground-truth
  ball-center set across the 5 clips.
- **T1.6 — Measure.** Per-frame center error vs labels; recall of "a peak within
  R px of the true ball" on clearly-visible-ball frames; **sustained-lock-on
  rate** (runs of ≥6 frames where the peak sits on a fixed wrong location);
  confidence calibration (does low confidence correlate with error?).

### Learning round

Write `ROUND1_DETECTOR_FEASIBILITY.md`. Answer: Is zero-shot WASB a usable
candidate miner? Where does it fail — blur, small scale, clutter? Does it lock
on? Is the gap fine-tunable or structural?

### Decision gate

- **Proceed to Round 2** if: (a) zero-shot puts a peak near the visible ball on
  ≥ 50 % of clear frames *or* the failures look like a domain gap fine-tuning
  will close, **and** (b) no systematic sustained lock-on.
- **Pivot** if zero-shot is weak but promising: jump straight to a small
  fine-tune (a few hundred frames) before judging — fold into Round 2.
- **Pivot hard** if WASB is structurally wrong for footbag: try **BlurBall**
  weights; or drop to a **per-user appearance model** (user clicks their bag
  once → per-clip template/color detector). The per-user route is the honest
  fallback and footbag players will accept one click.
- **Kill** the trained-detector goal only if *every* detector route fails the
  lock-on requirement — then the release descopes to "heuristic + human review,"
  labelled as such.

---

## Round 2 — Labeling Pipeline & First Dense Dataset

**Effort:** ~3 weeks  ·  **Entry condition:** Round 1 proceed/pivot gate passed.

**The bet:** A click-once-then-propagate loop (CoTracker3) plus arc-level review
turns dense labeling from infeasible into a 10–30× throughput win, enough to
build a real training set from the 27 clips.

### Strategize round

- From Round 1: how much can the bootstrap detector pre-label, reducing clicks?
- Dataset size target: set a concrete first-sprint number (the V2 plan suggests
  ~3–5 k dense frames + ≥ 500 hard negatives) and the train/val/test split —
  **the test split is frozen now and never relabeled.**
- Review UI: extend `review_app.py`, or a thin standalone? Decide; do not
  rebuild the review system.
- Honest constraint (EgoPoints): point trackers collapse on egocentric footage;
  the loop *must* be human-in-the-loop. Budget reviewer time accordingly.

### Tasks

- **T2.1 — Clip segmentation.** Split each video at ball-exit / long-occlusion
  into continuous-visibility segments (sidesteps the broken re-identification).
- **T2.2 — CoTracker3 propagation.** Click once per segment → dense per-frame
  center + visibility flags.
- **T2.3 — Drift auto-flag.** Flag frames where curvature, velocity, or
  visibility implies the track has drifted — for human attention.
- **T2.4 — Arc-level review UI.** Reviewer approves/rejects/corrects a *fitted
  arc* (one decision) instead of 180 frames. Re-click + re-propagate on reject.
- **T2.5 — Label the first dataset.** Reach the strategize-round target of clean
  dense train/val frames + hard negatives (shoes, socks, hands, grass, shadows,
  colored non-bag patches, and the exact Round 1 failure look-alikes).
- **T2.6 — Dataset export** in WASB/heatmap training format, with a manifest
  recording split provenance and counts.

### Learning round

Write `ROUND2_LABELING.md`: measured throughput (frames/reviewer-hour), label
quality from a spot-check audit, final dataset size and split, and a realistic
projection of how large the set can grow.

### Decision gate

- **Proceed** if: ≥ 1.5 k clean dense train/val frames + ≥ 500 hard negatives
  exist, and throughput is proven sustainable.
- **Pivot** if throughput is too low: narrow scope — fewer videos, or commit to
  the per-user model (which needs far less data).
- This round explicitly **does not train anything.** Labeling only.

---

## Round 3 — Train L1 & Close the Loop

**Effort:** ~2.5 weeks  ·  **Entry condition:** Round 2 dataset exists.

**The bet:** Fine-tuning WASB on the Round 2 dataset produces a detector that
passes the dense-validation gate and — critically — does not lock on.

### Strategize round

- Training recipe: frozen-backbone vs full fine-tune; learning rate; epochs.
- Augmentation: synthetic motion blur, color jitter, occlusion — decide the set
  up front (low-shot detection is augmentation-sensitive).
- Self-training: plan one pseudo-label → human-correct → retrain iteration.

### Tasks

- **T3.1 — Fine-tune WASB** from pretrained weights; save a versioned checkpoint
  + training manifest.
- **T3.2 — Dense-validation gate.** Evaluate: visible-frame center RMSE < 12 px,
  p95 < 30 px, missing-visible-frame rate < 5 %, hard-negative FP rate < 2 %.
  **Separately and explicitly: test for sustained lock-on** on long clips.
- **T3.3 — Self-training iteration.** Pseudo-label all clips with the fine-tuned
  model; human-review-correct the high-value disagreements; retrain.
- **T3.4 — Blur handling.** If blur is the dominant residual failure, switch the
  detector head to **BlurBall** (same lineage, bundles WASB weights).

### Learning round

Write `ROUND3_DETECTOR_TRAINING.md`: gate results, the lock-on test outcome,
remaining failure modes, and whether more labels (back to Round 2) are needed.

### Decision gate

- **Proceed** if the dense-validation gate passes **and** the lock-on test is
  clean.
- **Loop back to Round 2** if it fails on data volume — collect more, retrain.
- **Pivot** to BlurBall or per-user calibration if it fails structurally.

---

## Round 4 — Full Pipeline Integration (L0 + L2 + L3)

**Effort:** ~3 weeks  ·  **Entry condition:** Round 3 detector passes its gate.

**The bet:** With a clean L1 track, the Round-0 prototypes for L0 and L2 — made
production-grade — chain into an end-to-end tracker that hits the touch and
ball-center gates.

### Strategize round

- Re-confirm the L0 finding: ego-motion is needed only *locally* (chaining
  adjacent arcs), with drift control (keyframe re-anchoring). Decide the design.
- L2 robust fitting: the L2 prototype showed a local outlier filter is not
  enough — design whole-arc RANSAC / trajectory-consensus fitting that rejects
  sustained bad runs. (Less critical now that L1 is stateless, but keep it.)
- Stall handling in L3 — stalls are non-ballistic; decide how L3 represents them.

### Tasks

- **T4.1 — L0 module.** Productionize `prototype_ego_motion.py` into a local,
  drift-controlled ego-motion module.
- **T4.2 — L2 module.** Productionize `prototype_arc_touches.py`: whole-arc
  RANSAC robust fitting, factor-graph smoothing, stall handling.
- **T4.3 — L3 module.** Touch detection = arc intersection + residual test;
  drop/floor-reset detection.
- **T4.4 — Integrate.** Wire L1 → L2 → L3 into one tracker; expose it in
  `hackytrack.py` as a selectable ball-track backend alongside the heuristic.
- **T4.5 — Evaluate** touch precision/recall, duplicate rate, drop precision/
  recall, and ball-center pass rate on held-out reviewed videos.

### Learning round

Write `ROUND4_PIPELINE.md`: end-to-end metrics vs every gate, with per-failure
breakdown.

### Decision gate

- **Proceed** if: ball-center ≥ 95 %, touch P ≥ 90 % / R ≥ 85 %, duplicates
  < 5 %, drop ≥ 90 % / 90 %.
- **Loop back** to the responsible round for any missed gate (detector → R3;
  labels → R2).

---

## Round 5 — Contact Classification

**Effort:** ~1.5 weeks  ·  **Entry condition:** Round 4 pipeline passes.

**The bet:** Off-the-shelf pose (RTMPose) + the L2 trajectory kink gives side
and contact type geometrically, with little training data.

### Tasks

- **T5.1 — RTMPose integration** (COCO-WholeBody / RTMW, lower-body crop,
  top-down — tolerates partial POV bodies).
- **T5.2 — Contact classifier.** At each confirmed touch, features from
  ball-vs-foot-keypoint geometry + trajectory curvature → small classifier for
  side (L/R) and contact type.
- **T5.3 — Evaluate** side and type accuracy; ambiguous cases kept `unknown`;
  knee stays candidate-only until ≥ 20 reviewed examples exist.

### Learning round

Write `ROUND5_CONTACT.md`: side/type accuracy, `unknown` rate, failure cases.

### Decision gate

- **Proceed** if side ≥ 85 % and contact type ≥ 85 %.
- **Waive** (documented) if a sub-target is unmet but the rest of release holds
  — contact labels degrade gracefully to `unknown`.

---

## Round 6 — Release Evaluation & Decision

**Effort:** ~1.5 weeks  ·  **Entry condition:** Rounds 4–5 gates met.

### Tasks

- **T6.1 — Full 27-video release evaluation.** Model centers promoted only under
  the explicit validation gates; clean train/val/test separation.
- **T6.2 — Release-candidate report** — artifact paths, every metric vs gate,
  HUD verification with the model-backed track, remaining limitations.
- **T6.3 — Release-candidate changelog and tag.**

### Learning round + final decision

Write `ROUND6_RELEASE.md`. Then one of three honest outcomes:

- **Ship** — all gates met (or unmet ones explicitly waived with reasons).
- **Descope & ship** — trained detector promoted where it passes; heuristic +
  human review retained where it does not; the product is labelled accordingly.
- **Iterate** — a specific gate fails; loop to the round that owns it.

---

## Standing Off-Ramps (honest, available at any round)

- **Per-user calibration.** 27 videos is a narrow distribution. If a *universal*
  detector keeps failing, a one-click per-user appearance model is a far more
  tractable product and an acceptable release. Keep it in reach from Round 1 on.
- **Descoped release.** The heuristic + human-review pipeline already produces a
  working, explainable HUD. Shipping *that*, honestly labelled, is a valid
  outcome if the trained detector does not land — do not hold the release
  hostage to unsolved research.
- **Stop and reassess** if two consecutive rounds miss their gate — that is a
  signal the architecture is wrong, not that the next round needs more effort.

---

## Timeline & Artifact Summary

| Round | Focus | Effort | Required learning artifact |
|---|---|---|---|
| 0 | Research + L0/L2 prototypes | done | `PROTOTYPE_L0/L2_FINDINGS.md` ✓ |
| 1 | Detector feasibility | ~1.5 wk | `ROUND1_DETECTOR_FEASIBILITY.md` |
| 2 | Labeling + first dataset | ~3 wk | `ROUND2_LABELING.md` |
| 3 | Train L1, close the loop | ~2.5 wk | `ROUND3_DETECTOR_TRAINING.md` |
| 4 | Pipeline integration | ~3 wk | `ROUND4_PIPELINE.md` |
| 5 | Contact classification | ~1.5 wk | `ROUND5_CONTACT.md` |
| 6 | Release evaluation | ~1.5 wk | `ROUND6_RELEASE.md` |

Total: **~13 engineer-weeks** (~3–3.5 months) from here, *if* every gate passes
first time. Pivots and loop-backs extend it — and that is expected, not failure.
The roadmap is designed so a pivot costs one round, not the whole project.
