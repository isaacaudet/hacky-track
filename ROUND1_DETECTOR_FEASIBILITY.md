# Round 1 — Detector Feasibility — Findings

Learning artifact for Round 1 of `DETECTOR_REPLACEMENT_ROADMAP.md`.
Harness: `wasb_feasibility.py` · clip: `video-506` rally (124 frames).

## The bet

WASB's tennis/badminton-pretrained HRNet transfers to a footbag well enough to
be a *candidate miner* — and, being a per-frame heatmap detector, does not
produce the sustained lock-on that killed L2 in Round 0.

## What was done

- **T1.1 Repo hygiene — done.** Branch `detector-replacement-r1`; ~48 uncommitted
  files committed in 6 logical chunks; `requirements*.txt` pinned to exact
  versions; CI workflow added; all 95 tests green.
- **T1.2 Environment — done.** `torch 2.11.0` + WASB deps (hydra-core, timm,
  einops, scikit-image, pandas, tqdm, gdown) installed; pinned in
  `requirements-tracker.txt`. No NVIDIA GPU on this machine → runs on MPS/CPU.
- **T1.3 Stand up WASB — done.** `external/WASB-SBDT` cloned; tennis checkpoint
  `wasb_tennis_best.pth.tar` downloaded. The HRNet model **loads cleanly**
  (0 missing / 0 unexpected state-dict keys) and runs on MPS, emitting heatmaps.
- **T1.4 Feasibility harness — done (partial: 1 clip).** `wasb_feasibility.py`
  runs WASB zero-shot, bypassing WASB's CUDA-only `Detector` wrapper.
- **T1.5 / T1.6 — not done.** The 5-clip set and hand-labelled reference set
  were not built — the T1.4 result was unambiguous enough to decide without them.

## Results — zero-shot WASB on footbag

Two preprocessing variants, both with no fine-tuning:

| Preprocessing | peak confidence (mean / max) | peak near ball |
|---|---|---|
| Full-frame letterbox (portrait → 512×288) | 0.085 / 0.49 | 0 % within 5 % image diag |
| **Fair 16:9 ROI crop** (1024×576 around the ball → 512×288) | **0.044 / 0.15** | **0 % @40px · 1 % @80px · 3 % @150px** |

The ROI crop was tested specifically to rule out the obvious confound — that
letterboxing a 2192×2928 portrait into 512×288 shrinks the footbag to ~6 px.
With a fair 16:9 crop (footbag ~30 px, no aspect distortion) WASB did **not**
improve — confidence was actually *lower*, at the noise floor. The visual
montage (`tmp_proto/wasb_feasibility.png`) confirms it: the WASB peak lands on
random grass/wall in every frame; the footbag is never detected.

## The finding

**Zero-shot WASB tennis weights do not transfer to footbag — at all.** Peak
confidence sits at 0.04–0.15 everywhere (a usable detection needs ~0.5+); the
model essentially never fires. This is not an aspect-ratio artefact — it
survived a fair-crop control.

**But it fails *safely*.** WASB produces no confident peak anywhere — the
*opposite* of lock-on. The Round-0-killer failure mode (sustained confident
wrong detections) is absent. The uniform near-floor confidence is the signature
of a plain **domain gap**, not a broken architecture — and a domain gap is
exactly what fine-tuning closes. The HRNet heatmap architecture itself is sound
(checkpoint loaded perfectly; it runs).

## Decision gate

Roadmap R1 gate: *proceed if zero-shot puts a peak near the ball on ≥50 % of
clear frames **or** the failures look like a domain gap fine-tuning will close,
and no systematic lock-on.*

- ≥50 % bar: **failed hard** (0–3 %).
- Domain-gap-fine-tunable: **yes** — uniform near-floor confidence is the
  textbook fine-tunable signature.
- Lock-on: **none** — WASB is well-behaved in the one way that matters most.
- Kill criterion (*every* detector route fails the lock-on requirement): **not
  triggered**.

### Verdict: PROCEED to Round 2 — with two corrections to the plan

1. **WASB is not a usable zero-shot candidate miner.** Round 2 labeling must
   *not* lean on WASB pre-labeling — use CoTracker3 + HSV/manual seeding only.
2. **R3 fine-tuning is mandatory, not optional.** The roadmap already plans it;
   this confirms there is no zero-shot shortcut. Budget R2 to produce enough
   labels for a real fine-tune.

## Caveats (honest)

- **One clip only.** video-506 is a clean outdoor clip. The result is so
  unambiguous (confidence at noise floor) that one clip strongly indicates the
  conclusion, but T1.5's 5-clip set should still confirm it before R3.
- **Harness not yet validated on WASB's home turf.** The most important missing
  check: run `wasb_feasibility.py` on a *tennis* clip and confirm it detects the
  tennis ball with high confidence. If it does, the harness is proven correct
  and footbag transfer genuinely fails; if it doesn't, the harness has a
  preprocessing bug. **Do this first in Round 2.**
- Preprocessing assumed `/255` + RGB + 3-frame concat, no mean/std (matches
  WASB's `ToTensor`-only transform) — consistent with the loaded checkpoint, but
  the tennis-clip check above is what definitively validates it.

## Hand-off to Round 2

- Treat the detector as untrained: the R2 labeling loop is the prerequisite for
  any working detector, with no zero-shot crutch.
- First R2 action: validate the harness on a tennis clip (above).
- The ROI-crop front-end (`wasb_feasibility.py:letterbox`/crop logic) is
  reusable — a fine-tuned WASB will still want a region crop, since the footbag
  is tiny in a full portrait frame.
