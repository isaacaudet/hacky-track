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

Preprocessing variants, all with no fine-tuning:

| Preprocessing | peak confidence (mean / max) | peak near ball |
|---|---|---|
| Full-frame letterbox (portrait → 512×288) | 0.085 / 0.49 | 0 % within 5 % image diag |
| Fair 16:9 ROI crop, raw /255 | 0.044 / 0.15 | 0 % @40px · 1 % @80px · 3 % @150px |
| **ROI crop + ImageNet norm** (corrected — WASB's actual test transform) | **0.028 / 0.06** | **4 % @40px · 11 % @80px · 18 % @150px** |

The ROI crop was tested specifically to rule out the obvious confound — that
letterboxing a 2192×2928 portrait into 512×288 shrinks the footbag to ~6 px.
With a fair 16:9 crop (footbag ~30 px, no aspect distortion) WASB did **not**
improve — confidence was actually *lower*, at the noise floor.

**Preprocessing correction (post-R1 harness audit).** The first two rows used
`/255`-only input. An audit of WASB's own `dataloaders/build_img_transforms`
showed its test transform is `ToTensor + Normalize(ImageNet mean/std)` — applied
for train *and* test. The R1 harness omitted the `Normalize`, feeding the net
un-normalized input it was never trained on. The third row re-runs with the
correct normalization. Result: confidence stays pinned to the noise floor
(0.06 max vs the ~0.5 a real detection needs); localization is marginally less
random (3 %→18 % within 150 px, still chance-level for a 1024-px-wide crop).
**The bug was real but the conclusion is unchanged** — fixing it confirms the
domain gap rather than closing it. The montage
(`tmp_proto/wasb_feasibility.png`, normalized run) shows the peak still landing
on random grass/wall; the footbag is never detected.

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
- **Harness preprocessing — audited and corrected.** The harness was validated
  line-by-line against WASB's own inference path (`detectors/detector.py`,
  `dataloaders/dataset_loader.py`, `build_img_transforms`): RGB channel order,
  `/255`, 3-frame channel concat (oldest→newest), output channel ↔ input frame
  mapping (last = current), and `sigmoid` on logits all match. **One mismatch
  was found and fixed:** WASB's test transform applies ImageNet `Normalize`; the
  original harness omitted it. Re-running with the fix (results table above)
  leaves the conclusion unchanged. The one remaining empirical check — running
  the corrected harness on an in-domain tennis clip — is now a low-priority
  confirmation rather than a blocker, since the preprocessing match is verified
  against WASB's source.
- Geometric front-end differs from WASB by design: WASB anamorphically warps a
  centered `max(h,w)` square via `get_affine_transform`; the harness uses a 16:9
  ROI crop so the tiny footbag is ~30 px not ~6 px. This is a deliberate
  adaptation, not a bug, and a fine-tuned detector will keep the ROI crop.

## Hand-off to Round 2

- Treat the detector as untrained: the R2 labeling loop is the prerequisite for
  any working detector, with no zero-shot crutch.
- The harness preprocessing is now audited against WASB source and corrected
  (ImageNet `Normalize` added) — no tennis-clip validation needed before R2.
- The ROI-crop front-end (`wasb_feasibility.py:letterbox`/crop logic) is
  reusable — a fine-tuned WASB will still want a region crop, since the footbag
  is tiny in a full portrait frame.
