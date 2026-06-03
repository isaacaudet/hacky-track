# Open-Vocabulary Detector Findings

Date: 2026-05-22

## Verdict

OWLv2 should become the next L1 detector candidate. It is the first detector tested in
this project that simultaneously gives high visible-frame localization and zero
no-target false positives on the dense-label corpus without custom training.

This does not make the tracker solved by itself. At the strict operating point
(`threshold=0.2`), OWLv2 is below the 95% per-frame visible-pass target, but it is close
enough that the remaining gap is now a temporal/L2 integration problem rather than a
color-calibration or hand-labeling problem.

## Evaluation Setup

- Script: `oracle_detector_spike.py`
- Model: `google/owlv2-base-patch16-ensemble`
- Prompts: `a footbag`, `a hacky sack`, `a small ball`, `a small round bean bag`, `a ball`
- Labels: `runs/release-27-public/dense_trajectory_review_v2_with_sources/dense_trajectory_labels.reviewed.jsonl`
- Frames: all `744` reviewed target/no-target rows
- Positives: `629`
- No-target frames: `115`
- Sealed audit/test positives: `62`
- No training. No dense labels used by the model.

Outputs:

- `runs/release-27-public/oracle_owlv2_full_v1/report.md`
- `runs/release-27-public/oracle_owlv2_full_v1/summary.json`
- `runs/release-27-public/oracle_owlv2_full_v1/detections.jsonl`
- `runs/release-27-public/oracle_owlv2_full_v1/qa/top1_failures_thr0.2_tol12.jpg`
- `runs/release-27-public/oracle_owlv2_full_v1/qa/top1_fail_oracle_pass_thr0.2_tol12.jpg`

## Headline Numbers

At `threshold=0.2`:

| slice | tolerance | visible pass | oracle pass | fire rate | mean err | p95 err | no-target FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 12px | 0.895 | 0.898 | 0.951 | 8.2px | 13.4px | 0.000 |
| all | 15px | 0.921 | 0.924 | 0.951 | 8.2px | 13.4px | 0.000 |
| validation | 12px | 0.871 | 0.871 | 0.940 | 5.9px | 14.2px | 0.000 |
| validation | 15px | 0.897 | 0.897 | 0.940 | 5.9px | 14.2px | 0.000 |
| sealed audit/test | 12px | 0.919 | 0.919 | 0.984 | 3.8px | 13.6px | n/a |
| sealed audit/test | 15px | 0.935 | 0.935 | 0.984 | 3.8px | 13.6px | n/a |

This beats every existing custom detector on the validation slice by a large margin.
The previous best visible-frame baseline was `v11_greedy` at `0.474` visible pass and
`0.169` no-target false-positive rate. OWLv2 is roughly `0.897` visible pass at 15px
with `0.000` no-target false positives.

## Threshold Tradeoff

On all frames:

| threshold | 12px visible | 15px visible | no-target FP |
| ---: | ---: | ---: | ---: |
| 0.05 | 0.922 | 0.948 | 0.261 |
| 0.10 | 0.921 | 0.946 | 0.061 |
| 0.20 | 0.895 | 0.921 | 0.000 |
| 0.30 | 0.860 | 0.886 | 0.000 |

`0.2` is the clean operating point: it sacrifices some recall but eliminates the
lock-on/no-target failure mode. `0.1` is tempting for recall, but a `6.1%` no-target FP
rate means it needs a temporal or no-target gate before it can be trusted.

## Failure Anatomy

At `threshold=0.2`:

- 12px tolerance: `563/629` pass, `33` localized-but-outside-tolerance, `31` missing,
  `2` top-1-wrong/oracle-right.
- 15px tolerance: `579/629` pass, `17` localized-but-outside-tolerance, `31` missing,
  `2` top-1-wrong/oracle-right.

The important split:

- The dominant remaining issue is not hallucination. No-target FP is zero.
- Many 12px misses are visually on the ball but center-offset beyond a very strict
  tolerance, especially close hand-held frames.
- True top-1 ambiguity is rare. There are `70` multi-detection positive frames, but only
  `2` frames where top-1 is wrong while another detection is correct.
- Missing detections are clip-concentrated, especially `video-245_singular_display 2.MOV`
  and `video-486_singular_display.MOV`.

That means the next leverage is temporal integration: use OWLv2 as high-precision
anchors, then fill/drop/gate with L2 trajectory physics instead of trying to solve
everything per frame.

## Strategic Impact

The previous roadmap assumed L1 required a fine-tuned WASB detector trained from dense
human labels. This spike changes that assumption.

Recommended change:

1. Pause human dense-labeling as the critical path.
2. Promote OWLv2 to the next L1 candidate.
3. Wire OWLv2 detections into the existing detector-track/evaluate path.
4. Evaluate OWLv2 + L2 trajectory fill end-to-end on the dense labels and rally gates.
5. Use OWLv2 as a teacher/autolabeler later if speed or deployment cost requires a
   smaller distilled detector.

The existing 744 human labels were still essential: they turned this from a visual hunch
into a measured result with sealed-audit confirmation.

## Next Engineering Step

Build an OWLv2 detector-track adapter:

- Input: video or dense-review manifest.
- Output: the same detector track schema consumed by `evaluate_dense_trajectory.py` and
  downstream QA tooling.
- Use `threshold=0.2` as the safe baseline.
- Add optional `threshold=0.1 + temporal/no-target gate` as an experimental setting.
- Score:
  - raw OWLv2 per-frame,
  - OWLv2 + L2 interpolation/fill,
  - existing v10/v11 baselines.

Success criterion for the next round: recover the missing/localized-outside-tolerance
frames without reintroducing no-target lock-on.

## Follow-up: Conservative L2 Adapter Result

Implemented:

- Script: `owlv2_l2_eval.py`
- Tests: `tests/test_owlv2_l2_eval.py`
- Output: `runs/release-27-public/owlv2_l2_eval_v1/report.md`

The adapter writes normal `predictions.jsonl` rows and evaluates them with
`evaluate_dense_trajectory.py`. The L2 pass is intentionally conservative:

- raw OWLv2 anchor threshold: `0.2`
- fill candidate floor: `0.1`
- support radius: `25px`
- max gap: `3` frames / `0.18s`
- exact-frame dense eval, no 40ms time fallback

Result:

| track | tolerance | all visible pass | validation pass | sealed audit/test pass | no-target FP | longest no-target run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw OWLv2 | 12px | 0.895 | 0.871 | 0.919 | 0.000 | 0 |
| raw OWLv2 | 15px | 0.921 | 0.897 | 0.935 | 0.000 | 0 |
| OWLv2 + conservative L2 | 12px | 0.911 | 0.862 | 0.935 | 0.000 | 0 |
| OWLv2 + conservative L2 | 15px | 0.936 | 0.888 | 0.952 | 0.000 | 0 |

The conservative L2 pass is safe but not sufficient. It improves all-frame 15px pass
from `0.921` to `0.936` and sealed audit/test from `0.935` to `0.952`, while keeping
no-target FP at `0`. However, validation moves slightly down (`0.897` to `0.888`), so
this is not a clean promotion rule yet.

An unsafe variant was also tested:

- OWLv2 `threshold=0.1` plus trajectory cleaning reaches about `0.952` visible pass at
  15px, but creates `7-10` no-target false positives and a sustained no-target run up
  to `8` frames.

Conclusion: keep `threshold=0.2` as the safe baseline. The current L2 fill is useful
evidence, but the next improvement should target the miss-heavy clips/prompts or a
stronger OWLv2 model before relaxing the threshold.
