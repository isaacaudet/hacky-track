# Audio Fusion Findings

## Current Verdict

Audio fusion is now implemented in the intended direction:

- Loose audio is high-recall and low-precision.
- Trajectory evidence is the precision filter.
- The video-50 fused result is no longer vacuous: fused beats both trajectory-only and loose-audio-only.

This still does not certify rollout. Video-506 needs a completed OWLv2 stride-1 run, or an equivalent cached OWLv2 track, before the cross-clip claim is trustworthy.

## Video-50, OWLv2, Stride 1, Threshold 0.2

Output: `runs/release-27-public/owlv2_event_eval_video50_loose_audio_v2/`

| method | precision | recall | f1 | tp/gt | fp |
| --- | ---: | ---: | ---: | ---: | ---: |
| trajectory | 0.818 | 0.750 | 0.783 | 9/12 | 2 |
| audio_only | 0.545 | 1.000 | 0.706 | 12/12 | 10 |
| fused | 1.000 | 1.000 | 1.000 | 12/12 | 0 |

Rally 2 improved from trajectory-only `0.714 / 0.625 / 0.667` to fused `1.000 / 1.000 / 1.000`.

## Video-506 Sanity Checks

`data/video-506_singular_display.events.json` has `annotation_method: user_corrected_hud_ground_truth`, so it is the better non-circular test target.

Frozen loose-audio params on video-506:

| method | precision | recall | f1 | tp/gt | fp |
| --- | ---: | ---: | ---: | ---: | ---: |
| audio_only | 0.543 | 1.000 | 0.704 | 19/19 | 16 |

This confirms the desired audio shape cross-clip: audio catches all reviewed touches but brings many in-window non-touch onsets.

The OWLv2 stride-1 video-506 run did not complete in a reasonable local window and was stopped before producing a detection cache. Two diagnostic-only trajectory probes were run using older tracks:

| trajectory source | trajectory P/R/F1 | audio_only P/R/F1 | fused P/R/F1 |
| --- | ---: | ---: | ---: |
| v10 calibrated track | 0.909 / 0.526 / 0.667 | 0.543 / 1.000 / 0.704 | 0.833 / 0.526 / 0.645 |
| patch temporal track | 0.714 / 0.263 / 0.385 | 0.543 / 1.000 / 0.704 | 1.000 / 0.263 / 0.417 |

These probes do not evaluate the intended OWLv2 pipeline. They show that when the trajectory source misses too many contacts, fusion stays recall-limited by trajectory corroboration.

## Implementation Notes

The fused decision now accepts an audio onset only when nearby L2 breakpoints have enough support and one of:

- positive vertical-velocity discontinuity,
- sparse local samples where breakpoint velocity cannot be estimated,
- high breakpoint consensus, kept low-priority so temporal NMS can suppress nearby arc-apex/footstep candidates.

The default audio extractor is now intentionally loose: `delta=0.07`, `wait_sec=0.0`.

## Next Required Check

Complete a video-506 OWLv2 stride-1 detection cache, then rerun:

```bash
python3 owlv2_event_eval.py \
  --events data/video-506_singular_display.events.json \
  --frame-stride 1 \
  --out-dir runs/release-27-public/owlv2_event_eval_video506_loose_audio_v1
```

That is the first honest cross-clip certification point for the OWLv2 audio-fused event pipeline.
