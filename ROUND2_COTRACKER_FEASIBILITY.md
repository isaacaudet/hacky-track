# Round 2 CoTracker Feasibility Spike

Date: 2026-05-21

## Scope

This was a narrow spike to test whether CoTracker3 is a viable click-and-propagate assist for dense footbag labels. It did not build a review UI and did not label training data.

Inputs came from `runs/release-27-public/dense_trajectory_review_v2_with_sources/`.

## Environment

- `cotracker` / `cotracker3` are not available as PyPI packages in this environment.
- Official Torch Hub load works:
  - `torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")`
  - checkpoint downloaded: `facebook/cotracker3/scaled_offline.pth`
- Torch: `2.11.0`
- MPS available: yes
- Spike runs used CPU at `resize_scale=0.35` for stable repeatability.

## Script

Added `cotracker_feasibility.py`.

It takes one dense-review clip, one seed frame, and one clicked point, then writes:

- `propagated_points.jsonl`
- `summary.json`
- `overlay.mp4`
- sampled `overlay_contact_sheet.jpg` generated after the run

No labels are written back to the dense-review batch.

## Clips Tested

| Case | Clip | Seed | Result |
| --- | --- | --- | --- |
| easy | `video-498_singular_display__t0003032__26866c24` | frame `70`, `(334, 639)` | Holds cleanly |
| intermediate | `video-234_singular_display_2__t0004999__5d9366ed` | frame `149`, `(333, 675)` | Holds with contact/near-foot motion |
| hard | `video-352_singular_display_2__t0004999__331bc3c7` | frame `149`, `(461, 583)` | Useful only with visibility/re-click gating |

## Metrics

### Easy: `video-498`

- Frames: `62`
- v10 greedy median distance: `2.6 px`
- v10 temporal median distance: `2.7 px`
- v11 greedy median distance: `2.9 px`
- Within 30 px of v10/v11: `98-100%`
- Visual verdict: holds the footbag across the clip.

### Intermediate: `video-234`

- Frames: `63`
- v10 greedy median distance: `12.4 px`
- v10 temporal median distance: `12.3 px`
- v10 temporal within 30 px: `96.8%`
- v11 within 30 px: `74.1%`
- Visual verdict: holds the footbag through the shoe/contact-heavy region. This is the strongest positive signal for Round 2 because this clip is from the shoe-drift tail.

### Hard: `video-352`

- Frames: `63`
- v10/v11 agreement is poor, but detector hints are visibly wrong in this clip.
- CoTracker visibility is low for most frames:
  - visibility mean: `0.1587`
  - visibility below 0.5: `53/63 frames`
- Visual verdict: useful near the seeded visible ball segment, but not safe as an unattended one-click label source. When the ball leaves visibility or the frame content changes strongly, the point continues on background/nearby texture unless the reviewer marks the segment invalid or re-clicks.

## Artifacts

- `outputs/cotracker_feasibility/easy_498/overlay_contact_sheet.jpg`
- `outputs/cotracker_feasibility/intermediate_234/overlay_contact_sheet.jpg`
- `outputs/cotracker_feasibility/hard_352/overlay_contact_sheet.jpg`
- Per-clip JSONL and MP4 overlays in the same folders.

## Verdict

CoTracker3 is viable as a reviewer assist, not as an autonomous dense-label generator.

The result matches the middle outcome from the planning prompt:

> Holds on 1-2/3, drifts or becomes uncertain on others with re-click recovery.

More precisely:

- It holds on the easy and intermediate clips.
- It is still useful on the hard clip, but only if the UI is designed around visibility gating and frequent re-clicks.
- The CoTracker visibility signal is useful as a warning, but not sufficient alone; the reviewer still needs to see the overlay.

## UI Implication

Build the dense-label reviewer UI, but design it around short propagation segments rather than one-click-per-clip.

Required UI behavior:

- Click a seed point on a clear frame.
- Propagate forward/backward.
- Show the propagated overlay immediately.
- Let the reviewer mark a propagated span as accepted.
- Let the reviewer split/re-click whenever drift, occlusion, or out-of-frame occurs.
- Default low-visibility spans to `quality=pending` or `visibility=uncertain`, not training labels.

Do not build a UI that assumes one click can label a whole clip reliably.
