# Hacky Track MVP

Hacky Track is a small computer-vision/audio prototype for turning first-person footbag footage into a rally timeline: touches, drops, stalls, counts, proof frames, and annotated videos.

The current implementation is intentionally narrow. It was built around one Meta Glasses clip, `video-50_singular_display.MOV`, where a simple "count the kicks" request becomes surprisingly tricky once you try to make the result auditable.

## The Problem

Footbag rally tracking looks easy to a person watching the clip, but it is awkward for a naive video script:

- The bag is small, fast, and often motion-blurred.
- First-person camera motion makes object tracking unstable.
- The foot and bag can overlap for only a few frames.
- Audio spikes are useful contact clues, but footsteps, floor hits, and duplicated rebounds create false positives.
- A rally is not just a stream of contacts. The system also needs to understand floor resets, final drops, and stalls where the bag is controlled instead of kicked.
- The output has to be explainable. A raw count is not enough if a user thinks the tracker duplicated or missed a touch.

This repo captures an MVP answer to that problem: use audio transients to propose likely contact times, keep a checked event file as the source of truth, and render visual artifacts that make the count easy to inspect.

For the included annotation file, the clip contains:

- Rally 1: 2 touches, then a floor reset.
- Rally 2: 8 touches, then a floor reset.
- Rally 3: 2 touches, then a toe stall.

## What It Does

`hacky_mvp.py` takes a local video and a checked event JSON file, then generates:

- `rally_events.json`: structured event data enriched with nearest audio peak timing.
- `rally_events.csv`: spreadsheet-friendly event table.
- `event_sheet.jpg`: proof-frame contact sheet for the labeled events.
- `annotated_rallies.mp4`: video with event labels and a timeline.
- `hud_overlay.mp4`: richer HUD overlay with live rally count, total touches, best rally, event flashes, stall state, and timeline markers.

## How It Works

1. Extract mono audio from the input video with `ffmpeg`.
2. Compute short-window RMS energy and robust z-scores.
3. Detect transient peaks as contact candidates.
4. Load checked rally annotations from `data/video-50_singular_display.events.json`.
5. Attach nearest audio-peak evidence to each checked event.
6. Render JSON, CSV, proof sheet, annotated video, and HUD video outputs.

The MVP does not yet infer every event end-to-end from vision alone. The checked event file is deliberately explicit because the goal of this version is to create an auditable workflow and a clear target for future automation.

## Requirements

- Python 3.11+
- `ffmpeg` available on your `PATH`
- Python packages in `requirements.txt`

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, install `ffmpeg` with Homebrew if needed:

```bash
brew install ffmpeg
```

## Usage

Put the source clip somewhere local. The video itself is not committed to this repo.

```bash
python3 hacky_mvp.py /path/to/video-50_singular_display.MOV
```

Optional flags:

```bash
python3 hacky_mvp.py /path/to/video.MOV \
  --events data/video-50_singular_display.events.json \
  --out outputs/video-50_singular_display \
  --scale 0.5
```

The script prints a rally summary and writes generated artifacts under `outputs/`.

## Repository Layout

```text
.
├── data/
│   └── video-50_singular_display.events.json
├── hacky_mvp.py
├── requirements.txt
└── README.md
```

Generated media and local input videos are ignored by Git.

## Current Limitations

- It is tuned to one known clip and one annotation file.
- The input event JSON is still human-checked ground truth, not a fully automatic detector.
- Audio transients help find contacts, but they are not reliable enough alone for final event labeling.
- The script assumes the checked events use the same timeline as the input video.
- There is no model-based bag detector or pose tracker yet.

## Future Directions

- Add automatic bag detection and short-horizon tracking.
- Fuse pose, optical flow, and audio peaks for better contact confidence.
- Add a review UI for accepting/rejecting candidate touches.
- Generalize rally segmentation across arbitrary clips.
- Export highlight clips for each rally.
- Add tests around event summarization and audio-peak matching.
