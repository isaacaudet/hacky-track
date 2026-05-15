#!/usr/bin/env python3
"""Local touch-review app for Hacky Track candidate labels.

The detector writes candidate touches. This app keeps a separate human-review
layer so the raw model output stays reproducible while approved/rejected/manual
labels can feed the next training pass.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
REVIEWS_DIR = ROOT / "reviews"
DOWNLOADS_DIR = Path.home() / "Downloads"
DEFAULT_TRACK_SIZE = {"width": 688, "height": 912}


@dataclass(frozen=True)
class ReviewableVideo:
    stem: str
    source_video: str
    video_path: Path
    event_path: Path
    review_path: Path
    candidate_source: str
    track_size: dict[str, int]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def slug_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")


def find_video_path(source_video: str, event_path: Path) -> Path:
    candidates = [
        DOWNLOADS_DIR / source_video,
        ROOT / source_video,
        event_path.parent / source_video,
        event_path.parent / "source.mov",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return DOWNLOADS_DIR / source_video


def discover_videos() -> dict[str, ReviewableVideo]:
    found: dict[str, ReviewableVideo] = {}
    if not OUTPUTS_DIR.exists():
        return found

    for out_dir in sorted(path for path in OUTPUTS_DIR.iterdir() if path.is_dir()):
        event_path = out_dir / "trained_events.json"
        candidate_source = "trained_events.json"
        if not event_path.exists():
            event_path = out_dir / "candidate_events.json"
            candidate_source = "candidate_events.json"
        if not event_path.exists():
            continue

        try:
            doc = read_json(event_path)
        except json.JSONDecodeError:
            continue

        source_video = str(doc.get("source_video") or f"{out_dir.name}.MOV")
        stem = Path(source_video).stem or out_dir.name
        found[stem] = ReviewableVideo(
            stem=stem,
            source_video=source_video,
            video_path=find_video_path(source_video, event_path),
            event_path=event_path,
            review_path=REVIEWS_DIR / f"{stem}.review.json",
            candidate_source=candidate_source,
            track_size=DEFAULT_TRACK_SIZE.copy(),
        )
    return found


def candidate_item(rally: dict[str, Any], event: dict[str, Any], event_path: Path) -> dict[str, Any] | None:
    if event.get("type") != "touch":
        return None
    rally_id = int(rally.get("id") or 0)
    touch_number = int(event.get("touch_number") or 0)
    time_sec = round(float(event.get("time_sec") or 0.0), 3)
    item_id = f"cand-r{rally_id:03d}-t{touch_number:03d}-{int(round(time_sec * 1000)):07d}"
    item = {
        "id": item_id,
        "source": "candidate",
        "candidate_file": str(event_path.relative_to(ROOT)) if event_path.is_relative_to(ROOT) else str(event_path),
        "kind": "touch",
        "status": "pending",
        "rally_id": rally_id or None,
        "touch_number": touch_number or None,
        "time_sec": time_sec,
        "confidence": event.get("confidence"),
        "audio_z": event.get("audio_z"),
        "visual_score": event.get("visual_score"),
        "motion_score": event.get("motion_score"),
        "x": event.get("x"),
        "y": event.get("y"),
        "note": "",
    }
    return item


def initialize_review_doc(info: ReviewableVideo) -> dict[str, Any]:
    event_doc = read_json(info.event_path)
    items: list[dict[str, Any]] = []
    for rally in event_doc.get("rallies", []):
        for event in rally.get("events", []):
            item = candidate_item(rally, event, info.event_path)
            if item:
                items.append(item)

    items.sort(key=lambda item: (float(item.get("time_sec") or 0.0), item.get("id", "")))
    return {
        "version": 1,
        "source_video": info.source_video,
        "video_path": str(info.video_path),
        "candidate_source": str(info.event_path.relative_to(ROOT)) if info.event_path.is_relative_to(ROOT) else str(info.event_path),
        "track_size": info.track_size,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "items": items,
    }


def merged_review_doc(info: ReviewableVideo) -> dict[str, Any]:
    base = initialize_review_doc(info)
    if not info.review_path.exists():
        return base

    try:
        existing = read_json(info.review_path)
    except json.JSONDecodeError:
        return base

    existing_by_id = {str(item.get("id")): item for item in existing.get("items", []) if item.get("id")}
    merged: list[dict[str, Any]] = []
    base_ids: set[str] = set()

    for item in base["items"]:
        item_id = str(item["id"])
        base_ids.add(item_id)
        previous = existing_by_id.get(item_id)
        if previous:
            merged_item = {**item, **previous}
            merged.append(merged_item)
        else:
            merged.append(item)

    for item in existing.get("items", []):
        item_id = str(item.get("id") or "")
        if item_id and item_id not in base_ids and item.get("source") == "manual":
            merged.append(item)

    merged.sort(key=lambda item: (float(item.get("time_sec") or 0.0), item.get("id", "")))
    base["items"] = merged
    base["created_at"] = existing.get("created_at") or base["created_at"]
    base["updated_at"] = existing.get("updated_at") or base["updated_at"]
    return base


def review_summary(info: ReviewableVideo) -> dict[str, Any]:
    doc = merged_review_doc(info)
    counts = {"pending": 0, "approved": 0, "rejected": 0, "missing": 0}
    for item in doc.get("items", []):
        status = str(item.get("status") or "pending")
        if status in counts:
            counts[status] += 1
    return {
        "stem": info.stem,
        "source_video": info.source_video,
        "video_exists": info.video_path.exists(),
        "video_path": str(info.video_path),
        "events_path": str(info.event_path.relative_to(ROOT)) if info.event_path.is_relative_to(ROOT) else str(info.event_path),
        "review_path": str(info.review_path.relative_to(ROOT)) if info.review_path.is_relative_to(ROOT) else str(info.review_path),
        "candidate_source": info.candidate_source,
        "total_items": len(doc.get("items", [])),
        "counts": counts,
        "updated_at": doc.get("updated_at"),
    }


def accepted_events_doc(review_doc: dict[str, Any]) -> dict[str, Any]:
    accepted = [
        item
        for item in review_doc.get("items", [])
        if item.get("kind") == "touch" and item.get("status") in {"approved", "missing"}
    ]
    accepted.sort(key=lambda item: float(item.get("time_sec") or 0.0))

    rallies: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    rally_id = 1
    last_time: float | None = None
    for item in accepted:
        time_sec = float(item.get("time_sec") or 0.0)
        if current and last_time is not None and time_sec - last_time > 2.2:
            rallies.append(build_rally(rally_id, current))
            rally_id += 1
            current = []
        current.append(item)
        last_time = time_sec
    if current:
        rallies.append(build_rally(rally_id, current))

    return {
        "source_video": review_doc.get("source_video"),
        "annotation_method": "review_app_approved",
        "created_at": utc_now(),
        "rallies": rallies,
    }


def build_rally(rally_id: int, items: list[dict[str, Any]]) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        event = {
            "type": "touch",
            "touch_number": index,
            "time_sec": round(float(item.get("time_sec") or 0.0), 3),
            "label": "manual_missing" if item.get("status") == "missing" else "approved_candidate",
            "review_item_id": item.get("id"),
            "confidence": item.get("confidence"),
            "audio_z": item.get("audio_z"),
            "visual_score": item.get("visual_score"),
            "motion_score": item.get("motion_score"),
            "x": item.get("x"),
            "y": item.get("y"),
        }
        events.append({key: value for key, value in event.items() if value is not None})

    start = float(items[0].get("time_sec") or 0.0)
    end = float(items[-1].get("time_sec") or start)
    return {
        "id": rally_id,
        "label": f"Reviewed rally {rally_id}",
        "start_sec": round(start, 3),
        "end_sec": round(end, 3),
        "expected_touches": len(events),
        "expected_stalls": 0,
        "events": events,
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hacky Track Review</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101112;
      --panel: #181a1d;
      --panel-2: #202329;
      --line: #30343b;
      --text: #f1f3f4;
      --muted: #a6adb7;
      --dim: #747d8a;
      --green: #76d68a;
      --red: #ff7770;
      --yellow: #ffd65c;
      --blue: #76b9ff;
      --ink: #08090a;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }

    button, select, input, textarea {
      font: inherit;
    }

    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }

    header {
      display: flex;
      gap: 14px;
      align-items: center;
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      background: #141619;
    }

    .brand {
      display: flex;
      align-items: baseline;
      gap: 10px;
      min-width: 190px;
    }

    h1 {
      margin: 0;
      font-size: 16px;
      font-weight: 780;
      letter-spacing: 0;
    }

    .save-state {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }

    .top-select {
      min-width: 280px;
      color: var(--text);
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 8px 10px;
      outline: none;
    }

    .counts {
      display: flex;
      gap: 8px;
      align-items: center;
      margin-left: auto;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--muted);
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 12px;
      white-space: nowrap;
    }

    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--dim);
      flex: 0 0 auto;
    }

    .dot.pending { background: var(--yellow); }
    .dot.approved { background: var(--green); }
    .dot.rejected { background: var(--red); }
    .dot.missing { background: var(--blue); }

    main {
      display: grid;
      grid-template-columns: minmax(420px, 1fr) 430px;
      min-height: 0;
    }

    .viewer {
      min-width: 0;
      display: grid;
      grid-template-rows: 1fr auto;
      background: #0b0c0d;
      border-right: 1px solid var(--line);
    }

    .video-stage {
      min-height: 0;
      display: grid;
      place-items: center;
      padding: 16px;
      background:
        linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(0deg, rgba(255,255,255,0.025) 1px, transparent 1px),
        #0b0c0d;
      background-size: 28px 28px;
    }

    .video-wrap {
      position: relative;
      height: min(calc(100vh - 170px), 84vw);
      max-height: 100%;
      aspect-ratio: 688 / 912;
      background: #000;
      box-shadow: 0 18px 60px rgba(0,0,0,.45);
      overflow: hidden;
    }

    video {
      display: block;
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #000;
    }

    .marker {
      position: absolute;
      width: 54px;
      height: 54px;
      margin-left: -27px;
      margin-top: -27px;
      border: 3px solid var(--yellow);
      border-radius: 50%;
      box-shadow: 0 0 0 2px rgba(0,0,0,.8), 0 0 24px rgba(255,214,92,.35);
      pointer-events: none;
      opacity: 0;
      transform: scale(.9);
      transition: opacity .12s ease, transform .12s ease;
    }

    .marker.show {
      opacity: 1;
      transform: scale(1);
    }

    .control-strip {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: center;
      padding: 12px 16px;
      border-top: 1px solid var(--line);
      background: #141619;
    }

    .selected-line {
      min-width: 0;
      display: flex;
      gap: 10px;
      align-items: center;
      color: var(--muted);
      white-space: nowrap;
      overflow: hidden;
    }

    .selected-line strong {
      color: var(--text);
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .actions {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    button {
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--text);
      border-radius: 7px;
      padding: 8px 11px;
      cursor: pointer;
      min-height: 36px;
    }

    button:hover { border-color: #4b5360; background: #262a31; }
    button:active { transform: translateY(1px); }
    button.primary { background: #d9f99d; border-color: #d9f99d; color: var(--ink); font-weight: 760; }
    button.danger { background: #3a1d1e; border-color: #673031; color: #ffd0cd; }
    button.blue { background: #152a3d; border-color: #31577b; color: #d6ebff; }
    button.ghost { color: var(--muted); }

    aside {
      min-width: 0;
      min-height: 0;
      display: grid;
      grid-template-rows: auto auto 1fr auto;
      background: var(--panel);
    }

    .tool-row {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 8px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
    }

    .filter-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
    }

    input, textarea {
      width: 100%;
      color: var(--text);
      background: #111317;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 9px 10px;
      outline: none;
    }

    input:focus, textarea:focus, select:focus {
      border-color: #596473;
      box-shadow: 0 0 0 2px rgba(118,185,255,.14);
    }

    .queue {
      min-height: 0;
      overflow: auto;
    }

    .row {
      width: 100%;
      display: grid;
      grid-template-columns: 28px 72px 1fr 64px;
      gap: 8px;
      align-items: center;
      padding: 9px 12px;
      border: 0;
      border-bottom: 1px solid rgba(48,52,59,.8);
      background: transparent;
      color: var(--text);
      border-radius: 0;
      text-align: left;
      min-height: 48px;
    }

    .row:hover { background: rgba(255,255,255,.035); border-color: rgba(48,52,59,.8); }
    .row.active { background: #26303a; box-shadow: inset 3px 0 0 var(--yellow); }

    .row-index {
      color: var(--dim);
      font-variant-numeric: tabular-nums;
      font-size: 12px;
    }

    .row-time {
      color: var(--text);
      font-weight: 720;
      font-variant-numeric: tabular-nums;
    }

    .row-main {
      min-width: 0;
      display: grid;
      gap: 2px;
    }

    .row-title {
      display: flex;
      gap: 6px;
      align-items: center;
      min-width: 0;
    }

    .row-title span:last-child {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .row-sub {
      color: var(--muted);
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .score {
      color: var(--muted);
      font-variant-numeric: tabular-nums;
      text-align: right;
      font-size: 12px;
    }

    .details {
      border-top: 1px solid var(--line);
      padding: 12px;
      display: grid;
      gap: 10px;
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 8px;
    }

    .metric {
      background: #121417;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 8px;
      min-width: 0;
    }

    .metric-label {
      color: var(--dim);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }

    .metric-value {
      color: var(--text);
      font-size: 13px;
      font-weight: 720;
      font-variant-numeric: tabular-nums;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .note-row {
      display: grid;
      gap: 7px;
    }

    textarea {
      resize: vertical;
      min-height: 58px;
      max-height: 140px;
    }

    .footer-actions {
      display: flex;
      gap: 8px;
      justify-content: space-between;
      align-items: center;
    }

    .path-note {
      color: var(--dim);
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .empty {
      padding: 28px 18px;
      color: var(--muted);
    }

    @media (max-width: 980px) {
      header { align-items: stretch; flex-wrap: wrap; }
      .top-select { min-width: 100%; }
      .counts { margin-left: 0; justify-content: flex-start; }
      main { grid-template-columns: 1fr; grid-template-rows: minmax(420px, 58vh) minmax(420px, 42vh); }
      .viewer { border-right: 0; border-bottom: 1px solid var(--line); }
      aside { min-height: 420px; }
      .video-wrap { height: 100%; max-width: 100%; }
      .control-strip { grid-template-columns: 1fr; }
      .actions { justify-content: flex-start; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div class="brand">
        <h1>Hacky Track Review</h1>
        <span id="saveState" class="save-state">loading</span>
      </div>
      <select id="videoSelect" class="top-select"></select>
      <div id="counts" class="counts"></div>
    </header>

    <main>
      <section class="viewer">
        <div class="video-stage">
          <div class="video-wrap" id="videoWrap">
            <video id="video" controls playsinline preload="metadata"></video>
            <div id="marker" class="marker"></div>
          </div>
        </div>
        <div class="control-strip">
          <div class="selected-line">
            <span id="selectedDot" class="dot pending"></span>
            <strong id="selectedTitle">No touch selected</strong>
            <span id="selectedTime"></span>
          </div>
          <div class="actions">
            <button id="prevBtn" class="ghost" title="Previous candidate">Prev</button>
            <button id="approveBtn" class="primary" title="Approve selected touch">Approve</button>
            <button id="rejectBtn" class="danger" title="Reject selected touch">Reject</button>
            <button id="pendingBtn" class="ghost" title="Return selected touch to pending">Pending</button>
            <button id="addMissingBtn" class="blue" title="Add touch at current video time">Add Touch</button>
            <button id="nextBtn" class="ghost" title="Next candidate">Next</button>
          </div>
        </div>
      </section>

      <aside>
        <div class="tool-row">
          <button id="saveBtn">Save</button>
          <button id="exportBtn">Export Approved</button>
          <button id="playClipBtn">Play Clip</button>
        </div>
        <div class="filter-row">
          <input id="filterInput" type="search" placeholder="Filter status, time, note">
          <button id="pendingOnlyBtn" class="ghost">Pending</button>
        </div>
        <div id="queue" class="queue"></div>
        <div class="details">
          <div class="metrics">
            <div class="metric"><div class="metric-label">Status</div><div id="metricStatus" class="metric-value">-</div></div>
            <div class="metric"><div class="metric-label">Rally</div><div id="metricRally" class="metric-value">-</div></div>
            <div class="metric"><div class="metric-label">Conf</div><div id="metricConf" class="metric-value">-</div></div>
            <div class="metric"><div class="metric-label">Audio</div><div id="metricAudio" class="metric-value">-</div></div>
          </div>
          <div class="actions">
            <button id="nudgeBackBtn" class="ghost" title="Move selected time earlier">-0.05s</button>
            <button id="setCurrentBtn" class="ghost" title="Set selected time to current video time">Set Time</button>
            <button id="nudgeForwardBtn" class="ghost" title="Move selected time later">+0.05s</button>
            <button id="deleteManualBtn" class="danger" title="Delete selected manual touch">Delete Manual</button>
          </div>
          <div class="note-row">
            <textarea id="noteInput" placeholder="Note"></textarea>
          </div>
          <div class="footer-actions">
            <span id="pathNote" class="path-note"></span>
          </div>
        </div>
      </aside>
    </main>
  </div>

  <script>
    const state = {
      videos: [],
      stem: null,
      review: null,
      selectedId: null,
      dirty: false,
      saveTimer: null,
      filter: "",
      pendingOnly: false,
      lastExport: ""
    };

    const $ = (id) => document.getElementById(id);
    const video = $("video");
    const marker = $("marker");

    function fmtTime(value) {
      const t = Number(value || 0);
      const m = Math.floor(t / 60);
      const s = (t - m * 60).toFixed(3).padStart(6, "0");
      return `${m}:${s}`;
    }

    function fmtShort(value) {
      if (value === null || value === undefined || value === "") return "-";
      const num = Number(value);
      if (!Number.isFinite(num)) return String(value);
      return num.toFixed(num >= 10 ? 1 : 2);
    }

    function itemLabel(item) {
      if (!item) return "No touch selected";
      const source = item.source === "manual" ? "Manual" : "Candidate";
      const rally = item.rally_id ? `R${item.rally_id}` : "No rally";
      const touch = item.touch_number ? `T${item.touch_number}` : "touch";
      return `${source} ${rally} ${touch}`;
    }

    function sortedItems() {
      if (!state.review) return [];
      return [...state.review.items].sort((a, b) => {
        const dt = Number(a.time_sec || 0) - Number(b.time_sec || 0);
        return dt || String(a.id).localeCompare(String(b.id));
      });
    }

    function visibleItems() {
      const filter = state.filter.trim().toLowerCase();
      return sortedItems().filter((item) => {
        if (state.pendingOnly && item.status !== "pending") return false;
        if (!filter) return true;
        const haystack = [
          item.status,
          item.source,
          item.note,
          item.rally_id,
          item.touch_number,
          fmtTime(item.time_sec),
          Number(item.time_sec || 0).toFixed(3)
        ].join(" ").toLowerCase();
        return haystack.includes(filter);
      });
    }

    function currentItem() {
      if (!state.review || !state.selectedId) return null;
      return state.review.items.find((item) => item.id === state.selectedId) || null;
    }

    function setSaveState(text) {
      $("saveState").textContent = text;
    }

    function markDirty() {
      state.dirty = true;
      setSaveState("unsaved");
      window.clearTimeout(state.saveTimer);
      state.saveTimer = window.setTimeout(saveNow, 600);
    }

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: {"Content-Type": "application/json"},
        ...options
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `${response.status} ${response.statusText}`);
      }
      return response.json();
    }

    async function loadVideos() {
      const data = await api("/api/videos");
      state.videos = data.videos;
      const select = $("videoSelect");
      select.innerHTML = "";
      for (const entry of state.videos) {
        const option = document.createElement("option");
        option.value = entry.stem;
        option.textContent = `${entry.source_video} (${entry.total_items})`;
        select.appendChild(option);
      }
      if (!state.videos.length) {
        $("queue").innerHTML = '<div class="empty">No trained event files found under outputs.</div>';
        setSaveState("empty");
        return;
      }
      await loadReview(state.videos[0].stem);
    }

    async function loadReview(stem) {
      state.stem = stem;
      $("videoSelect").value = stem;
      state.review = await api(`/api/review/${encodeURIComponent(stem)}`);
      state.selectedId = null;
      state.dirty = false;
      state.lastExport = "";
      video.src = state.review.video_url;
      const firstPending = sortedItems().find((item) => item.status === "pending");
      const first = firstPending || sortedItems()[0] || null;
      if (first) state.selectedId = first.id;
      setSaveState("saved");
      render();
      if (first) seekToItem(first, false);
    }

    async function saveNow() {
      if (!state.review || !state.stem) return;
      window.clearTimeout(state.saveTimer);
      setSaveState("saving");
      state.review.items = sortedItems();
      const result = await api(`/api/review/${encodeURIComponent(state.stem)}`, {
        method: "POST",
        body: JSON.stringify(state.review)
      });
      state.review.updated_at = result.updated_at;
      state.dirty = false;
      setSaveState("saved");
      renderCounts();
    }

    async function exportApproved() {
      await saveNow();
      const result = await api(`/api/export/${encodeURIComponent(state.stem)}`, {method: "POST"});
      state.lastExport = result.export_path;
      $("pathNote").textContent = result.export_path;
      setSaveState("exported");
    }

    function seekToItem(item, play = false) {
      if (!item) return;
      const preRoll = 0.45;
      video.currentTime = Math.max(0, Number(item.time_sec || 0) - preRoll);
      if (play) {
        const stopAt = Number(item.time_sec || 0) + 0.65;
        video.play().catch(() => {});
        const stop = () => {
          if (video.currentTime >= stopAt) {
            video.pause();
            video.removeEventListener("timeupdate", stop);
          }
        };
        video.addEventListener("timeupdate", stop);
      }
    }

    function selectItem(item, seek = true) {
      if (!item) return;
      state.selectedId = item.id;
      if (seek) seekToItem(item, false);
      render();
    }

    function selectRelative(delta) {
      const items = visibleItems();
      if (!items.length) return;
      const index = Math.max(0, items.findIndex((item) => item.id === state.selectedId));
      const next = Math.min(items.length - 1, Math.max(0, index + delta));
      selectItem(items[next], true);
    }

    function setStatus(status) {
      const item = currentItem();
      if (!item) return;
      item.status = status;
      markDirty();
      render();
      if (status === "approved" || status === "rejected") {
        const items = visibleItems();
        const index = items.findIndex((entry) => entry.id === item.id);
        const next = items.slice(index + 1).find((entry) => entry.status === "pending") || items[index + 1];
        if (next) selectItem(next, true);
      }
    }

    function addMissingTouch() {
      if (!state.review) return;
      const timeSec = Math.max(0, video.currentTime || 0);
      const id = `manual-${Date.now()}-${Math.round(timeSec * 1000)}`;
      const item = {
        id,
        source: "manual",
        kind: "touch",
        status: "missing",
        rally_id: nearestRallyId(timeSec),
        touch_number: null,
        time_sec: Number(timeSec.toFixed(3)),
        confidence: null,
        audio_z: null,
        visual_score: null,
        motion_score: null,
        x: null,
        y: null,
        note: ""
      };
      state.review.items.push(item);
      state.selectedId = id;
      markDirty();
      render();
    }

    function nearestRallyId(timeSec) {
      const candidates = sortedItems().filter((item) => item.rally_id);
      let best = null;
      for (const item of candidates) {
        const dist = Math.abs(Number(item.time_sec || 0) - timeSec);
        if (!best || dist < best.dist) best = {dist, rally_id: item.rally_id};
      }
      return best && best.dist <= 2.5 ? best.rally_id : null;
    }

    function nudgeSelected(delta) {
      const item = currentItem();
      if (!item) return;
      item.time_sec = Number(Math.max(0, Number(item.time_sec || 0) + delta).toFixed(3));
      markDirty();
      render();
      seekToItem(item, false);
    }

    function setSelectedToCurrent() {
      const item = currentItem();
      if (!item) return;
      item.time_sec = Number(Math.max(0, video.currentTime || 0).toFixed(3));
      markDirty();
      render();
    }

    function deleteManual() {
      const item = currentItem();
      if (!item || item.source !== "manual") return;
      state.review.items = state.review.items.filter((entry) => entry.id !== item.id);
      const next = sortedItems().find((entry) => Number(entry.time_sec || 0) >= Number(item.time_sec || 0)) || sortedItems()[0] || null;
      state.selectedId = next ? next.id : null;
      markDirty();
      render();
    }

    function renderCounts() {
      const counts = {pending: 0, approved: 0, rejected: 0, missing: 0};
      if (state.review) {
        for (const item of state.review.items) {
          if (counts[item.status] !== undefined) counts[item.status] += 1;
        }
      }
      $("counts").innerHTML = Object.entries(counts).map(([key, value]) => (
        `<span class="pill"><span class="dot ${key}"></span>${key} ${value}</span>`
      )).join("");
    }

    function renderQueue() {
      const queue = $("queue");
      const items = visibleItems();
      if (!items.length) {
        queue.innerHTML = '<div class="empty">No touches match this view.</div>';
        return;
      }
      queue.innerHTML = "";
      items.forEach((item, index) => {
        const row = document.createElement("button");
        row.className = `row ${item.id === state.selectedId ? "active" : ""}`;
        row.type = "button";
        row.innerHTML = `
          <span class="row-index">${index + 1}</span>
          <span class="row-time">${fmtTime(item.time_sec)}</span>
          <span class="row-main">
            <span class="row-title"><span class="dot ${item.status}"></span><span>${itemLabel(item)}</span></span>
            <span class="row-sub">${item.source}${item.note ? " - " + escapeHtml(item.note) : ""}</span>
          </span>
          <span class="score">${fmtShort(item.confidence)}</span>
        `;
        row.addEventListener("click", () => selectItem(item, true));
        row.addEventListener("dblclick", () => seekToItem(item, true));
        queue.appendChild(row);
      });
    }

    function escapeHtml(value) {
      return String(value || "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }[char]));
    }

    function renderSelected() {
      const item = currentItem();
      $("selectedTitle").textContent = itemLabel(item);
      $("selectedTime").textContent = item ? fmtTime(item.time_sec) : "";
      $("selectedDot").className = `dot ${item ? item.status : "pending"}`;
      $("metricStatus").textContent = item ? item.status : "-";
      $("metricRally").textContent = item && item.rally_id ? `R${item.rally_id}` : "-";
      $("metricConf").textContent = item ? fmtShort(item.confidence) : "-";
      $("metricAudio").textContent = item ? fmtShort(item.audio_z) : "-";
      $("noteInput").value = item ? (item.note || "") : "";
      $("deleteManualBtn").disabled = !item || item.source !== "manual";

      const size = state.review && state.review.track_size ? state.review.track_size : {width: 688, height: 912};
      if (item && item.x !== null && item.x !== undefined && item.y !== null && item.y !== undefined) {
        marker.style.left = `${(Number(item.x) / Number(size.width || 688)) * 100}%`;
        marker.style.top = `${(Number(item.y) / Number(size.height || 912)) * 100}%`;
        marker.classList.add("show");
      } else {
        marker.classList.remove("show");
      }

      const path = state.lastExport || (state.review ? state.review.review_path || "" : "");
      $("pathNote").textContent = path;
    }

    function render() {
      renderCounts();
      renderQueue();
      renderSelected();
      $("pendingOnlyBtn").classList.toggle("primary", state.pendingOnly);
    }

    $("videoSelect").addEventListener("change", async (event) => {
      if (state.dirty) await saveNow();
      await loadReview(event.target.value);
    });
    $("saveBtn").addEventListener("click", saveNow);
    $("exportBtn").addEventListener("click", exportApproved);
    $("playClipBtn").addEventListener("click", () => seekToItem(currentItem(), true));
    $("prevBtn").addEventListener("click", () => selectRelative(-1));
    $("nextBtn").addEventListener("click", () => selectRelative(1));
    $("approveBtn").addEventListener("click", () => setStatus("approved"));
    $("rejectBtn").addEventListener("click", () => setStatus("rejected"));
    $("pendingBtn").addEventListener("click", () => setStatus("pending"));
    $("addMissingBtn").addEventListener("click", addMissingTouch);
    $("nudgeBackBtn").addEventListener("click", () => nudgeSelected(-0.05));
    $("nudgeForwardBtn").addEventListener("click", () => nudgeSelected(0.05));
    $("setCurrentBtn").addEventListener("click", setSelectedToCurrent);
    $("deleteManualBtn").addEventListener("click", deleteManual);
    $("filterInput").addEventListener("input", (event) => {
      state.filter = event.target.value;
      renderQueue();
    });
    $("pendingOnlyBtn").addEventListener("click", () => {
      state.pendingOnly = !state.pendingOnly;
      render();
    });
    $("noteInput").addEventListener("input", (event) => {
      const item = currentItem();
      if (!item) return;
      item.note = event.target.value;
      markDirty();
      renderQueue();
    });

    window.addEventListener("keydown", (event) => {
      const tag = document.activeElement && document.activeElement.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (event.key === "a" || event.key === "A") { setStatus("approved"); event.preventDefault(); }
      if (event.key === "r" || event.key === "R") { setStatus("rejected"); event.preventDefault(); }
      if (event.key === "p" || event.key === "P") { setStatus("pending"); event.preventDefault(); }
      if (event.key === "m" || event.key === "M") { addMissingTouch(); event.preventDefault(); }
      if (event.key === "ArrowDown" || event.key === "n" || event.key === "N") { selectRelative(1); event.preventDefault(); }
      if (event.key === "ArrowUp" || event.key === "b" || event.key === "B") { selectRelative(-1); event.preventDefault(); }
      if (event.key === "[") { nudgeSelected(-0.05); event.preventDefault(); }
      if (event.key === "]") { nudgeSelected(0.05); event.preventDefault(); }
      if (event.key === " ") {
        if (video.paused) video.play().catch(() => {});
        else video.pause();
        event.preventDefault();
      }
    });

    window.addEventListener("beforeunload", (event) => {
      if (!state.dirty) return;
      event.preventDefault();
      event.returnValue = "";
    });

    loadVideos().catch((error) => {
      console.error(error);
      $("queue").innerHTML = `<div class="empty">${escapeHtml(error.message)}</div>`;
      setSaveState("error");
    });
  </script>
</body>
</html>
"""


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "HackyTrackReview/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/":
            self.send_text(HTML, "text/html; charset=utf-8")
            return
        if path == "/api/videos":
            videos = [review_summary(info) for info in self.videos().values()]
            self.send_json({"videos": videos})
            return
        if path.startswith("/api/review/"):
            stem = slug_id(path.rsplit("/", 1)[-1])
            info = self.require_video(stem)
            if not info:
                return
            doc = merged_review_doc(info)
            doc["video_url"] = f"/media/{info.stem}"
            doc["review_path"] = str(info.review_path.relative_to(ROOT)) if info.review_path.is_relative_to(ROOT) else str(info.review_path)
            self.send_json(doc)
            return
        if path.startswith("/media/"):
            stem = slug_id(path.rsplit("/", 1)[-1])
            info = self.require_video(stem)
            if not info:
                return
            self.send_file(info.video_path)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path.startswith("/media/"):
            stem = slug_id(path.rsplit("/", 1)[-1])
            info = self.require_video(stem)
            if not info:
                return
            self.send_file(info.video_path, head_only=True)
            return
        if path == "/":
            body = HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path.startswith("/api/review/"):
            stem = slug_id(path.rsplit("/", 1)[-1])
            info = self.require_video(stem)
            if not info:
                return
            payload = self.read_body_json()
            if payload is None:
                return
            if not isinstance(payload.get("items"), list):
                self.send_error(HTTPStatus.BAD_REQUEST, "Review document must include an items list")
                return
            payload.pop("video_url", None)
            payload.pop("review_path", None)
            payload["source_video"] = info.source_video
            payload["video_path"] = str(info.video_path)
            payload["candidate_source"] = str(info.event_path.relative_to(ROOT)) if info.event_path.is_relative_to(ROOT) else str(info.event_path)
            payload["track_size"] = payload.get("track_size") or info.track_size
            payload["updated_at"] = utc_now()
            payload.setdefault("created_at", payload["updated_at"])
            payload["items"] = sorted(
                payload["items"],
                key=lambda item: (float(item.get("time_sec") or 0.0), str(item.get("id") or "")),
            )
            write_json(info.review_path, payload)
            self.send_json({"ok": True, "updated_at": payload["updated_at"], "review_path": str(info.review_path)})
            return

        if path.startswith("/api/export/"):
            stem = slug_id(path.rsplit("/", 1)[-1])
            info = self.require_video(stem)
            if not info:
                return
            review_doc = merged_review_doc(info)
            if info.review_path.exists():
                review_doc = read_json(info.review_path)
            export_doc = accepted_events_doc(review_doc)
            export_path = REVIEWS_DIR / f"{info.stem}.approved_events.json"
            write_json(export_path, export_doc)
            display_path = str(export_path.relative_to(ROOT)) if export_path.is_relative_to(ROOT) else str(export_path)
            self.send_json({"ok": True, "export_path": display_path, "rallies": len(export_doc["rallies"])})
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def videos(self) -> dict[str, ReviewableVideo]:
        return self.server.review_videos  # type: ignore[attr-defined]

    def require_video(self, stem: str) -> ReviewableVideo | None:
        info = self.videos().get(stem)
        if not info:
            self.send_error(HTTPStatus.NOT_FOUND, f"Unknown video: {stem}")
            return None
        return info

    def read_body_json(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON")
            return None
        if not isinstance(payload, dict):
            self.send_error(HTTPStatus.BAD_REQUEST, "Expected a JSON object")
            return None
        return payload

    def send_json(self, data: dict[str, Any]) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str, content_type: str) -> None:
        body = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, *, head_only: bool = False) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, f"File not found: {path}")
            return

        size = path.stat().st_size
        range_header = self.headers.get("Range")
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix.lower() == ".mov":
            content_type = "video/quicktime"

        start = 0
        end = size - 1
        status = HTTPStatus.OK
        if range_header:
            match = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if match:
                if match.group(1):
                    start = int(match.group(1))
                if match.group(2):
                    end = int(match.group(2))
                end = min(end, size - 1)
                status = HTTPStatus.PARTIAL_CONTENT

        if start < 0 or start >= size or end < start:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head_only:
            return

        with path.open("rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def free_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Hacky Track touch review app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos = discover_videos()
    if not videos:
        print(f"No reviewable videos found under {OUTPUTS_DIR}")
    port = free_port(args.port) if args.host in {"127.0.0.1", "localhost"} else args.port
    server = ThreadingHTTPServer((args.host, port), ReviewHandler)
    server.review_videos = videos  # type: ignore[attr-defined]
    print(f"review app: http://{args.host}:{port}/")
    print(f"videos: {len(videos)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
