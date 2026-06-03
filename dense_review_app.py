#!/usr/bin/env python3
"""Lean dense trajectory review app.

This is intentionally a small local-only reviewer for the Round 2 dense trajectory
batch. It reads the exported frame rows, lets a human stamp reviewed labels, and
keeps audit_only rows locked out of training by preserving the original split fields.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import socket
import subprocess
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).resolve().parent
DEFAULT_BATCH = ROOT / "runs" / "release-27-public" / "dense_trajectory_review_v2_with_sources"
TEMPLATE_NAME = "dense_trajectory_labels_template.jsonl"
REVIEWED_NAME = "dense_trajectory_labels.reviewed.jsonl"
VISIBLE_STATES = {"visible", "partially_occluded"}
NO_TARGET_STATES = {"fully_occluded", "out_of_frame"}
LABEL_VISIBILITY = VISIBLE_STATES | NO_TARGET_STATES | {"unlabeled", "uncertain"}
MUTABLE_FIELDS = {
    "x",
    "y",
    "visibility",
    "occlusion",
    "quality",
    "radius",
    "label_source",
    "reviewed_at",
    "review_note",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(path)


def row_key(row: dict[str, Any]) -> str:
    return f"{row.get('clip_id')}::{int(row.get('frame_index'))}"


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:160]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def find_free_port(host: str, preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


class DenseReviewStore:
    def __init__(self, batch_dir: Path, labels_out: Path | None = None) -> None:
        self.batch_dir = Path(batch_dir).resolve()
        self.template_path = self.batch_dir / TEMPLATE_NAME
        self.labels_out = Path(labels_out or (self.batch_dir / REVIEWED_NAME)).resolve()
        self.lock = RLock()
        self.rows: list[dict[str, Any]] = []
        self.original: dict[str, dict[str, Any]] = {}
        self.index_by_key: dict[str, int] = {}
        self.load()

    def load(self) -> None:
        template_rows = read_jsonl(self.template_path)
        if not template_rows:
            raise FileNotFoundError(f"no dense trajectory rows found at {self.template_path}")

        saved = {row_key(row): row for row in read_jsonl(self.labels_out)}
        merged: list[dict[str, Any]] = []
        original: dict[str, dict[str, Any]] = {}
        index_by_key: dict[str, int] = {}

        for index, base in enumerate(template_rows):
            key = row_key(base)
            row = dict(base)
            original[key] = dict(base)
            if key in saved:
                for field in MUTABLE_FIELDS:
                    if field in saved[key]:
                        row[field] = saved[key][field]
            row["split"] = base.get("split")
            row["training_use"] = base.get("training_use")
            row["frame_image"] = base.get("frame_image")
            row["model_hints"] = base.get("model_hints", {})
            merged.append(row)
            index_by_key[key] = index

        self.rows = merged
        self.original = original
        self.index_by_key = index_by_key

    def save(self) -> None:
        write_jsonl(self.labels_out, self.rows)

    def summary(self) -> dict[str, Any]:
        reviewed = sum(1 for row in self.rows if row.get("quality") == "reviewed")
        trainable = sum(1 for row in self.rows if row.get("training_use") == "train_or_calibration")
        audit_only = sum(1 for row in self.rows if row.get("training_use") == "audit_only")
        clips: dict[str, dict[str, Any]] = {}
        for index, row in enumerate(self.rows):
            clip_id = str(row["clip_id"])
            clip = clips.setdefault(
                clip_id,
                {
                    "clip_id": clip_id,
                    "source_video": row.get("source_video"),
                    "split": row.get("split"),
                    "training_use": row.get("training_use"),
                    "first_index": index,
                    "count": 0,
                    "reviewed": 0,
                },
            )
            clip["count"] += 1
            if row.get("quality") == "reviewed":
                clip["reviewed"] += 1
        return {
            "rows": len(self.rows),
            "reviewed": reviewed,
            "pending": len(self.rows) - reviewed,
            "train_or_calibration": trainable,
            "audit_only": audit_only,
            "clips": sorted(clips.values(), key=lambda item: (item["split"], item["source_video"], item["clip_id"])),
        }

    def state(self) -> dict[str, Any]:
        with self.lock:
            return {
                "batch_dir": str(self.batch_dir),
                "template": str(self.template_path),
                "labels_out": str(self.labels_out),
                "rows": self.rows,
                "summary": self.summary(),
            }

    def update_label(self, key: str, patch: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if key not in self.index_by_key:
                raise KeyError(f"unknown row key: {key}")
            row = dict(self.rows[self.index_by_key[key]])
            base = self.original[key]

            if any(field in patch for field in ("split", "training_use", "frame_image", "model_hints")):
                raise ValueError("split, training_use, frame_image, and model_hints are locked")

            clean: dict[str, Any] = {}
            for field, value in patch.items():
                if field not in MUTABLE_FIELDS:
                    continue
                clean[field] = value

            visibility = clean.get("visibility", row.get("visibility", "unlabeled"))
            quality = clean.get("quality", row.get("quality", "pending"))
            if visibility not in LABEL_VISIBILITY:
                raise ValueError(f"unsupported visibility: {visibility}")
            if quality not in {"pending", "reviewed"}:
                raise ValueError(f"unsupported quality: {quality}")

            row.update(clean)
            row["visibility"] = visibility
            row["quality"] = quality
            row["split"] = base.get("split")
            row["training_use"] = base.get("training_use")

            if row["visibility"] in NO_TARGET_STATES:
                row["x"] = None
                row["y"] = None
                row["occlusion"] = row["visibility"]
            elif row["visibility"] == "uncertain":
                row["quality"] = "pending"
            elif row["quality"] == "reviewed":
                if row["visibility"] not in VISIBLE_STATES:
                    raise ValueError("reviewed rows must be visible, partially_occluded, fully_occluded, or out_of_frame")
                row["x"] = float(row["x"])
                row["y"] = float(row["y"])
                row["reviewed_at"] = now_iso()

            if row.get("x") is not None:
                row["x"] = round(float(row["x"]), 3)
            if row.get("y") is not None:
                row["y"] = round(float(row["y"]), 3)

            row.setdefault("label_source", "human_dense_review")
            self.rows[self.index_by_key[key]] = row
            self.save()
            return row


class DenseReviewHandler(BaseHTTPRequestHandler):
    server_version = "HackyDenseReview/1.0"

    @property
    def store(self) -> DenseReviewStore:
        return self.server.store  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str, content_type: str = "text/html; charset=utf-8") -> None:
        body = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status: HTTPStatus, message: str) -> None:
        self.send_json({"error": message}, status=status)

    def read_body_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_text(APP_HTML)
            return
        if parsed.path == "/api/state":
            self.send_json(self.store.state())
            return
        if parsed.path == "/frame":
            qs = parse_qs(parsed.query)
            rel = unquote(qs.get("path", [""])[0])
            self.serve_frame(rel)
            return
        self.send_error_json(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self.read_body_json()
            if parsed.path == "/api/label":
                key = str(payload.get("key", ""))
                patch = payload.get("patch", {})
                if not isinstance(patch, dict):
                    raise ValueError("patch must be an object")
                row = self.store.update_label(key, patch)
                self.send_json({"row": row, "summary": self.store.summary()})
                return
            if parsed.path == "/api/cotracker":
                self.send_json(self.run_cotracker(payload))
                return
            self.send_error_json(HTTPStatus.NOT_FOUND, "not found")
        except Exception as exc:  # noqa: BLE001 - local reviewer should surface exact failure
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

    def serve_frame(self, rel_path: str) -> None:
        if not rel_path:
            self.send_error_json(HTTPStatus.BAD_REQUEST, "missing frame path")
            return
        target = (self.store.batch_dir / rel_path).resolve()
        try:
            target.relative_to(self.store.batch_dir)
        except ValueError:
            self.send_error_json(HTTPStatus.FORBIDDEN, "frame path escapes batch directory")
            return
        if not target.exists():
            self.send_error_json(HTTPStatus.NOT_FOUND, f"missing frame: {rel_path}")
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "max-age=3600")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def run_cotracker(self, payload: dict[str, Any]) -> dict[str, Any]:
        clip_id = str(payload.get("clip_id", ""))
        frame_index = int(payload.get("frame_index"))
        x = float(payload.get("x"))
        y = float(payload.get("y"))
        resize_scale = float(payload.get("resize_scale", 0.35))
        device = str(payload.get("device", "cpu"))
        if not clip_id:
            raise ValueError("clip_id is required")

        script = ROOT / "cotracker_feasibility.py"
        if not script.exists():
            raise FileNotFoundError(f"missing {script}")

        out_dir = self.store.batch_dir / "cotracker_suggestions"
        name = f"app_{safe_slug(clip_id)}_f{frame_index}"
        cmd = [
            sys.executable,
            str(script),
            "--batch-dir",
            str(self.store.batch_dir),
            "--clip-id",
            clip_id,
            "--seed-frame",
            str(frame_index),
            "--x",
            str(x),
            "--y",
            str(y),
            "--name",
            name,
            "--out-dir",
            str(out_dir),
            "--resize-scale",
            str(resize_scale),
            "--device",
            device,
        ]
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=240)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout or "CoTracker failed").strip())

        point_path = out_dir / name / "propagated_points.jsonl"
        summary_path = out_dir / name / "summary.json"
        points = read_jsonl(point_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        return {
            "points": points,
            "summary": summary,
            "point_path": str(point_path),
            "stdout": proc.stdout[-4000:],
        }


APP_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hacky Track Dense Review</title>
  <style>
    :root {
      --bg: #0f141b;
      --panel: #151d27;
      --panel-2: #111821;
      --line: #2a3545;
      --text: #e7edf6;
      --muted: #93a1b3;
      --accent: #4ea1ff;
      --good: #37d67a;
      --warn: #ffb84d;
      --bad: #ff5d73;
      --cyan: #4fe0ff;
      --magenta: #e06cff;
      --amber: #ffd24d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: #0d131a;
    }
    h1 {
      margin: 0;
      font-size: 17px;
      font-weight: 700;
      letter-spacing: 0;
    }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 390px;
      min-height: calc(100vh - 57px);
    }
    .stage {
      min-width: 0;
      padding: 16px;
      display: grid;
      grid-template-rows: minmax(0, 1fr) auto;
      gap: 12px;
    }
    .viewer {
      position: relative;
      min-height: 360px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      background: #05080d;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    #loupe {
      position: absolute;
      top: 14px;
      left: 14px;
      width: 168px;
      height: 168px;
      border: 1px solid #59687b;
      border-radius: 8px;
      background: #05080d;
      box-shadow: 0 10px 28px rgba(0,0,0,.45);
      display: none;
      image-rendering: pixelated;
      z-index: 5;
    }
    .frame-wrap {
      position: relative;
      display: inline-block;
      max-width: 100%;
      max-height: calc(100vh - 190px);
    }
    #frame {
      display: block;
      max-width: 100%;
      max-height: calc(100vh - 190px);
      object-fit: contain;
      user-select: none;
    }
    #overlay {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      cursor: crosshair;
    }
    aside {
      border-left: 1px solid var(--line);
      background: var(--panel);
      min-width: 0;
      overflow: auto;
    }
    section {
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
    }
    .section-title {
      margin: 0 0 10px;
      color: #b8c4d6;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: .14em;
      text-transform: uppercase;
    }
    .row {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 8px;
    }
    .row.wrap { flex-wrap: wrap; }
    .row > label { min-width: 74px; color: var(--muted); }
    select, input, button {
      border: 1px solid var(--line);
      background: #1b2532;
      color: var(--text);
      border-radius: 6px;
      padding: 8px 10px;
      font: inherit;
    }
    select { max-width: 100%; min-width: 0; }
    button {
      cursor: pointer;
      font-weight: 650;
    }
    button:hover { border-color: #49617e; background: #223047; }
    button.primary { border-color: #2c6fb7; background: #1f5d99; }
    button.good { border-color: #238d52; background: #1f7146; }
    button.warn { border-color: #9e6b24; background: #704b18; }
    button.active { outline: 2px solid var(--accent); outline-offset: 1px; }
    button:disabled { cursor: not-allowed; opacity: .55; }
    .statusbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      min-height: 34px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 3px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      background: #111821;
      white-space: nowrap;
      font-size: 12px;
      font-weight: 650;
    }
    .pill.good { color: #b9f8d0; border-color: #2c7c4d; background: #143522; }
    .pill.warn { color: #ffe1aa; border-color: #8f6429; background: #392918; }
    .pill.bad { color: #ffc4ce; border-color: #8d3343; background: #3a1820; }
    .pill.accent { color: #cae4ff; border-color: #2f6ea7; background: #17304a; }
    .meta {
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    .progress {
      width: 100%;
      height: 8px;
      border-radius: 99px;
      background: #0c1219;
      overflow: hidden;
      border: 1px solid var(--line);
    }
    .progress > div {
      height: 100%;
      background: linear-gradient(90deg, #2fa7ff, #35d27e);
      width: 0%;
    }
    .hint-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    .hint-table td {
      padding: 5px 0;
      border-bottom: 1px solid rgba(255,255,255,.06);
      color: var(--muted);
    }
    .hint-table tr { cursor: pointer; }
    .hint-table tr:hover td { color: var(--text); background: rgba(255,255,255,.04); }
    .hint-table td:first-child { color: var(--text); }
    .legend-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
      margin-right: 6px;
      vertical-align: -1px;
    }
    .error {
      color: #ffd0d7;
      background: #3b1821;
      border: 1px solid #853142;
      border-radius: 6px;
      padding: 8px 10px;
      margin-top: 8px;
      display: none;
    }
    .footer-help {
      color: var(--muted);
      font-size: 13px;
    }
    .wide { flex: 1; min-width: 0; }
    .frame-strip {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(14px, 1fr));
      gap: 4px;
      margin-top: 10px;
    }
    .frame-tick {
      height: 16px;
      min-width: 14px;
      border: 1px solid #2c394a;
      border-radius: 4px;
      background: #101822;
      padding: 0;
    }
    .frame-tick.reviewed { background: #1d7c4a; border-color: #34b66e; }
    .frame-tick.pending { background: #263247; }
    .frame-tick.current { outline: 2px solid #ffffff; outline-offset: 1px; }
    .frame-tick.audit { box-shadow: inset 0 -3px 0 var(--amber); }
    .mini-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 8px;
    }
    .stat-card {
      border: 1px solid var(--line);
      background: var(--panel-2);
      border-radius: 8px;
      padding: 8px 10px;
    }
    .stat-card strong {
      display: block;
      color: var(--text);
      font-size: 16px;
      line-height: 1.2;
    }
    .stat-card span {
      color: var(--muted);
      font-size: 12px;
    }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      aside { border-left: 0; border-top: 1px solid var(--line); }
      #frame, .frame-wrap { max-height: 58vh; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Hacky Track Dense Review</h1>
    <div id="headerMeta" class="meta">Loading...</div>
  </header>
  <main>
    <div class="stage">
      <div class="viewer">
        <canvas id="loupe" width="168" height="168"></canvas>
        <div class="frame-wrap">
          <img id="frame" alt="dense review frame">
          <canvas id="overlay"></canvas>
        </div>
      </div>
      <div class="statusbar">
        <button id="prevBtn">Prev</button>
        <button id="nextBtn">Next</button>
        <button id="nextPendingBtn">Next pending</button>
        <span id="frameStatus" class="pill"></span>
        <span id="qualityStatus" class="pill"></span>
        <span id="auditStatus" class="pill"></span>
        <span id="saveStatus" class="meta"></span>
      </div>
    </div>

    <aside>
      <section>
        <p class="section-title">Clip</p>
        <div class="row">
          <label for="clipSelect">Clip</label>
          <select id="clipSelect" class="wide"></select>
        </div>
        <div id="clipMeta" class="meta"></div>
        <div class="progress" title="reviewed rows in current clip"><div id="clipProgress"></div></div>
        <div class="mini-row">
          <div class="stat-card"><strong id="clipReviewedStat">0/0</strong><span>clip reviewed</span></div>
          <div class="stat-card"><strong id="globalReviewedStat">0/0</strong><span>batch reviewed</span></div>
        </div>
        <div id="frameStrip" class="frame-strip" aria-label="clip frame strip"></div>
      </section>

      <section>
        <p class="section-title">Label</p>
        <div class="row wrap">
          <button data-vis="visible">1 Visible</button>
          <button data-vis="partially_occluded">2 Partial</button>
          <button data-vis="fully_occluded">3 Occluded</button>
          <button data-vis="out_of_frame">4 Out</button>
          <button data-vis="uncertain">5 Uncertain</button>
        </div>
        <div class="row wrap">
          <button id="acceptBtn" class="good">Approve point</button>
          <button id="acceptNextBtn" class="primary">Approve + next</button>
          <button id="pendingBtn">Mark pending</button>
          <button id="clearBtn" class="warn">Clear point</button>
        </div>
        <div id="pointMeta" class="meta"></div>
        <div id="errorBox" class="error"></div>
      </section>

      <section>
        <p class="section-title">CoTracker Assist</p>
        <div class="row wrap">
          <button id="runCotrackerBtn" class="primary">Propagate from point</button>
          <button id="useSuggestionBtn">Use suggestion</button>
        </div>
        <div id="cotrackerMeta" class="meta">Click the ball on a clear frame, then propagate. Suggestions are overlays only until you accept them.</div>
      </section>

      <section>
        <p class="section-title">Model Hints</p>
        <table class="hint-table" id="hintTable"></table>
      </section>

      <section>
        <p class="section-title">Controls</p>
        <div class="footer-help">
          Click image to place the ball. Keys: j/k next/prev, arrows step frames,
          1-5 visibility, a approve current point, enter approve+next, p pending, n next pending.
          Test/audit rows can be reviewed for evaluation but remain audit_only and are excluded from training.
        </div>
      </section>
    </aside>
  </main>

  <script>
    const app = {
      rows: [],
      summary: null,
      index: 0,
      suggestions: {},
      saving: false,
    };
    const colors = {
      v10_greedy: "#4fe0ff",
      v10_temporal: "#ffd24d",
      v11_greedy: "#ff7ab6",
      label: "#37d67a",
      cotracker: "#e06cff",
    };
    const COTRACKER_AUTO_VISIBILITY = 0.25;
    const frame = document.getElementById("frame");
    const canvas = document.getElementById("overlay");
    const ctx = canvas.getContext("2d");
    const loupe = document.getElementById("loupe");
    const loupeCtx = loupe.getContext("2d");

    function keyOf(row) {
      return `${row.clip_id}::${row.frame_index}`;
    }

    function current() {
      return app.rows[app.index];
    }

    function currentSuggestion(row = current()) {
      return (app.suggestions[row.clip_id] || {})[row.frame_index] || null;
    }

    function suggestionIsAutoSafe(suggestion) {
      return suggestion && Number(suggestion.visibility_score || 0) >= COTRACKER_AUTO_VISIBILITY;
    }

    function clipRows(clipId) {
      return app.rows.filter((row) => row.clip_id === clipId);
    }

    function showError(message) {
      const box = document.getElementById("errorBox");
      box.textContent = message || "";
      box.style.display = message ? "block" : "none";
    }

    function setSaveStatus(message) {
      document.getElementById("saveStatus").textContent = message || "";
    }

    function frameUrl(row) {
      return `/frame?path=${encodeURIComponent(row.frame_image)}&t=${encodeURIComponent(row.quality || "pending")}`;
    }

    function resizeOverlay() {
      const rect = frame.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawOverlay();
    }

    function toScreen(x, y) {
      const rect = canvas.getBoundingClientRect();
      const sx = rect.width / (frame.naturalWidth || 1);
      const sy = rect.height / (frame.naturalHeight || 1);
      return [x * sx, y * sy];
    }

    function drawCircle(x, y, color, radius, label) {
      const [sx, sy] = toScreen(x, y);
      ctx.save();
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(sx, sy, radius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(sx - radius - 4, sy);
      ctx.lineTo(sx + radius + 4, sy);
      ctx.moveTo(sx, sy - radius - 4);
      ctx.lineTo(sx, sy + radius + 4);
      ctx.stroke();
      if (label) {
        ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
        ctx.fillText(label, sx + radius + 6, sy - radius - 2);
      }
      ctx.restore();
    }

    function drawLoupe(clientX, clientY) {
      if (!frame.naturalWidth || !frame.naturalHeight) return;
      const rect = canvas.getBoundingClientRect();
      const x = ((clientX - rect.left) / rect.width) * frame.naturalWidth;
      const y = ((clientY - rect.top) / rect.height) * frame.naturalHeight;
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;
      const crop = 56;
      const sx = Math.max(0, Math.min(frame.naturalWidth - crop, x - crop / 2));
      const sy = Math.max(0, Math.min(frame.naturalHeight - crop, y - crop / 2));
      loupeCtx.imageSmoothingEnabled = false;
      loupeCtx.clearRect(0, 0, loupe.width, loupe.height);
      loupeCtx.drawImage(frame, sx, sy, crop, crop, 0, 0, loupe.width, loupe.height);
      loupeCtx.strokeStyle = "#37d67a";
      loupeCtx.lineWidth = 1;
      loupeCtx.beginPath();
      loupeCtx.moveTo(loupe.width / 2 - 14, loupe.height / 2);
      loupeCtx.lineTo(loupe.width / 2 + 14, loupe.height / 2);
      loupeCtx.moveTo(loupe.width / 2, loupe.height / 2 - 14);
      loupeCtx.lineTo(loupe.width / 2, loupe.height / 2 + 14);
      loupeCtx.stroke();
      loupeCtx.fillStyle = "rgba(5,8,13,.72)";
      loupeCtx.fillRect(0, loupe.height - 22, loupe.width, 22);
      loupeCtx.fillStyle = "#e7edf6";
      loupeCtx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      loupeCtx.fillText(`x=${x.toFixed(1)} y=${y.toFixed(1)}`, 8, loupe.height - 7);
      loupe.style.display = "block";
    }

    function drawOverlay() {
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      const row = current();
      if (!row || !frame.naturalWidth) return;

      const hints = row.model_hints || {};
      Object.keys(hints).forEach((name) => {
        const hint = hints[name];
        if (hint && Number.isFinite(hint.x) && Number.isFinite(hint.y)) {
          drawCircle(hint.x, hint.y, colors[name] || "#ffffff", 6, name.replace("_", " "));
        }
      });

      const clipSuggestions = app.suggestions[row.clip_id] || {};
      const suggestion = clipSuggestions[row.frame_index];
      if (suggestion && Number.isFinite(suggestion.x) && Number.isFinite(suggestion.y)) {
        drawCircle(suggestion.x, suggestion.y, colors.cotracker, 9, `cot ${Number(suggestion.visibility_score || 0).toFixed(2)}`);
      }

      if (Number.isFinite(row.x) && Number.isFinite(row.y)) {
        drawCircle(row.x, row.y, colors.label, 11, "label");
      }
    }

    function render() {
      const row = current();
      if (!row) return;
      showError("");
      frame.src = frameUrl(row);

      const clips = app.summary.clips;
      const clipSelect = document.getElementById("clipSelect");
      if (clipSelect.options.length !== clips.length) {
        clipSelect.innerHTML = "";
        clips.forEach((clip) => {
          const option = document.createElement("option");
          option.value = clip.clip_id;
          option.textContent = `${clip.source_video} / ${clip.split} / ${clip.reviewed}/${clip.count}`;
          clipSelect.appendChild(option);
        });
      }
      clipSelect.value = row.clip_id;

      const rowsForClip = clipRows(row.clip_id);
      const clipIndex = rowsForClip.findIndex((item) => item.frame_index === row.frame_index);
      const reviewedInClip = rowsForClip.filter((item) => item.quality === "reviewed").length;
      const pct = rowsForClip.length ? Math.round((reviewedInClip / rowsForClip.length) * 100) : 0;
      document.getElementById("clipProgress").style.width = `${pct}%`;
      document.getElementById("clipReviewedStat").textContent = `${reviewedInClip}/${rowsForClip.length}`;
      document.getElementById("globalReviewedStat").textContent = `${app.summary.reviewed}/${app.summary.rows}`;
      document.getElementById("clipMeta").textContent =
        `${row.source_video} | ${row.split} | ${row.training_use} | clip frame ${clipIndex + 1}/${rowsForClip.length}`;

      document.getElementById("headerMeta").textContent =
        `${app.summary.reviewed}/${app.summary.rows} reviewed | ${app.summary.train_or_calibration} train/calibration | ${app.summary.audit_only} audit-only`;
      document.getElementById("frameStatus").textContent =
        `frame ${row.frame_index} | ${Number(row.time_sec).toFixed(3)}s`;
      const q = document.getElementById("qualityStatus");
      q.textContent = `${row.quality} / ${row.visibility}`;
      q.className = `pill ${row.quality === "reviewed" ? "good" : "accent"}`;
      const audit = document.getElementById("auditStatus");
      audit.textContent = row.training_use === "audit_only" ? "audit_only locked" : "train_or_calibration";
      audit.className = `pill ${row.training_use === "audit_only" ? "warn" : "good"}`;
      document.getElementById("pointMeta").textContent =
        Number.isFinite(row.x) && Number.isFinite(row.y)
          ? `x=${Number(row.x).toFixed(1)}, y=${Number(row.y).toFixed(1)}, radius=${row.radius}`
          : "No point set";

      renderHints(row);
      renderFrameStrip(rowsForClip, row);
      renderCotrackerMeta(row);
      drawOverlay();
    }

    function renderFrameStrip(rowsForClip, row) {
      const strip = document.getElementById("frameStrip");
      strip.innerHTML = "";
      rowsForClip.forEach((item) => {
        const tick = document.createElement("button");
        tick.className = [
          "frame-tick",
          item.quality === "reviewed" ? "reviewed" : "pending",
          item.frame_index === row.frame_index ? "current" : "",
          item.training_use === "audit_only" ? "audit" : "",
        ].filter(Boolean).join(" ");
        tick.title = `frame ${item.frame_index} | ${item.quality} / ${item.visibility}`;
        tick.addEventListener("click", () => {
          const index = app.rows.findIndex((candidate) => candidate.clip_id === item.clip_id && candidate.frame_index === item.frame_index);
          if (index >= 0) setIndex(index);
        });
        strip.appendChild(tick);
      });
    }

    function renderHints(row) {
      const table = document.getElementById("hintTable");
      table.innerHTML = "";
      const hints = row.model_hints || {};
      Object.keys(hints).sort().forEach((name) => {
        const hint = hints[name];
        const tr = document.createElement("tr");
        tr.title = `Use ${name} as the label point`;
        tr.addEventListener("click", () => useHint(name));
        const label = document.createElement("td");
        label.innerHTML = `<span class="legend-dot" style="background:${colors[name] || "#fff"}"></span>${name}`;
        const xy = document.createElement("td");
        xy.textContent = Number.isFinite(hint.x) ? `${Number(hint.x).toFixed(1)}, ${Number(hint.y).toFixed(1)}` : "-";
        const conf = document.createElement("td");
        conf.textContent = hint.confidence == null ? "" : `conf ${Number(hint.confidence).toFixed(3)}`;
        tr.appendChild(label);
        tr.appendChild(xy);
        tr.appendChild(conf);
        table.appendChild(tr);
      });
    }

    function renderCotrackerMeta(row) {
      const clipSuggestions = app.suggestions[row.clip_id] || {};
      const suggestion = currentSuggestion(row);
      const button = document.getElementById("useSuggestionBtn");
      if (!Object.keys(clipSuggestions).length) {
        button.textContent = "Use suggestion";
        button.className = "";
        button.disabled = true;
        document.getElementById("cotrackerMeta").textContent =
          "Click the ball on a clear frame, then propagate. Suggestions are overlays only until you accept them.";
        return;
      }
      if (suggestion) {
        const score = Number(suggestion.visibility_score || 0);
        const safe = suggestionIsAutoSafe(suggestion);
        const hasPoint = Number.isFinite(row.x) && Number.isFinite(row.y);
        button.textContent = safe ? "Use suggestion + next" : "Use low-vis suggestion";
        button.className = safe ? "good active" : "warn";
        button.disabled = false;
        if (safe && hasPoint) {
          document.getElementById("cotrackerMeta").textContent =
            `CoTracker is available, but Enter approves the current green point. Click Use suggestion + next only if the purple point is better. visibility=${score.toFixed(2)}`;
        } else if (safe) {
          document.getElementById("cotrackerMeta").textContent =
            `Default: Enter uses CoTracker, marks reviewed, and advances. x=${Number(suggestion.x).toFixed(1)}, y=${Number(suggestion.y).toFixed(1)}, visibility=${score.toFixed(2)}`;
        } else {
          document.getElementById("cotrackerMeta").textContent =
            `Low CoTracker visibility (${score.toFixed(2)}). Enter approves your current point if one exists; click the yellow button only if the purple point is actually right.`;
        }
      } else {
        button.textContent = "Use suggestion";
        button.className = "";
        button.disabled = true;
        document.getElementById("cotrackerMeta").textContent = "CoTracker suggestions loaded for this clip; none on this frame.";
      }
    }

    async function savePatch(patch) {
      const row = current();
      setSaveStatus("saving...");
      const response = await fetch("/api/label", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key: keyOf(row), patch }),
      });
      const data = await response.json();
      if (!response.ok) {
        showError(data.error || "save failed");
        setSaveStatus("save failed");
        return false;
      }
      app.rows[app.index] = data.row;
      app.summary = data.summary;
      setSaveStatus(`saved ${new Date().toLocaleTimeString()}`);
      render();
      return true;
    }

    function setIndex(index) {
      app.index = Math.max(0, Math.min(app.rows.length - 1, index));
      render();
    }

    function step(delta) {
      setIndex(app.index + delta);
    }

    async function acceptAndNext() {
      const row = current();
      const hasPoint = Number.isFinite(row.x) && Number.isFinite(row.y);
      const suggestion = currentSuggestion();
      if (!hasPoint && suggestionIsAutoSafe(suggestion)) {
        return useSuggestion({ review: true, advance: true });
      }
      return approveExistingPoint({ advance: true });
    }

    function nextPending() {
      for (let i = app.index + 1; i < app.rows.length; i += 1) {
        if (app.rows[i].quality !== "reviewed") return setIndex(i);
      }
      for (let i = 0; i < app.index; i += 1) {
        if (app.rows[i].quality !== "reviewed") return setIndex(i);
      }
    }

    function setVisibility(visibility) {
      const patch = { visibility, label_source: "human_dense_review" };
      if (visibility === "fully_occluded" || visibility === "out_of_frame") {
        patch.x = null;
        patch.y = null;
        patch.occlusion = visibility;
      } else if (visibility === "uncertain") {
        patch.quality = "pending";
        patch.occlusion = "unknown";
      } else {
        patch.occlusion = visibility === "partially_occluded" ? "partial" : "none";
      }
      savePatch(patch);
    }

    async function approveExistingPoint(options = {}) {
      const row = current();
      const hasPoint = Number.isFinite(row.x) && Number.isFinite(row.y);
      if (hasPoint) {
        const visibility = row.visibility === "partially_occluded" ? "partially_occluded" : "visible";
        const ok = await savePatch({
          visibility,
          occlusion: visibility === "partially_occluded" ? "partial" : "none",
          quality: "reviewed",
          label_source: row.label_source || "human_dense_review",
        });
        if (ok && options.advance) step(1);
        return ok;
      }
      if (row.visibility === "fully_occluded" || row.visibility === "out_of_frame") {
        const ok = await savePatch({ quality: "reviewed", label_source: "human_dense_review" });
        if (ok && options.advance) step(1);
        return ok;
      }
      const suggestion = currentSuggestion();
      if (suggestionIsAutoSafe(suggestion)) {
        return useSuggestion({ review: true, advance: Boolean(options.advance) });
      }
      if (suggestion) {
        showError("The visible CoTracker point is low-confidence. Click it only if it is right, or place your own point, then approve.");
        return false;
      }
      showError("No point to approve. Click the ball, use a hint, or mark it occluded/out.");
      return false;
    }

    function acceptReviewed() {
      return approveExistingPoint({ advance: false });
    }

    function markPending() {
      savePatch({ quality: "pending" });
    }

    function clearPoint() {
      savePatch({ x: null, y: null, visibility: "unlabeled", occlusion: "unknown", quality: "pending" });
    }

    function useHint(name) {
      const hint = (current().model_hints || {})[name];
      if (!hint || !Number.isFinite(hint.x) || !Number.isFinite(hint.y)) {
        showError(`No usable point for ${name}.`);
        return;
      }
      savePatch({
        x: hint.x,
        y: hint.y,
        visibility: current().visibility === "partially_occluded" ? "partially_occluded" : "visible",
        occlusion: current().visibility === "partially_occluded" ? "partial" : "none",
        label_source: `human_dense_review+${name}_hint`,
      });
    }

    function canvasClick(event) {
      const rect = canvas.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width) * frame.naturalWidth;
      const y = ((event.clientY - rect.top) / rect.height) * frame.naturalHeight;
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;
      const row = current();
      savePatch({
        x,
        y,
        visibility: row.visibility === "partially_occluded" ? "partially_occluded" : "visible",
        occlusion: row.visibility === "partially_occluded" ? "partial" : "none",
        label_source: "human_dense_review",
      });
    }

    async function runCotracker() {
      const row = current();
      if (!Number.isFinite(row.x) || !Number.isFinite(row.y)) {
        showError("Place a seed point before running CoTracker.");
        return;
      }
      const button = document.getElementById("runCotrackerBtn");
      button.disabled = true;
      button.textContent = "Running...";
      document.getElementById("cotrackerMeta").textContent = "Running CoTracker on this clip. CPU runs can take a minute.";
      try {
        const response = await fetch("/api/cotracker", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            clip_id: row.clip_id,
            frame_index: row.frame_index,
            x: row.x,
            y: row.y,
            resize_scale: 0.35,
            device: "cpu",
          }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "CoTracker failed");
        const byFrame = {};
        data.points.forEach((point) => { byFrame[point.frame_index] = point; });
        app.suggestions[row.clip_id] = byFrame;
        document.getElementById("cotrackerMeta").textContent =
          `Loaded ${data.points.length} suggestions. Enter approves the current point; if no point is set, it uses a safe CoTracker suggestion.`;
        render();
        drawOverlay();
      } catch (error) {
        showError(error.message || String(error));
      } finally {
        button.disabled = false;
        button.textContent = "Propagate from point";
      }
    }

    async function useSuggestion(options = {}) {
      const row = current();
      const suggestion = currentSuggestion(row);
      if (!suggestion) {
        showError("No CoTracker suggestion on this frame.");
        return false;
      }
      const visibility = suggestionIsAutoSafe(suggestion) ? "visible" : "uncertain";
      const patch = {
        x: suggestion.x,
        y: suggestion.y,
        visibility,
        occlusion: visibility === "visible" ? "none" : "unknown",
        label_source: "human_dense_review+cotracker_assist",
      };
      if (options.review && visibility === "visible") {
        patch.quality = "reviewed";
      }
      const ok = await savePatch(patch);
      if (!ok) return false;
      if (options.review && visibility !== "visible") {
        showError("CoTracker visibility is low on this frame. Point was placed as uncertain; review manually.");
        return false;
      }
      if (options.advance) step(1);
      return true;
    }

    async function loadState() {
      const response = await fetch("/api/state");
      const data = await response.json();
      app.rows = data.rows;
      app.summary = data.summary;
      const firstPending = app.rows.findIndex((row) => row.quality !== "reviewed");
      app.index = firstPending >= 0 ? firstPending : 0;
      render();
    }

    frame.addEventListener("load", resizeOverlay);
    window.addEventListener("resize", resizeOverlay);
    canvas.addEventListener("click", canvasClick);
    canvas.addEventListener("mousemove", (event) => drawLoupe(event.clientX, event.clientY));
    canvas.addEventListener("mouseleave", () => { loupe.style.display = "none"; });
    document.getElementById("prevBtn").addEventListener("click", () => step(-1));
    document.getElementById("nextBtn").addEventListener("click", () => step(1));
    document.getElementById("nextPendingBtn").addEventListener("click", nextPending);
    document.getElementById("acceptBtn").addEventListener("click", acceptReviewed);
    document.getElementById("acceptNextBtn").addEventListener("click", acceptAndNext);
    document.getElementById("pendingBtn").addEventListener("click", markPending);
    document.getElementById("clearBtn").addEventListener("click", clearPoint);
    document.getElementById("runCotrackerBtn").addEventListener("click", runCotracker);
    document.getElementById("useSuggestionBtn").addEventListener("click", () => useSuggestion({ review: true, advance: true }));
    document.querySelectorAll("button[data-vis]").forEach((button) => {
      button.addEventListener("click", () => setVisibility(button.dataset.vis));
    });
    document.getElementById("clipSelect").addEventListener("change", (event) => {
      const clipId = event.target.value;
      const index = app.rows.findIndex((row) => row.clip_id === clipId);
      if (index >= 0) setIndex(index);
    });
    window.addEventListener("keydown", (event) => {
      if (event.target && ["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) return;
      if (event.key === "j" || event.key === "ArrowRight") step(1);
      if (event.key === "k" || event.key === "ArrowLeft") step(-1);
      if (event.key === "n") nextPending();
      if (event.key === "a") acceptReviewed();
      if (event.key === "Enter") acceptAndNext();
      if (event.key === "p") markPending();
      if (event.key === "1") setVisibility("visible");
      if (event.key === "2") setVisibility("partially_occluded");
      if (event.key === "3") setVisibility("fully_occluded");
      if (event.key === "4") setVisibility("out_of_frame");
      if (event.key === "5") setVisibility("uncertain");
    });

    loadState().catch((error) => showError(error.message || String(error)));
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the dense trajectory review web app")
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--labels-out", type=Path, help="Reviewed JSONL output path; defaults inside batch dir")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8891)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = DenseReviewStore(args.batch_dir, args.labels_out)
    port = find_free_port(args.host, args.port)
    server = ThreadingHTTPServer((args.host, port), DenseReviewHandler)
    server.store = store  # type: ignore[attr-defined]
    url = f"http://{args.host}:{port}/"
    print(f"dense review app: {url}")
    print(f"batch: {store.batch_dir}")
    print(f"labels: {store.labels_out}")
    print(f"rows: {len(store.rows)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
