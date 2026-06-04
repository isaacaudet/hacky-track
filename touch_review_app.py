#!/usr/bin/env python3
"""Muted visual touch review app.

This is a lean event-level reviewer for Phase 1. It shows each video muted by
default, overlays loose candidate hints, and saves only human visual decisions
as event JSON. Audio candidates are hints, not labels.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import socket
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import parse_qs, urlparse

from touch_label_readiness import classify_video, clustered_review_hints, count_checked_hints, make_next_clips


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "runs/release-27-public/touch_corpus_v1"
DEFAULT_REVIEW_MANIFEST = DEFAULT_CORPUS / "touch_review_manifest.json"
DEFAULT_CANDIDATES_DIR = DEFAULT_CORPUS / "audio_candidates"
DEFAULT_LABELS_DIR = DEFAULT_CORPUS / "visual_touch_labels"
DEFAULT_CONTACT_SHEETS_DIR = DEFAULT_CORPUS / "touch_review_contact_sheets"
DEFAULT_MONTAGES_DIR = DEFAULT_CORPUS / "touch_review_montages"
DEFAULT_TRACK_ROOTS = [
    ROOT / "runs/release-27-public/detector_inference_v10_calibrated_batch_300_processed_temporal",
    ROOT / "runs/release-27-public/patch_detector_inference_v5_temporal_27_smoke",
]
EVENT_TYPES = {"touch", "drop_floor", "stall"}
CANDIDATE_CLUSTER_GAP_SEC = 0.04
CONTACT_SIDES = {"unknown", "left", "right", "center"}
CONTACT_TYPES = {"unknown", "kick", "foot", "knee", "stall", "drop_floor", "ground", "chest", "hand"}
CONTACT_SURFACES = {"unknown", "inner", "outer"}
CONTACT_SIDE_BASES = {"unknown", "wearer_limb", "screen_position", "pose_anatomical", "ambiguous", "legacy_unspecified"}
TRICK_LABELS = {
    "",
    "right_kick",
    "left_kick",
    "left_knee",
    "right_knee",
    "left_inner_kick",
    "left_outer_kick",
    "right_inner_kick",
    "right_outer_kick",
    "left_inner_knee",
    "left_outer_knee",
    "right_inner_knee",
    "right_outer_knee",
    "right_stall",
    "left_stall",
    "left_inner_stall",
    "left_outer_stall",
    "right_inner_stall",
    "right_outer_stall",
    "knee",
    "clipper",
    "inner_left",
    "inner_right",
    "outer_left",
    "outer_right",
    "around_the_world",
    "around_the_world_outer_right",
    "around_the_world_outer_left",
}
CONTACT_REVIEW_STATUSES = {"unreviewed", "reviewed"}


def side_from_trick_label(trick_label: str | None) -> str | None:
    raw = str(trick_label or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in {"l", "left"} or raw.startswith("left_") or raw.endswith("_left") or "_left_" in raw:
        return "left"
    if raw in {"r", "right"} or raw.startswith("right_") or raw.endswith("_right") or "_right_" in raw:
        return "right"
    return None


def infer_contact_side_basis(contact_side: str, trick_label: str | None, explicit_basis: str | None = None) -> str:
    if explicit_basis:
        return explicit_basis
    trick_side = side_from_trick_label(trick_label)
    if contact_side in {"left", "right"} and trick_side == contact_side:
        return "wearer_limb"
    if contact_side in {"left", "right"}:
        return "legacy_unspecified"
    return "unknown"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


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


class TouchReviewStore:
    def __init__(
        self,
        review_manifest: Path,
        candidates_dir: Path,
        labels_dir: Path,
        contact_sheets_dir: Path = DEFAULT_CONTACT_SHEETS_DIR,
        montages_dir: Path = DEFAULT_MONTAGES_DIR,
        track_roots: list[Path] | tuple[Path, ...] = tuple(DEFAULT_TRACK_ROOTS),
    ) -> None:
        self.review_manifest = review_manifest.resolve()
        self.candidates_dir = candidates_dir.resolve()
        self.labels_dir = labels_dir.resolve()
        self.contact_sheets_dir = contact_sheets_dir.resolve()
        self.montages_dir = montages_dir.resolve()
        self.track_roots = [path.resolve() for path in track_roots]
        self.lock = RLock()
        self.manifest = read_json(self.review_manifest)
        self.items = list(self.manifest.get("items", []))
        self.by_id = {str(item["video_id"]): item for item in self.items}
        self.contact_sheet_manifest = self.load_contact_sheet_manifest()
        self.montage_manifest = self.load_montage_manifest()
        self.labels_dir.mkdir(parents=True, exist_ok=True)

    def label_path(self, video_id: str) -> Path:
        return self.labels_dir / f"{safe_slug(video_id)}.events.json"

    def candidate_path(self, video_id: str) -> Path:
        return self.candidates_dir / f"{safe_slug(video_id)}.touch_candidates.json"

    def load_contact_sheet_manifest(self) -> dict[str, list[dict[str, Any]]]:
        manifest_path = self.contact_sheets_dir / "touch_review_contact_sheets.json"
        if not manifest_path.exists():
            return {}
        doc = read_json(manifest_path)
        by_video: dict[str, list[dict[str, Any]]] = {}
        for row in doc.get("videos", []):
            by_video[str(row.get("video_id"))] = list(row.get("sheets", []))
        return by_video

    def load_montage_manifest(self) -> dict[str, list[dict[str, Any]]]:
        manifest_path = self.montages_dir / "touch_review_montages.json"
        if not manifest_path.exists():
            return {}
        doc = read_json(manifest_path)
        by_video: dict[str, list[dict[str, Any]]] = {}
        for row in doc.get("videos", []):
            by_video[str(row.get("video_id"))] = list(row.get("montages", []))
        return by_video

    def load_events(self, video_id: str) -> dict[str, Any]:
        path = self.label_path(video_id)
        if path.exists():
            return read_json(path)
        item = self.by_id[video_id]
        existing_path = item.get("existing_events_path")
        events: list[dict[str, Any]] = []
        if existing_path and Path(existing_path).exists():
            doc = read_json(Path(existing_path))
            for rally in doc.get("rallies", []):
                for event in rally.get("events", []):
                    if event.get("type") in EVENT_TYPES and event.get("time_sec") is not None:
                        row = {
                            "type": event["type"],
                            "time_sec": round(float(event["time_sec"]), 3),
                            "review_status": "needs_muted_visual_confirmation",
                            "source": "existing_event_hint",
                        }
                        if event.get("duration_sec") is not None:
                            row["duration_sec"] = float(event["duration_sec"])
                        if event.get("contact_side") not in (None, ""):
                            row["contact_side"] = str(event["contact_side"])
                        if event.get("contact_type") not in (None, ""):
                            row["contact_type"] = str(event["contact_type"])
                        if event.get("contact_surface") not in (None, ""):
                            row["contact_surface"] = str(event["contact_surface"])
                        if event.get("trick_label") not in (None, ""):
                            row["trick_label"] = str(event["trick_label"])
                        if event.get("contact_review_status") not in (None, ""):
                            row["contact_review_status"] = str(event["contact_review_status"])
                        row["contact_side_basis"] = infer_contact_side_basis(
                            str(row.get("contact_side") or "unknown"),
                            str(row.get("trick_label") or ""),
                            None if event.get("contact_side_basis") in (None, "") else str(event["contact_side_basis"]),
                        )
                        events.append(row)
        return self.empty_doc(video_id, events)

    def empty_doc(self, video_id: str, events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        item = self.by_id[video_id]
        return {
            "schema_version": 1,
            "source_video": Path(item["video_path"]).name,
            "source_video_path": item["video_path"],
            "split": item["split"],
            "annotation_method": "muted_visual_touch_review",
            "audio_muted_during_review_required": True,
            "candidate_review_complete": False,
            "review_status": "in_progress",
            "review_completed_at": None,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "review_note": (
                "Events in this file are ground truth only after review_status=approved. "
                "Audio/candidate hints are not labels. Candidate rows are used for training "
                "only after candidate_review_complete=true."
            ),
            "candidate_reviews": [],
            "rallies": [
                {
                    "id": 1,
                    "label": "full-video visual review",
                    "start_sec": 0.0,
                    "end_sec": None,
                    "events": sorted(events or [], key=lambda row: float(row["time_sec"])),
                }
            ],
        }

    def load_candidates(self, video_id: str) -> dict[str, Any]:
        path = self.candidate_path(video_id)
        if path.exists():
            return read_json(path)
        return {"audio_candidates": [], "existing_event_hints": []}

    def detector_track_path(self, video_id: str) -> Path | None:
        item = self.by_id[video_id]
        stem = Path(str(item["video_path"])).stem
        candidates = [stem, safe_slug(stem), safe_slug(video_id), str(video_id)]
        seen: set[Path] = set()
        for root in self.track_roots:
            for name in candidates:
                path = (root / name / "detector_track.json").resolve()
                if path in seen:
                    continue
                seen.add(path)
                try:
                    path.relative_to(root)
                except ValueError:
                    continue
                if path.exists():
                    return path
        return None

    def load_track_graph(self, video_id: str) -> dict[str, Any]:
        path = self.detector_track_path(video_id)
        if path is None:
            return {"source": None, "track_points": [], "status": "missing_track_file"}
        doc = read_json(path)
        points = []
        for point in doc.get("track", []):
            center = point.get("center")
            if isinstance(center, list) and len(center) >= 2:
                x, y = center[0], center[1]
            else:
                x, y = point.get("x"), point.get("y")
            if point.get("time_sec") is None or x is None or y is None:
                continue
            points.append(
                {
                    "time_sec": round(float(point["time_sec"]), 6),
                    "frame_index": point.get("frame_index"),
                    "x": round(float(x), 3),
                    "y": round(float(y), 3),
                    "confidence": None if point.get("confidence") is None else round(float(point["confidence"]), 6),
                    "source": point.get("source"),
                }
            )
        points.sort(key=lambda row: float(row["time_sec"]))
        return {
            "source": str(path),
            "track_points": points,
            "status": "ok" if points else "empty_track",
        }

    def contact_sheet_path(self, video_id: str, name: str) -> Path:
        if Path(name).name != name or not name.endswith(".png"):
            raise ValueError(f"invalid contact sheet name: {name}")
        path = (self.contact_sheets_dir / safe_slug(video_id) / name).resolve()
        root = self.contact_sheets_dir.resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"contact sheet escapes sheet directory: {name}") from exc
        return path

    def montage_path(self, video_id: str, name: str) -> Path:
        if Path(name).name != name or not name.endswith(".mp4"):
            raise ValueError(f"invalid montage name: {name}")
        path = (self.montages_dir / safe_slug(video_id) / name).resolve()
        root = self.montages_dir.resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"montage escapes montage directory: {name}") from exc
        return path

    def contact_sheets(self, video_id: str) -> list[dict[str, Any]]:
        video_dir = self.contact_sheets_dir / safe_slug(video_id)
        if not video_dir.exists():
            return []
        manifest_by_name = {
            Path(str(row.get("path") or "")).name: row
            for row in self.contact_sheet_manifest.get(video_id, [])
            if row.get("path")
        }
        rows = []
        for path in sorted(video_dir.glob("*.png")):
            name = path.name
            manifest_row = manifest_by_name.get(name, {})
            if "_likely_unchecked_" in name:
                filter_name = "likely_unchecked"
                label = "Likely/model"
            elif "_audio_only_" in name:
                filter_name = "audio_only"
                label = "Audio-tail"
            else:
                filter_name = "other"
                label = "Contact sheet"
            rows.append(
                {
                    "name": name,
                    "filter": filter_name,
                    "label": label,
                    "url": f"/contact-sheet?id={video_id}&name={name}",
                    "path": str(path),
                    "candidate_count": int(manifest_row.get("candidate_count") or 0),
                    "times_sec": list(manifest_row.get("times_sec") or []),
                }
            )
        return rows

    def review_montages(self, video_id: str) -> list[dict[str, Any]]:
        video_dir = self.montages_dir / safe_slug(video_id)
        if not video_dir.exists():
            return []
        manifest_by_name = {
            Path(str(row.get("path") or "")).name: row
            for row in self.montage_manifest.get(video_id, [])
            if row.get("path")
        }
        rows = []
        for path in sorted(video_dir.glob("*.mp4")):
            name = path.name
            manifest_row = manifest_by_name.get(name, {})
            if "_likely_unchecked_" in name:
                filter_name = "likely_unchecked"
                label = "Likely/model"
            elif "_audio_only_" in name:
                filter_name = "audio_only"
                label = "Audio-tail"
            else:
                filter_name = "other"
                label = "Montage"
            rows.append(
                {
                    "name": name,
                    "filter": filter_name,
                    "label": label,
                    "url": f"/review-montage?id={video_id}&name={name}",
                    "path": str(path),
                    "candidate_count": int(manifest_row.get("candidate_count") or 0),
                    "duration_sec": manifest_row.get("duration_sec"),
                    "times_sec": list(manifest_row.get("times_sec") or []),
                }
            )
        return rows

    def require_complete_candidate_reviews(self, video_id: str, reviews: list[dict[str, Any]]) -> None:
        candidates_path = self.candidate_path(video_id)
        if not candidates_path.exists():
            raise ValueError(f"cannot complete: missing candidate file {candidates_path}")
        hints = clustered_review_hints(read_json(candidates_path), cluster_gap_sec=CANDIDATE_CLUSTER_GAP_SEC)
        checked = count_checked_hints(hints, reviews, tolerance_sec=0.05)
        missing = len(hints) - checked
        if missing:
            raise ValueError(f"cannot complete: {missing} unchecked hints")

    @staticmethod
    def contact_side_basis_stats(events: list[dict[str, Any]]) -> dict[str, int]:
        approved = [event for event in events if event.get("review_status") == "approved"]
        classifiable = [event for event in approved if event.get("type") in EVENT_TYPES]
        wearer = [
            event
            for event in classifiable
            if event.get("contact_side") in {"left", "right"} and event.get("contact_side_basis") == "wearer_limb"
        ]
        legacy = [
            event
            for event in classifiable
            if event.get("contact_side") in {"left", "right"}
            and event.get("contact_side_basis") in {None, "", "unknown", "legacy_unspecified"}
        ]
        ambiguous = [
            event
            for event in classifiable
            if event.get("contact_side_basis") in {"ambiguous", "screen_position", "pose_anatomical"}
        ]
        return {
            "contact_classifiable_events": len(classifiable),
            "wearer_side_events": len(wearer),
            "legacy_side_basis_events": len(legacy),
            "ambiguous_side_basis_events": len(ambiguous),
        }

    def state(self) -> dict[str, Any]:
        rows = []
        readiness_rows = []
        for item in self.items:
            video_id = str(item["video_id"])
            readiness = classify_video(
                item=item,
                labels_dir=self.labels_dir,
                candidates_dir=self.candidates_dir,
                review_match_tolerance_sec=0.05,
                candidate_cluster_gap_sec=CANDIDATE_CLUSTER_GAP_SEC,
            )
            readiness_rows.append(readiness)
            label_path = self.label_path(video_id)
            approved = 0
            pending = 0
            if label_path.exists():
                doc = read_json(label_path)
                events = [event for rally in doc.get("rallies", []) for event in rally.get("events", [])]
                approved = sum(1 for event in events if event.get("review_status") == "approved")
                pending = sum(1 for event in events if event.get("review_status") != "approved")
                complete = bool(doc.get("candidate_review_complete"))
                reviewed_hints = sum(1 for row in doc.get("candidate_reviews", []) if row.get("review_status") == "reviewed")
            else:
                events = []
                complete = False
                reviewed_hints = 0
                approved = 0
                pending = 0
            side_basis_stats = self.contact_side_basis_stats(events)
            rows.append(
                {
                    **item,
                    "label_path": str(label_path),
                    "approved_events": approved,
                    "pending_events": pending,
                    "candidate_review_complete": complete,
                    "reviewed_hints": reviewed_hints,
                    "readiness_status": readiness["status"],
                    "ready_for_training_table": readiness["ready_for_training_table"],
                    "total_hints": readiness["total_hint_count"],
                    "checked_hints": readiness["checked_hint_count"],
                    "unchecked_hints": readiness["unchecked_hint_count"],
                    "likely_hints": readiness["likely_hint_count"],
                    "likely_checked_hints": readiness["likely_checked_hint_count"],
                    "likely_unchecked_hints": readiness["likely_unchecked_hint_count"],
                    "audio_only_hints": readiness["audio_only_hint_count"],
                    "audio_only_checked_hints": readiness["audio_only_checked_hint_count"],
                    "audio_only_unchecked_hints": readiness["audio_only_unchecked_hint_count"],
                    "generated_hints": readiness["generated_event_hint_count"],
                    **side_basis_stats,
                    "review_priority": None,
                    "review_reason": None,
                }
            )
        queue = make_next_clips(readiness_rows, min_non_test_videos=3, min_frozen_test_videos=1)
        by_queue_id = {row["video_id"]: row for row in queue}
        for row in rows:
            queue_row = by_queue_id.get(row["video_id"])
            if queue_row:
                row["review_priority"] = queue_row["priority"]
                row["review_reason"] = queue_row["reason"]
        return {
            "manifest": str(self.review_manifest),
            "labels_dir": str(self.labels_dir),
            "items": rows,
            "review_queue": queue,
            "summary": {
                "videos": len(rows),
                "approved_events": sum(int(row["approved_events"]) for row in rows),
                "pending_events": sum(int(row["pending_events"]) for row in rows),
                "complete_videos": sum(1 for row in rows if row["candidate_review_complete"]),
                "reviewed_hints": sum(int(row["reviewed_hints"]) for row in rows),
                "unchecked_hints": sum(int(row["unchecked_hints"]) for row in rows),
                "complete_ready_videos": sum(1 for row in rows if row["readiness_status"] == "complete_ready"),
                "frozen_test_videos": sum(1 for row in rows if row["split"] == "test_frozen"),
                "wearer_side_events": sum(int(row["wearer_side_events"]) for row in rows),
                "legacy_side_basis_events": sum(int(row["legacy_side_basis_events"]) for row in rows),
                "ambiguous_side_basis_events": sum(int(row["ambiguous_side_basis_events"]) for row in rows),
            },
        }

    def video_payload(self, video_id: str) -> dict[str, Any]:
        if video_id not in self.by_id:
            raise KeyError(video_id)
        item = self.by_id[video_id]
        return {
            "item": item,
            "events": self.load_events(video_id),
            "candidates": self.load_candidates(video_id),
            "candidate_clusters": clustered_review_hints(self.load_candidates(video_id), cluster_gap_sec=CANDIDATE_CLUSTER_GAP_SEC),
            "contact_sheets": self.contact_sheets(video_id),
            "review_montages": self.review_montages(video_id),
            "track_graph": self.load_track_graph(video_id),
            "label_path": str(self.label_path(video_id)),
        }

    def save_events(
        self,
        video_id: str,
        events: list[dict[str, Any]],
        *,
        candidate_review_complete: bool = False,
        candidate_reviews: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            if video_id not in self.by_id:
                raise KeyError(video_id)
            clean_events: list[dict[str, Any]] = []
            for event in events:
                event_type = event.get("type")
                if event_type not in EVENT_TYPES:
                    raise ValueError(f"unsupported event type: {event_type}")
                if event.get("time_sec") is None:
                    raise ValueError("event missing time_sec")
                row = {
                    "type": event_type,
                    "time_sec": round(float(event["time_sec"]), 3),
                    "review_status": event.get("review_status") or "approved",
                    "source": event.get("source") or "muted_visual_review",
                }
                if event.get("duration_sec") not in (None, ""):
                    row["duration_sec"] = round(float(event["duration_sec"]), 3)
                contact_side = str(event.get("contact_side") or "unknown")
                if contact_side not in CONTACT_SIDES:
                    raise ValueError(f"unsupported contact_side: {contact_side}")
                contact_type = str(event.get("contact_type") or "")
                if not contact_type:
                    contact_type = "ground" if event_type == "drop_floor" else "stall" if event_type == "stall" else "unknown"
                if contact_type not in CONTACT_TYPES:
                    raise ValueError(f"unsupported contact_type: {contact_type}")
                contact_surface = str(event.get("contact_surface") or "unknown")
                if contact_surface not in CONTACT_SURFACES:
                    raise ValueError(f"unsupported contact_surface: {contact_surface}")
                trick_label = str(event.get("trick_label") or "")
                if trick_label not in TRICK_LABELS:
                    raise ValueError(f"unsupported trick_label: {trick_label}")
                row["contact_side"] = contact_side
                row["contact_type"] = contact_type
                row["contact_surface"] = contact_surface
                contact_side_basis = infer_contact_side_basis(
                    contact_side,
                    trick_label,
                    None if event.get("contact_side_basis") in (None, "") else str(event.get("contact_side_basis")),
                )
                if contact_side_basis not in CONTACT_SIDE_BASES:
                    raise ValueError(f"unsupported contact_side_basis: {contact_side_basis}")
                row["contact_side_basis"] = contact_side_basis
                if trick_label:
                    row["trick_label"] = trick_label
                contact_review_status = str(event.get("contact_review_status") or "unreviewed")
                if contact_review_status not in CONTACT_REVIEW_STATUSES:
                    raise ValueError(f"unsupported contact_review_status: {contact_review_status}")
                row["contact_review_status"] = contact_review_status
                clean_events.append(row)
            clean_candidate_reviews: list[dict[str, Any]] = []
            seen_reviews: set[tuple[float, str]] = set()
            for review in candidate_reviews or []:
                if review.get("time_sec") is None:
                    raise ValueError("candidate review missing time_sec")
                decision = str(review.get("decision") or "no_touch")
                if decision not in {"no_touch", "touch"}:
                    raise ValueError(f"unsupported candidate review decision: {decision}")
                time_sec = round(float(review["time_sec"]), 3)
                key = (time_sec, decision)
                if key in seen_reviews:
                    continue
                seen_reviews.add(key)
                clean_candidate_reviews.append(
                    {
                        "time_sec": time_sec,
                        "decision": decision,
                        "review_status": "reviewed",
                        "source": review.get("source") or "muted_visual_review",
                    }
                )
            doc = self.empty_doc(video_id, clean_events)
            doc["candidate_reviews"] = sorted(clean_candidate_reviews, key=lambda row: float(row["time_sec"]))
            if candidate_review_complete:
                self.require_complete_candidate_reviews(video_id, doc["candidate_reviews"])
            doc["candidate_review_complete"] = bool(candidate_review_complete)
            doc["review_status"] = "complete" if candidate_review_complete else "in_progress"
            doc["review_completed_at"] = now_iso() if candidate_review_complete else None
            doc["updated_at"] = now_iso()
            write_json(self.label_path(video_id), doc)
            return doc


class TouchReviewHandler(BaseHTTPRequestHandler):
    server_version = "HackyTouchReview/1.0"

    @property
    def store(self) -> TouchReviewStore:
        return self.server.store  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status: HTTPStatus, message: str) -> None:
        self.send_json({"error": message}, status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_html()
            return
        if parsed.path == "/api/state":
            self.send_json(self.store.state())
            return
        if parsed.path == "/api/video":
            video_id = parse_qs(parsed.query).get("id", [""])[0]
            try:
                self.send_json(self.store.video_payload(video_id))
            except KeyError:
                self.send_error_json(HTTPStatus.NOT_FOUND, f"unknown video_id: {video_id}")
            return
        if parsed.path == "/media":
            video_id = parse_qs(parsed.query).get("id", [""])[0]
            self.serve_media(video_id)
            return
        if parsed.path == "/contact-sheet":
            query = parse_qs(parsed.query)
            video_id = query.get("id", [""])[0]
            name = query.get("name", [""])[0]
            self.serve_contact_sheet(video_id, name)
            return
        if parsed.path == "/review-montage":
            query = parse_qs(parsed.query)
            video_id = query.get("id", [""])[0]
            name = query.get("name", [""])[0]
            self.serve_review_montage(video_id, name)
            return
        self.send_error_json(HTTPStatus.NOT_FOUND, "not found")

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/media":
            video_id = parse_qs(parsed.query).get("id", [""])[0]
            self.serve_media(video_id, send_body=False)
            return
        if parsed.path == "/contact-sheet":
            query = parse_qs(parsed.query)
            video_id = query.get("id", [""])[0]
            name = query.get("name", [""])[0]
            self.serve_contact_sheet(video_id, name, send_body=False)
            return
        if parsed.path == "/review-montage":
            query = parse_qs(parsed.query)
            video_id = query.get("id", [""])[0]
            name = query.get("name", [""])[0]
            self.serve_review_montage(video_id, name, send_body=False)
            return
        if parsed.path == "/":
            body = HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return
        self.send_error_json(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        if parsed.path == "/api/events":
            video_id = str(payload.get("video_id") or "")
            try:
                doc = self.store.save_events(
                    video_id,
                    list(payload.get("events") or []),
                    candidate_review_complete=bool(payload.get("candidate_review_complete")),
                    candidate_reviews=list(payload.get("candidate_reviews") or []),
                )
            except (KeyError, ValueError) as exc:
                self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self.send_json({"ok": True, "events": doc, "label_path": str(self.store.label_path(video_id))})
            return
        self.send_error_json(HTTPStatus.NOT_FOUND, "not found")

    def serve_media(self, video_id: str, send_body: bool = True) -> None:
        if video_id not in self.store.by_id:
            self.send_error_json(HTTPStatus.NOT_FOUND, f"unknown video_id: {video_id}")
            return
        path = Path(self.store.by_id[video_id]["video_path"]).resolve()
        if not path.exists():
            self.send_error_json(HTTPStatus.NOT_FOUND, f"missing video: {path}")
            return
        ctype = mimetypes.guess_type(path.name)[0] or "video/quicktime"
        size = path.stat().st_size
        range_header = self.headers.get("Range")
        start = 0
        end = size - 1
        status = HTTPStatus.OK
        if range_header and range_header.startswith("bytes="):
            status = HTTPStatus.PARTIAL_CONTENT
            raw = range_header.split("=", 1)[1].split(",", 1)[0]
            left, _, right = raw.partition("-")
            start = int(left) if left else 0
            end = int(right) if right else size - 1
            end = min(end, size - 1)
        length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not send_body:
            return
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(chunk)

    def serve_contact_sheet(self, video_id: str, name: str, send_body: bool = True) -> None:
        if video_id not in self.store.by_id:
            self.send_error_json(HTTPStatus.NOT_FOUND, f"unknown video_id: {video_id}")
            return
        try:
            path = self.store.contact_sheet_path(video_id, name)
        except ValueError as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        if not path.exists():
            self.send_error_json(HTTPStatus.NOT_FOUND, f"missing contact sheet: {name}")
            return
        body = b"" if not send_body else path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def serve_review_montage(self, video_id: str, name: str, send_body: bool = True) -> None:
        if video_id not in self.store.by_id:
            self.send_error_json(HTTPStatus.NOT_FOUND, f"unknown video_id: {video_id}")
            return
        try:
            path = self.store.montage_path(video_id, name)
        except ValueError as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        if not path.exists():
            self.send_error_json(HTTPStatus.NOT_FOUND, f"missing review montage: {name}")
            return
        ctype = mimetypes.guess_type(path.name)[0] or "video/mp4"
        size = path.stat().st_size
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.end_headers()
        if send_body:
            self.wfile.write(path.read_bytes())

    def send_html(self) -> None:
        body = HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hacky Touch Review</title>
  <style>
    :root { color-scheme: dark; --bg:#0e1218; --panel:#151c26; --line:#2b3442; --text:#ecf2f8; --muted:#98a6b8; --green:#31d07d; --red:#ff6b6b; --blue:#5bb7ff; --amber:#f8c14a; }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    header { height: 52px; display:flex; align-items:center; justify-content:space-between; padding:0 18px; border-bottom:1px solid var(--line); background:#0b1017; }
    h1 { margin:0; font-size:17px; letter-spacing:0; }
    main { display:grid; grid-template-columns: 280px 1fr 360px; min-height:calc(100vh - 52px); }
    aside, section { border-right:1px solid var(--line); min-width:0; }
    .sidebar { background:#111821; overflow:auto; max-height:calc(100vh - 52px); }
    .video-list button { width:100%; text-align:left; border:0; border-bottom:1px solid #202938; background:transparent; color:var(--text); padding:10px 12px; cursor:pointer; }
    .video-list button.active { background:#213047; }
    .video-list small { display:block; color:var(--muted); margin-top:3px; }
    .queue-card { padding:12px; border-bottom:1px solid #243044; background:#121b27; }
    .queue-card strong { display:block; margin-bottom:4px; }
    .queue-card button { width:100%; margin-top:8px; }
    .queue-card .queue-line { color:var(--muted); font-size:13px; margin-top:3px; }
    .stage { padding:14px; }
    video { width:100%; max-height:68vh; background:#05070a; border:1px solid var(--line); display:block; }
    .controls { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:10px 0; }
    button, select, input { border:1px solid #334156; background:#1b2533; color:var(--text); border-radius:6px; padding:7px 10px; }
    button { cursor:pointer; }
    .toggle { display:inline-flex; align-items:center; gap:6px; color:var(--muted); }
    .toggle input { margin:0; }
    button.primary { background:#1f6f46; border-color:#2ab36b; }
    button.danger { background:#5a2630; border-color:#c7495c; }
    button:disabled { opacity:.45; cursor:not-allowed; }
    .timeline { position:relative; height:54px; border:1px solid var(--line); background:#0b1016; margin-top:10px; overflow:hidden; }
    .tick { position:absolute; top:0; width:2px; height:100%; transform:translateX(-1px); }
    .tick.audio { background:rgba(248,193,74,.75); }
    .tick.reviewed { background:rgba(152,166,184,.65); }
    .tick.existing_touch { background:rgba(49,208,125,.9); width:3px; }
    .tick.touch { background:var(--green); width:4px; }
    .tick.drop_floor { background:var(--red); width:4px; }
    .tick.stall { background:var(--blue); width:4px; }
    .playhead { position:absolute; top:0; width:2px; height:100%; background:#fff; z-index:5; }
    .track-panel { border:1px solid var(--line); background:#0b1016; margin-top:10px; padding:9px; }
    .track-panel-head { display:flex; justify-content:space-between; align-items:center; gap:10px; margin-bottom:7px; }
    .track-panel-head strong { font-size:13px; }
    .track-actions { display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
    .track-actions button { padding:5px 8px; }
    .track-canvas { width:100%; height:160px; display:block; background:#081018; border:1px solid #1f2a3a; }
    .guide-card { margin:10px 0 12px; padding:12px; border:1px solid #36506f; background:#101b29; border-radius:10px; box-shadow:0 0 0 1px rgba(91,183,255,.08) inset; }
    .guide-top { display:grid; grid-template-columns:1fr; gap:10px; align-items:start; }
    .guide-eyebrow { color:var(--blue); font-size:12px; font-weight:750; letter-spacing:.08em; text-transform:uppercase; margin-bottom:4px; }
    .guide-headline { font-size:18px; font-weight:750; margin-bottom:4px; }
    .guide-copy { color:#c7d3e2; max-width:820px; }
    .guide-actions { display:grid; grid-template-columns:repeat(2, minmax(150px, 1fr)); gap:8px; min-width:0; }
    .guide-actions button { min-height:42px; font-weight:750; }
    .guide-actions .yes { background:#1f6f46; border-color:#31d07d; }
    .guide-actions .no { background:#253247; border-color:#667896; }
    .guide-actions .jump { background:#183456; border-color:#5bb7ff; }
    .guide-actions .done { background:#473816; border-color:#f8c14a; }
    .guide-steps { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:8px; margin-top:12px; }
    .guide-step { border:1px solid #2b3b52; background:#0b121c; border-radius:8px; padding:8px; color:#cbd6e5; }
    .guide-step strong { display:block; color:var(--text); margin-bottom:2px; }
    .guide-step.active { border-color:#f8c14a; background:#2f2612; color:#ffe8a8; }
    .quick-fix { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:10px 0; padding:9px 10px; border:1px solid #2b3b52; background:#111923; border-radius:8px; }
    .quick-fix strong { margin-right:4px; color:#d7e3f1; }
    .advanced-panel { margin:10px 0; border:1px solid var(--line); background:#111923; border-radius:8px; overflow:hidden; }
    .advanced-panel summary { cursor:pointer; padding:9px 11px; color:#d7e3f1; font-weight:650; background:#151f2d; }
    .advanced-panel .controls { margin:0; padding:10px; border-top:1px solid #253044; }
    .classify-panel { padding:10px; border-top:1px solid #253044; background:#0f1824; }
    .classify-status { display:grid; grid-template-columns:1fr auto; gap:8px; align-items:center; margin-bottom:8px; color:#cbd6e5; }
    .classify-status strong { color:var(--text); }
    .classify-actions { display:grid; grid-template-columns:repeat(6, minmax(0, 1fr)); gap:8px; }
    .classify-actions button { min-height:42px; font-weight:750; }
    .classify-actions .kick-left, .classify-actions .kick-right { background:#173823; border-color:#31d07d; }
    .classify-actions .knee { background:#2c2f48; border-color:#9aa8ff; }
    .classify-actions .stall { background:#17324a; border-color:#5bb7ff; }
    .classify-actions .ground { background:#4a2027; border-color:#ff6b6b; }
    .classify-actions .unknown { background:#263247; border-color:#667896; }
    .classify-actions .next { background:#473816; border-color:#f8c14a; }
    .semantic-note { grid-column:1 / -1; color:#c7d3e2; border:1px solid #334156; background:#111923; border-radius:8px; padding:8px 10px; line-height:1.35; }
    .side { background:var(--panel); padding:12px; overflow:auto; max-height:calc(100vh - 52px); border-right:0; }
    .event-row { display:grid; grid-template-columns:64px 1fr auto; gap:8px; align-items:center; padding:8px 0; border-bottom:1px solid #253044; }
    .pill { display:inline-flex; align-items:center; border-radius:999px; padding:2px 7px; font-size:12px; background:#263247; color:#cbd6e5; }
    .pill.test { color:#111; background:var(--amber); }
    .muted-warning { color:#111; background:var(--amber); padding:7px 9px; border-radius:6px; font-weight:650; }
    .meta { color:var(--muted); font-size:13px; }
    .candidate-list { max-height:170px; overflow:auto; border:1px solid var(--line); margin-top:8px; }
    .candidate-list button { width:100%; display:flex; justify-content:space-between; border:0; border-bottom:1px solid #253044; border-radius:0; background:#121923; }
    .candidate-list button.reviewed { background:#182638; color:#d7e3f1; }
    .candidate-list button.touch-reviewed { background:#163424; color:#dbffe9; }
    .progress-panel { margin:12px 0 14px; border:1px solid var(--line); border-radius:8px; overflow:hidden; background:#111923; }
    .progress-row { display:grid; grid-template-columns:1fr auto; gap:10px; padding:8px 10px; border-bottom:1px solid #253044; }
    .progress-row:last-child { border-bottom:0; }
    .progress-row strong { font-weight:650; }
    .progress-row.ready { color:#dfffea; background:#123223; }
    .progress-row.blocked { color:#ffe8a8; background:#2f2612; }
    .sheet-list { margin:8px 0 14px; border:1px solid var(--line); border-radius:8px; overflow:hidden; background:#111923; }
    .sheet-list a { display:grid; grid-template-columns:1fr auto; gap:10px; padding:8px 10px; border-bottom:1px solid #253044; color:var(--text); text-decoration:none; }
    .sheet-list .sheet-row { display:grid; grid-template-columns:1fr auto; gap:8px; align-items:center; padding:8px 10px; border-bottom:1px solid #253044; }
    .sheet-list .sheet-row:last-child { border-bottom:0; }
    .sheet-list a { padding:0; border-bottom:0; }
    .sheet-list a:hover, .sheet-list .sheet-row:hover { background:#213047; }
    .sheet-actions { display:flex; gap:6px; align-items:center; }
    .sheet-actions button { padding:5px 8px; }
    .sheet-time-grid { grid-column:1 / -1; display:flex; flex-wrap:wrap; gap:5px; margin-top:6px; }
    .sheet-time-grid button { padding:4px 6px; font-size:12px; color:#d7e3f1; background:#172233; }
    .sheet-time-grid button.reviewed { background:#263247; color:#cbd6e5; }
    .sheet-time-grid button.no-touch { background:#253145; color:#9fb0c8; }
    .sheet-time-grid button.touch { background:#143a27; color:#dfffea; border-color:#31d07d; }
    .sheet-list small { color:var(--muted); }
    .empty { color:var(--muted); padding:12px 0; }
    @media (max-width: 1180px) {
      main { grid-template-columns: 280px minmax(420px, 1fr); }
      .stage { border-right:0; }
      .side { grid-column:2; border-top:1px solid var(--line); max-height:none; }
      video { max-height:52vh; }
      .track-canvas { height:180px; }
    }
  </style>
</head>
<body>
<header>
  <h1>Hacky Touch Review</h1>
  <div id="summary" class="meta"></div>
</header>
<main>
  <aside class="sidebar">
    <div id="queueCard" class="queue-card"></div>
    <div id="videoList" class="video-list"></div>
  </aside>
  <section class="stage">
    <div id="currentTitle" class="meta">Loading...</div>
    <div id="workflowGuide" class="guide-card"></div>
    <div class="quick-fix">
      <strong>Fix a mistake</strong>
      <button id="undoLast" class="danger">Undo last action</button>
      <button id="quickClearHint">Clear current hint</button>
      <button id="quickDeleteEvent" class="danger" disabled>Delete selected touch/event</button>
      <button id="quickSaveDraft">Save fixed draft</button>
      <span id="status" class="meta"></span>
    </div>
    <video id="video" controls muted playsinline></video>
    <details class="advanced-panel">
      <summary>Manual event and save controls</summary>
      <div class="controls">
        <span class="muted-warning">Muted visual review</span>
        <button id="addTouch" class="primary">Add touch here</button>
        <button id="markNoTouch">Mark current hint no-touch</button>
        <button id="markNoTouchNext">No: not a touch + next</button>
        <button id="addTouchNext">Yes: touch + next</button>
        <button id="clearHintReview">Clear current hint review</button>
        <button id="addDrop">Add drop here</button>
        <button id="addStall">Add stall here</button>
        <button id="saveDraft">Save draft only</button>
        <button id="saveComplete" class="primary">Save complete (clip done)</button>
        <button id="saveCompleteNext" class="primary">Save complete + load next</button>
        <button id="deleteEvent" class="danger" disabled>Delete selected event</button>
      </div>
    </details>
    <details class="advanced-panel" open>
      <summary>v1.0 contact labels for selected event</summary>
      <div class="classify-panel">
        <div class="classify-status">
          <div>
            <strong id="classificationHeadline">Select an event to classify</strong>
            <div id="classificationCopy" class="meta">Use the event list, timeline, or Next unclassified.</div>
          </div>
          <span id="classificationProgress" class="pill">0/0</span>
        </div>
        <div class="classify-actions">
          <div class="semantic-note"><strong>Side means contacting limb / wearer side.</strong> Do not label side from screen-left or screen-right. Use Unknown/skip when the limb is not clear.</div>
          <button id="classifyLeftKick" class="kick-left">1 Left kick</button>
          <button id="classifyLeftInnerKick" class="kick-left">2 Left inner kick</button>
          <button id="classifyLeftOuterKick" class="kick-left">3 Left outer kick</button>
          <button id="classifyRightKick" class="kick-right">4 Right kick</button>
          <button id="classifyRightInnerKick" class="kick-right">5 Right inner kick</button>
          <button id="classifyRightOuterKick" class="kick-right">6 Right outer kick</button>
          <button id="classifyLeftKnee" class="knee">7 Left knee</button>
          <button id="classifyLeftInnerKnee" class="knee">Left inner knee</button>
          <button id="classifyLeftOuterKnee" class="knee">Left outer knee</button>
          <button id="classifyRightKnee" class="knee">8 Right knee</button>
          <button id="classifyRightInnerKnee" class="knee">Right inner knee</button>
          <button id="classifyRightOuterKnee" class="knee">Right outer knee</button>
          <button id="classifyLeftStall" class="stall">9 Left stall</button>
          <button id="classifyLeftInnerStall" class="stall">Left inner stall</button>
          <button id="classifyLeftOuterStall" class="stall">Left outer stall</button>
          <button id="classifyRightStall" class="stall">Right stall</button>
          <button id="classifyRightInnerStall" class="stall">Right inner stall</button>
          <button id="classifyRightOuterStall" class="stall">Right outer stall</button>
          <button id="classifyGround" class="ground">G Ground/drop</button>
          <button id="classifyUnknown" class="unknown">0 Unknown/skip</button>
          <button id="nextUnclassifiedEvent" class="next">Next unclassified</button>
          <button id="nextSideBasisReview" class="next">Next side-basis</button>
          <button id="clearClassification">Clear class</button>
        </div>
      </div>
      <div class="controls">
        <label>Side
          <select id="selectedContactSide">
            <option value="unknown">unknown</option>
            <option value="left">left</option>
            <option value="right">right</option>
            <option value="center">center</option>
          </select>
        </label>
        <label>Side basis
          <select id="selectedContactSideBasis">
            <option value="unknown">unknown / not usable</option>
            <option value="wearer_limb">contacting limb / wearer side</option>
            <option value="screen_position">screen position only</option>
            <option value="pose_anatomical">pose anatomical side only</option>
            <option value="ambiguous">ambiguous</option>
            <option value="legacy_unspecified">legacy unspecified</option>
          </select>
        </label>
        <label>Type
          <select id="selectedContactType">
            <option value="unknown">unknown</option>
            <option value="kick">kick</option>
            <option value="foot">foot</option>
            <option value="knee">knee</option>
            <option value="stall">stall</option>
            <option value="drop_floor">drop_floor</option>
            <option value="ground">ground</option>
            <option value="chest">chest</option>
            <option value="hand">hand</option>
          </select>
        </label>
        <label>Surface
          <select id="selectedContactSurface">
            <option value="unknown">unknown</option>
            <option value="inner">inner</option>
            <option value="outer">outer</option>
          </select>
        </label>
        <label>Trick
          <select id="selectedTrickLabel">
            <option value="">none</option>
            <option value="right_kick">right_kick</option>
            <option value="left_kick">left_kick</option>
            <option value="left_knee">left_knee</option>
            <option value="right_knee">right_knee</option>
            <option value="left_outer_kick">left_outer_kick</option>
            <option value="left_inner_kick">left_inner_kick</option>
            <option value="right_inner_kick">right_inner_kick</option>
            <option value="right_outer_kick">right_outer_kick</option>
            <option value="left_outer_knee">left_outer_knee</option>
            <option value="left_inner_knee">left_inner_knee</option>
            <option value="right_inner_knee">right_inner_knee</option>
            <option value="right_outer_knee">right_outer_knee</option>
            <option value="right_stall">right_stall</option>
            <option value="left_stall">left_stall</option>
            <option value="left_inner_stall">left_inner_stall</option>
            <option value="left_outer_stall">left_outer_stall</option>
            <option value="right_inner_stall">right_inner_stall</option>
            <option value="right_outer_stall">right_outer_stall</option>
            <option value="knee">knee</option>
            <option value="clipper">clipper</option>
            <option value="around_the_world">around_the_world</option>
            <option value="around_the_world_outer_right">around_the_world_outer_right</option>
            <option value="around_the_world_outer_left">around_the_world_outer_left</option>
          </select>
        </label>
        <span id="contactLabelReadout" class="meta"></span>
      </div>
    </details>
    <div id="timeline" class="timeline"></div>
    <div class="track-panel">
      <div class="track-panel-head">
        <div>
          <strong>Ball height graph</strong>
          <span id="trackGraphMeta" class="meta"></span>
        </div>
        <div class="track-actions">
          <button id="prevValley">Prev valley</button>
          <button id="nextValley">Next valley</button>
        </div>
      </div>
      <canvas id="trackGraph" class="track-canvas"></canvas>
    </div>
    <details class="advanced-panel" open>
      <summary>Hint navigation, filters, and graph controls</summary>
      <div class="controls">
        <button id="prevCandidate">Prev shown hint</button>
        <button id="nextCandidate">Next shown hint</button>
        <button id="nextLikelyUnchecked">Next likely touch</button>
        <button id="nextUnchecked">Next unchecked hint</button>
        <select id="candidateFilter" aria-label="Candidate filter">
          <option value="all">All hints</option>
          <option value="unchecked">Unchecked</option>
          <option value="likely_unchecked">Likely unchecked</option>
          <option value="likely">Likely/model</option>
          <option value="audio_only">Audio only</option>
          <option value="reviewed">Reviewed</option>
        </select>
        <button id="bulkNoTouchFiltered">Mark filtered audio-only no-touch</button>
        <button id="bulkAudioTailComplete">Finish audio-tail as no-touch</button>
        <span id="likelyReadout" class="meta"></span>
        <button id="replayHint">Replay current hint</button>
        <label class="toggle"><input type="checkbox" id="autoReplay" checked> Auto replay hint window</label>
        <button id="frameBack">-1 frame</button>
        <button id="frameForward">+1 frame</button>
        <select id="playbackRate" aria-label="Playback speed">
          <option value="0.25">0.25x</option>
          <option value="0.5" selected>0.5x</option>
          <option value="1">1x</option>
        </select>
        <span id="timeReadout" class="meta"></span>
      </div>
    </details>
    <div class="candidate-list" id="candidateList"></div>
  </section>
  <aside class="side">
    <div id="splitPill" class="pill"></div>
    <h3>Review Progress</h3>
    <div id="reviewProgress" class="progress-panel"></div>
    <h3>Montages</h3>
    <div id="reviewMontages" class="sheet-list"></div>
    <h3>Contact Sheets</h3>
    <div id="reviewSheets" class="sheet-list"></div>
    <h3>Events</h3>
    <div id="events"></div>
    <h3>What Counts</h3>
    <p class="meta"><strong>Touch:</strong> the sack clearly contacts the foot/body and changes direction. <strong>No-touch:</strong> footstep/noise, the sack just passes nearby, or you cannot see contact. Yellow graph dots are low ball positions; many real touches are near those valleys.</p>
  </aside>
</main>
<script>
const app = {
  state:null,
  current:null,
  videoPayload:null,
  events:[],
  candidateReviews:[],
  selected:null,
  replayUntil:null,
  replayCenter:null,
  trackMinima:[],
  dirty:false,
  dirtyVersion:0,
  autosaveTimer:null,
  draftSaveInFlight:null,
  candidateFilter:"all",
  autoReplay:true,
  undoStack:[]
};
const video = document.getElementById("video");
const timeline = document.getElementById("timeline");
const trackGraph = document.getElementById("trackGraph");
const HINT_ACTION_TOL_SEC = 0.20;
const REVIEW_WINDOW_SEC = 0.35;
const FRAME_STEP_SEC = 1 / 30;

function fmt(t){ return Number(t).toFixed(3); }
function eventLabel(e){
  const contact = [
    e.contact_side || "unknown",
    e.contact_surface || "unknown",
    e.contact_type || "unknown",
    e.trick_label || ""
  ].filter(Boolean).join(" · ");
  return `${fmt(e.time_sec)}s ${e.type}${contact ? " · " + contact : ""}`;
}
async function api(path, options){ const r = await fetch(path, options); if(!r.ok) throw new Error(await r.text()); return r.json(); }

async function loadState(selectFirst=true){
  app.state = await api("/api/state");
  document.getElementById("summary").textContent = `${app.state.summary.videos} videos | ${app.state.summary.complete_ready_videos} ready | ${app.state.summary.unchecked_hints} unchecked hints | ${app.state.summary.approved_events} approved events | wearer-side ${app.state.summary.wearer_side_events} | legacy side ${app.state.summary.legacy_side_basis_events} | ${app.state.summary.frozen_test_videos} frozen test`;
  renderVideoList();
  if(selectFirst && app.state.items.length) await loadVideo(initialVideoId());
}

function initialVideoId(){
  const params = new URLSearchParams(window.location.search);
  const requested = params.get("video_id") || params.get("id");
  if(requested && app.state.items.some(item => item.video_id === requested)) return requested;
  return app.state.review_queue?.[0]?.video_id || app.state.items[0].video_id;
}

function orderedItems(){
  const queueIndex = new Map((app.state.review_queue || []).map((row, index) => [row.video_id, index]));
  return [...app.state.items].sort((a, b) => {
    const ai = queueIndex.has(a.video_id) ? queueIndex.get(a.video_id) : 9999;
    const bi = queueIndex.has(b.video_id) ? queueIndex.get(b.video_id) : 9999;
    if(ai !== bi) return ai - bi;
    if(Number(a.legacy_side_basis_events || 0) !== Number(b.legacy_side_basis_events || 0)){
      return Number(b.legacy_side_basis_events || 0) - Number(a.legacy_side_basis_events || 0);
    }
    if(a.split !== b.split) return String(a.split).localeCompare(String(b.split));
    return String(a.video_name).localeCompare(String(b.video_name));
  });
}

function sideBasisQueue(){
  return [...(app.state.items || [])]
    .filter(item => Number(item.legacy_side_basis_events || 0) > 0)
    .sort((a, b) => {
      const debt = Number(b.legacy_side_basis_events || 0) - Number(a.legacy_side_basis_events || 0);
      if(debt !== 0) return debt;
      return String(a.video_name).localeCompare(String(b.video_name));
    });
}

function renderQueueCard(){
  const card = document.getElementById("queueCard");
  const next = app.state.review_queue?.[0] || null;
  const nextDifferent = (app.state.review_queue || []).find(row => !app.current || row.video_id !== app.current.video_id) || null;
  const sideQueue = sideBasisQueue();
  const sideTarget = sideQueue.find(row => !app.current || row.video_id !== app.current.video_id) || sideQueue[0] || null;
  const sideLine = sideQueue.length
    ? `<div class="queue-line">Side-basis debt: ${app.state.summary.legacy_side_basis_events} legacy left/right labels across ${sideQueue.length} clips.</div><button id="loadNextSideBasis" ${sideTarget ? "" : "disabled"}>${sideTarget ? "Load next side-basis clip" : "Current clip has side-basis debt"}</button>`
    : `<div class="queue-line">Side-basis debt: none.</div>`;
  if(!next){
    card.innerHTML = `<strong>Review queue</strong><div class="queue-line">All clips are complete-ready.</div>${sideLine}`;
    if(sideTarget) document.getElementById("loadNextSideBasis").onclick = () => loadVideo(sideTarget.video_id);
    return;
  }
  const target = app.current && next.video_id === app.current.video_id ? nextDifferent : next;
  const buttonText = app.current && next.video_id === app.current.video_id ? "Load next different clip" : "Load next priority";
  const currentLine = app.current && next.video_id === app.current.video_id
    ? `<div class="queue-line">Current clip is still top priority until all hints are checked and saved complete.</div>`
    : "";
  card.innerHTML = `<strong>Next review clip</strong>
    <div>${next.video_name}</div>
    <div class="queue-line">${next.split} · ${next.status} · ${next.unchecked_hint_count}/${next.total_hint_count} unchecked</div>
    <div class="queue-line">likely ${next.likely_unchecked_hint_count}/${next.likely_hint_count} · audio-tail ${next.audio_only_unchecked_hint_count}/${next.audio_only_hint_count}</div>
    <div class="queue-line">${next.reason}</div>
    ${currentLine}
    <button id="loadNextPriority" ${target ? "" : "disabled"}>${target ? buttonText : "No other priority clips"}</button>
    ${sideLine}`;
  if(target) document.getElementById("loadNextPriority").onclick = () => loadVideo(target.video_id);
  if(sideTarget) document.getElementById("loadNextSideBasis").onclick = () => loadVideo(sideTarget.video_id);
}

function renderVideoList(){
  renderQueueCard();
  const list = document.getElementById("videoList");
  list.innerHTML = "";
  orderedItems().forEach(item => {
    const b = document.createElement("button");
    b.className = app.current && app.current.video_id === item.video_id ? "active" : "";
    const priority = item.review_priority === null || item.review_priority === undefined ? "" : `P${item.review_priority} | `;
    const sideDebt = Number(item.legacy_side_basis_events || 0) ? ` | side-basis ${item.wearer_side_events}/${item.legacy_side_basis_events} wearer/legacy` : ` | wearer-side ${item.wearer_side_events}`;
    b.innerHTML = `<strong>${item.video_name}</strong><small>${priority}${item.split} | ${item.readiness_status} | hints ${item.checked_hints}/${item.total_hints} | likely left ${item.likely_unchecked_hints} | audio-tail left ${item.audio_only_unchecked_hints} | approved ${item.approved_events} | pending ${item.pending_events}${sideDebt}</small>`;
    b.onclick = () => loadVideo(item.video_id);
    list.appendChild(b);
  });
}

async function loadVideo(videoId){
  if(app.current && app.current.video_id !== videoId){
    await flushDraftSave();
    if(app.dirty && !window.confirm("Draft autosave did not finish. Switch clips and lose unsaved changes?")) return;
  }
  app.videoPayload = await api(`/api/video?id=${encodeURIComponent(videoId)}`);
  app.current = app.videoPayload.item;
  app.events = ((app.videoPayload.events.rallies || [])[0]?.events || []).map(e => ({...e}));
  app.candidateReviews = (app.videoPayload.events.candidate_reviews || []).map(e => ({...e}));
  app.selected = null;
  app.undoStack = [];
  app.replayUntil = null;
  app.dirty = false;
  chooseDefaultCandidateFilter();
  video.src = `/media?id=${encodeURIComponent(videoId)}`;
  enforceMuted();
  video.playbackRate = Number(document.getElementById("playbackRate").value || 0.5);
  const stateItem = currentStateItem();
  const readiness = stateItem ? ` | ${stateItem.readiness_status} | ${stateItem.unchecked_hints}/${stateItem.total_hints} unchecked` : "";
  document.getElementById("currentTitle").textContent = `${app.current.video_name} | ${app.current.split}${readiness}`;
  const url = new URL(window.location.href);
  url.searchParams.set("video_id", videoId);
  window.history.replaceState({}, "", url);
  const split = document.getElementById("splitPill");
  split.textContent = app.current.split;
  split.className = `pill ${app.current.split === "test_frozen" ? "test" : ""}`;
  renderAll();
}

function enforceMuted(){
  if(!video.muted || video.volume !== 0){
    video.muted = true;
    video.volume = 0;
  }
}

function currentStateItem(){
  if(!app.state || !app.current) return null;
  return app.state.items.find(item => item.video_id === app.current.video_id) || null;
}

function allCandidates(){
  const clusters = app.videoPayload?.candidate_clusters || [];
  if(clusters.length){
    return clusters.map(x => ({...x, kind:x.has_audio ? "audio" : "existing", source:x.source || "clustered_candidate", priority_score:candidatePriorityScore(x), priority_label:candidatePriorityLabel(x)})).sort((a,b)=>Number(a.time_sec)-Number(b.time_sec));
  }
  const c = app.videoPayload?.candidates || {};
  const audio = (c.audio_candidates || []).map(x => ({...x, kind:"audio"}));
  const existing = (c.existing_event_hints || []).map(x => ({...x, kind:x.source || "existing"}));
  const generated = (c.generated_event_hints || []).map(x => ({...x, kind:x.source || "generated"}));
  return audio.concat(existing).concat(generated).map(x => ({...x, priority_score:candidatePriorityScore(x), priority_label:candidatePriorityLabel(x)})).sort((a,b)=>Number(a.time_sec)-Number(b.time_sec));
}

function candidateHintTypes(c){
  return (c.existing_hint_types || (c.event_type ? [c.event_type] : []) || []).map(x => String(x));
}

function candidatePriorityScore(c){
  const types = candidateHintTypes(c);
  const sources = (c.raw_sources || [c.source || c.kind || ""]).map(x => String(x));
  let score = 0;
  if(types.includes("touch")) score += 100;
  if(types.includes("drop_floor") || types.includes("stall")) score += 25;
  if(sources.some(x => x.startsWith("generated_"))) score += 40;
  if(sources.some(x => x.startsWith("existing_"))) score += 30;
  if(c.has_audio || c.kind === "audio") score += 5;
  return score;
}

function candidatePriorityLabel(c){
  const types = candidateHintTypes(c);
  const sources = (c.raw_sources || [c.source || c.kind || ""]).map(x => String(x));
  if(types.includes("touch") && sources.some(x => x.startsWith("generated_"))) return "model touch";
  if(types.includes("touch")) return "touch hint";
  if(types.includes("drop_floor")) return "drop hint";
  if(types.includes("stall")) return "stall hint";
  return "audio only";
}

function reviewForCandidate(c){
  const t = Number(c.time_sec);
  return app.candidateReviews.find(r => Math.abs(Number(r.time_sec) - t) <= 0.05) || null;
}

function approvedEventNear(c){
  const t = Number(c.time_sec);
  return app.events.find(e => e.review_status === "approved" && Math.abs(Number(e.time_sec) - t) <= 0.20) || null;
}

function candidateCovered(c){
  return Boolean(reviewForCandidate(c));
}

function uncoveredCandidates(){
  return allCandidates().filter(c => !candidateCovered(c));
}

function isLikelyCandidate(c){
  return Number(c.priority_score || 0) >= 100;
}

function likelyCandidates(){
  return allCandidates().filter(c => isLikelyCandidate(c));
}

function likelyUncheckedCandidates(){
  return likelyCandidates().filter(c => !candidateCovered(c));
}

function candidateMatchesFilter(c){
  const review = reviewForCandidate(c);
  if(app.candidateFilter === "unchecked") return !review;
  if(app.candidateFilter === "likely_unchecked") return isLikelyCandidate(c) && !review;
  if(app.candidateFilter === "likely") return isLikelyCandidate(c);
  if(app.candidateFilter === "audio_only") return !isLikelyCandidate(c) && !review;
  if(app.candidateFilter === "reviewed") return Boolean(review);
  return true;
}

function filteredCandidates(){
  return allCandidates().filter(candidateMatchesFilter);
}

function candidateReviewStats(){
  const candidates = allCandidates();
  let checked = 0;
  let likelyTotal = 0;
  let likelyChecked = 0;
  let audioOnlyTotal = 0;
  let audioOnlyChecked = 0;
  let touchReviews = 0;
  let noTouchReviews = 0;
  candidates.forEach(candidate => {
    const review = reviewForCandidate(candidate);
    const likely = isLikelyCandidate(candidate);
    if(likely) likelyTotal += 1;
    else audioOnlyTotal += 1;
    if(review){
      checked += 1;
      if(likely) likelyChecked += 1;
      else audioOnlyChecked += 1;
      if(review.decision === "touch") touchReviews += 1;
      if(review.decision === "no_touch") noTouchReviews += 1;
    }
  });
  const total = candidates.length;
  return {
    total,
    checked,
    unchecked: Math.max(0, total - checked),
    likelyTotal,
    likelyChecked,
    likelyUnchecked: Math.max(0, likelyTotal - likelyChecked),
    audioOnlyTotal,
    audioOnlyChecked,
    audioOnlyUnchecked: Math.max(0, audioOnlyTotal - audioOnlyChecked),
    touchReviews,
    noTouchReviews,
    completeReady: total > 0 && checked === total
  };
}

function chooseDefaultCandidateFilter(){
  const stats = candidateReviewStats();
  const select = document.getElementById("candidateFilter");
  if(stats.likelyUnchecked > 0){
    app.candidateFilter = "likely_unchecked";
  } else if(stats.audioOnlyUnchecked > 0){
    app.candidateFilter = "audio_only";
  } else if(stats.unchecked > 0){
    app.candidateFilter = "unchecked";
  } else {
    app.candidateFilter = "reviewed";
  }
  if(select) select.value = app.candidateFilter;
}

function candidateDescriptor(c){
  if(!c) return "no hint";
  const strength = c.strength || c.audio_strength;
  const strengthText = strength ? ` · audio ${Number(strength).toFixed(2)}` : "";
  return `${fmt(c.time_sec)}s · ${c.priority_label || c.source || "hint"}${strengthText}`;
}

function nearestCandidateFrom(candidates){
  if(!candidates.length) return null;
  const t = Number(video.currentTime || 0);
  return candidates.reduce((best, c) => Math.abs(Number(c.time_sec) - t) < Math.abs(Number(best.time_sec) - t) ? c : best, candidates[0]);
}

function nearestCandidate(){
  return nearestCandidateFrom(allCandidates());
}

function nearestFilteredCandidate(){
  return nearestCandidateFrom(filteredCandidates());
}

function nearestCandidateDistance(candidate){
  if(!candidate) return Infinity;
  return Math.abs(Number(candidate.time_sec) - Number(video.currentTime || 0));
}

function nearestGlobalUncheckedCandidate(){
  const candidate = nearestCandidate();
  if(candidate && !candidateCovered(candidate) && nearestCandidateDistance(candidate) <= HINT_ACTION_TOL_SEC){
    return candidate;
  }
  return null;
}

function nextGlobalUncheckedCandidate(){
  const candidates = uncoveredCandidates();
  if(!candidates.length) return null;
  const t = Number(video.currentTime || 0);
  return candidates.find(x => Number(x.time_sec) > t + 0.03) || candidates[0];
}

function renderWorkflowGuide(){
  const root = document.getElementById("workflowGuide");
  const stats = candidateReviewStats();
  const currentHint = nearestGlobalUncheckedCandidate();
  const nextHint = nextGlobalUncheckedCandidate();
  const displayHint = currentHint || nextHint;
  const readyToDecide = Boolean(currentHint);
  const canDecideDisplayedHint = Boolean(displayHint);
  const completeReady = stats.unchecked === 0;
  const headline = completeReady
    ? "All hints are checked. Save this clip complete."
    : readyToDecide
      ? `Review this hint: ${candidateDescriptor(currentHint)}`
      : `Jump to the next unchecked hint (${stats.unchecked} left)`;
  const copy = completeReady
    ? "This clip has a touch/no-touch decision for every hint. Save complete, then move to the next priority clip."
    : readyToDecide
      ? "Watch the muted replay at the white playhead. If the sack visibly contacts the foot/body and changes direction, choose Yes. If it is only a footstep/noise/pass-by, choose No."
      : displayHint
        ? `Click Jump to cue ${candidateDescriptor(displayHint)}. The yellow graph dots are low ball positions; many true touches happen near those valleys.`
        : "No unchecked hints remain in this clip.";
  const stepClass = completeReady ? ["", "", "", "active"] : readyToDecide ? ["", "active", "active", ""] : ["active", "", "", ""];
  root.innerHTML = `
    <div class="guide-top">
      <div>
        <div class="guide-eyebrow">Current task</div>
        <div class="guide-headline">${headline}</div>
        <div class="guide-copy">${copy}</div>
      </div>
      <div class="guide-actions">
        <button id="guideNextUnchecked" class="jump" ${nextHint ? "" : "disabled"}>Jump to next unchecked</button>
        <button id="guideSaveCompleteNext" class="done" ${completeReady ? "" : "disabled"}>Save complete + next clip</button>
        <button id="guideTouchNext" class="yes" ${canDecideDisplayedHint ? "" : "disabled"}>Yes, touch → next (this hint)</button>
        <button id="guideNoTouchNext" class="no" ${canDecideDisplayedHint ? "" : "disabled"}>No, not touch → next (this hint)</button>
      </div>
    </div>
    <div class="guide-steps">
      <div class="guide-step ${stepClass[0]}"><strong>1. Cue a hint</strong>Use Jump to next unchecked.</div>
      <div class="guide-step ${stepClass[1]}"><strong>2. Watch muted</strong>Replay the contact moment.</div>
      <div class="guide-step ${stepClass[2]}"><strong>3. Decide</strong>Yes touch, or no-touch.</div>
      <div class="guide-step ${stepClass[3]}"><strong>4. Finish clip</strong>Save complete when 0 left.</div>
    </div>`;
  document.getElementById("guideNextUnchecked").onclick = stepAnyUncheckedCandidate;
  document.getElementById("guideSaveCompleteNext").onclick = () => save(true, true);
  document.getElementById("guideTouchNext").onclick = () => displayHint ? addTouchForCandidate(displayHint, true) : false;
  document.getElementById("guideNoTouchNext").onclick = () => displayHint ? setCandidateReviewAndAdvance(displayHint, "no_touch") : false;
}

function setStatus(message){
  document.getElementById("status").textContent = message;
}

function snapshotState(label){
  return {
    label,
    events: app.events.map(row => ({...row})),
    candidateReviews: app.candidateReviews.map(row => ({...row})),
    selected: app.selected
  };
}

function renderUndoControls(){
  const undo = document.getElementById("undoLast");
  if(undo) undo.disabled = app.undoStack.length === 0;
  const quickDelete = document.getElementById("quickDeleteEvent");
  if(quickDelete) quickDelete.disabled = app.selected === null;
}

function pushUndo(label){
  app.undoStack.push(snapshotState(label));
  if(app.undoStack.length > 30) app.undoStack.shift();
  renderUndoControls();
}

function undoLastAction(){
  const prior = app.undoStack.pop();
  if(!prior){
    setStatus("nothing to undo");
    renderUndoControls();
    return;
  }
  app.events = prior.events.map(row => ({...row}));
  app.candidateReviews = prior.candidateReviews.map(row => ({...row}));
  app.selected = prior.selected;
  markDirty(`undid ${prior.label}`);
  renderAll();
}

function markDirty(message="unsaved draft"){
  app.dirty = true;
  app.dirtyVersion += 1;
  setStatus(message);
  scheduleDraftSave();
}

function scheduleDraftSave(){
  if(app.autosaveTimer) clearTimeout(app.autosaveTimer);
  app.autosaveTimer = setTimeout(() => flushDraftSave(), 900);
}

async function flushDraftSave(){
  if(!app.current || !app.dirty) return;
  if(app.autosaveTimer){
    clearTimeout(app.autosaveTimer);
    app.autosaveTimer = null;
  }
  if(app.draftSaveInFlight) return app.draftSaveInFlight;
  const videoId = app.current.video_id;
  const version = app.dirtyVersion;
  const events = app.events.map(row => ({...row}));
  const candidateReviews = app.candidateReviews.map(row => ({...row}));
  app.draftSaveInFlight = api("/api/events", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({
      video_id: videoId,
      events,
      candidate_reviews: candidateReviews,
      candidate_review_complete:false
    })
  }).then(() => {
    if(app.current && app.current.video_id === videoId && app.dirtyVersion === version){
      app.dirty = false;
      setStatus(`draft autosaved ${new Date().toLocaleTimeString()}`);
    } else if(app.current && app.current.video_id === videoId && app.dirty){
      scheduleDraftSave();
    }
  }).catch(err => {
    if(app.current && app.current.video_id === videoId){
      setStatus(`autosave failed: ${err.message || err}`);
    }
  }).finally(() => {
    app.draftSaveInFlight = null;
  });
  return app.draftSaveInFlight;
}

function upsertCandidateReview(candidate, decision, source="muted_visual_review"){
  const t = Number(candidate.time_sec);
  app.candidateReviews = app.candidateReviews.filter(r => Math.abs(Number(r.time_sec) - t) > 0.05);
  app.candidateReviews.push({ time_sec: Number(t.toFixed(3)), decision, review_status:"reviewed", source });
}

function setCandidateReviewFor(candidate, decision, source="muted_visual_review"){
  upsertCandidateReview(candidate, decision, source);
  markDirty();
}

function setCandidateReview(decision, advance=false){
  const candidate = nearestFilteredCandidate();
  if(!candidate) return false;
  if(nearestCandidateDistance(candidate) > HINT_ACTION_TOL_SEC){
    setStatus(`no hint within ${HINT_ACTION_TOL_SEC.toFixed(2)}s`);
    return false;
  }
  pushUndo(`mark ${decision}`);
  setCandidateReviewFor(candidate, decision);
  if(advance) stepUncheckedCandidate(1);
  renderAll();
  return true;
}

function setNearestCandidateReview(decision, advance=false){
  const candidate = nearestCandidate();
  if(!candidate) return false;
  if(nearestCandidateDistance(candidate) > HINT_ACTION_TOL_SEC){
    setStatus(`no hint within ${HINT_ACTION_TOL_SEC.toFixed(2)}s`);
    return false;
  }
  pushUndo(`mark ${decision}`);
  setCandidateReviewFor(candidate, decision);
  if(advance) stepAnyUncheckedCandidate();
  renderAll();
  return true;
}

function setCandidateReviewAndAdvance(candidate, decision){
  if(!candidate) return false;
  pushUndo(`mark ${decision}`);
  setCandidateReviewFor(candidate, decision);
  stepAnyUncheckedCandidate();
  renderAll();
  return true;
}

function clearCandidateReview(){
  const candidate = nearestFilteredCandidate();
  if(!candidate) return;
  if(nearestCandidateDistance(candidate) > HINT_ACTION_TOL_SEC){
    setStatus(`no hint within ${HINT_ACTION_TOL_SEC.toFixed(2)}s`);
    return;
  }
  pushUndo("clear hint review");
  const t = Number(candidate.time_sec);
  app.candidateReviews = app.candidateReviews.filter(r => Math.abs(Number(r.time_sec) - t) > 0.05);
  markDirty("hint review cleared");
  renderAll();
}

function bulkNoTouchFiltered(){
  const candidates = filteredCandidates().filter(candidate => !candidateCovered(candidate));
  if(!candidates.length){
    setStatus("no unchecked hints in current filter");
    return;
  }
  const filter = document.getElementById("candidateFilter").value;
  if(filter !== "audio_only"){
    setStatus("bulk no-touch is only available with the Audio only filter");
    return;
  }
  const message = `Mark ${candidates.length} unchecked hints in filter "${filter}" as no-touch? This is a visual review decision.`;
  if(!window.confirm(message)) return;
  pushUndo("bulk mark no-touch");
  candidates.forEach(candidate => upsertCandidateReview(candidate, "no_touch", "muted_visual_review_bulk"));
  markDirty(`${candidates.length} filtered hints marked no-touch`);
  renderAll();
}

async function bulkAudioTailAndSaveComplete(){
  const stats = candidateReviewStats();
  if(stats.unchecked === 0){
    await save(true);
    return;
  }
  if(stats.likelyUnchecked > 0){
    setStatus(`review ${stats.likelyUnchecked} likely/model hints before bulk-completing audio-tail`);
    return;
  }
  const candidates = allCandidates().filter(candidate => !isLikelyCandidate(candidate) && !candidateCovered(candidate));
  if(!candidates.length){
    setStatus("no audio-tail hints left to bulk mark");
    return;
  }
  const message = `Mark ${candidates.length} remaining audio-tail hints as no-touch and save this clip complete? This should only be used after muted visual review confirms the tail contains no touches.`;
  if(!window.confirm(message)) return;
  pushUndo("bulk audio-tail no-touch");
  candidates.forEach(candidate => upsertCandidateReview(candidate, "no_touch", "muted_visual_review_audio_tail_complete"));
  renderAll();
  await save(true);
}

function sheetTimesNoTouch(rawTimes){
  const times = String(rawTimes || "").split(",").map(Number).filter(Number.isFinite);
  if(!times.length){
    setStatus("no sheet times to mark");
    return;
  }
  const candidates = allCandidates().filter(candidate => {
    if(isLikelyCandidate(candidate) || candidateCovered(candidate)) return false;
    return times.some(time => Math.abs(Number(candidate.time_sec) - time) <= 0.05);
  });
  if(!candidates.length){
    setStatus("no unchecked audio-tail hints left on this sheet");
    return;
  }
  const message = `Mark ${candidates.length} audio-tail hints from this sheet as no-touch? Use only after muted visual review confirms this sheet page has no touches.`;
  if(!window.confirm(message)) return;
  candidates.forEach(candidate => upsertCandidateReview(candidate, "no_touch", "muted_visual_review_sheet_bulk"));
  markDirty(`${candidates.length} sheet hints marked no-touch`);
  renderAll();
}

function duration(){
  return Number.isFinite(video.duration) && video.duration > 0 ? video.duration : Math.max(1, ...allCandidates().map(c=>Number(c.time_sec)), ...app.events.map(e=>Number(e.time_sec)), ...trackPoints().map(p=>Number(p.time_sec)));
}

function pct(t){ return Math.max(0, Math.min(100, 100 * Number(t) / duration())); }

function trackPoints(){
  return (app.videoPayload?.track_graph?.track_points || [])
    .map(p => ({...p, time_sec:Number(p.time_sec), x:Number(p.x), y:Number(p.y)}))
    .filter(p => Number.isFinite(p.time_sec) && Number.isFinite(p.y))
    .sort((a, b) => a.time_sec - b.time_sec);
}

function localValleys(points){
  if(points.length < 5) return [];
  const valleys = [];
  const win = 2;
  for(let i = win; i < points.length - win; i += 1){
    const y = Number(points[i].y);
    const neighborhood = points.slice(i - win, i + win + 1);
    const isLowestBall = neighborhood.every(p => y >= Number(p.y));
    const prominence = y - Math.min(...neighborhood.map(p => Number(p.y)));
    const separated = !valleys.length || points[i].time_sec - valleys[valleys.length - 1].time_sec >= 0.22;
    if(isLowestBall && prominence >= 4 && separated){
      valleys.push(points[i]);
    } else if(isLowestBall && prominence >= 4 && valleys.length){
      const prior = valleys[valleys.length - 1];
      if(points[i].time_sec - prior.time_sec < 0.22 && y > Number(prior.y)){
        valleys[valleys.length - 1] = points[i];
      }
    }
  }
  return valleys;
}

function resizeCanvas(canvas){
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(320, Math.floor(rect.width * ratio));
  const height = Math.max(120, Math.floor(rect.height * ratio));
  if(canvas.width !== width || canvas.height !== height){
    canvas.width = width;
    canvas.height = height;
  }
  return {width, height, ratio};
}

function drawTrackMarker(ctx, x, color, height, dash=false){
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  if(dash) ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(x, 0);
  ctx.lineTo(x, height);
  ctx.stroke();
  ctx.restore();
}

function renderTrackGraph(){
  if(!trackGraph) return;
  const {width, height} = resizeCanvas(trackGraph);
  const ctx = trackGraph.getContext("2d");
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#081018";
  ctx.fillRect(0, 0, width, height);
  const points = trackPoints();
  const meta = document.getElementById("trackGraphMeta");
  const graphStatus = app.videoPayload?.track_graph?.status || "missing_track_file";
  if(!points.length){
    app.trackMinima = [];
    meta.textContent = ` · no detector track for this clip (${graphStatus}); use hint ticks and video only`;
    ctx.fillStyle = "#98a6b8";
    ctx.font = "13px system-ui, sans-serif";
    ctx.fillText("No detector track graph available for this clip.", 14, 30);
    return;
  }
  const padX = 24;
  const padY = 16;
  const plotW = Math.max(1, width - padX * 2);
  const plotH = Math.max(1, height - padY * 2);
  const dur = duration();
  const ys = points.map(p => p.y);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const yRange = Math.max(1, yMax - yMin);
  const xFor = t => padX + plotW * Math.max(0, Math.min(1, Number(t) / dur));
  const yFor = y => padY + plotH * ((Number(y) - yMin) / yRange);

  ctx.strokeStyle = "#1f2a3a";
  ctx.lineWidth = 1;
  for(let i = 0; i <= 4; i += 1){
    const y = padY + (plotH * i / 4);
    ctx.beginPath();
    ctx.moveTo(padX, y);
    ctx.lineTo(width - padX, y);
    ctx.stroke();
  }

  allCandidates().forEach(c => {
    const review = reviewForCandidate(c);
    const color = review ? (review.decision === "touch" ? "rgba(49,208,125,.55)" : "rgba(152,166,184,.35)") : "rgba(248,193,74,.35)";
    drawTrackMarker(ctx, xFor(c.time_sec), color, height, true);
  });
  app.events.forEach(e => {
    const color = e.type === "touch" ? "rgba(49,208,125,.9)" : e.type === "drop_floor" ? "rgba(255,107,107,.85)" : "rgba(91,183,255,.85)";
    drawTrackMarker(ctx, xFor(e.time_sec), color, height, false);
  });

  ctx.strokeStyle = "#5bb7ff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  let started = false;
  let prior = null;
  points.forEach(p => {
    const x = xFor(p.time_sec);
    const y = yFor(p.y);
    if(!started || (prior && p.time_sec - prior.time_sec > 0.25)){
      ctx.moveTo(x, y);
      started = true;
    } else {
      ctx.lineTo(x, y);
    }
    prior = p;
  });
  ctx.stroke();

  app.trackMinima = localValleys(points);
  app.trackMinima.forEach(p => {
    const x = xFor(p.time_sec);
    const y = yFor(p.y);
    ctx.fillStyle = "#f8c14a";
    ctx.beginPath();
    ctx.arc(x, y, 4.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#111821";
    ctx.lineWidth = 1;
    ctx.stroke();
  });

  const playX = xFor(video.currentTime || 0);
  drawTrackMarker(ctx, playX, "#ffffff", height, false);
  ctx.fillStyle = "#98a6b8";
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillText("valleys = low ball positions", padX, height - 5);
  meta.textContent = ` · ${points.length} points · ${app.trackMinima.length} valleys`;
}

function graphTimeFromEvent(event){
  const rect = trackGraph.getBoundingClientRect();
  const x = Math.max(0, Math.min(rect.width, event.clientX - rect.left));
  return duration() * (x / Math.max(1, rect.width));
}

function renderTimeline(){
  timeline.innerHTML = "";
  allCandidates().forEach(c => {
    const d = document.createElement("div");
    const review = reviewForCandidate(c);
    d.className = `tick ${review ? "reviewed" : c.kind === "audio" ? "audio" : c.event_type || "existing_touch"}`;
    d.style.left = `${pct(c.time_sec)}%`;
    d.title = `${c.source} ${fmt(c.time_sec)}s`;
    timeline.appendChild(d);
  });
  app.events.forEach((e, i) => {
    const d = document.createElement("div");
    d.className = `tick ${e.type}`;
    d.style.left = `${pct(e.time_sec)}%`;
    d.title = eventLabel(e);
    d.onclick = () => { app.selected = i; video.currentTime = Number(e.time_sec); renderAll(); };
    timeline.appendChild(d);
  });
  const p = document.createElement("div");
  p.className = "playhead";
  p.style.left = `${pct(video.currentTime || 0)}%`;
  timeline.appendChild(p);
}

function renderEvents(){
  const root = document.getElementById("events");
  root.innerHTML = "";
  if(!app.events.length){
    root.innerHTML = `<div class="empty">No reviewed events yet.</div>`;
    renderContactLabelControls();
    return;
  }
  app.events.sort((a,b)=>Number(a.time_sec)-Number(b.time_sec));
  app.events.forEach((e, i) => {
    const row = document.createElement("div");
    row.className = "event-row";
    row.style.background = app.selected === i ? "#22314a" : "transparent";
    const basis = e.contact_side_basis && e.contact_side_basis !== "unknown" ? `basis:${e.contact_side_basis}` : "";
    const contact = [e.contact_side || "unknown", basis, e.contact_surface || "unknown", e.contact_type || "unknown", e.trick_label || ""].filter(Boolean).join(" · ");
    row.innerHTML = `<button>${fmt(e.time_sec)}s</button><span>${e.type}<br><small class="meta">${e.review_status || "approved"} · ${contact}</small></span><button data-i="${i}">go</button>`;
    row.onclick = () => { app.selected = i; video.currentTime = Number(e.time_sec); renderAll(); };
    root.appendChild(row);
  });
  document.getElementById("deleteEvent").disabled = app.selected === null;
  renderContactLabelControls();
  renderUndoControls();
}

function selectedEvent(){
  if(app.selected === null || app.selected === undefined) return null;
  return app.events[app.selected] || null;
}

function contactDefaultsForType(type){
  if(type === "drop_floor") return {contact_side:"unknown", contact_side_basis:"unknown", contact_type:"ground", contact_surface:"unknown", trick_label:""};
  if(type === "stall") return {contact_side:"unknown", contact_side_basis:"unknown", contact_type:"stall", contact_surface:"unknown", trick_label:""};
  return {contact_side:"unknown", contact_side_basis:"unknown", contact_type:"unknown", contact_surface:"unknown", trick_label:""};
}

function applyContactDefaults(row){
  const defaults = contactDefaultsForType(row.type);
  row.contact_side = row.contact_side || defaults.contact_side;
  if(!row.contact_side_basis && (row.contact_side === "left" || row.contact_side === "right")){
    row.contact_side_basis = "legacy_unspecified";
  } else {
    row.contact_side_basis = row.contact_side_basis || defaults.contact_side_basis;
  }
  row.contact_type = row.contact_type || defaults.contact_type;
  row.contact_surface = row.contact_surface || defaults.contact_surface;
  row.trick_label = row.trick_label || defaults.trick_label;
  row.contact_review_status = row.contact_review_status || "unreviewed";
  return row;
}

function renderContactLabelControls(){
  const item = selectedEvent();
  const side = document.getElementById("selectedContactSide");
  const sideBasis = document.getElementById("selectedContactSideBasis");
  const type = document.getElementById("selectedContactType");
  const surface = document.getElementById("selectedContactSurface");
  const trick = document.getElementById("selectedTrickLabel");
  const readout = document.getElementById("contactLabelReadout");
  const disabled = !item;
  [side, sideBasis, type, surface, trick].forEach(control => { if(control) control.disabled = disabled; });
  if(!item){
    if(side) side.value = "unknown";
    if(sideBasis) sideBasis.value = "unknown";
    if(type) type.value = "unknown";
    if(surface) surface.value = "unknown";
    if(trick) trick.value = "";
    if(readout) readout.textContent = "Select an event to label side/type.";
    renderClassificationPanel();
    return;
  }
  applyContactDefaults(item);
  if(side) side.value = item.contact_side || "unknown";
  if(sideBasis) sideBasis.value = item.contact_side_basis || "unknown";
  if(type) type.value = item.contact_type || "unknown";
  if(surface) surface.value = item.contact_surface || "unknown";
  if(trick) trick.value = item.trick_label || "";
  if(readout) readout.textContent = `${item.type} @ ${fmt(item.time_sec)}s`;
  renderClassificationPanel();
}

function inferTrickLabel(side, contactType, contactSurface="unknown"){
  if(side !== "left" && side !== "right") return "";
  if((contactType === "kick" || contactType === "foot" || contactType === "knee" || contactType === "stall") && (contactSurface === "inner" || contactSurface === "outer")){
    const normalizedType = contactType === "foot" ? "kick" : contactType;
    return `${side}_${contactSurface}_${normalizedType}`;
  }
  if(contactType === "stall") return `${side}_stall`;
  if(contactType === "kick" || contactType === "foot") return `${side}_kick`;
  if(contactType === "knee") return `${side}_knee`;
  return "";
}

function updateSelectedContactField(field, value){
  const item = selectedEvent();
  if(!item) return;
  pushUndo(`set ${field}`);
  applyContactDefaults(item);
  item[field] = value;
  if(field === "contact_side" && (value === "left" || value === "right") && (!item.contact_side_basis || item.contact_side_basis === "unknown")){
    item.contact_side_basis = "wearer_limb";
  }
  item.contact_review_status = "reviewed";
  if(field === "contact_side" || field === "contact_type" || field === "contact_surface"){
    const suggested = inferTrickLabel(item.contact_side, item.contact_type, item.contact_surface);
    if(suggested && !item.trick_label) item.trick_label = suggested;
  }
  markDirty(`updated ${field}`);
  renderAll();
}

function classifiableEvents(){
  return app.events.filter(event => (event.review_status || "approved") === "approved" && ["touch", "stall", "drop_floor"].includes(event.type));
}

function sideBasisIsExplicitWearer(event){
  return (event.contact_side === "left" || event.contact_side === "right") && event.contact_side_basis === "wearer_limb";
}

function sideBasisNeedsReview(event){
  const side = event.contact_side || "unknown";
  const basis = event.contact_side_basis || "unknown";
  return (side === "left" || side === "right") && (basis === "unknown" || basis === "legacy_unspecified");
}

function hasUsefulContactLabel(event){
  const side = event.contact_side || "unknown";
  const sideBasis = event.contact_side_basis || "unknown";
  const type = event.contact_type || "unknown";
  const surface = event.contact_surface || "unknown";
  const trainableSide = (side === "left" || side === "right") && sideBasis === "wearer_limb";
  return trainableSide || surface === "inner" || surface === "outer" || !["", "unknown"].includes(type) || Boolean(event.trick_label);
}

function eventContactClassified(event){
  return event.contact_review_status === "reviewed" || hasUsefulContactLabel(event);
}

function contactClassificationStats(){
  const events = classifiableEvents();
  const classified = events.filter(eventContactClassified);
  const usable = events.filter(hasUsefulContactLabel);
  const wearerSide = events.filter(sideBasisIsExplicitWearer);
  const sideBasisNeeds = events.filter(sideBasisNeedsReview);
  return {
    total: events.length,
    classified: classified.length,
    unclassified: Math.max(0, events.length - classified.length),
    usable: usable.length,
    wearerSide: wearerSide.length,
    sideBasisNeeds: sideBasisNeeds.length
  };
}

function renderClassificationPanel(){
  const item = selectedEvent();
  const stats = contactClassificationStats();
  const headline = document.getElementById("classificationHeadline");
  const copy = document.getElementById("classificationCopy");
  const progress = document.getElementById("classificationProgress");
  const buttons = [
    "classifyLeftKick",
    "classifyLeftOuterKick",
    "classifyLeftInnerKick",
    "classifyRightKick",
    "classifyRightInnerKick",
    "classifyRightOuterKick",
    "classifyLeftKnee",
    "classifyLeftOuterKnee",
    "classifyLeftInnerKnee",
    "classifyRightKnee",
    "classifyRightInnerKnee",
    "classifyRightOuterKnee",
    "classifyLeftStall",
    "classifyLeftInnerStall",
    "classifyLeftOuterStall",
    "classifyRightStall",
    "classifyRightInnerStall",
    "classifyRightOuterStall",
    "classifyGround",
    "classifyUnknown",
    "clearClassification"
  ].map(id => document.getElementById(id)).filter(Boolean);
  buttons.forEach(button => { button.disabled = !item; });
  const nextButton = document.getElementById("nextUnclassifiedEvent");
  if(nextButton) nextButton.disabled = stats.unclassified === 0;
  const nextSideBasisButton = document.getElementById("nextSideBasisReview");
  if(nextSideBasisButton) nextSideBasisButton.disabled = stats.sideBasisNeeds === 0;
  if(progress) progress.textContent = `${stats.classified}/${stats.total} classified · ${stats.wearerSide} wearer-side · ${stats.sideBasisNeeds} side-basis todo`;
  if(!item){
    if(headline) headline.textContent = "Select an event to classify";
    if(copy) copy.textContent = stats.unclassified ? `${stats.unclassified} events still need a side/type decision.` : "No classifiable events in this clip.";
    return;
  }
  const state = eventContactClassified(item) ? "classified" : "needs class";
  if(headline) headline.textContent = `${eventLabel(item)} · ${state}`;
  if(copy) copy.textContent = sideBasisNeedsReview(item)
    ? "This event has a legacy left/right label. Confirm it is the contacting limb side, or mark it ambiguous/unknown."
    : "Use one button. The UI saves a draft and moves to the next unclassified event.";
}

function setSelectedEventIndex(index, replay=false, message=null){
  if(index < 0 || index >= app.events.length) return false;
  app.selected = index;
  const item = app.events[index];
  if(replay){
    playEventWindow(item, message || `classify ${fmt(item.time_sec)}s`);
  } else {
    video.pause();
    video.currentTime = Number(item.time_sec);
    setStatus(message || `event ${fmt(item.time_sec)}s`);
    renderAll();
  }
  return true;
}

function nextUnclassifiedEventIndex(){
  if(!app.events.length) return -1;
  const start = app.selected === null || app.selected === undefined ? -1 : app.selected;
  for(let offset = 1; offset <= app.events.length; offset += 1){
    const index = (start + offset + app.events.length) % app.events.length;
    const event = app.events[index];
    if(["touch", "stall", "drop_floor"].includes(event.type) && !eventContactClassified(event)) return index;
  }
  return -1;
}

function nextSideBasisReviewIndex(){
  if(!app.events.length) return -1;
  const start = app.selected === null || app.selected === undefined ? -1 : app.selected;
  for(let offset = 1; offset <= app.events.length; offset += 1){
    const index = (start + offset + app.events.length) % app.events.length;
    const event = app.events[index];
    if(["touch", "stall"].includes(event.type) && sideBasisNeedsReview(event)) return index;
  }
  return -1;
}

function stepUnclassifiedEvent(){
  const index = nextUnclassifiedEventIndex();
  if(index < 0){
    setStatus("all events classified or intentionally skipped");
    renderAll();
    return false;
  }
  return setSelectedEventIndex(index, true, "next unclassified event");
}

function stepSideBasisReview(){
  const index = nextSideBasisReviewIndex();
  if(index < 0){
    setStatus("no legacy side labels need side-basis review");
    renderAll();
    return false;
  }
  return setSelectedEventIndex(index, true, "next side-basis review");
}

function playEventWindow(event, message){
  const center = Number(event.time_sec);
  app.replayCenter = center;
  app.replayUntil = center + REVIEW_WINDOW_SEC;
  video.currentTime = Math.max(0, center - REVIEW_WINDOW_SEC);
  video.playbackRate = Number(document.getElementById("playbackRate").value || 0.5);
  enforceMuted();
  setStatus(message || `replay event ${fmt(center)}s`);
  renderAll();
  const promise = video.play();
  if(promise && typeof promise.catch === "function"){
    promise.catch(() => {
      app.replayUntil = null;
      app.replayCenter = null;
      video.currentTime = center;
      setStatus(`ready ${fmt(center)}s`);
      renderAll();
    });
  }
}

function classificationPreset(preset){
  if(preset === "left_kick") return {contact_side:"left", contact_side_basis:"wearer_limb", contact_type:"kick", contact_surface:"unknown", trick_label:"left_kick"};
  if(preset === "left_inner_kick") return {contact_side:"left", contact_side_basis:"wearer_limb", contact_type:"kick", contact_surface:"inner", trick_label:"left_inner_kick"};
  if(preset === "left_outer_kick") return {contact_side:"left", contact_side_basis:"wearer_limb", contact_type:"kick", contact_surface:"outer", trick_label:"left_outer_kick"};
  if(preset === "right_kick") return {contact_side:"right", contact_side_basis:"wearer_limb", contact_type:"kick", contact_surface:"unknown", trick_label:"right_kick"};
  if(preset === "right_inner_kick") return {contact_side:"right", contact_side_basis:"wearer_limb", contact_type:"kick", contact_surface:"inner", trick_label:"right_inner_kick"};
  if(preset === "right_outer_kick") return {contact_side:"right", contact_side_basis:"wearer_limb", contact_type:"kick", contact_surface:"outer", trick_label:"right_outer_kick"};
  if(preset === "left_knee") return {contact_side:"left", contact_side_basis:"wearer_limb", contact_type:"knee", contact_surface:"unknown", trick_label:"left_knee"};
  if(preset === "left_inner_knee") return {contact_side:"left", contact_side_basis:"wearer_limb", contact_type:"knee", contact_surface:"inner", trick_label:"left_inner_knee"};
  if(preset === "left_outer_knee") return {contact_side:"left", contact_side_basis:"wearer_limb", contact_type:"knee", contact_surface:"outer", trick_label:"left_outer_knee"};
  if(preset === "right_knee") return {contact_side:"right", contact_side_basis:"wearer_limb", contact_type:"knee", contact_surface:"unknown", trick_label:"right_knee"};
  if(preset === "right_inner_knee") return {contact_side:"right", contact_side_basis:"wearer_limb", contact_type:"knee", contact_surface:"inner", trick_label:"right_inner_knee"};
  if(preset === "right_outer_knee") return {contact_side:"right", contact_side_basis:"wearer_limb", contact_type:"knee", contact_surface:"outer", trick_label:"right_outer_knee"};
  if(preset === "left_stall") return {contact_side:"left", contact_side_basis:"wearer_limb", contact_type:"stall", contact_surface:"unknown", trick_label:"left_stall"};
  if(preset === "left_inner_stall") return {contact_side:"left", contact_side_basis:"wearer_limb", contact_type:"stall", contact_surface:"inner", trick_label:"left_inner_stall"};
  if(preset === "left_outer_stall") return {contact_side:"left", contact_side_basis:"wearer_limb", contact_type:"stall", contact_surface:"outer", trick_label:"left_outer_stall"};
  if(preset === "right_stall") return {contact_side:"right", contact_side_basis:"wearer_limb", contact_type:"stall", contact_surface:"unknown", trick_label:"right_stall"};
  if(preset === "right_inner_stall") return {contact_side:"right", contact_side_basis:"wearer_limb", contact_type:"stall", contact_surface:"inner", trick_label:"right_inner_stall"};
  if(preset === "right_outer_stall") return {contact_side:"right", contact_side_basis:"wearer_limb", contact_type:"stall", contact_surface:"outer", trick_label:"right_outer_stall"};
  if(preset === "ground") return {contact_side:"unknown", contact_side_basis:"unknown", contact_type:"ground", contact_surface:"unknown", trick_label:""};
  return {contact_side:"unknown", contact_side_basis:"ambiguous", contact_type:"unknown", contact_surface:"unknown", trick_label:""};
}

function classifySelectedEvent(preset, advance=true){
  const item = selectedEvent();
  if(!item){
    stepUnclassifiedEvent();
    return false;
  }
  pushUndo(`classify ${preset}`);
  const values = classificationPreset(preset);
  item.contact_side = values.contact_side;
  item.contact_side_basis = values.contact_side_basis;
  item.contact_type = values.contact_type;
  item.contact_surface = values.contact_surface;
  item.trick_label = values.trick_label;
  item.contact_review_status = "reviewed";
  markDirty(`classified ${preset}`);
  if(advance) stepUnclassifiedEvent();
  else renderAll();
  return true;
}

function clearSelectedClassification(){
  const item = selectedEvent();
  if(!item) return;
  pushUndo("clear classification");
  const defaults = contactDefaultsForType(item.type);
  item.contact_side = defaults.contact_side;
  item.contact_side_basis = defaults.contact_side_basis || "unknown";
  item.contact_type = defaults.contact_type;
  item.contact_surface = defaults.contact_surface;
  item.trick_label = defaults.trick_label;
  item.contact_review_status = "unreviewed";
  markDirty("classification cleared");
  renderAll();
}

function renderCandidates(){
  const root = document.getElementById("candidateList");
  const candidates = filteredCandidates();
  const allCount = allCandidates().length;
  const filteredCount = candidates.length;
  const likelyTotal = likelyCandidates().length;
  const likelyUnchecked = likelyUncheckedCandidates().length;
  const likelyText = likelyTotal ? `${likelyUnchecked}/${likelyTotal} likely unchecked` : "no model-touch hints";
  document.getElementById("likelyReadout").textContent = `${filteredCount}/${allCount} shown · ${likelyText}`;
  renderUndoControls();
  root.innerHTML = "";
  if(!candidates.length){
    root.innerHTML = `<div class="empty">No hints match this filter.</div>`;
    return;
  }
  candidates.forEach(c => {
    const review = reviewForCandidate(c);
    const nearEvent = approvedEventNear(c);
    const state = review ? review.decision : nearEvent ? nearEvent.type : "unchecked";
    const b = document.createElement("button");
    b.className = review ? (review.decision === "touch" ? "touch-reviewed reviewed" : "reviewed") : "";
    const priority = c.priority_score > 0 ? ` · ${c.priority_label}` : "";
    const strength = c.strength || c.audio_strength;
    b.innerHTML = `<span>${fmt(c.time_sec)}s</span><span>${state} · ${c.source}${priority}${strength ? " " + Number(strength).toFixed(2) : ""}</span>`;
    b.onclick = () => { video.currentTime = Number(c.time_sec); };
    root.appendChild(b);
  });
}

function renderReviewProgress(){
  const root = document.getElementById("reviewProgress");
  const stats = candidateReviewStats();
  const classStats = contactClassificationStats();
  const completeClass = stats.unchecked === 0 ? "ready" : "blocked";
  const completeText = stats.unchecked === 0 ? "ready" : `${stats.unchecked} left`;
  const saveComplete = document.getElementById("saveComplete");
  const saveCompleteNext = document.getElementById("saveCompleteNext");
  const completeBlocked = stats.unchecked > 0;
  [saveComplete, saveCompleteNext].forEach(button => {
    if(!button) return;
    button.disabled = completeBlocked;
    button.title = completeBlocked ? `Review ${stats.unchecked} remaining hints before saving complete.` : "All hints checked; this clip can be saved complete.";
  });
  root.innerHTML = `
    <div class="progress-row"><span>All hints checked</span><strong>${stats.checked}/${stats.total}</strong></div>
    <div class="progress-row"><span>Likely/model checked</span><strong>${stats.likelyChecked}/${stats.likelyTotal}</strong></div>
    <div class="progress-row"><span>Audio-only checked</span><strong>${stats.audioOnlyChecked}/${stats.audioOnlyTotal}</strong></div>
    <div class="progress-row"><span>Touch / no-touch reviews</span><strong>${stats.touchReviews}/${stats.noTouchReviews}</strong></div>
    <div class="progress-row ${classStats.unclassified === 0 ? "ready" : "blocked"}"><span>v1.0 side/type classified</span><strong>${classStats.classified}/${classStats.total}</strong></div>
    <div class="progress-row ${classStats.sideBasisNeeds === 0 ? "ready" : "blocked"}"><span>Wearer-side basis labels</span><strong>${classStats.wearerSide} usable · ${classStats.sideBasisNeeds} legacy left/right</strong></div>
    <div class="progress-row ${completeClass}"><span>Save complete</span><strong>${completeText}</strong></div>
  `;
}

function renderReviewSheets(){
  const root = document.getElementById("reviewSheets");
  const sheets = app.videoPayload?.contact_sheets || [];
  if(!sheets.length){
    root.innerHTML = `<div class="empty">No sheets yet. Run build_touch_review_contact_sheets.py.</div>`;
    return;
  }
  root.innerHTML = sheets.map((sheet, index) => {
    const sheetTimes = Array.isArray(sheet.times_sec) ? sheet.times_sec.map(Number).filter(Number.isFinite) : [];
    const firstTime = sheetTimes.length ? sheetTimes[0] : null;
    const jumpButton = firstTime === null ? "" : `<button data-sheet-jump="${firstTime.toFixed(3)}">jump</button>`;
    const reviewedTimes = sheetTimes.filter(time => Boolean(reviewForCandidate({time_sec: time})));
    const sheetTotal = sheetTimes.length || Number(sheet.candidate_count || 0);
    const countText = sheetTotal ? `${reviewedTimes.length}/${sheetTotal} reviewed · ${sheetTotal} candidates` : (sheet.filter || "sheet");
    const noTouchButton = sheet.filter === "audio_only" && sheetTimes.length
      ? `<button data-sheet-no-touch="${sheetTimes.map(time => time.toFixed(3)).join(",")}">no-touch page</button>`
      : "";
    const timeButtons = sheetTimes.map((time, timeIndex) => {
      const review = reviewForCandidate({time_sec: time});
      const decisionClass = review ? (review.decision === "touch" ? "touch" : "no-touch") : "";
      const stateClass = review ? ` reviewed ${decisionClass}` : "";
      const labelPrefix = review ? (review.decision === "touch" ? "T" : "N") : `${timeIndex + 1}`;
      return `<button class="sheet-time${stateClass}" data-sheet-time="${time.toFixed(3)}">${labelPrefix}:${time.toFixed(3)}</button>`;
    }).join("");
    return `<div class="sheet-row">
      <a href="${sheet.url}" target="_blank" rel="noopener">
        <span>${sheet.label || "Contact sheet"}<br><small>${countText}</small></span>
      </a>
      <span class="sheet-actions">${jumpButton}${noTouchButton}<strong>${index + 1}/${sheets.length}</strong></span>
      <div class="sheet-time-grid">${timeButtons}</div>
    </div>`;
  }).join("");
  root.querySelectorAll("[data-sheet-jump], [data-sheet-time]").forEach(button => {
    button.onclick = (event) => {
      const target = Number(
        event.currentTarget.getAttribute("data-sheet-jump") ||
        event.currentTarget.getAttribute("data-sheet-time")
      );
      video.currentTime = target;
      setStatus(`sheet jump ${fmt(target)}s`);
      renderAll();
    };
  });
  root.querySelectorAll("[data-sheet-no-touch]").forEach(button => {
    button.onclick = (event) => {
      sheetTimesNoTouch(event.currentTarget.getAttribute("data-sheet-no-touch"));
    };
  });
}

function renderReviewMontages(){
  const root = document.getElementById("reviewMontages");
  const montages = app.videoPayload?.review_montages || [];
  if(!montages.length){
    root.innerHTML = `<div class="empty">No montages yet. Run build_touch_review_montages.py.</div>`;
    return;
  }
  root.innerHTML = montages.map((montage, index) => {
    const countText = montage.candidate_count ? `${montage.candidate_count} candidates` : (montage.filter || "montage");
    const duration = montage.duration_sec ? ` · ${Number(montage.duration_sec).toFixed(1)}s` : "";
    return `<div class="sheet-row">
      <a href="${montage.url}" target="_blank" rel="noopener">
        <span>${montage.label || "Montage"}<br><small>${countText}${duration}</small></span>
      </a>
      <span class="sheet-actions"><strong>${index + 1}/${montages.length}</strong></span>
    </div>`;
  }).join("");
}

function renderAll(){
  renderVideoList();
  renderWorkflowGuide();
  renderTimeline();
  renderTrackGraph();
  renderEvents();
  renderCandidates();
  renderReviewProgress();
  renderReviewMontages();
  renderReviewSheets();
  document.getElementById("timeReadout").textContent = `${fmt(video.currentTime || 0)}s`;
}

function addEvent(type, advance=false){
  const time = Number(video.currentTime || 0);
  pushUndo(`add ${type}`);
  const row = applyContactDefaults({ type, time_sec: Number(time.toFixed(3)), review_status:"approved", source:"muted_visual_review" });
  if(type === "stall") row.duration_sec = 0.5;
  app.events.push(row);
  markDirty();
  if(type === "touch"){
    const candidate = nearestFilteredCandidate();
    if(candidate && nearestCandidateDistance(candidate) <= HINT_ACTION_TOL_SEC){
      setCandidateReviewFor(candidate, "touch");
    }
  }
  if(type === "drop_floor" || type === "stall"){
    const candidate = nearestFilteredCandidate();
    if(candidate && nearestCandidateDistance(candidate) <= HINT_ACTION_TOL_SEC){
      setCandidateReviewFor(candidate, "no_touch");
    }
  }
  app.selected = app.events.length - 1;
  if(advance) stepUncheckedCandidate(1);
  renderAll();
}

function addTouchForNearestHint(advance=false){
  const candidate = nearestCandidate();
  if(!candidate) return false;
  if(nearestCandidateDistance(candidate) > HINT_ACTION_TOL_SEC){
    setStatus(`no hint within ${HINT_ACTION_TOL_SEC.toFixed(2)}s`);
    return false;
  }
  return addTouchForCandidate(candidate, advance);
}

function addTouchForCandidate(candidate, advance=false){
  if(!candidate) return false;
  pushUndo("mark touch");
  const time = Number(candidate.time_sec);
  const row = applyContactDefaults({ type:"touch", time_sec: Number(time.toFixed(3)), review_status:"approved", source:"muted_visual_review" });
  app.events.push(row);
  app.selected = app.events.length - 1;
  setCandidateReviewFor(candidate, "touch");
  if(advance) stepAnyUncheckedCandidate();
  renderAll();
  return true;
}

async function save(candidateReviewComplete=false, advanceAfterComplete=false){
  const missing = uncoveredCandidates();
  if(candidateReviewComplete && missing.length){
    document.getElementById("status").textContent = `not saved complete: ${missing.length} unchecked hints remain`;
    return;
  }
  if(app.autosaveTimer){
    clearTimeout(app.autosaveTimer);
    app.autosaveTimer = null;
  }
  if(app.draftSaveInFlight) await app.draftSaveInFlight;
  const videoId = app.current.video_id;
  await api("/api/events", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({ video_id: videoId, events: app.events, candidate_reviews: app.candidateReviews, candidate_review_complete: candidateReviewComplete }) });
  app.dirty = false;
  document.getElementById("status").textContent = `${candidateReviewComplete ? "complete" : "draft"} saved ${new Date().toLocaleTimeString()}`;
  await loadState(false);
  if(candidateReviewComplete && advanceAfterComplete){
    const next = (app.state.review_queue || []).find(row => row.video_id !== videoId);
    if(next){
      await loadVideo(next.video_id);
      return;
    }
  }
  await loadVideo(videoId);
}

function stepCandidate(direction){
  const c = filteredCandidates();
  if(!c.length) return;
  const t = Number(video.currentTime || 0);
  const sorted = direction > 0 ? c : [...c].reverse();
  const next = sorted.find(x => direction > 0 ? Number(x.time_sec) > t + 0.03 : Number(x.time_sec) < t - 0.03) || sorted[0];
  cueCandidate(next, `hint ${fmt(next.time_sec)}s`);
}

function stepUncheckedCandidate(direction){
  const c = filteredCandidates().filter(candidate => !candidateCovered(candidate));
  if(!c.length){
    setStatus("no unchecked hints in current filter");
    return;
  }
  const t = Number(video.currentTime || 0);
  const sorted = direction > 0 ? c : [...c].reverse();
  const next = sorted.find(x => direction > 0 ? Number(x.time_sec) > t + 0.03 : Number(x.time_sec) < t - 0.03) || sorted[0];
  cueCandidate(next, `unchecked ${fmt(next.time_sec)}s`);
}

function stepAnyUncheckedCandidate(){
  const c = uncoveredCandidates();
  if(!c.length){
    setStatus("all hints checked");
    return;
  }
  const t = Number(video.currentTime || 0);
  const next = c.find(x => Number(x.time_sec) > t + 0.03) || c[0];
  cueCandidate(next, `unchecked ${fmt(next.time_sec)}s`);
}

function stepLikelyUncheckedCandidate(){
  const c = likelyUncheckedCandidates();
  if(!c.length){
    setStatus("all likely hints checked");
    return;
  }
  const sorted = [...c].sort((a, b) => {
    const priorityDelta = Number(b.priority_score || 0) - Number(a.priority_score || 0);
    if(priorityDelta !== 0) return priorityDelta;
    return Number(a.time_sec) - Number(b.time_sec);
  });
  const next = sorted[0];
  cueCandidate(next, `likely unchecked: ${fmt(next.time_sec)}s ${next.priority_label}`);
}

function stepTrackValley(direction){
  const valleys = app.trackMinima || [];
  if(!valleys.length){
    setStatus("no tracked-ball valleys for this clip");
    return;
  }
  const t = Number(video.currentTime || 0);
  const sorted = direction > 0 ? valleys : [...valleys].reverse();
  const next = sorted.find(x => direction > 0 ? Number(x.time_sec) > t + 0.03 : Number(x.time_sec) < t - 0.03) || sorted[0];
  video.pause();
  video.currentTime = Number(next.time_sec);
  setStatus(`valley ${fmt(next.time_sec)}s`);
  renderAll();
}

function replayHintWindow(){
  const candidate = nearestFilteredCandidate();
  if(!candidate){
    setStatus("no hint to replay");
    return;
  }
  playCandidateWindow(candidate, `replay ${fmt(candidate.time_sec)}s`);
}

function cueCandidate(candidate, message){
  if(app.autoReplay){
    playCandidateWindow(candidate, message);
    return;
  }
  const center = Number(candidate.time_sec);
  app.replayUntil = null;
  app.replayCenter = null;
  video.pause();
  video.currentTime = center;
  setStatus(message || `hint ${fmt(center)}s`);
  renderAll();
}

function playCandidateWindow(candidate, message){
  const center = Number(candidate.time_sec);
  app.replayCenter = center;
  app.replayUntil = center + REVIEW_WINDOW_SEC;
  video.currentTime = Math.max(0, center - REVIEW_WINDOW_SEC);
  video.playbackRate = Number(document.getElementById("playbackRate").value || 0.5);
  enforceMuted();
  setStatus(message || `replay ${fmt(center)}s`);
  renderAll();
  const promise = video.play();
  if(promise && typeof promise.catch === "function"){
    promise.catch(() => {
      app.replayUntil = null;
      app.replayCenter = null;
      video.currentTime = center;
      setStatus(`ready ${fmt(center)}s`);
      renderAll();
    });
  }
}

function frameStep(direction){
  app.replayUntil = null;
  app.replayCenter = null;
  video.pause();
  video.currentTime = Math.max(0, Number(video.currentTime || 0) + direction * FRAME_STEP_SEC);
  setStatus(`${direction > 0 ? "+" : "-"}1 frame`);
  renderAll();
}

function applyPlaybackRate(){
  video.playbackRate = Number(document.getElementById("playbackRate").value || 0.5);
}

function stopReplayIfNeeded(){
  if(app.replayUntil !== null && Number(video.currentTime || 0) >= app.replayUntil){
    video.pause();
    video.currentTime = app.replayCenter === null ? app.replayUntil : app.replayCenter;
    app.replayUntil = null;
    app.replayCenter = null;
    renderAll();
  }
}

document.getElementById("addTouch").onclick = () => addEvent("touch");
document.getElementById("markNoTouch").onclick = () => setNearestCandidateReview("no_touch");
document.getElementById("markNoTouchNext").onclick = () => setNearestCandidateReview("no_touch", true);
document.getElementById("addTouchNext").onclick = () => addTouchForNearestHint(true);
document.getElementById("clearHintReview").onclick = clearCandidateReview;
document.getElementById("quickClearHint").onclick = clearCandidateReview;
document.getElementById("undoLast").onclick = undoLastAction;
document.getElementById("addDrop").onclick = () => addEvent("drop_floor");
document.getElementById("addStall").onclick = () => addEvent("stall");
document.getElementById("saveDraft").onclick = () => save(false);
document.getElementById("saveComplete").onclick = () => save(true);
document.getElementById("saveCompleteNext").onclick = () => save(true, true);
function deleteSelectedEvent(){
  if(app.selected === null) return;
  pushUndo("delete selected event");
  app.events.splice(app.selected, 1);
  app.selected = null;
  markDirty("event deleted");
  renderAll();
}
document.getElementById("deleteEvent").onclick = deleteSelectedEvent;
document.getElementById("quickDeleteEvent").onclick = deleteSelectedEvent;
document.getElementById("quickSaveDraft").onclick = () => save(false);
document.getElementById("prevCandidate").onclick = () => stepCandidate(-1);
document.getElementById("nextCandidate").onclick = () => stepCandidate(1);
document.getElementById("nextLikelyUnchecked").onclick = stepLikelyUncheckedCandidate;
document.getElementById("nextUnchecked").onclick = stepAnyUncheckedCandidate;
document.getElementById("prevValley").onclick = () => stepTrackValley(-1);
document.getElementById("nextValley").onclick = () => stepTrackValley(1);
document.getElementById("candidateFilter").onchange = (event) => { app.candidateFilter = event.target.value; renderAll(); };
document.getElementById("bulkNoTouchFiltered").onclick = bulkNoTouchFiltered;
document.getElementById("bulkAudioTailComplete").onclick = bulkAudioTailAndSaveComplete;
document.getElementById("replayHint").onclick = replayHintWindow;
document.getElementById("autoReplay").onchange = (event) => { app.autoReplay = Boolean(event.target.checked); };
document.getElementById("frameBack").onclick = () => frameStep(-1);
document.getElementById("frameForward").onclick = () => frameStep(1);
document.getElementById("selectedContactSide").onchange = (event) => updateSelectedContactField("contact_side", event.target.value);
document.getElementById("selectedContactSideBasis").onchange = (event) => updateSelectedContactField("contact_side_basis", event.target.value);
document.getElementById("selectedContactType").onchange = (event) => updateSelectedContactField("contact_type", event.target.value);
document.getElementById("selectedContactSurface").onchange = (event) => updateSelectedContactField("contact_surface", event.target.value);
document.getElementById("selectedTrickLabel").onchange = (event) => updateSelectedContactField("trick_label", event.target.value);
document.getElementById("classifyLeftKick").onclick = () => classifySelectedEvent("left_kick");
document.getElementById("classifyRightKick").onclick = () => classifySelectedEvent("right_kick");
document.getElementById("classifyLeftOuterKick").onclick = () => classifySelectedEvent("left_outer_kick");
document.getElementById("classifyLeftInnerKick").onclick = () => classifySelectedEvent("left_inner_kick");
document.getElementById("classifyRightInnerKick").onclick = () => classifySelectedEvent("right_inner_kick");
document.getElementById("classifyRightOuterKick").onclick = () => classifySelectedEvent("right_outer_kick");
document.getElementById("classifyLeftKnee").onclick = () => classifySelectedEvent("left_knee");
document.getElementById("classifyRightKnee").onclick = () => classifySelectedEvent("right_knee");
document.getElementById("classifyLeftOuterKnee").onclick = () => classifySelectedEvent("left_outer_knee");
document.getElementById("classifyLeftInnerKnee").onclick = () => classifySelectedEvent("left_inner_knee");
document.getElementById("classifyRightInnerKnee").onclick = () => classifySelectedEvent("right_inner_knee");
document.getElementById("classifyRightOuterKnee").onclick = () => classifySelectedEvent("right_outer_knee");
document.getElementById("classifyLeftStall").onclick = () => classifySelectedEvent("left_stall");
document.getElementById("classifyLeftInnerStall").onclick = () => classifySelectedEvent("left_inner_stall");
document.getElementById("classifyLeftOuterStall").onclick = () => classifySelectedEvent("left_outer_stall");
document.getElementById("classifyRightStall").onclick = () => classifySelectedEvent("right_stall");
document.getElementById("classifyRightInnerStall").onclick = () => classifySelectedEvent("right_inner_stall");
document.getElementById("classifyRightOuterStall").onclick = () => classifySelectedEvent("right_outer_stall");
document.getElementById("classifyGround").onclick = () => classifySelectedEvent("ground");
document.getElementById("classifyUnknown").onclick = () => classifySelectedEvent("unknown");
document.getElementById("nextUnclassifiedEvent").onclick = stepUnclassifiedEvent;
document.getElementById("nextSideBasisReview").onclick = stepSideBasisReview;
document.getElementById("clearClassification").onclick = clearSelectedClassification;
document.getElementById("playbackRate").onchange = applyPlaybackRate;
video.addEventListener("timeupdate", () => { renderTimeline(); renderTrackGraph(); stopReplayIfNeeded(); });
trackGraph.addEventListener("click", (event) => {
  video.pause();
  video.currentTime = graphTimeFromEvent(event);
  setStatus(`graph scrub ${fmt(video.currentTime)}s`);
  renderAll();
});
video.addEventListener("volumechange", enforceMuted);
video.addEventListener("play", enforceMuted);
video.addEventListener("loadedmetadata", () => { enforceMuted(); renderAll(); });
window.addEventListener("beforeunload", (event) => {
  if(!app.dirty) return;
  event.preventDefault();
  event.returnValue = "";
});
window.addEventListener("keydown", (event) => {
  if(event.target && ["INPUT","TEXTAREA","SELECT"].includes(event.target.tagName)) return;
  if(event.key === "t") addEvent("touch");
  if(event.key === "T") addEvent("touch", true);
  if(event.key === "x") setCandidateReview("no_touch");
  if(event.key === "X") setCandidateReview("no_touch", true);
  if(event.key === "c") clearCandidateReview();
  if(event.key === "d") addEvent("drop_floor");
  if(event.key === "s" && !event.metaKey) { event.preventDefault(); addEvent("stall"); }
  if(event.key === "Enter") save(false);
  if(event.key === "S" && event.shiftKey) { event.preventDefault(); save(true); }
  if(event.key === "j") stepCandidate(-1);
  if(event.key === "k") stepCandidate(1);
  if(event.key === "l") stepLikelyUncheckedCandidate();
  if(event.key === "u") stepUncheckedCandidate(1);
  if(event.key === "v") stepTrackValley(1);
  if(event.key === "V") stepTrackValley(-1);
  if(event.key === "N") bulkNoTouchFiltered();
  if(event.key === "A" && event.shiftKey) { event.preventDefault(); bulkAudioTailAndSaveComplete(); }
  if(event.key === "r") replayHintWindow();
  if(event.key === "[") frameStep(-1);
  if(event.key === "]") frameStep(1);
  if(event.key === "1") { event.preventDefault(); classifySelectedEvent("left_kick"); }
  if(event.key === "2") { event.preventDefault(); classifySelectedEvent("left_inner_kick"); }
  if(event.key === "3") { event.preventDefault(); classifySelectedEvent("left_outer_kick"); }
  if(event.key === "4") { event.preventDefault(); classifySelectedEvent("right_kick"); }
  if(event.key === "5") { event.preventDefault(); classifySelectedEvent("right_inner_kick"); }
  if(event.key === "6") { event.preventDefault(); classifySelectedEvent("right_outer_kick"); }
  if(event.key === "7") { event.preventDefault(); classifySelectedEvent("left_knee"); }
  if(event.key === "8") { event.preventDefault(); classifySelectedEvent("right_knee"); }
  if(event.key === "9") { event.preventDefault(); classifySelectedEvent("left_stall"); }
  if(event.key === "g" || event.key === "G") { event.preventDefault(); classifySelectedEvent("ground"); }
  if(event.key === "0") { event.preventDefault(); classifySelectedEvent("unknown"); }
  if(event.key === "m") { event.preventDefault(); stepUnclassifiedEvent(); }
  if(event.key === " ") { event.preventDefault(); video.paused ? video.play() : video.pause(); }
});
loadState().catch(err => { document.body.innerHTML = `<pre>${err.stack || err}</pre>`; });
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run muted visual touch review app")
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW_MANIFEST)
    parser.add_argument("--candidates-dir", type=Path, default=DEFAULT_CANDIDATES_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--contact-sheets-dir", type=Path, default=DEFAULT_CONTACT_SHEETS_DIR)
    parser.add_argument("--montages-dir", type=Path, default=DEFAULT_MONTAGES_DIR)
    parser.add_argument("--track-root", type=Path, action="append", default=[], help="Detector track root containing <video-stem>/detector_track.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8892)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    track_roots = args.track_root or DEFAULT_TRACK_ROOTS
    store = TouchReviewStore(args.review_manifest, args.candidates_dir, args.labels_dir, args.contact_sheets_dir, args.montages_dir, track_roots)
    port = find_free_port(args.host, args.port)
    server = ThreadingHTTPServer((args.host, port), TouchReviewHandler)
    server.store = store  # type: ignore[attr-defined]
    print(f"touch review app: http://{args.host}:{port}")
    print(f"labels: {store.labels_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
