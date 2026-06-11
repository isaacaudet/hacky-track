#!/usr/bin/env python3
"""Audit whether QA event ball centers visually land on the footbag.

This is a verification layer, not another detector pass. It samples the QA
events, opens the actual frame used for each event, checks whether color/shape
evidence supports the stored ball center, and writes metrics plus failure
contact sheets. The goal is to make ball-center quality measurable instead of
only inspectable by hand.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from detect_atw_overlay import detect_ball, red_ball_mask
from paint_hud import detect_bag_center


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "outputs" / "full_training_27_qa" / "qa_manifest.json"
DEFAULT_OUT_DIR = ROOT / "outputs" / "ball_tracking_audit"
OUT_SIZE = (688, 912)
AUDIT_KINDS = {"touch", "drop_floor", "stall"}
TILE_W = 330
TILE_H = 382
CROP_W = 300
CROP_H = 244
SHEET_COLUMNS = 4


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def event_time(event: dict[str, Any]) -> float:
    return float(event.get("time_sec", event.get("start_sec", 0.0)) or 0.0)


def frame_time(event: dict[str, Any]) -> float:
    return float(event.get("qa_frame_time_sec", event_time(event)) or 0.0)


def resolve_event_path(run: dict[str, Any]) -> Path:
    raw = str(run.get("qa_events_path") or "")
    path = ROOT / raw
    if path.exists():
        return path
    return Path(raw)


def resolve_video_path(run: dict[str, Any], source_video: str) -> Path:
    raw = str(run.get("video") or "")
    path = Path(raw)
    if path.exists():
        return path
    downloads = Path.home() / "Downloads" / Path(source_video).name
    return downloads if downloads.exists() else path


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def independent_blob_center(frame: np.ndarray, qa_center: tuple[float, float]) -> tuple[float | None, float | None, float, str, int]:
    """Find local color evidence near the proposed QA ball center."""

    qx, qy = qa_center
    red = red_ball_mask(frame)
    h, w = red.shape[:2]
    gate = np.zeros_like(red)
    cv2.circle(gate, (int(round(qx)), int(round(qy))), 72, 255, -1)
    local = cv2.bitwise_and(red, gate)
    contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[float, float, float, int] | None = None
    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area < 8 or area > 5200:
            continue
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius < 3 or radius > 65:
            continue
        dist = math.hypot(float(x) - qx, float(y) - qy)
        score = area * math.exp(-(dist * dist) / (2 * 44 * 44)) / max(radius, 1.0)
        if best is None or score > best[0]:
            best = (score, float(x), float(y), area)

    if best is not None:
        _score, x, y, area = best
        return x, y, math.hypot(x - qx, y - qy), "local_red_mask", area

    detected, conf = detect_ball(frame, (qx, qy, 18.0))
    if detected is not None and conf >= 0.05:
        x, y, _r = detected
        return float(x), float(y), math.hypot(float(x) - qx, float(y) - qy), "detect_ball_near_qa", int(conf * 1000)

    center = detect_bag_center(frame)
    if center is not None:
        x, y = center
        return float(x), float(y), math.hypot(float(x) - qx, float(y) - qy), "global_color_blob", 0

    return None, None, math.inf, "missing_visual_blob", 0


def local_pixel_support(frame: np.ndarray, qa_center: tuple[float, float]) -> tuple[int, int]:
    qx, qy = qa_center
    mask = red_ball_mask(frame)
    h, w = mask.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    dist2 = (xx - qx) ** 2 + (yy - qy) ** 2
    near24 = int(np.count_nonzero((mask > 0) & (dist2 <= 24 * 24)))
    near42 = int(np.count_nonzero((mask > 0) & (dist2 <= 42 * 42)))
    return near24, near42


def audit_status(
    record: dict[str, Any],
    frame: np.ndarray | None,
) -> tuple[str, dict[str, Any]]:
    qx = maybe_float(record.get("qa_ball_x"))
    qy = maybe_float(record.get("qa_ball_y"))
    if frame is None:
        return "uncertain", {"reason": "missing_frame"}
    if qx is None or qy is None:
        return "uncertain", {"reason": "missing_qa_center"}
    h, w = frame.shape[:2]
    if qx < 0 or qx >= w or qy < 0 or qy >= h:
        return "fail", {"reason": "qa_center_out_of_frame"}

    blob_x, blob_y, dist, source, blob_area = independent_blob_center(frame, (qx, qy))
    near24, near42 = local_pixel_support(frame, (qx, qy))
    accuracy = str(record.get("qa_ball_accuracy") or "")
    radius = maybe_float(record.get("qa_ball_radius")) or 18.0
    tight_dist = max(18.0, min(28.0, radius * 1.25))
    loose_dist = max(24.0, min(36.0, radius * 1.75))
    evidence = {
        "reason": source,
        "independent_x": None if blob_x is None else round(blob_x, 2),
        "independent_y": None if blob_y is None else round(blob_y, 2),
        "independent_distance_px": None if math.isinf(dist) else round(float(dist), 2),
        "blob_area": blob_area,
        "local_red_pixels_r24": near24,
        "local_red_pixels_r42": near42,
        "audit_tight_distance_px": round(tight_dist, 2),
        "audit_loose_distance_px": round(loose_dist, 2),
    }

    # A pass should prove the stored QA center itself lands on the sack, not
    # merely that a red/orange blob exists somewhere nearby.
    if near24 >= 18:
        return "pass", evidence
    if (
        str(record.get("kind") or "") == "drop_floor"
        and source == "global_color_blob"
        and dist <= tight_dist
        and qy >= h * 0.88
    ):
        evidence["reason"] = "near_floor_global_color_blob"
        return "pass", evidence
    if dist <= tight_dist and near42 >= 12:
        return "pass", evidence
    if dist <= loose_dist and near24 >= 6 and near42 >= 36:
        return "pass", evidence
    if near24 == 0 and near42 >= 24:
        evidence["reason"] = f"{source}_nearby_blob_center_offset"
        return "uncertain", evidence
    if accuracy == "low":
        return "uncertain", evidence
    if source == "missing_visual_blob" and near42 < 8:
        return "fail", evidence
    return "uncertain", evidence


def collect_records(manifest_path: Path, min_confidence: float) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path)
    records: list[dict[str, Any]] = []
    for video_index, run in enumerate(manifest.get("runs", [])):
        event_path = resolve_event_path(run)
        if not event_path.exists():
            continue
        doc = read_json(event_path)
        source_video = str(doc.get("source_video") or Path(str(run.get("video") or event_path.parent.name)).name)
        video_path = resolve_video_path(run, source_video)
        for event_index, event in enumerate(doc.get("events", [])):
            kind = str(event.get("type") or "")
            if kind not in AUDIT_KINDS:
                continue
            confidence = maybe_float(event.get("confidence"))
            if confidence is None or confidence < min_confidence:
                continue
            records.append(
                {
                    "audit_item_id": f"{video_index:02d}-{event_index:04d}",
                    "source_video": source_video,
                    "video_path": str(video_path),
                    "qa_events_path": rel(event_path),
                    "video_index": video_index,
                    "event_index": event_index,
                    "kind": kind,
                    "time_sec": round(event_time(event), 3),
                    "frame_time_sec": round(frame_time(event), 3),
                    "confidence": confidence,
                    "qa_ball_x": event.get("qa_ball_x", event.get("x")),
                    "qa_ball_y": event.get("qa_ball_y", event.get("y")),
                    "qa_ball_radius": event.get("qa_ball_radius"),
                    "qa_ball_accuracy": event.get("qa_ball_accuracy"),
                    "qa_ball_source": event.get("qa_ball_source"),
                    "qa_ball_correction_px": event.get("qa_ball_correction_px"),
                    "contact_side": event.get("contact_side"),
                    "contact_type": event.get("contact_type"),
                    "contact_confidence": event.get("contact_confidence"),
                    "label": event.get("label"),
                    "note": event.get("note"),
                }
            )
    return records


def sample_records(records: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]]:
    if len(records) <= max_items:
        return sorted(records, key=lambda item: (item["video_index"], item["time_sec"], item["event_index"]))
    # Keep a balanced deterministic sample: top risks first, then chronological fill.
    def risk(item: dict[str, Any]) -> tuple[float, int, float]:
        correction = maybe_float(item.get("qa_ball_correction_px")) or 0.0
        kind_bonus = {"drop_floor": 0.5, "stall": 0.25, "touch": 0.0}.get(str(item.get("kind")), 0.0)
        accuracy_bonus = 0.9 if item.get("qa_ball_accuracy") == "low" else 0.35 if item.get("qa_ball_accuracy") == "medium" else 0.0
        return (kind_bonus + accuracy_bonus + min(1.2, correction / 130.0), -int(item["video_index"]), -float(item["time_sec"]))

    selected: dict[str, dict[str, Any]] = {}
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        by_video[str(item["source_video"])].append(item)
    for video_records in by_video.values():
        for item in sorted(video_records, key=risk, reverse=True)[:4]:
            selected[item["audit_item_id"]] = item
    for item in sorted(records, key=risk, reverse=True):
        if len(selected) >= max_items:
            break
        selected[item["audit_item_id"]] = item
    return sorted(selected.values(), key=lambda item: (item["video_index"], item["time_sec"], item["event_index"]))


def read_frame(cap: cv2.VideoCapture, fps: float, time_sec: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(time_sec * fps))))
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.resize(frame, OUT_SIZE, interpolation=cv2.INTER_AREA)


def crop_pil(frame: np.ndarray, record: dict[str, Any]) -> tuple[Image.Image, tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    qx = maybe_float(record.get("qa_ball_x")) or w / 2
    qy = maybe_float(record.get("qa_ball_y")) or h * 0.72
    size = 330
    left = max(0, min(w - size, int(round(qx - size / 2))))
    top = max(0, min(h - size, int(round(qy - size / 2))))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb).crop((left, top, left + size, top + size)).resize((CROP_W, CROP_H), Image.Resampling.LANCZOS)
    return img, (left, top, left + size, top + size)


def draw_marker(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    x: Any,
    y: Any,
    color: tuple[int, int, int],
    *,
    radius: int = 13,
) -> None:
    fx = maybe_float(x)
    fy = maybe_float(y)
    if fx is None or fy is None:
        return
    left, top, right, bottom = box
    px = (fx - left) * CROP_W / (right - left)
    py = (fy - top) * CROP_H / (bottom - top)
    if px < -radius or px > CROP_W + radius or py < -radius or py > CROP_H + radius:
        return
    draw.ellipse((px - radius, py - radius, px + radius, py + radius), outline=(0, 0, 0), width=5)
    draw.ellipse((px - radius, py - radius, px + radius, py + radius), outline=color, width=3)


def tile_for_result(index: int, result: dict[str, Any], frame: np.ndarray | None, fonts: dict[str, ImageFont.ImageFont]) -> Image.Image:
    tile = Image.new("RGB", (TILE_W, TILE_H), (20, 22, 25))
    draw = ImageDraw.Draw(tile)
    status = str(result["audit_status"])
    color = {"pass": (105, 228, 130), "uncertain": (255, 211, 82), "fail": (255, 93, 86)}.get(status, (220, 220, 220))
    draw.rectangle((0, 0, TILE_W, 7), fill=color)
    draw.text((10, 13), f"#{index:03d} {status.upper()} {result['kind']} {result['time_sec']:.2f}s", fill=(248, 248, 248), font=fonts["bold"])
    draw.text((10, 35), str(result["source_video"])[:42], fill=(168, 176, 186), font=fonts["small"])
    if frame is None:
        draw.rectangle((14, 58, 14 + CROP_W, 58 + CROP_H), fill=(5, 6, 8), outline=(76, 82, 92))
        draw.text((36, 166), "missing frame", fill=(255, 93, 86), font=fonts["body"])
    else:
        crop, box = crop_pil(frame, result)
        crop_draw = ImageDraw.Draw(crop)
        draw_marker(crop_draw, box, result.get("qa_ball_x"), result.get("qa_ball_y"), (255, 235, 40), radius=15)
        draw_marker(crop_draw, box, result.get("independent_x"), result.get("independent_y"), (70, 190, 255), radius=10)
        tile.paste(crop, (14, 58))
        draw.rectangle((14, 58, 14 + CROP_W, 58 + CROP_H), outline=(76, 82, 92), width=1)
    meta = (
        f"dist {result.get('independent_distance_px')} "
        f"pix24 {result.get('local_red_pixels_r24')} "
        f"conf {result.get('confidence'):.2f}"
    )
    draw.text((10, 315), meta[:52], fill=(238, 238, 238), font=fonts["small"])
    draw.text((10, 335), f"{result.get('contact_side')}/{result.get('contact_type')}"[:46], fill=(172, 208, 255), font=fonts["small"])
    draw.text((10, 355), "yellow=QA blue=independent", fill=(128, 136, 146), font=fonts["tiny"])
    return tile


def write_sheet(results: list[dict[str, Any]], frames: dict[str, np.ndarray | None], out_path: Path, *, max_items: int = 80) -> None:
    items = results[:max_items]
    if not items:
        return
    fonts = {
        "bold": load_font(15, bold=True),
        "body": load_font(14),
        "small": load_font(12),
        "tiny": load_font(10),
    }
    rows = math.ceil(len(items) / SHEET_COLUMNS)
    sheet = Image.new("RGB", (SHEET_COLUMNS * TILE_W, rows * TILE_H), (12, 13, 15))
    for index, item in enumerate(items, start=1):
        tile = tile_for_result(index, item, frames.get(str(item["audit_item_id"])), fonts)
        x = ((index - 1) % SHEET_COLUMNS) * TILE_W
        y = ((index - 1) // SHEET_COLUMNS) * TILE_H
        sheet.paste(tile, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def audit_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray | None]]:
    caps: dict[str, tuple[cv2.VideoCapture, float]] = {}
    frames: dict[str, np.ndarray | None] = {}
    results: list[dict[str, Any]] = []
    for record in records:
        video_path = str(record["video_path"])
        cap_tuple = caps.get(video_path)
        if cap_tuple is None:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap_tuple = (cap, fps)
            caps[video_path] = cap_tuple
        cap, fps = cap_tuple
        frame = read_frame(cap, fps, float(record["frame_time_sec"])) if cap.isOpened() else None
        status, evidence = audit_status(record, frame)
        result = {**record, **evidence, "audit_status": status}
        results.append(result)
        frames[str(record["audit_item_id"])] = frame
    for cap, _fps in caps.values():
        cap.release()
    return results, frames


def summarize(results: list[dict[str, Any]], total_high_confidence_events: int) -> dict[str, Any]:
    status_counts = Counter(str(item["audit_status"]) for item in results)
    kind_status: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    video_status: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    correction_counts = {"large_correction_events": 0, "huge_correction_events": 0}
    unsupported_confident = 0
    for item in results:
        kind_status[str(item["kind"])][str(item["audit_status"])] += 1
        video_status[str(item["source_video"])][str(item["audit_status"])] += 1
        correction = maybe_float(item.get("qa_ball_correction_px")) or 0.0
        if correction >= 75:
            correction_counts["large_correction_events"] += 1
        if correction >= 125:
            correction_counts["huge_correction_events"] += 1
        if item["audit_status"] == "fail" and float(item.get("confidence") or 0.0) >= 0.75:
            unsupported_confident += 1
    total = len(results)
    pass_count = status_counts.get("pass", 0)
    pass_rate = None if total == 0 else pass_count / total
    return {
        "sampled_events": total,
        "total_high_confidence_events": total_high_confidence_events,
        "status_counts": dict(status_counts),
        "pass_rate": None if pass_rate is None else round(pass_rate, 4),
        "target_pass_rate": 0.90,
        "target_met": bool(pass_rate is not None and pass_rate >= 0.90),
        "unsupported_confident_events": unsupported_confident,
        "correction_counts": correction_counts,
        "kind_status_counts": {kind: dict(counts) for kind, counts in sorted(kind_status.items())},
        "video_status_counts": {video: dict(counts) for video, counts in sorted(video_status.items())},
    }


def write_report(path: Path, metrics: dict[str, Any], out_dir: Path) -> None:
    lines = [
        "# Ball Tracking Audit",
        "",
        f"- Sampled events: {metrics['sampled_events']} of {metrics['total_high_confidence_events']} high-confidence contact/drop/stall events",
        f"- Pass rate: {metrics['pass_rate'] if metrics['pass_rate'] is not None else 'n/a'}",
        f"- Target: {metrics['target_pass_rate']:.2f}",
        f"- Target met: {metrics['target_met']}",
        f"- Unsupported confident events: {metrics['unsupported_confident_events']}",
        f"- Failure/uncertain sheet: `{rel(out_dir / 'audit_failures_uncertain_sheet.jpg')}`",
        f"- All audited sheet: `{rel(out_dir / 'audit_sample_sheet.jpg')}`",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in sorted(metrics["status_counts"].items()):
        lines.append(f"- {status}: {count}")
    lines.extend(["", "## Correction Counts", ""])
    for key, value in metrics["correction_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Kind Status", "", "| Kind | Pass | Uncertain | Fail |", "| --- | ---: | ---: | ---: |"])
    for kind, counts in metrics["kind_status_counts"].items():
        lines.append(f"| `{kind}` | {counts.get('pass', 0)} | {counts.get('uncertain', 0)} | {counts.get('fail', 0)} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Yellow circles are the QA event center; blue circles are the independent local color/blob center.",
            "- `fail` means a confident QA center lacks local visual support or is out of frame.",
            "- `uncertain` means the audit could not prove the center visually, so the event needs review instead of being treated as verified.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit QA ball-center accuracy")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--max-events", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_records = collect_records(args.manifest, args.min_confidence)
    records = sample_records(all_records, args.max_events)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results, frames = audit_records(records)
    metrics = summarize(results, len(all_records))
    metrics.update(
        {
            "manifest_path": rel(args.manifest),
            "min_confidence": args.min_confidence,
            "max_events": args.max_events,
            "results_path": rel(args.out_dir / "audit_results.jsonl"),
            "report_path": rel(args.out_dir / "audit_report.md"),
        }
    )
    write_json(args.out_dir / "audit_metrics.json", metrics)
    write_jsonl(args.out_dir / "audit_results.jsonl", results)
    failures = [item for item in results if item["audit_status"] in {"fail", "uncertain"}]
    failures.sort(key=lambda item: (0 if item["audit_status"] == "fail" else 1, -float(item.get("confidence") or 0.0), item["source_video"], item["time_sec"]))
    write_sheet(failures, frames, args.out_dir / "audit_failures_uncertain_sheet.jpg", max_items=80)
    sample_sheet_items = sorted(results, key=lambda item: (item["audit_status"] != "fail", item["audit_status"] != "uncertain", item["source_video"], item["time_sec"]))
    write_sheet(sample_sheet_items, frames, args.out_dir / "audit_sample_sheet.jpg", max_items=96)
    write_report(args.out_dir / "audit_report.md", metrics, args.out_dir)
    print(f"metrics: {args.out_dir / 'audit_metrics.json'}")
    print(f"report: {args.out_dir / 'audit_report.md'}")
    print(f"pass_rate: {metrics['pass_rate']} target_met={metrics['target_met']}")


if __name__ == "__main__":
    main()
