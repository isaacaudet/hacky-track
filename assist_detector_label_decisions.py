#!/usr/bin/env python3
"""Create conservative CV-assisted detector-label decisions.

This is not a replacement for review. It only writes completed decisions for
obvious centered footbags and obvious non-footbag centers, leaving ambiguous
items pending so they cannot contaminate the training set.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def portable(path: Path, base: Path = ROOT) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except (OSError, ValueError):
        return path.name if path.is_absolute() else str(path)


def resolve_path(raw: Any, *, base: Path) -> Path:
    path = Path(str(raw or "")).expanduser()
    if path.exists() or path.is_absolute():
        return path
    base_candidate = base / path
    if base_candidate.exists():
        return base_candidate
    return ROOT / path


def numeric(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if math.isfinite(value_f) else None


def disk_mask(shape: tuple[int, int], center: tuple[float, float], radius: float) -> np.ndarray:
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    return ((xx - center[0]) ** 2 + (yy - center[1]) ** 2) <= radius * radius


def color_masks(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    red = (((h <= 12) | (h >= 165)) & (s >= 65) & (v >= 45)).astype(np.uint8)
    orange = (((h >= 5) & (h <= 28)) & (s >= 80) & (v >= 55)).astype(np.uint8)
    yellow = (((h >= 18) & (h <= 44)) & (s >= 45) & (v >= 60)).astype(np.uint8)
    blue = (((h >= 82) & (h <= 132)) & (s >= 35) & (v >= 45)).astype(np.uint8)
    dark = ((v <= 70) & (s >= 20)).astype(np.uint8)
    # Green is deliberately excluded from the primary connected-component mask:
    # outdoor grass otherwise becomes one giant "ball-color" component. Blue,
    # yellow, and red/orange carry stronger footbag-panel evidence.
    color = cv2.morphologyEx(((red | orange | yellow | blue) * 255).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return color, dark, hsv


def auxiliary_color_masks(hsv: np.ndarray) -> dict[str, np.ndarray]:
    h, s, v = cv2.split(hsv)
    blue = (((h >= 82) & (h <= 132)) & (s >= 35) & (v >= 45))
    yellow = (((h >= 18) & (h <= 44)) & (s >= 45) & (v >= 60))
    green = (((h >= 44) & (h <= 88)) & (s >= 55) & (v >= 45))
    red_or_orange = ((((h <= 12) | (h >= 165)) & (s >= 65) & (v >= 45)) | (((h >= 5) & (h <= 28)) & (s >= 80) & (v >= 55)))
    skin = ((h <= 25) & (s >= 20) & (s <= 180) & (v >= 70))
    return {
        "blue_yellow": ((blue | yellow) * 255).astype(np.uint8),
        "green": (green * 255).astype(np.uint8),
        "red_or_orange": (red_or_orange * 255).astype(np.uint8),
        "skin": (skin * 255).astype(np.uint8),
    }


def nearest_color_component(color_mask: np.ndarray, center: tuple[float, float]) -> dict[str, float] | None:
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(color_mask, 8)
    best: dict[str, float] | None = None
    for idx in range(1, num):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area < 8:
            continue
        cx, cy = float(centroids[idx][0]), float(centroids[idx][1])
        dist = math.hypot(cx - center[0], cy - center[1])
        score = area / (1.0 + dist)
        if best is None or score > best["score"]:
            best = {
                "area": area,
                "cx": cx,
                "cy": cy,
                "dist": dist,
                "score": score,
                "width": float(stats[idx, cv2.CC_STAT_WIDTH]),
                "height": float(stats[idx, cv2.CC_STAT_HEIGHT]),
            }
    return best


def analyze_crop(crop_path: Path, record: dict[str, Any]) -> dict[str, Any]:
    image = cv2.imread(str(crop_path))
    if image is None:
        return {"status": "pending", "confidence": 0.0, "reasons": ["missing_crop"]}
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    radius = numeric(record.get("radius")) or 18.0
    search_radius = max(22.0, min(58.0, radius * 2.2))
    center_disk = disk_mask((h, w), center, search_radius)
    color_mask, dark_mask, _hsv = color_masks(image)
    aux_masks = auxiliary_color_masks(_hsv)
    color_pixels = color_mask > 0
    dark_pixels = dark_mask > 0
    blue_yellow_pixels = aux_masks["blue_yellow"] > 0
    skin_pixels = aux_masks["skin"] > 0
    component = nearest_color_component(color_mask, center)

    center_area = max(1, int(center_disk.sum()))
    center_color_ratio = float((color_pixels & center_disk).sum()) / center_area
    center_blue_yellow_ratio = float((blue_yellow_pixels & center_disk).sum()) / center_area
    center_dark_ratio = float((dark_pixels & center_disk).sum()) / center_area
    center_skin_ratio = float((skin_pixels & center_disk).sum()) / center_area
    total_color_ratio = float(color_pixels.sum()) / max(1, h * w)
    correction = numeric(record.get("qa_ball_correction_px")) or 0.0
    qa_confidence = numeric(record.get("qa_ball_confidence")) or 0.0
    suggested = str(record.get("suggested_detector_label") or "")

    reasons: list[str] = []
    status = "pending"
    confidence = 0.0
    if component is not None:
        reasons.append(f"nearest_color_component_area={component['area']:.0f}")
        reasons.append(f"nearest_color_component_dist={component['dist']:.1f}")
    reasons.append(f"center_color_ratio={center_color_ratio:.4f}")
    reasons.append(f"center_blue_yellow_ratio={center_blue_yellow_ratio:.4f}")
    reasons.append(f"center_dark_ratio={center_dark_ratio:.4f}")
    reasons.append(f"center_skin_ratio={center_skin_ratio:.4f}")

    footbag_near_limit = max(12.0, radius * 0.65) if suggested == "not_footbag" else max(24.0, radius * 1.7)
    near_component = component is not None and component["dist"] <= footbag_near_limit
    compact_component = (
        component is not None
        and 6.0 <= component["width"] <= search_radius * 2.4
        and 6.0 <= component["height"] <= search_radius * 2.4
        and component["area"] <= math.pi * (search_radius * 1.45) ** 2
    )
    strong_center_color = center_color_ratio >= 0.018
    distinct_panel_color = center_blue_yellow_ratio >= 0.007
    patterned_center = center_color_ratio >= 0.008 and center_dark_ratio >= 0.04
    skin_dominates = (
        center_skin_ratio >= 0.18
        and center_skin_ratio >= max(0.03, center_blue_yellow_ratio * 3.0)
        and center_dark_ratio < 0.04
    )
    if near_component and compact_component and (distinct_panel_color or strong_center_color or patterned_center) and not skin_dominates:
        status = "footbag"
        confidence = min(
            0.98,
            0.58
            + center_color_ratio * 8.0
            + center_blue_yellow_ratio * 8.0
            + min(0.20, component["area"] / 360.0),
        )
        reasons.append("centered_compact_footbag_color_pattern")
    elif suggested == "not_footbag" and (
        skin_dominates
        or (not compact_component and center_blue_yellow_ratio < 0.010)
        or (total_color_ratio < 0.003 and center_color_ratio < 0.001)
    ):
        status = "not_footbag"
        confidence = 0.80 if skin_dominates else 0.74
        reasons.append("suggested_hard_negative_without_centered_compact_footbag")
    elif (
        suggested == "verify_or_correct"
        and correction >= 110.0
        and qa_confidence < 0.78
        and total_color_ratio < 0.002
        and center_color_ratio < 0.001
    ):
        status = "not_footbag"
        confidence = 0.72
        reasons.append("large_correction_no_red_or_orange_candidate")
    else:
        reasons.append("ambiguous_left_pending")

    return {
        "status": status,
        "confidence": round(float(confidence), 4),
        "reasons": reasons,
        "metrics": {
            "center_color_ratio": round(center_color_ratio, 6),
            "center_blue_yellow_ratio": round(center_blue_yellow_ratio, 6),
            "center_dark_ratio": round(center_dark_ratio, 6),
            "center_skin_ratio": round(center_skin_ratio, 6),
            "total_color_ratio": round(total_color_ratio, 6),
            "nearest_component": component,
        },
    }


def build_assisted_decisions(
    review_manifest: Path,
    out_path: Path,
    *,
    min_confidence: float = 0.70,
    dry_run: bool = False,
) -> dict[str, Any]:
    manifest = read_json(review_manifest)
    base = review_manifest.parent
    decisions: list[dict[str, Any]] = []
    counts = {"footbag": 0, "not_footbag": 0, "pending": 0, "skip": 0}
    for item in manifest.get("items", []):
        crop_path = resolve_path(item.get("crop_path"), base=base)
        analysis = analyze_crop(crop_path, item)
        status = analysis["status"] if float(analysis["confidence"]) >= min_confidence else "pending"
        counts[status] = counts.get(status, 0) + 1
        decisions.append(
            {
                "detector_label_id": item.get("detector_label_id"),
                "source_video": item.get("source_video"),
                "detector_status": status,
                "corrected_x": None,
                "corrected_y": None,
                "radius": item.get("radius"),
                "confidence": analysis["confidence"],
                "evidence": "; ".join(analysis["reasons"]),
                "reviewer": "cv_assisted_conservative",
                "review_mode": "assisted_not_human_final",
                "metrics": analysis["metrics"],
            }
        )
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "review_manifest": portable(review_manifest),
        "out_path": portable(out_path),
        "min_confidence": min_confidence,
        "counts": counts,
        "instructions": "Conservative CV-assisted decisions. Pending rows must be reviewed manually. Treat non-pending rows as assisted labels unless a human reviewer validates them.",
        "decisions": decisions,
    }
    if not dry_run:
        write_json(out_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create conservative CV-assisted detector-label decisions")
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_assisted_decisions(
        args.review_manifest,
        args.out,
        min_confidence=args.min_confidence,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(f"decisions: {args.out}")
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
