#!/usr/bin/env python3
"""Export reviewed dense trajectory labels as a center/heatmap dataset.

The Round 2 dense labels are center labels. The `radius` field is the review
template default and must not be interpreted as object size or exported as a
bounding box. This exporter writes center manifests and optional Gaussian
heatmaps for a WASB-lineage heatmap detector.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_BATCH_DIR = ROOT / "runs" / "release-27-public" / "dense_trajectory_review_v2_with_sources"
DEFAULT_LABELS = DEFAULT_BATCH_DIR / "dense_trajectory_labels.reviewed.jsonl"
DEFAULT_OUT_DIR = ROOT / "runs" / "release-27-public" / "dense_trajectory_dataset_v1"
POSITIVE_VISIBILITY = {"visible", "partially_occluded"}
NO_TARGET_VISIBILITY = {"fully_occluded", "out_of_frame"}
IGNORED_VISIBILITY = {"unlabeled", "uncertain"}
KNOWN_VISIBILITY = POSITIVE_VISIBILITY | NO_TARGET_VISIBILITY | IGNORED_VISIBILITY
MANIFEST_NAMES = {
    "train": "train.jsonl",
    "validation": "val.jsonl",
    "test_audit": "test_audit.jsonl",
    "negative_holdout": "negative_holdout.jsonl",
}


@dataclass(frozen=True)
class ExportRecord:
    image_path: str
    center_x: float | None
    center_y: float | None
    visibility: str
    split: str
    is_target: bool
    clip_id: str
    source_video: str
    frame_index: int
    time_sec: float | None
    training_use: str
    sigma_px: float
    heatmap_path: str | None = None
    diagnostic_role: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "image_path": self.image_path,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "visibility": self.visibility,
            "split": self.split,
            "is_target": self.is_target,
            "clip_id": self.clip_id,
            "source_video": self.source_video,
            "frame_index": self.frame_index,
            "time_sec": self.time_sec,
            "training_use": self.training_use,
            "sigma_px": self.sigma_px,
            "heatmap_path": self.heatmap_path,
            "diagnostic_role": self.diagnostic_role,
        }


def numeric(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def portable(path: Path, base: Path = ROOT) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except (OSError, ValueError):
        return str(path)


def frame_path_for(row: dict[str, Any], batch_dir: Path) -> Path:
    raw = row.get("frame_image")
    if not raw:
        raise ValueError(f"missing frame_image for {row_identity(row)}")
    path = Path(str(raw))
    if not path.is_absolute():
        path = batch_dir / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"missing frame image for {row_identity(row)}: {path}")
    return path


def row_identity(row: dict[str, Any]) -> str:
    return f"{row.get('clip_id')} frame={row.get('frame_index')} visibility={row.get('visibility')}"


def gaussian_heatmap(width: int, height: int, center_x: float | None, center_y: float | None, sigma_px: float) -> np.ndarray:
    if center_x is None or center_y is None:
        return np.zeros((height, width), dtype=np.float32)
    sigma = max(1e-3, float(sigma_px))
    yy, xx = np.mgrid[0:height, 0:width]
    heatmap = np.exp(-(((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma * sigma)))
    return heatmap.astype(np.float32)


def write_heatmap(path: Path, width: int, height: int, center_x: float | None, center_y: float | None, sigma_px: float) -> None:
    heatmap = gaussian_heatmap(width, height, center_x, center_y, sigma_px)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.clip(heatmap * 255.0, 0, 255).astype(np.uint8))


def assert_heatmap_peak(center_x: float, center_y: float, heatmap: np.ndarray, tolerance_px: float = 1.5) -> None:
    peak_y, peak_x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    if math.hypot(float(peak_x) - center_x, float(peak_y) - center_y) > tolerance_px:
        raise AssertionError(
            f"heatmap peak ({peak_x}, {peak_y}) is not on labeled center ({center_x:.3f}, {center_y:.3f})"
        )


def reviewed_target_kind(row: dict[str, Any]) -> str | None:
    visibility = str(row.get("visibility") or "")
    if visibility not in KNOWN_VISIBILITY:
        raise ValueError(f"unknown visibility value for {row_identity(row)}")
    if row.get("quality") != "reviewed":
        return None
    if visibility in POSITIVE_VISIBILITY:
        return "positive"
    if visibility in NO_TARGET_VISIBILITY:
        return "no_target"
    raise ValueError(f"reviewed row is not exportable because visibility is {visibility!r}: {row_identity(row)}")


def validate_clip_disjoint(rows: list[dict[str, Any]]) -> None:
    split_by_clip: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("quality") != "reviewed":
            continue
        clip_id = str(row.get("clip_id") or "")
        split = str(row.get("split") or "")
        split_by_clip[clip_id].add(split)
    offenders = {clip: sorted(splits) for clip, splits in split_by_clip.items() if len(splits) > 1}
    if offenders:
        raise AssertionError(f"clips appear in multiple splits: {offenders}")


def split_key_for(row: dict[str, Any]) -> str:
    training_use = str(row.get("training_use") or "train_or_calibration")
    split = str(row.get("split") or "")
    if training_use == "audit_only":
        return "test_audit"
    if split == "train":
        return "train"
    if split == "validation":
        return "validation"
    if split == "test":
        raise AssertionError(f"test row is not audit_only and would leak: {row_identity(row)}")
    raise ValueError(f"unsupported split for {row_identity(row)}: {split!r}")


def heatmap_output_path(out_dir: Path, manifest_key: str, row: dict[str, Any]) -> Path:
    return out_dir / "heatmaps" / manifest_key / str(row["clip_id"]) / f"frame_{int(row['frame_index']):06d}.png"


def record_from_row(
    row: dict[str, Any],
    *,
    batch_dir: Path,
    out_dir: Path,
    sigma_scale: float,
    render_heatmaps: bool,
    diagnostic_role: str | None = None,
) -> ExportRecord:
    kind = reviewed_target_kind(row)
    if kind is None:
        raise ValueError(f"cannot export non-reviewed row: {row_identity(row)}")

    frame_path = frame_path_for(row, batch_dir)
    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read frame image for {row_identity(row)}: {frame_path}")
    height, width = image.shape[:2]

    radius = numeric(row.get("radius"))
    if radius is None:
        raise ValueError(f"missing radius for sigma derivation in {row_identity(row)}")
    sigma_px = float(radius) * sigma_scale
    if sigma_px <= 0:
        raise ValueError(f"sigma must be positive for {row_identity(row)}")

    center_x: float | None = None
    center_y: float | None = None
    is_target = kind == "positive"
    if is_target:
        center_x = numeric(row.get("x"))
        center_y = numeric(row.get("y"))
        if center_x is None or center_y is None:
            raise ValueError(f"reviewed ball-present row has null center: {row_identity(row)}")
        if not (0 <= center_x < width and 0 <= center_y < height):
            raise ValueError(f"center outside frame bounds for {row_identity(row)}: ({center_x}, {center_y}) vs {width}x{height}")

    manifest_key = split_key_for(row)
    heatmap_path: Path | None = None
    if render_heatmaps:
        heatmap_path = heatmap_output_path(out_dir, manifest_key, row)
        write_heatmap(heatmap_path, width, height, center_x, center_y, sigma_px)

    return ExportRecord(
        image_path=portable(frame_path),
        center_x=None if center_x is None else round(center_x, 3),
        center_y=None if center_y is None else round(center_y, 3),
        visibility=str(row["visibility"]),
        split=str(row.get("split") or ""),
        is_target=is_target,
        clip_id=str(row["clip_id"]),
        source_video=str(row.get("source_video") or ""),
        frame_index=int(row["frame_index"]),
        time_sec=numeric(row.get("time_sec")),
        training_use=str(row.get("training_use") or "train_or_calibration"),
        sigma_px=round(float(sigma_px), 3),
        heatmap_path=None if heatmap_path is None else portable(heatmap_path),
        diagnostic_role=diagnostic_role,
    )


def choose_negative_holdout(rows: list[dict[str, Any]]) -> str | None:
    counts: Counter[str] = Counter()
    for row in rows:
        if row.get("quality") != "reviewed":
            continue
        if str(row.get("training_use") or "") == "audit_only":
            continue
        if str(row.get("visibility") or "") in NO_TARGET_VISIBILITY:
            counts[str(row["clip_id"])] += 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def summarize_records(records_by_manifest: dict[str, list[ExportRecord]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, records in records_by_manifest.items():
        clips = sorted({record.clip_id for record in records})
        positives = sum(1 for record in records if record.is_target)
        negatives = sum(1 for record in records if not record.is_target)
        summary[key] = {
            "records": len(records),
            "positive": positives,
            "no_target": negatives,
            "clips": clips,
            "clip_count": len(clips),
        }
    return summary


def assert_expected_counts(summary: dict[str, Any], expected: dict[str, int]) -> None:
    actual = {
        "train_positive": summary["train"]["positive"],
        "train_no_target": summary["train"]["no_target"],
        "train_clips": summary["train"]["clip_count"],
        "validation_positive": summary["validation"]["positive"],
        "validation_no_target": summary["validation"]["no_target"],
        "validation_clips": summary["validation"]["clip_count"],
        "test_audit_positive": summary["test_audit"]["positive"],
        "test_audit_no_target": summary["test_audit"]["no_target"],
        "test_audit_clips": summary["test_audit"]["clip_count"],
    }
    actual["total_positive"] = actual["train_positive"] + actual["validation_positive"] + actual["test_audit_positive"]
    actual["total_no_target"] = actual["train_no_target"] + actual["validation_no_target"] + actual["test_audit_no_target"]
    actual["total_reviewed"] = actual["total_positive"] + actual["total_no_target"]
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise AssertionError(f"dense heatmap export counts did not match expected values: {mismatches}")


def validate_no_audit_leak(records_by_manifest: dict[str, list[ExportRecord]]) -> None:
    leaks: list[dict[str, Any]] = []
    for key in ("train", "validation"):
        for record in records_by_manifest[key]:
            if record.training_use == "audit_only":
                leaks.append(record.to_json())
    if leaks:
        raise AssertionError(f"audit_only rows leaked into train/validation: {leaks[:5]}")


def render_positive_contact_sheet(
    *,
    records: list[ExportRecord],
    out_dir: Path,
    seed: int,
    sample_count: int,
) -> Path:
    positives = [record for record in records if record.is_target]
    if not positives:
        raise AssertionError("cannot render contact sheet: no positive records")
    rng = random.Random(seed)
    samples = rng.sample(positives, k=min(sample_count, len(positives)))
    tiles: list[np.ndarray] = []
    for record in samples:
        frame_path = ROOT / record.image_path if not Path(record.image_path).is_absolute() else Path(record.image_path)
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not read sampled frame: {frame_path}")
        if record.center_x is None or record.center_y is None:
            raise AssertionError(f"sampled positive without center: {record}")
        heatmap = gaussian_heatmap(image.shape[1], image.shape[0], record.center_x, record.center_y, record.sigma_px)
        assert_heatmap_peak(record.center_x, record.center_y, heatmap)
        heat_u8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
        color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
        overlay = cv2.addWeighted(image, 0.7, color, 0.3, 0.0)
        center = (int(round(record.center_x)), int(round(record.center_y)))
        cv2.circle(overlay, center, 12, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(overlay, center, 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 52), (8, 12, 18), -1)
        cv2.putText(overlay, f"{record.source_video[:34]}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 248, 252), 1, cv2.LINE_AA)
        cv2.putText(overlay, f"f{record.frame_index} sigma={record.sigma_px:g}", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (215, 225, 238), 1, cv2.LINE_AA)
        tiles.append(cv2.resize(overlay, (240, 318), interpolation=cv2.INTER_AREA))

    cols = min(5, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.full((rows * 318 + 48, cols * 240, 3), (15, 20, 27), dtype=np.uint8)
    cv2.putText(canvas, "Dense heatmap export QA: sampled positives", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 248, 252), 2, cv2.LINE_AA)
    for index, tile in enumerate(tiles):
        x = (index % cols) * 240
        y = 48 + (index // cols) * 318
        canvas[y : y + 318, x : x + 240] = tile
    out_path = out_dir / "qa" / "positive_heatmap_contact_sheet.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return out_path


def export_dense_heatmap_dataset(
    *,
    labels_jsonl: Path = DEFAULT_LABELS,
    batch_dir: Path = DEFAULT_BATCH_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    sigma_scale: float = 1.0,
    render_heatmaps: bool = False,
    seed: int = 7,
    contact_sheet_samples: int = 5,
    expected_counts: dict[str, int] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    labels_jsonl = labels_jsonl.resolve()
    batch_dir = batch_dir.resolve()
    out_dir = out_dir.resolve()
    rows = load_jsonl(labels_jsonl)

    for row in rows:
        visibility = str(row.get("visibility") or "")
        if visibility not in KNOWN_VISIBILITY:
            raise ValueError(f"unknown visibility value for {row_identity(row)}")
    validate_clip_disjoint(rows)

    negative_holdout_clip = choose_negative_holdout(rows)
    records_by_manifest: dict[str, list[ExportRecord]] = {
        "train": [],
        "validation": [],
        "test_audit": [],
        "negative_holdout": [],
    }

    reviewed_count = 0
    for row in rows:
        kind = reviewed_target_kind(row)
        if kind is None:
            continue
        reviewed_count += 1
        key = split_key_for(row)
        record = record_from_row(
            row,
            batch_dir=batch_dir,
            out_dir=out_dir,
            sigma_scale=sigma_scale,
            render_heatmaps=render_heatmaps and not dry_run,
        )
        records_by_manifest[key].append(record)
        if row.get("clip_id") == negative_holdout_clip and kind == "no_target":
            records_by_manifest["negative_holdout"].append(
                record_from_row(
                    row,
                    batch_dir=batch_dir,
                    out_dir=out_dir,
                    sigma_scale=sigma_scale,
                    render_heatmaps=False,
                    diagnostic_role="DIAGNOSTIC_NON_PRISTINE_NEGATIVE_HOLDOUT",
                )
            )

    validate_no_audit_leak(records_by_manifest)
    split_summary = summarize_records(records_by_manifest)
    if expected_counts is not None:
        assert_expected_counts(split_summary, expected_counts)

    all_primary_records = records_by_manifest["train"] + records_by_manifest["validation"] + records_by_manifest["test_audit"]
    positive_sheet = None if dry_run else render_positive_contact_sheet(records=all_primary_records, out_dir=out_dir, seed=seed, sample_count=contact_sheet_samples)

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_reviewed_jsonl": portable(labels_jsonl),
        "source_batch_dir": portable(batch_dir),
        "out_dir": portable(out_dir),
        "row_selection": {
            "quality": "reviewed",
            "positive_visibility": sorted(POSITIVE_VISIBILITY),
            "no_target_visibility": sorted(NO_TARGET_VISIBILITY),
            "ignored_visibility": sorted(IGNORED_VISIBILITY),
        },
        "radius_semantics": "radius is a review-template default used only to derive Gaussian sigma / eval tolerance; it is not object size and no boxes are exported.",
        "sigma": {
            "source": "row.radius * sigma_scale",
            "sigma_scale": sigma_scale,
            "observed_positive_radius_values": sorted({numeric(row.get("radius")) for row in rows if row.get("quality") == "reviewed" and str(row.get("visibility")) in POSITIVE_VISIBILITY}),
        },
        "heatmaps": {
            "per_frame_heatmaps_rendered": render_heatmaps and not dry_run,
            "contact_sheet": None if positive_sheet is None else portable(positive_sheet),
        },
        "reviewed_rows_exported": reviewed_count,
        "splits": split_summary,
        "outputs": {
            key: portable(out_dir / filename)
            for key, filename in MANIFEST_NAMES.items()
        }
        | {"dataset_manifest": portable(out_dir / "dataset_manifest.json")},
        "negative_holdout": {
            "clip_id": negative_holdout_clip,
            "role": "DIAGNOSTIC / NON-pristine same-batch holdout for measuring false positives until a real sealed no-target test clip is collected.",
        },
    }

    if dry_run:
        return manifest

    out_dir.mkdir(parents=True, exist_ok=True)
    for key, filename in MANIFEST_NAMES.items():
        write_jsonl(out_dir / filename, [record.to_json() for record in records_by_manifest[key]])
    write_json(out_dir / "dataset_manifest.json", manifest)
    return manifest


def default_expected_counts() -> dict[str, int]:
    return {
        "train_positive": 451,
        "train_no_target": 44,
        "train_clips": 8,
        "validation_positive": 116,
        "validation_no_target": 71,
        "validation_clips": 3,
        "test_audit_positive": 62,
        "test_audit_no_target": 0,
        "test_audit_clips": 1,
        "total_positive": 629,
        "total_no_target": 115,
        "total_reviewed": 744,
    }


def print_split_table(manifest: dict[str, Any]) -> None:
    print("| split | positive | no_target | clips | records |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for key in ("train", "validation", "test_audit", "negative_holdout"):
        split = manifest["splits"][key]
        print(f"| {key} | {split['positive']} | {split['no_target']} | {split['clip_count']} | {split['records']} |")
    print(f"manifest: {manifest['outputs']['dataset_manifest']}")
    print(f"negative_holdout: {manifest['negative_holdout']['clip_id']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export reviewed dense center labels for heatmap-detector training")
    parser.add_argument("--labels-jsonl", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sigma-scale", type=float, default=1.0)
    parser.add_argument("--render-heatmaps", action="store_true", help="Render one grayscale Gaussian heatmap per exported frame")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--contact-sheet-samples", type=int, default=5)
    parser.add_argument("--no-count-assertions", action="store_true", help="Disable the pinned current-dataset acceptance counts")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = export_dense_heatmap_dataset(
        labels_jsonl=args.labels_jsonl,
        batch_dir=args.batch_dir,
        out_dir=args.out_dir,
        sigma_scale=args.sigma_scale,
        render_heatmaps=args.render_heatmaps,
        seed=args.seed,
        contact_sheet_samples=args.contact_sheet_samples,
        expected_counts=None if args.no_count_assertions else default_expected_counts(),
        dry_run=args.dry_run,
    )
    print_split_table(manifest)


if __name__ == "__main__":
    main()
