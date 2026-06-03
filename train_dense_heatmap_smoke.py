#!/usr/bin/env python3
"""Validation-only smoke fine-tune for dense center heatmap labels.

This is deliberately not YOLO training. It uses the center-only manifests from
`export_dense_heatmap_dataset.py`, generates Gaussian heatmap targets on the fly,
trains a tiny fully-convolutional heatmap model on train.jsonl, and evaluates only
on val.jsonl. The sealed audit/test manifest is loaded only to verify it stays
sealed and is never used for training or validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "runs" / "release-27-public" / "dense_trajectory_dataset_v1"
DEFAULT_OUT = ROOT / "runs" / "release-27-public" / "dense_heatmap_smoke_v1"
VISIBLE_TOLERANCE_PX = 15.0
DEFAULT_THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def portable(path: Path | None, base: Path = ROOT) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except (OSError, ValueError):
        return str(path)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def gaussian_target(width: int, height: int, center_x: float | None, center_y: float | None, sigma_px: float) -> np.ndarray:
    if center_x is None or center_y is None:
        return np.zeros((height, width), dtype=np.float32)
    sigma = max(1e-3, sigma_px)
    yy, xx = np.mgrid[0:height, 0:width]
    heatmap = np.exp(-(((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma * sigma)))
    return heatmap.astype(np.float32)


@dataclass(frozen=True)
class SampleMeta:
    row_index: int
    image_path: str
    clip_id: str
    source_video: str
    frame_index: int
    visibility: str
    split: str
    is_target: bool
    center_x: float | None
    center_y: float | None
    sigma_px: float
    orig_width: int
    orig_height: int


class DenseHeatmapDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], *, width: int, height: int) -> None:
        self.records = records
        self.width = width
        self.height = height

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        record = self.records[index]
        image_path = resolve_path(record["image_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not read image: {image_path}")
        orig_height, orig_width = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        image_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

        is_target = bool(record["is_target"])
        center_x = None if record.get("center_x") is None else float(record["center_x"])
        center_y = None if record.get("center_y") is None else float(record["center_y"])
        sigma_px = float(record["sigma_px"])
        if is_target:
            if center_x is None or center_y is None:
                raise ValueError(f"target record is missing center: {record}")
            scaled_x = center_x * (self.width / orig_width)
            scaled_y = center_y * (self.height / orig_height)
            scaled_sigma = max(1.0, sigma_px * ((self.width / orig_width + self.height / orig_height) / 2.0))
            target = gaussian_target(self.width, self.height, scaled_x, scaled_y, scaled_sigma)
        else:
            target = np.zeros((self.height, self.width), dtype=np.float32)
        target_tensor = torch.from_numpy(target)[None, :, :]
        meta = {
            "row_index": index,
            "image_path": record["image_path"],
            "clip_id": record["clip_id"],
            "source_video": record["source_video"],
            "frame_index": int(record["frame_index"]),
            "visibility": record["visibility"],
            "split": record["split"],
            "is_target": is_target,
            "center_x": center_x,
            "center_y": center_y,
            "sigma_px": sigma_px,
            "orig_width": orig_width,
            "orig_height": orig_height,
        }
        return image_tensor, target_tensor, meta


def collate_heatmap_batch(batch: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[Any]]]:
    images, targets, metas = zip(*batch)
    keys = metas[0].keys()
    return (
        torch.stack(list(images), dim=0),
        torch.stack(list(targets), dim=0),
        {key: [meta[key] for meta in metas] for key in keys},
    )


class TinyHeatmapNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(96 + 48, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(48 + 24, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(24, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = torch.nn.functional.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = torch.nn.functional.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def weighted_heatmap_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Sparse heatmaps otherwise encourage the all-background solution. The peak
    # weighting only affects heatmap confidence training; it does not imply size.
    weights = 1.0 + target * 80.0
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (bce * weights).mean()


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    losses: list[float] = []
    for images, targets, _meta in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = weighted_heatmap_loss(logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(mean(losses)) if losses else 0.0


def predict_records(model: nn.Module, loader: DataLoader, device: torch.device) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for images, _targets, meta in loader:
            logits = model(images.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]
            batch_size = probs.shape[0]
            for item in range(batch_size):
                heatmap = probs[item]
                peak_flat = int(np.argmax(heatmap))
                peak_y, peak_x = np.unravel_index(peak_flat, heatmap.shape)
                confidence = float(heatmap[peak_y, peak_x])
                orig_width = int(meta["orig_width"][item])
                orig_height = int(meta["orig_height"][item])
                pred_x = (float(peak_x) + 0.5) * orig_width / heatmap.shape[1]
                pred_y = (float(peak_y) + 0.5) * orig_height / heatmap.shape[0]
                center_x = meta["center_x"][item]
                center_y = meta["center_y"][item]
                is_target = bool(meta["is_target"][item])
                error = None
                if is_target:
                    error = math.hypot(float(pred_x) - float(center_x), float(pred_y) - float(center_y))
                rows.append(
                    {
                        "row_index": int(meta["row_index"][item]),
                        "image_path": meta["image_path"][item],
                        "clip_id": meta["clip_id"][item],
                        "source_video": meta["source_video"][item],
                        "frame_index": int(meta["frame_index"][item]),
                        "visibility": meta["visibility"][item],
                        "split": meta["split"][item],
                        "is_target": is_target,
                        "expected_x": None if center_x is None else round(float(center_x), 3),
                        "expected_y": None if center_y is None else round(float(center_y), 3),
                        "prediction_x": round(pred_x, 3),
                        "prediction_y": round(pred_y, 3),
                        "confidence": round(confidence, 6),
                        "center_error_px": None if error is None else round(error, 3),
                    }
                )
    return rows


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_predictions(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    positives = [row for row in rows if row["is_target"]]
    negatives = [row for row in rows if not row["is_target"]]
    confident_positives = [row for row in positives if float(row["confidence"]) >= threshold]
    missing = len(positives) - len(confident_positives)
    passed = sum(1 for row in confident_positives if float(row["center_error_px"]) <= VISIBLE_TOLERANCE_PX)
    failed = len(confident_positives) - passed
    errors = [float(row["center_error_px"]) for row in confident_positives if row["center_error_px"] is not None]
    fp = sum(1 for row in negatives if float(row["confidence"]) >= threshold)
    return {
        "threshold": threshold,
        "visible_frames": len(positives),
        "passed_visible_frames": passed,
        "failed_visible_frames": failed,
        "missing_visible_frames": missing,
        "visible_pass_rate": None if not positives else round(passed / len(positives), 6),
        "visible_missing_rate": None if not positives else round(missing / len(positives), 6),
        "mean_center_error_px": None if not errors else round(float(mean(errors)), 3),
        "p95_center_error_px": None if not errors else round(float(percentile(errors, 0.95)), 3),
        "no_target_frames": len(negatives),
        "false_positive_no_target_frames": fp,
        "no_target_false_positive_rate": None if not negatives else round(fp / len(negatives), 6),
    }


def read_baselines(paths: list[Path]) -> dict[str, Any]:
    baselines: dict[str, Any] = {}
    for path in paths:
        if not path.exists():
            continue
        data = read_json(path)
        name = path.parent.name.replace("eval_reviewed_", "")
        baselines[name] = data.get("by_split", {}).get("validation", data.get("overall", {}))
    return baselines


def compare_with_baselines(threshold_sweep: list[dict[str, Any]], baselines: dict[str, Any]) -> dict[str, Any]:
    best_visible = max(threshold_sweep, key=lambda row: float(row["visible_pass_rate"] or 0.0))
    best_balanced = max(
        threshold_sweep,
        key=lambda row: float(row["visible_pass_rate"] or 0.0) - 0.25 * float(row["no_target_false_positive_rate"] or 0.0),
    )
    best_baseline_visible_name = None
    best_baseline_visible_rate = -1.0
    best_baseline_fp_name = None
    best_baseline_fp_rate = math.inf
    for name, row in baselines.items():
        visible_rate = row.get("visible_pass_rate")
        if visible_rate is not None and float(visible_rate) > best_baseline_visible_rate:
            best_baseline_visible_name = name
            best_baseline_visible_rate = float(visible_rate)
        fp_rate = row.get("no_target_false_positive_rate")
        if fp_rate is not None and float(fp_rate) < best_baseline_fp_rate:
            best_baseline_fp_name = name
            best_baseline_fp_rate = float(fp_rate)
    return {
        "best_smoke_visible": best_visible,
        "best_smoke_balanced": best_balanced,
        "best_baseline_visible": {
            "name": best_baseline_visible_name,
            "visible_pass_rate": None if best_baseline_visible_name is None else round(best_baseline_visible_rate, 6),
        },
        "best_baseline_no_target_fp": {
            "name": best_baseline_fp_name,
            "no_target_false_positive_rate": None if best_baseline_fp_name is None else round(best_baseline_fp_rate, 6),
        },
        "beats_best_baseline_visible": (
            False
            if best_baseline_visible_name is None
            else float(best_visible["visible_pass_rate"] or 0.0) > best_baseline_visible_rate
        ),
        "beats_best_baseline_no_target_fp": (
            False
            if best_baseline_fp_name is None
            else float(best_balanced["no_target_false_positive_rate"] or 1.0) < best_baseline_fp_rate
        ),
    }


def validate_dataset_discipline(dataset_dir: Path) -> dict[str, Any]:
    train = read_jsonl(dataset_dir / "train.jsonl")
    val = read_jsonl(dataset_dir / "val.jsonl")
    audit = read_jsonl(dataset_dir / "test_audit.jsonl")
    leaks = [row for row in train + val if row.get("training_use") == "audit_only"]
    if leaks:
        raise AssertionError(f"audit_only rows leaked into train/val: {leaks[:3]}")
    train_clips = {row["clip_id"] for row in train}
    val_clips = {row["clip_id"] for row in val}
    audit_clips = {row["clip_id"] for row in audit}
    overlaps = {
        "train_val": sorted(train_clips & val_clips),
        "train_audit": sorted(train_clips & audit_clips),
        "val_audit": sorted(val_clips & audit_clips),
    }
    if any(overlaps.values()):
        raise AssertionError(f"clip split leakage detected: {overlaps}")
    return {
        "train_records": len(train),
        "validation_records": len(val),
        "test_audit_records_sealed": len(audit),
        "train_clips": sorted(train_clips),
        "validation_clips": sorted(val_clips),
        "test_audit_clips": sorted(audit_clips),
    }


def render_prediction_contact_sheet(rows: list[dict[str, Any]], out_dir: Path, seed: int, max_items: int = 8) -> Path:
    positives = [row for row in rows if row["is_target"] and row.get("center_error_px") is not None]
    if not positives:
        raise AssertionError("no positive predictions to render")
    hardest = sorted(positives, key=lambda row: float(row["center_error_px"]), reverse=True)[: max_items // 3]
    best = sorted(positives, key=lambda row: float(row["center_error_px"]))[: max_items // 3]
    negatives = sorted(
        [row for row in rows if not row["is_target"]],
        key=lambda row: float(row["confidence"]),
        reverse=True,
    )[: max_items // 3]
    rng = random.Random(seed)
    sample_pool = positives[:]
    rng.shuffle(sample_pool)
    selected = hardest + best + negatives + sample_pool[: max(0, max_items - len(hardest) - len(best) - len(negatives))]
    selected = selected[:max_items]
    tiles: list[np.ndarray] = []
    for row in selected:
        image_path = row.get("image_path")
        if image_path is None:
            continue
        image = cv2.imread(str(resolve_path(str(image_path))), cv2.IMREAD_COLOR)
        if image is None:
            continue
        tile_width = 360
        scale = tile_width / image.shape[1]
        tile_height = max(1, int(round(image.shape[0] * scale)))
        tile = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        pred = (int(round(float(row["prediction_x"]) * scale)), int(round(float(row["prediction_y"]) * scale)))
        is_target = bool(row["is_target"])
        confidence = float(row["confidence"])
        if is_target:
            expected = (int(round(float(row["expected_x"]) * scale)), int(round(float(row["expected_y"]) * scale)))
            error = float(row["center_error_px"])
            color = (0, 220, 0) if error <= VISIBLE_TOLERANCE_PX else (0, 120, 255)
            cv2.circle(tile, expected, 9, (0, 255, 0), 2)
            cv2.drawMarker(tile, pred, color, markerType=cv2.MARKER_TILTED_CROSS, markerSize=22, thickness=2)
            label = f"{row['source_video']} f{row['frame_index']} err={error:.1f}px conf={confidence:.2f}"
        else:
            cv2.drawMarker(tile, pred, (255, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=22, thickness=2)
            label = f"{row['source_video']} f{row['frame_index']} no-target conf={confidence:.2f}"
        cv2.rectangle(tile, (0, 0), (tile_width, 34), (0, 0, 0), -1)
        cv2.putText(tile, label[:72], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        raise AssertionError("no prediction contact sheet tiles could be rendered")
    cols = 2
    rows_needed = math.ceil(len(tiles) / cols)
    max_h = max(tile.shape[0] for tile in tiles)
    sheet = np.full((rows_needed * max_h, cols * 360, 3), 24, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        y = (idx // cols) * max_h
        x = (idx % cols) * 360
        sheet[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    out_path = out_dir / "qa" / "validation_predictions_contact_sheet.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    return out_path


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Dense Heatmap Smoke Fine-tune",
        "",
        "Scope: train on `train.jsonl`, evaluate on `val.jsonl`, keep `test_audit.jsonl` sealed. Center heatmaps only; no boxes.",
        "",
        "## Dataset Discipline",
        "",
        f"- Train records: `{summary['dataset_discipline']['train_records']}`",
        f"- Validation records: `{summary['dataset_discipline']['validation_records']}`",
        f"- Sealed audit/test records: `{summary['dataset_discipline']['test_audit_records_sealed']}`",
        "",
        "## Heatmap Model",
        "",
        f"- Device: `{summary['parameters']['device']}`",
        f"- Input: `{summary['parameters']['image_width']}x{summary['parameters']['image_height']}`",
        f"- Epochs: `{summary['parameters']['epochs']}`",
        "",
        "## Validation Threshold Sweep",
        "",
        "| threshold | visible pass | missing | mean error | p95 error | no-target FP |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["validation_threshold_sweep"]:
        lines.append(
            f"| {row['threshold']} | {row['visible_pass_rate']} | {row['visible_missing_rate']} | "
            f"{row['mean_center_error_px']} | {row['p95_center_error_px']} | {row['no_target_false_positive_rate']} |"
        )
    lines.extend(["", "## Existing Detector Baselines On Validation", "", "| baseline | visible pass | missing | mean error | p95 error | no-target FP |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for name, row in sorted(summary["baselines"].items()):
        lines.append(
            f"| {name} | {row.get('visible_pass_rate')} | {row.get('visible_missing_rate')} | "
            f"{row.get('mean_center_error_px')} | {row.get('p95_center_error_px')} | {row.get('no_target_false_positive_rate')} |"
        )
    lines.extend(
        [
            "",
            "## Best Smoke Result",
            "",
            f"- Best threshold: `{summary['best_validation']['threshold']}`",
            f"- Visible pass rate: `{summary['best_validation']['visible_pass_rate']}`",
            f"- No-target false-positive rate: `{summary['best_validation']['no_target_false_positive_rate']}`",
            "",
            "## Baseline Comparison",
            "",
            f"- Best existing visible pass: `{summary['baseline_comparison']['best_baseline_visible']['name']}` "
            f"at `{summary['baseline_comparison']['best_baseline_visible']['visible_pass_rate']}`",
            f"- Best smoke visible pass: `{summary['baseline_comparison']['best_smoke_visible']['visible_pass_rate']}` "
            f"at threshold `{summary['baseline_comparison']['best_smoke_visible']['threshold']}`",
            f"- Beats visible baseline: `{summary['baseline_comparison']['beats_best_baseline_visible']}`",
            f"- Best existing no-target FP: `{summary['baseline_comparison']['best_baseline_no_target_fp']['name']}` "
            f"at `{summary['baseline_comparison']['best_baseline_no_target_fp']['no_target_false_positive_rate']}`",
            f"- Best smoke balanced no-target FP: `{summary['baseline_comparison']['best_smoke_balanced']['no_target_false_positive_rate']}` "
            f"at threshold `{summary['baseline_comparison']['best_smoke_balanced']['threshold']}`",
            f"- Beats no-target FP baseline: `{summary['baseline_comparison']['beats_best_baseline_no_target_fp']}`",
            "",
            "Interpretation: this is a smoke result on validation only. It does not use or claim the sealed audit/test rows.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_dense_heatmap_smoke(
    *,
    dataset_dir: Path,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    image_width: int,
    image_height: int,
    learning_rate: float,
    device_name: str,
    seed: int,
    num_workers: int,
    thresholds: tuple[float, ...],
    baseline_metrics: list[Path],
    dry_run: bool = False,
) -> dict[str, Any]:
    seed_everything(seed)
    dataset_dir = dataset_dir.resolve()
    out_dir = out_dir.resolve()
    discipline = validate_dataset_discipline(dataset_dir)
    train_records = read_jsonl(dataset_dir / "train.jsonl")
    val_records = read_jsonl(dataset_dir / "val.jsonl")
    if dry_run:
        return {"dataset_discipline": discipline, "train_records": len(train_records), "validation_records": len(val_records)}

    device = choose_device(device_name)
    train_dataset = DenseHeatmapDataset(train_records, width=image_width, height=image_height)
    val_dataset = DenseHeatmapDataset(val_records, width=image_width, height=image_height)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_heatmap_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_heatmap_batch)
    model = TinyHeatmapNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    epoch_rows: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_score = -1.0
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        predictions = predict_records(model, val_loader, device)
        metrics_at_default = summarize_predictions(predictions, threshold=0.2)
        epoch_row = {"epoch": epoch, "train_loss": round(loss, 6), **metrics_at_default}
        epoch_rows.append(epoch_row)
        score = float(metrics_at_default["visible_pass_rate"] or 0.0) - 0.25 * float(metrics_at_default["no_target_false_positive_rate"] or 0.0)
        if score > best_score:
            best_score = score
            best_state = {
                "epoch": epoch,
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "predictions": predictions,
            }
        print(
            f"epoch {epoch:02d} loss={loss:.4f} val_pass={metrics_at_default['visible_pass_rate']} "
            f"val_fp={metrics_at_default['no_target_false_positive_rate']}"
        )

    if best_state is None:
        raise AssertionError("training produced no best state")
    model.load_state_dict(best_state["model"])
    val_predictions = predict_records(model, val_loader, device)
    threshold_sweep = [summarize_predictions(val_predictions, threshold=threshold) for threshold in thresholds]
    best_validation = max(
        threshold_sweep,
        key=lambda row: float(row["visible_pass_rate"] or 0.0) - 0.25 * float(row["no_target_false_positive_rate"] or 0.0),
    )

    baselines = read_baselines(baseline_metrics)
    baseline_comparison = compare_with_baselines(threshold_sweep, baselines)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "tiny_heatmap_smoke.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model": "TinyHeatmapNet",
            "image_width": image_width,
            "image_height": image_height,
            "best_epoch": best_state["epoch"],
            "threshold_sweep": threshold_sweep,
        },
        checkpoint_path,
    )
    prediction_rows = []
    for row in val_predictions:
        prediction_rows.append(dict(row))
    predictions_csv = out_dir / "validation_predictions.csv"
    write_csv(predictions_csv, prediction_rows)
    write_csv(out_dir / "epoch_metrics.csv", epoch_rows)
    prediction_contact_sheet = render_prediction_contact_sheet(val_predictions, out_dir, seed=seed)
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": portable(dataset_dir),
        "out_dir": portable(out_dir),
        "dataset_discipline": discipline,
        "parameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "image_width": image_width,
            "image_height": image_height,
            "learning_rate": learning_rate,
            "device": str(device),
            "seed": seed,
            "visible_tolerance_px": VISIBLE_TOLERANCE_PX,
            "thresholds": list(thresholds),
        },
        "best_epoch": best_state["epoch"],
        "validation_threshold_sweep": threshold_sweep,
        "best_validation": best_validation,
        "baselines": baselines,
        "baseline_comparison": baseline_comparison,
        "outputs": {
            "checkpoint": portable(checkpoint_path),
            "validation_predictions_csv": portable(predictions_csv),
            "epoch_metrics_csv": portable(out_dir / "epoch_metrics.csv"),
            "validation_predictions_contact_sheet": portable(prediction_contact_sheet),
            "summary_json": portable(out_dir / "summary.json"),
            "report_md": portable(out_dir / "smoke_report.md"),
        },
    }
    write_json(out_dir / "summary.json", summary)
    write_markdown_report(out_dir / "smoke_report.md", summary)
    return summary


def parse_thresholds(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one threshold is required")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run center-heatmap detector smoke fine-tune on dense labels")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=296)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--thresholds", type=parse_thresholds, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--baseline-metrics", type=Path, action="append", default=[
        ROOT / "runs/release-27-public/dense_trajectory_review_v2_with_sources/eval_reviewed_v10_greedy/dense_trajectory_metrics.json",
        ROOT / "runs/release-27-public/dense_trajectory_review_v2_with_sources/eval_reviewed_v10_temporal/dense_trajectory_metrics.json",
        ROOT / "runs/release-27-public/dense_trajectory_review_v2_with_sources/eval_reviewed_v11_greedy/dense_trajectory_metrics.json",
    ])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_dense_heatmap_smoke(
        dataset_dir=args.dataset_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_width=args.image_width,
        image_height=args.image_height,
        learning_rate=args.learning_rate,
        device_name=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
        thresholds=args.thresholds,
        baseline_metrics=args.baseline_metrics,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return
    print(f"summary: {args.out_dir / 'summary.json'}")
    print(json.dumps(summary["best_validation"], indent=2))


if __name__ == "__main__":
    main()
