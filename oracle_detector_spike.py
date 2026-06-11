#!/usr/bin/env python3
"""Zero-shot open-vocabulary detector oracle spike.

This evaluates whether a large pretrained open-vocabulary detector can localize
the footbag from the Round 2 dense center labels without training. It is still a
spike, but the scoring is strict enough to decide whether OWLv2 should become an
L1 candidate in the real pipeline:

- cache per-frame detections so multiple tolerances can be scored without
  rerunning the model;
- score by split, including sealed audit/test rows as evaluation only;
- report top-1 pass, oracle pass, fire rate, no-target false positives, and
  top-1-vs-oracle disagreements that often indicate multi-sack ambiguity.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
BATCH = ROOT / "runs/release-27-public/dense_trajectory_review_v2_with_sources"
LABELS = BATCH / "dense_trajectory_labels.reviewed.jsonl"
VIS_POS = {"visible", "partially_occluded"}
VIS_NEG = {"fully_occluded", "out_of_frame"}
DEFAULT_PROMPTS = ["a footbag", "a hacky sack", "a small ball", "a small round bean bag", "a ball"]
DEFAULT_THRESHOLDS = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5)
DEFAULT_TOLERANCES = (12.0, 15.0)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_float_list(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one numeric value is required")
    return values


def load_rows(labels: Path) -> list[dict[str, Any]]:
    rows = [row for row in read_jsonl(labels) if row.get("quality") == "reviewed"]
    out: list[dict[str, Any]] = []
    for row in rows:
        visibility = row.get("visibility")
        is_target = visibility in VIS_POS and row.get("x") is not None and row.get("y") is not None
        is_no_target = visibility in VIS_NEG
        if not is_target and not is_no_target:
            continue
        out.append(row)
    return out


def stratified(rows: list[dict[str, Any]], per_clip: int, seed: int = 11) -> list[dict[str, Any]]:
    if per_clip == 0:
        return rows
    by_clip: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_clip[row["clip_id"]].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for clip_id in sorted(by_clip):
        clip_rows = by_clip[clip_id][:]
        rng.shuffle(clip_rows)
        selected.extend(clip_rows[:per_clip])
    return selected


def frame_path(row: dict[str, Any], batch_dir: Path) -> Path:
    return batch_dir / row["frame_image"]


def row_key(row: dict[str, Any]) -> str:
    return f"{row['clip_id']}:{int(row['frame_index'])}"


def slice_name(row: dict[str, Any]) -> str:
    if row.get("training_use") == "audit_only":
        return "test_audit"
    return str(row.get("split") or "unknown")


def is_target_row(row: dict[str, Any]) -> bool:
    return row.get("visibility") in VIS_POS and row.get("x") is not None and row.get("y") is not None


def is_no_target_row(row: dict[str, Any]) -> bool:
    return row.get("visibility") in VIS_NEG


# Detection backends return list of {"score": float, "x": float, "y": float}.


def make_yoloworld(prompts: list[str], device: str):
    from ultralytics import YOLOWorld

    model = YOLOWorld("yolov8x-worldv2.pt")
    model.set_classes(prompts)

    def detect(path: Path) -> list[dict[str, float]]:
        result = model.predict(str(path), conf=0.001, verbose=False, device=device)
        detections: list[dict[str, float]] = []
        for box in result[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({"score": float(box.conf[0]), "x": (x1 + x2) / 2.0, "y": (y1 + y2) / 2.0})
        detections.sort(key=lambda item: -item["score"])
        return detections

    return detect


def make_owlv2(prompts: list[str], device: str, checkpoint: str = "google/owlv2-base-patch16-ensemble"):
    import torch
    from PIL import Image
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    processor = Owlv2Processor.from_pretrained(checkpoint)
    model = Owlv2ForObjectDetection.from_pretrained(checkpoint).to(device)
    model.train(False)

    def detect(path: Path) -> list[dict[str, float]]:
        image = Image.open(path).convert("RGB")
        inputs = processor(text=[prompts], images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([[image.size[1], image.size[0]]], device=device)
        result = processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=target_sizes)[0]
        detections: list[dict[str, float]] = []
        for score, box in zip(result["scores"].tolist(), result["boxes"].tolist()):
            x1, y1, x2, y2 = box
            detections.append({"score": float(score), "x": (x1 + x2) / 2.0, "y": (y1 + y2) / 2.0})
        detections.sort(key=lambda item: -item["score"])
        return detections

    return detect


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else intersection / union


def normalize_moondream_objects(
    objects: list[dict[str, Any]],
    *,
    image_width: int,
    image_height: int,
    prompt: str,
    prompt_index: int,
    score: float,
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for object_index, obj in enumerate(objects):
        try:
            x1 = float(obj["x_min"]) * image_width
            y1 = float(obj["y_min"]) * image_height
            x2 = float(obj["x_max"]) * image_width
            y2 = float(obj["y_max"]) * image_height
        except KeyError as exc:
            raise ValueError(f"Moondream detection is missing normalized bbox key: {obj}") from exc
        item_score = float(obj.get("score", score))
        detections.append(
            {
                "score": item_score,
                "x": (x1 + x2) / 2.0,
                "y": (y1 + y2) / 2.0,
                "bbox": [x1, y1, x2, y2],
                "prompt": prompt,
                "prompt_index": prompt_index,
                "object_index": object_index,
                "source": "moondream_detect",
            }
        )
    return detections


def dedupe_moondream_detections(detections: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
    ordered = sorted(
        detections,
        key=lambda item: (
            -float(item.get("score", 0.0)),
            int(item.get("prompt_index", 0)),
            int(item.get("object_index", 0)),
        ),
    )
    if iou_threshold <= 0.0:
        return ordered

    kept: list[dict[str, Any]] = []
    for detection in ordered:
        bbox = detection.get("bbox")
        if bbox is None:
            kept.append(detection)
            continue
        if all(bbox_iou([float(v) for v in bbox], [float(v) for v in kept_item.get("bbox", [])]) < iou_threshold for kept_item in kept if kept_item.get("bbox")):
            kept.append(detection)
    return kept


def make_moondream(
    prompts: list[str],
    *,
    api_key_env: str,
    model_name: str | None,
    local: bool,
    score: float,
    dedupe_iou: float,
):
    if not prompts:
        raise ValueError("Moondream detection requires at least one prompt.")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {api_key_env}=... before running Moondream detection.")
    try:
        import moondream as md  # type: ignore
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install Moondream dependencies with `pip install -r requirements-moondream.txt`.") from exc

    kwargs: dict[str, Any] = {"api_key": api_key}
    if local:
        kwargs["local"] = True
    if model_name:
        kwargs["model"] = model_name
    model = md.vl(**kwargs)

    def detect(path: Path) -> list[dict[str, Any]]:
        image = Image.open(path).convert("RGB")
        try:
            detector_image = model.encode_image(image)
        except AttributeError:
            detector_image = image
        detections: list[dict[str, Any]] = []
        for prompt_index, prompt in enumerate(prompts):
            result = model.detect(detector_image, prompt)
            detections.extend(
                normalize_moondream_objects(
                    list(result.get("objects", [])),
                    image_width=image.width,
                    image_height=image.height,
                    prompt=prompt,
                    prompt_index=prompt_index,
                    score=score,
                )
            )
        return dedupe_moondream_detections(detections, dedupe_iou)

    return detect


def dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def detection_records(
    *,
    rows: list[dict[str, Any]],
    batch_dir: Path,
    model_name: str,
    prompts: list[str],
    device: str,
    out_dir: Path,
    force: bool,
    moondream_model: str | None = None,
    moondream_local: bool = False,
    moondream_api_key_env: str = "MOONDREAM_API_KEY",
    moondream_score: float = 1.0,
    moondream_dedupe_iou: float = 0.85,
) -> list[dict[str, Any]]:
    cache_path = out_dir / "detections.jsonl"
    if cache_path.exists() and not force:
        cached = read_jsonl(cache_path)
        print(f"using cached detections: {cache_path} ({len(cached)} rows)")
        return cached

    detect = {
        "yoloworld": lambda: make_yoloworld(prompts, device),
        "owlv2": lambda: make_owlv2(prompts, device),
        "owlv2-large": lambda: make_owlv2(prompts, device, "google/owlv2-large-patch14-ensemble"),
        "moondream": lambda: make_moondream(
            prompts,
            api_key_env=moondream_api_key_env,
            model_name=moondream_model,
            local=moondream_local,
            score=moondream_score,
            dedupe_iou=moondream_dedupe_iou,
        ),
    }[model_name]()

    records: list[dict[str, Any]] = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        path = frame_path(row, batch_dir)
        if not path.exists():
            raise FileNotFoundError(f"missing frame image: {path}")
        detections = detect(path)
        records.append(
            {
                "key": row_key(row),
                "clip_id": row["clip_id"],
                "source_video": row.get("source_video"),
                "frame_index": int(row["frame_index"]),
                "time_sec": row.get("time_sec"),
                "frame_image": row["frame_image"],
                "split": row.get("split"),
                "training_use": row.get("training_use"),
                "slice": slice_name(row),
                "visibility": row.get("visibility"),
                "is_target": is_target_row(row),
                "is_no_target": is_no_target_row(row),
                "x": None if row.get("x") is None else float(row["x"]),
                "y": None if row.get("y") is None else float(row["y"]),
                "detections": detections,
            }
        )
        if index % 25 == 0 or index == total:
            print(f"  detected {index}/{total}")
    write_jsonl(cache_path, records)
    return records


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


def score_subset(records: list[dict[str, Any]], threshold: float, tolerance: float) -> dict[str, Any]:
    positives = [record for record in records if record["is_target"]]
    negatives = [record for record in records if record["is_no_target"]]
    top1_pass = 0
    oracle_pass = 0
    fired = 0
    errors: list[float] = []
    top1_fail_oracle_pass = 0
    multi_detection_frames = 0

    for record in positives:
        above = [det for det in record["detections"] if float(det["score"]) >= threshold]
        if len(above) >= 2:
            multi_detection_frames += 1
        if not above:
            continue
        fired += 1
        top = above[0]
        top_error = dist(float(top["x"]), float(top["y"]), float(record["x"]), float(record["y"]))
        errors.append(top_error)
        top_ok = top_error <= tolerance
        oracle_ok = any(dist(float(det["x"]), float(det["y"]), float(record["x"]), float(record["y"])) <= tolerance for det in above)
        if top_ok:
            top1_pass += 1
        if oracle_ok:
            oracle_pass += 1
        if oracle_ok and not top_ok:
            top1_fail_oracle_pass += 1

    no_target_fp = sum(1 for record in negatives if any(float(det["score"]) >= threshold for det in record["detections"]))
    return {
        "threshold": threshold,
        "tolerance_px": tolerance,
        "visible_frames": len(positives),
        "top1_pass_frames": top1_pass,
        "oracle_pass_frames": oracle_pass,
        "fired_visible_frames": fired,
        "top1_pass_rate": None if not positives else top1_pass / len(positives),
        "oracle_pass_rate": None if not positives else oracle_pass / len(positives),
        "fire_rate": None if not positives else fired / len(positives),
        "mean_error_px": None if not errors else sum(errors) / len(errors),
        "p95_error_px": percentile(errors, 0.95),
        "no_target_frames": len(negatives),
        "no_target_fp_frames": no_target_fp,
        "no_target_fp_rate": None if not negatives else no_target_fp / len(negatives),
        "multi_detection_visible_frames": multi_detection_frames,
        "top1_fail_oracle_pass_frames": top1_fail_oracle_pass,
    }


def build_slices(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    slices: dict[str, list[dict[str, Any]]] = {"all": records}
    for name in sorted({record["slice"] for record in records}):
        slices[name] = [record for record in records if record["slice"] == name]
    slices["train_validation"] = [record for record in records if record["slice"] in {"train", "validation"}]
    return slices


def score_all(records: list[dict[str, Any]], thresholds: tuple[float, ...], tolerances: tuple[float, ...]) -> dict[str, Any]:
    by_slice: dict[str, Any] = {}
    for name, subset in build_slices(records).items():
        by_tolerance: dict[str, list[dict[str, Any]]] = {}
        for tolerance in tolerances:
            by_tolerance[str(tolerance)] = [score_subset(subset, threshold, tolerance) for threshold in thresholds]
        by_slice[name] = by_tolerance
    return by_slice


def metric_at(summary: dict[str, Any], slice_key: str, tolerance: float, threshold: float) -> dict[str, Any]:
    rows = summary["by_slice"][slice_key][str(tolerance)]
    for row in rows:
        if float(row["threshold"]) == float(threshold):
            return row
    raise KeyError((slice_key, tolerance, threshold))


def format_rate(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def render_qa_sheets(
    records: list[dict[str, Any]],
    *,
    batch_dir: Path,
    out_dir: Path,
    threshold: float,
    tolerance: float,
    max_items: int,
) -> dict[str, str]:
    from PIL import Image, ImageDraw

    def best_oracle(record: dict[str, Any], above: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not above or record.get("x") is None or record.get("y") is None:
            return None
        return min(above, key=lambda det: dist(float(det["x"]), float(det["y"]), float(record["x"]), float(record["y"])))

    def make_sheet(selected: list[dict[str, Any]], path: Path) -> None:
        if not selected:
            return
        cols = 4
        cell = 260
        rows_n = math.ceil(len(selected) / cols)
        sheet = Image.new("RGB", (cols * cell, rows_n * cell), "black")
        for index, record in enumerate(selected):
            image = Image.open(batch_dir / record["frame_image"]).convert("RGB")
            draw = ImageDraw.Draw(image)
            above = [det for det in record["detections"] if float(det["score"]) >= threshold]
            if record["is_target"]:
                x = float(record["x"])
                y = float(record["y"])
                draw.ellipse([x - 12, y - 12, x + 12, y + 12], outline="lime", width=5)
            if above:
                top = above[0]
                draw.ellipse([top["x"] - 12, top["y"] - 12, top["x"] + 12, top["y"] + 12], outline="red", width=5)
                oracle = best_oracle(record, above)
                if oracle is not None and oracle is not top:
                    draw.rectangle([oracle["x"] - 12, oracle["y"] - 12, oracle["x"] + 12, oracle["y"] + 12], outline="yellow", width=5)
            label = f"{record['source_video']} f{record['frame_index']} {record['visibility']}"
            draw.rectangle([0, 0, image.width, 34], fill=(0, 0, 0))
            draw.text((8, 8), label[:80], fill=(255, 255, 255))
            image.thumbnail((cell, cell))
            sheet.paste(image, ((index % cols) * cell, (index // cols) * cell))
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(path)

    positives = [record for record in records if record["is_target"]]
    top1_fail_oracle_pass = []
    top1_fail = []
    for record in positives:
        above = [det for det in record["detections"] if float(det["score"]) >= threshold]
        if not above:
            continue
        top = above[0]
        top_ok = dist(float(top["x"]), float(top["y"]), float(record["x"]), float(record["y"])) <= tolerance
        oracle_ok = any(dist(float(det["x"]), float(det["y"]), float(record["x"]), float(record["y"])) <= tolerance for det in above)
        if oracle_ok and not top_ok:
            top1_fail_oracle_pass.append(record)
        if not top_ok:
            top1_fail.append(record)

    high_conf_neg = sorted(
        [record for record in records if record["is_no_target"] and any(float(det["score"]) >= threshold for det in record["detections"])],
        key=lambda record: max(float(det["score"]) for det in record["detections"]),
        reverse=True,
    )

    outputs: dict[str, str] = {}
    sheets = [
        ("top1_failures", top1_fail),
        ("top1_fail_oracle_pass", top1_fail_oracle_pass),
        ("no_target_false_positives", high_conf_neg),
    ]
    for name, selected in sheets:
        path = out_dir / "qa" / f"{name}_thr{threshold}_tol{int(tolerance)}.jpg"
        make_sheet(selected[:max_items], path)
        if path.exists():
            outputs[name] = str(path)
    return outputs


def write_report(path: Path, summary: dict[str, Any], headline_threshold: float) -> None:
    lines = [
        f"# Oracle Detector Validation — {summary['model']}",
        "",
        "Zero-shot open-vocabulary detection vs Round 2 dense center labels. No training.",
        f"Prompts: `{summary['prompts']}`",
        f"Frames scored: `{summary['record_count']}` reviewed rows.",
        "",
    ]
    for tolerance in summary["tolerances"]:
        lines.extend(
            [
                f"## Headline @ {headline_threshold} / {tolerance}px",
                "",
                "| slice | visible | top-1 pass | oracle pass | fire | mean err | p95 err | no-target | FP | multi-det | oracle-over-top1 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for slice_key in ["all", "train", "validation", "test_audit", "train_validation"]:
            if slice_key not in summary["by_slice"]:
                continue
            row = metric_at(summary, slice_key, tolerance, headline_threshold)
            mean_error = "-" if row["mean_error_px"] is None else f"{row['mean_error_px']:.1f}"
            p95_error = "-" if row["p95_error_px"] is None else f"{row['p95_error_px']:.1f}"
            lines.append(
                f"| {slice_key} | {row['visible_frames']} | {format_rate(row['top1_pass_rate'])} | "
                f"{format_rate(row['oracle_pass_rate'])} | {format_rate(row['fire_rate'])} | {mean_error} | {p95_error} | "
                f"{row['no_target_frames']} | {format_rate(row['no_target_fp_rate'])} | "
                f"{row['multi_detection_visible_frames']} | {row['top1_fail_oracle_pass_frames']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Full Threshold Sweep: All Frames",
            "",
            "| tolerance | threshold | top-1 pass | oracle pass | fire | mean err | p95 err | no-target FP |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for tolerance in summary["tolerances"]:
        for row in summary["by_slice"]["all"][str(tolerance)]:
            mean_error = "-" if row["mean_error_px"] is None else f"{row['mean_error_px']:.1f}"
            p95_error = "-" if row["p95_error_px"] is None else f"{row['p95_error_px']:.1f}"
            lines.append(
                f"| {tolerance} | {row['threshold']} | {format_rate(row['top1_pass_rate'])} | "
                f"{format_rate(row['oracle_pass_rate'])} | {format_rate(row['fire_rate'])} | {mean_error} | "
                f"{p95_error} | {format_rate(row['no_target_fp_rate'])} |"
            )
    lines.extend(
        [
            "",
            "## QA Images",
            "",
            "Green circle = ground truth, red circle = top-1 detection, yellow square = closest detection when top-1 is wrong.",
        ]
    )
    for name, image_path in sorted(summary.get("qa_sheets", {}).items()):
        lines.append(f"- `{name}`: `{image_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate zero-shot open-vocab detector against dense labels")
    parser.add_argument("--model", choices=["yoloworld", "owlv2", "owlv2-large", "moondream"], default="owlv2")
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--batch-dir", type=Path, default=BATCH)
    parser.add_argument("--per-clip", type=int, default=15, help="0 = all reviewed frames")
    parser.add_argument("--tolerances", type=parse_float_list, default=DEFAULT_TOLERANCES)
    parser.add_argument("--thresholds", type=parse_float_list, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--headline-threshold", type=float, default=0.2)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--qa-frames", type=int, default=12)
    parser.add_argument("--force", action="store_true", help="ignore cached detections.jsonl")
    parser.add_argument("--clips", nargs="*", default=None,
                        help="only score rows whose clip_id starts with one of these prefixes")
    parser.add_argument("--moondream-model", default=None, help="Optional Moondream model name, e.g. moondream2")
    parser.add_argument("--moondream-local", action="store_true", help="Run Moondream through local Photon instead of cloud")
    parser.add_argument("--moondream-api-key-env", default="MOONDREAM_API_KEY")
    parser.add_argument(
        "--moondream-score",
        type=float,
        default=1.0,
        help="Synthetic score assigned to Moondream boxes because detect does not return confidence.",
    )
    parser.add_argument(
        "--moondream-dedupe-iou",
        type=float,
        default=0.85,
        help="Suppress duplicate boxes from multiple Moondream prompts above this IoU; <=0 disables suppression.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (ROOT / f"runs/release-27-public/oracle_{args.model}_v1")
    out_dir = out_dir.resolve()
    labels = args.labels if args.labels.is_absolute() else (ROOT / args.labels)
    batch_dir = args.batch_dir if args.batch_dir.is_absolute() else (ROOT / args.batch_dir)
    rows = load_rows(labels)
    if args.clips:
        rows = [r for r in rows if any(r["clip_id"].startswith(p) for p in args.clips)]
    rows = stratified(rows, args.per_clip)

    print(f"model={args.model} prompts={args.prompts}")
    print(f"scoring {len(rows)} reviewed target/no-target frames (per_clip={args.per_clip})")
    records = detection_records(
        rows=rows,
        batch_dir=batch_dir,
        model_name=args.model,
        prompts=args.prompts,
        device=args.device,
        out_dir=out_dir,
        force=args.force,
        moondream_model=args.moondream_model,
        moondream_local=args.moondream_local,
        moondream_api_key_env=args.moondream_api_key_env,
        moondream_score=args.moondream_score,
        moondream_dedupe_iou=args.moondream_dedupe_iou,
    )
    by_slice = score_all(records, args.thresholds, args.tolerances)
    qa_sheets = render_qa_sheets(
        records,
        batch_dir=batch_dir,
        out_dir=out_dir,
        threshold=args.headline_threshold,
        tolerance=args.tolerances[0],
        max_items=args.qa_frames,
    )
    summary = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "prompts": args.prompts,
        "labels": str(labels),
        "batch_dir": str(batch_dir),
        "record_count": len(records),
        "per_clip": args.per_clip,
        "thresholds": list(args.thresholds),
        "tolerances": list(args.tolerances),
        "headline_threshold": args.headline_threshold,
        "model_parameters": {
            "moondream_model": args.moondream_model,
            "moondream_local": args.moondream_local,
            "moondream_api_key_env": args.moondream_api_key_env,
            "moondream_score": args.moondream_score,
            "moondream_dedupe_iou": args.moondream_dedupe_iou,
        }
        if args.model == "moondream"
        else {},
        "by_slice": by_slice,
        "qa_sheets": qa_sheets,
    }
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "report.md", summary, args.headline_threshold)
    print((out_dir / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
