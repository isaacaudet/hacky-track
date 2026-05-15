#!/usr/bin/env python3
"""Render a rough MS Paint-style HUD and reusable transparent HUD assets.

This is a second visual pass for the checked rally annotations. The event
JSON remains the source of truth for touch/rally counts; this script focuses
on a reusable, intentionally janky overlay style and small contact effects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_EVENTS = Path("data/video-50_singular_display.events.json")
DEFAULT_ANCHORS = Path("data/video-50_singular_display.paint_anchors.json")
DEFAULT_ASSETS = Path("assets/ms_paint_hud")

RGBA = tuple[int, int, int, int]

YELLOW: RGBA = (255, 235, 28, 255)
GREEN: RGBA = (48, 255, 42, 255)
CYAN: RGBA = (27, 217, 255, 255)
PINK: RGBA = (204, 105, 255, 255)
RED: RGBA = (255, 54, 54, 255)
WHITE: RGBA = (250, 250, 238, 255)
BLACK: RGBA = (5, 5, 5, 255)
INK: RGBA = (0, 0, 0, 255)
PAPER: RGBA = (255, 252, 229, 245)


@dataclass(frozen=True)
class Event:
    rally_id: int
    rally_label: str
    type: str
    time_sec: float
    touch_number: int | None = None
    duration_sec: float = 0.0
    label: str = ""


@dataclass(frozen=True)
class Rally:
    id: int
    label: str
    start_sec: float
    end_sec: float
    expected_touches: int
    expected_stalls: int
    events: tuple[Event, ...]


def stable_seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") & 0xFFFFFFFF


def stable_rng(key: str) -> np.random.Generator:
    return np.random.default_rng(stable_seed(key))


@lru_cache(maxsize=64)
def load_font(size: int, *, heavy: bool = True) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
        "/System/Library/Fonts/Supplemental/Arial Black.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/MarkerFelt.ttc",
        "/Library/Fonts/Arial Bold.ttf",
    ]
    if not heavy:
        candidates = [
            "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default()


def text_size(text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def rough_rect_points(width: int, height: int, pad: int, jitter: int, key: str) -> list[tuple[int, int]]:
    rng = stable_rng(key)
    left, top, right, bottom = pad, pad, width - pad - 1, height - pad - 1
    points: list[tuple[int, int]] = []
    steps = 5

    for i in range(steps + 1):
        x = round(left + (right - left) * i / steps)
        points.append((x + int(rng.integers(-jitter, jitter + 1)), top + int(rng.integers(-jitter, jitter + 1))))
    for i in range(1, steps + 1):
        y = round(top + (bottom - top) * i / steps)
        points.append((right + int(rng.integers(-jitter, jitter + 1)), y + int(rng.integers(-jitter, jitter + 1))))
    for i in range(1, steps + 1):
        x = round(right - (right - left) * i / steps)
        points.append((x + int(rng.integers(-jitter, jitter + 1)), bottom + int(rng.integers(-jitter, jitter + 1))))
    for i in range(1, steps):
        y = round(bottom - (bottom - top) * i / steps)
        points.append((left + int(rng.integers(-jitter, jitter + 1)), y + int(rng.integers(-jitter, jitter + 1))))
    return points


def draw_rough_rect(
    draw: ImageDraw.ImageDraw,
    size: tuple[int, int],
    *,
    fill: RGBA,
    outline: RGBA = INK,
    width: int = 4,
    pad: int = 7,
    jitter: int = 4,
    key: str = "rect",
) -> None:
    points = rough_rect_points(size[0], size[1], pad, jitter, key)
    draw.polygon(points, fill=fill)
    for i in range(3):
        jittered = rough_rect_points(size[0], size[1], pad, max(1, jitter - 1), f"{key}:outline:{i}")
        draw.line(jittered + [jittered[0]], fill=outline, width=width, joint="curve")


def draw_janky_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    *,
    fill: RGBA = WHITE,
    outline: RGBA = INK,
    stroke: int = 3,
    key: str,
) -> None:
    rng = stable_rng(key)
    x, y = xy
    offsets = [(0, 0)]
    for _ in range(max(14, stroke * 8)):
        dx = int(rng.integers(-stroke, stroke + 1))
        dy = int(rng.integers(-stroke, stroke + 1))
        if dx * dx + dy * dy <= (stroke + 1) * (stroke + 1):
            offsets.append((dx, dy))
    for dx, dy in offsets:
        draw.text((x + dx, y + dy), text, font=font, fill=outline)
    for _ in range(2):
        draw.text(
            (x + int(rng.integers(-1, 2)), y + int(rng.integers(-1, 2))),
            text,
            font=font,
            fill=fill,
        )


def rotate_janky(img: Image.Image, key: str, max_degrees: float = 2.4) -> Image.Image:
    rng = stable_rng(key)
    angle = float(rng.uniform(-max_degrees, max_degrees))
    return img.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)


def make_label_asset(
    text: str,
    *,
    fill: RGBA,
    text_fill: RGBA = BLACK,
    font_size: int = 30,
    pad_x: int = 18,
    pad_y: int = 10,
    key: str | None = None,
    rotate: bool = True,
) -> Image.Image:
    key = key or f"label:{text}:{fill}:{text_fill}:{font_size}"
    font = load_font(font_size)
    w, h = text_size(text, font)
    img = Image.new("RGBA", (w + pad_x * 2 + 12, h + pad_y * 2 + 12), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw_rough_rect(draw, img.size, fill=fill, outline=INK, width=4, pad=6, jitter=3, key=key)
    draw_janky_text(
        draw,
        (pad_x + 6, pad_y + 2),
        text,
        font,
        fill=text_fill,
        outline=INK if text_fill != BLACK else WHITE,
        stroke=2 if text_fill == BLACK else 3,
        key=f"{key}:text",
    )
    return rotate_janky(img, key) if rotate else img


def make_bare_text(
    text: str,
    *,
    fill: RGBA = WHITE,
    outline: RGBA = INK,
    font_size: int = 36,
    key: str | None = None,
) -> Image.Image:
    key = key or f"text:{text}:{font_size}:{fill}"
    font = load_font(font_size)
    w, h = text_size(text, font)
    img = Image.new("RGBA", (w + 18, h + 18), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw_janky_text(draw, (9, 5), text, font, fill=fill, outline=outline, stroke=3, key=key)
    return img


def make_digit_asset(char: str, fill: RGBA, key: str) -> Image.Image:
    font = load_font(96)
    bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 18
    img = Image.new("RGBA", (max(76, w + pad * 2), h + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw_janky_text(draw, (pad - bbox[0], pad - bbox[1]), char, font, fill=fill, outline=INK, stroke=5, key=key)
    return rotate_janky(img, key, max_degrees=3.0)


def draw_star(draw: ImageDraw.ImageDraw, center: tuple[int, int], r_outer: int, r_inner: int, fill: RGBA, key: str) -> None:
    rng = stable_rng(key)
    cx, cy = center
    points: list[tuple[int, int]] = []
    for i in range(18):
        radius = r_outer if i % 2 == 0 else r_inner
        radius += int(rng.integers(-5, 6))
        angle = -math.pi / 2 + i * math.tau / 18
        points.append((round(cx + math.cos(angle) * radius), round(cy + math.sin(angle) * radius)))
    draw.polygon(points, fill=fill)
    for i in range(3):
        jittered = [
            (x + int(rng.integers(-2, 3)), y + int(rng.integers(-2, 3)))
            for x, y in points
        ]
        draw.line(jittered + [jittered[0]], fill=INK, width=4 - min(i, 2))


def make_footbag_icon() -> Image.Image:
    img = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((13, 12, 83, 84), fill=WHITE, outline=INK, width=5)
    patches = [
        ((30, 14, 56, 40), RED),
        ((15, 34, 40, 61), GREEN),
        ((47, 36, 78, 64), CYAN),
        ((31, 56, 58, 83), YELLOW),
        ((51, 14, 76, 38), RED),
    ]
    for box, color in patches:
        draw.pieslice(box, 0, 360, fill=color, outline=INK, width=3)
    draw.line((48, 14, 48, 83), fill=INK, width=3)
    draw.line((18, 47, 79, 47), fill=INK, width=3)
    return rotate_janky(img, "footbag_icon", 4.0)


def make_foot_icon() -> Image.Image:
    img = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    skin = (244, 210, 164, 255)
    pts = [(28, 78), (30, 41), (43, 20), (55, 25), (55, 52), (68, 57), (72, 71), (62, 82), (42, 82)]
    draw.polygon(pts, fill=skin, outline=INK)
    draw.line(pts + [pts[0]], fill=INK, width=4)
    for i, x in enumerate([44, 51, 58, 64]):
        draw.ellipse((x, 53 + i, x + 9, 64 + i), fill=skin, outline=INK, width=2)
    return rotate_janky(img, "foot_icon", 4.0)


def make_spark_icon() -> Image.Image:
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw_star(draw, (50, 50), 42, 15, RED, "spark_icon")
    draw_star(draw, (50, 50), 25, 8, YELLOW, "spark_icon:inner")
    return img


def make_atw_star_icon() -> Image.Image:
    img = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw_star(draw, (64, 64), 54, 18, PINK, "atw_star:outer")
    draw_star(draw, (64, 64), 34, 11, YELLOW, "atw_star:inner")
    for i, box in enumerate([(13, 34, 115, 94), (24, 20, 104, 108)]):
        draw.arc(box, start=196 + i * 11, end=340 + i * 9, fill=INK, width=6)
        draw.arc(box, start=196 + i * 11, end=340 + i * 9, fill=CYAN, width=3)
    font = load_font(22)
    draw_janky_text(draw, (38, 51), "ATW", font, fill=BLACK, outline=WHITE, stroke=2, key="atw_star:text")
    return rotate_janky(img, "atw_star", 5.0)


def make_ticks_icon() -> Image.Image:
    img = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    lines = [((18, 24), (70, 8)), ((24, 48), (82, 48)), ((18, 72), (70, 88))]
    for i, (p1, p2) in enumerate(lines):
        draw.line(p1 + p2, fill=INK, width=9)
        draw.line((p1[0] + 1, p1[1] - 1, p2[0] + 1, p2[1] - 1), fill=CYAN, width=5)
    return rotate_janky(img, "ticks_icon", 3.0)


def make_pip(on: bool) -> Image.Image:
    img = Image.new("RGBA", (28, 28), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    fill = GREEN if on else (30, 30, 30, 70)
    outline = INK if on else (230, 230, 230, 210)
    draw.ellipse((4, 4, 23, 23), fill=fill, outline=outline, width=4)
    if on:
        draw.ellipse((9, 7, 13, 11), fill=(255, 255, 255, 170))
    return img


def make_panel(width: int, height: int, *, fill: RGBA, outline: RGBA, key: str) -> Image.Image:
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw_rough_rect(draw, img.size, fill=fill, outline=outline, width=5, pad=7, jitter=5, key=key)
    return img


def save_png(img: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def generate_assets(asset_dir: Path) -> list[Path]:
    generated: list[Path] = []
    for subdir in ["digits", "labels", "badges", "icons", "effects", "panels", "pips", "moves"]:
        (asset_dir / subdir).mkdir(parents=True, exist_ok=True)

    digit_colors = [YELLOW, GREEN, GREEN, CYAN, RED, RED, (255, 146, 31, 255), YELLOW, CYAN, PINK]
    for i, color in enumerate(digit_colors):
        generated.append(save_png(make_digit_asset(str(i), color, f"digit:{i}"), asset_dir / "digits" / f"digit_{i}.png"))
    generated.append(save_png(make_digit_asset("/", WHITE, "digit:slash"), asset_dir / "digits" / "slash.png"))
    generated.append(save_png(make_digit_asset("+", YELLOW, "digit:plus"), asset_dir / "digits" / "plus.png"))

    labels = [
        ("LIVE COUNT", GREEN, BLACK, "live_count"),
        ("RALLY", YELLOW, BLACK, "rally"),
        ("TOUCH", GREEN, BLACK, "touch"),
        ("TOUCHES", GREEN, BLACK, "touches"),
        ("TOTAL", CYAN, BLACK, "total"),
        ("BEST", PINK, BLACK, "best"),
        ("STALL", CYAN, BLACK, "stall"),
        ("DROP", PINK, BLACK, "drop"),
        ("RESET", RED, BLACK, "reset"),
        ("RALLY LIVE", YELLOW, BLACK, "rally_live"),
    ]
    for text, fill, text_fill, name in labels:
        generated.append(
            save_png(
                make_label_asset(text, fill=fill, text_fill=text_fill, font_size=30, key=f"label:{name}"),
                asset_dir / "labels" / f"label_{name}.png",
            )
        )

    generated.append(save_png(make_panel(260, 165, fill=(2, 2, 2, 225), outline=YELLOW, key="panel:live"), asset_dir / "panels" / "panel_live_counter.png"))
    generated.append(save_png(make_panel(340, 66, fill=(2, 2, 2, 215), outline=WHITE, key="panel:timeline"), asset_dir / "panels" / "panel_timeline.png"))
    generated.append(save_png(make_panel(168, 74, fill=PAPER, outline=INK, key="panel:chip"), asset_dir / "panels" / "panel_chip_blank.png"))

    generated.append(save_png(make_label_asset("FLOOR RESET!", fill=RED, text_fill=BLACK, font_size=34, key="badge:reset"), asset_dir / "badges" / "badge_floor_reset.png"))
    generated.append(save_png(make_label_asset("STALL!", fill=CYAN, text_fill=BLACK, font_size=36, key="badge:stall"), asset_dir / "badges" / "badge_stall.png"))
    generated.append(save_png(make_label_asset("+1", fill=YELLOW, text_fill=BLACK, font_size=42, key="badge:plus_one"), asset_dir / "badges" / "badge_plus_one.png"))
    generated.append(save_png(make_label_asset("AROUND THE WORLD", fill=PINK, text_fill=BLACK, font_size=30, key="badge:around_the_world"), asset_dir / "badges" / "badge_around_the_world.png"))
    generated.append(save_png(make_label_asset("ATW CLEAN", fill=PINK, text_fill=BLACK, font_size=32, key="badge:atw_clean"), asset_dir / "badges" / "badge_atw_clean.png"))

    move_specs = [
        ("READY", WHITE, BLACK, "ready"),
        ("DROP FROM HAND", WHITE, BLACK, "drop_from_hand"),
        ("RIGHT KICK", YELLOW, BLACK, "right_kick"),
        ("LEFT KICK", GREEN, BLACK, "left_kick"),
        ("OUTER RIGHT", CYAN, BLACK, "outer_right"),
        ("OUTER LEFT", CYAN, BLACK, "outer_left"),
        ("RIGHT STALL", CYAN, BLACK, "right_stall"),
        ("AROUND THE WORLD", PINK, BLACK, "around_the_world"),
    ]
    for text, fill, text_fill, name in move_specs:
        generated.append(
            save_png(
                make_label_asset(text, fill=fill, text_fill=text_fill, font_size=22, pad_x=14, pad_y=8, key=f"move:{name}"),
                asset_dir / "moves" / f"move_{name}.png",
            )
        )

    generated.append(save_png(make_footbag_icon(), asset_dir / "icons" / "footbag_icon.png"))
    generated.append(save_png(make_foot_icon(), asset_dir / "icons" / "foot_icon.png"))
    generated.append(save_png(make_spark_icon(), asset_dir / "effects" / "effect_spark.png"))
    generated.append(save_png(make_atw_star_icon(), asset_dir / "effects" / "effect_atw_star.png"))
    generated.append(save_png(make_ticks_icon(), asset_dir / "effects" / "effect_ticks.png"))
    generated.append(save_png(make_pip(True), asset_dir / "pips" / "pip_on.png"))
    generated.append(save_png(make_pip(False), asset_dir / "pips" / "pip_off.png"))

    generated.append(save_png(make_asset_sheet(asset_dir, generated), asset_dir / "asset_sheet.png"))
    return generated


def make_asset_sheet(asset_dir: Path, paths: list[Path]) -> Image.Image:
    thumbs = [p for p in paths if p.suffix.lower() == ".png" and p.name != "asset_sheet.png"]
    cols = 5
    cell_w, cell_h = 220, 150
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGBA", (cols * cell_w, rows * cell_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    label_font = load_font(14, heavy=False)
    for idx, path in enumerate(thumbs):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline=(218, 218, 218, 255), width=1)
        img = Image.open(path).convert("RGBA")
        img.thumbnail((160, 92), Image.Resampling.LANCZOS)
        sheet.alpha_composite(img, (x + (cell_w - img.width) // 2, y + 10))
        rel = str(path.relative_to(asset_dir))
        draw.text((x + 10, y + 112), rel, fill=(0, 0, 0, 255), font=label_font)
    return sheet.convert("RGB")


def load_event_doc(path: Path) -> tuple[list[Rally], list[Event], dict[str, Any]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    rallies: list[Rally] = []
    flat: list[Event] = []
    for rally_doc in doc["rallies"]:
        events: list[Event] = []
        for event_doc in rally_doc["events"]:
            event = Event(
                rally_id=int(rally_doc["id"]),
                rally_label=str(rally_doc["label"]),
                type=str(event_doc["type"]),
                time_sec=float(event_doc["time_sec"]),
                touch_number=event_doc.get("touch_number"),
                duration_sec=float(event_doc.get("duration_sec", 0.0)),
                label=str(event_doc.get("label", "")),
            )
            events.append(event)
            flat.append(event)
        rallies.append(
            Rally(
                id=int(rally_doc["id"]),
                label=str(rally_doc["label"]),
                start_sec=float(rally_doc["start_sec"]),
                end_sec=float(rally_doc["end_sec"]),
                expected_touches=int(rally_doc.get("expected_touches", 0)),
                expected_stalls=int(rally_doc.get("expected_stalls", 0)),
                events=tuple(events),
            )
        )
    return rallies, sorted(flat, key=lambda item: item.time_sec), doc


def event_key(event: Event) -> str:
    number = event.touch_number if event.touch_number is not None else 0
    return f"r{event.rally_id}:{event.type}:{number}:{event.time_sec:.2f}"


def load_anchors(path: Path) -> dict[str, tuple[float, float]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    anchors: dict[str, tuple[float, float]] = {}
    for item in data.get("anchors", []):
        anchors[str(item["key"])] = (float(item["x"]), float(item["y"]))
    return anchors


def resize_rgba(src: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize RGBA with premultiplied alpha so soft cutout edges stay clean."""
    if src.shape[1] == size[0] and src.shape[0] == size[1]:
        return src
    interpolation = cv2.INTER_AREA if size[0] < src.shape[1] or size[1] < src.shape[0] else cv2.INTER_LANCZOS4
    src_f = src.astype(np.float32)
    alpha = src_f[:, :, 3:4] / 255.0
    rgb_premultiplied = src_f[:, :, :3] * alpha
    rgb_resized = cv2.resize(rgb_premultiplied, size, interpolation=interpolation)
    alpha_resized = cv2.resize(src_f[:, :, 3], size, interpolation=interpolation)
    alpha_norm = np.maximum(alpha_resized[:, :, None] / 255.0, 1e-6)
    rgb = np.clip(rgb_resized / alpha_norm, 0, 255)
    out = np.dstack([rgb, np.clip(alpha_resized, 0, 255)])
    out[alpha_resized <= 0] = 0
    return out.astype(np.uint8)


def overlay_rgba(frame_bgr: np.ndarray, asset_rgba: np.ndarray, x: int, y: int, scale: float = 1.0) -> None:
    if asset_rgba.size == 0:
        return
    src = asset_rgba
    if abs(scale - 1.0) > 1e-3:
        new_w = max(1, int(round(src.shape[1] * scale)))
        new_h = max(1, int(round(src.shape[0] * scale)))
        src = resize_rgba(src, (new_w, new_h))
    h, w = frame_bgr.shape[:2]
    src_h, src_w = src.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + src_w), min(h, y + src_h)
    if x1 >= x2 or y1 >= y2:
        return
    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
    region = src[sy1:sy2, sx1:sx2].astype(np.float32)
    alpha = region[:, :, 3:4] / 255.0
    rgb = region[:, :, :3][:, :, ::-1]
    dst = frame_bgr[y1:y2, x1:x2].astype(np.float32)
    frame_bgr[y1:y2, x1:x2] = np.clip(rgb * alpha + dst * (1.0 - alpha), 0, 255).astype(np.uint8)


def pil_to_rgba_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGBA"))


class AssetBank:
    def __init__(self, asset_dir: Path) -> None:
        self.asset_dir = asset_dir
        self._rgba_cache: dict[str, np.ndarray] = {}
        self._pil_cache: dict[str, Image.Image] = {}

    def rgba(self, rel_path: str) -> np.ndarray:
        if rel_path not in self._rgba_cache:
            self._rgba_cache[rel_path] = np.array(Image.open(self.asset_dir / rel_path).convert("RGBA"))
        return self._rgba_cache[rel_path]

    def pil(self, rel_path: str) -> Image.Image:
        if rel_path not in self._pil_cache:
            self._pil_cache[rel_path] = Image.open(self.asset_dir / rel_path).convert("RGBA")
        return self._pil_cache[rel_path]

    @lru_cache(maxsize=256)
    def digit_string(self, text: str, height: int) -> np.ndarray:
        parts: list[Image.Image] = []
        for char in text:
            if char.isdigit():
                img = self.pil(f"digits/digit_{char}.png").copy()
            elif char == "/":
                img = self.pil("digits/slash.png").copy()
            elif char == "+":
                img = self.pil("digits/plus.png").copy()
            else:
                spacer = Image.new("RGBA", (height // 3, height), (0, 0, 0, 0))
                parts.append(spacer)
                continue
            ratio = height / max(1, img.height)
            img = img.resize((max(1, round(img.width * ratio)), height), Image.Resampling.LANCZOS)
            parts.append(img)
        if not parts:
            return np.zeros((1, 1, 4), dtype=np.uint8)
        overlap = max(0, height // 10)
        width = sum(part.width for part in parts) - overlap * (len(parts) - 1) + 8
        canvas = Image.new("RGBA", (width, height + 8), (0, 0, 0, 0))
        x = 4
        rng = stable_rng(f"digits:{text}:{height}")
        for part in parts:
            y = int(rng.integers(0, 6))
            canvas.alpha_composite(part, (x, y))
            x += part.width - overlap
        return pil_to_rgba_np(canvas)

    @lru_cache(maxsize=256)
    def chip(self, text: str, fill: RGBA, text_fill: RGBA, font_size: int) -> np.ndarray:
        return pil_to_rgba_np(
            make_label_asset(
                text,
                fill=fill,
                text_fill=text_fill,
                font_size=font_size,
                key=f"dynamic_chip:{text}:{fill}:{font_size}",
                rotate=True,
            )
        )

    @lru_cache(maxsize=128)
    def bare_text(self, text: str, fill: RGBA, font_size: int) -> np.ndarray:
        return pil_to_rgba_np(make_bare_text(text, fill=fill, font_size=font_size, key=f"bare:{text}:{font_size}:{fill}"))


def find_active_rally(t: float, rallies: list[Rally]) -> Rally | None:
    for rally in rallies:
        if rally.start_sec - 0.2 <= t <= rally.end_sec + 0.8:
            return rally
    if rallies and t < rallies[0].start_sec:
        return rallies[0]
    return None


def rally_state(t: float, rallies: list[Rally], events: list[Event]) -> dict[str, Any]:
    count_t = t + 0.045
    active = find_active_rally(t, rallies)
    current_touches = 0
    expected = 0
    rally_id = 0
    if active is not None:
        rally_id = active.id
        expected = active.expected_touches
        current_touches = sum(1 for event in active.events if event.type == "touch" and event.time_sec <= count_t)
        if t > active.end_sec + 0.35 and not any(event.type == "stall" and event.time_sec <= t <= event.time_sec + event.duration_sec for event in active.events):
            current_touches = 0
    total_touches = sum(1 for event in events if event.type == "touch" and event.time_sec <= count_t)
    best = 0
    for rally in rallies:
        touches = sum(1 for event in rally.events if event.type == "touch" and event.time_sec <= min(count_t, rally.end_sec))
        if rally.start_sec <= t:
            best = max(best, touches)
    stall = next((event for event in events if event.type == "stall" and event.time_sec <= t <= event.time_sec + max(0.6, event.duration_sec)), None)
    recent = [event for event in events if -0.04 <= t - event.time_sec <= 0.62]
    return {
        "active": active,
        "rally_id": rally_id,
        "rally_total": len(rallies),
        "current_touches": current_touches,
        "expected": expected,
        "total_touches": total_touches,
        "best": best,
        "stall": stall,
        "recent": recent,
    }


def detect_bag_center(frame_bgr: np.ndarray) -> tuple[float, float] | None:
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([28, 35, 35]), np.array([92, 255, 255]))
    yellow = cv2.inRange(hsv, np.array([12, 55, 45]), np.array([37, 255, 255]))
    mask = green | yellow
    gate = np.zeros_like(mask)
    gate[int(h * 0.38) :, :] = 255
    mask = cv2.bitwise_and(mask, gate)
    mask = cv2.medianBlur(mask, 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: tuple[float, float, float] | None = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 10 or area > 1200:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 3 or bh < 3 or bw > 80 or bh > 80:
            continue
        aspect = bw / max(1, bh)
        if aspect < 0.35 or aspect > 2.8:
            continue
        peri = cv2.arcLength(contour, True)
        circularity = 4 * math.pi * area / (peri * peri + 1e-6)
        cx, cy = x + bw / 2, y + bh / 2
        score = area * (0.5 + max(0.0, circularity)) * (0.65 + 0.7 * (cy / h))
        if cy < h * 0.58:
            score *= 0.42
        if cy < h * 0.52 and area > 250:
            score *= 0.25
        if best is None or score > best[0]:
            best = (score, cx, cy)
    if best is None:
        return None
    return best[1], best[2]


def auto_anchor_centers(video: Path, events: list[Event], anchors: dict[str, tuple[float, float]], scale: float) -> dict[str, tuple[float, float]]:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    centers: dict[str, tuple[float, float]] = {}
    for event in events:
        if event.type not in {"touch", "stall", "drop_floor"} or event_key(event) in anchors:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, round(event.time_sec * fps)))
        ok, frame = cap.read()
        if not ok:
            continue
        if abs(scale - 1.0) > 1e-3:
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        center = detect_bag_center(frame)
        if center is not None:
            h, w = frame.shape[:2]
            centers[event_key(event)] = (center[0] / w, center[1] / h)
    cap.release()
    return centers


def event_center(event: Event, frame_shape: tuple[int, int, int], anchors: dict[str, tuple[float, float]], auto_centers: dict[str, tuple[float, float]]) -> tuple[int, int]:
    h, w = frame_shape[:2]
    normalized = anchors.get(event_key(event)) or auto_centers.get(event_key(event))
    if normalized is None:
        drift = math.sin(event.time_sec * 2.1) * 0.12
        normalized = (0.5 + drift, 0.74)
    return int(round(normalized[0] * w)), int(round(normalized[1] * h))


def draw_effect_lines(frame: np.ndarray, center: tuple[int, int], age: float, key: str, *, stall: bool = False) -> None:
    if age < 0:
        return
    life = 0.72 if stall else 0.34
    if age > life:
        return
    fade = max(0.0, 1.0 - age / life)
    h, w = frame.shape[:2]
    effect = np.zeros((h, w, 4), dtype=np.uint8)
    rng = stable_rng(f"effect:{key}")
    cx, cy = center
    if stall:
        radius = int(28 + 6 * math.sin(age * 18))
        cv2.circle(effect, (cx, cy), radius, (*GREEN[:3], int(150 * fade)), 3, cv2.LINE_AA)
        cv2.circle(effect, (cx, cy), radius + 9, (*CYAN[:3], int(90 * fade)), 2, cv2.LINE_AA)
    else:
        for i in range(6):
            angle = i * math.tau / 6 + float(rng.uniform(-0.24, 0.24))
            r1 = 31 + int(rng.integers(-3, 5))
            r2 = 54 + int(rng.integers(-5, 8))
            p1 = (round(cx + math.cos(angle) * r1), round(cy + math.sin(angle) * r1))
            p2 = (round(cx + math.cos(angle) * r2), round(cy + math.sin(angle) * r2))
            cv2.line(effect, p1, p2, (0, 0, 0, int(150 * fade)), 4, cv2.LINE_AA)
            color = YELLOW if i % 2 == 0 else CYAN
            cv2.line(effect, p1, p2, (*color[:3], int(185 * fade)), 2, cv2.LINE_AA)
    overlay_rgba(frame, effect, 0, 0)


def tag_position(
    center: tuple[int, int],
    tag_shape: tuple[int, int, int],
    frame_shape: tuple[int, int, int],
    scale_ui: float,
) -> tuple[int, int]:
    h, w = frame_shape[:2]
    tag_h, tag_w = tag_shape[:2]
    gap = int(round(64 * scale_ui))
    margin = int(round(10 * scale_ui))
    candidates = [
        (center[0] + gap, center[1] - gap),
        (center[0] - tag_w - gap, center[1] - gap),
        (center[0] + gap, center[1] + int(gap * 0.55)),
        (center[0] - tag_w - gap, center[1] + int(gap * 0.55)),
    ]
    best = candidates[0]
    best_penalty = float("inf")
    for x, y in candidates:
        clamped_x = min(max(margin, x), max(margin, w - tag_w - margin))
        clamped_y = min(max(margin, y), max(margin, h - tag_h - margin))
        penalty = abs(clamped_x - x) + abs(clamped_y - y)
        # Prefer keeping the label outside the lower-center contact zone.
        if abs((clamped_x + tag_w / 2) - center[0]) < gap * 0.75 and abs((clamped_y + tag_h / 2) - center[1]) < gap * 0.75:
            penalty += 1000
        if penalty < best_penalty:
            best_penalty = penalty
            best = (clamped_x, clamped_y)
    return int(best[0]), int(best[1])


def draw_timeline(frame: np.ndarray, t: float, events: list[Event], duration: float, scale_ui: float) -> None:
    h, w = frame.shape[:2]
    x1, x2 = int(32 * scale_ui), w - int(42 * scale_ui)
    y = h - int(46 * scale_ui)
    rng = stable_rng("timeline")
    for pass_idx in range(3):
        yy = y + int(rng.integers(-2, 3))
        cv2.line(frame, (x1, yy), (x2, yy + int(rng.integers(-2, 3))), (0, 0, 0), int(7 * scale_ui), cv2.LINE_8)
    cv2.line(frame, (x1, y), (x2, y), (245, 245, 245), max(2, int(3 * scale_ui)), cv2.LINE_8)
    cv2.arrowedLine(frame, (x2 - int(22 * scale_ui), y), (x2, y), (245, 245, 245), max(2, int(3 * scale_ui)), tipLength=0.6)

    for event in events:
        if event.type not in {"touch", "stall", "drop_floor"}:
            continue
        x = int(round(x1 + (event.time_sec / duration) * (x2 - x1)))
        if event.type == "drop_floor":
            color = (45, 45, 255)
        elif event.type == "stall":
            color = (255, 220, 50)
        else:
            color = (25, 245, 255) if event.time_sec <= t else (85, 85, 85)
        radius = int((7 if event.time_sec <= t else 5) * scale_ui)
        cv2.circle(frame, (x, y), radius + 3, (0, 0, 0), -1, cv2.LINE_8)
        cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_8)

    progress_x = int(round(x1 + min(1.0, max(0.0, t / duration)) * (x2 - x1)))
    cv2.circle(frame, (progress_x, y), int(9 * scale_ui), (255, 255, 255), -1, cv2.LINE_8)
    cv2.circle(frame, (progress_x, y), int(5 * scale_ui), (0, 0, 0), -1, cv2.LINE_8)


def draw_hud(
    frame: np.ndarray,
    t: float,
    rallies: list[Rally],
    events: list[Event],
    duration: float,
    assets: AssetBank,
    anchors: dict[str, tuple[float, float]],
    auto_centers: dict[str, tuple[float, float]],
) -> None:
    h, w = frame.shape[:2]
    scale_ui = w / 688.0
    s = lambda value: int(round(value * scale_ui))
    state = rally_state(t, rallies, events)

    overlay_rgba(frame, assets.rgba("panels/panel_live_counter.png"), s(14), s(16), scale_ui)
    overlay_rgba(frame, assets.rgba("labels/label_live_count.png"), s(32), s(28), scale_ui * 0.78)
    expected = max(1, int(state["expected"] or state["current_touches"] or 1))
    count_text = f"{int(state['current_touches'])}/{expected}"
    overlay_rgba(frame, assets.digit_string(count_text, s(68)), s(30), s(66), 1.0)

    pip_y = s(138)
    pip_x = s(36)
    pip_gap = s(18)
    for idx in range(min(expected, 10)):
        rel = "pips/pip_on.png" if idx < int(state["current_touches"]) else "pips/pip_off.png"
        overlay_rgba(frame, assets.rgba(rel), pip_x + idx * pip_gap, pip_y, scale_ui * 0.68)

    overlay_rgba(frame, assets.bare_text("rally live", GREEN, s(20)), s(38), s(150), 1.0)

    chip_y = s(20)
    chips = [
        (f"RALLY {state['rally_id'] or 1}/{state['rally_total']}", YELLOW),
        (f"TOTAL {state['total_touches']}", CYAN),
        (f"BEST {state['best']}", PINK),
    ]
    chip_x = s(288)
    for text, color in chips:
        chip = assets.chip(text, color, BLACK, s(20))
        if chip_x + chip.shape[1] > w - s(8):
            chip_x = s(288)
            chip_y += s(42)
        overlay_rgba(frame, chip, chip_x, chip_y, 1.0)
        chip_x += int(chip.shape[1] + s(7))

    if state["stall"] is not None:
        overlay_rgba(frame, assets.rgba("badges/badge_stall.png"), w - s(162), s(86), scale_ui * 0.72)

    for event in state["recent"]:
        age = max(0.0, t - event.time_sec)
        if event.type == "touch":
            center = event_center(event, frame.shape, anchors, auto_centers)
            draw_effect_lines(frame, center, age, event_key(event))
            if age <= 0.46:
                tag = assets.chip("+1", YELLOW, BLACK, s(32))
                tag_x, tag_y = tag_position(center, tag.shape, frame.shape, scale_ui)
                overlay_rgba(frame, tag, tag_x, tag_y, 1.0)
        elif event.type == "stall":
            center = event_center(event, frame.shape, anchors, auto_centers)
            draw_effect_lines(frame, center, age, event_key(event), stall=True)
            overlay_rgba(frame, assets.rgba("badges/badge_stall.png"), center[0] + s(18), center[1] - s(54), scale_ui * 0.56)
        elif event.type == "drop_floor" and age <= 0.55:
            overlay_rgba(frame, assets.rgba("badges/badge_floor_reset.png"), w // 2 - s(116), h - s(138), scale_ui * 0.66)

    draw_timeline(frame, t, events, duration, scale_ui)


def mux_audio(video_no_audio: Path, source_video: Path, output_video: Path) -> None:
    if not shutil.which("ffmpeg"):
        shutil.copyfile(video_no_audio, output_video)
        return
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_no_audio),
            "-i",
            str(source_video),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_video),
        ],
        check=True,
    )


def render_video(
    video: Path,
    out_dir: Path,
    events_path: Path,
    anchors_path: Path,
    asset_dir: Path,
    *,
    scale: float,
    max_seconds: float | None = None,
) -> Path:
    rallies, events, _ = load_event_doc(events_path)
    anchors = load_anchors(anchors_path)
    auto_centers = auto_anchor_centers(video, events, anchors, scale)
    assets = AssetBank(asset_dir)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = max(2, int(round(in_w * scale)) // 2 * 2)
    out_h = max(2, int(round(in_h * scale)) // 2 * 2)
    duration = total_frames / fps if total_frames else max(event.time_sec for event in events) + 1.0
    max_frames = total_frames
    if max_seconds is not None:
        max_frames = min(total_frames, max(1, int(round(max_seconds * fps))))
    rendered_until = max_frames / fps
    out_dir.mkdir(parents=True, exist_ok=True)
    output_video = out_dir / "paint_hud_overlay.mp4"

    with tempfile.TemporaryDirectory() as tmp:
        raw_video = Path(tmp) / "paint_hud_no_audio.mp4"
        writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create video writer: {raw_video}")
        frame_index = 0
        while True:
            if frame_index >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != (out_w, out_h):
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            t = frame_index / fps
            draw_hud(frame, t, rallies, events, duration, assets, anchors, auto_centers)
            writer.write(frame)
            frame_index += 1
        writer.release()
        cap.release()
        mux_audio(raw_video, video, output_video)
    preview_events = [event for event in events if event.time_sec <= rendered_until]
    write_preview_artifacts(output_video, preview_events or events, out_dir)
    return output_video


def read_frame_at(cap: cv2.VideoCapture, time_sec: float) -> np.ndarray | None:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, round(time_sec * fps)))
    ok, frame = cap.read()
    return frame if ok else None


def write_preview_artifacts(video: Path, events: list[Event], out_dir: Path) -> None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return
    preview_time = next((event.time_sec + 0.08 for event in events if event.rally_id == 2 and event.touch_number == 8), events[-1].time_sec)
    preview = read_frame_at(cap, preview_time)
    if preview is not None:
        cv2.imwrite(str(out_dir / "paint_hud_preview.jpg"), preview)

    thumbs: list[np.ndarray] = []
    for event in events:
        if event.type not in {"touch", "stall", "drop_floor"}:
            continue
        frame = read_frame_at(cap, event.time_sec + 0.08)
        if frame is None:
            continue
        label = f"R{event.rally_id} {event.type}"
        if event.touch_number is not None:
            label += f" {event.touch_number}"
        label += f" @ {event.time_sec:.2f}s"
        cv2.putText(frame, label, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, label, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (25, 255, 255), 2, cv2.LINE_AA)
        thumb_w = 344
        thumb_h = round(frame.shape[0] * thumb_w / frame.shape[1])
        thumbs.append(cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA))
    cap.release()
    if not thumbs:
        return
    cols = 4
    rows = math.ceil(len(thumbs) / cols)
    thumb_h, thumb_w = thumbs[0].shape[:2]
    sheet = np.full((rows * thumb_h, cols * thumb_w, 3), 255, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * thumb_h
        sheet[y : y + thumb_h, x : x + thumb_w] = thumb
    cv2.imwrite(str(out_dir / "paint_hud_contact_sheet.jpg"), sheet)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the sketchy MS Paint-style hacky sack HUD overlay.")
    parser.add_argument("video", type=Path, help="Source video path")
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS, help="Checked event JSON")
    parser.add_argument("--anchors", type=Path, default=DEFAULT_ANCHORS, help="Optional normalized contact anchor JSON")
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS, help="Output/reuse directory for HUD PNG assets")
    parser.add_argument("--out", type=Path, default=None, help="Output directory; defaults to outputs/<video stem>")
    parser.add_argument("--scale", type=float, default=0.75, help="Output video scale relative to source")
    parser.add_argument("--max-seconds", type=float, default=None, help="Render only the first N seconds")
    parser.add_argument("--assets-only", action="store_true", help="Only generate the reusable PNG asset pack")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    out_dir = args.out or Path("outputs") / args.video.stem
    generated = generate_assets(args.assets)
    if args.assets_only:
        print(f"Generated {len(generated)} assets in {args.assets}")
        return
    output = render_video(args.video, out_dir, args.events, args.anchors, args.assets, scale=args.scale, max_seconds=args.max_seconds)
    print(f"Wrote {output}")
    print(f"Wrote assets under {args.assets}")
    print(f"Wrote preview {out_dir / 'paint_hud_preview.jpg'}")
    print(f"Wrote contact sheet {out_dir / 'paint_hud_contact_sheet.jpg'}")


if __name__ == "__main__":
    main()
