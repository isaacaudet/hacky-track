#!/usr/bin/env python3
"""Attach audio timbre features to touch candidate rows.

Audio onset strength alone is not reliable for this project because footsteps
can be louder than soft footbag touches. This step adds short-window spectral
shape features around each candidate so the classifier can learn contact sound
timbre without treating loudness as the whole signal.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = ROOT / "runs/release-27-public/touch_corpus_v1/touch_training_dataset_v1"
DEFAULT_AUDIO_WAV_DIR = ROOT / "runs/release-27-public/touch_corpus_v1/audio_candidates/wav"


@dataclass
class AudioCacheItem:
    samples: np.ndarray
    sr: int


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def wav_path_for_video(audio_wav_dir: Path, video_name: str) -> Path:
    return audio_wav_dir / f"{Path(video_name).stem}.wav"


def load_audio(path: Path) -> AudioCacheItem:
    import soundfile as sf

    samples, sr = sf.read(str(path), always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = samples.astype(np.float32, copy=False)
    return AudioCacheItem(samples=samples, sr=int(sr))


def safe_mean(value: np.ndarray) -> float | None:
    if value.size == 0:
        return None
    numeric = float(np.mean(value))
    return None if math.isnan(numeric) or math.isinf(numeric) else numeric


def band_energy_ratio(power: np.ndarray, freqs: np.ndarray, lo: float, hi: float, total: float) -> float | None:
    if total <= 1e-12:
        return None
    mask = (freqs >= lo) & (freqs < hi)
    if not bool(mask.any()):
        return 0.0
    return float(np.sum(power[mask]) / total)


def missing_features(status: str) -> dict[str, Any]:
    return {
        "audio_timbre_status": status,
        "audio_window_rms": None,
        "audio_window_peak": None,
        "audio_peak_to_rms": None,
        "audio_zero_crossing_rate": None,
        "audio_spectral_centroid": None,
        "audio_spectral_bandwidth": None,
        "audio_spectral_rolloff85": None,
        "audio_spectral_flatness": None,
        "audio_low_band_ratio": None,
        "audio_mid_band_ratio": None,
        "audio_high_band_ratio": None,
        "audio_pre_rms": None,
        "audio_post_rms": None,
        "audio_attack_ratio": None,
        "audio_mfcc_1": None,
        "audio_mfcc_2": None,
        "audio_mfcc_3": None,
        "audio_mfcc_4": None,
        "audio_mfcc_5": None,
        "audio_mfcc_6": None,
    }


def compute_audio_features(
    item: AudioCacheItem,
    time_sec: float,
    *,
    window_sec: float,
    n_fft: int,
    hop_length: int,
) -> dict[str, Any]:
    import librosa

    half = window_sec / 2.0
    sr = item.sr
    start = max(0, int(round((time_sec - half) * sr)))
    end = min(len(item.samples), int(round((time_sec + half) * sr)))
    if end - start < max(128, hop_length * 2):
        return missing_features("too_short_audio_window")
    y = np.asarray(item.samples[start:end], dtype=np.float32)
    if not np.any(np.isfinite(y)):
        return missing_features("nonfinite_audio_window")
    y = np.nan_to_num(y)

    rms = float(np.sqrt(np.mean(y * y)))
    peak = float(np.max(np.abs(y)))
    peak_to_rms = None if rms <= 1e-12 else peak / rms
    zcr = safe_mean(librosa.feature.zero_crossing_rate(y, frame_length=min(n_fft, len(y)), hop_length=hop_length))

    fft = min(n_fft, max(256, 2 ** int(np.floor(np.log2(len(y))))))
    if fft < 256:
        return missing_features("too_short_audio_fft")
    stft = np.abs(librosa.stft(y, n_fft=fft, hop_length=hop_length, center=True))
    power = stft * stft
    freqs = librosa.fft_frequencies(sr=sr, n_fft=fft)
    power_mean = np.mean(power, axis=1)
    total_power = float(np.sum(power_mean))

    centroid = safe_mean(librosa.feature.spectral_centroid(S=stft, sr=sr))
    bandwidth = safe_mean(librosa.feature.spectral_bandwidth(S=stft, sr=sr))
    rolloff = safe_mean(librosa.feature.spectral_rolloff(S=stft, sr=sr, roll_percent=0.85))
    flatness = safe_mean(librosa.feature.spectral_flatness(S=stft))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=6, n_fft=fft, hop_length=hop_length)
    mfcc_means = [float(np.mean(mfcc[i])) for i in range(min(6, mfcc.shape[0]))]

    center = int(round(time_sec * sr)) - start
    pre = y[: max(0, center)]
    post = y[min(len(y), center) :]
    pre_rms = None if len(pre) < 16 else float(np.sqrt(np.mean(pre * pre)))
    post_rms = None if len(post) < 16 else float(np.sqrt(np.mean(post * post)))
    attack_ratio = None
    if pre_rms is not None and post_rms is not None:
        attack_ratio = post_rms / max(pre_rms, 1e-9)

    out = {
        "audio_timbre_status": "ok",
        "audio_window_rms": round(rms, 9),
        "audio_window_peak": round(peak, 9),
        "audio_peak_to_rms": None if peak_to_rms is None else round(float(peak_to_rms), 6),
        "audio_zero_crossing_rate": None if zcr is None else round(float(zcr), 6),
        "audio_spectral_centroid": None if centroid is None else round(float(centroid), 6),
        "audio_spectral_bandwidth": None if bandwidth is None else round(float(bandwidth), 6),
        "audio_spectral_rolloff85": None if rolloff is None else round(float(rolloff), 6),
        "audio_spectral_flatness": None if flatness is None else round(float(flatness), 9),
        "audio_low_band_ratio": None if (v := band_energy_ratio(power_mean, freqs, 20, 300, total_power)) is None else round(v, 6),
        "audio_mid_band_ratio": None if (v := band_energy_ratio(power_mean, freqs, 300, 1600, total_power)) is None else round(v, 6),
        "audio_high_band_ratio": None if (v := band_energy_ratio(power_mean, freqs, 1600, sr / 2.0, total_power)) is None else round(v, 6),
        "audio_pre_rms": None if pre_rms is None else round(pre_rms, 9),
        "audio_post_rms": None if post_rms is None else round(post_rms, 9),
        "audio_attack_ratio": None if attack_ratio is None else round(float(attack_ratio), 6),
    }
    for i in range(6):
        out[f"audio_mfcc_{i + 1}"] = round(mfcc_means[i], 6) if i < len(mfcc_means) else None
    return out


def attach_audio_to_rows(
    rows: list[dict[str, Any]],
    *,
    audio_wav_dir: Path,
    window_sec: float,
    n_fft: int,
    hop_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audio_cache: dict[str, AudioCacheItem | None] = {}
    out_rows: list[dict[str, Any]] = []
    per_video: dict[str, dict[str, Any]] = {}
    for row in rows:
        video_name = str(row["video_name"])
        if video_name not in audio_cache:
            wav_path = wav_path_for_video(audio_wav_dir, video_name)
            audio_cache[video_name] = load_audio(wav_path) if wav_path.exists() else None
        audio = audio_cache[video_name]
        features = missing_features("missing_wav") if audio is None else compute_audio_features(
            audio,
            float(row["candidate_time_sec"]),
            window_sec=window_sec,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        out = dict(row)
        out.update(features)
        out_rows.append(out)
        stats = per_video.setdefault(
            video_name,
            {
                "video_name": video_name,
                "video_id": row.get("video_id"),
                "split": row.get("split"),
                "rows": 0,
                "positive_rows": 0,
                "ok_rows": 0,
                "missing_wav_rows": 0,
            },
        )
        stats["rows"] += 1
        stats["positive_rows"] += int(bool(row.get("label_is_touch")))
        stats["ok_rows"] += int(features.get("audio_timbre_status") == "ok")
        stats["missing_wav_rows"] += int(features.get("audio_timbre_status") == "missing_wav")
    summary = {
        "rows": len(out_rows),
        "ok_rows": sum(1 for row in out_rows if row.get("audio_timbre_status") == "ok"),
        "missing_wav_rows": sum(1 for row in out_rows if row.get("audio_timbre_status") == "missing_wav"),
        "videos": list(per_video.values()),
    }
    return out_rows, summary


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Touch Audio Timbre Feature Attachment",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Dataset dir: `{manifest['dataset_dir']}`",
        f"- Audio WAV dir: `{manifest['audio_wav_dir']}`",
        f"- Window: `{manifest['window_sec']}` sec",
        f"- Train/val rows: `{manifest['train_val']['rows']}`",
        f"- Train/val timbre rows: `{manifest['train_val']['ok_rows']}`",
        f"- Frozen-test rows: `{manifest['test_frozen']['rows']}`",
        f"- Frozen-test timbre rows: `{manifest['test_frozen']['ok_rows']}`",
        "",
        "## Per-Video",
        "",
        "| video | split | rows | positives | timbre | missing wav |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in manifest["videos"]:
        lines.append(
            f"| `{row['video_name']}` | {row.get('split')} | {row['rows']} | {row['positive_rows']} | "
            f"{row['ok_rows']} | {row['missing_wav_rows']} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- These are spectral/timbre descriptors around the candidate moment, not labels.",
            "- They are intended to help distinguish soft footbag contact sounds from louder footsteps.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def attach_dataset(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else dataset_dir
    audio_wav_dir = args.audio_wav_dir.resolve()
    train_rows = read_jsonl(dataset_dir / "touch_training_candidates.jsonl")
    test_rows = read_jsonl(dataset_dir / "touch_training_test_frozen.jsonl")
    train_out, train_summary = attach_audio_to_rows(
        train_rows,
        audio_wav_dir=audio_wav_dir,
        window_sec=args.window_sec,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )
    test_out, test_summary = attach_audio_to_rows(
        test_rows,
        audio_wav_dir=audio_wav_dir,
        window_sec=args.window_sec,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )
    if any(row.get("split") == "test_frozen" for row in train_out):
        raise AssertionError("test_frozen row leaked into train/validation output")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "touch_training_candidates.jsonl", train_out)
    write_jsonl(out_dir / "touch_training_test_frozen.jsonl", test_out)
    status = "features_attached" if train_summary["ok_rows"] or test_summary["ok_rows"] else "no_audio_timbre_features"
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "audio_wav_dir": str(audio_wav_dir),
        "window_sec": args.window_sec,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "train_val": train_summary,
        "test_frozen": test_summary,
        "videos": train_summary["videos"] + test_summary["videos"],
    }
    write_json(out_dir / "touch_audio_feature_manifest.json", manifest)
    write_report(out_dir / "touch_audio_feature_report.md", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach audio timbre features to touch training candidates")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--audio-wav-dir", type=Path, default=DEFAULT_AUDIO_WAV_DIR)
    parser.add_argument("--window-sec", type=float, default=0.16)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    manifest = attach_dataset(parse_args())
    out_dir = Path(manifest["out_dir"])
    print(f"manifest: {out_dir / 'touch_audio_feature_manifest.json'}")
    print(f"report:   {out_dir / 'touch_audio_feature_report.md'}")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "train_val_rows": manifest["train_val"]["rows"],
                "train_val_ok_rows": manifest["train_val"]["ok_rows"],
                "test_frozen_rows": manifest["test_frozen"]["rows"],
                "test_frozen_ok_rows": manifest["test_frozen"]["ok_rows"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
