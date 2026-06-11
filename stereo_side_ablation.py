#!/usr/bin/env python3
"""Does the Ray-Ban Meta stereo mic pair carry contact SIDE (left vs right foot)?

A left-foot contact happens ~30cm left of the head, so the left channel should
get the transient slightly earlier (ITD, ~<1ms) and slightly louder (ILD).
Neither cue has ever been tested: every existing audio feature is mono.

Method: for each of the 76 side-labeled contacts, slice a +/-80ms stereo window
at the contact time, isolate the transient band, and compute interaural cues
(RMS ratio dB, onset-energy ratio, GCC-PHAT lag). Evaluate leave-one-video-out
logistic regression against the 0.671 majority ("right") baseline, and also
report the single-cue sign rules (no fitting at all).

Read-only on videos; writes a report + out-of-fold score file next to the other
ablations under touch_corpus_v1/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import audio_type_ablation as ab
import train_release_contact_classifier as tc

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = tc.DEFAULT_CORPUS / "stereo_side_ablation_v1"
SAMPLE_RATE = 48000
WINDOW_SEC = 0.08
BAND_LO, BAND_HI = 300.0, 8000.0
MAX_ITD_SEC = 0.0015  # head-width acoustic path bound, generous
BASELINE = 0.671


def video_file_for(video_id: str, search_dirs: list[Path]) -> Path | None:
    # video-340_singular_display-2 -> "video-340_singular_display 2.MOV"
    stem = video_id.replace("-2", " 2") if video_id.endswith("-2") else video_id
    for d in search_dirs:
        for cand in (d / f"{stem}.MOV", d / f"{stem}.mov", d / f"{video_id}.MOV"):
            if cand.exists():
                return cand
    return None


def load_stereo(path: Path) -> np.ndarray:
    """Decode the full stereo track to float32 [2, n] at SAMPLE_RATE."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(SAMPLE_RATE), "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T
    return audio


def bandpass(x: np.ndarray) -> np.ndarray:
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.shape[-1], 1.0 / SAMPLE_RATE)
    spec[..., (freqs < BAND_LO) | (freqs > BAND_HI)] = 0.0
    return np.fft.irfft(spec, n=x.shape[-1])


def gcc_phat_lag(left: np.ndarray, right: np.ndarray) -> float:
    """Lag (seconds) of left relative to right; negative = left leads."""
    n = left.shape[0] * 2
    lf, rf = np.fft.rfft(left, n=n), np.fft.rfft(right, n=n)
    cross = lf * np.conj(rf)
    denom = np.abs(cross)
    denom[denom < 1e-12] = 1e-12
    cc = np.fft.irfft(cross / denom, n=n)
    max_lag = int(MAX_ITD_SEC * SAMPLE_RATE)
    cc = np.concatenate((cc[-max_lag:], cc[: max_lag + 1]))
    return (int(np.argmax(cc)) - max_lag) / SAMPLE_RATE


def stereo_features(audio: np.ndarray, t_sec: float) -> dict[str, float] | None:
    half = int(WINDOW_SEC * SAMPLE_RATE)
    center = int(t_sec * SAMPLE_RATE)
    lo, hi = center - half, center + half
    if lo < 0 or hi > audio.shape[1]:
        return None
    win = bandpass(audio[:, lo:hi].astype(np.float64))
    left, right = win[0], win[1]
    eps = 1e-10
    rms_db = 20.0 * np.log10((np.sqrt(np.mean(left**2)) + eps) / (np.sqrt(np.mean(right**2)) + eps))
    # Onset: the loudest 10ms of the summed envelope -- the contact transient itself.
    env = left**2 + right**2
    k = int(0.010 * SAMPLE_RATE)
    csum = np.cumsum(env)
    seg = np.argmax(csum[k:] - csum[:-k])
    ol, orr = left[seg: seg + k], right[seg: seg + k]
    onset_db = 20.0 * np.log10((np.sqrt(np.mean(ol**2)) + eps) / (np.sqrt(np.mean(orr**2)) + eps))
    lag = gcc_phat_lag(ol, orr)
    return {
        "stereo_rms_db": float(rms_db),
        "stereo_onset_db": float(onset_db),
        "stereo_itd_ms": float(lag * 1000.0),
    }


def lovo_eval(rows: list[dict[str, Any]], feat_keys: list[str]) -> list[dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    videos = sorted({r["video_id"] for r in rows})
    preds = []
    for vid in videos:
        train = [r for r in rows if r["video_id"] != vid]
        test = [r for r in rows if r["video_id"] == vid]
        if len({r["contact_side"] for r in train}) < 2:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        model.fit([[r[k] for k in feat_keys] for r in train], [r["contact_side"] for r in train])
        feats = [[r[k] for k in feat_keys] for r in test]
        out = model.predict(feats)
        conf = model.predict_proba(feats).max(axis=1)
        for r, p, c in zip(test, out, conf):
            preds.append({"video_id": r["video_id"], "candidate_time_sec": r["candidate_time_sec"],
                          "label": r["contact_side"], "prediction": str(p),
                          "confidence": round(float(c), 6), "correct": r["contact_side"] == str(p)})
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--search-dir", type=Path, action="append", default=[Path.home() / "Downloads"])
    args = parser.parse_args()

    rows = [r for r in ab.load_labeled_rows() if r.get("contact_side") in ("left", "right")]
    print(f"side-labeled contacts: {len(rows)}  dist={Counter(r['contact_side'] for r in rows)}")

    cache: dict[str, np.ndarray | None] = {}
    feat_rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for r in rows:
        vid = str(r["video_id"])
        if vid not in cache:
            path = video_file_for(vid, args.search_dir)
            cache[vid] = load_stereo(path) if path else None
            if path is None:
                print(f"!! no video file for {vid}")
        audio = cache[vid]
        if audio is None:
            skipped.append(vid)
            continue
        t = float(r.get("contact_label_time_sec") or r.get("candidate_time_sec"))
        feats = stereo_features(audio, t)
        if feats is None:
            skipped.append(f"{vid}@{t}")
            continue
        feat_rows.append({"video_id": vid, "candidate_time_sec": float(r["candidate_time_sec"]),
                          "contact_side": str(r["contact_side"]), **feats})

    keys = ["stereo_rms_db", "stereo_onset_db", "stereo_itd_ms"]
    report: dict[str, Any] = {"n": len(feat_rows), "skipped": len(skipped), "baseline_majority": BASELINE,
                              "window_sec": WINDOW_SEC, "band_hz": [BAND_LO, BAND_HI]}

    # Zero-fit sign rules: left foot should be louder-left (db > 0) and earlier-left (itd < 0).
    for key, rule in [("stereo_rms_db", lambda v: "left" if v > 0 else "right"),
                      ("stereo_onset_db", lambda v: "left" if v > 0 else "right"),
                      ("stereo_itd_ms", lambda v: "left" if v < 0 else "right")]:
        acc = float(np.mean([rule(r[key]) == r["contact_side"] for r in feat_rows]))
        means = {s: float(np.mean([r[key] for r in feat_rows if r["contact_side"] == s])) for s in ("left", "right")}
        report[f"sign_rule_{key}"] = {"accuracy": round(acc, 4), "class_means": {k: round(v, 4) for k, v in means.items()}}
        print(f"sign rule {key}: acc={acc:.3f}  means L/R = {means['left']:.3f} / {means['right']:.3f}")

    preds = lovo_eval(feat_rows, keys)
    acc = float(np.mean([p["correct"] for p in preds])) if preds else 0.0
    per_class = {s: float(np.mean([p["correct"] for p in preds if p["label"] == s])) for s in ("left", "right")}
    bal = float(np.mean(list(per_class.values())))
    report["lovo"] = {"n": len(preds), "accuracy": round(acc, 4), "balanced_accuracy": round(bal, 4),
                      "per_class_recall": {k: round(v, 4) for k, v in per_class.items()},
                      "beats_majority_baseline": acc > BASELINE}
    print(f"LOVO stereo-only: acc={acc:.3f} bal={bal:.3f} (baseline {BASELINE}) "
          f"recall L/R = {per_class['left']:.3f}/{per_class['right']:.3f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "stereo_side_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.out_dir / "contact_side_oof_scores.jsonl").open("w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(p, sort_keys=True) + "\n")
    with (args.out_dir / "stereo_features.jsonl").open("w", encoding="utf-8") as fh:
        for r in feat_rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
