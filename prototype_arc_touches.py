#!/usr/bin/env python3
"""Prototype: L2/L3 physics layer — recover touches from the ball trajectory.

Tests the second premise of DETECTOR_REPLACEMENT_PLAN_V2.md: a footbag rally is
a chain of ballistic arcs, and a *touch* is the breakpoint between two arcs.
So touch detection = optimally segmenting the ball track into piecewise-
parabolic pieces; each breakpoint is a candidate touch.

Method:
  1. Detect the ball each frame (HSV blob — reused from prototype_ego_motion).
  2. Optimal piecewise fit by dynamic programming: partition the (t, x, y)
     track into segments, each scored by a quadratic-in-t fit to y and a
     linear-in-t fit to x. A per-break penalty (lambda) controls sensitivity.
     Prefix sums make each segment cost O(1).
  3. Each breakpoint -> touch time, refined as the intersection of the two
     adjacent fitted parabolas.
  4. Match detected touches to ground-truth touches (events.json); report
     precision / recall over a lambda sweep.

The L0 prototype showed short ballistic arcs are parabolic in *raw* image
coordinates, so this runs without ego-motion compensation.

Run: python3 prototype_arc_touches.py   (outputs in tmp_proto/)
"""
import argparse
import json
import os

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prototype_ego_motion import detect_ball


# --------------------------------------------------------------------------
# Prefix-sum machinery for O(1) segment-fit cost
# --------------------------------------------------------------------------
class SegmentCost:
    """O(1) cost of fitting y~quadratic(t) and x~linear(t) to points [i, j]."""

    def __init__(self, t, x, y):
        t = np.asarray(t, float)
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        n = len(t)
        # prefix sums of t powers and cross terms (index m = sum over [0, m))
        def pre(a):
            return np.concatenate([[0.0], np.cumsum(a)])
        self.P = {p: pre(t ** p) for p in range(5)}
        self.Sy = pre(y)
        self.Sty = pre(t * y)
        self.St2y = pre(t * t * y)
        self.Syy = pre(y * y)
        self.Sx = pre(x)
        self.Stx = pre(t * x)
        self.Sxx = pre(x * x)
        self.n = n

    def _range(self, arr, i, j):
        return arr[j + 1] - arr[i]

    def cost(self, i, j):
        cnt = j - i + 1
        if cnt < 4:
            return np.inf
        S = {p: self._range(self.P[p], i, j) for p in range(5)}
        # quadratic fit to y: design columns [1, t, t^2]
        My = np.array([[S[0], S[1], S[2]],
                       [S[1], S[2], S[3]],
                       [S[2], S[3], S[4]]])
        by = np.array([self._range(self.Sy, i, j),
                       self._range(self.Sty, i, j),
                       self._range(self.St2y, i, j)])
        # linear fit to x: design columns [1, t]
        Mx = np.array([[S[0], S[1]], [S[1], S[2]]])
        bx = np.array([self._range(self.Sx, i, j), self._range(self.Stx, i, j)])
        try:
            beta_y = np.linalg.solve(My, by)
            beta_x = np.linalg.solve(Mx, bx)
        except np.linalg.LinAlgError:
            return np.inf
        sse_y = self._range(self.Syy, i, j) - beta_y @ by
        sse_x = self._range(self.Sxx, i, j) - beta_x @ bx
        return max(sse_y, 0.0) + max(sse_x, 0.0)


def segment(t, x, y, lam):
    """DP optimal piecewise fit. Returns list of (start_idx, end_idx) segments."""
    sc = SegmentCost(t, x, y)
    n = len(t)
    dp = np.full(n + 1, np.inf)
    dp[0] = 0.0
    prev = np.zeros(n + 1, int)
    for m in range(4, n + 1):  # dp[m] = best cost covering points [0, m)
        for i in range(0, m - 3):  # segment = points [i, m-1]
            c = dp[i] + sc.cost(i, m - 1) + (lam if i > 0 else 0.0)
            if c < dp[m]:
                dp[m] = c
                prev[m] = i
    segs = []
    m = n
    while m > 0:
        i = prev[m]
        segs.append((i, m - 1))
        m = i
    return segs[::-1]


def parabola(t, y):
    return np.polyfit(np.asarray(t, float), np.asarray(y, float), 2)


def touch_time_at_break(tl, yl, tr, yr, gap_lo, gap_hi):
    """Touch time = intersection of the two adjacent parabolas, else gap mid."""
    cl, cr = parabola(tl, yl), parabola(tr, yr)
    d = cl - cr  # roots of (left - right) = 0
    roots = np.roots(d) if abs(d[0]) > 1e-9 else (
        np.array([-d[2] / d[1]]) if abs(d[1]) > 1e-9 else np.array([]))
    inside = [float(r) for r in roots if np.isreal(r) and gap_lo <= r.real <= gap_hi]
    return inside[0] if inside else 0.5 * (gap_lo + gap_hi)


def match(detected, truth, tol):
    """Greedy match. Returns (true_pos, n_detected, n_truth)."""
    truth = sorted(truth)
    used = [False] * len(truth)
    tp = 0
    for d in sorted(detected):
        best, bj = tol + 1e-9, -1
        for j, g in enumerate(truth):
            if not used[j] and abs(d - g) < best:
                best, bj = abs(d - g), j
        if bj >= 0:
            used[bj] = True
            tp += 1
    return tp, len(detected), len(truth)


def build_synthetic(touch_times, fps=30.0, sigma=0.0, outlier_frac=0.0,
                    dropout=0.0, n_bursts=0, seed=0):
    """Synthetic footbag rally: exact ballistic arcs between known touches.

    Each inter-touch arc is a true parabola (ball rises from foot level and
    falls back); x is linear per arc. Touches are real velocity kinks. With
    all corruption params 0 the track is noise-free ground truth.
      sigma        Gaussian centroid jitter (px)
      outlier_frac fraction of frames replaced by an i.i.d. wrong-blob detection
      dropout      fraction of frames with no detection (gap)
      n_bursts     sustained wrong-blob lock-on runs (8-16 frames each)
    """
    rng = np.random.default_rng(seed)
    ts, xs, ys = [], [], []
    y_foot, x_cur = 800.0, 350.0
    for k in range(len(touch_times) - 1):
        ti, tj = touch_times[k], touch_times[k + 1]
        d = tj - ti
        if d <= 0.05:
            continue
        height = rng.uniform(250, 600)          # kick height (px)
        a = 4.0 * height / d**2                 # parabola curvature
        vx = rng.uniform(-180, 180)
        for f in range(int(np.ceil(ti * fps)), int(np.floor(tj * fps)) + 1):
            tt = f / fps
            ys.append(y_foot + a * (tt - ti) * (tt - tj))
            xs.append(x_cur + vx * (tt - ti))
            ts.append(tt)
        x_cur += vx * d
    ts, xs, ys = np.array(ts), np.array(xs), np.array(ys)
    if sigma > 0:
        xs = xs + rng.normal(0, sigma, len(xs))
        ys = ys + rng.normal(0, sigma, len(ys))
    if outlier_frac > 0:                        # i.i.d. wrong-blob detections
        m = rng.random(len(xs)) < outlier_frac
        xs[m] = rng.uniform(50, 680, int(m.sum()))
        ys[m] = rng.uniform(50, 920, int(m.sum()))
    for _ in range(n_bursts):                   # sustained wrong-blob lock-on
        if len(xs) < 40:
            break
        s = int(rng.integers(0, len(xs) - 20))
        L = int(rng.integers(8, 17))
        wx, wy = rng.uniform(60, 670), rng.uniform(60, 900)
        xs[s:s + L] = wx + rng.normal(0, 5, min(L, len(xs) - s))
        ys[s:s + L] = wy + rng.normal(0, 5, min(L, len(xs) - s))
    if dropout > 0:                             # missed detections (gaps)
        keep = rng.random(len(xs)) >= dropout
        ts, xs, ys = ts[keep], xs[keep], ys[keep]
    return ts, xs, ys


def robust_clean(t, x, y, win=6, k=3.5, floor_px=12.0, iters=2):
    """Drop gross outliers: points far from a local median-scaled quadratic
    trend of their neighbours. Catches wrong-blob detections; keeps real arcs.
    """
    keep = np.ones(len(t), bool)
    for _ in range(iters):
        idx = np.where(keep)[0]
        drop = []
        for pos, i in enumerate(idx):
            nb = idx[max(0, pos - win):pos + win + 1]
            nb = nb[nb != i]
            if len(nb) < 6:
                continue
            cy = np.polyfit(t[nb], y[nb], 2)
            cx = np.polyfit(t[nb], x[nb], 2)
            res = np.hypot(y[nb] - np.polyval(cy, t[nb]),
                           x[nb] - np.polyval(cx, t[nb]))
            scale = 1.4826 * np.median(res)
            r = np.hypot(y[i] - np.polyval(cy, t[i]), x[i] - np.polyval(cx, t[i]))
            if r > k * scale + floor_px:
                drop.append(i)
        if not drop:
            break
        keep[drop] = False
    return keep


def detect_touches(tc, x, y, t0, lam):
    """Run segmentation, return detected touch times (absolute)."""
    segs = segment(tc, x, y, lam)
    return [touch_time_at_break(tc[s0:e0 + 1], y[s0:e0 + 1],
                                tc[s1:e1 + 1], y[s1:e1 + 1], tc[e0], tc[s1]) + t0
            for (s0, e0), (s1, e1) in zip(segs, segs[1:])]


def _best_score(ts, xs, ys, gt, tol):
    """Best touch P/R/F1 over a lambda sweep for one (ts,xs,ys) track."""
    tc = ts - ts.mean()
    best = {"f1": -1.0}
    for lam in (200, 600, 1800, 5000, 14000):
        det = detect_touches(tc, xs, ys, ts.mean(), lam)
        tp, nd, ng = match(det, gt, tol)
        prec = tp / nd if nd else 0.0
        rec = tp / ng if ng else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best["f1"]:
            best = {"precision": round(prec, 3), "recall": round(rec, 3),
                    "f1": round(f1, 3)}
    return best


def synthetic_conditions(touch_times, tol, out_dir):
    """Inject each realistic corruption in turn to localise what breaks L2."""
    gt = sorted(touch_times)[1:-1]   # interior touches are the arc breakpoints
    conds = [
        ("clean (sigma=0)", dict()),
        ("Gaussian noise sigma=12px", dict(sigma=12)),
        ("+ 8% i.i.d. outliers", dict(sigma=12, outlier_frac=0.08)),
        ("+ 8% i.i.d. outliers + robust filter", dict(sigma=12, outlier_frac=0.08)),
        ("+ 25% dropout", dict(sigma=12, dropout=0.25)),
        ("+ outlier BURSTS (sustained lock-on)", dict(sigma=12, n_bursts=5)),
        ("+ outlier bursts + robust filter", dict(sigma=12, n_bursts=5)),
    ]
    rows = []
    for name, kw in conds:
        ts, xs, ys = build_synthetic(touch_times, seed=7, **kw)
        if "robust filter" in name:
            keep = robust_clean(ts, xs, ys)
            ts, xs, ys = ts[keep], xs[keep], ys[keep]
        sc = _best_score(ts, xs, ys, gt, tol)
        rows.append({"condition": name, **sc})
        print(f"  {name:<38} P={sc['precision']:.2f} "
              f"R={sc['recall']:.2f} F1={sc['f1']:.2f}")
    fig, axp = plt.subplots(figsize=(11, 5))
    pos = np.arange(len(rows))
    axp.bar(pos - 0.22, [r["precision"] for r in rows], 0.22, label="precision")
    axp.bar(pos, [r["recall"] for r in rows], 0.22, label="recall")
    axp.bar(pos + 0.22, [r["f1"] for r in rows], 0.22, label="F1")
    axp.set_xticks(pos)
    axp.set_xticklabels([r["condition"] for r in rows], rotation=20, ha="right", fontsize=7)
    axp.set_ylabel("touch detection score")
    axp.set_title("Synthetic control — what breaks L2 touch detection, and the fix\n"
                  "(algorithm is exact when clean; gross outliers are the killer; "
                  "a robust pre-filter recovers it)")
    axp.legend()
    axp.set_ylim(0, 1.05)
    axp.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/synthetic_conditions.png", dpi=110)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="/Users/isaacaudet/Downloads/video-506_singular_display.MOV")
    ap.add_argument("--events", default="data/video-506_singular_display.events.json")
    ap.add_argument("--work-width", type=int, default=731)
    ap.add_argument("--tolerance-sec", type=float, default=0.20)
    ap.add_argument("--out-dir", default="tmp_proto")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ev = json.load(open(args.events))
    rally = ev["rallies"][0]
    touches_gt = sorted(e["time_sec"] for r in ev["rallies"]
                        for e in r["events"] if e.get("type") == "touch")
    stalls_gt = [(e["time_sec"], e["time_sec"] + e.get("duration_sec", 0.0))
                 for r in ev["rallies"] for e in r["events"] if e.get("type") == "stall"]
    f_lo = max(0, int(rally["start_sec"] * 30) - 8)
    f_hi = int(rally["end_sec"] * 30) + 8

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    work_h = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * args.work_width
                       / cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_lo)

    track = []  # (t_sec, x, y)
    prev_xy = None
    fi = f_lo
    while fi <= f_hi:
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.resize(fr, (args.work_width, work_h))
        b = detect_ball(fr, prev_xy)
        if b is not None:
            track.append((fi / fps, b[0], b[1]))
            prev_xy = (b[0], b[1])
        fi += 1
    cap.release()

    t = np.array([p[0] for p in track])
    x = np.array([p[1] for p in track])
    y = np.array([p[2] for p in track])
    t0 = t.mean()
    tc = t - t0  # centred for numerical stability

    def pr(detected):
        tp, nd, ng = match(detected, touches_gt, args.tolerance_sec)
        prec = tp / nd if nd else 0.0
        rec = tp / ng if ng else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {"n_detected": nd, "true_pos": tp, "precision": round(prec, 3),
                "recall": round(rec, 3), "f1": round(f1, 3)}

    # --- Stage 1: pure trajectory segmentation (lambda sweep) -------------
    seg_sweep = []
    for lam in (200, 400, 800, 1500, 3000, 6000, 12000):
        segs = segment(tc, x, y, lam)
        det = [touch_time_at_break(tc[s0:e0 + 1], y[s0:e0 + 1],
                                   tc[s1:e1 + 1], y[s1:e1 + 1], tc[e0], tc[s1]) + t0
               for (s0, e0), (s1, e1) in zip(segs, segs[1:])]
        seg_sweep.append({"lambda": lam, "n_segments": len(segs), **pr(det)})

    # --- Stage 2: velocity-kink confirmation -----------------------------
    # Fix a high-recall lambda, then keep only breaks where the ball's
    # velocity changes sharply (a real touch redirects the ball; detector
    # noise barely moves it). This is intrinsic to the trajectory -- no model.
    lam = 800
    segs = segment(tc, x, y, lam)
    breaks = []
    for (s0, e0), (s1, e1) in zip(segs, segs[1:]):
        cyL, cyR = parabola(tc[s0:e0 + 1], y[s0:e0 + 1]), parabola(tc[s1:e1 + 1], y[s1:e1 + 1])
        cxL = np.polyfit(tc[s0:e0 + 1], x[s0:e0 + 1], 1)
        cxR = np.polyfit(tc[s1:e1 + 1], x[s1:e1 + 1], 1)
        tt = touch_time_at_break(tc[s0:e0 + 1], y[s0:e0 + 1],
                                 tc[s1:e1 + 1], y[s1:e1 + 1], tc[e0], tc[s1])
        # speed change at the break (px/s): foot redirects the ball
        dvy = (2 * cyR[0] * tt + cyR[1]) - (2 * cyL[0] * tt + cyL[1])
        dvx = cxR[0] - cxL[0]
        breaks.append({"t": tt + t0, "delta_v": float(np.hypot(dvx, dvy))})

    dv_sweep = []
    for thr in (0, 150, 300, 500, 800, 1200, 1800, 2600):
        det = [b["t"] for b in breaks if b["delta_v"] >= thr]
        dv_sweep.append({"delta_v_threshold": thr, "n_kept": len(det), **pr(det)})
    best = max(dv_sweep, key=lambda r: r["f1"])
    best_det = [b["t"] for b in breaks if b["delta_v"] >= best["delta_v_threshold"]]

    # --- Stage 3: synthetic control — localise what breaks L2 ------------
    print("\nSynthetic control (exact ballistic arcs, known touches):")
    synth = synthetic_conditions(touches_gt, args.tolerance_sec, args.out_dir)

    # --- Stage 4: robust pre-filter applied to the real HSV track --------
    keep = robust_clean(t, x, y)
    tcl, xcl, ycl = t[keep], x[keep], y[keep]
    tccl = tcl - tcl.mean()
    real_clean = []
    for lam in (200, 400, 800, 1500, 3000, 6000, 12000):
        real_clean.append({"lambda": lam,
                            **pr(detect_touches(tccl, xcl, ycl, tcl.mean(), lam))})
    real_raw_best = max(r["f1"] for r in seg_sweep)
    real_clean_best = max(r["f1"] for r in real_clean)

    summary = {
        "video": os.path.basename(args.video),
        "ball_detections": len(track),
        "ground_truth_touches": len(touches_gt),
        "tolerance_sec": args.tolerance_sec,
        "stage1_segmentation_only": seg_sweep,
        "stage2_velocity_confirmation": dv_sweep,
        "stage1_best_f1": max(r["f1"] for r in seg_sweep),
        "stage2_best": {k: best[k] for k in
                        ("delta_v_threshold", "precision", "recall", "f1")},
        "stage3_synthetic_conditions": synth,
        "stage4_real_track_robust_filtered": real_clean,
        "stage4_real_detections_kept": int(keep.sum()),
        "stage4_real_detections_dropped": int((~keep).sum()),
        "real_clip_f1_raw_vs_cleaned": [real_raw_best, real_clean_best],
    }
    json.dump(summary, open(f"{args.out_dir}/arc_touch_metrics.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))

    # plot
    fig, ax = plt.subplots(2, 1, figsize=(14, 8))
    ax[0].plot(t, y, ".", ms=4, color="0.6", label="ball y(t) (HSV detection)")
    for (s, e) in segs:
        c = parabola(tc[s:e + 1], y[s:e + 1])
        ts = np.linspace(tc[s], tc[e], 40)
        ax[0].plot(ts + t0, np.polyval(c, ts), "-", color="green", lw=2)
    for g in touches_gt:
        ax[0].axvline(g, color="tab:blue", ls="-", lw=1, alpha=0.55)
    for d in best_det:
        ax[0].axvline(d, color="tab:red", ls="--", lw=1.3)
    for s0, s1 in stalls_gt:
        ax[0].axvspan(s0, s1, color="orange", alpha=0.15)
    ax[0].invert_yaxis()
    ax[0].set_ylabel("ball y (px)")
    ax[0].set_xlabel("time (s)")
    ax[0].set_title(f"L2/L3 — video-506  |  green = piecewise parabola fit  |  "
                    f"blue = true touch, red dashed = detected (after velocity confirmation)  "
                    f"P={best['precision']:.2f} R={best['recall']:.2f} F1={best['f1']:.2f}")
    ax[0].legend(loc="upper right", fontsize=8)
    thr = [r["delta_v_threshold"] for r in dv_sweep]
    ax[1].plot(thr, [r["precision"] for r in dv_sweep], "o-", label="precision")
    ax[1].plot(thr, [r["recall"] for r in dv_sweep], "s-", label="recall")
    ax[1].plot(thr, [r["f1"] for r in dv_sweep], "^-", label="F1")
    ax[1].axhline(max(r["f1"] for r in seg_sweep), color="0.5", ls=":",
                  label="best F1, segmentation only")
    ax[1].set_xlabel("velocity-kink threshold (px/s)  —  confirmation strength")
    ax[1].set_title("touch precision/recall vs velocity-kink confirmation")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/arc_touches.png", dpi=110)

    print(f"\nStage 1  real clip, segmentation only : best F1 = {real_raw_best:.2f}")
    print(f"Stage 4  real clip, + robust filter   : best F1 = {real_clean_best:.2f}  "
          f"({keep.sum()}/{len(keep)} detections kept)")


if __name__ == "__main__":
    main()
