#!/usr/bin/env python3
"""Prototype: camera ego-motion compensation (L0) for footbag tracking.

Validates the core premise of DETECTOR_REPLACEMENT_PLAN_V2.md: once camera
head-motion is removed, the footbag's image-space path between touches becomes
piecewise-parabolic (ballistic), so physics-based arc fitting becomes valid.

Vision-only: consumer Ray-Ban Meta footage exposes no IMU/gyro.

Pipeline for one clip:
  1. Detect the ball each frame (HSV colour blob — distinctive coral footbag).
  2. Estimate frame-to-frame camera motion (LK optical flow on background
     features, with the ball region masked out) -> 4-DOF similarity transform.
  3. Accumulate transforms into a per-frame map back to a reference frame.
  4. Warp raw ball centres into the reference frame -> "stabilised" track.
  5. Fit a parabola to y(t) on each inter-touch arc; report fit residual for
     the raw vs stabilised track. Lower stabilised residual = premise holds.

Outputs (under --out-dir):
  ego_motion_tracks.png   raw vs stabilised x(t)/y(t) with per-arc parabola fits
  stabilised_average.png  / raw_average.png  visual stabilisation check
  prototype_metrics.json  numeric summary
"""
import argparse
import json
import os

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Ball detection (HSV colour blob)
# --------------------------------------------------------------------------
# The coral footbag is a saturated red (hue near the 0/180 wrap); dead grass is
# a desaturated yellow-green (hue ~38). Hue + saturation cleanly separate them.
# These thresholds are per-clip on purpose — DETECTOR_REPLACEMENT_PLAN_V2.md
# treats ball appearance as a per-video model, not a universal detector.
def detect_ball(frame_bgr, prev_xy=None):
    """Return (x, y, radius) of the coral footbag blob, or None.

    With prev_xy supplied, prefers the candidate nearest the last known
    position (continuity) over the merely-largest blob.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.int16)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    red_hue = (h < 13) | (h > 166)
    mask = (red_hue & (s > 130) & (v > 110)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 15 or area > 8000:
            continue
        m = cv2.moments(c)
        if m["m00"] <= 0:
            continue
        x = m["m10"] / m["m00"]  # intensity-stable centroid
        y = m["m01"] / m["m00"]
        r = float(np.sqrt(area / np.pi))
        cands.append((float(x), float(y), r, area))
    if not cands:
        return None
    if prev_xy is not None and prev_xy[0] is not None:
        px, py = prev_xy
        cands.sort(key=lambda c: (c[0] - px) ** 2 + (c[1] - py) ** 2)
        best = cands[0]
        if (best[0] - px) ** 2 + (best[1] - py) ** 2 > 260 ** 2:
            best = max(cands, key=lambda c: c[3])  # continuity lost; take largest
    else:
        best = max(cands, key=lambda c: c[3])
    return best[0], best[1], best[2]


# --------------------------------------------------------------------------
# L0: camera ego-motion estimation
# --------------------------------------------------------------------------
def estimate_motion(prev_gray, gray, ball_xy_r):
    """Estimate the similarity transform mapping prev_gray -> gray.

    Background features only: the ball region is masked out so the fast-moving
    footbag never contaminates the camera-motion estimate. Returns a 2x3 affine
    matrix and the number of RANSAC inliers (None if estimation failed).
    """
    h, w = prev_gray.shape
    bg_mask = np.full((h, w), 255, np.uint8)
    if ball_xy_r is not None:
        bx, by, br = ball_xy_r
        cv2.circle(bg_mask, (int(bx), int(by)), int(max(40, br * 4)), 0, -1)
    pts = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=700, qualityLevel=0.01, minDistance=8, mask=bg_mask
    )
    if pts is None or len(pts) < 12:
        return None, 0
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, pts, None, winSize=(21, 21), maxLevel=3
    )
    st = st.reshape(-1).astype(bool)
    p0, p1 = pts[st], nxt[st]
    if len(p0) < 12:
        return None, 0
    affine, inliers = cv2.estimateAffinePartial2D(
        p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if affine is None:
        return None, 0
    return affine, int(inliers.sum())


def to_3x3(affine):
    m = np.eye(3, dtype=np.float64)
    m[:2, :] = affine
    return m


# --------------------------------------------------------------------------
# Physics check: parabola fit residual per inter-touch arc
# --------------------------------------------------------------------------
def arc_fit_residual(times, ys, xs):
    """Fit y=quadratic(t), x=linear(t) over one arc. Return RMSE in px."""
    if len(times) < 5:
        return None
    t = np.asarray(times)
    cy = np.polyfit(t, ys, 2)
    cx = np.polyfit(t, xs, 1)
    ry = np.asarray(ys) - np.polyval(cy, t)
    rx = np.asarray(xs) - np.polyval(cx, t)
    rmse = float(np.sqrt(np.mean(ry**2 + rx**2)))
    return rmse, cy, cx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="/Users/isaacaudet/Downloads/video-506_singular_display.MOV")
    ap.add_argument("--events", default="data/video-506_singular_display.events.json")
    ap.add_argument("--start-frame", type=int, default=30)
    ap.add_argument("--end-frame", type=int, default=165)
    ap.add_argument("--work-width", type=int, default=731)
    ap.add_argument("--out-dir", default="tmp_proto")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    scale = args.work_width / src_w
    work_h = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale))

    # event times (seconds): touches are drawn as markers; touches + stall
    # start/end are arc boundaries (a stall is not one ballistic segment).
    touches, kink_times = [], []
    if os.path.exists(args.events):
        ev = json.load(open(args.events))
        for rally in ev.get("rallies", []):
            for e in rally.get("events", []):
                if e.get("type") == "touch":
                    touches.append(e["time_sec"])
                    kink_times.append(e["time_sec"])
                elif e.get("type") == "stall":
                    kink_times.append(e["time_sec"])
                    kink_times.append(e["time_sec"] + e.get("duration_sec", 0.0))
    touches.sort()
    kink_times.sort()

    frames, balls, grays = [], [], []
    fi = args.start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    while fi <= args.end_frame:
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.resize(fr, (args.work_width, work_h))
        frames.append(fi)
        grays.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        balls.append(fr)
        fi += 1
    cap.release()

    # detect ball per frame, using continuity from the previous detection
    raw_ball = []
    prev_xy = None
    for f in balls:
        b = detect_ball(f, prev_xy)
        raw_ball.append(b)
        if b is not None:
            prev_xy = (b[0], b[1])
    n = len(frames)

    # accumulate per-frame transform: H[i] maps reference(frame 0) -> frame i
    H = [np.eye(3)]
    last_affine = np.eye(2, 3, dtype=np.float64)
    inlier_counts = []
    for i in range(1, n):
        affine, inl = estimate_motion(grays[i - 1], grays[i], raw_ball[i])
        if affine is None:
            affine = last_affine  # reuse previous motion on a bad frame
        else:
            last_affine = affine
        inlier_counts.append(inl)
        H.append(to_3x3(affine) @ H[-1])

    # stabilise ball centres: map frame-i coords back to the reference frame
    t = [(frames[i] - frames[0]) / fps for i in range(n)]
    raw_xy = [(b[0], b[1]) if b else (None, None) for b in raw_ball]
    stab_xy = []
    for i in range(n):
        if raw_ball[i] is None:
            stab_xy.append((None, None))
            continue
        inv = np.linalg.inv(H[i])
        p = inv @ np.array([raw_ball[i][0], raw_ball[i][1], 1.0])
        stab_xy.append((float(p[0]), float(p[1])))

    # camera-motion magnitude: how far a fixed reference point (frame centre)
    # is pushed by head motion, expressed in reference-frame pixels.
    cx, cy = args.work_width / 2.0, work_h / 2.0
    ref_center = []
    for i in range(n):
        p = np.linalg.inv(H[i]) @ np.array([cx, cy, 1.0])
        ref_center.append((p[0], p[1]))
    cam_step = [0.0] + [
        float(np.hypot(ref_center[i][0] - ref_center[i - 1][0],
                       ref_center[i][1] - ref_center[i - 1][1]))
        for i in range(1, n)
    ]
    cam_path_total = float(np.sum(cam_step))

    # L0 self-check: warp every frame into the reference frame, save the
    # average. (A drifting average is expected over a long pan — the rigorous
    # check is the per-arc residual below, not this picture.)
    acc_stab = np.zeros((work_h, args.work_width, 3), np.float64)
    for i in range(n):
        acc_stab += cv2.warpAffine(balls[i], np.linalg.inv(H[i])[:2],
                                   (args.work_width, work_h))
    cv2.imwrite(f"{args.out_dir}/stabilised_average.png", (acc_stab / n).astype(np.uint8))

    # per-arc parabola fit residual (raw vs stabilised)
    bounds = [t[0] - 1] + kink_times + [t[-1] + 1]
    arcs = []
    for k in range(len(bounds) - 1):
        lo, hi = bounds[k], bounds[k + 1]
        idx = [i for i in range(n) if lo + 0.05 < t[i] < hi - 0.05 and raw_ball[i]]
        if len(idx) < 5:
            continue
        tt = [t[i] for i in idx]
        raw_r = arc_fit_residual(tt, [raw_xy[i][1] for i in idx], [raw_xy[i][0] for i in idx])
        stab_r = arc_fit_residual(tt, [stab_xy[i][1] for i in idx], [stab_xy[i][0] for i in idx])
        if raw_r and stab_r:
            i0, i1 = idx[0], idx[-1]
            cam = float(np.hypot(ref_center[i1][0] - ref_center[i0][0],
                                 ref_center[i1][1] - ref_center[i0][1]))
            arcs.append({"t_lo": round(lo, 2), "t_hi": round(hi, 2), "n": len(idx),
                         "camera_motion_px": round(cam, 1),
                         "raw_rmse_px": round(raw_r[0], 2),
                         "stab_rmse_px": round(stab_r[0], 2)})

    # plot: raw vs stabilised tracks overlaid, parabola fits, camera motion
    fig, ax = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    for row, axis in enumerate(("y", "x")):  # row 0 = y(t), row 1 = x(t)
        comp = 1 if axis == "y" else 0
        tv = [t[i] for i in range(n) if raw_ball[i]]
        rv = [raw_xy[i][comp] for i in range(n) if raw_ball[i]]
        sv = [stab_xy[i][comp] for i in range(n) if raw_ball[i]]
        ax[row].plot(tv, rv, ".", ms=5, color="0.7", label="raw (camera motion mixed in)")
        ax[row].plot(tv, sv, ".", ms=5, color="tab:red", label="stabilised (ego-motion removed)")
        for a in arcs:
            idx = [i for i in range(n) if a["t_lo"] + 0.05 < t[i] < a["t_hi"] - 0.05 and raw_ball[i]]
            tt = np.array([t[i] for i in idx])
            fit = arc_fit_residual(tt, [stab_xy[i][1] for i in idx], [stab_xy[i][0] for i in idx])
            if fit:
                ts = np.linspace(tt.min(), tt.max(), 60)
                coef = fit[1] if axis == "y" else fit[2]
                ax[row].plot(ts, np.polyval(coef, ts), "-", color="green", lw=2, alpha=0.8)
                if axis == "y":
                    ax[row].annotate(f"raw {a['raw_rmse_px']:.0f}px\nstab {a['stab_rmse_px']:.0f}px",
                                     (tt.mean(), np.polyval(coef, tt.mean())),
                                     fontsize=7, ha="center", color="green")
        for tch in touches:
            if tv and tv[0] <= tch <= tv[-1]:
                ax[row].axvline(tch, color="gray", ls=":", lw=1)
        ax[row].set_ylabel(f"ball {axis} (px)")
        ax[row].legend(fontsize=8, loc="upper right")
    ax[0].set_title("ball y(t) — green = ballistic parabola fit per inter-touch arc "
                    "(dotted line = touch)")
    ax[0].invert_yaxis()
    ax[1].set_title("ball x(t)")
    cam_cum = [float(np.hypot(ref_center[i][0] - ref_center[0][0],
                              ref_center[i][1] - ref_center[0][1])) for i in range(n)]
    ax[2].plot(t, cam_cum, "-", color="tab:blue")
    ax[2].fill_between(t, cam_cum, color="tab:blue", alpha=0.15)
    ax[2].set_title("camera ego-motion: reference-frame displacement from start (px)")
    ax[2].set_ylabel("camera shift (px)")
    ax[2].set_xlabel("time (s)")
    fig.suptitle("L0 ego-motion prototype — video-506  |  "
                 "clean ballistic arcs fit a parabola; camera motion is real "
                 "but short-arc fits absorb it", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/ego_motion_tracks.png", dpi=110)

    metrics = {
        "video": os.path.basename(args.video),
        "frames": [frames[0], frames[-1]],
        "fps": round(fps, 3),
        "ball_detections": sum(1 for b in raw_ball if b),
        "ball_detection_rate": round(sum(1 for b in raw_ball if b) / n, 3),
        "median_motion_inliers": int(np.median(inlier_counts)) if inlier_counts else 0,
        "camera_path_total_px": round(cam_path_total, 1),
        "arcs": arcs,
    }
    json.dump(metrics, open(f"{args.out_dir}/prototype_metrics.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))
    if arcs:
        raw_mean = np.mean([a["raw_rmse_px"] for a in arcs])
        stab_mean = np.mean([a["stab_rmse_px"] for a in arcs])
        print(f"\nMean ballistic-arc fit residual:  raw {raw_mean:.2f} px"
              f"  ->  stabilised {stab_mean:.2f} px")


if __name__ == "__main__":
    main()
