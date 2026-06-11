#!/usr/bin/env python3
"""Round 1 / T1.3-T1.4 — WASB zero-shot feasibility on footbag video.

Loads the WASB tennis-pretrained HRNet checkpoint and runs it, with no
fine-tuning, on a footbag clip. Bypasses WASB's CUDA-only `TracknetV2Detector`
wrapper (this Mac has no NVIDIA GPU) and does preprocessing + peak extraction
directly, so it runs on CPU/MPS.

The bet (DETECTOR_REPLACEMENT_ROADMAP.md, R1): WASB's tennis weights transfer to
a footbag well enough to be a candidate miner. This script measures whether the
heatmap peak lands on the ball.

Prereqs: external/WASB-SBDT cloned; pretrained_weights/wasb_tennis_best.pth.tar
downloaded (see roadmap T1.2-T1.3). Run: python3 wasb_feasibility.py
"""
import os
import sys

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

WASB_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "external", "WASB-SBDT", "src")
sys.path.insert(0, WASB_SRC)
from models import build_model  # noqa: E402

from prototype_ego_motion import detect_ball  # noqa: E402  HSV reference

# WASB model config (external/WASB-SBDT/src/configs/model/wasb.yaml)
WASB_CFG = {
    "name": "hrnet", "frames_in": 3, "frames_out": 3,
    "inp_height": 288, "inp_width": 512, "out_height": 288, "out_width": 512,
    "rgb_diff": False, "out_scales": [0],
    "MODEL": {"EXTRA": {
        "FINAL_CONV_KERNEL": 1, "PRETRAINED_LAYERS": ["*"],
        "STEM": {"INPLANES": 64, "STRIDES": [1, 1]},
        "STAGE1": {"NUM_MODULES": 1, "NUM_BRANCHES": 1, "BLOCK": "BOTTLENECK",
                   "NUM_BLOCKS": [1], "NUM_CHANNELS": [32], "FUSE_METHOD": "SUM"},
        "STAGE2": {"NUM_MODULES": 1, "NUM_BRANCHES": 2, "BLOCK": "BASIC",
                   "NUM_BLOCKS": [2, 2], "NUM_CHANNELS": [16, 32], "FUSE_METHOD": "SUM"},
        "STAGE3": {"NUM_MODULES": 1, "NUM_BRANCHES": 3, "BLOCK": "BASIC",
                   "NUM_BLOCKS": [2, 2, 2], "NUM_CHANNELS": [16, 32, 64], "FUSE_METHOD": "SUM"},
        "STAGE4": {"NUM_MODULES": 1, "NUM_BRANCHES": 4, "BLOCK": "BASIC",
                   "NUM_BLOCKS": [2, 2, 2, 2], "NUM_CHANNELS": [16, 32, 64, 128],
                   "FUSE_METHOD": "SUM"},
        "DECONV": {"NUM_DECONVS": 0, "KERNEL_SIZE": [], "NUM_BASIC_BLOCKS": 2},
    }, "INIT_WEIGHTS": True},
}
INP_W, INP_H = 512, 288


def letterbox(frame_bgr):
    """Resize keeping aspect into INP_W x INP_H, black-padded (WASB convention).
    Returns the letterboxed RGB frame, the scale, and the (x,y) paste offset."""
    h, w = frame_bgr.shape[:2]
    if INP_H / INP_W >= h / w:
        nw, nh = INP_W, int(INP_W * h / w)
    else:
        nw, nh = int(INP_H * w / h), INP_H
    resized = cv2.resize(frame_bgr, (nw, nh))
    canvas = np.zeros((INP_H, INP_W, 3), np.uint8)
    ox, oy = (INP_W - nw) // 2, (INP_H - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return rgb, nw / w, (ox, oy)


def unletterbox(px, py, scale, offset):
    ox, oy = offset
    return (px - ox) / scale, (py - oy) / scale


def main():
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    ckpt = "pretrained_weights/wasb_tennis_best.pth.tar"
    video = "/Users/isaacaudet/Downloads/video-506_singular_display.MOV"

    model = build_model(OmegaConf.create({"model": WASB_CFG}))
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"WASB HRNet loaded on {device}  "
          f"(missing={len(missing)}, unexpected={len(unexpected)} keys)")
    model = model.to(device)
    model.train(False)                                  # inference mode

    cap = cv2.VideoCapture(video)
    f0, f1 = 34, 160                                    # the video-506 rally
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    frames = []
    while len(frames) < (f1 - f0):
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()

    # HSV reference: ball position per frame (region proposal + GT stand-in).
    def hsv_ball(fr):
        s = 731.0 / fr.shape[1]
        b = detect_ball(cv2.resize(fr, (731, int(fr.shape[0] * s))))
        return (b[0] / s, b[1] / s) if b else None
    refs = [hsv_ball(fr) for fr in frames]

    # WASB's own test transform (dataloaders/build_img_transforms) is
    # ToTensor + Normalize(ImageNet mean/std) — applied for BOTH train and test.
    # R1's harness omitted the Normalize, feeding the net un-normalized [0,1]
    # input it was never trained on. This run measures with and without it.
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Fair test: a 16:9 ROI crop around the ball (== WASB input aspect), so the
    # footbag is ~30 px rather than ~6 px after the full-portrait letterbox.
    CW, CH = 1024, 576

    def score(normalize, save_montage=False):
        """Run WASB over the rally; one row per frame. If `normalize`, apply the
        ImageNet mean/std from WASB's own test transform (build_img_transforms)."""
        rows, overlays = [], []
        for i in range(2, len(frames)):
            c = refs[i]
            if c is None:
                continue
            H, W = frames[i].shape[:2]
            ox = int(np.clip(c[0] - CW / 2, 0, max(0, W - CW)))
            oy = int(np.clip(c[1] - CH / 2, 0, max(0, H - CH)))
            chans = []
            for fr in frames[i - 2:i + 1]:
                crop = fr[oy:oy + CH, ox:ox + CW]
                rgb = cv2.cvtColor(cv2.resize(crop, (INP_W, INP_H)), cv2.COLOR_BGR2RGB)
                t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                if normalize:
                    t = (t - imagenet_mean) / imagenet_std
                chans.append(t)
            inp = torch.cat(chans, dim=0).unsqueeze(0).to(device)   # (1,9,288,512)
            with torch.no_grad():
                out = model(inp)
            hm = out[0] if isinstance(out, (list, tuple)) else (
                out[next(iter(out))] if isinstance(out, dict) else out)
            hm = torch.sigmoid(hm)[0, -1].cpu().numpy()             # last frame's heatmap
            if hm.shape != (INP_H, INP_W):
                hm = cv2.resize(hm, (INP_W, INP_H))
            peak = float(hm.max())
            py, px = np.unravel_index(int(hm.argmax()), hm.shape)
            fx = ox + px * (CW / INP_W)
            fy = oy + py * (CH / INP_H)
            err = float(np.hypot(fx - c[0], fy - c[1]))
            rows.append({"frame": f0 + i, "peak_conf": peak, "err_px": err})
            if save_montage and i % 14 == 2 and len(overlays) < 9:
                vis = cv2.resize(frames[i], (366, int(366 * H / W)))
                sx = 366 / W
                cv2.circle(vis, (int(fx * sx), int(fy * sx)), 12, (0, 0, 255), 3)
                cv2.circle(vis, (int(c[0] * sx), int(c[1] * sx)), 8, (0, 255, 0), 2)
                cv2.putText(vis, f"f{f0+i} c={peak:.2f}", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                overlays.append(vis)
        if save_montage and overlays:
            hh = min(o.shape[0] for o in overlays)
            overlays = [cv2.resize(o, (int(o.shape[1] * hh / o.shape[0]), hh))
                        for o in overlays]
            rowimgs = [np.hstack(overlays[k:k + 3]) for k in range(0, len(overlays), 3)
                       if len(overlays[k:k + 3]) == 3]
            if rowimgs:
                os.makedirs("tmp_proto", exist_ok=True)
                cv2.imwrite("tmp_proto/wasb_feasibility.png", np.vstack(rowimgs))
        return rows

    def report(label, rows):
        confs = np.array([r["peak_conf"] for r in rows])
        errs = np.array([r["err_px"] for r in rows])
        print(f"\n[{label}]  frames scored: {len(rows)}")
        print(f"  peak confidence: mean={confs.mean():.3f} max={confs.max():.3f} "
              f"min={confs.min():.3f}")
        if len(errs):
            print(f"  peak vs HSV ball error (px): median={np.median(errs):.0f} "
                  f"p90={np.percentile(errs, 90):.0f}")
            for thr in (40, 80, 150):
                print(f"    within {thr}px of ball: {(errs < thr).mean():.2f}")

    rows_raw = score(normalize=False)
    rows_norm = score(normalize=True, save_montage=True)
    report("raw /255  (R1 original — INCORRECT preprocessing)", rows_raw)
    report("ImageNet-normalized  (matches WASB build_img_transforms)", rows_norm)
    print("\nmontage (normalized run): tmp_proto/wasb_feasibility.png")


if __name__ == "__main__":
    main()
