"""Stage 2: person + pose detection on sampled frames (cached to parquet)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .calibrate import load_config

KP_NAMES = [
    "nose",
    "leye",
    "reye",
    "lear",
    "rear",
    "lsho",
    "rsho",
    "lelb",
    "relb",
    "lwri",
    "rwri",
    "lhip",
    "rhip",
    "lknee",
    "rknee",
    "lank",
    "rank",
]


def colour_descriptor(frame_hsv, box, kps):
    """Median HSV of the shorts region (hips to knees); a cheap colour cue stored with each detection."""
    x0, y0, x1, y1 = box
    hips = kps[[11, 12]]
    knees = kps[[13, 14]]
    if (hips[:, 2] > 0.3).all() and (knees[:, 2] > 0.3).any():
        yt = hips[:, 1].mean()
        yb = knees[knees[:, 2] > 0.3, 1].mean()
        xc = hips[:, 0].mean()
        hw = max(8, abs(hips[0, 0] - hips[1, 0]) * 0.9)
        xa, xb = xc - hw, xc + hw
    else:
        h = y1 - y0
        yt, yb = y0 + 0.45 * h, y0 + 0.68 * h
        xa, xb = x0 + 0.2 * (x1 - x0), x1 - 0.2 * (x1 - x0)
    H, W = frame_hsv.shape[:2]
    xa, xb = int(max(0, xa)), int(min(W, xb))
    yt, yb = int(max(0, yt)), int(min(H, yb))
    if xb - xa < 3 or yb - yt < 3:
        return dict(c_h=np.nan, c_s=np.nan, c_v=np.nan)
    crop = frame_hsv[yt:yb, xa:xb].reshape(-1, 3).astype(float)
    h, s, v = crop[:, 0], crop[:, 1], crop[:, 2]
    return dict(c_h=float(np.median(h)), c_s=float(np.median(s)), c_v=float(np.median(v)))


def run(cfg=None, force=False):
    cfg = cfg or load_config()
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "detections.parquet"
    if out.exists() and not force:
        print("detections exist, skipping")
        return pd.read_parquet(out)
    from ultralytics import YOLO

    model = YOLO(cfg["model"])
    cap = cv2.VideoCapture(cfg["video"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    w_img, h_img = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi_mask = None
    if cfg.get("roi_px"):
        roi_mask = np.zeros((h_img, w_img), np.uint8)
        cv2.fillPoly(roi_mask, [np.array(cfg["roi_px"], np.int32)], 255)
        print(
            f"ROI mask active: {len(cfg['roi_px'])}-point polygon, "
            f"{roi_mask.mean() / 255 * 100:.0f}% of the frame kept (bodies outside it are ignored)"
        )
    stride = int(cfg["frame_stride"])
    f0, f1 = int(cfg["t_start"] * fps), int(cfg["t_end"] * fps)
    rows = []
    t_wall = time.time()
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    fi = f0
    done = 0
    total = (f1 - f0) // stride
    while fi <= f1:
        ok = cap.grab()
        if not ok:
            break
        if (fi - f0) % stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            if roi_mask is not None:
                frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
            res = model.predict(frame, imgsz=cfg["imgsz"], conf=cfg["conf"], classes=[0], device="cpu", verbose=False)[0]
            hsv = None
            if res.boxes is not None and len(res.boxes):
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                kpts = res.keypoints.data.cpu().numpy() if res.keypoints is not None else np.zeros((len(boxes), 17, 3))
                for b, c, k in zip(boxes, confs, kpts):
                    row = dict(frame=fi, t=fi / fps, x0=b[0], y0=b[1], x1=b[2], y1=b[3], conf=float(c))
                    for j, name in enumerate(KP_NAMES):
                        row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_c"] = k[j]
                    row.update(colour_descriptor(hsv, b, k))
                    rows.append(row)
            done += 1
            if done % 200 == 0:
                el = time.time() - t_wall
                print(
                    f"{done}/{total} frames  t={fi / fps:.0f}s  {el / done:.3f}s/frame  ETA {(total - done) * el / done / 60:.1f} min",
                    flush=True,
                )
        fi += 1
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    print(f"saved {len(df)} detections from {done} frames in {(time.time() - t_wall) / 60:.1f} min")
    return df


if __name__ == "__main__":
    run(force="--force" in sys.argv)
