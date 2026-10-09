"""Keypoint-anchored colour patches (chest, thighs) for every detection -> output/patches.parquet"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .calibrate import load_config


def patch_stats(hsv, cx, cy, r):
    H, W = hsv.shape[:2]
    x0, x1 = int(max(0, cx - r)), int(min(W, cx + r + 1))
    y0, y1 = int(max(0, cy - r)), int(min(H, cy + r + 1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return [np.nan] * 3
    p = hsv[y0:y1, x0:x1].reshape(-1, 3).astype(float)
    return [np.median(p[:, 0]), np.median(p[:, 1]), np.median(p[:, 2])]


def run(cfg=None, force=False):
    cfg = cfg or load_config()
    out_dir = Path(cfg["output_dir"])
    out = out_dir / "patches.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    det = pd.read_parquet(out_dir / "detections.parquet").reset_index().rename(columns={"index": "det_id"})
    cap = cv2.VideoCapture(cfg["video"])
    by_frame = {f: g for f, g in det.groupby("frame")}
    frames = sorted(by_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    fi = frames[0]
    rows = []
    while fi <= frames[-1]:
        if not cap.grab():
            break
        if fi in by_frame:
            _, frame = cap.retrieve()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            for r in by_frame[fi].itertuples():
                h = r.y1 - r.y0
                rad = max(2, h * 0.04)
                sho = np.array([[r.lsho_x, r.lsho_y, r.lsho_c], [r.rsho_x, r.rsho_y, r.rsho_c]])
                hip = np.array([[r.lhip_x, r.lhip_y, r.lhip_c], [r.rhip_x, r.rhip_y, r.rhip_c]])
                kne = np.array([[r.lknee_x, r.lknee_y, r.lknee_c], [r.rknee_x, r.rknee_y, r.rknee_c]])
                vals = [r.det_id]
                ok_t = (sho[:, 2] > 0.3).all() and (hip[:, 2] > 0.3).all()
                if ok_t:
                    cx, cy = (sho[:, 0].mean() + hip[:, 0].mean()) / 2, sho[:, 1].mean() * 0.6 + hip[:, 1].mean() * 0.4
                    vals += patch_stats(hsv, cx, cy, rad)
                else:
                    vals += [np.nan] * 3
                for side in (0, 1):
                    if hip[side, 2] > 0.3 and kne[side, 2] > 0.3:
                        cx, cy = hip[side, 0] * 0.6 + kne[side, 0] * 0.4, hip[side, 1] * 0.6 + kne[side, 1] * 0.4
                        vals += patch_stats(hsv, cx, cy, rad)
                    else:
                        vals += [np.nan] * 3
                rows.append(vals)
        fi += 1
    cols = ["det_id", "chest_h", "chest_s", "chest_v", "lth_h", "lth_s", "lth_v", "rth_h", "rth_s", "rth_v"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_parquet(out, index=False)
    print(
        "patches:",
        len(df),
        "chest ok:",
        df.chest_v.notna().mean().round(3),
        "thigh ok:",
        (df.lth_v.notna() | df.rth_v.notna()).mean().round(3),
    )
    return df


if __name__ == "__main__":
    run(force="--force" in sys.argv)
