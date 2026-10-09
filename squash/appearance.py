"""Stage 2b: appearance descriptors for every detection and a learned 2-player separation axis.

Descriptor: HSV histograms of the torso+shorts region (shoulders to knees, central 60% of the box).
The separating direction is the first principal component of the descriptor differences between the
two detections in frames that contain exactly two players (sign-free), so no colour thresholds are needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .calibrate import load_config

NB_H, NB_S, NB_V = 12, 8, 8


def descriptor(hsv, r):
    x0, y0, x1, y1 = r.x0, r.y0, r.x1, r.y1
    sho = np.array([[r.lsho_x, r.lsho_y, r.lsho_c], [r.rsho_x, r.rsho_y, r.rsho_c]])
    kne = np.array([[r.lknee_x, r.lknee_y, r.lknee_c], [r.rknee_x, r.rknee_y, r.rknee_c]])
    hip = np.array([[r.lhip_x, r.lhip_y, r.lhip_c], [r.rhip_x, r.rhip_y, r.rhip_c]])
    top = sho[sho[:, 2] > 0.3, 1].mean() if (sho[:, 2] > 0.3).any() else y0 + 0.2 * (y1 - y0)
    bot = kne[kne[:, 2] > 0.3, 1].mean() if (kne[:, 2] > 0.3).any() else y0 + 0.7 * (y1 - y0)
    mid = hip[hip[:, 2] > 0.3, 1].mean() if (hip[:, 2] > 0.3).any() else (top + bot) / 2
    cx = (x0 + x1) / 2
    hw = 0.3 * (x1 - x0)
    H, W = hsv.shape[:2]
    xa, xb = int(max(0, cx - hw)), int(min(W, cx + hw))
    ya, yb = int(max(0, top)), int(min(H, bot))
    if xb - xa < 3 or yb - ya < 6:
        return None
    crop = hsv[ya:yb, xa:xb].reshape(-1, 3).astype(np.float32)
    h, s, v = crop[:, 0], crop[:, 1], crop[:, 2]
    hh = np.histogram(h, bins=NB_H, range=(0, 180), weights=s / 255)[0]
    hs = np.histogram(s, bins=NB_S, range=(0, 256))[0]
    hv = np.histogram(v, bins=NB_V, range=(0, 256))[0]
    d = np.concatenate([hh / max(hh.sum(), 1e-6), hs / len(s), hv / len(v)])
    ym = int(np.clip(mid, ya + 1, yb - 1))
    shirt = hsv[ya:ym, xa:xb].reshape(-1, 3).astype(np.float32)
    shorts = hsv[ym:yb, xa:xb].reshape(-1, 3).astype(np.float32)
    extra = [
        shirt[:, 2].mean() if len(shirt) else np.nan,
        shorts[:, 1].mean() if len(shorts) else np.nan,
        shorts[:, 2].mean() if len(shorts) else np.nan,
    ]
    return np.concatenate([d, extra])


def run(cfg=None, force=False):
    cfg = cfg or load_config()
    out_dir = Path(cfg["output_dir"])
    out = out_dir / "appearance.parquet"
    if out.exists() and not force:
        print("appearance exists, skipping")
        return pd.read_parquet(out)
    det = pd.read_parquet(out_dir / "detections.parquet").reset_index().rename(columns={"index": "det_id"})
    cap = cv2.VideoCapture(cfg["video"])
    by_frame = {f: g for f, g in det.groupby("frame")}
    frames = sorted(by_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    fi = frames[0]
    rows = []
    while fi <= frames[-1]:
        ok = cap.grab()
        if not ok:
            break
        if fi in by_frame:
            ok, frame = cap.retrieve()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            for r in by_frame[fi].itertuples():
                d = descriptor(hsv, r)
                if d is not None:
                    rows.append((r.det_id, *d))
        fi += 1
    cols = ["det_id"] + [f"d{i}" for i in range(NB_H + NB_S + NB_V)] + ["shirt_v", "shorts_s", "shorts_v"]
    app = pd.DataFrame(rows, columns=cols)
    # learn the separating axis from frames with exactly two detections
    D = app.set_index("det_id")
    dcols = [f"d{i}" for i in range(NB_H + NB_S + NB_V)]
    diffs = []
    for f, g in det.groupby("frame"):
        if len(g) == 2 and g.det_id.isin(D.index).all():
            a, b = D.loc[g.det_id.iloc[0], dcols].to_numpy(float), D.loc[g.det_id.iloc[1], dcols].to_numpy(float)
            diffs.append(a - b)
    diffs = np.array(diffs)
    _, _, vt = np.linalg.svd(diffs - 0, full_matrices=False)
    axis = vt[0]
    X = D[dcols].to_numpy(float)
    score = X @ axis
    # orient the axis so that positive = higher shorts saturation (an arbitrary but stable sign)
    if np.corrcoef(score, D["shorts_s"].fillna(D["shorts_s"].median()))[0, 1] < 0:
        axis, score = -axis, -score
    # normalise: two-means on the score
    c = np.percentile(score, [25, 75])
    for _ in range(20):
        lab = score > c.mean()
        c = np.array([score[~lab].mean(), score[lab].mean()])
    thr = c.mean()
    app["score"] = (score - thr) / (c[1] - c[0]) * 2  # ~ -1 .. +1 between the two clusters
    app.to_parquet(out, index=False)
    # separation quality in two-detection frames
    S = app.set_index("det_id")["score"]
    gaps = []
    for f, g in det.groupby("frame"):
        if len(g) == 2 and g.det_id.isin(S.index).all():
            gaps.append(abs(S.loc[g.det_id.iloc[0]] - S.loc[g.det_id.iloc[1]]))
    gaps = np.array(gaps)
    print(
        f"descriptors: {len(app)}; two-player frames: {len(gaps)}; score gap quantiles (5/25/50%): {np.percentile(gaps, [5, 25, 50]).round(2).tolist()}"
    )
    print(f"share of two-player frames with opposite signs: {np.mean([1 for _ in gaps]) if False else ''}")
    opp = 0
    for f, g in det.groupby("frame"):
        if len(g) == 2 and g.det_id.isin(S.index).all():
            opp += (S.loc[g.det_id.iloc[0]] * S.loc[g.det_id.iloc[1]]) < 0
    print(f"two-player frames with opposite-sign scores: {opp}/{len(gaps)} ({opp / len(gaps) * 100:.1f}%)")
    print("score hist:", np.histogram(app.score, bins=np.arange(-3, 3.1, 0.5))[0].tolist())
    return app


if __name__ == "__main__":
    run(force="--force" in sys.argv)


# ---------------------------------------------------------------------------------------------
# CNN embedding (ResNet18, ImageNet weights) of the full person crop -> learned separation axis
# ---------------------------------------------------------------------------------------------
def cnn_scores(cfg=None, force=False):
    import torch
    import torchvision

    cfg = cfg or load_config()
    out_dir = Path(cfg["output_dir"])
    out = out_dir / "appearance_cnn.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    det = pd.read_parquet(out_dir / "detections.parquet").reset_index().rename(columns={"index": "det_id"})
    torch.set_num_threads(8)
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()
    m.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    cap = cv2.VideoCapture(cfg["video"])
    by_frame = {f: g for f, g in det.groupby("frame")}
    frames = sorted(by_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    fi = frames[0]
    ids, feats, batch, bids = [], [], [], []

    def flush():
        if not batch:
            return
        x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float() / 255
        x = (x - mean) / std
        with torch.no_grad():
            f = m(x).numpy()
        feats.extend(f)
        ids.extend(bids)
        batch.clear()
        bids.clear()

    while fi <= frames[-1]:
        if not cap.grab():
            break
        if fi in by_frame:
            _, frame = cap.retrieve()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for r in by_frame[fi].itertuples():
                x0, y0, x1, y1 = int(max(0, r.x0)), int(max(0, r.y0)), int(min(rgb.shape[1], r.x1)), int(min(rgb.shape[0], r.y1))
                if x1 - x0 < 8 or y1 - y0 < 16:
                    continue
                crop = cv2.resize(rgb[y0:y1, x0:x1], (96, 192))
                batch.append(crop)
                bids.append(r.det_id)
                if len(batch) >= 64:
                    flush()
        fi += 1
    flush()
    F = np.array(feats)
    F = F / np.linalg.norm(F, axis=1, keepdims=True)
    np.save(out_dir / "cnn_features.npy", F)
    np.save(out_dir / "cnn_det_ids.npy", np.array(ids))
    D = pd.DataFrame(F, index=ids)
    diffs = []
    for f, g in det.groupby("frame"):
        if len(g) == 2 and g.det_id.isin(D.index).all():
            diffs.append(D.loc[g.det_id.iloc[0]].to_numpy() - D.loc[g.det_id.iloc[1]].to_numpy())
    diffs = np.array(diffs)
    _, sv, vt = np.linalg.svd(diffs, full_matrices=False)
    axis = vt[0]
    score = F @ axis
    c = np.percentile(score, [25, 75])
    for _ in range(30):
        lab = score > c.mean()
        c = np.array([score[~lab].mean(), score[lab].mean()])
    score = (score - c.mean()) / (c[1] - c[0]) * 2
    app = pd.DataFrame(dict(det_id=ids, score_cnn=score))
    app.to_parquet(out, index=False)
    S = app.set_index("det_id").score_cnn
    opp = tot = 0
    gaps = []
    for f, g in det.groupby("frame"):
        if len(g) == 2 and g.det_id.isin(S.index).all():
            a, b = S.loc[g.det_id.iloc[0]], S.loc[g.det_id.iloc[1]]
            tot += 1
            opp += (a * b) < 0
            gaps.append(abs(a - b))
    print(
        f"cnn: {len(app)} embeddings; explained var of axis {sv[0] ** 2 / np.sum(sv**2):.2f}; "
        f"two-player frames opposite-sign {opp}/{tot} ({opp / tot * 100:.1f}%); gap quantiles 5/25/50%: {np.percentile(gaps, [5, 25, 50]).round(2).tolist()}"
    )
    print("score hist:", np.histogram(score, bins=np.arange(-3, 3.1, 0.5))[0].tolist())
    return app


if __name__ == "__main__" and "--cnn" in sys.argv:
    cnn_scores(force=True)
