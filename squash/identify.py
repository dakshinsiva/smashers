"""Stage 3: player identity, floor projection and smoothing -> positions.csv"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

from . import court
from .calibrate import img_to_court, load_config

PLAYERS = ("p1", "p2")
MARGIN = 0.45  # metres outside the court still accepted (then clamped)
MAX_GAP_S = 0.6  # interpolate gaps up to this long
MAX_SPEED = 8.0  # m/s, above this a sample is a glitch
ANKLE_CONF = 0.5
SG_WIN = 11  # Savitzky-Golay window in sampled frames (~1.1 s at 10 fps)


def foot_points(df, img_h, y_horizon=385.0, cam_h=0.6, ankle_h=0.08):
    """Ground-contact point per detection.
    Depth comes from the bounding-box bottom (the shoe sole is on the floor); the ankle keypoint sits
    ~8 cm above the floor, which from this floor-level camera is worth half a metre of depth near the T.
    Lateral position comes from the ankles when they are confident. When the feet are cut off by the
    frame edge, the ankle row is projected down to the floor using the camera height."""
    la = df[["lank_x", "lank_y", "lank_c"]].to_numpy()
    ra = df[["rank_x", "rank_y", "rank_c"]].to_numpy()
    lv, rv = la[:, 2] > ANKLE_CONF, ra[:, 2] > ANKLE_CONF
    ankle_x = np.where(lv & rv, (la[:, 0] + ra[:, 0]) / 2, np.where(lv, la[:, 0], np.where(rv, ra[:, 0], (df.x0 + df.x1) / 2)))
    ankle_y = np.where(lv & rv, (la[:, 1] + ra[:, 1]) / 2, np.where(lv, la[:, 1], np.where(rv, ra[:, 1], df.y1)))
    ankle_vis = lv | rv
    feet_cut = df.y1.to_numpy() >= img_h - 3
    y_floor_from_ankle = ankle_y + (ankle_y - y_horizon) * ankle_h / (cam_h - ankle_h)
    fx = ankle_x
    fy = np.where(feet_cut, y_floor_from_ankle, df.y1.to_numpy())
    return fx, fy, ankle_vis, feet_cut


def build_tracklets(det, gate_base=0.45, gate_per_frame=0.35, max_miss=3, cross_dist=0.8):
    """Link detections across sampled frames by court position into tracklets.
    Tracklets are cut whenever two players come within `cross_dist` m of each other, so identity
    is re-decided by appearance after every possible crossing."""
    det = det.sort_values(["frame", "conf"], ascending=[True, False]).reset_index(drop=True)
    det["tracklet"] = -1
    active = []  # dicts: id, last_frame, x, y
    next_id = 0
    xs, ys = det.x.to_numpy(), det.y.to_numpy()
    idx_by_frame = det.groupby("frame").indices
    for f in sorted(idx_by_frame):
        idx = idx_by_frame[f]
        # close stale tracks
        active = [a for a in active if f - a["last_frame"] <= max_miss * 3]  # frames are sampled every 3
        if active and len(idx):
            cost = np.zeros((len(active), len(idx)))
            for i, a in enumerate(active):
                gap = max(1, (f - a["last_frame"]) // 3)
                d = np.hypot(xs[idx] - a["x"], ys[idx] - a["y"])
                gate = gate_base + gate_per_frame * gap
                cost[i] = np.where(d < gate, d, 1e6)
            r, c = linear_sum_assignment(cost)
            used_a, used_d = set(), set()
            for i, j in zip(r, c):
                if cost[i, j] < 1e5:
                    a = active[i]
                    det.at[idx[j], "tracklet"] = a["id"]
                    a.update(last_frame=f, x=xs[idx[j]], y=ys[idx[j]])
                    used_a.add(i)
                    used_d.add(j)
        else:
            used_a, used_d = set(), set()
        for j in range(len(idx)):
            if j not in used_d:
                det.at[idx[j], "tracklet"] = next_id
                active.append(dict(id=next_id, last_frame=f, x=xs[idx[j]], y=ys[idx[j]]))
                next_id += 1
        # crossing guard: cut tracklets that are close to each other in this frame
        cur = [a for a in active if a["last_frame"] == f]
        for i in range(len(cur)):
            for j in range(i + 1, len(cur)):
                if np.hypot(cur[i]["x"] - cur[j]["x"], cur[i]["y"] - cur[j]["y"]) < cross_dist:
                    for a in (cur[i], cur[j]):
                        a["last_frame"] = -(10**9)  # force closure; next frame starts new tracklets
    return det


def split_on_flips(det, score_col="score", win=5, margin=0.5):
    """Cut a tracklet where the rolling appearance score changes sign decisively (missed crossing)."""
    det = det.sort_values(["tracklet", "frame"]).copy()
    next_id = det.tracklet.max() + 1
    new_ids = det.tracklet.to_numpy().copy()
    n_split = 0
    for tid, g in det.groupby("tracklet"):
        if len(g) < 2 * win:
            continue
        s = g[score_col].rolling(win, center=True, min_periods=2).mean().to_numpy()
        idx = g.index.to_numpy()
        cur = tid
        for k in range(win, len(s) - win):
            if s[k - 1] > margin and s[k] < -margin or s[k - 1] < -margin and s[k] > margin:
                cur = next_id
                next_id += 1
                n_split += 1
            new_ids[det.index.get_loc(idx[k])] = cur
    det["tracklet"] = new_ids
    print(f"tracklets split on appearance flips: {n_split}")
    return det


def assign_identities(det, fps_eff, score_col="score"):
    """Label tracklets by mean appearance score, resolving overlaps in time."""
    det = build_tracklets(det)
    det = split_on_flips(det, score_col)
    tl = det.groupby("tracklet").agg(
        f0=("frame", "min"), f1=("frame", "max"), n=("frame", "size"), s=(score_col, lambda v: np.nanmean(v)), conf=("conf", "mean")
    )
    tl["label"] = np.where(tl.s > 0, "p1", "p2")
    tl["w"] = np.abs(tl.s.fillna(0)) * np.sqrt(tl.n)
    tl["flipped"] = False
    det["player"] = det.tracklet.map(tl.label)
    det["tw"] = det.tracklet.map(tl.w)
    # frame-level resolution: if two detections share a label, the one from the stronger tracklet keeps it;
    # the other takes the free label when its own score does not contradict it, else it is dropped.
    n_reassigned = n_dropped = 0
    for f, g in det.groupby("frame"):
        if len(g) < 2:
            continue
        for lab in PLAYERS:
            same = g[g.player == lab].sort_values("tw", ascending=False)
            if len(same) < 2:
                continue
            other = "p2" if lab == "p1" else "p1"
            other_taken = (g.player == other).any()
            for k, (i, r) in enumerate(same.iterrows()):
                if k == 0:
                    continue
                s = r[score_col]
                if not other_taken and ((other == "p1" and s > -0.8) or (other == "p2" and s < 0.8)):
                    det.at[i, "player"] = other
                    other_taken = True
                    n_reassigned += 1
                else:
                    det.at[i, "player"] = ""
                    n_dropped += 1
    print(f"frame-level conflicts: {n_reassigned} reassigned, {n_dropped} dropped")
    swaps = [
        dict(tracklet=int(i), frames=[int(r.f0), int(r.f1)], n=int(r.n), mean_score=float(r.s), label=r.label)
        for i, r in tl.iterrows()
        if r.flipped or abs(r.s) < 0.25
    ]
    stats = dict(
        n_tracklets=int(len(tl)),
        frames_weighted_abs_score=float((np.abs(tl.s) * tl.n).sum() / tl.n.sum()),
        dropped_detections=int((det.player == "").sum()),
        ambiguous_frames=int(tl.n[np.abs(tl.s) < 0.25].sum()),
    )
    print("tracklets:", stats)
    return det, swaps


def smooth_track(g, fps_eff):
    g = g.sort_values("t").drop_duplicates("frame")
    # resample onto the regular sampled-frame grid to make gaps explicit
    frames = np.arange(g.frame.min(), g.frame.max() + 1, g.frame.diff().mode().iloc[0])
    s = g.set_index("frame").reindex(frames)
    s["t"] = s["t"].interpolate()
    s["interpolated"] = s["x"].isna()
    max_gap = int(MAX_GAP_S * fps_eff)
    for col in ("x", "y", "conf"):
        s[col] = s[col].interpolate(limit=max_gap, limit_area="inside")
    for col in ("ankle_vis", "feet_cut"):
        s[col] = s[col].astype(float).interpolate(limit=max_gap, limit_area="inside") > 0.5
    valid = s["x"].notna()
    # smooth each contiguous valid run
    xs, ys = s["x"].to_numpy().copy(), s["y"].to_numpy().copy()
    run_id = (valid != valid.shift()).cumsum()
    for _, run in s[valid].groupby(run_id[valid]):
        i = s.index.get_indexer(run.index)
        if len(i) >= SG_WIN:
            xs[i] = savgol_filter(xs[i], SG_WIN, 2)
            ys[i] = savgol_filter(ys[i], SG_WIN, 2)
    s["x"], s["y"] = np.clip(xs, 0, court.WIDTH), np.clip(ys, 0, court.LENGTH)
    dt = np.gradient(s["t"].to_numpy())
    vx, vy = np.gradient(s["x"].to_numpy()) / dt, np.gradient(s["y"].to_numpy()) / dt
    sp = np.hypot(vx, vy)
    sp[~valid.to_numpy()] = np.nan
    sp = np.where(sp > MAX_SPEED, np.nan, sp)
    s["speed"] = pd.Series(sp, index=s.index).interpolate(limit=2, limit_area="inside")
    s["dist_t"] = court.dist_to_t(s["x"], s["y"])
    s["frame"] = s.index
    return s.reset_index(drop=True)


def run(cfg=None, force=False):
    cfg = cfg or load_config()
    out_dir = Path(cfg["output_dir"])
    out = out_dir / "positions.csv"
    if out.exists() and not force:
        print("positions exist, skipping")
        return pd.read_csv(out)
    calib = json.load(open(out_dir / "calibration.json"))  # dict: H + optional lens model
    img_w, img_h = calib["image_size"]
    det = pd.read_parquet(out_dir / "detections.parquet")
    # effective sample rate from the detections themselves (t = frame / fps), so this stage never needs the video
    fps = float(np.median(det.frame[det.frame > 0] / det.t[det.frame > 0]))
    fps_eff = fps / cfg["frame_stride"]

    fx, fy, ankle_vis, feet_cut = foot_points(det, img_h)
    xy = img_to_court(calib, np.stack([fx, fy], 1))
    det["fx"], det["fy"] = fx, fy
    det["x"], det["y"] = xy[:, 0], xy[:, 1]
    det["ankle_vis"], det["feet_cut"] = ankle_vis, feet_cut
    det = det.reset_index().rename(columns={"index": "det_id"})
    probs = pd.read_parquet(out_dir / "identity_probs.parquet").set_index("det_id")
    det = det.join(probs, on="det_id")
    if "prob_other" in det:  # 3-class model: drop spectators / non-players first
        n_other = int((det.prob_other > 0.5).sum())
        det = det[~(det.prob_other > 0.5)].copy()
        print(f"dropped {n_other} detections classified as non-players")
        det["score"] = det["prob_p1"] - det["prob_p2"]  # -1 = p2 .. +1 = p1
    else:
        det["score"] = 2 * det["prob_p1"] - 1  # -1 = p2 .. +1 = p1
    n0 = len(det)
    inside = (det.x > -MARGIN) & (det.x < court.WIDTH + MARGIN) & (det.y > -MARGIN) & (det.y < court.LENGTH + 0.8)
    det = det[inside & (det.t >= cfg["t_start"]) & (det.t <= cfg["t_end"])].copy()
    print(f"detections: {n0} -> {len(det)} inside the court window")

    det, swaps = assign_identities(det, fps_eff)
    det = det[det.player != ""]
    print("assigned:", det.player.value_counts().to_dict(), " colour/continuity disagreements:", len(swaps))
    json.dump(swaps, open(out_dir / "identity_swaps.json", "w"), indent=1)

    depth_factor = np.clip(1 - (det.y - 7.0) / 5.0, 0.55, 1.0)
    det["pos_conf"] = det.conf * np.where(det.feet_cut, 0.5, 1.0) * np.where(det.ankle_vis, 1.0, 0.8) * depth_factor
    det["x_raw"], det["y_raw"] = det.x, det.y
    tracks = []
    for p in PLAYERS:
        g = det[det.player == p][["frame", "t", "x", "y", "pos_conf", "ankle_vis", "feet_cut", "fx", "fy", "x_raw", "y_raw"]].rename(
            columns={"pos_conf": "conf"}
        )
        s = smooth_track(g, fps_eff)
        s["player"] = p
        tracks.append(s)
    pos = pd.concat(tracks, ignore_index=True)
    pos = pos[
        ["player", "frame", "t", "x", "y", "speed", "dist_t", "conf", "interpolated", "ankle_vis", "feet_cut", "fx", "fy", "x_raw", "y_raw"]
    ]
    pos.to_csv(out, index=False, float_format="%.4f")
    for p in PLAYERS:
        g = pos[pos.player == p]
        print(
            f"{p}: {g.x.notna().mean() * 100:.1f}% of sampled frames tracked, median conf {g.conf.median():.2f}, "
            f"feet_cut {g.feet_cut.mean() * 100:.1f}%"
        )
    return pos


if __name__ == "__main__":
    run(force="--force" in sys.argv)
