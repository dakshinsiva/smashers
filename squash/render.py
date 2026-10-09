"""Stage 7: annotated video (or stills) with a top-down mini-map.

python -m squash.render                 -> output/annotated.mp4 (whole analysed window)
python -m squash.render --stills 30,95  -> output/debug_frames/t0030.jpg ... (verification)
python -m squash.render --rallies-only  -> skip rest periods
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import court
from .calibrate import load_config

PLAYERS = ("p1", "p2")
BGR = {"p1": (214, 120, 42), "p2": (52, 104, 235)}  # same hues as the figures (BGR)
MAP_SCALE = 16  # px per metre in the mini-map
TRAIL_S = 2.0


def load(cfg):
    out_dir = Path(cfg["output_dir"])
    pos = pd.read_csv(out_dir / "positions.csv")
    det = pd.read_parquet(out_dir / "detections.parquet")
    rallies = pd.read_csv(out_dir / "rallies.csv") if (out_dir / "rallies.csv").exists() else None
    return pos, det, rallies


def interp_tracks(pos, frames_all):
    """Per player: x, y, fx, fy interpolated onto every frame (short gaps only)."""
    tr = {}
    for p in PLAYERS:
        g = pos[pos.player == p].set_index("frame").sort_index()
        s = g[["x", "y", "fx", "fy", "conf"]].reindex(frames_all)
        s = s.interpolate(limit=3, limit_area="inside")
        tr[p] = s
    return tr


def draw_minimap(frame, tr, fi, fps, rally_txt):
    W, Hh = int(court.WIDTH * MAP_SCALE) + 20, int(court.LENGTH * MAP_SCALE) + 20
    mm = np.full((Hh, W, 3), 30, np.uint8)
    court.draw_court_cv(mm, MAP_SCALE, (10, 10), (170, 170, 170), 1)
    for p in PLAYERS:
        s = tr[p]
        lo = max(s.index.min(), fi - int(TRAIL_S * fps))
        seg = s.loc[lo:fi].dropna(subset=["x"])
        pts = np.stack([10 + seg.x * MAP_SCALE, 10 + seg.y * MAP_SCALE], 1).astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(mm, [pts], False, BGR[p], 1, cv2.LINE_AA)
        if fi in s.index and not np.isnan(s.at[fi, "x"]):
            cv2.circle(mm, tuple(pts[-1]), 4, BGR[p], -1, cv2.LINE_AA)
    h, w = mm.shape[:2]
    x0, y0 = frame.shape[1] - w - 10, 10
    roi = frame[y0 : y0 + h, x0 : x0 + w]
    cv2.addWeighted(mm, 0.85, roi, 0.15, 0, roi)
    if rally_txt:
        fs = 0.45 * min(1.0, frame.shape[1] / 1024) + 0.1
        cv2.putText(frame, rally_txt, (x0, y0 + h + 16), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def score_at(rallies, t):
    """(score_p1, score_p2, note) as it stands at time t: the score after the last rally that has
    ended; during a rally the score it is being played at."""
    if rallies is None or not len(rallies):
        return 0, 0, ""
    done = rallies[rallies.t_end <= t]
    if not len(done):
        return 0, 0, ""
    last = done.iloc[-1]
    note = ""
    if last.name == rallies.index[-1] and not isinstance(last.winner, str) and not (isinstance(last.note, str) and "let" in last.note):
        note = "last point not decided on camera"
    return int(last.score_p1), int(last.score_p2), note


def draw_scoreboard(frame, t, rallies, labels):
    """Always-visible score panel, top-left."""
    st, sb, note = score_at(rallies, t)
    line1 = f"{labels['p1']} {st} - {sb} {labels['p2']}"
    line2 = ""
    if rallies is not None:
        cur = rallies[(rallies.t_start <= t) & (rallies.t_end >= t)]
        if len(cur):
            r = cur.iloc[0]
            srv = f" - {labels[r.server]} serving ({r.box} box)" if isinstance(r.server, str) else ""
            line2 = f"rally {int(r.rally)}{srv}"
        elif note:
            line2 = note
    W = frame.shape[1]
    s1 = 0.9 * min(1.0, W / 1024) + 0.15
    s2 = 0.5 * min(1.0, W / 1024) + 0.1
    (w1, h1), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_DUPLEX, s1, 2)
    (w2, h2), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, s2, 1) if line2 else ((0, 0), 0)
    pw, ph = max(w1, w2) + 24, h1 + (h2 + 10 if line2 else 0) + 22
    panel = frame[8 : 8 + ph, 8 : 8 + pw]
    cv2.addWeighted(np.full_like(panel, 20), 0.75, panel, 0.25, 0, panel)
    cv2.putText(frame, line1, (20, 8 + 12 + h1), cv2.FONT_HERSHEY_DUPLEX, s1, (255, 255, 255), 2, cv2.LINE_AA)
    if line2:
        cv2.putText(frame, line2, (20, 8 + 12 + h1 + 10 + h2), cv2.FONT_HERSHEY_SIMPLEX, s2, (220, 220, 220), 1, cv2.LINE_AA)
    return frame


def annotate(frame, fi, t, tr, det_by_frame, labels, rallies, fps):
    if fi in det_by_frame:
        for d in det_by_frame[fi].itertuples():
            cv2.rectangle(frame, (int(d.x0), int(d.y0)), (int(d.x1), int(d.y1)), (200, 200, 200), 1)
    for p in PLAYERS:
        s = tr[p]
        if fi in s.index and not np.isnan(s.at[fi, "fx"]):
            fx, fy = int(s.at[fi, "fx"]), int(s.at[fi, "fy"])
            cv2.circle(frame, (fx, fy), 6, BGR[p], 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"{labels[p]} ({s.at[fi, 'x']:.1f},{s.at[fi, 'y']:.1f})",
                (fx + 8, fy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5 * min(1.0, frame.shape[1] / 1024) + 0.1,
                BGR[p],
                2,
                cv2.LINE_AA,
            )
    draw_scoreboard(frame, t, rallies, labels)
    txt = f"t={t:6.1f}s"
    return draw_minimap(frame, tr, fi, fps, txt)


def run(cfg=None, stills=None, rallies_only=False):
    cfg = cfg or load_config()
    out_dir = Path(cfg["output_dir"])
    labels = {p: cfg["players"][p]["label"] for p in PLAYERS}
    pos, det, rallies = load(cfg)
    cap = cv2.VideoCapture(cfg["video"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, Hh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f0, f1 = int(cfg["t_start"] * fps), int(cfg["t_end"] * fps)
    tr = interp_tracks(pos, np.arange(f0, f1 + 1))
    det_by_frame = {f: g for f, g in det.groupby("frame")}
    if stills:
        d = out_dir / "debug_frames"
        d.mkdir(exist_ok=True)
        for t in stills:
            fi = int(round(t * fps))
            fi = fi - (fi - f0) % cfg["frame_stride"]  # snap to a sampled frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            annotate(frame, fi, fi / fps, tr, det_by_frame, labels, rallies, fps)
            cv2.imwrite(str(d / f"t{int(t):04d}.jpg"), frame)
        print("stills written to", d)
        return
    out = out_dir / "annotated.mp4"
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), fps, (W, Hh))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    fi = f0
    n = 0
    while fi <= f1:
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps
        if rallies_only and rallies is not None:
            if not ((rallies.t_start - 1.0 <= t) & (rallies.t_end + 1.0 >= t)).any():
                fi += 1
                continue
        annotate(frame, fi, t, tr, det_by_frame, labels, rallies, fps)
        vw.write(frame)
        n += 1
        if n % 1500 == 0:
            print(f"rendered {n} frames (t={t:.0f}s)", flush=True)
        fi += 1
    vw.release()
    print(f"wrote {out} ({n} frames)")


if __name__ == "__main__":
    args = sys.argv[1:]
    stills = None
    if "--stills" in args:
        stills = [float(v) for v in args[args.index("--stills") + 1].split(",")]
    run(stills=stills, rallies_only="--rallies-only" in args)
