"""Stage 4: rally segmentation, serve detection and PAR score reconstruction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import court
from .calibrate import load_config

# Thresholds on the 1 s rolling sum of both players' speeds (m/s). The defaults suit a slow club
# game sampled at ~10 fps; set_params() rescales them from the game's own pace (percentiles of the
# activity and speed signals), and config `rally_params` can pin any of them.
ACT_HI, ACT_LO = 1.8, 0.9
MIN_RALLY_S, MERGE_GAP_S = 2.0, 1.2
SLOW = 0.8  # m/s: "standing" for serve configuration
SERVE_MIN_S = 0.4  # serve configuration must hold at least this long
BOX_PAD = 0.35  # metres of tolerance around the service boxes


RECEIVER_RULE = "opposite_half"  # or "past_inner_line": where the receiver may stand for a serve to count


def set_params(act, speed, fps_eff, overrides=None):
    """Scale the pause / serve thresholds to this game's pace. Returns the values used.
    Config `rally_params` can pin any of them; the one that most often needs a human decision is
    `receiver_rule`: "opposite_half" (receiver waits in the other half, the usual club pattern) or
    "past_inner_line" (receiver waits anywhere beyond the server's box, seen with fast-serving juniors)."""
    a10, a25, a50, a75 = np.nanpercentile(act, [10, 25, 50, 75])
    s10 = np.nanpercentile(speed, 10)
    p = dict(
        TROUGH_LO=max(0.8, 1.1 * a10),
        LULL_ACT=max(1.6, a25),
        ACT_PLAY=max(2.0, 0.85 * a50),
        MIN_PLAY_ACT=max(3.0, a75),
        SLOW=max(0.8, 2.5 * s10),
        BOX_PAD=0.35,
        SERVE_MIN_S=max(0.4, 4.0 / fps_eff),
        RELAX_SLOW=max(1.0, 3.2 * s10),
        RECEIVER_RULE="opposite_half",
    )
    p.update({(k.upper() if k.islower() else k): v for k, v in (overrides or {}).items()})
    globals().update(p)
    return p


def receiver_ok(xr, yr, box):
    """Is the receiver positioned for a serve from `box` (array or scalar inputs)?"""
    if RECEIVER_RULE == "past_inner_line":
        opp = np.where(box == "left", xr > court.BOX + 0.3, np.where(box == "right", xr < court.WIDTH - court.BOX - 0.3, False))
    else:
        opp = np.where(box == "left", xr > court.WIDTH / 2 + 0.3, np.where(box == "right", xr < court.WIDTH / 2 - 0.3, False))
    return opp & (yr > court.SHORT_LINE - 1.5)


def wide_pivot(pos):
    p = pos.pivot_table(index="frame", columns="player", values=["t", "x", "y", "speed", "dist_t"])
    return p


def times(p):
    return p[("t", "p1")].fillna(p[("t", "p2")]).to_numpy()


def activity_signal(p, fps_eff):
    sp = p["speed"].fillna(0)
    act = sp.sum(axis=1)
    win = max(3, int(round(1.0 * fps_eff)))
    return act.rolling(win, center=True, min_periods=1).mean()


def active_segments(act, t, fps_eff):
    """Hysteresis thresholding -> list of (start_idx, end_idx) inclusive."""
    on = False
    segs, start = [], None
    a = act.to_numpy()
    for i, v in enumerate(a):
        if not on and v > ACT_HI:
            on, start = True, i
        elif on and v < ACT_LO:
            on = False
            segs.append([start, i])
    if on:
        segs.append([start, len(a) - 1])
    # extend each segment backwards/forwards to where activity crossed ACT_LO
    ext = []
    for s, e in segs:
        while s > 0 and a[s - 1] > ACT_LO:
            s -= 1
        ext.append([s, e])
    merged = []
    for s, e in ext:
        if merged and t[s] - t[merged[-1][1]] < MERGE_GAP_S:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if t[e] - t[s] >= MIN_RALLY_S]


def box_of(x, y):
    lb = (
        (x >= court.LEFT_BOX[0] - BOX_PAD)
        & (x <= court.LEFT_BOX[2] + BOX_PAD)
        & (y >= court.LEFT_BOX[1] - BOX_PAD)
        & (y <= court.LEFT_BOX[3] + BOX_PAD)
    )
    rb = (
        (x >= court.RIGHT_BOX[0] - BOX_PAD)
        & (x <= court.RIGHT_BOX[2] + BOX_PAD)
        & (y >= court.RIGHT_BOX[1] - BOX_PAD)
        & (y <= court.RIGHT_BOX[3] + BOX_PAD)
    )
    return np.where(lb, "left", np.where(rb, "right", ""))


def serve_events(p, fps_eff):
    """Frames where one player stands in a service box and the other waits on the opposite side.
    When both interpretations hold, the one whose server is slower wins."""
    t = times(p)
    n = len(p)
    best_s = np.full(n, "", dtype=object)
    best_b = np.full(n, "", dtype=object)
    best_c = np.full(n, np.inf)
    for s, r in (("p1", "p2"), ("p2", "p1")):
        xs, ys, vs = p[("x", s)].to_numpy(), p[("y", s)].to_numpy(), p[("speed", s)].to_numpy()
        xr, yr, vr = p[("x", r)].to_numpy(), p[("y", r)].to_numpy(), p[("speed", r)].to_numpy()
        b = box_of(xs, ys)
        ok = (b != "") & receiver_ok(xr, yr, b) & (vs < SLOW) & (vr < 4.0)
        ok &= ~np.isnan(xs) & ~np.isnan(xr)
        cost = np.where(ok, vs + 0.2 * vr, np.inf)
        better = cost < best_c
        best_c[better], best_s[better], best_b[better] = cost[better], s, b[better]
    events = []
    i = 0
    while i < n:
        if best_s[i] == "":
            i += 1
            continue
        j = i
        while j + 1 < n and best_s[j + 1] == best_s[i]:
            j += 1
        if t[j] - t[i] >= SERVE_MIN_S:
            bx = pd.Series(best_b[i : j + 1]).mode().iloc[0]
            events.append(dict(server=best_s[i], box=bx, i0=i, i1=j, t0=float(t[i]), t1=float(t[j])))
        i = j + 1
    return events


def serve_clusters(events, gap=3.0):
    """Merge serve events closer than `gap` s; server/box by duration-weighted majority."""
    clusters = []
    for ev in events:
        if clusters and ev["t0"] - clusters[-1]["t1"] < gap:
            clusters[-1]["events"].append(ev)
            clusters[-1]["t1"] = ev["t1"]
        else:
            clusters.append(dict(t0=ev["t0"], t1=ev["t1"], events=[ev]))
    for c in clusters:
        w = {}
        for ev in c["events"]:
            k = (ev["server"], ev["box"])
            w[k] = w.get(k, 0) + (ev["t1"] - ev["t0"]) + 0.2
        (c["server"], c["box"]), _ = max(w.items(), key=lambda kv: kv[1])
        c["ambiguous"] = len(w) > 1 and sorted(w.values())[-1] < 1.5 * sorted(w.values())[-2]
    return clusters


ACT_PLAY = 2.0  # activity that only rally play reaches
TROUGH_LO, TROUGH_MIN_S = 0.8, 1.5


LULL_ACT, LULL_MIN_S = 1.6, 4.0


def first_lull(act, t, t0, t1):
    """Earliest start of a period in (t0, t1) where the activity stays below LULL_ACT for LULL_MIN_S:
    both players have stopped playing (walking, waiting, talking)."""
    idx = np.nonzero((t > t0) & (t < t1))[0]
    low = act[idx] < LULL_ACT
    i = 0
    while i < len(idx):
        if low[i]:
            j = i
            while j + 1 < len(idx) and low[j + 1]:
                j += 1
            if t[idx[j]] - t[idx[i]] >= LULL_MIN_S:
                return float(t[idx[i]])
            i = j + 1
        else:
            i += 1
    return t1


STILL_MIN_S, STILL_SPEED = 2.0, 0.6  # both players standing for this long ends a rally


def next_pause(a, t, t0, t1, both_slow=None):
    """First pause after t0: a deep trough (< TROUGH_LO for TROUGH_MIN_S), a lull (< LULL_ACT for
    LULL_MIN_S) or both players standing (speeds < SLOW for STILL_MIN_S). Returns (start, end) or None."""
    idx = np.nonzero((t > t0) & (t < t1))[0]
    best = None
    tests = [(a[idx] < TROUGH_LO, TROUGH_MIN_S), (a[idx] < LULL_ACT, LULL_MIN_S)]
    if both_slow is not None:
        tests.append((both_slow[idx], STILL_MIN_S))
    for low, min_s in tests:
        i = 0
        while i < len(idx):
            if low[i]:
                j = i
                while j + 1 < len(idx) and low[j + 1]:
                    j += 1
                if t[idx[j]] - t[idx[i]] >= min_s:
                    cand = (float(t[idx[i]]), float(t[idx[j]]))
                    if best is None or cand[0] < best[0]:
                        best = cand
                    break
                i = j + 1
            else:
                i += 1
    return best


def segment_rallies(clusters, act, t, t_end, p_speeds=None):
    """Rallies between consecutive serve clusters. The first rally of a cluster starts at the serve
    moment; play is cut at pauses (deep trough, sustained lull, or both players standing) and any play
    after a pause becomes a further rally whose serve was not detected (resolved by infer_missed_serves)."""
    a = act.to_numpy()
    both_slow = None
    if p_speeds is not None:
        sp = [pd.Series(v).rolling(5, center=True, min_periods=2).mean().to_numpy() for v in p_speeds.values()]
        both_slow = np.all([np.nan_to_num(v, nan=9.0) < STILL_SPEED for v in sp], axis=0)
    rallies = []
    for k, c in enumerate(clusters):
        ts = c["t1"]
        tn = clusters[k + 1]["t0"] if k + 1 < len(clusters) else t_end
        idx = np.nonzero((t > ts) & (t < tn))[0]
        if not len(idx):
            continue
        hi = idx[a[idx] > ACT_PLAY]
        if len(hi) == 0 or t[hi[-1]] - ts < 1.0:
            continue
        te_full = min(t[hi[-1]] + 1.0, tn - 0.3)
        start, server, box = ts, c["server"], c["box"]
        note = "serve ambiguous" if c["ambiguous"] else ""
        while start < te_full - 1.0:
            pause = next_pause(a, t, start + 1.5, te_full - 1.0, both_slow)
            if pause is None:
                rallies.append(dict(t_start=float(start), t_end=float(te_full), server=server, box=box, note=note))
                break
            p0, p1 = pause
            before = idx[(t[idx] > start) & (t[idx] < p0) & (a[idx] > ACT_PLAY)]
            end = min(float(t[before[-1]] + 1.0), p0) if len(before) else p0
            rallies.append(dict(t_start=float(start), t_end=float(end), server=server, box=box, note=note))
            after = idx[(t[idx] > p1) & (a[idx] > MIN_PLAY_ACT)]
            if not len(after):
                break
            start, server, box, note = p1, None, None, "serve not detected"
    for r in rallies:
        r["duration"] = float(r["t_end"] - r["t_start"])
    return [r for r in rallies if r["duration"] >= MIN_RALLY_S]


RELAX_PAD, RELAX_SLOW, RELAX_WINDOW_S, MIN_PLAY_ACT = 0.6, 1.0, 12.0, 3.0


def infer_missed_serves(rallies, p, act, t):
    """For rallies whose serve was not detected, look forward from the split point for a relaxed serve
    configuration (one player in a padded box, the other on the opposite side, both slow) before play
    starts; re-cut the rally start there. Serve-less segments that never reach play-level activity are
    dropped (walking, picking up the ball, discussions)."""
    a = act.to_numpy()
    xs = {pl: p[("x", pl)].to_numpy() for pl in ("p1", "p2")}
    ys = {pl: p[("y", pl)].to_numpy() for pl in ("p1", "p2")}
    vs = {pl: p[("speed", pl)].to_numpy() for pl in ("p1", "p2")}
    keep, dropped = [], []
    for r in rallies:
        if r["server"] is not None:
            keep.append(r)
            continue
        m = (t >= r["t_start"]) & (t <= r["t_end"])
        paths = [np.nansum(np.hypot(np.diff(xs[pl][m]), np.diff(ys[pl][m]))) for pl in ("p1", "p2")]
        if a[m].max() < MIN_PLAY_ACT or r["duration"] < 3.0 or min(paths) < 2.5:
            r["note"] = "dropped: no serve detected and no two-player play"
            dropped.append(r)
            continue
        idx = np.nonzero((t >= r["t_start"]) & (t <= min(r["t_start"] + RELAX_WINDOW_S, r["t_end"] - 1.0)))[0]
        hi = idx[a[idx] > MIN_PLAY_ACT]
        limit = hi[0] if len(hi) else idx[-1]
        found = None
        for i in idx:
            if i > limit:
                break
            for s, o in (("p1", "p2"), ("p2", "p1")):
                if np.isnan(xs[s][i]) or np.isnan(xs[o][i]):
                    continue
                pad = RELAX_PAD
                inl = (-pad <= xs[s][i] <= court.BOX + pad) and (court.SHORT_LINE - pad <= ys[s][i] <= court.SHORT_LINE + court.BOX + pad)
                inr = (court.WIDTH - court.BOX - pad <= xs[s][i] <= court.WIDTH + pad) and (
                    court.SHORT_LINE - pad <= ys[s][i] <= court.SHORT_LINE + court.BOX + pad
                )
                if not (inl or inr):
                    continue
                if receiver_ok(np.array(xs[o][i]), np.array(ys[o][i]), np.array("left" if inl else "right")) and vs[o][i] < 4.0:
                    found = (i, s, "left" if inl else "right")
        if found:
            i, s, b = found
            r.update(t_start=float(t[i]), server=s, box=b, note="serve inferred (relaxed)")
            r["duration"] = float(r["t_end"] - r["t_start"])
            keep.append(r)
        else:
            keep.append(r)
    return keep, dropped


LONG_RALLY_S = 16.0


def split_long_rallies(rallies, p, act, t):
    """Inside rallies longer than LONG_RALLY_S, look for a serve configuration during a quiet moment
    (server standing in a padded box, receiver past the inner line, activity below play level for the
    previous second) and split the rally there. Fast servers do not produce a visible pause."""
    a = act.to_numpy()
    xs = {pl: p[("x", pl)].to_numpy() for pl in ("p1", "p2")}
    ys = {pl: p[("y", pl)].to_numpy() for pl in ("p1", "p2")}
    vs = {pl: p[("speed", pl)].to_numpy() for pl in ("p1", "p2")}
    pad = RELAX_PAD
    out = []
    n_split = 0
    for r in rallies:
        parts = [r]
        if r["duration"] >= LONG_RALLY_S:
            idx = np.nonzero((t >= r["t_start"] + 4.0) & (t <= r["t_end"] - 3.0))[0]
            found = []
            for i in idx:
                if found and t[i] - found[-1][0] < 4.0:
                    continue
                quiet = a[(t >= t[i] - 1.0) & (t <= t[i])].max() < ACT_PLAY
                if not quiet:
                    continue
                for s_, o in (("p1", "p2"), ("p2", "p1")):
                    if np.isnan(xs[s_][i]) or np.isnan(xs[o][i]) or vs[s_][i] > RELAX_SLOW:
                        continue
                    inl = (-pad <= xs[s_][i] <= court.BOX + pad) and (
                        court.SHORT_LINE - pad <= ys[s_][i] <= court.SHORT_LINE + court.BOX + pad
                    )
                    inr = (court.WIDTH - court.BOX - pad <= xs[s_][i] <= court.WIDTH + pad) and (
                        court.SHORT_LINE - pad <= ys[s_][i] <= court.SHORT_LINE + court.BOX + pad
                    )
                    if not (inl or inr):
                        continue
                    if receiver_ok(np.array(xs[o][i]), np.array(ys[o][i]), np.array("left" if inl else "right")):
                        found.append((float(t[i]), s_, "left" if inl else "right"))
                        break
            if found:
                parts = []
                start, server, box, note = r["t_start"], r["server"], r["box"], r.get("note", "")
                for ts, s_, b in found:
                    parts.append(dict(t_start=float(start), t_end=float(ts - 0.5), server=server, box=box, note=note))
                    start, server, box, note = ts, s_, b, "serve inferred (quiet moment in a long rally)"
                parts.append(dict(t_start=float(start), t_end=float(r["t_end"]), server=server, box=box, note=note))
                n_split += len(found)
        for pr in parts:
            pr["duration"] = float(pr["t_end"] - pr["t_start"])
            if pr["duration"] >= MIN_RALLY_S:
                out.append(pr)
    if n_split:
        print(f"long rallies split at inferred serves: {n_split}")
    return out


MIN_RALLY_S_FINAL, MIN_RECEIVER_PATH_M = 2.5, 2.0


def plausible_rallies(rallies, p, act, t):
    """Drop segments a real rally cannot produce: too short, never reaching play-level activity, or one
    player moving while the other stands (a server crossing to the other box, players walking off)."""
    a = act.to_numpy()
    xs = {pl: p[("x", pl)].to_numpy() for pl in ("p1", "p2")}
    ys = {pl: p[("y", pl)].to_numpy() for pl in ("p1", "p2")}
    keep, dropped = [], []
    for r in rallies:
        m = (t >= r["t_start"]) & (t <= r["t_end"])
        paths = [np.nansum(np.hypot(np.diff(xs[pl][m]), np.diff(ys[pl][m]))) for pl in ("p1", "p2")]
        why = None
        if r["duration"] < MIN_RALLY_S_FINAL:
            why = f"shorter than {MIN_RALLY_S_FINAL:.1f} s"
        elif not m.any() or a[m].max() < 1.2 * ACT_PLAY:
            why = "never reached play-level activity"
        elif min(paths) < MIN_RECEIVER_PATH_M and r["duration"] < 4.0:
            why = "one player did not move (not a rally)"
        if why:
            r["note"] = (r.get("note", "") + "; dropped: " + why).strip("; ")
            dropped.append(r)
        else:
            keep.append(r)
    if dropped:
        print(f"implausible segments dropped: {len(dropped)} ({', '.join(d['note'].split('dropped: ')[-1] for d in dropped)})")
    return keep, dropped


def reconstruct_score(rallies):
    """PAR: winner of rally k = server of rally k+1. Same server & same box twice -> let."""
    score = {"p1": 0, "p2": 0}
    out = []
    for k, r in enumerate(rallies):
        nxt = rallies[k + 1] if k + 1 < len(rallies) else None
        winner, note = None, ""
        if r["server"] is None:
            note = "no serve detected"
        elif nxt is None:
            if score["p1"] == 14 and score["p2"] < 14:
                winner = "p1"
            elif score["p2"] == 14 and score["p1"] < 14:
                winner = "p2"
            else:
                note = "last rally: winner unknown"
        elif nxt["server"] is None:
            note = "next serve not detected: winner unknown"
        elif nxt["server"] == r["server"] and nxt["box"] == r["box"]:
            note = "let (same server, same box)"
        else:
            winner = nxt["server"]
        if winner:
            score[winner] += 1
        out.append(dict(winner=winner, score_p1=score["p1"], score_p2=score["p2"], note=note))
    return out


def run(cfg=None, force=False):
    cfg = cfg or load_config()
    out_dir = Path(cfg["output_dir"])
    out = out_dir / "rallies.csv"
    if out.exists() and not force:
        print("rallies exist, skipping")
        return pd.read_csv(out)
    pos = pd.read_csv(out_dir / "positions.csv")
    fps_eff = 1 / np.median(np.diff(np.sort(pos.t.unique())))
    p = wide_pivot(pos)
    t = times(p)
    act = activity_signal(p, fps_eff)
    params = set_params(act.to_numpy(), pos.speed.to_numpy(), fps_eff, cfg.get("rally_params"))
    print("rally thresholds for this game:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in params.items()})
    events = serve_events(p, fps_eff)
    clusters = serve_clusters(events)
    pd.DataFrame(dict(frame=p.index, t=t, activity=act.to_numpy())).to_csv(out_dir / "activity.csv", index=False)
    json.dump(events, open(out_dir / "serve_events.json", "w"), indent=1)
    json.dump([{k: v for k, v in c.items() if k != "events"} for c in clusters], open(out_dir / "serve_clusters.json", "w"), indent=1)
    p_speeds = {pl: p[("speed", pl)].to_numpy() for pl in ("p1", "p2")}
    rallies = segment_rallies(clusters, act, t, float(cfg["t_end"]), p_speeds)
    rallies, dropped = infer_missed_serves(rallies, p, act, t)
    rallies = split_long_rallies(rallies, p, act, t)
    rallies, implausible = plausible_rallies(rallies, p, act, t)
    dropped += implausible
    json.dump(dropped, open(out_dir / "dropped_segments.json", "w"), indent=1)
    sc = reconstruct_score(rallies)
    rows = []
    for k, (r, s) in enumerate(zip(rallies, sc), start=1):
        m = (t >= r["t_start"]) & (t <= r["t_end"])
        seg = p[m]
        row = dict(rally=k, **{kk: v for kk, v in r.items() if kk != "note"}, **s)
        row["note"] = "; ".join(x for x in (r.get("note", ""), s.get("note", "")) if x)
        row["rest_before"] = float(r["t_start"] - rallies[k - 2]["t_end"]) if k > 1 else np.nan
        for pl in ("p1", "p2"):
            x, y = seg[("x", pl)].to_numpy(), seg[("y", pl)].to_numpy()
            d = np.nansum(np.hypot(np.diff(x), np.diff(y)))
            dt_ = seg[("dist_t", pl)].to_numpy()
            row[f"dist_{pl}"] = float(d)
            row[f"tdist_{pl}"] = float(np.nanmean(dt_)) if np.isfinite(dt_).any() else np.nan
            row[f"tshare_{pl}"] = float(np.nanmean(dt_ < 1.5)) if np.isfinite(dt_).any() else np.nan
            sp = seg[("speed", pl)].to_numpy()
            row[f"maxspeed_{pl}"] = float(np.nanmax(sp)) if np.isfinite(sp).any() else np.nan
        d1, d2 = seg[("dist_t", "p1")].to_numpy(), seg[("dist_t", "p2")].to_numpy()
        both = ~np.isnan(d1) & ~np.isnan(d2)
        row["closer_to_t_p1"] = float(np.mean(d1[both] < d2[both])) if both.any() else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False, float_format="%.3f")
    print(f"serve events: {len(events)}, serve clusters: {len(clusters)}, rallies: {len(df)}, dropped segments: {len(dropped)}")
    print(df[["rally", "t_start", "t_end", "duration", "server", "box", "winner", "score_p1", "score_p2", "note"]].to_string(index=False))
    return df


if __name__ == "__main__":
    run(force="--force" in sys.argv)
