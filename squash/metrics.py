"""Stage 5: metrics per player and per game -> summary.json"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import court
from .calibrate import load_config

PLAYERS = ("p1", "p2")
GRID = 0.25
T_R1, T_R2 = 1.0, 1.5
BURST_SPEED, BURST_MIN_S = 3.0, 0.5


def rally_mask(pos, rallies):
    m = np.zeros(len(pos), bool)
    rid = np.full(len(pos), 0)
    t = pos.t.to_numpy()
    for r in rallies.itertuples():
        sel = (t >= r.t_start) & (t <= r.t_end)
        m |= sel
        rid[sel] = r.rally
    return m, rid


def excursions(dist_t, t, radius=T_R2):
    """Median time (s) to get back inside `radius` of the T after leaving it."""
    out = dist_t > radius
    times = []
    i = 0
    while i < len(out):
        if out[i] and not np.isnan(dist_t[i]):
            j = i
            while j + 1 < len(out) and out[j + 1]:
                j += 1
            if j + 1 < len(out) and (t[j + 1] - t[i]) < 15:
                times.append(t[j + 1] - t[i])
            i = j + 1
        else:
            i += 1
    return float(np.median(times)) if times else float("nan"), len(times)


def bursts(speed, t):
    fast = speed > BURST_SPEED
    n, i = 0, 0
    while i < len(fast):
        if fast[i]:
            j = i
            while j + 1 < len(fast) and fast[j + 1]:
                j += 1
            if t[j] - t[i] >= BURST_MIN_S:
                n += 1
            i = j + 1
        else:
            i += 1
    return n


def player_metrics(p, pos_p, opp, rallies, in_rally, rid, handed):
    g = pos_p[in_rally]
    o = opp[in_rally]
    x, y, sp, dt_, t = g.x.to_numpy(), g.y.to_numpy(), g.speed.to_numpy(), g.dist_t.to_numpy(), g.t.to_numpy()
    valid = ~np.isnan(x)
    row, col = court.zone_of(x[valid], y[valid])
    zones = {}
    for r in range(3):
        for c in range(2):
            zones[court.ZONE_NAMES[r][c]] = float(np.mean((row == r) & (col == c)))
    fh_side = 1 if handed == "right" else 0  # right-handed: forehand on the right (x > 3.2)
    heat, _, _ = np.histogram2d(x[valid], y[valid], bins=[np.arange(0, court.WIDTH + GRID, GRID), np.arange(0, court.LENGTH + GRID, GRID)])
    step = np.hypot(np.diff(x), np.diff(y))
    step[np.isnan(step)] = 0
    med_rec, n_exc = excursions(dt_, t)
    odt = o.dist_t.to_numpy()
    both = valid & ~np.isnan(odt)
    opp_at_t = both & (odt < T_R1)
    per_rally = rallies.copy()
    won = per_rally[per_rally.winner == p]
    lost = per_rally[(per_rally.winner.notna()) & (per_rally.winner != p)]
    served = per_rally[per_rally.server == p]
    received = per_rally[(per_rally.server.notna()) & (per_rally.server != p)]
    m = dict(
        frames_in_rally=int(valid.sum()),
        t_share_1m=float(np.nanmean(dt_ < T_R1)),
        t_share_1_5m=float(np.nanmean(dt_ < T_R2)),
        closer_to_t_share=float(np.mean(dt_[both] < odt[both])),
        mean_dist_t=float(np.nanmean(dt_)),
        median_dist_t=float(np.nanmedian(dt_)),
        recovery_to_t_median_s=med_rec,
        n_excursions=n_exc,
        zones=zones,
        front_share=float(zones["front-left"] + zones["front-right"]),
        mid_share=float(zones["mid-left"] + zones["mid-right"]),
        back_share=float(zones["back-left"] + zones["back-right"]),
        left_share=float(np.mean(col == 0)),
        right_share=float(np.mean(col == 1)),
        forehand_side_share=float(np.mean(col == fh_side)),
        backhand_side_share=float(np.mean(col != fh_side)),
        handedness=handed,
        back_when_opp_at_t=float(np.mean(y[opp_at_t] > court.ZONE_Y_EDGES[2])) if opp_at_t.any() else float("nan"),
        distance_rally_m=float(step.sum()),
        distance_per_rally_m=float(rallies[f"dist_{p}"].mean()),
        distance_won_rallies_m=float(won[f"dist_{p}"].mean()) if len(won) else float("nan"),
        distance_lost_rallies_m=float(lost[f"dist_{p}"].mean()) if len(lost) else float("nan"),
        mean_speed=float(np.nanmean(sp)),
        peak_speed=float(np.nanpercentile(sp, 99)),
        bursts=bursts(np.nan_to_num(sp), t),
        points_won=int(len(won)),
        points_lost=int(len(lost)),
        rallies_served=int(len(served)),
        points_won_on_serve=int((served.winner == p).sum()),
        rallies_received=int(len(received)),
        points_won_on_receive=int((received.winner == p).sum()),
        mean_rally_won_s=float(won.duration.mean()) if len(won) else float("nan"),
        mean_rally_lost_s=float(lost.duration.mean()) if len(lost) else float("nan"),
        heatmap=heat.T.tolist(),  # rows = depth bins (front first), cols = x bins
    )
    # longest run of consecutive points
    run = best = 0
    for w in per_rally.winner:
        run = run + 1 if w == p else 0
        best = max(best, run)
    m["longest_run"] = int(best)
    return m


def run(cfg=None, force=False):
    cfg = cfg or load_config()
    out_dir = Path(cfg["output_dir"])
    out = out_dir / "summary.json"
    pos = pd.read_csv(out_dir / "positions.csv")
    rallies = pd.read_csv(out_dir / "rallies.csv")
    calib = json.load(open(out_dir / "calibration.json"))
    piv = {p: pos[pos.player == p].set_index("frame") for p in PLAYERS}
    frames = sorted(set(piv["p1"].index) | set(piv["p2"].index))
    for p in PLAYERS:
        piv[p] = piv[p].reindex(frames)
        piv[p]["t"] = piv[p]["t"].fillna(piv["p2" if p == "p1" else "p1"]["t"])
    in_rally, rid = rally_mask(piv["p1"], rallies)
    players = {}
    for p in PLAYERS:
        opp = piv["p2" if p == "p1" else "p1"]
        players[p] = player_metrics(p, piv[p], opp, rallies, in_rally, rid, cfg["players"][p]["handedness"])
        players[p]["label"] = cfg["players"][p]["label"]
    # game-level
    sep = np.hypot(piv["p1"].x - piv["p2"].x, piv["p1"].y - piv["p2"].y).to_numpy()[in_rally]
    scored = rallies[rallies.winner.notna()]
    lets = rallies[rallies.note.fillna("").str.contains("let (same", regex=False)]
    t_first, t_last = rallies.t_start.min(), rallies.t_end.max()
    total_rally = float(rallies.duration.sum())
    rest = rallies.rest_before.dropna()
    final = dict(p1=int(rallies.score_p1.iloc[-1]), p2=int(rallies.score_p2.iloc[-1]))
    game = dict(
        video=cfg["video"],
        analysed_window_s=[cfg["t_start"], cfg["t_end"]],
        game_window_s=[float(t_first), float(t_last)],
        game_duration_s=float(t_last - t_first),
        n_rallies=int(len(rallies)),
        n_points=int(len(scored)),
        n_lets=int(len(lets)),
        n_unknown=int(((rallies.winner.isna()) & ~rallies.note.fillna("").str.contains("let (same", regex=False)).sum()),
        final_score=final,
        known_final_score=cfg.get("known_final_score"),
        mean_rally_s=float(rallies.duration.mean()),
        median_rally_s=float(rallies.duration.median()),
        longest_rally_s=float(rallies.duration.max()),
        longest_rally_no=int(rallies.loc[rallies.duration.idxmax(), "rally"]),
        total_rally_time_s=total_rally,
        total_rest_time_s=float(rest.sum()),
        mean_rest_s=float(rest.mean()),
        work_rest_ratio=float(total_rally / rest.sum()) if rest.sum() else float("nan"),
        mean_separation_m=float(np.nanmean(sep)),
        crowding_share_1m=float(np.nanmean(sep < 1.0)),
        serve_box_alternation_ok=None,
    )
    # serve-box alternation check: consecutive rallies with same server after a won point must alternate boxes
    ok = tot = 0
    for a, b in zip(rallies.itertuples(), rallies.iloc[1:].itertuples()):
        if a.server == b.server and a.winner == a.server:
            tot += 1
            ok += int(a.box != b.box)
    game["serve_box_alternation_ok"] = f"{ok}/{tot}"
    elevated = calib.get("camera_profile") == "elevated"
    fps_eff = 1 / np.median(np.diff(np.sort(pos.t.unique())))
    reliability = dict(
        depth_scale_note=(
            "Floor calibration fitted on the painted lines with a lens model; positions are good to about 10 cm over the playing area, "
            "degrading to 20-30 cm in the far corners near the frame edge."
            if elevated
            else "Depth (front-back) scale rests on a single reference (the T); expect ~5% depth error, growing towards the back wall."
        ),
        accuracy_note=(
            "to within about 10 cm over the playing area, 20–30 cm in the far corners near the edge of the picture"
            if elevated
            else "to within about 10 cm across the court and 20–40 cm front-to-back (less precise towards the back wall, where the camera is)"
        ),
        samples_per_s=f"{fps_eff:.0f}",
        frame_bottom_depth_m=calib["frame_bottom_depth_m"],
        feet_cut_share={p: float(piv[p].feet_cut.fillna(False).astype(bool)[in_rally].mean()) for p in PLAYERS},
        tracked_share_in_rally={p: float(piv[p].x.notna()[in_rally].mean()) for p in PLAYERS},
        identity_disagreements=len(json.load(open(out_dir / "identity_swaps.json"))),
        ball_tracking="not attempted (576p, low camera, black ball)",
    )
    summary = dict(game=game, players=players, reliability=reliability, rallies=json.loads(rallies.to_json(orient="records")))
    json.dump(summary, open(out, "w"), indent=1)
    show = {k: v for k, v in game.items()}
    print(json.dumps(show, indent=1))
    for p in PLAYERS:
        m = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in players[p].items() if k not in ("heatmap", "zones")}
        print(p, json.dumps(m))
    return summary


if __name__ == "__main__":
    run(force="--force" in sys.argv)
