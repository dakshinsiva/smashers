"""Stage 6: figures -> output/figures/*.png  (palette: dataviz reference, slots 1+2)"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter

from . import court
from .calibrate import load_config

PLAYERS = ("p1", "p2")
THEMES = {
    "light": dict(
        COL={"p1": "#2a78d6", "p2": "#eb6834"},
        RAMP={"p1": ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95"], "p2": ["#fcfcfb", "#fbdccf", "#f3a07c", "#eb6834", "#a33c14"]},
        SURFACE="#fcfcfb",
        INK="#0b0b0b",
        INK2="#52514e",
        GRID="#e5e4e0",
        NEUTRAL="#b8b7b2",
        SHADE="#f1f0ec",
    ),
    "dark": dict(
        COL={"p1": "#3987e5", "p2": "#d95926"},
        RAMP={"p1": ["#1a1a19", "#1c3a5e", "#1c5cab", "#3987e5", "#9ec5f4"], "p2": ["#1a1a19", "#4a2617", "#a33c14", "#d95926", "#f3a07c"]},
        SURFACE="#1a1a19",
        INK="#ffffff",
        INK2="#c3c2b7",
        GRID="#33332f",
        NEUTRAL="#5e5d58",
        SHADE="#242422",
    ),
}
COL = RAMP = SURFACE = INK = INK2 = GRID = NEUTRAL = SHADE = None


def set_theme(name):
    global COL, RAMP, SURFACE, INK, INK2, GRID, NEUTRAL, SHADE
    th = THEMES[name]
    COL, RAMP, SURFACE, INK, INK2, GRID, NEUTRAL, SHADE = (
        th["COL"],
        th["RAMP"],
        th["SURFACE"],
        th["INK"],
        th["INK2"],
        th["GRID"],
        th["NEUTRAL"],
        th["SHADE"],
    )
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "legend.frameon": False,
            "figure.dpi": 160,
            "legend.labelcolor": INK,
        }
    )


def cmap(p):
    return LinearSegmentedColormap.from_list(p, RAMP[p])


def label(cfg, p):
    return cfg["players"][p]["label"]


def fig_heatmaps(cfg, summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.2))
    for ax, p in zip(axes, PLAYERS):
        h = np.array(summary["players"][p]["heatmap"])  # rows depth, cols x
        h = gaussian_filter(h, 1.2)
        h = h / h.sum() * 100
        ax.imshow(
            h,
            extent=[0, court.WIDTH, court.LENGTH, 0],
            cmap=cmap(p),
            vmin=0,
            vmax=np.percentile(h, 99.5),
            interpolation="bilinear",
            zorder=0,
            aspect="equal",
        )
        court.draw_court_mpl(ax, color=NEUTRAL)
        ax.grid(False)
        m = summary["players"][p]
        ax.set_title(f"{label(cfg, p)}  ·  {m['t_share_1_5m'] * 100:.0f}% within 1.5 m of the T", pad=18)
        ax.text(court.WIDTH / 2, -0.35, "front wall", ha="center", va="bottom", color=INK2, fontsize=8)
    fig.suptitle("Where each player stood during rallies (share of rally time)", x=0.02, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(out / "heatmaps.png")
    plt.close(fig)


def fig_zones(cfg, summary, out):
    names = [n for row in court.ZONE_NAMES for n in row]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    yv = np.arange(len(names))
    for i, p in enumerate(PLAYERS):
        vals = [summary["players"][p]["zones"][n] * 100 for n in names]
        bars = ax.barh(yv + (i - 0.5) * 0.36, vals, height=0.32, color=COL[p], label=label(cfg, p), zorder=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_width() + 0.6, b.get_y() + b.get_height() / 2, f"{v:.0f}%", va="center", fontsize=8, color=INK2)
    ax.set_yticks(yv, [n.replace("-", " ") for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("share of rally time (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Court zone occupancy (left/right as seen from the back wall)")
    fig.tight_layout()
    fig.savefig(out / "zones.png")
    plt.close(fig)


def fig_t_timeline(cfg, pos, rallies, out):
    t0, t1 = rallies.t_start.min() - 5, rallies.t_end.max() + 5
    n_rows = 3
    edges = np.linspace(t0, t1, n_rows + 1)
    fig, axes = plt.subplots(n_rows, 1, figsize=(7.2, 6.4), sharey=True)
    for k, ax in enumerate(axes):
        a, b = edges[k], edges[k + 1]
        for r in rallies.itertuples():
            if r.t_end < a or r.t_start > b:
                continue
            ax.axvspan(r.t_start, r.t_end, color=SHADE, zorder=0)
            ax.text((r.t_start + r.t_end) / 2, 5.9, str(r.rally), ha="center", fontsize=7, color=INK2)
        for p in PLAYERS:
            g = pos[(pos.player == p) & (pos.t >= a) & (pos.t <= b)]
            ax.plot(g.t, g.dist_t, color=COL[p], lw=1.2, label=label(cfg, p), zorder=3)
        ax.axhline(1.5, color=NEUTRAL, lw=0.8, ls="--", zorder=1)
        ax.set_xlim(a, b)
        ax.set_ylim(0, 6.4)
        ax.grid(axis="x", visible=False)
        if k == 0:
            ax.text(a + 1, 1.62, "1.5 m from T", fontsize=7, color=INK2)
        ax.set_ylabel("m from T")
    axes[-1].set_xlabel("video time (s)")
    fig.suptitle("Distance from the T through the game (shaded = rallies)", x=0.02, ha="left", fontsize=11, fontweight="semibold")
    fig.legend(
        handles=[Patch(color=COL[p], label=label(cfg, p)) for p in PLAYERS],
        loc="upper right",
        bbox_to_anchor=(0.99, 0.965),
        ncol=2,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "t_timeline.png")
    plt.close(fig)


def fig_rallies(cfg, rallies, out):
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for r in rallies.itertuples():
        c = COL.get(r.winner, NEUTRAL) if isinstance(r.winner, str) else NEUTRAL
        ax.bar(r.rally, r.duration, color=c, width=0.7, zorder=2)
        if isinstance(r.note, str) and "let" in r.note:
            ax.text(r.rally, r.duration + 0.4, "let", ha="center", fontsize=7, color=INK2)
    ax.set_xlabel("rally")
    ax.set_ylabel("seconds")
    ax.grid(axis="x", visible=False)
    ax.legend(
        handles=[Patch(color=COL[p], label=f"won by {label(cfg, p)}") for p in PLAYERS] + [Patch(color=NEUTRAL, label="let / unknown")],
        loc="upper left",
        ncol=3,
    )
    ax.set_title("Rally length and who won it")
    fig.tight_layout()
    fig.savefig(out / "rallies.png")
    plt.close(fig)


def fig_distance(cfg, rallies, out):
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for i, p in enumerate(PLAYERS):
        ax.bar(rallies.rally + (i - 0.5) * 0.38, rallies[f"dist_{p}"], width=0.36, color=COL[p], label=label(cfg, p), zorder=2)
    ax.set_xlabel("rally")
    ax.set_ylabel("metres run")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("Distance covered per rally")
    fig.tight_layout()
    fig.savefig(out / "distance.png")
    plt.close(fig)


def fig_score(cfg, rallies, out):
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.r_[0, rallies.rally]
    for p in PLAYERS:
        y = np.r_[0, rallies[f"score_{p}"]]
        ax.step(x, y, where="post", color=COL[p], lw=2, label=label(cfg, p), zorder=3)
        ax.text(x[-1] + 0.3, y[-1], f"{int(y[-1])}", color=INK, va="center", fontsize=9, fontweight="semibold")
    ax.set_xlabel("rally")
    ax.set_ylabel("points")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    ax.set_title("Reconstructed score (winner of a rally serves the next)")
    fig.tight_layout()
    fig.savefig(out / "score.png")
    plt.close(fig)


def fig_trajectories(cfg, pos, rallies, out):
    n = len(rallies)
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 1.55 * rows + 0.4))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, r in zip(axes.flat, rallies.itertuples()):
        ax.set_visible(True)
        court.draw_court_mpl(ax, color=NEUTRAL, lw=0.6)
        ax.grid(False)
        for p in PLAYERS:
            g = pos[(pos.player == p) & (pos.t >= r.t_start) & (pos.t <= r.t_end)]
            ax.plot(g.x, g.y, color=COL[p], lw=0.8, alpha=0.9, zorder=3)
            if len(g):
                ax.plot(g.x.iloc[0], g.y.iloc[0], "o", ms=3, color=COL[p], zorder=4)
        w = label(cfg, r.winner) if isinstance(r.winner, str) else "—"
        ax.set_title(f"R{r.rally}  {r.duration:.0f}s  → {w}", fontsize=7, pad=2)
    fig.suptitle("Movement paths per rally (dot = position at the serve)", x=0.02, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(out / "trajectories.png")
    plt.close(fig)


def fig_speed(cfg, pos, rallies, out):
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    m = np.zeros(len(pos), bool)
    for r in rallies.itertuples():
        m |= (pos.t >= r.t_start) & (pos.t <= r.t_end)
    bins = np.arange(0, 6.5, 0.25)
    for p in PLAYERS:
        g = pos[(pos.player == p) & m].speed.dropna()
        ax.hist(
            g, bins=bins, histtype="step", lw=1.8, color=COL[p], label=f"{label(cfg, p)} (mean {g.mean():.2f} m/s)", density=True, zorder=3
        )
    ax.set_xlabel("speed (m/s)")
    ax.set_ylabel("density")
    ax.legend(loc="upper right")
    ax.set_title("Speed distribution during rallies")
    fig.tight_layout()
    fig.savefig(out / "speed.png")
    plt.close(fig)


def run(cfg=None, force=False):
    cfg = cfg or load_config()
    out_dir = Path(cfg["output_dir"])
    summary = json.load(open(out_dir / "summary.json"))
    pos = pd.read_csv(out_dir / "positions.csv")
    rallies = pd.read_csv(out_dir / "rallies.csv")
    for theme in THEMES:
        set_theme(theme)
        out = out_dir / "figures" / theme
        out.mkdir(parents=True, exist_ok=True)
        fig_heatmaps(cfg, summary, out)
        fig_zones(cfg, summary, out)
        fig_t_timeline(cfg, pos, rallies, out)
        fig_rallies(cfg, rallies, out)
        fig_distance(cfg, rallies, out)
        fig_score(cfg, rallies, out)
        fig_trajectories(cfg, pos, rallies, out)
        fig_speed(cfg, pos, rallies, out)
        print(theme, "figures written:", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    run(force="--force" in sys.argv)
