"""Stage 9: a standalone strengths-and-weaknesses page per player -> output/strengths.html (+ .pdf)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .calibrate import load_config
from .report import CHROME, match_findings, num, pct, player_findings

PLAYERS = ("p1", "p2")


def profile(S, cfg, u):
    """{'strengths': [(title, evidence, action)], 'weaknesses': [...]} for player u."""
    P = S["players"]
    L = {p: cfg["players"][p]["label"] for p in PLAYERS}
    o = "p2" if u == "p1" else "p1"
    U, Opp = P[u], P[o]
    R = pd.DataFrame(S["rallies"])
    scored = R[R.winner.notna()]
    S_, W_ = [], []

    # findings shared with the main report
    for _, kind, title, evidence, action in player_findings(S, cfg, u):
        (S_ if kind == "strength" else W_).append((title, evidence, action))

    # --- extra strength rules -------------------------------------------------------------
    tu, to = U["t_share_1_5m"], Opp["t_share_1_5m"]
    if tu >= 1.3 * to and tu >= 0.25:
        S_.append(
            (
                f"{L[u]} controls the middle of the court",
                f"{pct(tu)} of rally time within 1.5 m of the T against {pct(to)} for {L[o]}; the closer player to the T {pct(U['closer_to_t_share'])} of the time.",
                "Keep making the T the first destination after every shot; it is the base this game is built on.",
            )
        )
    if "closer_to_t_p1" in R and len(scored) >= 8:
        ct = R.closer_to_t_p1 if u == "p1" else 1 - R.closer_to_t_p1
        held = scored[ct.loc[scored.index] > 0.6]
        if len(held) >= 8 and (held.winner == u).mean() >= 0.6 and not any(t.startswith(f"{L[u]} wins when") for t, _, _ in S_):
            S_.append(
                (
                    f"{L[u]} wins most rallies as the T player",
                    f"Won {int((held.winner == u).sum())} of the {len(held)} rallies in which {L[u]} was the player nearer the T.",
                    "Get there first and the point tends to follow; the next step is to attack from there rather than hit another length.",
                )
            )
    du, do = U["distance_per_rally_m"], Opp["distance_per_rally_m"]
    if U["bursts"] >= 2 * max(Opp["bursts"], 1) and U["bursts"] >= 5:
        S_.append(
            (
                f"{L[u]} retrieves well and keeps going",
                f"{U['bursts']} sprints over 3 m/s to {Opp['bursts']} for {L[o]}, {num(du)} m per rally, top speed {num(U['peak_speed'])} m/s.",
                "The engine is there; spend it on getting back to the T, not only on chasing the ball.",
            )
        )
    if du <= 0.92 * do and U["points_won"] >= Opp["points_won"] - 1:
        S_.append(
            (
                f"{L[u]} wins without doing the running",
                f"{num(du)} m per rally against {num(do)} m for {L[o]} for the same number of points.",
                "Economy is a strength; make the opponent run further still with better width and height.",
            )
        )
    if (
        U["rallies_received"] >= 5
        and U["points_won_on_receive"] / U["rallies_received"] >= 0.5
        and not any("Return" in t for t, _, _ in S_)
    ):
        S_.append(
            (
                "Return of serve is a strength",
                f"{L[u]} won {U['points_won_on_receive']} of {U['rallies_received']} rallies as the receiver.",
                "Keep attacking the serve early.",
            )
        )
    if len(scored) >= 6:
        med = scored.duration.median()
        long_ = scored[scored.duration > med]
        if len(long_) >= 4 and (long_.winner == u).mean() >= 0.65 and not any("long" in t for t, _, _ in S_):
            S_.append(
                (
                    f"{L[u]} wins the long rallies",
                    f"Won {int((long_.winner == u).sum())} of {len(long_)} rallies longer than {num(med, 0)} s.",
                    "Patience is paying; keep the ball deep and let the opponent err.",
                )
            )

    # --- extra weakness rules -------------------------------------------------------------
    if U["front_share"] < 0.05 and not any("short" in t for t, _, _ in W_):
        W_.append(
            (
                f"{L[u]} never plays short",
                f"{pct(U['front_share'])} of rally time in the front third of the court.",
                "Add a straight drop from the T when the opponent is deep; without it every rally is a length contest.",
            )
        )
    if U["back_when_opp_at_t"] > 0.65 and U["back_when_opp_at_t"] >= Opp["back_when_opp_at_t"] + 0.10:
        W_.append(
            (
                f"{L[u]} gets pushed to the back wall",
                f"In the back third {pct(U['back_when_opp_at_t'])} of the time the opponent held the T (opponent: {pct(Opp['back_when_opp_at_t'])}).",
                "Volley more from the T so the ball does not reach the back; when it does, lob to buy time.",
            )
        )
    if U["rallies_served"] >= 5 and not any("serve" in t.lower() for t, _, _ in W_):
        rs = U["points_won_on_serve"] / U["rallies_served"]
        if rs < 0.4:
            W_.append(
                (
                    f"The serve is not earning {L[u]} points",
                    f"Won {U['points_won_on_serve']} of {U['rallies_served']} rallies on serve ({pct(rs)}).",
                    "Higher lob serve into the side wall; practise 20 a side.",
                )
            )
    if U["mean_rally_lost_s"] > 1.2 * U["mean_rally_won_s"] and U["points_lost"] >= 4:
        W_.append(
            (
                f"{L[u]} loses the longer exchanges",
                f"Rallies {L[u]} lost averaged {num(U['mean_rally_lost_s'], 0)} s; rallies won {num(U['mean_rally_won_s'], 0)} s.",
                "Finish earlier when in front, or build the fitness to outlast the opponent.",
            )
        )

    # "had the T in only N rallies" and "rarely on the T" are one point: keep the first, fold the numbers in
    only = [i for i in W_ if "had the T in only" in i[0]]
    rare = [i for i in W_ if "is rarely on the T" in i[0]]
    if only and rare:
        t, e, a = only[0]
        W_ = [(t, e + " " + rare[0][1].split(", and")[0] + ".", a) if i is only[0] else i for i in W_ if i is not rare[0]]

    def dedupe(items):
        seen, out = set(), []
        for t, e, a in items:
            k = t.lower()
            if k not in seen:
                seen.add(k)
                out.append((t, e, a))
        return out

    return dict(strengths=dedupe(S_), weaknesses=dedupe(W_))


def build(cfg, S, out_dir):
    P, G = S["players"], S["game"]
    L = {p: cfg["players"][p]["label"] for p in PLAYERS}
    prof = {p: profile(S, cfg, p) for p in PLAYERS}
    shared = match_findings(S, cfg)

    def items(lst, cls):
        if not lst:
            return "<li class='none'>Nothing the movement data can single out.</li>"
        return "".join(f"<li class='{cls}'><b>{t}</b><span class='ev'>{e}</span><span class='act'>{a}</span></li>" for t, e, a in lst)

    def col(p):
        return f"""<section class="player {p}">
  <h2><span class="dot"></span>{L[p]}<small>{cfg["players"][p].get("kit_desc", "")}</small></h2>
  <h3 class="s">Strengths</h3><ul>{items(prof[p]["strengths"], "s")}</ul>
  <h3 class="w">Weaknesses</h3><ul>{items(prof[p]["weaknesses"], "w")}</ul>
</section>"""

    def row(lab, a, b):
        return f"<tr><th>{lab}</th><td>{a}</td><td>{b}</td></tr>"

    u, o = "p1", "p2"
    numbers = "".join(
        [
            row("Points won", P[u]["points_won"], P[o]["points_won"]),
            row(
                "Won on serve / on return",
                f"{P[u]['points_won_on_serve']} of {P[u]['rallies_served']} / {P[u]['points_won_on_receive']} of {P[u]['rallies_received']}",
                f"{P[o]['points_won_on_serve']} of {P[o]['rallies_served']} / {P[o]['points_won_on_receive']} of {P[o]['rallies_received']}",
            ),
            row("Rally time within 1.5 m of the T", pct(P[u]["t_share_1_5m"]), pct(P[o]["t_share_1_5m"])),
            row("Closer to the T than the opponent", pct(P[u]["closer_to_t_share"]), pct(P[o]["closer_to_t_share"])),
            row("Median recovery to the T", num(P[u]["recovery_to_t_median_s"], 1, " s"), num(P[o]["recovery_to_t_median_s"], 1, " s")),
            row(
                "Distance per rally / sprints",
                f"{num(P[u]['distance_per_rally_m'])} m / {P[u]['bursts']}",
                f"{num(P[o]['distance_per_rally_m'])} m / {P[o]['bursts']}",
            ),
            row(
                "Front / middle / back third",
                f"{pct(P[u]['front_share'])} / {pct(P[u]['mid_share'])} / {pct(P[u]['back_share'])}",
                f"{pct(P[o]['front_share'])} / {pct(P[o]['mid_share'])} / {pct(P[o]['back_share'])}",
            ),
        ]
    )
    shared_html = "".join(f"<li><b>{t}.</b> {e}</li>" for _, _, t, e, _ in shared) or "<li>Nothing further.</li>"
    fs = G["final_score"]
    title = cfg.get("strengths_title", f"{L['p1']} vs {L['p2']} Strengths")
    html = f"""<meta charset="utf-8">
<title>{title}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;700&family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: two player columns side by side, strengths above weaknesses in each; shared notes and a numbers strip below; fits two A4 pages */
:root {{
  --bg:#f7f4ee; --paper:#fdfcf9; --ink:#191714; --ink-2:#5a554c; --line:#e3ded3; --shade:#efeae0;
  --court:#c8102e; --p1:#2a78d6; --p2:#eb6834; --good:#1a8f2e; --bad:#c8102e;
  --font-display:"Barlow Condensed","Arial Narrow",sans-serif; --font-body:"Source Sans 3","Helvetica Neue",Arial,sans-serif; --font-mono:"IBM Plex Mono",Menlo,monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#1a1a19; --paper:#222220; --ink:#f3f1ea; --ink-2:#bdb9ae; --line:#3a3934; --shade:#2a2927; --court:#ff5a6b; --p1:#3987e5; --p2:#d95926; --good:#3fbf3f; --bad:#ff5a6b; color-scheme:dark }} }}
:root[data-theme="dark"] {{ --bg:#1a1a19; --paper:#222220; --ink:#f3f1ea; --ink-2:#bdb9ae; --line:#3a3934; --shade:#2a2927; --court:#ff5a6b; --p1:#3987e5; --p2:#d95926; --good:#3fbf3f; --bad:#ff5a6b; color-scheme:dark }}
body {{ background:var(--bg); color:var(--ink); font-family:var(--font-body); font-size:16px; line-height:1.45; margin:0; }}
.wrap {{ max-width:1000px; margin:0 auto; padding-block:28px 56px; padding-inline:20px; }}
h1,h2,h3 {{ font-family:var(--font-display); margin:0; line-height:1.05; text-wrap:balance; }}
h1 {{ font-size:clamp(34px,5vw,50px); font-weight:700; }}
.eyebrow {{ font-family:var(--font-mono); font-size:12px; letter-spacing:.12em; text-transform:uppercase; color:var(--ink-2); }}
.lead {{ color:var(--ink-2); max-width:80ch; margin:8px 0 22px; }}
.cols {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; align-items:start; }}
section.player {{ background:var(--paper); border:1px solid var(--line); padding:18px 20px; min-width:0; }}
section.player h2 {{ font-size:30px; font-weight:700; display:flex; align-items:baseline; gap:10px; margin-bottom:6px; }}
section.player h2 small {{ font-family:var(--font-body); font-size:13px; color:var(--ink-2); font-weight:400; }}
.dot {{ width:12px; height:12px; border-radius:50%; display:inline-block; align-self:center; }}
section.p1 .dot {{ background:var(--p1); }} section.p2 .dot {{ background:var(--p2); }}
section.player h3 {{ font-size:14px; letter-spacing:.12em; text-transform:uppercase; font-family:var(--font-mono); font-weight:500; margin:16px 0 6px; padding-bottom:4px; border-bottom:2px solid var(--line); }}
h3.s {{ color:var(--good); border-color:var(--good) !important; }} h3.w {{ color:var(--bad); border-color:var(--bad) !important; }}
section.player ul {{ list-style:none; margin:0; padding:0; display:grid; gap:10px; }}
section.player li {{ padding-left:14px; border-left:3px solid var(--line); }}
section.player li.s {{ border-left-color:var(--good); }} section.player li.w {{ border-left-color:var(--bad); }}
section.player li b {{ display:block; font-size:16px; }}
section.player li .ev {{ display:block; font-size:14px; color:var(--ink-2); }}
section.player li .act {{ display:block; font-size:14px; margin-top:2px; }}
section.player li .act::before {{ content:"→ "; color:var(--ink-2); }}
li.none {{ color:var(--ink-2); font-size:14px; }}
.shared {{ margin-top:22px; background:var(--shade); padding:14px 18px; }}
.shared h2 {{ font-size:22px; margin-bottom:6px; }}
.shared ul {{ margin:0; padding-left:18px; font-size:15px; }} .shared li {{ margin:4px 0; }}
table {{ border-collapse:collapse; width:100%; margin-top:18px; font-variant-numeric:tabular-nums; font-size:14px; }}
th,td {{ text-align:left; padding:6px 10px; border-bottom:1px solid var(--line); }}
thead th {{ font-family:var(--font-mono); font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--ink-2); }}
thead th:nth-child(2) {{ color:var(--p1); }} thead th:nth-child(3) {{ color:var(--p2); }}
tbody th {{ font-weight:400; color:var(--ink-2); width:46%; }}
td {{ font-family:var(--font-mono); }}
.foot {{ font-size:13px; color:var(--ink-2); margin-top:14px; max-width:90ch; }}
@media (max-width:700px) {{ .cols {{ grid-template-columns:1fr; }} }}
@page {{ size: A4; margin: 12mm; }}
@media print {{ :root {{ --bg:#fff; --paper:#fff; --ink:#000; --ink-2:#444; --line:#bbb; --shade:#f2f2f2; }} body {{ font-size:11px; }} .wrap {{ max-width:none; padding:0; }} h1 {{ font-size:28px; }} section.player h2 {{ font-size:22px; }} section.player li b {{ font-size:12px; }} section.player li .ev, section.player li .act, .shared ul, table {{ font-size:10.5px; }} .cols {{ gap:12px; }} section.player {{ padding:10px 12px; }} section.player li, .shared, table {{ break-inside:avoid; }} section.player h3 {{ break-after:avoid; }} }}
</style>
<div class="wrap">
<div class="eyebrow">Squash · strengths and weaknesses from video · {cfg.get("game_date", "")} · {L["p1"]} {fs["p1"]} – {fs["p2"]} {L["p2"]} on camera</div>
<h1>{L["p1"]} vs {L["p2"]}: strengths and weaknesses</h1>
<p class="lead">From tracking both players' feet through the whole clip ({G["n_rallies"]} rallies, {G["n_points"]} decided points). Each point carries the number behind it and one thing to do about it. The ball is not tracked, so shot quality is inferred from where the players ended up, not from the stroke itself.</p>
<div class="cols">{col("p1")}{col("p2")}</div>
<div class="shared"><h2>True of both players</h2><ul>{shared_html}</ul></div>
<table><thead><tr><th>Head to head</th><th>{L["p1"]}</th><th>{L["p2"]}</th></tr></thead><tbody>{numbers}</tbody></table>
<p class="foot">Handedness assumed right for both. Positions are accurate to roughly 10 cm over the playing area. The full report has the charts, the rally-by-rally table and the method.</p>
</div>
"""
    return html


def run(cfg=None, force=True):
    cfg = cfg or load_config()
    out_dir = Path(cfg["output_dir"])
    S = json.load(open(out_dir / "summary.json"))
    html = build(cfg, S, out_dir)
    (out_dir / "strengths.html").write_text(html)
    print(f"strengths page written ({len(html) / 1e3:.0f} kB)")
    return html


def to_pdf(cfg=None):
    import shutil
    import subprocess

    cfg = cfg or load_config()
    out_dir = Path(cfg["output_dir"]).resolve()
    chrome = CHROME if Path(CHROME).exists() else shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        print("no Chrome/Chromium found: skipping PDF")
        return None
    pdf = out_dir / "strengths.pdf"
    subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--virtual-time-budget=15000",
            f"--print-to-pdf={pdf}",
            f"file://{out_dir / 'strengths.html'}",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"pdf written: {pdf} ({pdf.stat().st_size / 1e6:.2f} MB)")
    return pdf


if __name__ == "__main__":
    run()
    to_pdf()
