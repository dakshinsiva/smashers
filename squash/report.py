"""Stage 8: coach-facing HTML report -> output/report.html (figures embedded, light + dark variants)."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .calibrate import load_config

PLAYERS = ("p1", "p2")


def mmss(s):
    return f"{int(s) // 60}:{int(s) % 60:02d}"


def pct(v, d=0):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.{d}f}%"


def num(v, d=1, unit=""):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}{unit}"


def img_pair(out_dir, name, alt, caption=""):
    def b64(p):
        return base64.b64encode(open(p, "rb").read()).decode()

    light = out_dir / "figures" / "light" / f"{name}.png"
    d = out_dir / "figures" / "dark" / f"{name}.png"
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return (
        f'<figure><img class="fig-light" src="data:image/png;base64,{b64(light)}" alt="{alt}">'
        f'<img class="fig-dark" src="data:image/png;base64,{b64(d)}" alt="{alt}">{cap}</figure>'
    )


# ------------------------------------------------------------------------------------------------
# Coaching takeaways, written from the point of view of the user's player (config: user_player)
# ------------------------------------------------------------------------------------------------
def takeaways(S, cfg):
    P, G = S["players"], S["game"]
    L = {p: cfg["players"][p]["label"] for p in PLAYERS}
    u = cfg.get("user_player") or "p1"
    o = "p2" if u == "p1" else "p1"
    U, Opp = P[u], P[o]
    R = pd.DataFrame(S["rallies"])
    scored = R[R.winner.notna()]
    items = []  # (priority, kind, title, evidence, action)

    # 1. recovery to the T
    ru, ro = U["recovery_to_t_median_s"], Opp["recovery_to_t_median_s"]
    if not np.isnan(ru) and not np.isnan(ro) and ru > 1.3 * ro:
        items.append(
            (
                1,
                "work",
                f"{L[u]} is slow back to the T",
                f"After leaving the T area {L[u]} took a median {num(ru)} s to get back inside 1.5 m of it; {L[o]} took {num(ro)} s. "
                f"Over a rally that gap is where the opponent gets a free look at the front court.",
                "Two quick steps back towards the T the moment the ball leaves the racket, then watch the opponent from there. "
                "Ghosting (six-point star, 30 s on / 30 s off) trains exactly this.",
            )
        )
    elif not np.isnan(ru) and not np.isnan(ro) and ro > 1.3 * ru:
        items.append(
            (
                4,
                "strength",
                f"{L[u]} recovers to the T faster than {L[o]}",
                f"Median recovery {num(ru)} s against {num(ro)} s for {L[o]}.",
                "Keep it. This is the habit that wins the long rallies.",
            )
        )

    # 2. running cost of lost points
    wu, lu = U["distance_won_rallies_m"], U["distance_lost_rallies_m"]
    if not np.isnan(wu) and not np.isnan(lu) and lu > 1.25 * wu:
        items.append(
            (
                2,
                "work",
                f"Points slip away when {L[u]} is made to run",
                f"{L[u]} covered {num(lu)} m in the average lost rally but only {num(wu)} m in the average won one. "
                f"{L[o]} shows the opposite pattern ({num(Opp['distance_won_rallies_m'])} m won, {num(Opp['distance_lost_rallies_m'])} m lost): "
                f"when {L[o]} does the running, {L[o]} tends to win anyway.",
                "Hit for length first: a ball that dies in the back corner stops the opponent attacking. "
                "When stretched, lift a lob instead of a flat drive so there is time to get back.",
            )
        )

    # 3. long rallies
    if len(scored) >= 6:
        med = scored.duration.median()
        long_, short_ = scored[scored.duration > med], scored[scored.duration <= med]
        su = (long_.winner == u).mean()
        if su <= 0.4:
            items.append(
                (
                    3,
                    "work",
                    f"{L[o]} wins the long exchanges",
                    f"Rallies longer than the median ({num(med, 0)} s): {L[u]} won {int((long_.winner == u).sum())} of {len(long_)}. "
                    f"Rallies shorter than that: {L[u]} won {int((short_.winner == u).sum())} of {len(short_)}.",
                    f"Either finish the point earlier when {L[u]} is in front (see the short-game point below) or build the legs to outlast {L[o]}. "
                    f"Conditioned games to 7 with 'winner must win in under 6 shots' make the first option a habit.",
                )
            )
        elif su >= 0.65:
            items.append(
                (
                    4,
                    "strength",
                    f"{L[u]} wins the long exchanges",
                    f"{L[u]} won {int((long_.winner == u).sum())} of {len(long_)} rallies longer than the median ({num(med, 0)} s).",
                    "Patience is paying off; keep the length and let the opponent make the error.",
                )
            )

    # 4. serve
    if U["rallies_served"] >= 5:
        rs = U["points_won_on_serve"] / U["rallies_served"]
        if rs < 0.4:
            items.append(
                (
                    3,
                    "work",
                    f"The serve is not earning {L[u]} points",
                    f"{L[u]} won {U['points_won_on_serve']} of {U['rallies_served']} rallies on serve ({pct(rs)}); "
                    f"{L[o]} won {Opp['points_won_on_serve']} of {Opp['rallies_served']} on {L[o]}'s serve.",
                    "A high lob serve that clings to the side wall and dies in the back corner forces a weak return. "
                    "Practise 20 serves a side, aiming to hit the side wall above the service line.",
                )
            )
    # 5. return
    if U["rallies_received"] >= 5:
        rr = U["points_won_on_receive"] / U["rallies_received"]
        if rr >= 0.5:
            items.append(
                (
                    5,
                    "strength",
                    "Return of serve is a strength",
                    f"{L[u]} won {U['points_won_on_receive']} of {U['rallies_received']} rallies as the receiver ({pct(rr)}).",
                    "Keep attacking the serve with a straight volley; it is currently the most productive shot in the game.",
                )
            )

    # 6. short game
    fu, fo = U["front_share"], Opp["front_share"]
    if fu < 0.05 and fo < 0.05:
        items.append(
            (
                2,
                "tactic",
                "Nobody played the front of the court",
                f"Time spent in the front third: {pct(fu)} for {L[u]}, {pct(fo)} for {L[o]}. Every rally was drives and lobs into the back.",
                f"When {L[u]} is on the T and {L[o]} is behind (which was {pct(Opp['back_when_opp_at_t'])} of the time {L[o]} held the T), "
                f"a straight drop or a trickle boast wins the point or forces a loose ball. Drop-and-drive routines build the confidence to use it.",
            )
        )
    elif fu < 0.05:
        items.append(
            (
                2,
                "tactic",
                f"{L[u]} never went short",
                f"Only {pct(fu)} of {L[u]}'s rally time was in the front third.",
                "Add the straight drop from the T when the opponent is deep.",
            )
        )

    # 7. T control did not decide points?
    if "closer_to_t_p1" in R:
        ct = R.closer_to_t_p1 if u == "p1" else 1 - R.closer_to_t_p1
        held = scored[ct.loc[scored.index] > 0.6]
        lost_t = scored[ct.loc[scored.index] < 0.4]
        if len(held) >= 4 and len(lost_t) >= 4:
            wh, wl = (held.winner == u).mean(), (lost_t.winner == u).mean()
            if abs(wh - wl) < 0.15:
                items.append(
                    (
                        3,
                        "tactic",
                        "Holding the T did not decide points",
                        f"When {L[u]} was the T player, {L[u]} won {int((held.winner == u).sum())} of {len(held)} rallies; when {L[o]} held the T, "
                        f"{L[u]} still won {int((lost_t.winner == u).sum())} of {len(lost_t)}. Overall time within 1.5 m of the T: {pct(U['t_share_1_5m'])} vs {pct(Opp['t_share_1_5m'])}.",
                        "The T is only worth something if the player on it attacks: volley early, take the ball short. Right now both players hold the T and then hit another length.",
                    )
                )
            elif wh - wl >= 0.25:
                items.append(
                    (
                        3,
                        "tactic",
                        f"{L[u]} wins when {L[u]} holds the T",
                        f"As the T player {L[u]} won {int((held.winner == u).sum())} of {len(held)} rallies, against {int((lost_t.winner == u).sum())} of {len(lost_t)} when {L[o]} held it.",
                        "Getting to the T first is the whole game plan; the recovery drills above pay off directly.",
                    )
                )

    # 8. backhand side
    # T presence gap
    tu, to = U["t_share_1_5m"], Opp["t_share_1_5m"]
    if to >= 0.2 and tu < 0.65 * to:
        items.append(
            (
                1,
                "work",
                f"{L[u]} is rarely on the T",
                f"{L[u]} spent {pct(tu)} of rally time within 1.5 m of the T against {pct(to)} for {L[o]}, and was the closer player to the T only "
                f"{pct(U['closer_to_t_share'])} of the time. Mean distance from the T: {num(U['mean_dist_t'], 1)} m vs {num(Opp['mean_dist_t'], 1)} m.",
                "Every shot should end with a move back towards the T, not a step back to the wall. Until that is a habit, "
                f"{L[u]} is playing each rally from behind and has to run further to win it.",
            )
        )
    # wins that cost more running than losses: winning with legs, not position
    if not np.isnan(wu) and not np.isnan(lu) and wu > 1.15 * lu and wu >= 14:
        items.append(
            (
                2,
                "work",
                f"{L[u]} wins points by running, not by position",
                f"{L[u]} ran {num(wu)} m in the average rally won but {num(lu)} m in the average rally lost, with {U['bursts']} sprints over 3 m/s "
                f"to {Opp['bursts']} for {L[o]}. The points are being retrieved rather than controlled.",
                "That works against this opponent; it will not against one who finishes points. Pair the retrieving with a return to the T so the next ball is taken early.",
            )
        )
    # low mobility relative to the opponent (only when it is a clear gap)
    if Opp["bursts"] >= 5 and U["bursts"] <= 0.4 * Opp["bursts"] and U["mean_speed"] < 0.9 * Opp["mean_speed"]:
        items.append(
            (
                3,
                "tactic",
                f"{L[u]} is the less explosive mover",
                f"{U['bursts']} sprints over 3 m/s against {Opp['bursts']} for {L[o]}; average speed {num(U['mean_speed'], 2)} m/s vs {num(Opp['mean_speed'], 2)} m/s. "
                f"{L[u]} is winning on position and the return, not on pace.",
                "Fine as long as the opponent keeps hitting length. An opponent who takes the ball short will expose it; "
                "add a short-ball ghosting set (T to front corners) to the warm-up.",
            )
        )
    bh = U["backhand_side_share"]
    if bh > 0.58 and Opp["backhand_side_share"] <= 0.58:  # both high -> a match pattern, see match_findings
        items.append(
            (
                4,
                "tactic",
                f"{L[u]} was kept on the backhand side",
                f"{pct(bh)} of {L[u]}'s rally time was on the backhand side (assuming {U['handedness']}-handed).",
                "Check on the video whether it is the opponent's targeting or drifting; a stronger backhand length changes the pattern either way.",
            )
        )

    # 9. crowding
    if G["crowding_share_1m"] > 0.12:
        items.append(
            (
                4,
                "tactic",
                "Crowded rallies",
                f"Players within 1 m of each other {pct(G['crowding_share_1m'])} of rally time; {G['n_lets']} lets recorded.",
                "Clear the ball faster after each shot.",
            )
        )
    items.sort(key=lambda x: x[0])
    return items


def player_findings(S, cfg, u):
    """Findings about a single player u, from u's own perspective - no shared/both-players branches
    (those live in match_findings). Used for the symmetric two-player report (user_player: null)."""
    P = S["players"]
    L = {p: cfg["players"][p]["label"] for p in PLAYERS}
    o = "p2" if u == "p1" else "p1"
    U, Opp = P[u], P[o]
    R = pd.DataFrame(S["rallies"])
    scored = R[R.winner.notna()]
    items = []

    ru, ro = U["recovery_to_t_median_s"], Opp["recovery_to_t_median_s"]
    if not np.isnan(ru) and not np.isnan(ro) and ru > 1.3 * ro:
        items.append(
            (
                1,
                "work",
                f"{L[u]} is slow back to the T",
                f"After leaving the T area {L[u]} took a median {num(ru)} s to get back inside 1.5 m of it; {L[o]} took {num(ro)} s. "
                f"Over a rally that gap is where the opponent gets a free look at the front court.",
                "Two quick steps back towards the T the moment the ball leaves the racket, then watch the opponent from there. "
                "Ghosting (six-point star, 30 s on / 30 s off) trains exactly this.",
            )
        )
    elif not np.isnan(ru) and not np.isnan(ro) and ro > 1.3 * ru:
        items.append(
            (
                4,
                "strength",
                f"{L[u]} recovers to the T faster than {L[o]}",
                f"Median recovery {num(ru)} s against {num(ro)} s for {L[o]}.",
                "Keep it. This is the habit that wins the long rallies.",
            )
        )

    wu, lu = U["distance_won_rallies_m"], U["distance_lost_rallies_m"]
    if not np.isnan(wu) and not np.isnan(lu) and lu > 1.25 * wu:
        items.append(
            (
                2,
                "work",
                f"Points slip away when {L[u]} is made to run",
                f"{L[u]} covered {num(lu)} m in the average lost rally but only {num(wu)} m in the average won one.",
                "Hit for length first: a ball that dies in the back corner stops the opponent attacking. "
                "When stretched, lift a lob instead of a flat drive so there is time to get back.",
            )
        )

    if len(scored) >= 6:
        med = scored.duration.median()
        long_, short_ = scored[scored.duration > med], scored[scored.duration <= med]
        su = (long_.winner == u).mean()
        if su <= 0.4:
            items.append(
                (
                    3,
                    "work",
                    f"{L[o]} wins the long exchanges",
                    f"Rallies longer than the median ({num(med, 0)} s): {L[u]} won {int((long_.winner == u).sum())} of {len(long_)}. "
                    f"Rallies shorter than that: {L[u]} won {int((short_.winner == u).sum())} of {len(short_)}.",
                    f"Either finish the point earlier when {L[u]} is in front (see the short-game point below) or build the legs to outlast {L[o]}. "
                    f"Conditioned games to 7 with 'winner must win in under 6 shots' make the first option a habit.",
                )
            )
        elif su >= 0.65:
            items.append(
                (
                    4,
                    "strength",
                    f"{L[u]} wins the long exchanges",
                    f"{L[u]} won {int((long_.winner == u).sum())} of {len(long_)} rallies longer than the median ({num(med, 0)} s).",
                    "Patience is paying off; keep the length and let the opponent make the error.",
                )
            )

    if U["rallies_served"] >= 5:
        rs = U["points_won_on_serve"] / U["rallies_served"]
        if rs < 0.4:
            items.append(
                (
                    3,
                    "work",
                    f"The serve is not earning {L[u]} points",
                    f"{L[u]} won {U['points_won_on_serve']} of {U['rallies_served']} rallies on serve ({pct(rs)}).",
                    "A high lob serve that clings to the side wall and dies in the back corner forces a weak return. "
                    "Practise 20 serves a side, aiming to hit the side wall above the service line.",
                )
            )
        elif rs >= 0.65:
            items.append(
                (
                    5,
                    "strength",
                    f"{L[u]}'s serve is winning points outright",
                    f"{L[u]} won {U['points_won_on_serve']} of {U['rallies_served']} rallies on serve ({pct(rs)}).",
                    "Keep serving to the same spot; there is no sign the opponent has solved it yet.",
                )
            )

    if U["rallies_received"] >= 5:
        rr = U["points_won_on_receive"] / U["rallies_received"]
        if rr >= 0.5:
            items.append(
                (
                    5,
                    "strength",
                    "Return of serve is a strength",
                    f"{L[u]} won {U['points_won_on_receive']} of {U['rallies_received']} rallies as the receiver ({pct(rr)}).",
                    "Keep attacking the serve with a straight volley; it is currently the most productive shot in the game.",
                )
            )
        elif rr <= 0.25:
            items.append(
                (
                    2,
                    "work",
                    "The return of serve is giving points away",
                    f"{L[u]} won only {U['points_won_on_receive']} of {U['rallies_received']} rallies as the receiver ({pct(rr)}).",
                    "A loose return hands the server the T for free. Aim the return straight and deep rather than across the court.",
                )
            )

    fu = U["front_share"]
    if fu < 0.05 and Opp["front_share"] >= 0.05:  # shared version lives in match_findings
        items.append(
            (
                3,
                "tactic",
                f"{L[u]} never went short",
                f"Only {pct(fu)} of {L[u]}'s rally time was in the front third.",
                "Add the straight drop from the T when the opponent is deep; right now every rally is being played at length.",
            )
        )

    if "closer_to_t_p1" in R:
        ct = R.closer_to_t_p1 if u == "p1" else 1 - R.closer_to_t_p1
        held = scored[ct.loc[scored.index] > 0.6]
        lost_t = scored[ct.loc[scored.index] < 0.4]
        if len(held) <= 2 and len(lost_t) >= 8:
            items.append(
                (
                    1,
                    "work",
                    f"{L[u]} had the T in only {len(held)} of {len(scored)} decided rallies",
                    f"{L[o]} was the T player in {len(lost_t)} rallies; {L[u]} still won {int((lost_t.winner == u).sum())} of those from behind.",
                    f"Winning from behind is the hardest way to play squash. The target for {L[u]} is to be the T player in at least half the rallies.",
                )
            )
        if len(held) >= 4:
            wh = (held.winner == u).mean()
            if wh >= 0.75:
                items.append(
                    (
                        4,
                        "strength",
                        f"{L[u]} wins when {L[u]} holds the T",
                        f"As the T player {L[u]} won {int((held.winner == u).sum())} of {len(held)} rallies.",
                        "Getting to the T first is working; keep prioritising the recovery over the pace of the shot.",
                    )
                )
            elif wh <= 0.4 and len(held) >= 5:
                items.append(
                    (
                        2,
                        "work",
                        f"{L[u]} holds the T but does not convert it",
                        f"As the T player {L[u]} won only {int((held.winner == u).sum())} of {len(held)} rallies from there.",
                        "Holding the T only pays off if the shot from there is attacking (a tight volley or a drop). "
                        "Review what shot was played on these rallies - it is probably another length ball.",
                    )
                )

    # T presence gap
    tu, to = U["t_share_1_5m"], Opp["t_share_1_5m"]
    if to >= 0.2 and tu < 0.65 * to:
        items.append(
            (
                1,
                "work",
                f"{L[u]} is rarely on the T",
                f"{L[u]} spent {pct(tu)} of rally time within 1.5 m of the T against {pct(to)} for {L[o]}, and was the closer player to the T only "
                f"{pct(U['closer_to_t_share'])} of the time. Mean distance from the T: {num(U['mean_dist_t'], 1)} m vs {num(Opp['mean_dist_t'], 1)} m.",
                "Every shot should end with a move back towards the T, not a step back to the wall. Until that is a habit, "
                f"{L[u]} is playing each rally from behind and has to run further to win it.",
            )
        )
    # wins that cost more running than losses: winning with legs, not position
    if not np.isnan(wu) and not np.isnan(lu) and wu > 1.15 * lu and wu >= 14:
        items.append(
            (
                2,
                "work",
                f"{L[u]} wins points by running, not by position",
                f"{L[u]} ran {num(wu)} m in the average rally won but {num(lu)} m in the average rally lost, with {U['bursts']} sprints over 3 m/s "
                f"to {Opp['bursts']} for {L[o]}. The points are being retrieved rather than controlled.",
                "That works against this opponent; it will not against one who finishes points. Pair the retrieving with a return to the T so the next ball is taken early.",
            )
        )
    # low mobility relative to the opponent (only when it is a clear gap)
    if Opp["bursts"] >= 5 and U["bursts"] <= 0.4 * Opp["bursts"] and U["mean_speed"] < 0.9 * Opp["mean_speed"]:
        items.append(
            (
                3,
                "tactic",
                f"{L[u]} is the less explosive mover",
                f"{U['bursts']} sprints over 3 m/s against {Opp['bursts']} for {L[o]}; average speed {num(U['mean_speed'], 2)} m/s vs {num(Opp['mean_speed'], 2)} m/s. "
                f"{L[u]} is winning on position and the return, not on pace.",
                "Fine as long as the opponent keeps hitting length. An opponent who takes the ball short will expose it; "
                "add a short-ball ghosting set (T to front corners) to the warm-up.",
            )
        )
    bh = U["backhand_side_share"]
    if bh > 0.58 and Opp["backhand_side_share"] <= 0.58:  # both high -> a match pattern, see match_findings
        items.append(
            (
                4,
                "tactic",
                f"{L[u]} was kept on the backhand side",
                f"{pct(bh)} of {L[u]}'s rally time was on the backhand side (assuming {U['handedness']}-handed).",
                "Check on the video whether it is the opponent's targeting or drifting; a stronger backhand length changes the pattern either way.",
            )
        )
    elif bh < 0.40:
        items.append(
            (
                5,
                "strength",
                f"{L[u]} dominates the forehand side and still competes",
                f"Only {pct(bh)} of {L[u]}'s rally time was on the backhand side (assuming {U['handedness']}-handed).",
                "The forehand corner is a stronghold; look to move the opponent there more often on serve and on the attacking shot.",
            )
        )

    items.sort(key=lambda x: x[0])
    return items


def match_findings(S, cfg):
    """Findings about the match as a whole, not attributable to one player - shown once, shared."""
    P, G = S["players"], S["game"]
    L = {p: cfg["players"][p]["label"] for p in PLAYERS}
    R = pd.DataFrame(S["rallies"])
    scored = R[R.winner.notna()]
    items = []

    fa, fb = P["p1"]["front_share"], P["p2"]["front_share"]
    if fa < 0.05 and fb < 0.05:
        items.append(
            (
                2,
                "tactic",
                "Neither player played the front of the court",
                f"Time spent in the front third: {pct(fa)} for {L['p1']}, {pct(fb)} for {L['p2']}. Every rally was drives and lobs into the back.",
                "A straight drop or a trickle boast from the T wins the point or forces a loose ball outright against two players who never expect it. "
                "Drop-and-drive routines in practice build the confidence to use it in a match.",
            )
        )

    if "closer_to_t_p1" in R and len(scored) >= 8:
        ct = R.closer_to_t_p1
        held = scored[ct.loc[scored.index] > 0.6]
        lost_t = scored[ct.loc[scored.index] < 0.4]
        if len(held) >= 4 and len(lost_t) >= 4:
            wh, wl = (held.winner == "p1").mean(), (lost_t.winner == "p1").mean()
            if abs(wh - wl) < 0.15:
                items.append(
                    (
                        3,
                        "tactic",
                        "Holding the T did not decide points for either player",
                        f"When {L['p1']} held the T, {L['p1']} won {pct(wh)} of those rallies; when {L['p2']} held the T, {L['p1']} still won {pct(wl)}. "
                        f"Overall time within 1.5 m of the T: {pct(P['p1']['t_share_1_5m'])} ({L['p1']}) vs {pct(P['p2']['t_share_1_5m'])} ({L['p2']}).",
                        "The T is only worth something if the player on it attacks. Right now both players hold the T and then hit another length, "
                        "so the position is being wasted by both sides.",
                    )
                )

    sa = P["p1"]["points_won_on_serve"] / P["p1"]["rallies_served"] if P["p1"]["rallies_served"] else np.nan
    sb = P["p2"]["points_won_on_serve"] / P["p2"]["rallies_served"] if P["p2"]["rallies_served"] else np.nan
    if P["p1"]["rallies_served"] >= 5 and P["p2"]["rallies_served"] >= 5 and sa < 0.4 and sb < 0.4:
        items.append(
            (
                3,
                "tactic",
                "Serving was a disadvantage for both players",
                f"The server won only {P['p1']['points_won_on_serve']} of {P['p1']['rallies_served']} rallies ({L['p1']}) and "
                f"{P['p2']['points_won_on_serve']} of {P['p2']['rallies_served']} ({L['p2']}). Both serves are being returned comfortably.",
                "A higher lob serve into the side wall, for both players, would change the pattern - right now serving is close to a disadvantage in this match.",
            )
        )

    ba, bb = P["p1"]["backhand_side_share"], P["p2"]["backhand_side_share"]
    if ba > 0.58 and bb > 0.58:
        items.append(
            (
                3,
                "tactic",
                "The match was played down the backhand wall",
                f"{pct(ba)} of {L['p1']}'s rally time and {pct(bb)} of {L['p2']}'s was on the left (backhand) side, assuming both are right-handed. "
                f"Both players kept hitting to the same wall.",
                "Whoever first moves the ball to the forehand side with a cross-court or a straight length on the right gets an easier ball back; "
                "it also tests whether the opponent's forehand length is as tight as the backhand.",
            )
        )
    if len(scored) >= 8:
        recv = int((scored.winner != scored.server).sum())
        if recv / len(scored) >= 0.7:
            items.append(
                (
                    1,
                    "tactic",
                    f"The receiver won {recv} of {len(scored)} decided points",
                    f"Serving was a liability for both: {L['p1']} held serve {P['p1']['points_won_on_serve']} of {P['p1']['rallies_served']} times, "
                    f"{L['p2']} {P['p2']['points_won_on_serve']} of {P['p2']['rallies_served']}. The points swapped sides almost every rally.",
                    "Serve to win the first exchange, not just to start the rally: a high lob serve that dies in the back corner, or a hard low serve to the body. "
                    "The receiver's first shot is clearly the strongest shot in this match for both players.",
                )
            )
            items = [i for i in items if i[2] != "Serving was a disadvantage for both players"]

    if G["crowding_share_1m"] > 0.12:
        items.append(
            (
                4,
                "tactic",
                "Crowded rallies",
                f"The players were within 1 m of each other for {pct(G['crowding_share_1m'])} of rally time; {G['n_lets']} lets recorded.",
                "Both players should clear the ball towards the T faster after each shot.",
            )
        )

    items.sort(key=lambda x: x[0])
    return items


def review_moments_neutral(S, cfg):
    """Rally-review list for the symmetric (no single user) report."""
    L = {p: cfg["players"][p]["label"] for p in PLAYERS}
    R = pd.DataFrame(S["rallies"])
    out = []
    r = R.loc[R.duration.idxmax()]
    out.append(
        (
            r,
            f"Longest rally ({num(r.duration, 0)} s), {('won by ' + L[r.winner]) if isinstance(r.winner, str) else 'winner not decided on camera'}: the shape of the whole match in one exchange.",
        )
    )
    for p in PLAYERS:
        o = "p2" if p == "p1" else "p1"
        lost = R[R.winner == o].sort_values(f"dist_{p}", ascending=False).head(2)
        for r in lost.itertuples():
            out.append((r, f"{L[p]} lost this rally after {num(getattr(r, f'dist_{p}'), 0)} m of running: where did the length go loose?"))
        won = R[R.winner == p].sort_values(f"tshare_{p}", ascending=False).head(1)
        for r in won.itertuples():
            out.append((r, f"{L[p]} won while holding the T {pct(getattr(r, f'tshare_{p}'))} of the rally: the model to repeat."))
    for r in R[R.note.fillna("").str.contains("let", regex=False)].itertuples():
        out.append((r, "Counted as a let (same player served twice from the same box)."))
    seen, uniq = set(), []
    for r, why in out:
        k = int(r.rally)
        if k not in seen:
            seen.add(k)
            uniq.append((r, why))
    return sorted(uniq, key=lambda x: int(x[0].rally))


def review_moments(S, cfg):
    L = {p: cfg["players"][p]["label"] for p in PLAYERS}
    u = cfg.get("user_player") or "p1"
    o = "p2" if u == "p1" else "p1"
    R = pd.DataFrame(S["rallies"])
    out = []
    r = R.loc[R.duration.idxmax()]
    out.append(
        (
            r,
            f"Longest rally ({num(r.duration, 0)} s), won by {L[r.winner] if isinstance(r.winner, str) else '–'}: the shape of the whole match in one exchange.",
        )
    )
    lost = R[R.winner == o].sort_values(f"dist_{u}", ascending=False).head(2)
    for r in lost.itertuples():
        out.append((r, f"Lost after {num(getattr(r, f'dist_{u}'), 0)} m of running: where did the length go loose?"))
    won = R[R.winner == u].sort_values(f"tshare_{u}", ascending=False).head(2)
    for r in won.itertuples():
        out.append((r, f"Won while holding the T {pct(getattr(r, f'tshare_{u}'))} of the rally: the model to repeat."))
    made_run = R[R.winner == u].sort_values(f"dist_{o}", ascending=False).head(1)
    for r in made_run.itertuples():
        out.append((r, f"{L[o]} was made to run {num(getattr(r, f'dist_{o}'), 0)} m and lost the point."))
    for r in R[R.note.fillna("").str.contains("let", regex=False)].itertuples():
        out.append((r, "Counted as a let (same player served twice from the same box)."))
    seen, uniq = set(), []
    for r, why in out:
        k = int(r.rally)
        if k not in seen:
            seen.add(k)
            uniq.append((r, why))
    return sorted(uniq, key=lambda x: int(x[0].rally))


def table_rallies(S, cfg):
    L = {p: cfg["players"][p]["label"] for p in PLAYERS}
    rows = []
    for r in S["rallies"]:
        srv = f"{L[r['server']]} ({r['box']})" if r.get("server") else "not seen"
        win = L[r["winner"]] if r.get("winner") else "–"
        note = r.get("note") or ""
        note = note.replace("serve inferred (relaxed)", "serve inferred").replace("let (same server, same box)", "let")
        rows.append(
            f"<tr><td>{r['rally']}</td><td>{mmss(r['t_start'])}</td><td>{r['duration']:.0f}</td><td>{srv}</td><td>{win}</td>"
            f"<td>{r['score_p1']}–{r['score_p2']}</td><td>{r['dist_p1']:.0f}</td><td>{r['dist_p2']:.0f}</td>"
            f"<td>{pct(r['tshare_p1'])}</td><td>{pct(r['tshare_p2'])}</td><td class='note'>{note}</td></tr>"
        )
    return (
        "<div class='scroll'><table><thead><tr><th>#</th><th>video time</th><th>length (s)</th><th>serve</th><th>won by</th><th>score after</th>"
        f"<th>{L['p1']} ran (m)</th><th>{L['p2']} ran (m)</th><th>{L['p1']} at T</th><th>{L['p2']} at T</th><th>note</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def build(cfg, S, out_dir):
    P, G, Rl = S["players"], S["game"], S["reliability"]
    L = {p: cfg["players"][p]["label"] for p in PLAYERS}
    u = cfg.get("user_player") or "p1"
    o = "p2" if u == "p1" else "p1"
    fs = G["final_score"]
    R = pd.DataFrame(S["rallies"])

    def F(n, alt, cap=""):
        return img_pair(out_dir, n, alt, cap)

    def stat(v, lab, sub=""):
        return f"<div class='stat'><div class='v'>{v}</div><div class='l'>{lab}</div>{f'<div class=s>{sub}</div>' if sub else ''}</div>"

    def row(lab, va, vb):
        return f"<tr><th>{lab}</th><td>{va}</td><td>{vb}</td></tr>"

    kinds = {"work": "Work on", "tactic": "Tactical", "strength": "Strength"}

    def tk_list(items):
        return "".join(
            f"<li class='tk {k}'><span class='kind'>{kinds[k]}</span><h3>{t}</h3><p>{e}</p><p class='act'><b>What to do:</b> {a}</p></li>"
            for _, k, t, e, a in items
        )

    if cfg.get("user_player"):
        # single coachee: one prioritised list from that player's perspective
        summary_html = f"<h2>Coach's summary for {L[u]}</h2><ol class='tks'>{tk_list(takeaways(S, cfg))}</ol>"
        moments_src = review_moments(S, cfg)
    else:
        # symmetric report: one section per player, then what the match as a whole shows
        parts = []
        for p in PLAYERS:
            items = player_findings(S, cfg, p)
            work = [i for i in items if i[1] != "strength"]
            good = [i for i in items if i[1] == "strength"]
            body = (
                tk_list(work)
                if work
                else "<li class='tk tactic'><h3>Nothing stands out as a weakness in the numbers</h3><p>The movement data does not single out a problem for this player; look at the shared observations below.</p></li>"
            )
            if good:
                body += tk_list(good)
            parts.append(f"<h2>{L[p]}: what to work on</h2><ol class='tks'>{body}</ol>")
        shared = match_findings(S, cfg)
        if shared:
            parts.append(f"<h2>What the match as a whole shows</h2><ol class='tks'>{tk_list(shared)}</ol>")
        summary_html = "".join(parts)
        moments_src = review_moments_neutral(S, cfg)

    moments = "".join(
        f"<li><b>{mmss(r.t_start)}</b> · rally {int(r.rally)} ({num(r.duration, 0)} s) — {why}</li>" for r, why in moments_src
    )

    def serve_rows(p):
        m = P[p]
        srv = R[R.server == p]
        by_box = {b: (int((srv[srv.box == b].winner == p).sum()), int(len(srv[srv.box == b]))) for b in ("left", "right")}
        return (m, by_box)

    (mu, bu), (mo, bo) = serve_rows(u), serve_rows(o)
    serve_table = (
        f"<table class='cmp'><thead><tr><th></th><th>{L[u]}</th><th>{L[o]}</th></tr></thead><tbody>"
        + row("Rallies served", mu["rallies_served"], mo["rallies_served"])
        + row(
            "Points won on serve",
            f"{mu['points_won_on_serve']} ({pct(mu['points_won_on_serve'] / max(mu['rallies_served'], 1))})",
            f"{mo['points_won_on_serve']} ({pct(mo['points_won_on_serve'] / max(mo['rallies_served'], 1))})",
        )
        + row(
            "…from the left box / right box",
            f"{bu['left'][0]} of {bu['left'][1]} / {bu['right'][0]} of {bu['right'][1]}",
            f"{bo['left'][0]} of {bo['left'][1]} / {bo['right'][0]} of {bo['right'][1]}",
        )
        + row("Rallies received", mu["rallies_received"], mo["rallies_received"])
        + row(
            "Points won on return",
            f"{mu['points_won_on_receive']} ({pct(mu['points_won_on_receive'] / max(mu['rallies_received'], 1))})",
            f"{mo['points_won_on_receive']} ({pct(mo['points_won_on_receive'] / max(mo['rallies_received'], 1))})",
        )
        + row("Longest run of points", mu["longest_run"], mo["longest_run"])
        + "</tbody></table>"
    )

    t_table = (
        f"<table class='cmp'><thead><tr><th></th><th>{L[u]}</th><th>{L[o]}</th></tr></thead><tbody>"
        + row("Rally time within 1 m of the T", pct(P[u]["t_share_1m"]), pct(P[o]["t_share_1m"]))
        + row("Rally time within 1.5 m of the T", pct(P[u]["t_share_1_5m"]), pct(P[o]["t_share_1_5m"]))
        + row("Closer to the T than the opponent", pct(P[u]["closer_to_t_share"]), pct(P[o]["closer_to_t_share"]))
        + row(
            "Median time to get back to the T", num(P[u]["recovery_to_t_median_s"], 1, " s"), num(P[o]["recovery_to_t_median_s"], 1, " s")
        )
        + row("In the back third while the other held the T", pct(P[u]["back_when_opp_at_t"]), pct(P[o]["back_when_opp_at_t"]))
        + row(
            "Front / middle / back third of the court",
            f"{pct(P[u]['front_share'])} / {pct(P[u]['mid_share'])} / {pct(P[u]['back_share'])}",
            f"{pct(P[o]['front_share'])} / {pct(P[o]['mid_share'])} / {pct(P[o]['back_share'])}",
        )
        + row("Backhand side (assumes right-handed)", pct(P[u]["backhand_side_share"]), pct(P[o]["backhand_side_share"]))
        + "</tbody></table>"
    )

    work_table = (
        f"<table class='cmp'><thead><tr><th></th><th>{L[u]}</th><th>{L[o]}</th></tr></thead><tbody>"
        + row("Distance run during rallies", num(P[u]["distance_rally_m"], 0, " m"), num(P[o]["distance_rally_m"], 0, " m"))
        + row("Per rally", num(P[u]["distance_per_rally_m"], 1, " m"), num(P[o]["distance_per_rally_m"], 1, " m"))
        + row(
            "Per rally won / lost",
            f"{num(P[u]['distance_won_rallies_m'], 1)} / {num(P[u]['distance_lost_rallies_m'], 1)} m",
            f"{num(P[o]['distance_won_rallies_m'], 1)} / {num(P[o]['distance_lost_rallies_m'], 1)} m",
        )
        + row(
            "Average / top speed",
            f"{num(P[u]['mean_speed'], 2)} / {num(P[u]['peak_speed'], 1)} m/s",
            f"{num(P[o]['mean_speed'], 2)} / {num(P[o]['peak_speed'], 1)} m/s",
        )
        + row("Sprints (over 3 m/s for half a second)", P[u]["bursts"], P[o]["bursts"])
        + row(
            "Average length of rallies won / lost",
            f"{num(P[u]['mean_rally_won_s'], 0)} / {num(P[u]['mean_rally_lost_s'], 0)} s",
            f"{num(P[o]['mean_rally_won_s'], 0)} / {num(P[o]['mean_rally_lost_s'], 0)} s",
        )
        + "</tbody></table>"
    )

    known = G.get("known_final_score")
    score_note = ""
    if known:
        score_note = f" The final score reported by the players was {known}."
    if max(fs.values()) < 15:
        starts_at_play = (R.t_start.min() - cfg.get("t_start", 0)) < 10
        why = (
            "so the clip does not contain the whole game"
            if starts_at_play
            else "so the recording most likely starts part-way through the game"
        )
        score_note += (
            f" The footage holds {G['n_points']} decided points, {G['n_lets']} lets and {G['n_unknown']} undecided rall{'y' if G['n_unknown'] == 1 else 'ies'}, "
            f"which is short of a game to 15, {why}. Read the score as the points played on camera."
        )

    html = f"""<meta charset="utf-8">
<title>{cfg.get("report_title", "Smashers Game Report")}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: one reading column; coach's summary before any chart; every chart carries a one-line caption; tables are the data view */
:root {{
  --bg:#f7f4ee; --paper:#fdfcf9; --ink:#191714; --ink-2:#5a554c; --line:#e3ded3; --shade:#efeae0;
  --court:#c8102e; --p1:#2a78d6; --p2:#eb6834; --good:#0ca30c;
  --font-display:"Barlow Condensed","Arial Narrow",sans-serif; --font-body:"Source Sans 3","Helvetica Neue",Arial,sans-serif; --font-mono:"IBM Plex Mono",Menlo,monospace;
  --show-light:block; --show-dark:none;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#1a1a19; --paper:#222220; --ink:#f3f1ea; --ink-2:#bdb9ae; --line:#3a3934; --shade:#2a2927; --court:#ff5a6b; --p1:#3987e5; --p2:#d95926; --good:#3fbf3f; --show-light:none; --show-dark:block; color-scheme:dark }} }}
:root[data-theme="dark"] {{ --bg:#1a1a19; --paper:#222220; --ink:#f3f1ea; --ink-2:#bdb9ae; --line:#3a3934; --shade:#2a2927; --court:#ff5a6b; --p1:#3987e5; --p2:#d95926; --good:#3fbf3f; --show-light:none; --show-dark:block; color-scheme:dark }}
body {{ background:var(--bg); color:var(--ink); font-family:var(--font-body); font-size:17px; line-height:1.5; margin:0; }}
.wrap {{ max-width:900px; margin:0 auto; padding-block:28px 64px; padding-inline:20px; }}
h1,h2,h3 {{ font-family:var(--font-display); text-wrap:balance; margin:0; line-height:1.05; }}
h1 {{ font-size:clamp(36px,6vw,56px); font-weight:700; letter-spacing:.01em; }}
h2 {{ font-size:30px; font-weight:700; margin-top:52px; padding-top:14px; border-top:2px solid var(--court); }}
h3 {{ font-size:22px; font-weight:600; }}
p {{ max-width:70ch; margin:10px 0; }}
.eyebrow {{ font-family:var(--font-mono); font-size:12px; letter-spacing:.12em; text-transform:uppercase; color:var(--ink-2); }}
.score {{ display:grid; grid-template-columns:1fr auto 1fr; gap:16px; align-items:center; margin-top:18px; padding:18px 20px; background:var(--paper); border:1px solid var(--line); }}
.score .p {{ font-family:var(--font-display); font-size:28px; font-weight:600; }}
.score .p small {{ display:block; font-family:var(--font-body); font-size:13px; color:var(--ink-2); font-weight:400; }}
.score .n {{ font-family:var(--font-display); font-size:64px; font-weight:700; font-variant-numeric:tabular-nums; }}
.score .right {{ text-align:right; }}
.dot {{ display:inline-block; width:12px; height:12px; border-radius:50%; margin-right:8px; vertical-align:middle; }}
.dot.p1 {{ background:var(--p1); }} .dot.p2 {{ background:var(--p2); }}
.lead {{ font-size:18px; }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:12px; margin-top:20px; }}
.stat {{ background:var(--paper); border:1px solid var(--line); padding:12px 14px; }}
.stat .v {{ font-family:var(--font-display); font-size:32px; font-weight:700; font-variant-numeric:tabular-nums; line-height:1; }}
.stat .l {{ font-size:13px; color:var(--ink-2); margin-top:6px; }}
.stat .s {{ font-size:12px; color:var(--ink-2); }}
ol.tks {{ list-style:none; padding:0; margin:18px 0 0; display:grid; gap:14px; }}
li.tk {{ background:var(--paper); border:1px solid var(--line); border-left:5px solid var(--court); padding:14px 18px; }}
li.tk.strength {{ border-left-color:var(--good); }} li.tk.tactic {{ border-left-color:var(--ink-2); }}
li.tk .kind {{ font-family:var(--font-mono); font-size:11px; letter-spacing:.1em; text-transform:uppercase; color:var(--ink-2); }}
li.tk h3 {{ margin:2px 0 6px; }} li.tk p {{ margin:6px 0; max-width:none; }} li.tk .act {{ color:var(--ink); }}
figure {{ margin:18px 0 6px; }}
figure img {{ width:100%; height:auto; display:block; border:1px solid var(--line); background:var(--paper); }}
figcaption {{ font-size:14px; color:var(--ink-2); margin-top:6px; max-width:80ch; }}
img.fig-light {{ display:var(--show-light); }} img.fig-dark {{ display:var(--show-dark); }}
table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; font-size:15px; }}
th,td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); vertical-align:top; }}
thead th {{ font-family:var(--font-mono); font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--ink-2); }}
tbody th {{ font-weight:400; color:var(--ink-2); width:52%; }}
.cmp {{ margin-top:14px; }} .cmp td {{ font-family:var(--font-mono); font-size:14px; }}
.cmp thead th:nth-child(2) {{ color:var(--{u}); }} .cmp thead th:nth-child(3) {{ color:var(--{o}); }}
.scroll {{ overflow-x:auto; margin-top:14px; }} .scroll table {{ min-width:820px; font-size:14px; }}
td.note {{ color:var(--ink-2); font-size:13px; }}
ul.moments {{ padding-left:20px; }} ul.moments li {{ margin:8px 0; max-width:80ch; }}
details {{ margin-top:16px; }} summary {{ cursor:pointer; font-weight:600; }}
.box {{ background:var(--shade); padding:16px 18px; margin-top:14px; font-size:15px; }} .box p {{ margin:6px 0; max-width:none; }}
dl {{ display:grid; grid-template-columns:max-content 1fr; gap:6px 16px; font-size:15px; margin:10px 0 0; }} dt {{ font-weight:600; }} dd {{ margin:0; }}
@media (max-width:560px) {{ .score {{ grid-template-columns:1fr; text-align:center; }} .score .right {{ text-align:center; }} .score .n {{ font-size:48px; }} dl {{ grid-template-columns:1fr; }} }}
@page {{ size: A4; margin: 14mm 14mm 16mm; }}
@media print {{ :root {{ --bg:#fff; --paper:#fff; --ink:#000; --ink-2:#444; --line:#bbb; --shade:#f2f2f2; --show-light:block; --show-dark:none; }} body {{ font-size:11.5px; }} .wrap {{ max-width:none; padding:0; }} h1 {{ font-size:34px; }} h2 {{ font-size:22px; margin-top:26px; break-after:avoid; }} h3 {{ font-size:15px; }} .lead {{ font-size:12px; }} .score .n {{ font-size:40px; }} .score .p {{ font-size:18px; }} .stat .v {{ font-size:22px; }} li.tk, figure, .stat, table {{ break-inside:avoid; }} figure img {{ max-height:118mm; width:auto; max-width:100%; margin:0 auto; }} .scroll {{ overflow:visible; }} .scroll table {{ min-width:0; font-size:9.5px; }} th,td {{ padding:4px 6px; }} summary {{ display:none; }} h2.pb {{ break-before:page; margin-top:0; }} }}
</style>
<div class="wrap">
<div class="eyebrow">Squash · game analysis from video · recorded {cfg.get("game_date", "")} · {
        mmss(G["game_duration_s"])
    } of play on camera</div>
<h1>{L["p1"]} vs {L["p2"]}</h1>
<div class="score">
  <div class="p"><span class="dot p1"></span>{L["p1"]}<small>{cfg["players"]["p1"].get("kit_desc", "p1")}</small></div>
  <div class="n">{fs["p1"]}&thinsp;–&thinsp;{fs["p2"]}</div>
  <div class="p right">{L["p2"]}<span class="dot p2" style="margin:0 0 0 8px"></span><small>{
        cfg["players"]["p2"].get("kit_desc", "p2")
    }</small></div>
</div>
<p class="lead">Every number here comes from tracking both players' feet ten times a second in the video and mapping them onto the court. The ball is not tracked, so shot choice and errors are not visible; positioning, movement and rally structure are. Video times in this report match the tracked video and the original clip.</p>
<p>The score is reconstructed from who served: in point-a-rally scoring the winner of a rally serves the next one.{score_note}</p>

{summary_html}

<h2>The game at a glance</h2>
<div class="stats">
  {stat(G["n_rallies"], "rallies on camera", f"{G['n_points']} points, {G['n_lets']} lets")}
  {stat(num(G["mean_rally_s"], 0, " s"), "average rally", f"median {num(G['median_rally_s'], 0)} s")}
  {
        stat(
            num(G["longest_rally_s"], 0, " s"),
            "longest rally",
            f"rally {G['longest_rally_no']} at {mmss(R.loc[R.rally == G['longest_rally_no'], 't_start'].iloc[0])}",
        )
    }
  {
        stat(
            num(G["mean_rest_s"], 0, " s"),
            "rest between rallies",
            f"play : rest = 1 : {num(1 / G['work_rest_ratio'], 1) if G['work_rest_ratio'] else '–'}",
        )
    }
  {stat(num(P[u]["distance_rally_m"], 0, " m"), f"run by {L[u]}", f"{pct(P[u]['t_share_1_5m'])} of rally time near the T")}
  {stat(num(P[o]["distance_rally_m"], 0, " m"), f"run by {L[o]}", f"{pct(P[o]['t_share_1_5m'])} of rally time near the T")}
</div>

<h2 class="pb">Where the game was played</h2>
{
        F(
            "heatmaps",
            "Heatmaps of each player position during rallies",
            "Front wall at the top of each court. Darker means more rally time on that spot; the dot on the short line is the T. Rest periods between rallies are excluded.",
        )
    }
{
        F(
            "zones",
            "Share of rally time in each of six court zones per player",
            "The court split into thirds (front: within 3 m of the front wall; back: behind the service boxes) and halves (left/right as seen from the back wall).",
        )
    }

<h2>T control and recovery</h2>
<p>The T is where the short line meets the half-court line. The player who gets back to it after each shot sees the whole court and takes the ball early. "Recovery" is the time from leaving a 1.5 m circle around the T to being back inside it.</p>
{t_table}
{
        F(
            "t_timeline",
            "Distance from the T over the whole game for both players",
            "Each line is a player’s distance from the T. Grey bands are rallies (numbered); the dashed line is the 1.5 m circle. Dips to zero are visits to the T; long stretches above 2 m are time spent retrieving.",
        )
    }

<h2>Work rate</h2>
{work_table}
{F("distance", "Distance covered per rally by each player", "Metres run by each player in every rally, in match order.")}
{
        F(
            "speed",
            "Speed distributions during rallies",
            "How fast each player was moving during rallies. A fatter right-hand tail means more sprinting; a taller peak near zero means more standing.",
        )
    }

<h2>Serve and return</h2>
{serve_table}

<h2 class="pb">Rally by rally</h2>
<p><b>Moments worth reviewing on the video</b> (times are minutes:seconds in the clip):</p>
<ul class="moments">{moments}</ul>
{
        F(
            "rallies",
            "Rally durations coloured by the winner",
            "Length of every rally, coloured by who won it. Grey bars are lets or rallies whose winner could not be seen.",
        )
    }
{F("score", "Reconstructed score progression", "Points as the game went on. A flat line is a run of points for the other player.")}
{table_rallies(S, cfg)}
<p class="eyebrow" style="margin-top:8px">"at T" = share of the rally spent within 1.5 m of the T · serve shows the service box used · "serve inferred" = the serve happened too quickly to be seen standing still, so it was inferred from the positions</p>
<details open><summary>Movement path of every rally</summary>
{
        F(
            "trajectories",
            "Movement paths for every rally",
            "Each small court is one rally, front wall at the top. Lines are the paths of the two players; the dot is where each stood at the serve.",
        )
    }
</details>

<h2 class="pb">How to read this report</h2>
<dl>
<dt>T</dt><dd>The junction of the short line and the half-court line, 5.44 m from the front wall in the middle of the court.</dd>
<dt>Service box</dt><dd>The 1.6 m square on each side behind the short line; the server must have one foot in it.</dd>
<dt>Point-a-rally</dt><dd>Every rally scores a point for its winner, who then serves. That rule is what lets the score be rebuilt from the video.</dd>
<dt>Let</dt><dd>A rally replayed with the same server from the same box. Lets and strokes decided between the players cannot be told apart on the video.</dd>
<dt>Front / middle / back third</dt><dd>Within 3 m of the front wall; between 3 m and the back of the service boxes; behind the service boxes.</dd>
</dl>
<div class="box">
<p><b>What the analysis can see.</b> Where each player's feet were, {Rl.get("samples_per_s", "about ten")} times a second, {
        Rl.get(
            "accuracy_note",
            "to within about 10 cm across the court and 20–40 cm front-to-back (less precise towards the back wall, where the camera is)",
        )
    }. Rally starts are taken from the server standing in the box; rally ends from both players stopping.</p>
<p><b>What it cannot see.</b> The ball, shot selection, errors, and referee decisions. A player standing in the last metre in front of the camera has their feet out of frame and is placed at the frame edge ({
        L["p1"]
    } {pct(Rl["feet_cut_share"]["p1"])} of rally time, {L["p2"]} {
        pct(Rl["feet_cut_share"]["p2"])
    }), so back-corner running is slightly under-counted.</p>
<p><b>Confidence.</b> Player identity was checked by hand on the longest tracks and is right in the sampled frames ({
        Rl["identity_disagreements"]
    } frames flagged as doubtful). The serve-box alternation rule, which must hold whenever a server keeps serving, held in {
        G["serve_box_alternation_ok"]
    } cases, so the rally structure and the score sequence are trustworthy.</p>
</div>
</div>
"""
    return html


def run(cfg=None, force=True):
    cfg = cfg or load_config()
    out_dir = Path(cfg["output_dir"])
    S = json.load(open(out_dir / "summary.json"))
    html = build(cfg, S, out_dir)
    (out_dir / "report.html").write_text(html)
    print(f"report written ({len(html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    run()


CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def to_pdf(cfg=None):
    """Print output/report.html to output/report.pdf with headless Chrome (A4, print stylesheet)."""
    import shutil
    import subprocess

    cfg = cfg or load_config()
    out_dir = Path(cfg["output_dir"]).resolve()
    chrome = CHROME if Path(CHROME).exists() else shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        print("no Chrome/Chromium found: skipping PDF")
        return None
    pdf = out_dir / "report.pdf"
    subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--virtual-time-budget=15000",
            f"--print-to-pdf={pdf}",
            f"file://{out_dir / 'report.html'}",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"pdf written: {pdf} ({pdf.stat().st_size / 1e6:.1f} MB)")
    return pdf
