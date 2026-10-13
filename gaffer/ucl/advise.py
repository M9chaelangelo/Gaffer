"""Advise on one UCL matchday: captain, eleven, and the one transfer.

Reads the committed snapshot, so it runs without network.

Two things make UCL different from FPL and both are modelled here:

  - Ball recoveries pay 1 per 3 with no threshold, so a holding midfielder is
    a floor rather than a lottery ticket. The feed gives `bR`, so this is a
    measured rate rather than a guess.
  - There is no vice-captain, but the armband may be moved mid-matchday to a
    player who has not yet kicked off. Captaining someone who plays on the
    first day therefore buys a free second swing, and that option is priced
    below rather than asserted.

Team goal rates come from devigged bookmaker odds — measured. Player shares
come from this season's own counting stats, shrunk hard toward a price-and-
position prior, because one matchday of history is not a rate.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from .rules import (ASSIST, CLEAN_SHEET, CONCEDED_PER_POINT, GOAL,
                    PLAYER_OF_THE_MATCH, RECOVERIES_PER_POINT, SAVES_PER_POINT)

RNG = np.random.default_rng(11)
N = 100_000
DC_RHO = -0.05

# Devigged 1X2 from the books, by the two clubs' short names in UEFA's feed.
# Measured input; update before each matchday.
ODDS = {
    ("Viking", "Bayern München"): (0.061, 0.102, 0.836),
    ("Galatasaray", "Barcelona"): (0.122, 0.166, 0.713),
    ("Atlético", "Man United"): (0.506, 0.233, 0.261),
    ("Man City", "Paris"): (0.439, 0.239, 0.322),
    ("Bodø/Glimt", "Dortmund"): (0.300, 0.232, 0.468),
    ("Slovan Bratislava", "Stuttgart"): (0.160, 0.220, 0.620),
    ("Lens", "Sporting CP"): (0.400, 0.280, 0.320),
}
DEFAULT_ODDS = (0.40, 0.26, 0.34)      # a neutral tie when the book is unknown


def dc(lh, la, maxg=9):
    h = poisson.pmf(np.arange(maxg + 1), lh)
    a = poisson.pmf(np.arange(maxg + 1), la)
    m = np.outer(h, a)
    m[0, 0] *= 1 - lh * la * DC_RHO
    m[0, 1] *= 1 + lh * DC_RHO
    m[1, 0] *= 1 + la * DC_RHO
    m[1, 1] *= 1 - DC_RHO
    return m / m.sum()


def lambdas(ph, pd_, pa):
    def loss(x):
        m = dc(*np.exp(x))
        return ((np.tril(m, -1).sum() - ph) ** 2 + (np.trace(m) - pd_) ** 2
                + (np.triu(m, 1).sum() - pa) ** 2)
    r = minimize(loss, np.log([1.3, 1.3]), method="Nelder-Mead",
                 options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 4000})
    return tuple(np.exp(r.x))


def match_odds(home, away):
    for (h, a), v in ODDS.items():
        if h.lower() in home.lower() and a.lower() in away.lower():
            return v
        if h.lower() in away.lower() and a.lower() in home.lower():
            ph, pdw, pa = v
            return (pa, pdw, ph)
    return DEFAULT_ODDS


# --- player rates -----------------------------------------------------------
# One matchday of history is not a rate. Shrink the observed per-90 toward what
# a player of this price and position typically does, with the weight on the
# observation rising with minutes played.
PRIOR90 = {   # goals, assists, recoveries per 90 for a mid-priced player
    "GK":  (0.00, 0.01, 1.2),
    "DEF": (0.05, 0.05, 6.5),
    "MID": (0.12, 0.14, 5.5),
    "FWD": (0.35, 0.12, 2.2),
}
PRIOR_MINUTES = 180.0      # observation reaches half weight at two full games


def rates(p):
    g0, a0, r0 = PRIOR90[p["pos"]]
    # price carries UEFA's own view of a player; scale the attacking prior by it
    scale = np.clip(p["price"] / {"GK": 5.0, "DEF": 5.5,
                                  "MID": 7.0, "FWD": 8.0}[p["pos"]], 0.5, 1.8)
    g0, a0 = g0 * scale, a0 * scale
    w = p["minutes"] / (p["minutes"] + PRIOR_MINUTES)
    return (w * p["goals90"] + (1 - w) * g0,
            w * p["assists90"] + (1 - w) * a0,
            w * p["recoveries90"] + (1 - w) * r0)


UNAVAILABLE = ("not in contention", "injured", "suspended", "doubt")


def start_prob(p):
    t = (p["trained"] or "").lower()
    if any(k in t for k in UNAVAILABLE):
        return 0.10
    base = 0.85 if "contention" in t else 0.70
    if p["team_played"] and p["minutes"] >= 60:
        base = min(base + 0.08, 0.93)
    elif p["team_played"] and p["minutes"] == 0:
        base = max(base - 0.35, 0.10)
    return base


def simulate(squad, fixtures_by_team):
    """(N, len(squad)) points, with team-mates sharing one scoreline draw."""
    goals, lam = {}, {}
    for team, fx in fixtures_by_team.items():
        key = tuple(sorted((team, fx["opponent"])))
        if key in goals:
            continue
        home, away = (team, fx["opponent"]) if fx["home"] else (fx["opponent"], team)
        lh, la = lambdas(*match_odds(home, away))
        lam[key] = (home, away, lh, la)
        goals[key] = (RNG.poisson(lh, N), RNG.poisson(la, N))

    out = np.zeros((N, len(squad)))
    for j, p in enumerate(squad):
        fx = fixtures_by_team[p["team"]]
        key = tuple(sorted((p["team"], fx["opponent"])))
        gh, ga = goals[key]
        home, away, _, _ = lam[key]
        scored, conceded = (gh, ga) if p["team"] == home else (ga, gh)

        g90, a90, r90 = rates(p)
        played = RNG.random(N) < start_prob(p)
        full = played & (RNG.random(N) < 0.80)
        mins = np.where(full, 1.0, np.where(played, 0.45, 0.0))

        # a player's share of his side's goals, from his rate against a typical
        # team rate of ~1.4 goals a game
        gshare = np.clip(g90 / 1.4, 0, 0.6)
        ashare = np.clip(a90 / 1.4, 0, 0.6)

        pts = np.where(played, 1, 0) + np.where(full, 1, 0)
        mygoals = RNG.binomial(scored, gshare) * played
        myass = RNG.binomial(scored, ashare) * played
        pts = pts + mygoals * GOAL[p["pos"]] + myass * ASSIST

        cs = (conceded == 0) & full
        pts = pts + cs * CLEAN_SHEET[p["pos"]]
        if p["pos"] in ("GK", "DEF"):
            pts = pts - (conceded // CONCEDED_PER_POINT) * full
        if p["pos"] == "GK":
            pts = pts + RNG.poisson(np.maximum(p["saves90"] * mins, 1e-9)) // SAVES_PER_POINT

        pts = pts + RNG.poisson(np.maximum(r90 * mins, 1e-9)) // RECOVERIES_PER_POINT
        pts = pts - (RNG.random(N) < 0.08 * played)
        motm = (RNG.random(N) < np.clip(0.04 + 0.18 * mygoals + 0.08 * myass, 0, 0.6)) & played
        pts = pts + motm * PLAYER_OF_THE_MATCH
        out[:, j] = pts
    return out, lam
