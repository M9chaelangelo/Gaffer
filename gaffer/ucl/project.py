"""Project one UCL matchday: a distribution of points per player.

Points are simulated, not averaged, for the same reason they are on the FPL
side: the captaincy and armband decisions are decisions about a tail, and a
mean cannot answer them. Team-mates share one scoreline draw per fixture, so
a clean sheet is correlated across a back four the way it is in life.

Inputs, in order of preference:

  1. devigged bookmaker odds from `config/ucl.yaml`, when they exist, and
  2. the feed's own strength model otherwise (see `strength.py`).

Which was used is reported per fixture, because a plan built on a model of a
model should say so.
"""
from __future__ import annotations

import re

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from . import strength
from .rules import (ASSIST, CLEAN_SHEET, CONCEDED_PER_POINT, GOAL,
                    PLAYER_OF_THE_MATCH, RECOVERIES_PER_POINT, SAVES_PER_POINT)

N = 60_000
DC_RHO = -0.05
SEED = 11

# One matchday of history is not a rate. Shrink the observed per-90 toward
# what a player of this price and position typically does, with the weight on
# the observation rising with minutes played.
PRIOR90 = {   # goals, assists, recoveries per 90
    "GK":  (0.00, 0.01, 1.2),
    "DEF": (0.05, 0.05, 6.5),
    "MID": (0.12, 0.14, 5.5),
    "FWD": (0.35, 0.12, 2.2),
}
PRIOR_MINUTES = 180.0          # observation reaches half weight at two games
TYPICAL_PRICE = {"GK": 5.0, "DEF": 5.5, "MID": 7.0, "FWD": 8.0}
TYPICAL_TEAM_GOALS = 1.4       # the denominator a player's share is taken of

# UEFA writes availability as prose, not a flag — and the prose covers two
# very different states that the first version of this collapsed into one.
# A suspension is a certainty; a knock is a coin toss. Olise is suspended for
# matchday three, and a model that gives him a one-in-ten chance of playing
# also gives him a tenth of a haul, which is enough to keep him in an eleven.
OUT = ("suspended", "injured", "not in contention", "out of contention",
       "unavailable", "ruled out")
DOUBT = ("doubt", "knock", "fitness test", "assessed", "late test")
DOUBT_START = 0.25
FULL_GIVEN_START = 0.80        # P(60+ | started)
PART_MINUTES = 0.45            # minutes fraction for a cameo
YELLOW_RATE = 0.08


def dixon_coles(lh, la, maxg=9):
    h = poisson.pmf(np.arange(maxg + 1), lh)
    a = poisson.pmf(np.arange(maxg + 1), la)
    m = np.outer(h, a)
    m[0, 0] *= 1 - lh * la * DC_RHO
    m[0, 1] *= 1 + lh * DC_RHO
    m[1, 0] *= 1 + la * DC_RHO
    m[1, 1] *= 1 - DC_RHO
    return m / m.sum()


def lambdas_from_odds(ph, pd_, pa):
    """Goal rates whose Dixon-Coles scoreline matrix reproduces the book."""
    def loss(x):
        m = dixon_coles(*np.exp(x))
        return ((np.tril(m, -1).sum() - ph) ** 2 + (np.trace(m) - pd_) ** 2
                + (np.triu(m, 1).sum() - pa) ** 2)
    r = minimize(loss, np.log([1.3, 1.3]), method="Nelder-Mead",
                 options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 4000})
    return tuple(float(v) for v in np.exp(r.x))


def _odds_for(odds, home, away):
    """A configured 1X2 for this fixture, oriented home-first, or None."""
    for key, v in (odds or {}).items():
        parts = [s.strip().lower() for s in
                 re.split(r"\s+v(?:s)?\s+|\|", str(key)) if s.strip()]
        if len(parts) != 2:
            continue
        h, a = parts
        if h in home.lower() and a in away.lower():
            return tuple(float(x) for x in v)
        if h in away.lower() and a in home.lower():
            ph, pdw, pa = (float(x) for x in v)
            return (pa, pdw, ph)
    return None


def rates(p):
    """Shrunk per-90 goals, assists and recoveries."""
    g0, a0, r0 = PRIOR90[p["pos"]]
    scale = float(np.clip(p["price"] / TYPICAL_PRICE[p["pos"]], 0.5, 1.8))
    g0, a0 = g0 * scale, a0 * scale
    w = p["minutes"] / (p["minutes"] + PRIOR_MINUTES)
    return (w * p["goals90"] + (1 - w) * g0,
            w * p["assists90"] + (1 - w) * a0,
            w * p["recoveries90"] + (1 - w) * r0)


def start_prob(p):
    """P(starts). Zero means zero: a suspended player is not a long shot."""
    if p.get("forced_out"):
        return 0.0
    words = f"{p.get('trained') or ''} {p.get('status') or ''}".lower()
    if any(k in words for k in OUT):
        return 0.0
    if any(k in words for k in DOUBT):
        return DOUBT_START
    t = (p.get("trained") or "").lower()
    base = 0.85 if "contention" in t else 0.70
    if p["team_played"] and p["minutes"] >= 60 * p["team_played"] * 0.7:
        base = min(base + 0.08, 0.93)
    elif p["team_played"] and p["minutes"] == 0:
        base = max(base - 0.35, 0.10)
    return base


def fixture_rates(fixtures, players, odds=None):
    """Goal rates per fixture, with the provenance of each."""
    rate, mu = strength.ratings(players)
    out = []
    for f in fixtures:
        book = _odds_for(odds, f["home"], f["away"])
        if book:
            lh, la = lambdas_from_odds(*book)
            src = "odds"
        else:
            lh, la = strength.lambdas(rate, f["home"], f["away"], mu)
            src = "model"
        out.append(dict(f, lh=lh, la=la, source=src))
    return out, rate, mu


def simulate(players, fixtures, odds=None, n=N, seed=SEED):
    """(n, len(players)) simulated points, plus the fixture rates used.

    Players whose club has no fixture this matchday score nothing — in UCL
    that is a real and common state, not an error.
    """
    rng = np.random.default_rng(seed)
    fx, _rate, _mu = fixture_rates(fixtures, players, odds)

    draws, by_team = {}, {}
    for f in fx:
        key = (f["home"], f["away"])
        draws[key] = (rng.poisson(f["lh"], n), rng.poisson(f["la"], n))
        by_team[f["home"]] = (key, True, f)
        by_team[f["away"]] = (key, False, f)

    out = np.zeros((n, len(players)))
    for j, p in enumerate(players):
        slot = by_team.get(p["team"])
        if slot is None:
            continue                      # no fixture this matchday
        key, at_home, _f = slot
        gh, ga = draws[key]
        scored, conceded = (gh, ga) if at_home else (ga, gh)

        g90, a90, r90 = rates(p)
        played = rng.random(n) < start_prob(p)
        full = played & (rng.random(n) < FULL_GIVEN_START)
        mins = np.where(full, 1.0, np.where(played, PART_MINUTES, 0.0))

        gshare = float(np.clip(g90 / TYPICAL_TEAM_GOALS, 0, 0.6))
        ashare = float(np.clip(a90 / TYPICAL_TEAM_GOALS, 0, 0.6))

        pts = played.astype(float) + full.astype(float)
        mygoals = rng.binomial(scored, gshare) * played
        myass = rng.binomial(scored, ashare) * played
        pts += mygoals * GOAL[p["pos"]] + myass * ASSIST

        cs = (conceded == 0) & full
        pts += cs * CLEAN_SHEET[p["pos"]]
        if p["pos"] in ("GK", "DEF"):
            pts -= (conceded // CONCEDED_PER_POINT) * full
        if p["pos"] == "GK":
            pts += rng.poisson(np.maximum(p["saves90"] * mins, 1e-9)) // SAVES_PER_POINT

        # No threshold, unlike FPL's defensive contribution: every third
        # recovery pays from the first, which is why a holding midfielder is
        # a floor here and a lottery ticket there.
        pts += rng.poisson(np.maximum(r90 * mins, 1e-9)) // RECOVERIES_PER_POINT
        pts -= (rng.random(n) < YELLOW_RATE * played)
        motm = (rng.random(n) < np.clip(0.04 + 0.18 * mygoals + 0.08 * myass,
                                        0, 0.6)) & played
        pts += motm * PLAYER_OF_THE_MATCH
        out[:, j] = pts
    return out, fx


def summarise(players, samples):
    """One row per player: mean, floor, ceiling and the shape between."""
    rows = []
    for j, p in enumerate(players):
        s = samples[:, j]
        rows.append(dict(
            p,
            ep=float(s.mean()),
            sd=float(s.std()),
            p_blank=float((s <= 0).mean()),
            p_haul=float((s >= 10).mean()),
            p90=float(np.percentile(s, 90)),
            value=float(s.mean() / p["price"]) if p["price"] else 0.0,
        ))
    return rows
