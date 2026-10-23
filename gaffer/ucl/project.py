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
# How much of a player's own record to believe. At 180 minutes — half
# weight after two games — Barkley's two goals in 106 minutes survived
# shrinkage as 0.69 goals a game and the model projected him at 14 points,
# ahead of Kane. Goals are a rare count; two of them are not a rate. 450
# puts half weight at five full matches, which is still generous.
PRIOR_MINUTES = 450.0
TYPICAL_PRICE = {"GK": 5.0, "DEF": 5.5, "MID": 7.0, "FWD": 8.0}
TYPICAL_TEAM_GOALS = 1.4       # fallback when a club's own rate is unknown
MIN_TEAM_GOALS = 0.8           # no club's baseline is lower than this

# A player's share of his side's goals. The cap was 0.6, which let one man
# take three fifths of a team expected to score three times — nobody in
# football does that. Haaland takes about a third of Manchester City's.
MAX_GOAL_SHARE = 0.35
MAX_ASSIST_SHARE = 0.30

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

# --- how much of the available football a player has actually played --------
#
# The previous version of start_prob read the prose, then nudged a flat base
# up or down by eight points. Measured across the 701 players with a matchday
# three fixture it returned 0.93 for almost all of them: Haaland, who has
# played every minute, and Barkley, who has played 106 of 180 and started once,
# and Kvaratskhelia, who has played 24, were all 0.93. A start probability that
# does not distinguish those three is not a start probability.
#
# UEFA's feed gives cumulative minutes and nothing per match, so the usable
# signal is the SHARE of his club's available minutes a player has taken.
# Inverting the share into a start probability needs two assumptions, both
# measurable later and neither doing much work at the extremes:
STARTER_SHARE = 0.88    # of a match, for a man who starts it
CAMEO_SHARE = 0.25      # of a match, for a man who comes off the bench
BENCH_USE = 0.60        # P(gets on at all | does not start)
# Two matches is not a rate, so shrink toward the pool. Light on purpose: a
# man who has played 180 of 180 minutes should not be dragged to the middle.
PRIOR_MATCHES = 0.5
PRIOR_START = 0.60
# Nobody is certain and nobody is impossible while his name is in the squad.
MIN_START, MAX_START = 0.03, 0.95


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
    """A configured 1X2 for this fixture, oriented home-first, or None.

    Accepts either a (home, away) tuple key — what `odds.match_fixtures`
    produces from a live feed — or the "Home v Away" string a human writes
    in config.
    """
    if isinstance(odds, dict):
        direct = odds.get((home, away))
        if direct:
            return tuple(float(x) for x in direct)
        flipped = odds.get((away, home))
        if flipped:
            a, d, h = (float(x) for x in flipped)
            return (h, d, a)
    return _odds_for_named(odds, home, away)


def _named(mapping, home, away):
    """(value, was it written the other way round) for "Home v Away" keys."""
    for key, v in (mapping or {}).items():
        parts = [s.strip().lower() for s in
                 re.split(r"\s+v(?:s)?\s+|\|", str(key)) if s.strip()]
        if len(parts) != 2:
            continue
        h, a = parts
        if h in home.lower() and a in away.lower():
            return v, False
        if h in away.lower() and a in home.lower():
            return v, True
    return None, False


def _odds_for_named(odds, home, away):
    """A configured 1X2 for this fixture, oriented home-first, or None."""
    v, flipped = _named(odds, home, away)
    if v is None:
        return None
    ph, pdw, pa = (float(x) for x in v)
    return (pa, pdw, ph) if flipped else (ph, pdw, pa)


def market_xg(table, home, away):
    """Expected goals typed straight in, oriented home-first, or None.

    The published market cards — thefantasytool, FPL Schaden and the rest —
    print the devigged expected goals for every fixture a day before the
    deadline. That is the same quantity `lambdas_from_odds` works back to
    from a 1X2, arrived at from more markets than three, so when someone has
    gone to the trouble of typing it in it wins.

    Written in config as `market_xg: {"Man City v AEK Athens": [3.3, 0.6]}`,
    home side first. A fixture written the other way round is turned around
    here rather than being silently wrong.
    """
    v, flipped = _named(table, home, away)
    if v is None:
        return None
    a, b = (float(x) for x in v)
    return (b, a) if flipped else (a, b)


def unmatched_xg(table, fixtures):
    """Keys in the table that match no fixture. Typos must not be silent."""
    return [k for k in (table or {})
            if not any(market_xg({k: v}, f["home"], f["away"])
                       for v in [table[k]] for f in fixtures)]


def rates(p):
    """Shrunk per-90 goals, assists and recoveries."""
    g0, a0, r0 = PRIOR90[p["pos"]]
    scale = float(np.clip(p["price"] / TYPICAL_PRICE[p["pos"]], 0.5, 1.8))
    g0, a0 = g0 * scale, a0 * scale
    w = p["minutes"] / (p["minutes"] + PRIOR_MINUTES)
    return (w * p["goals90"] + (1 - w) * g0,
            w * p["assists90"] + (1 - w) * a0,
            w * p["recoveries90"] + (1 - w) * r0)


def minutes_share(p):
    """Fraction of his club's available minutes this player has taken."""
    played = int(p.get("matches") or p.get("team_played") or 0)
    if not played:
        return None
    return min(1.0, p["minutes"] / (90.0 * played))


def start_prob(p):
    """P(starts). Zero means zero: a suspended player is not a long shot.

    Order matters. The prose is about THIS matchday and the minutes are about
    his role, so a suspension or an injury overrides however much football he
    has played; an availability flag scales what is left.
    """
    if p.get("forced_out"):
        return 0.0
    words = f"{p.get('trained') or ''} {p.get('status') or ''}".lower()
    if any(k in words for k in OUT):
        return 0.0
    # An outside flag — FPL's, for the English clubs — beats UEFA's own prose,
    # which lags a day or more. On 19 Oct 2026 FPL had Haaland at a 75% chance
    # of playing four hours before UEFA still called him "in contention".
    chance = p.get("availability")
    if any(k in words for k in DOUBT):
        return DOUBT_START if chance is None else min(DOUBT_START, chance)

    # Per-match minutes where the history log has managed to difference them —
    # see history.py. Strictly better than the share below, because it can tell
    # a man who starts and is replaced on sixty from a man who starts one game
    # and watches the next; the share reads both as 0.67.
    log = p.get("match_minutes")
    if log:
        starts = sum(1.0 for m in log if m >= 45.0)
        base = ((starts + PRIOR_MATCHES * PRIOR_START)
                / (len(log) + PRIOR_MATCHES))
        base = min(MAX_START, max(MIN_START, base))
        return base if chance is None else base * chance

    share = minutes_share(p)
    if share is None:
        # Nothing played yet by anyone at his club: all we have is the prose.
        return 0.85 if "contention" in words else 0.70
    played = int(p.get("matches") or p.get("team_played") or 0)
    floor = CAMEO_SHARE * BENCH_USE
    raw = (share - floor) / (STARTER_SHARE - floor)
    raw = min(1.0, max(0.0, raw))
    base = ((played * raw + PRIOR_MATCHES * PRIOR_START)
            / (played + PRIOR_MATCHES))
    base = min(MAX_START, max(MIN_START, base))
    return base if chance is None else base * chance


def team_baselines(rate, mu):
    """Each club's own goals-per-game, the denominator for a player's share.

    Taking the share against a league-wide 1.4 was the second half of the
    Barkley bug: a Villa player accumulated his record while Villa scored
    at Villa's rate, so dividing by someone else's rate and then
    multiplying by Villa's expected goals counts the same strength twice.
    """
    out = {}
    for name, t in (rate or {}).items():
        out[name] = max(float(t.get("attack_goals") or mu), MIN_TEAM_GOALS)
    return out


def fixture_rates(fixtures, players, odds=None, xg=None):
    """Goal rates per fixture, with the provenance of each.

    Three sources, best first: expected goals typed in from a published
    market card, a devigged 1X2, and the strength model. Measured beats
    modelled, and every fixture line says which it got — scored against the
    market, the model is at 0.33 goals mean error, so the difference is not
    cosmetic.
    """
    rate, mu = strength.ratings(players)
    out = []
    for f in fixtures:
        direct = market_xg(xg, f["home"], f["away"])
        book = None if direct else _odds_for(odds, f["home"], f["away"])
        if direct:
            lh, la = direct
            src = "market xg"
        elif book:
            lh, la = lambdas_from_odds(*book)
            src = "odds"
        else:
            lh, la = strength.lambdas(rate, f["home"], f["away"], mu)
            src = "model"
        out.append(dict(f, lh=lh, la=la, source=src))
    return out, rate, mu


def simulate(players, fixtures, odds=None, n=N, seed=SEED,
             xg=None):
    """(n, len(players)) simulated points, plus the fixture rates used.

    Players whose club has no fixture this matchday score nothing — in UCL
    that is a real and common state, not an error.
    """
    rng = np.random.default_rng(seed)
    fx, rate, mu = fixture_rates(fixtures, players, odds, xg)
    baseline = team_baselines(rate, mu)

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

        own = baseline.get(p["team"], TYPICAL_TEAM_GOALS)
        gshare = float(np.clip(g90 / own, 0, MAX_GOAL_SHARE))
        ashare = float(np.clip(a90 / own, 0, MAX_ASSIST_SHARE))

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
