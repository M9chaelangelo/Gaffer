"""Team goal rates for a UCL matchday, from the feed alone.

The FPL side of this engine gets its goal rates from devigged bookmaker odds,
which are the best forecast anyone publishes. For UCL we usually cannot reach
a book from the runner, and by matchday three a team has played twice, so the
observed rate is noise. This module makes a rate out of what the feed does
carry, and says so plainly rather than pretending it is odds.

Two measured inputs:

  - what each team has actually scored and conceded so far, and
  - what UEFA priced its squad at, which is UEFA's own view of the side.

The price prior is not asserted. The exponent linking squad value to goals is
*fitted* on this season's own scoring across all 36 teams, so if value turns
out not to predict goals the fit flattens and the prior does nothing.

Measured odds always win: pass them in `config/ucl.yaml` and `project` uses
those instead of anything here.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

# A team has played twice by matchday three, and two Poisson draws are not
# a rate. At PRIOR_GAMES = 2 the observation carried half the weight and
# six of eighteen fixtures came out at 3.6 to 4.0 expected goals, which is
# not a Champions League scoreline. Six keeps two matches at a quarter —
# standard shrinkage for a count this sparse — and should fall as the
# league phase goes on. Judgement, not measurement: the ledger is what will
# settle it.
PRIOR_GAMES = 6.0

# Home advantage in the league phase, as a multiplier on the home side's goal
# rate (and its reciprocal away). This one is an assumption, not a fit: the
# feed carries season totals with no home/away split, so there is nothing in
# it to estimate from. 1.10 is the long-run European club-competition figure.
HOME = 1.10

# Clamps. A fitted exponent on two matchdays can run away; these bound the
# prior to a factor of roughly two either side of the mean team.
# Attack and defence multiply, so a factor-of-two clamp on each permits a
# factor of four on the fixture. The best side in Europe scores perhaps
# half as much again as the average one.
MAX_RATIO = 1.55
MIN_RATIO = 0.65

# And a clamp on the rate itself. The ratio clamp bounds a team relative to
# the league; it does nothing if the league mean is wrong, which it will be
# if the feed's counting stats arrive in a shape this did not expect. Four
# goals is well beyond any Champions League fixture's true rate.
MAX_LAMBDA = 3.25
MIN_LAMBDA = 0.2


def aggregate(players):
    """Per-team goals for, goals against, matches played and squad value.

    Goals against come from the goalkeepers only. Every outfield row also
    carries `gC`, so summing the column over a whole squad counts each goal
    conceded once per player on the pitch — about elevenfold.
    """
    teams = {}
    for p in players:
        t = teams.setdefault(p["team"], {
            "team": p["team"], "team_id": p["team_id"],
            "scored": 0.0, "conceded": 0.0, "played": 0, "value": 0.0,
            "prices": []})
        t["scored"] += p["goals"]
        if p["pos"] == "GK":
            t["conceded"] += p["conceded"]
        # `matches`, not UEFA's `teamPlayed` — see fetch._attach_matches.
        t["played"] = max(t["played"], int(p.get("matches")
                                           or p.get("team_played") or 0))
        t["prices"].append(p["price"])
    for t in teams.values():
        # The fifteen dearest players: a squad's spine, not its academy.
        t["value"] = sum(sorted(t["prices"], reverse=True)[:15])
        t.pop("prices")
    return teams


def _fit_exponent(ratios, rates, mu):
    """k such that mu * ratio**k best explains the observed rates."""
    r = np.asarray(ratios, dtype=float)
    y = np.asarray(rates, dtype=float)
    if len(y) < 4 or mu <= 0:
        return 0.0
    f = minimize(lambda k: float(((mu * r ** k[0]) - y) ** 2 @ np.ones(len(y))),
                 [0.5], method="Nelder-Mead",
                 options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 500})
    k = float(f.x[0])
    # A fit this thin can produce an exponent that says value predicts goals
    # ten times over. Keep it in a range a football season could support.
    return max(-2.0, min(2.0, k))


def ratings(players):
    """Attack and defence multipliers per team, 1.0 being an average side."""
    teams = aggregate(players)
    live = [t for t in teams.values() if t["played"] > 0]
    if not live:
        return {name: {"attack": 1.0, "defence": 1.0, "mu": 1.35, **t}
                for name, t in teams.items()}, 1.35

    mu = sum(t["scored"] for t in live) / sum(t["played"] for t in live)
    mean_value = sum(t["value"] for t in live) / len(live)
    ratios = [t["value"] / mean_value if mean_value else 1.0 for t in live]
    att_obs = [t["scored"] / t["played"] for t in live]
    def_obs = [t["conceded"] / t["played"] for t in live]

    ka = _fit_exponent(ratios, att_obs, mu)
    # Defence runs the other way: a dearer squad concedes less, so the fit
    # should land negative on its own. It is not forced to.
    kd = _fit_exponent(ratios, def_obs, mu)

    w = None
    out = {}
    for name, t in teams.items():
        ratio = (t["value"] / mean_value) if mean_value else 1.0
        pa = mu * ratio ** ka
        pd = mu * ratio ** kd
        played = t["played"]
        w = played / (played + PRIOR_GAMES)
        att = (w * (t["scored"] / played) + (1 - w) * pa) if played else pa
        dfc = (w * (t["conceded"] / played) + (1 - w) * pd) if played else pd
        out[name] = dict(
            t,
            attack=_clamp(att / mu), defence=_clamp(dfc / mu),
            attack_goals=att, defence_goals=dfc,
            value_ratio=ratio)
    return out, mu


def _clamp(x):
    if not math.isfinite(x):
        return 1.0
    return max(MIN_RATIO, min(MAX_RATIO, x))


def lambdas(rate, home, away, mu):
    """Expected goals for one fixture, (home, away)."""
    h = rate.get(home) or {"attack": 1.0, "defence": 1.0}
    a = rate.get(away) or {"attack": 1.0, "defence": 1.0}
    lh = mu * h["attack"] * a["defence"] * HOME
    la = mu * a["attack"] * h["defence"] / HOME
    return (min(max(lh, MIN_LAMBDA), MAX_LAMBDA),
            min(max(la, MIN_LAMBDA), MAX_LAMBDA))
