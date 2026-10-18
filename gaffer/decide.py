"""Rank decisions by where they leave you, not by what they score.

The solver maximises expected points. From 40th of 42 that is the wrong
target, and `position.py` has the right one — but P(finishing first) is an
order statistic over 42 correlated totals, and no amount of cleverness makes
that a linear objective. So the MILP is not replaced; it is demoted to a
**candidate generator**, and the simulator decides between its candidates.

That division matters. The previous attempt to make the optimiser
leverage-aware was a multiplier on expected points, which modelled nothing
and eventually captained a 6.19 over a 9.14. A surrogate with better
marketing would be the same mistake. Here nothing is approximated in the
objective: each candidate is played out and counted.

**Correlation is the whole difficulty.** `distribution.py` builds each
player's points distribution exactly, by convolution — which assumes every
player is independent. For a single squad total that is a mild sin. For
P(first) it is fatal: if a rival and I both own Haaland, our scores move
together, and whether I can overtake him depends entirely on how much of
our two totals is the same randomness.

So the marginals stay exactly as `distribution.py` computes them — the
numbers every other view quotes — and correlation is imposed on top with a
Gaussian copula: one shared normal per fixture, a positive loading for the
home side and a negative one for the away side, since the week in which one
defence keeps a clean sheet is usually the week the other attack did not
score. Sampling a player is then inverting his own PMF at the copula's
uniform, so his marginal distribution is unchanged by construction.
"""
from __future__ import annotations

import numpy as np

from . import distribution

# How much of a player's outcome is the match going a particular way rather
# than the player having a particular day. A judgement, and the one number
# in this module worth arguing about, so `sensitivity()` exists to show what
# turns on it. Zero reproduces the independence assumption exactly.
RHO = 0.45


def _quantiles(pmf):
    """Cumulative probabilities and the point values they map to."""
    pts = np.arange(len(pmf), dtype=float)
    cdf = np.cumsum(np.asarray(pmf, dtype=float))
    cdf[-1] = 1.0
    return cdf, pts


def sample_players(cards, n=20000, rho=RHO, rng=None):
    """(n, players) correlated points, each with its exact marginal.

    `cards` are projection rows carrying `id`, `team` and whatever
    `distribution.fit` needs. A player with no usable distribution scores a
    flat zero rather than being dropped, so column order always matches the
    input and a caller cannot silently mis-index a squad.
    """
    rng = rng or np.random.default_rng(17)
    rho = float(np.clip(rho, 0.0, 0.95))

    # One shared draw per fixture. Teams not playing each other get their own,
    # which is correct: their weeks are unrelated.
    fixtures = {}
    for c in cards:
        key = c.get("fixture_key") or c.get("team")
        fixtures.setdefault(key, len(fixtures))
    z_fix = rng.standard_normal((n, max(1, len(fixtures))))

    out = np.zeros((n, len(cards)))
    a, b = rho, float(np.sqrt(max(0.0, 1.0 - rho * rho)))
    for j, c in enumerate(cards):
        pmf = distribution.player_points(c)
        if not pmf:
            continue
        cdf, pts = _quantiles(pmf)
        side = 1.0 if c.get("home", True) else -1.0
        z = a * side * z_fix[:, fixtures[c.get("fixture_key") or c.get("team")]] \
            + b * rng.standard_normal(n)
        u = _norm_cdf(z)
        out[:, j] = pts[np.searchsorted(cdf, u, side="left").clip(0, len(pts) - 1)]
    return out


def _norm_cdf(z):
    from math import sqrt
    from scipy.special import erf          # vectorised, and exact enough
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def squad_totals(samples, columns, picks, captain=None, chip=None):
    """What one team scores per draw, from its own picks.

    `picks` may be ids (an eleven) or {id: multiplier} (a captured squad,
    where the multiplier already carries the lineup and the armband).
    """
    n = samples.shape[0]
    total = np.zeros(n)
    if isinstance(picks, dict):
        items = picks.items()
    else:
        items = ((pid, 2 if pid == captain else 1) for pid in picks)
    for pid, mult in items:
        col = columns.get(str(pid))
        if col is None or not mult:
            continue
        total += mult * samples[:, col]
    return total


def sensitivity(build_options, rival_gw, my_total, rival_totals,
                rhos=(0.0, 0.25, 0.45, 0.65), **kw):
    """The same ranking at several correlation assumptions.

    RHO is a judgement. If the best option changes when it moves, the
    recommendation is an artefact of that judgement and should be presented
    as one; if it does not, the judgement did not matter here. Either answer
    is worth more than quoting a single number as though it were measured.
    """
    from . import position

    out = []
    for r in rhos:
        rows = position.compare(build_options(r), rival_gw, my_total,
                                rival_totals, **kw)
        key = "p_first_season" if "p_first_season" in rows[0] else "p_first_now"
        out.append({"rho": r, "best": rows[0]["option"],
                    "p_first": rows[0][key],
                    "order": [x["option"] for x in rows]})
    return out
