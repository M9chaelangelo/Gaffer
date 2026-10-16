"""Match-level scoreline maths.

Everything downstream — clean sheet probability, goals-conceded deductions, the
correlation between a defender's clean sheet and his team-mates' returns — comes
from a joint distribution over the scoreline. This module builds that
distribution with a Dixon-Coles adjusted double Poisson, which corrects the
independent-Poisson model's well-known underestimate of 0-0, 1-0, 0-1 and 1-1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson

MAX_GOALS = 10


def _tau(home_goals: np.ndarray, away_goals: np.ndarray, lam: float, mu: float, rho: float):
    """Dixon-Coles low-score correction applied to the independent product."""
    tau = np.ones_like(home_goals, dtype=float)
    tau = np.where((home_goals == 0) & (away_goals == 0), 1.0 - lam * mu * rho, tau)
    tau = np.where((home_goals == 0) & (away_goals == 1), 1.0 + lam * rho, tau)
    tau = np.where((home_goals == 1) & (away_goals == 0), 1.0 + mu * rho, tau)
    tau = np.where((home_goals == 1) & (away_goals == 1), 1.0 - rho, tau)
    return np.clip(tau, 1e-9, None)


def scoreline_matrix(
    lam_home: float, lam_away: float, rho: float = -0.03, max_goals: int = MAX_GOALS
) -> np.ndarray:
    """Joint probability of every scoreline up to ``max_goals`` each.

    Returns an array ``P`` where ``P[h, a]`` is the probability of the home side
    scoring ``h`` and the away side ``a``. Rows are truncated and renormalised,
    which is harmless: the mass beyond 10 goals is around 1e-7.
    """
    lam_home = max(float(lam_home), 1e-4)
    lam_away = max(float(lam_away), 1e-4)
    goals = np.arange(max_goals + 1)
    home_pmf = poisson.pmf(goals, lam_home)
    away_pmf = poisson.pmf(goals, lam_away)
    grid = np.outer(home_pmf, away_pmf)
    hg, ag = np.meshgrid(goals, goals, indexing="ij")
    grid = grid * _tau(hg, ag, lam_home, lam_away, rho)
    total = grid.sum()
    return grid / total if total > 0 else grid


@dataclass
class MatchProbabilities:
    """Everything the points model needs to know about one fixture."""

    lam_home: float
    lam_away: float
    rho: float = -0.03

    def matrix(self) -> np.ndarray:
        return scoreline_matrix(self.lam_home, self.lam_away, self.rho)

    def outcome(self) -> tuple[float, float, float]:
        """(home win, draw, away win)."""
        m = self.matrix()
        draw = float(np.trace(m))
        home = float(np.tril(m, -1).sum())
        away = float(np.triu(m, 1).sum())
        return home, draw, away

    def clean_sheet(self, home: bool) -> float:
        """Probability the given side concedes nothing."""
        m = self.matrix()
        return float(m[:, 0].sum()) if home else float(m[0, :].sum())

    def goals_conceded_pmf(self, home: bool) -> np.ndarray:
        """Distribution of goals conceded by the given side."""
        m = self.matrix()
        return m.sum(axis=0) if home else m.sum(axis=1)

    def goals_scored_pmf(self, home: bool) -> np.ndarray:
        m = self.matrix()
        return m.sum(axis=1) if home else m.sum(axis=0)

    def expected_goals(self, home: bool) -> float:
        pmf = self.goals_scored_pmf(home)
        return float(np.dot(np.arange(len(pmf)), pmf))

    def over(self, line: float) -> float:
        """Probability total goals exceed ``line`` (e.g. 2.5)."""
        m = self.matrix()
        goals = np.arange(m.shape[0])
        totals = goals[:, None] + goals[None, :]
        return float(m[totals > line].sum())


def outcome_probabilities(lam_home: float, lam_away: float, rho: float = -0.03) -> dict[str, float]:
    home, draw, away = MatchProbabilities(lam_home, lam_away, rho).outcome()
    return {"home": home, "draw": draw, "away": away}


def solve_lambdas_from_market(
    p_home: float,
    p_draw: float,
    p_away: float,
    p_over: float | None = None,
    line: float = 2.5,
    rho: float = -0.03,
) -> tuple[float, float]:
    """Recover both sides' expected goals from bookmaker probabilities.

    A 1X2 market pins down the *difference* in strength well but the *total*
    only loosely; an over/under market pins down the total. Given both, there is
    a unique (lambda_home, lambda_away) that reproduces them under Dixon-Coles.
    This solves for it by minimising squared error on the log-odds scale, which
    weights the tails sensibly.

    Falls back gracefully: with no totals market, the total is inferred from the
    1X2 shape alone, anchored on a league-average 2.75 goals.
    """
    from scipy.optimize import minimize

    total_p = p_home + p_draw + p_away
    if total_p <= 0:
        raise ValueError("outcome probabilities must be positive")
    p_home, p_draw, p_away = (p / total_p for p in (p_home, p_draw, p_away))

    def logit(p: float) -> float:
        p = min(max(p, 1e-6), 1 - 1e-6)
        return float(np.log(p / (1 - p)))

    targets = [logit(p_home), logit(p_draw), logit(p_away)]
    if p_over is not None:
        targets.append(logit(p_over))

    def loss(params: np.ndarray) -> float:
        lam_home, lam_away = np.exp(params)
        probs = MatchProbabilities(lam_home, lam_away, rho)
        home, draw, away = probs.outcome()
        model = [logit(home), logit(draw), logit(away)]
        if p_over is not None:
            model.append(logit(probs.over(line)))
        return float(np.sum((np.array(model) - np.array(targets)) ** 2))

    # Start from a sensible league-average split skewed by the market.
    start_total = 2.75
    skew = 0.5 + 0.35 * (p_home - p_away)
    x0 = np.log([max(start_total * skew, 0.2), max(start_total * (1 - skew), 0.2)])

    result = minimize(loss, x0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-8})
    lam_home, lam_away = np.exp(result.x)
    return float(np.clip(lam_home, 0.1, 6.0)), float(np.clip(lam_away, 0.1, 6.0))


def devig_proportional(odds: dict[str, float]) -> dict[str, float]:
    """Remove the bookmaker margin by scaling implied probabilities to sum to 1."""
    inverse = {k: 1.0 / v for k, v in odds.items() if v and v > 1.0}
    total = sum(inverse.values())
    if total <= 0:
        raise ValueError("no usable odds")
    return {k: v / total for k, v in inverse.items()}


def devig_shin(odds: dict[str, float], max_iter: int = 100) -> dict[str, float]:
    """Remove the margin with Shin's model.

    Shin treats the overround as compensation for insider trading, which pulls
    more margin out of longshots than out of favourites. That matches observed
    bookmaker behaviour better than proportional scaling, and it matters here
    because heavy favourites are exactly where captaincy decisions are made.
    """
    inverse = {k: 1.0 / v for k, v in odds.items() if v and v > 1.0}
    if not inverse:
        raise ValueError("no usable odds")
    keys = list(inverse)
    pi = np.array([inverse[k] for k in keys], dtype=float)
    booksum = pi.sum()
    if booksum <= 1.0:
        return {k: float(v / booksum) for k, v in zip(keys, pi)}

    z = 0.0
    for _ in range(max_iter):
        root = np.sqrt(z**2 + 4.0 * (1.0 - z) * pi**2 / booksum)
        probs = (root - z) / (2.0 * (1.0 - z)) if z < 1 else pi / booksum
        total = probs.sum()
        if abs(total - 1.0) < 1e-10:
            break
        # Newton-free bisection-style update: nudge z toward closing the gap.
        z += (total - 1.0) * 0.5
        z = float(np.clip(z, 0.0, 0.5))

    root = np.sqrt(z**2 + 4.0 * (1.0 - z) * pi**2 / booksum)
    probs = (root - z) / (2.0 * (1.0 - z))
    probs = probs / probs.sum()
    return {k: float(p) for k, p in zip(keys, probs)}


def devig(odds: dict[str, float], method: str = "shin") -> dict[str, float]:
    if method == "proportional":
        return devig_proportional(odds)
    try:
        return devig_shin(odds)
    except (ValueError, FloatingPointError):
        return devig_proportional(odds)
