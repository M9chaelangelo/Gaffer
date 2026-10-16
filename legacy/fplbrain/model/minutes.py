"""How long a player is likely to be on the pitch.

Minutes dominate everything else in FPL. A £4.5m defender who plays 90 minutes
every week beats a £7.0m one who is rotated, and the most expensive mistakes are
almost always started players who did not start. So the minutes model is kept
separate, explicit and inspectable rather than buried inside a points regression.

The output for each player and fixture is a small distribution:

    p_start          probability of being in the starting XI
    p_appear         probability of playing at least one minute
    p_60             probability of reaching the 60-minute scoring threshold
    expected_minutes unconditional expectation, used for per-90 rate scaling
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..data.entities import Player

# Priors for a player with no usable history, by position and price bracket. A
# £4.0m squad filler is a very different proposition to a £12m forward, and
# price is the only signal available for a new signing in gameweek 1.
_PRICE_START_PRIOR = [
    # (min price, p_start)
    (11.0, 0.90),
    (9.0, 0.85),
    (7.5, 0.78),
    (6.0, 0.66),
    (5.0, 0.52),
    (4.5, 0.38),
    (0.0, 0.22),
]

# A starter's chance of still being on at 60 minutes, by position. Forwards and
# attacking midfielders are hooked far more often than centre-backs.
_P60_GIVEN_START = {"GKP": 0.995, "DEF": 0.93, "MID": 0.86, "FWD": 0.80}

# Minutes played when a starter is substituted before the end.
_MEAN_MINUTES_IF_SUBBED = {"GKP": 45.0, "DEF": 55.0, "MID": 58.0, "FWD": 56.0}

# Minutes played by a substitute who does get on.
_MEAN_MINUTES_IF_SUB_APPEARANCE = 18.0


@dataclass
class MinutesProjection:
    element: int
    p_start: float
    p_appear: float
    p_60: float
    expected_minutes: float
    reason: str = ""

    @property
    def p_bench_appearance(self) -> float:
        return max(0.0, self.p_appear - self.p_start)

    def scaled(self, factor: float) -> "MinutesProjection":
        """Scale all probabilities, e.g. by an availability discount."""
        factor = float(np.clip(factor, 0.0, 1.0))
        return MinutesProjection(
            element=self.element,
            p_start=self.p_start * factor,
            p_appear=self.p_appear * factor,
            p_60=self.p_60 * factor,
            expected_minutes=self.expected_minutes * factor,
            reason=self.reason,
        )


def _price_prior(price: float) -> float:
    for threshold, prior in _PRICE_START_PRIOR:
        if price >= threshold:
            return prior
    return 0.2


def _decayed_rates(
    minutes_history: Sequence[float], half_life: float
) -> tuple[float, float, float]:
    """Weighted (start rate, appearance rate, mean minutes when starting).

    ``minutes_history`` is ordered oldest to newest. A start is inferred from
    minutes: a player on for 60+ minutes almost certainly started, and one on
    for under 25 almost certainly did not. The 25-60 band is ambiguous, so it is
    counted as a partial start rather than forced either way.
    """
    if not minutes_history:
        return float("nan"), float("nan"), float("nan")

    n = len(minutes_history)
    ages = np.arange(n - 1, -1, -1, dtype=float)
    weights = 0.5 ** (ages / max(half_life, 0.5))
    minutes = np.asarray(minutes_history, dtype=float)

    started = np.where(minutes >= 60, 1.0, np.where(minutes >= 25, 0.5, 0.0))
    appeared = (minutes > 0).astype(float)

    total_weight = weights.sum()
    if total_weight <= 0:
        return float("nan"), float("nan"), float("nan")

    start_rate = float((weights * started).sum() / total_weight)
    appear_rate = float((weights * appeared).sum() / total_weight)

    start_mask = minutes >= 60
    if start_mask.any():
        mean_start_minutes = float(
            (weights[start_mask] * minutes[start_mask]).sum() / weights[start_mask].sum()
        )
    else:
        mean_start_minutes = float("nan")
    return start_rate, appear_rate, mean_start_minutes


def project_minutes(
    player: Player,
    minutes_history: Sequence[float] | None = None,
    *,
    half_life: float = 4.0,
    prior_games: float = 3.0,
    availability_override: float | None = None,
    rotation_risk: float = 0.0,
) -> MinutesProjection:
    """Project one player's minutes for a single fixture.

    ``minutes_history`` should be this season's minutes per appearance in
    chronological order. ``rotation_risk`` in [0, 1] is an external discount for
    fixture congestion or a manager who rotates heavily.
    """
    position = player.position
    price_prior = _price_prior(player.price)

    start_rate, appear_rate, mean_start_minutes = _decayed_rates(
        minutes_history or [], half_life
    )

    if math.isnan(start_rate):
        # No games this season: fall back on season aggregates, then on price.
        if player.minutes > 0:
            games_equivalent = max(player.minutes / 90.0, 0.5)
            season_start_rate = (
                player.starts / games_equivalent if player.starts else player.minutes / 90.0 / 38
            )
            start_rate = float(np.clip(season_start_rate, 0.0, 1.0))
            appear_rate = min(1.0, start_rate + 0.15)
            effective_games = min(games_equivalent, 10.0)
        else:
            start_rate = price_prior
            appear_rate = min(1.0, price_prior + 0.2)
            effective_games = 0.0
    else:
        effective_games = float(min(len(minutes_history or []), 10))

    # Shrink toward the price prior; a player with two games of evidence should
    # not be assumed nailed on because he happened to start both.
    weight = effective_games / (effective_games + prior_games)
    p_start = weight * start_rate + (1 - weight) * price_prior
    p_appear = weight * appear_rate + (1 - weight) * min(1.0, price_prior + 0.2)
    p_appear = max(p_appear, p_start)

    availability = (
        availability_override if availability_override is not None else player.availability
    )
    p_start *= availability * (1.0 - rotation_risk)
    p_appear *= availability * (1.0 - 0.5 * rotation_risk)
    p_appear = max(p_appear, p_start)

    p60_given_start = _P60_GIVEN_START.get(position, 0.85)
    if not math.isnan(mean_start_minutes) and effective_games >= 3:
        # Blend the positional prior with what this player actually does: a
        # striker averaging 88 minutes a start is not a 70-minute risk.
        observed = float(np.clip((mean_start_minutes - 45.0) / 45.0, 0.0, 1.0))
        p60_given_start = 0.5 * p60_given_start + 0.5 * (0.5 + 0.5 * observed)

    p_60 = p_start * p60_given_start
    minutes_if_subbed = _MEAN_MINUTES_IF_SUBBED.get(position, 55.0)
    expected_minutes = (
        p_start * (p60_given_start * 82.0 + (1 - p60_given_start) * minutes_if_subbed)
        + (p_appear - p_start) * _MEAN_MINUTES_IF_SUB_APPEARANCE
    )

    reason = ""
    if availability < 1.0:
        reason = f"availability {availability:.0%}"
    elif rotation_risk > 0:
        reason = f"rotation risk {rotation_risk:.0%}"
    elif effective_games < 2:
        reason = "little evidence; price-based prior"

    return MinutesProjection(
        element=player.id,
        p_start=float(np.clip(p_start, 0.0, 1.0)),
        p_appear=float(np.clip(p_appear, 0.0, 1.0)),
        p_60=float(np.clip(p_60, 0.0, 1.0)),
        expected_minutes=float(np.clip(expected_minutes, 0.0, 90.0)),
        reason=reason,
    )


def minutes_history_from_summary(history_rows: Sequence[dict]) -> list[float]:
    """Pull the ordered minutes series out of an ``element-summary`` history."""
    rows = sorted(history_rows, key=lambda r: (r.get("round", 0), r.get("kickoff_time", "")))
    return [float(r.get("minutes", 0) or 0) for r in rows]


def congestion_rotation_risk(
    fixtures_in_gameweek: int, days_since_last_match: float | None = None
) -> float:
    """A crude rotation discount for congested schedules.

    Double gameweeks and short turnarounds raise the chance any given player
    sits one out. This is intentionally mild: over-discounting doubles is a
    classic way to talk yourself out of good Bench Boost weeks.
    """
    risk = 0.0
    if fixtures_in_gameweek >= 2:
        risk += 0.10
    if days_since_last_match is not None and days_since_last_match < 3.5:
        risk += 0.08
    return float(min(risk, 0.30))
