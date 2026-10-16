"""Per-90 rates for each player: goals, assists, defensive actions, saves, cards.

Raw season totals are a poor guide early on — three games is not enough to tell
a 0.6 xG/90 striker from a 0.2 one. Every rate here is therefore shrunk toward a
positional prior with a weight set by how many 90s the player has actually
played, and blended with last season where the API exposes it.

Underlying xG and xA are preferred to actual goals and assists throughout: they
are a far better predictor of future returns than finishing that has already
happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..data.entities import Player

# League-average per-90 rates by position, used as shrinkage targets.
POSITION_PRIORS: dict[str, dict[str, float]] = {
    "GKP": {
        "xg90": 0.005,
        "xa90": 0.010,
        "cbit90": 1.2,
        "cbirt90": 3.5,
        "saves90": 3.0,
        "yellow90": 0.06,
        "red90": 0.004,
        "xgc90": 1.35,
    },
    "DEF": {
        "xg90": 0.055,
        "xa90": 0.060,
        "cbit90": 7.5,
        "cbirt90": 12.5,
        "saves90": 0.0,
        "yellow90": 0.16,
        "red90": 0.010,
        "xgc90": 1.35,
    },
    "MID": {
        "xg90": 0.150,
        "xa90": 0.130,
        "cbit90": 3.4,
        "cbirt90": 9.0,
        "saves90": 0.0,
        "yellow90": 0.17,
        "red90": 0.008,
        "xgc90": 1.35,
    },
    "FWD": {
        "xg90": 0.360,
        "xa90": 0.120,
        "cbit90": 1.8,
        "cbirt90": 5.5,
        "saves90": 0.0,
        "yellow90": 0.13,
        "red90": 0.006,
        "xgc90": 1.35,
    },
}

# Ball recoveries are noisy and positionally driven; the DefCon threshold for
# midfielders and forwards is high enough that only genuine ball-winners clear
# it, so this rate carries real information and is worth shrinking gently.
_DEFCON_STAT_BY_POSITION = {"DEF": "cbit90", "MID": "cbirt90", "FWD": "cbirt90", "GKP": None}


@dataclass
class PlayerRates:
    """Per-90 rates for one player, already shrunk and blended."""

    element: int
    position: str
    xg90: float
    xa90: float
    cbit90: float
    cbirt90: float
    saves90: float
    yellow90: float
    red90: float
    xgc90: float
    penalty_taker: bool = False
    penalty_order: int | None = None
    set_piece_taker: bool = False
    ninetys: float = 0.0
    # Empirical share of appearances that cleared the DefCon threshold, when the
    # player has enough games for it to mean anything.
    defcon_hit_rate: float | None = None
    defcon_games: int = 0

    def defcon_rate(self) -> float | None:
        stat = _DEFCON_STAT_BY_POSITION.get(self.position)
        if stat is None:
            return None
        return getattr(self, stat)

    def scaled_to_team(self, team_attack_multiplier: float) -> "PlayerRates":
        """Scale attacking output by how good the team's fixture is.

        A striker's xG per 90 is roughly proportional to his team's expected
        goals in that match. The exponent damps this: individual quality does
        not scale one-for-one with team output.
        """
        factor = float(np.clip(team_attack_multiplier, 0.3, 3.0)) ** 0.85
        return PlayerRates(
            **{
                **self.__dict__,
                "xg90": self.xg90 * factor,
                "xa90": self.xa90 * factor,
            }
        )


@dataclass
class RateEstimator:
    """Builds :class:`PlayerRates` from FPL season stats and history."""

    shrinkage_90s: float = 6.0
    last_season_weight: float = 0.45
    league_xgc90: float = 1.35

    def estimate(
        self,
        player: Player,
        *,
        past_seasons: Sequence[dict] = (),
        history_rows: Sequence[dict] = (),
    ) -> PlayerRates:
        position = player.position
        priors = POSITION_PRIORS.get(position, POSITION_PRIORS["MID"])
        ninetys = player.minutes / 90.0

        current = self._current_season_rates(player)
        previous = self._past_season_rates(past_seasons, position)

        # Weight of evidence: this season's minutes, plus a discounted credit
        # for last season. Both are shrunk toward the positional prior.
        weight_current = ninetys
        weight_previous = self.last_season_weight * (previous.pop("_ninetys", 0.0) if previous else 0.0)

        def blend(key: str) -> float:
            prior = priors[key]
            total = weight_current + weight_previous + self.shrinkage_90s
            value = prior * self.shrinkage_90s
            if weight_current > 0 and key in current:
                value += current[key] * weight_current
            else:
                value += prior * weight_current
            if weight_previous > 0 and previous and key in previous:
                value += previous[key] * weight_previous
            else:
                value += prior * weight_previous
            return value / total

        defcon_hit_rate, defcon_games = self._empirical_defcon(history_rows, position)

        return PlayerRates(
            element=player.id,
            position=position,
            xg90=blend("xg90"),
            xa90=blend("xa90"),
            cbit90=blend("cbit90"),
            cbirt90=blend("cbirt90"),
            saves90=blend("saves90") if position == "GKP" else 0.0,
            yellow90=blend("yellow90"),
            red90=blend("red90"),
            xgc90=blend("xgc90"),
            penalty_taker=bool(player.penalties_order and player.penalties_order <= 1),
            penalty_order=player.penalties_order,
            set_piece_taker=bool(
                (player.corners_order and player.corners_order <= 2)
                or (player.freekicks_order and player.freekicks_order <= 2)
            ),
            ninetys=ninetys,
            defcon_hit_rate=defcon_hit_rate,
            defcon_games=defcon_games,
        )

    def _current_season_rates(self, player: Player) -> dict[str, float]:
        if player.minutes <= 0:
            return {}
        per90 = 90.0 / player.minutes
        cbi = player.clearances_blocks_interceptions
        tackles = player.tackles
        recoveries = player.recoveries
        return {
            "xg90": player.expected_goals * per90,
            "xa90": player.expected_assists * per90,
            "cbit90": (cbi + tackles) * per90,
            "cbirt90": (cbi + tackles + recoveries) * per90,
            "saves90": player.saves * per90,
            "yellow90": float(player.raw.get("yellow_cards", 0) or 0) * per90,
            "red90": float(player.raw.get("red_cards", 0) or 0) * per90,
            "xgc90": player.expected_goals_conceded * per90,
        }

    def _past_season_rates(
        self, past_seasons: Sequence[dict], position: str
    ) -> dict[str, float]:
        """Rates from the most recent completed season, if the API exposes them.

        ``history_past`` gives season totals but not always the newer defensive
        stats, so missing keys simply fall back to the positional prior.
        """
        if not past_seasons:
            return {}
        last = past_seasons[-1]
        minutes = float(last.get("minutes", 0) or 0)
        if minutes < 450:  # under five full games is not evidence
            return {}
        per90 = 90.0 / minutes
        out: dict[str, float] = {"_ninetys": minutes / 90.0}

        def maybe(key: str, *fields: str) -> None:
            total = 0.0
            found = False
            for f in fields:
                if last.get(f) is not None:
                    total += float(last.get(f) or 0)
                    found = True
            if found:
                out[key] = total * per90

        maybe("xg90", "expected_goals")
        maybe("xa90", "expected_assists")
        maybe("cbit90", "clearances_blocks_interceptions", "tackles")
        maybe("cbirt90", "clearances_blocks_interceptions", "tackles", "recoveries")
        maybe("saves90", "saves")
        maybe("yellow90", "yellow_cards")
        maybe("red90", "red_cards")
        maybe("xgc90", "expected_goals_conceded")

        # Fall back to actual goals/assists when xG is not published for the
        # past season; noisier, but better than pretending there is no history.
        if "xg90" not in out and last.get("goals_scored") is not None:
            out["xg90"] = float(last.get("goals_scored", 0) or 0) * per90
        if "xa90" not in out and last.get("assists") is not None:
            out["xa90"] = float(last.get("assists", 0) or 0) * per90
        return out

    @staticmethod
    def _empirical_defcon(
        history_rows: Sequence[dict], position: str
    ) -> tuple[float | None, int]:
        """Observed rate of clearing the DefCon threshold, per start.

        The FPL history rows expose ``defensive_contribution`` as the *points*
        awarded, so a non-zero value means the threshold was met.
        """
        if _DEFCON_STAT_BY_POSITION.get(position) is None:
            return None, 0
        qualifying = [r for r in history_rows if float(r.get("minutes", 0) or 0) >= 60]
        if len(qualifying) < 4:
            return None, len(qualifying)
        hits = sum(1 for r in qualifying if float(r.get("defensive_contribution", 0) or 0) > 0)
        return hits / len(qualifying), len(qualifying)


def team_attack_multiplier(fixture_xg: float, team_average_xg: float) -> float:
    """How much better or worse than usual a team's attacking fixture is."""
    if team_average_xg <= 0:
        return 1.0
    return float(np.clip(fixture_xg / team_average_xg, 0.3, 3.0))


def average_team_xg(forecast_xgs: Sequence[float]) -> float:
    """A team's mean expected goals across its projected fixtures."""
    values = [x for x in forecast_xgs if x > 0]
    return float(np.mean(values)) if values else 1.4


@dataclass
class SquadShares:
    """Within-team shares, so a team's goals get distributed sensibly.

    Used by the simulator: once a team's goal count is drawn, the goals have to
    be attributed to players in a way that respects who actually scores them.
    """

    goal_share: dict[int, float] = field(default_factory=dict)
    assist_share: dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_rates(
        cls, rates: dict[int, PlayerRates], minutes: dict[int, float]
    ) -> "SquadShares":
        goal_weights = {
            eid: max(r.xg90, 0.0) * max(minutes.get(eid, 0.0), 0.0) / 90.0
            for eid, r in rates.items()
        }
        assist_weights = {
            eid: max(r.xa90, 0.0) * max(minutes.get(eid, 0.0), 0.0) / 90.0
            for eid, r in rates.items()
        }
        total_goals = sum(goal_weights.values())
        total_assists = sum(assist_weights.values())
        return cls(
            goal_share={
                eid: (w / total_goals if total_goals > 0 else 0.0)
                for eid, w in goal_weights.items()
            },
            assist_share={
                eid: (w / total_assists if total_assists > 0 else 0.0)
                for eid, w in assist_weights.items()
            },
        )
