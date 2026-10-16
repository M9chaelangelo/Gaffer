"""Correlated Monte Carlo simulation of gameweeks.

Adding up independent player distributions understates the spread of a real
gameweek badly. FPL outcomes are correlated in ways that matter:

* team-mates share a clean sheet — three defenders from the same club is one
  bet, not three;
* team-mates share goals — when a side scores four, several of them return;
* a captain doubles whatever correlation he already carries.

So the simulator works top-down, the way a match actually unfolds. It draws a
scoreline for each fixture, then attributes those goals and assists to
individual players, then resolves minutes, defensive contributions, cards, saves
and bonus. Everything a manager cares about — the shape of a gameweek, the
chance of a green arrow, the value of a chip — falls out of the resulting sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..config import Rules, get_rules
from ..model.engine import ProjectionSet
from ..model.points import negbinom_tail
from ..optimize.lineup import _formation_legal

# Share of goals that are assisted. The rest are unassisted, penalties, or from
# a rebound that FPL does not credit.
ASSIST_RATE = 0.68


@dataclass
class GameweekSample:
    """Simulated points for a set of players over one gameweek."""

    gameweek: int
    elements: list[int]
    points: np.ndarray  # shape (draws, len(elements))
    minutes: np.ndarray  # shape (draws, len(elements)), minutes played

    def index_of(self, element: int) -> int:
        return self.elements.index(element)

    def for_element(self, element: int) -> np.ndarray:
        return self.points[:, self.index_of(element)]

    @property
    def draws(self) -> int:
        return int(self.points.shape[0])


@dataclass
class SquadOutcome:
    """The distribution of a squad's gameweek score."""

    total: np.ndarray
    starters_total: np.ndarray
    bench_total: np.ndarray
    autosub_points: np.ndarray
    captain_points: np.ndarray

    def summary(self) -> dict[str, float]:
        return {
            "mean": float(self.total.mean()),
            "median": float(np.median(self.total)),
            "std": float(self.total.std()),
            "p10": float(np.percentile(self.total, 10)),
            "p25": float(np.percentile(self.total, 25)),
            "p75": float(np.percentile(self.total, 75)),
            "p90": float(np.percentile(self.total, 90)),
            "p_over_60": float((self.total >= 60).mean()),
            "p_over_80": float((self.total >= 80).mean()),
            "captain_mean": float(self.captain_points.mean()),
            "autosub_mean": float(self.autosub_points.mean()),
        }


class GameweekSimulator:
    """Draws correlated player outcomes for a gameweek."""

    def __init__(
        self,
        projections: ProjectionSet,
        *,
        rules: Rules | None = None,
        seed: int = 20262027,
    ):
        self.projections = projections
        self.rules = rules or get_rules()
        self.rng = np.random.default_rng(seed)
        scoring = self.rules.scoring
        self.goal_points = scoring["goal"]
        self.assist_points = int(scoring["assist"])
        self.cs_points = scoring["clean_sheet"]
        self.gc_per = int(scoring["goals_conceded"]["per"])
        self.gc_points = int(scoring["goals_conceded"]["points"])
        self.gc_positions = set(scoring["goals_conceded"]["positions"])
        self.saves_per = int(scoring["saves"]["per"])
        self.defcon_points = self.rules.defcon_points

    def simulate(
        self, elements: Sequence[int], gameweek: int, draws: int = 10000
    ) -> GameweekSample:
        """Simulate ``draws`` gameweeks, covering every club the players belong to.

        The requested players are always included, but the simulation is widened
        to their team-mates first. It has to be: a drawn scoreline is shared out
        among whoever is on the pitch, so simulating three players from a club
        that scored twice would hand them both goals. Widening keeps the shares
        honest, and the extra columns are cheap.
        """
        elements = self._expand_to_full_squads(elements, gameweek)
        index = {element: i for i, element in enumerate(elements)}
        points = np.zeros((draws, len(elements)))
        minutes_out = np.zeros((draws, len(elements)))

        by_fixture: dict[int, list[int]] = {}
        for element in elements:
            projection = self.projections.get(element, gameweek)
            for fixture in projection.fixtures:
                by_fixture.setdefault(fixture.fixture_id, []).append(element)

        for fixture_id, involved in by_fixture.items():
            forecast = self.projections.forecasts.get(fixture_id)
            if forecast is None:
                continue
            self._simulate_fixture(
                forecast, involved, gameweek, draws, points, minutes_out, index
            )

        return GameweekSample(
            gameweek=gameweek, elements=elements, points=points, minutes=minutes_out
        )

    def _expand_to_full_squads(self, elements: Sequence[int], gameweek: int) -> list[int]:
        """Add every team-mate of the requested players who has a fixture."""
        teams = {self.projections.team_of(element) for element in elements}
        expanded = set(elements)
        for element, player in self.projections.players.items():
            if player.team_id not in teams:
                continue
            if self.projections.get(element, gameweek).n_fixtures == 0:
                continue
            expanded.add(element)
        return sorted(expanded)

    def _simulate_fixture(
        self,
        forecast,
        involved: Sequence[int],
        gameweek: int,
        draws: int,
        points: np.ndarray,
        minutes_out: np.ndarray,
        index: dict[int, int],
    ) -> None:
        rng = self.rng
        matrix = forecast.probabilities().matrix()
        flat = matrix.ravel()
        flat = flat / flat.sum()
        size = matrix.shape[0]
        draw_index = rng.choice(len(flat), size=draws, p=flat)
        home_goals = draw_index // size
        away_goals = draw_index % size

        for team_id in (forecast.home_team, forecast.away_team):
            is_home = team_id == forecast.home_team
            scored = home_goals if is_home else away_goals
            conceded = away_goals if is_home else home_goals

            squad = [e for e in involved if self.projections.team_of(e) == team_id]
            if not squad:
                continue

            played, played_60, minutes = self._draw_minutes(squad, gameweek, draws)
            for position, element in enumerate(squad):
                minutes_out[:, index[element]] += minutes[:, position]

            goals, assists = self._attribute_goals(squad, scored, minutes, draws)
            clean_sheet = conceded == 0

            for position, element in enumerate(squad):
                player_position = self.projections.position(element)
                column = index[element]
                element_points = np.zeros(draws)

                element_points += played[:, position].astype(float)
                element_points += played_60[:, position].astype(float)

                element_points += goals[:, position] * self.goal_points.get(player_position, 4)
                element_points += assists[:, position] * self.assist_points

                cs_value = self.cs_points.get(player_position, 0)
                if cs_value:
                    element_points += cs_value * (clean_sheet & played_60[:, position])

                if player_position in self.gc_positions:
                    element_points += (
                        (conceded // self.gc_per) * self.gc_points * played[:, position]
                    )

                if player_position == "GKP":
                    element_points += self._draw_saves(
                        element, conceded, minutes[:, position], draws
                    )

                element_points += self._draw_defcon(
                    element, player_position, minutes[:, position], draws
                )
                element_points += self._draw_cards(element, minutes[:, position], draws)

                points[:, column] += element_points

            # Bonus is a within-match ranking, so it is resolved once both sides
            # have their raw contributions.
            self._apply_bonus(
                squad, points, index, goals, assists, played, played_60, clean_sheet, draws
            )

    def _draw_minutes(
        self, squad: Sequence[int], gameweek: int, draws: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = self.rng
        n = len(squad)
        played = np.zeros((draws, n), dtype=bool)
        played_60 = np.zeros((draws, n), dtype=bool)
        minutes = np.zeros((draws, n))

        for position, element in enumerate(squad):
            projection = self.projections.get(element, gameweek)
            p_appear = 0.0
            p_60 = 0.0
            for fixture in projection.fixtures:
                if fixture.minutes is not None:
                    p_appear = max(p_appear, fixture.minutes.p_appear)
                    p_60 = max(p_60, fixture.minutes.p_60)
            uniform = rng.random(draws)
            played[:, position] = uniform < p_appear
            played_60[:, position] = uniform < p_60
            minutes[:, position] = np.where(
                played_60[:, position], 85.0, np.where(played[:, position], 30.0, 0.0)
            )
        return played, played_60, minutes

    def _attribute_goals(
        self, squad: Sequence[int], team_goals: np.ndarray, minutes: np.ndarray, draws: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Share a team's goals and assists out among the players on the pitch.

        Each goal picks a scorer in proportion to that player's expected-goals
        rate, restricted to whoever is actually on the pitch in that draw. The
        selection is done by inverse-CDF sampling across the whole array at once
        rather than a loop over draws, which matters: this is the hot path of
        the simulator.
        """
        rng = self.rng
        n = len(squad)
        goals = np.zeros((draws, n))
        assists = np.zeros((draws, n))
        max_goals = int(team_goals.max()) if team_goals.size else 0
        if n == 0 or max_goals == 0:
            return goals, assists

        goal_weights = np.array([max(self.projections.rates[e].xg90, 1e-4) for e in squad])
        assist_weights = np.array([max(self.projections.rates[e].xa90, 1e-4) for e in squad])
        on_pitch = (minutes > 0).astype(float)

        def sample(weights: np.ndarray, count: int) -> np.ndarray:
            """Indices of ``count`` weighted picks per draw, shape (draws, count)."""
            masked = weights[None, :] * on_pitch
            totals = masked.sum(axis=1, keepdims=True)
            # If nobody is on the pitch in a draw, fall back to the unmasked
            # weights: a goal was scored, so somebody must be credited with it.
            empty = totals[:, 0] <= 0
            if empty.any():
                masked[empty] = weights[None, :]
                totals[empty, 0] = weights.sum()
            cumulative = np.cumsum(masked / totals, axis=1)
            # Guard against floating point leaving the last bin below 1.
            cumulative[:, -1] = 1.0
            uniforms = rng.random((draws, count))
            return (uniforms[:, :, None] > cumulative[:, None, :]).sum(axis=2).clip(0, n - 1)

        scorers = sample(goal_weights, max_goals)
        helpers = sample(assist_weights, max_goals)

        # Only the goals a team actually scored in that draw count.
        slots = np.arange(max_goals)[None, :]
        scored = slots < team_goals[:, None]
        rows = np.repeat(np.arange(draws)[:, None], max_goals, axis=1)

        np.add.at(goals, (rows[scored], scorers[scored]), 1.0)

        # Not every goal is assisted, and nobody assists their own goal.
        assisted = scored & (rng.random((draws, max_goals)) < ASSIST_RATE)
        assisted &= helpers != scorers
        np.add.at(assists, (rows[assisted], helpers[assisted]), 1.0)

        return goals, assists

    def _draw_saves(
        self, element: int, conceded: np.ndarray, minutes: np.ndarray, draws: int
    ) -> np.ndarray:
        rates = self.projections.rates[element]
        expected = np.maximum(0.2, 1.9 * np.maximum(conceded, 0.4)) * (minutes / 90.0)
        if rates.saves90 > 0:
            expected = 0.5 * expected + 0.5 * rates.saves90 * (minutes / 90.0)
        saves = self.rng.poisson(np.maximum(expected, 1e-6))
        return (saves // self.saves_per).astype(float)

    def _draw_defcon(
        self, element: int, position: str, minutes: np.ndarray, draws: int
    ) -> np.ndarray:
        rule = self.rules.defcon_rule(position)
        if not rule:
            return np.zeros(draws)
        rates = self.projections.rates[element]
        rate = rates.defcon_rate()
        if rate is None:
            return np.zeros(draws)
        threshold = int(rule["threshold"])
        share = minutes / 90.0
        probability = np.array(
            [negbinom_tail(rate * s, threshold) if s > 0 else 0.0 for s in np.unique(share)]
        )
        lookup = dict(zip(np.unique(share), probability))
        probabilities = np.array([lookup[s] for s in share])
        if rates.defcon_hit_rate is not None and rates.defcon_games >= 4:
            weight = min(rates.defcon_games / (rates.defcon_games + 6.0), 0.75)
            probabilities = weight * (rates.defcon_hit_rate * np.minimum(share / 0.85, 1.0)) + (
                1 - weight
            ) * probabilities
        return (self.rng.random(draws) < probabilities).astype(float) * self.defcon_points

    def _draw_cards(self, element: int, minutes: np.ndarray, draws: int) -> np.ndarray:
        rates = self.projections.rates[element]
        share = minutes / 90.0
        yellow = self.rng.random(draws) < np.clip(rates.yellow90 * share, 0, 0.6)
        red = self.rng.random(draws) < np.clip(rates.red90 * share, 0, 0.15)
        return yellow * -1.0 + red * -3.0

    def _apply_bonus(
        self,
        squad: Sequence[int],
        points: np.ndarray,
        index: dict[int, int],
        goals: np.ndarray,
        assists: np.ndarray,
        played: np.ndarray,
        played_60: np.ndarray,
        clean_sheet: np.ndarray,
        draws: int,
    ) -> None:
        """Award 3-2-1 bonus by simulated BPS within each side.

        This is an approximation of the full within-match ranking: bonus is
        contested across both teams, but the dominant driver is goal
        involvement, and resolving it per side keeps the simulator tractable
        while preserving the correlation that matters (a hat-trick scorer almost
        always takes three bonus points off his own team-mates).
        """
        weights = self.rules.bps_weights
        n = len(squad)
        if n == 0:
            return
        bps = np.zeros((draws, n))
        for position, element in enumerate(squad):
            player_position = self.projections.position(element)
            bps[:, position] += goals[:, position] * float(
                weights.get("goal", {}).get(player_position, 18)
            )
            bps[:, position] += assists[:, position] * float(weights.get("assist", 9))
            bps[:, position] += played_60[:, position] * float(
                weights.get("played_60_plus", 6)
            )
            cs_weight = float(weights.get("clean_sheet", {}).get(player_position, 0))
            if cs_weight:
                bps[:, position] += cs_weight * (clean_sheet & played_60[:, position])
            # Baseline defensive and possession BPS, which decides most bonus
            # in matches where nobody scores twice.
            rates = self.projections.rates[element]
            bps[:, position] += played_60[:, position] * (
                rates.cbit90 / max(float(weights.get("cbi_per", 3)), 1.0)
                + max(rates.cbirt90 - rates.cbit90, 0.0)
                / max(float(weights.get("recovery_per", 4)), 1.0)
            )
            bps[:, position] += self.rng.normal(0.0, 4.0, draws)

        # Someone who never came on cannot win bonus points.
        bps = np.where(played, bps, -1e6)

        order = np.argsort(-bps, axis=1)
        for rank, award in enumerate(self.rules.scoring.get("bonus", [3, 2, 1])):
            if rank >= n:
                break
            winners = order[:, rank]
            for position, element in enumerate(squad):
                # A winner who never played is an artefact of the masking above.
                mask = (winners == position) & played[:, position]
                # Only half the bonus is awarded per side, since the real award
                # is contested across both teams in the fixture.
                points[mask, index[element]] += award * 0.5


def simulate_squad(
    sample: GameweekSample,
    starters: Sequence[int],
    bench: Sequence[int],
    captain: int,
    vice_captain: int,
    projections: ProjectionSet,
    *,
    captain_multiplier: int = 2,
    bench_boost: bool = False,
) -> SquadOutcome:
    """Score a squad from a simulated gameweek, resolving autosubs and the armband."""
    draws = sample.draws
    starters = list(starters)
    bench = list(bench)

    def column(element: int) -> int:
        return sample.index_of(element)

    played = {e: sample.minutes[:, column(e)] > 0 for e in starters + bench}
    points = {e: sample.points[:, column(e)] for e in starters + bench}

    starters_total = np.zeros(draws)
    for element in starters:
        starters_total += points[element] * played[element]

    bench_total = np.zeros(draws)
    for element in bench:
        bench_total += points[element] * played[element]

    autosub_points = np.zeros(draws)
    if not bench_boost:
        positions = {e: projections.position(e) for e in starters + bench}
        bench_keepers = [e for e in bench if positions[e] == "GKP"]
        bench_outfield = [e for e in bench if positions[e] != "GKP"]

        absent = np.zeros(draws, dtype=bool)
        for element in starters:
            absent |= ~played[element]

        for draw in np.flatnonzero(absent):
            missing = [e for e in starters if not played[e][draw]]
            live = [positions[e] for e in starters if played[e][draw]]
            used: set[int] = set()
            for out in missing:
                if positions[out] == "GKP":
                    replacement = next(
                        (k for k in bench_keepers if played[k][draw] and k not in used), None
                    )
                    if replacement is not None:
                        used.add(replacement)
                        autosub_points[draw] += points[replacement][draw]
                        live.append("GKP")
                    continue
                for candidate in bench_outfield:
                    if candidate in used or not played[candidate][draw]:
                        continue
                    if _formation_legal(live + [positions[candidate]]):
                        used.add(candidate)
                        autosub_points[draw] += points[candidate][draw]
                        live.append(positions[candidate])
                        break

    captain_played = played.get(captain, np.zeros(draws, dtype=bool))
    captain_points = points.get(captain, np.zeros(draws)) * (captain_multiplier - 1) * captain_played
    if vice_captain in points:
        captain_points += (
            points[vice_captain]
            * (captain_multiplier - 1)
            * (~captain_played)
            * played[vice_captain]
        )

    total = starters_total + captain_points
    if bench_boost:
        total = total + bench_total
    else:
        total = total + autosub_points

    return SquadOutcome(
        total=total,
        starters_total=starters_total,
        bench_total=bench_total,
        autosub_points=autosub_points,
        captain_points=captain_points,
    )
