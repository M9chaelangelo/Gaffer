"""Chip strategy.

2026/27 gives eight chips: a Wildcard, Free Hit, Triple Captain and Bench Boost
for the first half of the season, and the same four again for the second. The
first set expires at the Gameweek 19 deadline and cannot be carried over, which
turns chip planning into a scheduling problem with a hard constraint rather than
a matter of waiting for the perfect week.

Each chip is valued as the *marginal* points it adds over playing that gameweek
normally:

    Bench Boost      what the bench actually scores
    Triple Captain   one extra copy of the captain
    Free Hit         best possible one-week squad, minus your own squad
    Wildcard         value of re-optimising the squad for the rest of the horizon

Scenarios are then assembled respecting one chip per gameweek and the half-season
deadlines, and the best few are re-solved jointly so the comparison is honest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from ..config import Rules, get_rules
from ..model.engine import ProjectionSet
from .lineup import choose_starting_eleven, optimise_lineup
from .milp import ChipPlan, OptimisationResult, SquadOptimiser, shortlist

log = logging.getLogger(__name__)

CHIP_LABELS = {
    "wildcard": "Wildcard",
    "freehit": "Free Hit",
    "benchboost": "Bench Boost",
    "triplecaptain": "Triple Captain",
}


@dataclass
class ChipOpportunity:
    chip: str
    gameweek: int
    gain: float
    detail: str = ""
    fixtures_played: int = 0
    double_gameweek_players: int = 0

    @property
    def label(self) -> str:
        return CHIP_LABELS.get(self.chip, self.chip)


@dataclass
class ChipScenario:
    plan: ChipPlan
    expected_points: float
    result: OptimisationResult | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.plan.describe()


def gameweek_fixture_profile(
    projections: ProjectionSet, gameweek: int
) -> tuple[int, dict[int, int]]:
    """(number of fixtures, fixtures per club) for a gameweek.

    A normal gameweek has ten fixtures and every club plays once. Fewer means a
    blank; a club with two entries has a double.
    """
    per_club: dict[int, int] = {team_id: 0 for team_id in projections.teams}
    count = 0
    for forecast in projections.forecasts.values():
        if forecast.gameweek != gameweek:
            continue
        count += 1
        per_club[forecast.home_team] = per_club.get(forecast.home_team, 0) + 1
        per_club[forecast.away_team] = per_club.get(forecast.away_team, 0) + 1
    return count, per_club


def evaluate_bench_boost(
    squad: Sequence[int], projections: ProjectionSet, gameweek: int
) -> ChipOpportunity:
    """Bench Boost is worth exactly what the four bench players score."""
    starters, bench, _ = choose_starting_eleven(squad, projections, gameweek)
    gain = sum(projections.expected_points(e, gameweek) for e in bench)
    fixtures, per_club = gameweek_fixture_profile(projections, gameweek)
    doubles = sum(1 for e in squad if per_club.get(projections.team_of(e), 0) >= 2)
    detail = ", ".join(
        f"{projections.name(e)} {projections.expected_points(e, gameweek):.1f}" for e in bench
    )
    return ChipOpportunity(
        chip="benchboost",
        gameweek=gameweek,
        gain=gain,
        detail=detail,
        fixtures_played=fixtures,
        double_gameweek_players=doubles,
    )


def evaluate_triple_captain(
    squad: Sequence[int], projections: ProjectionSet, gameweek: int
) -> ChipOpportunity:
    """Triple Captain adds one more copy of whoever wears the armband."""
    starters, _, _ = choose_starting_eleven(squad, projections, gameweek)
    if not starters:
        return ChipOpportunity("triplecaptain", gameweek, 0.0)
    best = max(starters, key=lambda e: projections.expected_points(e, gameweek))
    gain = projections.expected_points(best, gameweek)
    projection = projections.get(best, gameweek)
    fixtures, per_club = gameweek_fixture_profile(projections, gameweek)
    detail = (
        f"{projections.name(best)}: {gain:.1f} xPts, "
        f"{projection.p_haul:.0%} chance of 10+, {projection.n_fixtures} fixture(s)"
    )
    return ChipOpportunity(
        chip="triplecaptain",
        gameweek=gameweek,
        gain=gain,
        detail=detail,
        fixtures_played=fixtures,
        double_gameweek_players=1 if projection.n_fixtures >= 2 else 0,
    )


def evaluate_free_hit(
    squad: Sequence[int],
    projections: ProjectionSet,
    gameweek: int,
    *,
    budget: float,
    optimiser: SquadOptimiser | None = None,
    candidate_pool: Sequence[int] | None = None,
) -> ChipOpportunity:
    """Free Hit is worth the best one-week squad money can buy, minus your own."""
    optimiser = optimiser or SquadOptimiser()
    pool = list(candidate_pool or shortlist(projections))

    result = optimiser.solve(
        projections,
        gameweeks=[gameweek],
        current_squad=[],
        bank=budget,
        free_transfers=15,
        candidate_pool=pool,
        time_limit=30,
    )
    if not result.ok or not result.decisions:
        return ChipOpportunity("freehit", gameweek, 0.0, detail="no feasible free hit squad")

    best = result.decisions[0]
    baseline_lineup = optimise_lineup(squad, projections, gameweek, draws=1500)
    free_hit_points = best.expected_points  # already includes the captain's extra copy
    gain = free_hit_points - baseline_lineup.expected_points
    fixtures, per_club = gameweek_fixture_profile(projections, gameweek)
    names = ", ".join(projections.name(e) for e in best.starters[:5])
    return ChipOpportunity(
        chip="freehit",
        gameweek=gameweek,
        gain=gain,
        detail=f"best XI starts {names}…",
        fixtures_played=fixtures,
    )


def evaluate_wildcard(
    squad: Sequence[int],
    projections: ProjectionSet,
    gameweek: int,
    *,
    horizon: Sequence[int],
    bank: float,
    selling_prices: dict[int, float],
    free_transfers: int,
    optimiser: SquadOptimiser | None = None,
    candidate_pool: Sequence[int] | None = None,
) -> ChipOpportunity:
    """Wildcard is worth the whole rest of the horizon, re-optimised.

    Comparing a wildcard against "do nothing" would overstate it: without the
    chip you would still make transfers. The baseline is therefore the normal
    optimiser run from the same starting point.
    """
    optimiser = optimiser or SquadOptimiser()
    pool = list(candidate_pool or shortlist(projections))
    window = [gw for gw in horizon if gw >= gameweek]
    if not window:
        return ChipOpportunity("wildcard", gameweek, 0.0, detail="outside the horizon")

    baseline = optimiser.solve(
        projections,
        gameweeks=window,
        current_squad=squad,
        selling_prices=selling_prices,
        bank=bank,
        free_transfers=free_transfers,
        candidate_pool=pool,
        time_limit=45,
    )
    wildcard = optimiser.solve(
        projections,
        gameweeks=window,
        current_squad=squad,
        selling_prices=selling_prices,
        bank=bank,
        free_transfers=free_transfers,
        chip_plan=ChipPlan(wildcard=gameweek),
        candidate_pool=pool,
        time_limit=45,
    )
    if not (baseline.ok and wildcard.ok):
        return ChipOpportunity("wildcard", gameweek, 0.0, detail="solver could not compare")

    gain = wildcard.total_expected_points() - baseline.total_expected_points()
    changes = len(wildcard.decisions[0].transfers_in) if wildcard.decisions else 0
    return ChipOpportunity(
        chip="wildcard",
        gameweek=gameweek,
        gain=gain,
        detail=f"{changes} changes, evaluated over GW{window[0]}-{window[-1]}",
    )


class ChipPlanner:
    """Scores every chip in every remaining gameweek and assembles a strategy."""

    def __init__(self, rules: Rules | None = None, optimiser: SquadOptimiser | None = None):
        self.rules = rules or get_rules()
        self.optimiser = optimiser or SquadOptimiser()

    def available_chips(self, gameweek: int, chips_used: Sequence[str]) -> list[str]:
        """Chips still in hand for the half-season containing ``gameweek``.

        Chip names are matched loosely because the API has used more than one
        spelling over the years (``3xc`` and ``triplecaptain``, for instance).
        """
        chip_set = self.rules.chip_set_for_gameweek(gameweek)
        if chip_set is None:
            return []
        window_lo, window_hi = self.rules.chip_window(chip_set)
        used_in_window = {_canonical_chip(name) for name in chips_used}
        available = []
        for chip in next(
            s["chips"] for s in self.rules.chip_sets if s["name"] == chip_set
        ):
            if chip not in used_in_window:
                available.append(chip)
        return available

    def opportunities(
        self,
        squad: Sequence[int],
        projections: ProjectionSet,
        *,
        gameweeks: Sequence[int],
        bank: float,
        selling_prices: dict[int, float],
        free_transfers: int,
        chips_used: Sequence[str] = (),
        include_slow: bool = True,
        slow_candidates: int = 2,
    ) -> list[ChipOpportunity]:
        """Value every available chip across the horizon.

        Bench Boost and Triple Captain are cheap to evaluate, so every gameweek
        gets one. Free Hit and Wildcard each require solving a fresh squad, so
        they are only evaluated at the ``slow_candidates`` most promising weeks —
        chosen by where the current squad is weakest, which is exactly where
        those chips pay.
        """
        squad_value = sum(selling_prices.get(e, projections.price(e)) for e in squad)
        pool = shortlist(projections)
        results: list[ChipOpportunity] = []

        for gameweek in gameweeks:
            available = self.available_chips(gameweek, chips_used)
            if "benchboost" in available:
                results.append(evaluate_bench_boost(squad, projections, gameweek))
            if "triplecaptain" in available:
                results.append(evaluate_triple_captain(squad, projections, gameweek))

        if include_slow:
            weakest = self._weakest_gameweeks(squad, projections, gameweeks, slow_candidates)
            for gameweek in weakest:
                available = self.available_chips(gameweek, chips_used)
                if "freehit" in available:
                    results.append(
                        evaluate_free_hit(
                            squad,
                            projections,
                            gameweek,
                            budget=squad_value + bank,
                            optimiser=self.optimiser,
                            candidate_pool=pool,
                        )
                    )
                if "wildcard" in available:
                    results.append(
                        evaluate_wildcard(
                            squad,
                            projections,
                            gameweek,
                            horizon=gameweeks,
                            bank=bank,
                            selling_prices=selling_prices,
                            free_transfers=free_transfers,
                            optimiser=self.optimiser,
                            candidate_pool=pool,
                        )
                    )

        results.sort(key=lambda o: -o.gain)
        return results

    @staticmethod
    def _weakest_gameweeks(
        squad: Sequence[int],
        projections: ProjectionSet,
        gameweeks: Sequence[int],
        n: int,
    ) -> list[int]:
        """Gameweeks where the current squad projects worst.

        A cheap proxy for where a Wildcard or Free Hit is worth most: count the
        players who blank, and score the eleven that would actually play.
        """
        scored: list[tuple[float, int]] = []
        for gameweek in gameweeks:
            blanks = sum(
                1 for e in squad if projections.get(e, gameweek).n_fixtures == 0
            )
            starters, _, _ = choose_starting_eleven(squad, projections, gameweek)
            strength = sum(projections.expected_points(e, gameweek) for e in starters)
            # Blanks dominate: a gameweek where five of your squad do not play is
            # a Free Hit week almost regardless of the rest.
            scored.append((strength - 6.0 * blanks, gameweek))
        scored.sort()
        return [gameweek for _, gameweek in scored[: max(1, n)]]

    def recommend(
        self,
        opportunities: Sequence[ChipOpportunity],
        *,
        thresholds: dict[str, float] | None = None,
        deadline_gameweek: int | None = None,
    ) -> list[str]:
        """Turn chip valuations into advice, including expiry warnings.

        A chip is only worth playing when its gain clears the value of holding
        it for a better week later. The thresholds encode roughly what a good
        week looks like: a Bench Boost below about 12 points is usually worth
        saving, while one over 20 rarely is.
        """
        thresholds = thresholds or {
            "benchboost": 14.0,
            "triplecaptain": 9.0,
            "freehit": 12.0,
            "wildcard": 12.0,
        }
        lines: list[str] = []
        best_by_chip: dict[str, ChipOpportunity] = {}
        for opportunity in opportunities:
            current = best_by_chip.get(opportunity.chip)
            if current is None or opportunity.gain > current.gain:
                best_by_chip[opportunity.chip] = opportunity

        for chip, opportunity in sorted(best_by_chip.items(), key=lambda kv: -kv[1].gain):
            threshold = thresholds.get(chip, 10.0)
            verdict = "play it" if opportunity.gain >= threshold else "hold"
            line = (
                f"{opportunity.label}: best in GW{opportunity.gameweek} "
                f"(+{opportunity.gain:.1f} pts) — {verdict}"
            )
            if opportunity.detail:
                line += f". {opportunity.detail}"
            if (
                deadline_gameweek is not None
                and verdict == "hold"
                and opportunity.gameweek >= deadline_gameweek - 2
            ):
                line += (
                    f" Note: this chip expires after GW{deadline_gameweek}; "
                    "holding much longer means losing it."
                )
            lines.append(line)
        return lines

    def build_scenarios(
        self,
        squad: Sequence[int],
        projections: ProjectionSet,
        opportunities: Sequence[ChipOpportunity],
        *,
        gameweeks: Sequence[int],
        bank: float,
        selling_prices: dict[int, float],
        free_transfers: int,
        top_k: int = 3,
    ) -> list[ChipScenario]:
        """Re-solve the best chip combinations jointly and rank them.

        Independent chip valuations can double-count: playing a Wildcard in
        GW10 changes the squad that a GW12 Bench Boost would boost. Solving the
        top few combinations through the full optimiser removes that error.
        """
        pool = shortlist(projections)
        best_per_chip: dict[str, list[ChipOpportunity]] = {}
        for opportunity in opportunities:
            best_per_chip.setdefault(opportunity.chip, []).append(opportunity)
        for chip in best_per_chip:
            best_per_chip[chip].sort(key=lambda o: -o.gain)

        plans: list[ChipPlan] = [ChipPlan()]
        for chip, options in best_per_chip.items():
            new_plans: list[ChipPlan] = []
            for plan in plans:
                new_plans.append(plan)
                for option in options[:top_k]:
                    if option.gain <= 0:
                        continue
                    candidate = ChipPlan(**plan.__dict__)
                    attribute = {
                        "wildcard": "wildcard",
                        "freehit": "freehit",
                        "benchboost": "bench_boost",
                        "triplecaptain": "triple_captain",
                    }[chip]
                    if getattr(candidate, attribute) is not None:
                        continue
                    if option.gameweek in _occupied_gameweeks(candidate):
                        continue
                    setattr(candidate, attribute, option.gameweek)
                    new_plans.append(candidate)
            plans = new_plans

        # Free Hit is not representable inside the multi-gameweek flow model, so
        # scenarios containing it are scored with the standalone estimate added
        # to a plan that otherwise ignores it.
        scenarios: list[ChipScenario] = []
        seen: set[tuple] = set()
        for plan in plans[:24]:
            key = (plan.wildcard, plan.freehit, plan.bench_boost, plan.triple_captain)
            if key in seen:
                continue
            seen.add(key)

            solve_plan = ChipPlan(
                wildcard=plan.wildcard,
                bench_boost=plan.bench_boost,
                triple_captain=plan.triple_captain,
            )
            result = self.optimiser.solve(
                projections,
                gameweeks=gameweeks,
                current_squad=squad,
                selling_prices=selling_prices,
                bank=bank,
                free_transfers=free_transfers,
                chip_plan=solve_plan,
                candidate_pool=pool,
                time_limit=30,
            )
            if not result.ok:
                continue
            total = result.total_expected_points()
            notes = []
            if plan.freehit is not None:
                free_hit_gain = next(
                    (
                        o.gain
                        for o in opportunities
                        if o.chip == "freehit" and o.gameweek == plan.freehit
                    ),
                    0.0,
                )
                total += free_hit_gain
                notes.append(
                    f"Free Hit in GW{plan.freehit} valued separately (+{free_hit_gain:.1f})"
                )
            scenarios.append(
                ChipScenario(plan=plan, expected_points=total, result=result, notes=notes)
            )

        scenarios.sort(key=lambda s: -s.expected_points)
        return scenarios


def _occupied_gameweeks(plan: ChipPlan) -> set[int]:
    return {
        gw
        for gw in (plan.wildcard, plan.freehit, plan.bench_boost, plan.triple_captain)
        if gw is not None
    }


def _canonical_chip(name: str) -> str:
    """Normalise the several spellings FPL has used for chip names."""
    key = (name or "").strip().lower().replace(" ", "").replace("_", "")
    return {
        "3xc": "triplecaptain",
        "triplecaptain": "triplecaptain",
        "tc": "triplecaptain",
        "bboost": "benchboost",
        "benchboost": "benchboost",
        "bb": "benchboost",
        "freehit": "freehit",
        "fh": "freehit",
        "wildcard": "wildcard",
        "wc": "wildcard",
    }.get(key, key)
