"""Comparing strategies: the different paths you could take from here.

A projection tells you what one plan is worth on average. It does not tell you
whether "roll the transfer and move next week" is safer than "take the -4 now",
or whether a Bench Boost in GW14 beats one in GW17 once the rest of the plan
adapts around it. Those are questions about whole paths, and the honest way to
answer them is to plan each path properly and then simulate it.

Each :class:`Strategy` is turned into a concrete sequence of squads by the
optimiser, and that sequence is then run through the correlated gameweek
simulator. What comes back is a distribution over the horizon, so the comparison
can be made on more than the mean: a plan with the same expected points but a
much worse tenth percentile is a different proposition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..config import Rules, get_rules
from ..model.engine import ProjectionSet
from ..optimize.lineup import optimise_lineup
from ..optimize.milp import ChipPlan, OptimisationResult, SquadOptimiser, shortlist
from .montecarlo import GameweekSimulator, simulate_squad
from .rivals import RivalFieldSimulator, build_field_model

log = logging.getLogger(__name__)


@dataclass
class Strategy:
    """One candidate plan."""

    name: str
    chip_plan: ChipPlan = field(default_factory=ChipPlan)
    max_total_hits: int | None = 0
    max_transfers_per_gameweek: int = 3
    description: str = ""

    def label(self) -> str:
        bits = [self.name]
        if self.chip_plan.describe() != "no chips":
            bits.append(self.chip_plan.describe())
        if self.max_total_hits:
            bits.append(f"up to {self.max_total_hits} hits")
        elif self.max_total_hits == 0:
            bits.append("no hits")
        return " — ".join(bits)


@dataclass
class PathOutcome:
    strategy: Strategy
    plan: OptimisationResult
    totals: np.ndarray  # simulated total points over the horizon
    per_gameweek_means: list[float]
    hits_taken: int
    feasible: bool = True
    note: str = ""

    def summary(self) -> dict[str, float]:
        if not self.feasible or self.totals.size == 0:
            return {"mean": 0.0, "p10": 0.0, "p90": 0.0, "std": 0.0}
        return {
            "mean": float(self.totals.mean()),
            "median": float(np.median(self.totals)),
            "p10": float(np.percentile(self.totals, 10)),
            "p90": float(np.percentile(self.totals, 90)),
            "std": float(self.totals.std()),
            "hits": float(self.hits_taken),
        }

    def probability_beats(self, other: "PathOutcome") -> float:
        """P(this path outscores the other), draw for draw."""
        if self.totals.size == 0 or other.totals.size == 0:
            return 0.0
        n = min(len(self.totals), len(other.totals))
        return float((self.totals[:n] > other.totals[:n]).mean())


def default_strategies(
    available_chips: Sequence[str], gameweeks: Sequence[int], best_chip_gameweeks: dict[str, int]
) -> list[Strategy]:
    """A sensible menu of paths to compare.

    Always includes the two baselines every FPL decision is really measured
    against — do nothing, and make only free transfers — plus a hit-taking
    variant and one path per chip at its best-looking gameweek.
    """
    strategies = [
        Strategy(
            "Hold",
            max_total_hits=0,
            max_transfers_per_gameweek=0,
            description="Roll transfers, change nothing.",
        ),
        Strategy(
            "Free transfers only",
            max_total_hits=0,
            description="Use free transfers as they come, never take a hit.",
        ),
        Strategy(
            "Aggressive",
            max_total_hits=2,
            description="Willing to take up to two -4 hits over the horizon.",
        ),
    ]

    for chip in available_chips:
        gameweek = best_chip_gameweeks.get(chip)
        if gameweek is None or gameweek not in gameweeks:
            continue
        plan = ChipPlan()
        attribute = {
            "wildcard": "wildcard",
            "freehit": "freehit",
            "benchboost": "bench_boost",
            "triplecaptain": "triple_captain",
        }[chip]
        setattr(plan, attribute, gameweek)
        strategies.append(
            Strategy(
                f"{chip.title()} GW{gameweek}",
                chip_plan=plan,
                max_total_hits=0 if chip in ("wildcard", "freehit") else 1,
                description=f"Play the {chip} in gameweek {gameweek}.",
            )
        )
    return strategies


class PathSimulator:
    """Plans and simulates a set of strategies from the same starting point."""

    def __init__(
        self,
        projections: ProjectionSet,
        *,
        rules: Rules | None = None,
        optimiser: SquadOptimiser | None = None,
        seed: int = 20262027,
    ):
        self.projections = projections
        self.rules = rules or get_rules()
        self.optimiser = optimiser or SquadOptimiser()
        self.seed = seed

    def run(
        self,
        strategies: Sequence[Strategy],
        *,
        squad: Sequence[int],
        gameweeks: Sequence[int],
        bank: float,
        selling_prices: dict[int, float],
        free_transfers: int,
        draws: int = 4000,
    ) -> list[PathOutcome]:
        pool = shortlist(self.projections)
        outcomes: list[PathOutcome] = []

        for strategy in strategies:
            # A Free Hit is the one chip the multi-gameweek flow model cannot
            # express: the squad reverts the following gameweek, so it is not a
            # transfer at all. Handing it to the optimiser would make it behave
            # exactly like a Wildcard — a permanent rebuild — and the two paths
            # would come back with identical numbers. Instead the squad path is
            # planned as if the chip were never played, and the Free Hit
            # gameweek is swapped for a standalone squad in `_simulate_plan`.
            planning_plan = ChipPlan(
                wildcard=strategy.chip_plan.wildcard,
                bench_boost=strategy.chip_plan.bench_boost,
                triple_captain=strategy.chip_plan.triple_captain,
            )
            result = self.optimiser.solve(
                self.projections,
                gameweeks=gameweeks,
                current_squad=squad,
                selling_prices=selling_prices,
                bank=bank,
                free_transfers=free_transfers,
                chip_plan=planning_plan,
                candidate_pool=pool,
                max_transfers_per_gameweek=strategy.max_transfers_per_gameweek,
                max_total_hits=strategy.max_total_hits,
                # Your permanent squad is frozen during a Free Hit: transfers
                # that week apply to the temporary squad, and the free transfer
                # survives to be spent afterwards.
                frozen_gameweeks=(
                    [strategy.chip_plan.freehit]
                    if strategy.chip_plan.freehit is not None
                    else []
                ),
                time_limit=30,
            )
            if not result.ok:
                outcomes.append(
                    PathOutcome(
                        strategy=strategy,
                        plan=result,
                        totals=np.array([]),
                        per_gameweek_means=[],
                        hits_taken=0,
                        feasible=False,
                        note=result.infeasible_reason or result.status,
                    )
                )
                continue

            totals, means = self._simulate_plan(
                result,
                strategy,
                gameweeks,
                draws,
                bank=bank,
                selling_prices=selling_prices,
                pool=pool,
            )
            outcomes.append(
                PathOutcome(
                    strategy=strategy,
                    plan=result,
                    totals=totals,
                    per_gameweek_means=means,
                    hits_taken=sum(d.hits for d in result.decisions),
                )
            )

        outcomes.sort(key=lambda o: -(o.totals.mean() if o.totals.size else -1e9))
        return outcomes

    def _simulate_plan(
        self,
        result: OptimisationResult,
        strategy: Strategy,
        gameweeks: Sequence[int],
        draws: int,
        *,
        bank: float = 0.0,
        selling_prices: dict[int, float] | None = None,
        pool: Sequence[int] | None = None,
    ) -> tuple[np.ndarray, list[float]]:
        totals = np.zeros(draws)
        means: list[float] = []
        selling_prices = selling_prices or {}

        for decision in result.decisions:
            chip = strategy.chip_plan.chip_in(decision.gameweek)
            squad = decision.squad
            hits = decision.hits

            if chip == "freehit":
                free_hit_squad = self._free_hit_squad(
                    decision.squad, decision.gameweek, bank, selling_prices, pool
                )
                if free_hit_squad:
                    # For this gameweek only. The plan's own squad carries on
                    # untouched into the next gameweek, which is the point of
                    # the chip and the reason it is not modelled as transfers.
                    squad = free_hit_squad
                    hits = 0

            lineup = optimise_lineup(
                squad,
                self.projections,
                decision.gameweek,
                chip=chip,
                draws=800,
            )
            simulator = GameweekSimulator(
                self.projections, rules=self.rules, seed=self.seed + decision.gameweek
            )
            sample = simulator.simulate(squad, decision.gameweek, draws=draws)
            outcome = simulate_squad(
                sample,
                lineup.starters,
                lineup.bench,
                lineup.captain,
                lineup.vice_captain,
                self.projections,
                captain_multiplier=(
                    self.rules.triple_captain_multiplier if chip == "triplecaptain" else 2
                ),
                bench_boost=chip == "benchboost",
            )
            gameweek_points = outcome.total - self.rules.hit_cost * hits
            totals += gameweek_points
            means.append(float(gameweek_points.mean()))

        return totals, means

    def _free_hit_squad(
        self,
        current_squad: Sequence[int],
        gameweek: int,
        bank: float,
        selling_prices: dict[int, float],
        pool: Sequence[int] | None,
    ) -> list[int]:
        """The best one-week squad the Free Hit budget can buy.

        Budget is what you could raise by selling the current squad, plus the
        bank — the same money a real Free Hit gives you, since the squad comes
        back afterwards regardless of what it cost.
        """
        budget = bank + sum(
            selling_prices.get(e, self.projections.price(e)) for e in current_squad
        )
        result = self.optimiser.solve(
            self.projections,
            gameweeks=[gameweek],
            current_squad=[],
            bank=budget,
            free_transfers=self.rules.squad_size,
            candidate_pool=list(pool) if pool is not None else None,
            time_limit=30,
        )
        if not result.ok or not result.decisions:
            log.info("free hit squad infeasible for GW%s; keeping the planned squad", gameweek)
            return []
        return result.decisions[0].squad

    def with_rank(
        self,
        outcome: PathOutcome,
        gameweek: int,
        *,
        draws: int = 2000,
        n_rivals: int = 1500,
    ):
        """Attach a rank projection to the first gameweek of a path."""
        decision = next((d for d in outcome.plan.decisions if d.gameweek == gameweek), None)
        if decision is None:
            return None

        field_model = build_field_model(self.projections, gameweek)
        universe = sorted(
            {
                element
                for element, player in self.projections.players.items()
                if player.selected_by_percent > 0.4
            }
            | set(decision.squad)
        )
        simulator = GameweekSimulator(self.projections, rules=self.rules, seed=self.seed)
        sample = simulator.simulate(universe, gameweek, draws=draws)

        lineup = optimise_lineup(decision.squad, self.projections, gameweek, draws=800)
        yours = simulate_squad(
            sample,
            lineup.starters,
            lineup.bench,
            lineup.captain,
            lineup.vice_captain,
            self.projections,
        )
        rivals = RivalFieldSimulator(self.projections, field_model, rules=self.rules)
        field_scores = rivals.score_field(sample, n_rivals=n_rivals, max_draws=draws)
        return rivals.project_rank(yours.total, field_scores)


def compare(outcomes: Sequence[PathOutcome]) -> list[dict]:
    """A ranked table of paths, with the odds each beats the baseline."""
    if not outcomes:
        return []
    baseline = next((o for o in outcomes if o.strategy.name == "Hold"), outcomes[-1])
    rows = []
    for outcome in outcomes:
        summary = outcome.summary()
        rows.append(
            {
                "strategy": outcome.strategy.label(),
                "description": outcome.strategy.description,
                "mean": round(summary.get("mean", 0.0), 1),
                "p10": round(summary.get("p10", 0.0), 1),
                "p90": round(summary.get("p90", 0.0), 1),
                "hits": outcome.hits_taken,
                "vs_hold": round(
                    summary.get("mean", 0.0) - baseline.summary().get("mean", 0.0), 1
                ),
                "p_beats_hold": round(outcome.probability_beats(baseline), 3),
                "feasible": outcome.feasible,
                "note": outcome.note,
            }
        )
    return rows


@dataclass
class RecommendedPath:
    """One of the routes offered for the week, with why it is on the list."""

    outcome: PathOutcome
    role: str          # "recommended", "safe", "aggressive", "chip"
    headline: str
    rationale: str
    is_recommendation: bool = False

    @property
    def label(self) -> str:
        return self.outcome.strategy.label()

    def summary(self) -> dict:
        stats = self.outcome.summary()
        return {
            "role": self.role,
            "headline": self.headline,
            "rationale": self.rationale,
            "recommended": self.is_recommendation,
            "mean": round(stats.get("mean", 0.0), 1),
            "p10": round(stats.get("p10", 0.0), 1),
            "p90": round(stats.get("p90", 0.0), 1),
            "hits": self.outcome.hits_taken,
        }


def recommend_paths(
    outcomes: Sequence[PathOutcome], projections=None, n: int = 3
) -> list[RecommendedPath]:
    """Pick a small, genuinely different set of routes to choose between.

    Offering the three highest-scoring plans is close to useless when all three
    are the same plan with a player swapped. What a manager actually wants is a
    spread: the best expected outcome, the safest floor, and the one that swings
    hardest — so each slot is filled by a different *kind* of plan, and a plan
    already chosen cannot fill a second slot.
    """
    feasible = [o for o in outcomes if o.feasible and o.totals.size]
    if not feasible:
        return []

    by_mean = sorted(feasible, key=lambda o: -o.totals.mean())
    best = by_mean[0]
    chosen: list[RecommendedPath] = [
        RecommendedPath(
            outcome=best,
            role="recommended",
            headline="Highest expected points",
            rationale=(
                f"{best.summary()['mean']:.0f} points over the horizon, "
                f"{best.hits_taken} hit(s) taken."
            ),
            is_recommendation=True,
        )
    ]
    used = {id(best)}

    # The safest route is the one with the best tenth percentile, which is a
    # different question from the best average and often a different plan.
    safe = max(
        (o for o in feasible if id(o) not in used),
        key=lambda o: o.summary().get("p10", 0.0),
        default=None,
    )
    if safe is not None:
        gap = best.totals.mean() - safe.totals.mean()
        chosen.append(
            RecommendedPath(
                outcome=safe,
                role="safe",
                headline="Best floor",
                rationale=(
                    f"p10 of {safe.summary()['p10']:.0f} — the sturdiest bad week, "
                    f"for {gap:.1f} points less on average."
                ),
            )
        )
        used.add(id(safe))

    # The most aggressive route is the widest, which is what you want when you
    # are chasing rather than protecting.
    bold = max(
        (o for o in feasible if id(o) not in used),
        key=lambda o: o.summary().get("p90", 0.0),
        default=None,
    )
    if bold is not None:
        chosen.append(
            RecommendedPath(
                outcome=bold,
                role="aggressive",
                headline="Highest ceiling",
                rationale=(
                    f"p90 of {bold.summary()['p90']:.0f}, and beats holding "
                    f"{bold.probability_beats(feasible[-1]):.0%} of the time."
                ),
            )
        )
        used.add(id(bold))

    return chosen[:n]
