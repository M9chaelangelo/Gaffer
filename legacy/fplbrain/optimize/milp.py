"""Multi-gameweek squad and transfer optimisation.

The decision is not "who is the best player to bring in this week" — it is
"which sequence of transfers over the next N weeks maximises points, given a
budget, three-per-club limits, a stock of free transfers that banks up to five,
and the option to take -4 hits". That is a mixed-integer program, and solving it
as one is the difference between a tool that suggests obvious moves and a tool
that plans.

Formulation, per gameweek t and player p:

    squad[p, t]     player is in the 15
    start[p, t]     player is in the starting XI
    captain[p, t]   player wears the armband
    buy[p, t]       transferred in ahead of gameweek t
    sell[p, t]      transferred out ahead of gameweek t

Money, free transfers and hits are tracked as continuous/integer variables with
the usual flow constraints. Chips are given to the model as fixed decisions, so
that the chip planner can enumerate scenarios and compare them like for like.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import pulp

from ..config import Rules, get_rules
from ..model.engine import ProjectionSet

log = logging.getLogger(__name__)

# How much a bench slot is worth relative to a starter, by bench order. A first
# substitute comes on reasonably often; the fourth almost never does. These are
# only used inside the MILP — the lineup optimiser does the exact autosub maths.
BENCH_WEIGHTS = (0.16, 0.09, 0.05, 0.03)

# Later gameweeks are discounted: the projection is less certain, and a plan
# that far out will be re-optimised long before it is executed.
DEFAULT_DECAY = 0.86


@dataclass
class ChipPlan:
    """Which chip, if any, is played in each gameweek."""

    wildcard: int | None = None
    freehit: int | None = None
    bench_boost: int | None = None
    triple_captain: int | None = None

    def chip_in(self, gameweek: int) -> str | None:
        for name, gw in (
            ("wildcard", self.wildcard),
            ("freehit", self.freehit),
            ("benchboost", self.bench_boost),
            ("triplecaptain", self.triple_captain),
        ):
            if gw == gameweek:
                return name
        return None

    def describe(self) -> str:
        parts = [
            f"{name} GW{gw}"
            for name, gw in (
                ("Wildcard", self.wildcard),
                ("Free Hit", self.freehit),
                ("Bench Boost", self.bench_boost),
                ("Triple Captain", self.triple_captain),
            )
            if gw is not None
        ]
        return ", ".join(parts) if parts else "no chips"


@dataclass
class GameweekDecision:
    gameweek: int
    squad: list[int]
    starters: list[int]
    bench: list[int]
    captain: int | None
    vice_captain: int | None
    transfers_in: list[int] = field(default_factory=list)
    transfers_out: list[int] = field(default_factory=list)
    hits: int = 0
    free_transfers_before: float = 0.0
    bank_after: float = 0.0
    chip: str | None = None
    expected_points: float = 0.0

    @property
    def net_expected_points(self) -> float:
        return self.expected_points - 4 * self.hits


@dataclass
class OptimisationResult:
    decisions: list[GameweekDecision]
    objective: float
    status: str
    chip_plan: ChipPlan = field(default_factory=ChipPlan)
    infeasible_reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "Optimal"

    @property
    def first(self) -> GameweekDecision | None:
        return self.decisions[0] if self.decisions else None

    def total_expected_points(self) -> float:
        return sum(d.net_expected_points for d in self.decisions)


class SquadOptimiser:
    """Solves the multi-gameweek transfer problem."""

    def __init__(self, rules: Rules | None = None, *, decay: float = DEFAULT_DECAY):
        self.rules = rules or get_rules()
        self.decay = decay

    def solve(
        self,
        projections: ProjectionSet,
        *,
        gameweeks: Sequence[int],
        current_squad: Sequence[int] | None = None,
        selling_prices: dict[int, float] | None = None,
        bank: float = 0.0,
        free_transfers: int = 1,
        chip_plan: ChipPlan | None = None,
        candidate_pool: Sequence[int] | None = None,
        max_transfers_per_gameweek: int = 5,
        max_transfers_first_gameweek: int | None = None,
        forced_out: Sequence[int] = (),
        forced_in: Sequence[int] = (),
        banned: Sequence[int] = (),
        frozen_gameweeks: Sequence[int] = (),
        time_limit: int = 120,
        max_total_hits: int | None = None,
    ) -> OptimisationResult:
        """Optimise transfers over ``gameweeks``.

        With ``current_squad`` empty this becomes an initial squad build: the
        whole 15 is bought in the first gameweek with no transfer cost, which is
        exactly the pre-season problem.
        """
        gameweeks = list(gameweeks)
        chip_plan = chip_plan or ChipPlan()
        current_squad = list(current_squad or [])
        banned_set = set(banned)

        players = self._candidate_players(projections, candidate_pool, current_squad, banned_set)
        if len(players) < 15:
            return OptimisationResult([], 0.0, "Infeasible", chip_plan, "too few candidates")

        prices = {p: projections.price(p) for p in players}
        sell = dict(selling_prices or {})
        for p in current_squad:
            sell.setdefault(p, prices.get(p, projections.price(p)))

        positions = {p: projections.position(p) for p in players}
        clubs = {p: projections.team_of(p) for p in players}
        points = {
            (p, gw): projections.expected_points(p, gw) for p in players for gw in gameweeks
        }

        problem = pulp.LpProblem("fpl_multi_gameweek", pulp.LpMaximize)

        squad = pulp.LpVariable.dicts("squad", (players, gameweeks), cat="Binary")
        start = pulp.LpVariable.dicts("start", (players, gameweeks), cat="Binary")
        captain = pulp.LpVariable.dicts("captain", (players, gameweeks), cat="Binary")
        buy = pulp.LpVariable.dicts("buy", (players, gameweeks), cat="Binary")
        sell_var = pulp.LpVariable.dicts("sell", (players, gameweeks), cat="Binary")
        bench_slots = pulp.LpVariable.dicts(
            "bench", (players, gameweeks, list(range(len(BENCH_WEIGHTS)))), cat="Binary"
        )

        free = pulp.LpVariable.dicts(
            "free_transfers", gameweeks, lowBound=0, upBound=self.rules.max_banked_transfers
        )
        carried = pulp.LpVariable.dicts("carried", gameweeks, lowBound=0)
        carry_flag = pulp.LpVariable.dicts("carry_flag", gameweeks, cat="Binary")
        hits = pulp.LpVariable.dicts("hits", gameweeks, lowBound=0, cat="Integer")
        money = pulp.LpVariable.dicts("bank", gameweeks, lowBound=0)

        squad_positions = self.rules.squad_positions
        formation = self.rules.formation_bounds
        big_m = 20.0

        objective_terms = []

        for index, gameweek in enumerate(gameweeks):
            weight = self.decay**index
            chip = chip_plan.chip_in(gameweek)
            captain_multiplier = (
                self.rules.triple_captain_multiplier if chip == "triplecaptain" else 2
            )

            # --- squad shape ------------------------------------------------
            problem += (
                pulp.lpSum(squad[p][gameweek] for p in players) == self.rules.squad_size,
                f"squad_size_{gameweek}",
            )
            for position, count in squad_positions.items():
                problem += (
                    pulp.lpSum(squad[p][gameweek] for p in players if positions[p] == position)
                    == count,
                    f"squad_{position}_{gameweek}",
                )
            for club in set(clubs.values()):
                problem += (
                    pulp.lpSum(squad[p][gameweek] for p in players if clubs[p] == club)
                    <= self.rules.max_per_club,
                    f"club_{club}_{gameweek}",
                )

            # --- starting XI --------------------------------------------------
            problem += (
                pulp.lpSum(start[p][gameweek] for p in players) == 11,
                f"xi_size_{gameweek}",
            )
            for position, (lo, hi) in formation.items():
                in_position = [p for p in players if positions[p] == position]
                problem += (
                    pulp.lpSum(start[p][gameweek] for p in in_position) >= lo,
                    f"xi_{position}_min_{gameweek}",
                )
                problem += (
                    pulp.lpSum(start[p][gameweek] for p in in_position) <= hi,
                    f"xi_{position}_max_{gameweek}",
                )
            for p in players:
                problem += start[p][gameweek] <= squad[p][gameweek], f"start_in_squad_{p}_{gameweek}"
                problem += (
                    captain[p][gameweek] <= start[p][gameweek],
                    f"captain_starts_{p}_{gameweek}",
                )
            problem += (
                pulp.lpSum(captain[p][gameweek] for p in players) == 1,
                f"one_captain_{gameweek}",
            )

            # --- bench slots --------------------------------------------------
            # Each bench position holds at most one player, and a benched player
            # occupies exactly one slot. This lets the objective value the bench
            # by order rather than treating all four the same.
            for slot in range(len(BENCH_WEIGHTS)):
                problem += (
                    pulp.lpSum(bench_slots[p][gameweek][slot] for p in players) <= 1,
                    f"bench_slot_{slot}_{gameweek}",
                )
            for p in players:
                problem += (
                    pulp.lpSum(bench_slots[p][gameweek][slot] for slot in range(len(BENCH_WEIGHTS)))
                    == squad[p][gameweek] - start[p][gameweek],
                    f"bench_assign_{p}_{gameweek}",
                )

            # --- transfers ----------------------------------------------------
            for p in players:
                previous = (
                    squad[p][gameweeks[index - 1]]
                    if index > 0
                    else (1 if p in current_squad else 0)
                )
                problem += (
                    squad[p][gameweek] == previous + buy[p][gameweek] - sell_var[p][gameweek],
                    f"flow_{p}_{gameweek}",
                )
                problem += (
                    buy[p][gameweek] + sell_var[p][gameweek] <= 1,
                    f"no_churn_{p}_{gameweek}",
                )

            transfers_made = pulp.lpSum(buy[p][gameweek] for p in players)

            if index == 0 and not current_squad:
                # Building a squad from scratch: the initial 15 are free. The
                # free-transfer variable is capped at the banking limit, so it
                # cannot be used to express "unlimited" here — hits are simply
                # forced to zero instead.
                problem += hits[gameweek] == 0, f"no_hits_initial_{gameweek}"
                problem += (
                    free[gameweek] == min(free_transfers, self.rules.max_banked_transfers),
                    f"free_initial_{gameweek}",
                )
            elif gameweek in frozen_gameweeks:
                # The permanent squad cannot move this gameweek — a Free Hit
                # week, where any transfer applies to the temporary squad only.
                # The free transfer is not consumed and still banks.
                problem += transfers_made == 0, f"frozen_{gameweek}"
                problem += hits[gameweek] == 0, f"frozen_no_hits_{gameweek}"
            elif chip in ("wildcard", "freehit"):
                problem += hits[gameweek] == 0, f"chip_no_hits_{gameweek}"
                problem += (
                    transfers_made <= self.rules.squad_size,
                    f"chip_transfer_cap_{gameweek}",
                )
            else:
                # A "what should I do this week" comparison has to leave later
                # gameweeks free: constraining the whole horizon would compare
                # "one transfer now" against "never transfer again", which
                # inflates every option's apparent gain.
                cap = max_transfers_per_gameweek
                if index == 0 and max_transfers_first_gameweek is not None:
                    cap = max_transfers_first_gameweek
                problem += (transfers_made <= cap, f"transfer_cap_{gameweek}")
                problem += (
                    hits[gameweek] >= transfers_made - free[gameweek],
                    f"hits_{gameweek}",
                )

            # Free transfer dynamics: next = min(cap, max(0, free - used) + 1).
            # `carried` is the max(0, ...) term, linearised with a big-M switch.
            if index == 0 and current_squad:
                problem += (
                    free[gameweek] == min(free_transfers, self.rules.max_banked_transfers),
                    f"free_start_{gameweek}",
                )
            if index + 1 < len(gameweeks):
                next_gameweek = gameweeks[index + 1]
                used = 0 if chip in ("wildcard", "freehit") else transfers_made
                problem += carried[gameweek] >= free[gameweek] - used, f"carry_lb_{gameweek}"
                problem += (
                    carried[gameweek] <= free[gameweek] - used + big_m * carry_flag[gameweek],
                    f"carry_ub_a_{gameweek}",
                )
                problem += (
                    carried[gameweek] <= big_m * (1 - carry_flag[gameweek]),
                    f"carry_ub_b_{gameweek}",
                )
                problem += (
                    free[next_gameweek] <= carried[gameweek] + 1,
                    f"free_next_{gameweek}",
                )
                problem += (
                    free[next_gameweek] <= self.rules.max_banked_transfers,
                    f"free_cap_{gameweek}",
                )

            # --- money --------------------------------------------------------
            spend = pulp.lpSum(prices[p] * buy[p][gameweek] for p in players)
            income = pulp.lpSum(sell.get(p, prices[p]) * sell_var[p][gameweek] for p in players)
            previous_bank = money[gameweeks[index - 1]] if index > 0 else bank
            problem += money[gameweek] == previous_bank + income - spend, f"bank_{gameweek}"

            # --- objective ------------------------------------------------------
            for p in players:
                expected = points[(p, gameweek)]
                objective_terms.append(weight * expected * start[p][gameweek])
                objective_terms.append(
                    weight * expected * (captain_multiplier - 1) * captain[p][gameweek]
                )
                if chip == "benchboost":
                    objective_terms.append(
                        weight * expected * (squad[p][gameweek] - start[p][gameweek])
                    )
                else:
                    for slot, bench_weight in enumerate(BENCH_WEIGHTS):
                        objective_terms.append(
                            weight * expected * bench_weight * bench_slots[p][gameweek][slot]
                        )
            objective_terms.append(-weight * self.rules.hit_cost * hits[gameweek])

        # --- global constraints -------------------------------------------------
        for p in forced_in:
            if p in players:
                problem += squad[p][gameweeks[0]] == 1, f"forced_in_{p}"
        for p in forced_out:
            if p in players:
                problem += squad[p][gameweeks[0]] == 0, f"forced_out_{p}"
        if max_total_hits is not None:
            problem += (
                pulp.lpSum(hits[gw] for gw in gameweeks) <= max_total_hits,
                "total_hits_cap",
            )

        # A free hit squad is reverted the following gameweek. Rather than model
        # the revert inside the flow constraints, the chip planner solves free
        # hit weeks as a standalone one-week problem; see `optimize.chips`.

        problem += pulp.lpSum(objective_terms)

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
        problem.solve(solver)
        status = pulp.LpStatus[problem.status]
        if status != "Optimal":
            return OptimisationResult(
                [], 0.0, status, chip_plan, f"solver returned {status}"
            )

        decisions = self._extract(
            problem,
            projections,
            players,
            gameweeks,
            squad,
            start,
            captain,
            buy,
            sell_var,
            hits,
            free,
            money,
            chip_plan,
        )
        return OptimisationResult(
            decisions, float(pulp.value(problem.objective) or 0.0), status, chip_plan
        )

    # -- helpers -------------------------------------------------------------

    def _candidate_players(
        self,
        projections: ProjectionSet,
        candidate_pool: Sequence[int] | None,
        current_squad: Sequence[int],
        banned: set[int],
    ) -> list[int]:
        if candidate_pool is not None:
            pool = set(candidate_pool)
        else:
            pool = set(shortlist(projections))
        pool |= set(current_squad)
        pool -= banned
        return sorted(pool)

    @staticmethod
    def _extract(
        problem,
        projections: ProjectionSet,
        players: Sequence[int],
        gameweeks: Sequence[int],
        squad,
        start,
        captain,
        buy,
        sell_var,
        hits,
        free,
        money,
        chip_plan: ChipPlan,
    ) -> list[GameweekDecision]:
        def chosen(var) -> bool:
            return bool(var.value() and var.value() > 0.5)

        decisions: list[GameweekDecision] = []
        for gameweek in gameweeks:
            in_squad = [p for p in players if chosen(squad[p][gameweek])]
            starters = [p for p in players if chosen(start[p][gameweek])]
            bench = [p for p in in_squad if p not in starters]
            captains = [p for p in players if chosen(captain[p][gameweek])]

            # Vice captain: the best remaining starter, since the MILP does not
            # need to model it (it only pays out if the captain does not play).
            ranked = sorted(
                (p for p in starters if p not in captains),
                key=lambda p: -projections.expected_points(p, gameweek),
            )
            expected = sum(projections.expected_points(p, gameweek) for p in starters)
            if captains:
                expected += projections.expected_points(captains[0], gameweek)

            decisions.append(
                GameweekDecision(
                    gameweek=gameweek,
                    squad=in_squad,
                    starters=starters,
                    bench=bench,
                    captain=captains[0] if captains else None,
                    vice_captain=ranked[0] if ranked else None,
                    transfers_in=[p for p in players if chosen(buy[p][gameweek])],
                    transfers_out=[p for p in players if chosen(sell_var[p][gameweek])],
                    hits=int(round(hits[gameweek].value() or 0)),
                    free_transfers_before=float(free[gameweek].value() or 0),
                    bank_after=float(money[gameweek].value() or 0),
                    chip=chip_plan.chip_in(gameweek),
                    expected_points=float(expected),
                )
            )
        return decisions


def shortlist(
    projections: ProjectionSet,
    *,
    per_position: dict[str, int] | None = None,
    min_price_coverage: int = 6,
) -> list[int]:
    """Reduce the player universe to a tractable, sensible candidate pool.

    The MILP scales with the number of players, and 700 of them makes it slow
    without making it smarter — most are unpickable. The pool keeps the best by
    expected points in each position, and separately keeps the cheapest options
    in each position, because budget enablers matter to the solution even though
    they never look good on points alone.
    """
    per_position = per_position or {"GKP": 25, "DEF": 70, "MID": 80, "FWD": 45}
    pool: set[int] = set()

    by_position: dict[str, list[int]] = {}
    for element, player in projections.players.items():
        if projections.minutes.get(element) and projections.minutes[element].p_appear < 0.02:
            continue
        by_position.setdefault(player.position, []).append(element)

    for position, elements in by_position.items():
        ranked = sorted(elements, key=lambda e: -projections.horizon_points(e))
        pool.update(ranked[: per_position.get(position, 40)])

        # Cheap enablers: the best few players at each low price point.
        cheapest = sorted(elements, key=lambda e: (projections.price(e), -projections.horizon_points(e)))
        pool.update(cheapest[:min_price_coverage])

        # Best value per million, which is where bench fodder actually comes from.
        by_value = sorted(
            elements,
            key=lambda e: -(projections.horizon_points(e) / max(projections.price(e), 0.1)),
        )
        pool.update(by_value[:20])

    return sorted(pool)
