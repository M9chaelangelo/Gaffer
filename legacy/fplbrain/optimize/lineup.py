"""Starting XI, bench order and captaincy.

The MILP picks a squad; this module decides what to do with it on the day. Three
decisions, each with its own subtlety:

**Starting XI** — not simply the eleven highest projections, because the
formation has to be legal and because a defender with a high floor can be worth
more than a midfielder with a higher mean once autosubs are accounted for.

**Bench order** — worth roughly half a point a week on average and considerably
more in a blank gameweek. The correct objective is the expected value of the
autosub chain, which depends on each bench player's chance of *not playing* as
well as his points if he does.

**Captaincy** — the highest-variance decision in the game. Maximising expected
points is right if you are trying to maximise expected points, but that is not
always the goal: a manager chasing a rank needs variance and a manager
protecting one needs safety. The objective is configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable, Sequence

import numpy as np

from ..config import Rules, get_rules
from ..model.engine import ProjectionSet
from ..model.points import GRID, PlayerGameweekProjection

# Legal outfield shapes: (defenders, midfielders, forwards) summing to 10.
VALID_FORMATIONS = [
    (d, m, f)
    for d in range(3, 6)
    for m in range(2, 6)
    for f in range(1, 4)
    if d + m + f == 10
]


@dataclass
class LineupDecision:
    gameweek: int
    starters: list[int]
    bench: list[int]  # in bench order, goalkeeper first
    captain: int
    vice_captain: int
    formation: tuple[int, int, int]
    expected_points: float
    expected_points_no_autosub: float
    chip: str | None = None

    @property
    def autosub_value(self) -> float:
        return self.expected_points - self.expected_points_no_autosub

    def formation_label(self) -> str:
        return "-".join(str(x) for x in self.formation)


@dataclass
class CaptainCandidate:
    element: int
    name: str
    expected_points: float
    ceiling: float  # probability of a double-figure haul
    floor: float  # probability of two points or fewer
    effective_ownership: float
    differential_score: float
    rank_utility: float
    recommended: bool = False
    reason: str = ""


def choose_starting_eleven(
    squad: Sequence[int],
    projections: ProjectionSet,
    gameweek: int,
    *,
    bench_boost: bool = False,
) -> tuple[list[int], list[int], tuple[int, int, int]]:
    """Pick the XI and the bench, trying every legal formation.

    Under Bench Boost every player scores, so the formation stops mattering for
    the total and the XI is chosen only to place the captain sensibly.
    """
    by_position: dict[str, list[int]] = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
    for element in squad:
        by_position[projections.position(element)].append(element)

    for position in by_position:
        by_position[position].sort(key=lambda e: -projections.expected_points(e, gameweek))

    best: tuple[float, list[int], tuple[int, int, int]] | None = None
    for defenders, midfielders, forwards in VALID_FORMATIONS:
        if (
            len(by_position["DEF"]) < defenders
            or len(by_position["MID"]) < midfielders
            or len(by_position["FWD"]) < forwards
            or len(by_position["GKP"]) < 1
        ):
            continue
        eleven = (
            by_position["GKP"][:1]
            + by_position["DEF"][:defenders]
            + by_position["MID"][:midfielders]
            + by_position["FWD"][:forwards]
        )
        total = sum(projections.expected_points(e, gameweek) for e in eleven)
        if best is None or total > best[0]:
            best = (total, eleven, (defenders, midfielders, forwards))

    if best is None:  # pragma: no cover - only if the squad is malformed
        eleven = list(squad)[:11]
        return eleven, list(squad)[11:], (4, 4, 2)

    _, eleven, formation = best
    bench = [e for e in squad if e not in eleven]
    return eleven, bench, formation


def order_bench(
    bench: Sequence[int],
    starters: Sequence[int],
    projections: ProjectionSet,
    gameweek: int,
    *,
    draws: int = 4000,
    seed: int = 7,
) -> tuple[list[int], float]:
    """Find the bench order that maximises expected points after autosubs.

    The outfield bench has at most 3! = 6 orderings, so this evaluates all of
    them exactly by simulation rather than reasoning about them heuristically.
    The goalkeeper is not ordered: he can only ever replace the other keeper.
    """
    keepers = [e for e in bench if projections.position(e) == "GKP"]
    outfield = [e for e in bench if projections.position(e) != "GKP"]

    if not outfield:
        return list(keepers), 0.0

    best_order: list[int] = list(outfield)
    best_value = -np.inf
    for candidate in permutations(outfield):
        value = _simulate_lineup_points(
            starters, list(candidate), keepers, projections, gameweek, draws=draws, seed=seed
        )
        if value > best_value:
            best_value = value
            best_order = list(candidate)

    baseline = _simulate_lineup_points(
        starters, best_order, keepers, projections, gameweek, draws=draws, seed=seed,
        allow_autosubs=False,
    )
    return keepers + best_order, float(best_value - baseline)


def _sample_player(
    projection: PlayerGameweekProjection, rng: np.random.Generator, draws: int
) -> tuple[np.ndarray, np.ndarray]:
    """Draw points and an "appeared" flag for one player.

    Appearance is not independent of points: a player who scores zero has often
    not played. The pmf already mixes over minutes, so appearance is drawn from
    the minutes model directly and points are drawn conditionally.
    """
    pmf = projection.pmf
    points = rng.choice(GRID, size=draws, p=pmf / pmf.sum())
    p_appear = 0.0
    for fixture in projection.fixtures:
        if fixture.minutes is not None:
            p_appear = max(p_appear, fixture.minutes.p_appear)
    if not projection.fixtures:
        p_appear = 0.0
    appeared = rng.random(draws) < p_appear
    # A player who did not appear scores nothing.
    points = np.where(appeared, points, 0)
    return points, appeared


def _simulate_lineup_points(
    starters: Sequence[int],
    bench_outfield: Sequence[int],
    bench_keepers: Sequence[int],
    projections: ProjectionSet,
    gameweek: int,
    *,
    draws: int = 4000,
    seed: int = 7,
    allow_autosubs: bool = True,
    captain: int | None = None,
    captain_multiplier: int = 2,
    vice_captain: int | None = None,
) -> float:
    """Mean gameweek points for a lineup, with the autosub chain resolved."""
    rng = np.random.default_rng(seed)

    all_players = list(starters) + list(bench_outfield) + list(bench_keepers)
    samples: dict[int, np.ndarray] = {}
    appeared: dict[int, np.ndarray] = {}
    for element in all_players:
        points, played = _sample_player(projections.get(element, gameweek), rng, draws)
        samples[element] = points
        appeared[element] = played

    total = np.zeros(draws, dtype=float)
    counted = {element: np.array(appeared[element], dtype=bool) for element in starters}

    for element in starters:
        total += samples[element] * counted[element]

    if allow_autosubs:
        positions = {e: projections.position(e) for e in all_players}
        # Only draws with an absent starter need the substitution chain resolved,
        # which is a small minority of them.
        absent_any = np.zeros(draws, dtype=bool)
        for element in starters:
            absent_any |= ~counted[element]

        for draw in np.flatnonzero(absent_any):
            missing = [e for e in starters if not counted[e][draw]]
            live = [positions[e] for e in starters if counted[e][draw]]
            used: set[int] = set()

            for out in missing:
                if positions[out] == "GKP":
                    replacement = next(
                        (k for k in bench_keepers if appeared[k][draw] and k not in used),
                        None,
                    )
                    if replacement is not None:
                        used.add(replacement)
                        total[draw] += samples[replacement][draw]
                        live.append("GKP")
                    continue

                for candidate in bench_outfield:
                    if candidate in used or not appeared[candidate][draw]:
                        continue
                    if _formation_legal(live + [positions[candidate]]):
                        used.add(candidate)
                        total[draw] += samples[candidate][draw]
                        live.append(positions[candidate])
                        break

    if captain is not None:
        captain_points = samples.get(captain)
        if captain_points is not None:
            captain_played = appeared[captain]
            total += captain_points * (captain_multiplier - 1) * captain_played
            if vice_captain is not None and vice_captain in samples:
                # The vice only takes over when the captain does not appear.
                total += (
                    samples[vice_captain]
                    * (captain_multiplier - 1)
                    * (~captain_played)
                    * appeared[vice_captain]
                )

    return float(total.mean())


def _formation_legal(positions: Sequence[str]) -> bool:
    """Whether a set of on-pitch positions forms a legal FPL team.

    FPL only makes a substitution if the resulting team still has at least three
    defenders and one forward, and never more than one goalkeeper.
    """
    counts = {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for position in positions:
        counts[position] += 1
    if counts["GKP"] > 1:
        return False
    return counts["DEF"] >= 3 and counts["FWD"] >= 1 and len(positions) <= 11


def optimise_lineup(
    squad: Sequence[int],
    projections: ProjectionSet,
    gameweek: int,
    *,
    chip: str | None = None,
    captain_objective: str = "ev",
    effective_ownership: dict[int, float] | None = None,
    rules: Rules | None = None,
    draws: int = 3000,
) -> LineupDecision:
    """Full matchday decision for one gameweek."""
    rules = rules or get_rules()
    bench_boost = chip == "benchboost"

    starters, bench, formation = choose_starting_eleven(
        squad, projections, gameweek, bench_boost=bench_boost
    )
    bench_order, _ = order_bench(bench, starters, projections, gameweek, draws=draws)

    candidates = captaincy_analysis(
        starters,
        projections,
        gameweek,
        objective=captain_objective,
        effective_ownership=effective_ownership,
    )
    captain = candidates[0].element if candidates else starters[0]
    vice = candidates[1].element if len(candidates) > 1 else starters[0]

    multiplier = rules.triple_captain_multiplier if chip == "triplecaptain" else 2
    keepers = [e for e in bench_order if projections.position(e) == "GKP"]
    outfield = [e for e in bench_order if projections.position(e) != "GKP"]

    expected = _simulate_lineup_points(
        starters,
        outfield,
        keepers,
        projections,
        gameweek,
        draws=draws,
        captain=captain,
        captain_multiplier=multiplier,
        vice_captain=vice,
    )
    expected_no_autosub = _simulate_lineup_points(
        starters,
        outfield,
        keepers,
        projections,
        gameweek,
        draws=draws,
        allow_autosubs=False,
        captain=captain,
        captain_multiplier=multiplier,
        vice_captain=vice,
    )
    if bench_boost:
        expected += sum(projections.expected_points(e, gameweek) for e in bench_order)
        expected_no_autosub += sum(projections.expected_points(e, gameweek) for e in bench_order)

    return LineupDecision(
        gameweek=gameweek,
        starters=starters,
        bench=bench_order,
        captain=captain,
        vice_captain=vice,
        formation=formation,
        expected_points=expected,
        expected_points_no_autosub=expected_no_autosub,
        chip=chip,
    )


def captaincy_analysis(
    candidates: Iterable[int],
    projections: ProjectionSet,
    gameweek: int,
    *,
    objective: str = "ev",
    effective_ownership: dict[int, float] | None = None,
    template_threshold: float = 25.0,
) -> list[CaptainCandidate]:
    """Rank captaincy options under the chosen objective.

    ``rank`` utility trades expected points against how many rivals share the
    pick. Captaining a 60%-owned player is close to a no-op for rank: you gain
    on the 40% who did not, and lose nothing to the 60% who did. A 5%-owned
    captain with a similar projection moves rank far more in both directions.
    """
    effective_ownership = effective_ownership or {}
    rows: list[CaptainCandidate] = []

    for element in candidates:
        projection = projections.get(element, gameweek)
        if projection.n_fixtures == 0:
            continue
        expected = projection.expected_points
        ceiling = projection.p_haul
        floor = projection.p_blank
        player = projections.players.get(element)
        ownership = effective_ownership.get(
            element, player.selected_by_percent if player else 0.0
        )

        # Differential value: how much of the field you gain ground on if this
        # captain hauls, scaled by how likely the haul is.
        differential = ceiling * (1.0 - min(ownership, 100.0) / 100.0)

        if objective == "ceiling":
            utility = expected + 6.0 * ceiling
        elif objective == "rank":
            # Expected points still dominates; ownership adjusts at the margin.
            crowding = min(ownership, 100.0) / 100.0
            utility = expected * (1.0 - 0.18 * crowding) + 4.0 * differential
        else:
            utility = expected

        rows.append(
            CaptainCandidate(
                element=element,
                name=projections.name(element),
                expected_points=expected,
                ceiling=ceiling,
                floor=floor,
                effective_ownership=ownership,
                differential_score=differential,
                rank_utility=utility,
            )
        )

    rows.sort(key=lambda r: -r.rank_utility)
    if rows:
        rows[0].recommended = True
        rows[0].reason = _captain_reason(rows[0], rows[1] if len(rows) > 1 else None, objective)
    return rows


def _captain_reason(
    best: CaptainCandidate, runner_up: CaptainCandidate | None, objective: str
) -> str:
    parts = [f"{best.expected_points:.1f} xPts", f"{best.ceiling:.0%} chance of 10+"]
    if best.effective_ownership:
        parts.append(f"{best.effective_ownership:.0f}% owned")
    if runner_up is not None:
        gap = best.expected_points - runner_up.expected_points
        if abs(gap) < 0.3:
            parts.append(
                f"effectively tied with {runner_up.name} on points; "
                f"{'differential' if objective == 'rank' else 'ceiling'} breaks the tie"
            )
        else:
            parts.append(f"{gap:+.1f} xPts clear of {runner_up.name}")
    return "; ".join(parts)


def bench_order_report(
    bench: Sequence[int], projections: ProjectionSet, gameweek: int
) -> list[dict]:
    """Explain a bench order: points if they play, and how likely that is."""
    rows = []
    for order, element in enumerate(bench, start=1):
        projection = projections.get(element, gameweek)
        p_appear = 0.0
        for fixture in projection.fixtures:
            if fixture.minutes is not None:
                p_appear = max(p_appear, fixture.minutes.p_appear)
        rows.append(
            {
                "order": order,
                "element": element,
                "name": projections.name(element),
                "position": projections.position(element),
                "xpts": round(projection.expected_points, 2),
                "p_appear": round(p_appear, 3),
                "fixtures": projection.n_fixtures,
            }
        )
    return rows


@dataclass
class LineupChange:
    """One difference between the squad as you have it set and as modelled."""

    kind: str  # "captain", "vice", "start", "bench_order"
    detail: str
    gain: float = 0.0

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.detail


def diff_lineup(
    squad,
    lineup: LineupDecision,
    projections: ProjectionSet,
    gameweek: int,
) -> list[LineupChange]:
    """Compare the lineup you have set against the one the model would set.

    The engine has always computed its own XI, armband and bench order, but it
    reported them as if from nothing — leaving you to spot the differences
    against your real team by eye. These are the free points: no transfer, no
    hit, just moving what you already own.
    """
    changes: list[LineupChange] = []
    current_starters = {p.element for p in squad.picks if p.is_starter}
    current_bench = [p.element for p in sorted(squad.picks, key=lambda p: p.position)
                     if not p.is_starter]

    def points(element: int) -> float:
        return projections.expected_points(element, gameweek)

    # -- armband -------------------------------------------------------------
    if squad.captain is not None and squad.captain != lineup.captain:
        gain = points(lineup.captain) - points(squad.captain)
        changes.append(
            LineupChange(
                "captain",
                f"Captain {projections.name(squad.captain)} → "
                f"{projections.name(lineup.captain)} ({gain:+.1f} xPts on the double)",
                gain,
            )
        )

    if squad.vice_captain is not None and squad.vice_captain != lineup.vice_captain:
        current_vice_position = projections.position(squad.vice_captain)
        note = (
            f"Vice {projections.name(squad.vice_captain)} → "
            f"{projections.name(lineup.vice_captain)}"
        )
        # A goalkeeper as vice is a common and costly slip: if the captain does
        # not play, the armband lands on the lowest-ceiling player in the squad.
        if current_vice_position == "GKP":
            note += " — a goalkeeper as vice caps your upside if the captain sits out"
        changes.append(LineupChange("vice", note))

    # -- starting eleven -----------------------------------------------------
    to_start = [e for e in lineup.starters if e not in current_starters]
    to_bench = [e for e in current_starters if e not in lineup.starters]
    for coming_in, going_out in zip(to_start, to_bench):
        gain = points(coming_in) - points(going_out)
        changes.append(
            LineupChange(
                "start",
                f"Start {projections.name(coming_in)} over "
                f"{projections.name(going_out)} ({gain:+.1f} xPts)",
                gain,
            )
        )

    # -- bench order ---------------------------------------------------------
    modelled_bench = [e for e in lineup.bench if projections.position(e) != "GKP"]
    current_outfield = [e for e in current_bench if projections.position(e) != "GKP"]
    if current_outfield and modelled_bench and current_outfield != modelled_bench:
        order = " → ".join(projections.name(e) for e in modelled_bench)
        changes.append(
            LineupChange(
                "bench_order",
                f"Reorder the bench: {order} "
                f"(worth {lineup.autosub_value:+.2f} xPts through autosubs)",
                lineup.autosub_value,
            )
        )

    return changes
