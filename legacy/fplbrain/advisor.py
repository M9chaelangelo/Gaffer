"""The decision layer: everything the engine wants to tell you, in one object.

The modules below this one each answer a narrow question. This one asks all of
them on your behalf, in the right order, and assembles an answer that a person
can act on: what to do this week, what it is worth, what the alternatives were,
and what would change the recommendation.

Nothing here does any modelling. If a number looks wrong, it came from
:mod:`fplbrain.model`; if a decision looks wrong, it came from
:mod:`fplbrain.optimize`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence


from .config import Rules, Settings, get_rules, get_settings
from .data.entities import ManagerSquad
from .data.crowd import CrowdClient, CrowdSnapshot
from .data.fpl import FplClient, summarise_availability
from .model.engine import ProjectionEngine, ProjectionSet
from .optimize.chips import ChipOpportunity, ChipPlanner
from .optimize.lineup import (
    CaptainCandidate,
    LineupChange,
    LineupDecision,
    bench_order_report,
    captaincy_analysis,
    diff_lineup,
    optimise_lineup,
)
from .optimize.milp import GameweekDecision, SquadOptimiser, shortlist
from .sim.montecarlo import GameweekSimulator, simulate_squad
from .sim.rivals import FieldModel, RankProjection, RivalFieldSimulator, build_field_model
from .sim.scenarios import (
    PathOutcome,
    PathSimulator,
    RecommendedPath,
    compare,
    default_strategies,
    recommend_paths,
)

log = logging.getLogger(__name__)


@dataclass
class TransferOption:
    """One candidate move, with what it costs and what it buys."""

    out_elements: list[int]
    in_elements: list[int]
    gain: float
    hits: int
    net_gain: float
    horizon: list[int]
    description: str = ""

    @property
    def is_hold(self) -> bool:
        return not self.in_elements


@dataclass
class Advice:
    """The complete answer for one gameweek."""

    gameweek: int
    deadline: datetime | None
    squad: ManagerSquad
    projections: ProjectionSet
    plan: list[GameweekDecision]
    lineup: LineupDecision
    captains: list[CaptainCandidate]
    bench: list[dict]
    transfer_options: list[TransferOption]
    chip_opportunities: list[ChipOpportunity]
    chip_advice: list[str]
    lineup_changes: list[LineupChange] = field(default_factory=list)
    paths: list[PathOutcome] = field(default_factory=list)
    routes: list[RecommendedPath] = field(default_factory=list)
    crowd: CrowdSnapshot | None = None
    rank: RankProjection | None = None
    field_model: FieldModel | None = None
    outcome_summary: dict[str, float] = field(default_factory=dict)
    risks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def recommended_transfer(self) -> TransferOption | None:
        return self.transfer_options[0] if self.transfer_options else None

    def headline(self) -> str:
        move = self.recommended_transfer
        if move is None or move.is_hold:
            action = "Hold — roll the transfer"
        else:
            outs = ", ".join(self.projections.name(e) for e in move.out_elements)
            ins = ", ".join(self.projections.name(e) for e in move.in_elements)
            action = f"{outs} → {ins}"
            if move.hits:
                action += f" (-{move.hits * 4})"
        return f"GW{self.gameweek}: {action}; captain {self.projections.name(self.lineup.captain)}"


class Advisor:
    """Runs the whole pipeline and produces :class:`Advice`."""

    def __init__(
        self,
        client: FplClient,
        *,
        settings: Settings | None = None,
        rules: Rules | None = None,
    ):
        self.client = client
        self.settings = settings or get_settings()
        self.rules = rules or get_rules()

    def advise(
        self,
        entry_id: int | None = None,
        *,
        gameweek: int | None = None,
        horizon: int | None = None,
        projections: ProjectionSet | None = None,
        squad: ManagerSquad | None = None,
        max_hits: int = 1,
        include_paths: bool = True,
        include_rank: bool = True,
        include_chips: bool = True,
        crowd_sample: int = 0,
        draws: int = 4000,
    ) -> Advice:
        boot = self.client.bootstrap()
        next_gameweek = boot.next_gameweek()
        gameweek = gameweek or (next_gameweek.id if next_gameweek else 1)
        horizon_length = horizon or self.settings.planning_horizon
        gameweeks = [gw for gw in range(gameweek, min(gameweek + horizon_length, 39))]

        if squad is None:
            entry_id = entry_id or self.settings.entry_id
            if entry_id is None:
                raise ValueError(
                    "No FPL entry id configured. Set manager.entry_id in "
                    "config/settings.yaml, or pass --entry."
                )
            squad = self.client.squad(entry_id)

        if projections is None:
            engine = ProjectionEngine(self.client, settings=self.settings, rules=self.rules)
            projections = engine.run(gameweeks)

        notes = list(projections.notes)
        if squad.source == "public":
            notes.append(
                "Squad read from the last published gameweek. Bank and free "
                "transfers are inferred, not read — add an FPL cookie for exact values."
            )

        selling = squad.selling_prices({e: projections.price(e) for e in squad.element_ids})
        pool = shortlist(projections)
        optimiser = SquadOptimiser(self.rules)

        plan_result = optimiser.solve(
            projections,
            gameweeks=gameweeks,
            current_squad=squad.element_ids,
            selling_prices=selling,
            bank=squad.bank,
            free_transfers=squad.free_transfers,
            candidate_pool=pool,
            max_total_hits=max_hits,
            time_limit=60,
        )
        if not plan_result.ok:
            notes.append(f"Optimiser returned {plan_result.status}; showing your current squad.")
            plan_decisions: list[GameweekDecision] = []
            working_squad = list(squad.element_ids)
        else:
            plan_decisions = plan_result.decisions
            working_squad = plan_decisions[0].squad if plan_decisions else list(squad.element_ids)

        transfer_options = self._transfer_options(
            projections, squad, selling, pool, gameweeks, optimiser, max_hits
        )

        chip = None
        lineup = optimise_lineup(
            working_squad,
            projections,
            gameweek,
            chip=chip,
            captain_objective=str(self.settings.risk.get("captain_objective", "ev")),
            rules=self.rules,
            draws=min(draws, 3000),
        )

        crowd = self._sample_crowd(boot, crowd_sample)
        if crowd is not None and crowd.available:
            notes.append(
                f"Captaincy shares measured from {len(crowd)} top-ranked squads in "
                f"GW{crowd.gameweek} (±{crowd.margin_of_error():.0f}pp), not modelled."
            )
        elif crowd is not None and crowd.note:
            notes.append(crowd.note)

        field_model = build_field_model(
            projections, gameweek, boot.total_players, crowd=crowd
        )
        captains = captaincy_analysis(
            lineup.starters,
            projections,
            gameweek,
            objective=str(self.settings.risk.get("captain_objective", "ev")),
            effective_ownership=field_model.effective_ownership(),
        )
        bench = bench_order_report(lineup.bench, projections, gameweek)
        # Free points: what to change without spending a transfer.
        lineup_changes = diff_lineup(squad, lineup, projections, gameweek)

        chip_opportunities: list[ChipOpportunity] = []
        chip_advice: list[str] = []
        if include_chips:
            planner = ChipPlanner(self.rules, optimiser)
            chip_opportunities = planner.opportunities(
                working_squad,
                projections,
                gameweeks=gameweeks,
                bank=squad.bank,
                selling_prices=selling,
                free_transfers=squad.free_transfers,
                chips_used=squad.chips_used,
                include_slow=len(gameweeks) <= 8,
            )
            chip_set = self.rules.chip_set_for_gameweek(gameweek)
            deadline_gameweek = self.rules.chip_window(chip_set)[1] if chip_set else None
            chip_advice = planner.recommend(
                chip_opportunities, deadline_gameweek=deadline_gameweek
            )

        outcome_summary: dict[str, float] = {}
        rank: RankProjection | None = None
        if include_rank:
            outcome_summary, rank = self._simulate_gameweek(
                projections, working_squad, lineup, gameweek, field_model, draws
            )

        paths: list[PathOutcome] = []
        if include_paths:
            paths = self._compare_paths(
                projections,
                squad,
                selling,
                gameweeks,
                chip_opportunities,
                optimiser,
                draws=min(draws, 3000),
            )

        return Advice(
            gameweek=gameweek,
            deadline=next_gameweek.deadline if next_gameweek else None,
            squad=squad,
            projections=projections,
            plan=plan_decisions,
            lineup=lineup,
            captains=captains,
            bench=bench,
            lineup_changes=lineup_changes,
            transfer_options=transfer_options,
            chip_opportunities=chip_opportunities,
            chip_advice=chip_advice,
            paths=paths,
            routes=recommend_paths(paths, projections),
            crowd=crowd,
            rank=rank,
            field_model=field_model,
            outcome_summary=outcome_summary,
            risks=self._risks(projections, squad, gameweek),
            notes=notes,
        )

    # -- pieces --------------------------------------------------------------

    def _transfer_options(
        self,
        projections: ProjectionSet,
        squad: ManagerSquad,
        selling: dict[int, float],
        pool: Sequence[int],
        gameweeks: Sequence[int],
        optimiser: SquadOptimiser,
        max_hits: int,
    ) -> list[TransferOption]:
        """Rank holding, one transfer, and a hit against each other.

        Only *this* gameweek's transfers are constrained; later weeks stay free
        in every option. That is what makes the comparison honest — the question
        is "what should I do now, given that I will keep optimising", not "what
        if I never transfer again".

        Gains are quoted against holding, over the full planning horizon, so a
        -4 has to earn its four points somewhere in the window rather than
        immediately.
        """
        options: list[TransferOption] = []
        baseline = optimiser.solve(
            projections,
            gameweeks=gameweeks,
            current_squad=squad.element_ids,
            selling_prices=selling,
            bank=squad.bank,
            free_transfers=squad.free_transfers,
            candidate_pool=pool,
            max_transfers_first_gameweek=0,
            max_total_hits=0,
            time_limit=25,
        )
        baseline_points = baseline.total_expected_points() if baseline.ok else 0.0

        for transfers, hits_allowed, label in (
            (0, 0, "Hold and roll the transfer"),
            (1, 0, "One free transfer"),
            (2, max(0, max_hits), "Two moves, taking a hit if it pays"),
            (3, max(0, max_hits + 1), "Three moves"),
        ):
            if transfers > 0 and transfers > squad.free_transfers + max_hits:
                continue
            result = optimiser.solve(
                projections,
                gameweeks=gameweeks,
                current_squad=squad.element_ids,
                selling_prices=selling,
                bank=squad.bank,
                free_transfers=squad.free_transfers,
                candidate_pool=pool,
                max_transfers_first_gameweek=transfers,
                max_total_hits=hits_allowed,
                time_limit=25,
            )
            if not result.ok or not result.decisions:
                continue
            first = result.decisions[0]
            total = result.total_expected_points()
            options.append(
                TransferOption(
                    out_elements=first.transfers_out,
                    in_elements=first.transfers_in,
                    gain=total - baseline_points,
                    hits=first.hits,
                    net_gain=total - baseline_points,
                    horizon=list(gameweeks),
                    description=label,
                )
            )

        # Deduplicate plans that came out identical under different caps.
        seen: set[tuple] = set()
        unique: list[TransferOption] = []
        for option in sorted(options, key=lambda o: -o.net_gain):
            key = (tuple(sorted(option.out_elements)), tuple(sorted(option.in_elements)))
            if key in seen:
                continue
            seen.add(key)
            unique.append(option)
        return unique

    def _simulate_gameweek(
        self,
        projections: ProjectionSet,
        squad: Sequence[int],
        lineup: LineupDecision,
        gameweek: int,
        field_model: FieldModel,
        draws: int,
    ) -> tuple[dict[str, float], RankProjection | None]:
        universe = sorted(
            {
                element
                for element, player in projections.players.items()
                if player.selected_by_percent > 0.4
            }
            | set(squad)
        )
        simulator = GameweekSimulator(
            projections, rules=self.rules, seed=int(self.settings.get("simulation.seed", 1))
        )
        sample = simulator.simulate(universe, gameweek, draws=draws)
        outcome = simulate_squad(
            sample,
            lineup.starters,
            lineup.bench,
            lineup.captain,
            lineup.vice_captain,
            projections,
        )
        rivals = RivalFieldSimulator(projections, field_model, rules=self.rules)
        field_scores = rivals.score_field(
            sample,
            n_rivals=int(self.settings.get("simulation.rival_field", 2000)),
            max_draws=min(draws, 2000),
        )
        rank = rivals.project_rank(outcome.total, field_scores)
        return outcome.summary(), rank

    def _compare_paths(
        self,
        projections: ProjectionSet,
        squad: ManagerSquad,
        selling: dict[int, float],
        gameweeks: Sequence[int],
        chip_opportunities: Sequence[ChipOpportunity],
        optimiser: SquadOptimiser,
        draws: int,
    ) -> list[PathOutcome]:
        # `chip_opportunities` arrives sorted by gain, so the first entry for a
        # chip is its best gameweek.
        best_by_chip: dict[str, int] = {}
        for opportunity in chip_opportunities:
            best_by_chip.setdefault(opportunity.chip, opportunity.gameweek)

        planner = ChipPlanner(self.rules, optimiser)
        available = planner.available_chips(gameweeks[0], squad.chips_used)
        strategies = default_strategies(available, gameweeks, best_by_chip)

        simulator = PathSimulator(projections, rules=self.rules, optimiser=optimiser)
        return simulator.run(
            strategies,
            squad=squad.element_ids,
            gameweeks=gameweeks,
            bank=squad.bank,
            selling_prices=selling,
            free_transfers=squad.free_transfers,
            draws=draws,
        )

    def _sample_crowd(self, boot, sample_size: int) -> CrowdSnapshot | None:
        """Read what the best managers actually did, if asked to.

        Off by default: it costs one request per manager sampled, which is a
        real delay on every run, and the engine's modelled captaincy is a decent
        stand-in. When it is on, measured shares replace the model.
        """
        if sample_size <= 0:
            return None
        finished = boot.finished_gameweeks()
        if not finished:
            return CrowdSnapshot(
                gameweek=0,
                note=(
                    "No gameweek has finished, so no squads are public yet — "
                    "captaincy shares are modelled rather than measured."
                ),
            )
        return CrowdClient(self.client, self.settings).elite(max(finished), sample_size)

    def _risks(
        self, projections: ProjectionSet, squad: ManagerSquad, gameweek: int
    ) -> list[str]:
        """Things about your own squad that need a human decision."""
        risks: list[str] = []
        for element in squad.element_ids:
            player = projections.players.get(element)
            if player is None:
                continue
            note = summarise_availability(player)
            if note:
                risks.append(f"{player.web_name}: {note}")
            minutes = projections.minutes.get(element)
            if minutes and 0.05 < minutes.p_start < 0.55 and player.price >= 5.5:
                risks.append(
                    f"{player.web_name} is a {minutes.p_start:.0%} starter at £{player.price}m "
                    "— rotation risk in a position you are paying for."
                )
            if projections.get(element, gameweek).n_fixtures == 0:
                risks.append(f"{player.web_name} blanks in GW{gameweek}.")
        return risks


def format_advice_text(advice: Advice) -> str:
    """A plain-text brief, for the terminal or a message."""
    projections = advice.projections
    lines: list[str] = []
    deadline = (
        advice.deadline.strftime("%a %d %b %H:%M UTC") if advice.deadline else "unknown"
    )
    lines.append(f"Gameweek {advice.gameweek} — deadline {deadline}")
    lines.append("=" * 60)
    lines.append("")

    if advice.lineup_changes:
        lines.append("FREE CHANGES (no transfer needed)")
        for change in advice.lineup_changes:
            lines.append(f"  - {change.detail}")
        lines.append("")

    lines.append("TRANSFERS")
    for index, option in enumerate(advice.transfer_options[:4]):
        marker = "->" if index == 0 else "  "
        if option.is_hold:
            body = "hold, roll the free transfer"
        else:
            outs = ", ".join(projections.name(e) for e in option.out_elements)
            ins = ", ".join(projections.name(e) for e in option.in_elements)
            body = f"out {outs} / in {ins}"
        hit = f", -{option.hits * 4} hit" if option.hits else ""
        lines.append(
            f" {marker} {option.description}: {body}{hit} "
            f"({option.net_gain:+.1f} pts over {len(option.horizon)} GWs)"
        )
    lines.append("")

    lines.append(f"STARTING XI ({advice.lineup.formation_label()})")
    for element in advice.lineup.starters:
        tag = ""
        if element == advice.lineup.captain:
            tag = " (C)"
        elif element == advice.lineup.vice_captain:
            tag = " (V)"
        lines.append(
            f"  {projections.position(element):3} {projections.name(element):<16}"
            f"{projections.expected_points(element, advice.gameweek):5.1f}{tag}"
        )
    lines.append("")

    lines.append("BENCH (in order)")
    for row in advice.bench:
        lines.append(
            f"  {row['order']}. {row['name']:<16} {row['xpts']:5.1f} "
            f"(plays {row['p_appear']:.0%})"
        )
    lines.append(f"  autosub value: {advice.lineup.autosub_value:+.2f} pts")
    lines.append("")

    lines.append("CAPTAIN")
    for candidate in advice.captains[:5]:
        marker = "->" if candidate.recommended else "  "
        lines.append(
            f" {marker} {candidate.name:<16} {candidate.expected_points:5.1f} xPts  "
            f"ceiling {candidate.ceiling:5.1%}  EO {candidate.effective_ownership:5.1f}%"
        )
    if advice.captains and advice.captains[0].reason:
        lines.append(f"    {advice.captains[0].reason}")
    lines.append("")

    if advice.chip_advice:
        lines.append("CHIPS")
        for line in advice.chip_advice:
            lines.append(f"  {line}")
        lines.append("")

    if advice.paths:
        lines.append("PATHS")
        for row in compare(advice.paths)[:6]:
            lines.append(
                f"  {row['strategy']:<34} mean {row['mean']:6.1f}  "
                f"p10 {row['p10']:6.1f}  p90 {row['p90']:6.1f}  "
                f"beats hold {row['p_beats_hold']:.0%}"
            )
        lines.append("")

    if advice.outcome_summary:
        summary = advice.outcome_summary
        lines.append("THIS GAMEWEEK")
        lines.append(
            f"  projected {summary.get('mean', 0):.1f} pts "
            f"(p10 {summary.get('p10', 0):.0f}, p90 {summary.get('p90', 0):.0f})"
        )
        if advice.rank:
            rank = advice.rank.summary()
            lines.append(
                f"  field average {rank['field_mean']:.1f}, your edge {rank['edge']:+.1f}, "
                f"green arrow {rank['p_green_arrow']:.0%}"
            )
        lines.append("")

    if advice.risks:
        lines.append("RISKS")
        for risk in advice.risks[:10]:
            lines.append(f"  - {risk}")
        lines.append("")

    if advice.notes:
        lines.append("DATA")
        for note in advice.notes:
            lines.append(f"  - {note}")

    return "\n".join(lines)
