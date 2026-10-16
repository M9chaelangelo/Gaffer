"""A synthetic field of rival managers, for rank projections.

Points are only a means to an end: what actually changes is your rank, and rank
depends on what everyone else owns. A gameweek where you score 70 is excellent
if the template scored 50 and disastrous if it scored 90. Two ideas do most of
the work here.

**Effective ownership** is ownership plus captaincy — a player owned by 40% and
captained by 30% has an effective ownership of 70%, because captains count
twice. It is the right measure of how much of the field a player's points
accrue to.

**A sampled field** turns that into a distribution. Rather than assume every
rival owns the template, the simulator draws thousands of plausible rival squads
weighted by real ownership, scores them against the same simulated gameweek your
own squad faces, and reads your rank off the resulting distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..config import Rules, get_rules
from ..data.crowd import CrowdSnapshot
from ..model.engine import ProjectionSet
from ..optimize.lineup import choose_starting_eleven
from .montecarlo import GameweekSample

# Captaincy concentrates far more than ownership: the field's armband goes
# overwhelmingly to a handful of premium attackers with good fixtures.
CAPTAINCY_CONCENTRATION = 2.6


@dataclass
class FieldModel:
    """Ownership and captaincy shares across the manager population."""

    ownership: dict[int, float]  # percent of managers owning
    captaincy: dict[int, float]  # percent of managers captaining
    total_managers: int = 0
    captaincy_source: str = "modelled from expected points"

    def effective_ownership(self) -> dict[int, float]:
        return {
            element: share + self.captaincy.get(element, 0.0)
            for element, share in self.ownership.items()
        }

    def top_captains(self, n: int = 8) -> list[tuple[int, float]]:
        return sorted(self.captaincy.items(), key=lambda kv: -kv[1])[:n]


def build_field_model(
    projections: ProjectionSet,
    gameweek: int,
    total_managers: int = 0,
    crowd: "CrowdSnapshot | None" = None,
) -> FieldModel:
    """Estimate ownership and captaincy shares for a gameweek.

    Ownership comes straight from the API. Captaincy is not published, so it is
    modelled: managers captain a player roughly in proportion to how attractive
    he is, weighted by how many of them own him at all.

    When a ``crowd`` sample of real squads is supplied, its *observed* captaincy
    replaces the model — a measured 46% armband on one player beats any
    inference from expected points. Ownership stays on the published figure,
    which is a census of the whole game rather than a sample of it; the sampled
    ownership is more useful as a contrast to that, and is reported separately.
    """
    ownership = {
        element: player.selected_by_percent for element, player in projections.players.items()
    }

    if crowd is not None and crowd.available:
        observed = crowd.captaincy()
        if observed:
            return FieldModel(
                ownership=ownership,
                captaincy=observed,
                total_managers=total_managers,
                captaincy_source=f"observed in {len(crowd)} {crowd.source} squads",
            )

    scores: dict[int, float] = {}
    for element, player in projections.players.items():
        own = ownership.get(element, 0.0)
        if own < 1.0:
            continue
        expected = projections.expected_points(element, gameweek)
        if expected <= 0:
            continue
        # Captaincy appeal grows superlinearly in expected points, and you can
        # only be captained by managers who own you.
        scores[element] = (own / 100.0) * (expected**CAPTAINCY_CONCENTRATION)

    total = sum(scores.values())
    captaincy = (
        {element: 100.0 * score / total for element, score in scores.items()} if total > 0 else {}
    )
    return FieldModel(ownership=ownership, captaincy=captaincy, total_managers=total_managers)


@dataclass
class RankProjection:
    mean_field_score: float
    your_mean: float
    percentile: float
    p_green_arrow: float
    p_top_1_percent_week: float
    quantiles: dict[str, float] = field(default_factory=dict)
    field_scores: np.ndarray | None = None

    def summary(self) -> dict[str, float]:
        return {
            "your_mean": round(self.your_mean, 2),
            "field_mean": round(self.mean_field_score, 2),
            "edge": round(self.your_mean - self.mean_field_score, 2),
            "percentile": round(self.percentile, 3),
            "p_green_arrow": round(self.p_green_arrow, 3),
            "p_top_1pct_week": round(self.p_top_1_percent_week, 4),
        }


class RivalFieldSimulator:
    """Samples rival squads and scores them against a simulated gameweek."""

    def __init__(
        self,
        projections: ProjectionSet,
        field_model: FieldModel,
        *,
        rules: Rules | None = None,
        seed: int = 99,
    ):
        self.projections = projections
        self.field = field_model
        self.rules = rules or get_rules()
        self.rng = np.random.default_rng(seed)

    def sample_squads(self, n_rivals: int, universe: Sequence[int]) -> list[list[int]]:
        """Draw plausible rival squads weighted by real ownership.

        The constraints (position counts, three per club) are enforced by
        rejection within each position, which is fast because ownership is
        concentrated: most sampled squads are legal on the first attempt.
        """
        universe = [e for e in universe if self.field.ownership.get(e, 0.0) > 0.1]
        by_position: dict[str, list[int]] = {}
        for element in universe:
            by_position.setdefault(self.projections.position(element), []).append(element)

        squads: list[list[int]] = []
        needed = self.rules.squad_positions
        for _ in range(n_rivals):
            squad: list[int] = []
            club_counts: dict[int, int] = {}
            for position, count in needed.items():
                pool = by_position.get(position, [])
                if len(pool) < count:
                    squad.extend(pool)
                    continue
                weights = np.array([self.field.ownership.get(e, 0.1) for e in pool], dtype=float)
                weights = np.maximum(weights, 0.01)
                weights /= weights.sum()
                chosen: list[int] = []
                attempts = 0
                while len(chosen) < count and attempts < 200:
                    attempts += 1
                    pick = int(self.rng.choice(len(pool), p=weights))
                    element = pool[pick]
                    if element in chosen:
                        continue
                    club = self.projections.team_of(element)
                    if club_counts.get(club, 0) >= self.rules.max_per_club:
                        continue
                    chosen.append(element)
                    club_counts[club] = club_counts.get(club, 0) + 1
                squad.extend(chosen)
            squads.append(squad)
        return squads

    def score_field(
        self,
        sample: GameweekSample,
        n_rivals: int = 2000,
        *,
        max_draws: int = 2000,
    ) -> np.ndarray:
        """Field scores, shaped ``(draws, n_rivals)``.

        Each rival plays a legal XI from their sampled 15 and captains according
        to the modelled captaincy shares.
        """
        draws = min(sample.draws, max_draws)
        universe = list(sample.elements)
        squads = self.sample_squads(n_rivals, universe)

        captain_elements = list(self.field.captaincy)
        captain_weights = np.array(
            [self.field.captaincy[e] for e in captain_elements], dtype=float
        )
        if captain_weights.sum() > 0:
            captain_weights = captain_weights / captain_weights.sum()

        scores = np.zeros((draws, len(squads)))
        for rival, squad in enumerate(squads):
            available = [e for e in squad if e in sample.elements]
            if not available:
                continue

            # Rivals pick their XI in advance, like everyone else. Choosing the
            # best eleven from the simulated results instead would hand the whole
            # field hindsight worth close to ten points a week and make your own
            # rank look far worse than it is.
            starters, bench, _ = choose_starting_eleven(
                available, self.projections, sample.gameweek
            )
            starter_columns = [sample.index_of(e) for e in starters]
            bench_columns = [sample.index_of(e) for e in bench]

            starter_points = sample.points[:draws, starter_columns]
            starter_live = sample.minutes[:draws, starter_columns] > 0
            total = (starter_points * starter_live).sum(axis=1)

            if bench_columns:
                # Approximate autosubs: each absent starter is replaced by the
                # best-scoring bench player who did appear. This ignores the
                # formation rules, which is worth a fraction of a point and
                # applies equally to every rival.
                absent = (~starter_live).sum(axis=1)
                bench_points = sample.points[:draws, bench_columns]
                bench_live = sample.minutes[:draws, bench_columns] > 0
                usable = np.where(bench_live, bench_points, 0.0)
                ordered = -np.sort(-usable, axis=1)
                cumulative = np.cumsum(ordered, axis=1)
                padded = np.hstack([np.zeros((draws, 1)), cumulative])
                total += padded[np.arange(draws), np.minimum(absent, len(bench_columns))]

            if len(captain_elements) and captain_weights.sum() > 0:
                captain = captain_elements[
                    int(self.rng.choice(len(captain_elements), p=captain_weights))
                ]
                if captain in squad and captain in sample.elements:
                    captain_column = sample.index_of(captain)
                    total += sample.points[:draws, captain_column] * (
                        sample.minutes[:draws, captain_column] > 0
                    )
            scores[:, rival] = total
        return scores

    def project_rank(
        self,
        your_scores: np.ndarray,
        field_scores: np.ndarray,
        *,
        current_rank: int | None = None,
        total_managers: int | None = None,
    ) -> RankProjection:
        """Compare your simulated scores against the simulated field."""
        draws = min(len(your_scores), field_scores.shape[0])
        yours = your_scores[:draws]
        rivals = field_scores[:draws]

        beaten = (yours[:, None] > rivals).mean(axis=1)
        percentile = float(beaten.mean())
        field_mean = float(rivals.mean())

        # A green arrow means finishing the week above the average manager.
        p_green = float((yours > rivals.mean(axis=1)).mean())
        p_top_1 = float((beaten >= 0.99).mean())

        return RankProjection(
            mean_field_score=field_mean,
            your_mean=float(yours.mean()),
            percentile=percentile,
            p_green_arrow=p_green,
            p_top_1_percent_week=p_top_1,
            quantiles={
                "p10": float(np.percentile(yours, 10)),
                "p50": float(np.percentile(yours, 50)),
                "p90": float(np.percentile(yours, 90)),
                "field_p50": float(np.percentile(rivals, 50)),
            },
            field_scores=rivals,
        )


def effective_ownership_table(
    field_model: FieldModel, projections: ProjectionSet, elements: Sequence[int]
) -> list[dict]:
    """Ownership, captaincy and effective ownership for a set of players."""
    effective = field_model.effective_ownership()
    rows = []
    for element in elements:
        rows.append(
            {
                "element": element,
                "name": projections.name(element),
                "ownership": round(field_model.ownership.get(element, 0.0), 1),
                "captaincy": round(field_model.captaincy.get(element, 0.0), 1),
                "effective_ownership": round(effective.get(element, 0.0), 1),
            }
        )
    rows.sort(key=lambda r: -r["effective_ownership"])
    return rows
