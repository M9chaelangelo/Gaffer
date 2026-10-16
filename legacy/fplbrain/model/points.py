"""The points model: a full distribution for every player in every fixture.

Most FPL tools stop at an expected-points number. That is not enough to make the
decisions this engine has to make. Captaincy is about the upper tail, bench order
is about the probability of a blank, and chip timing is about the shape of the
whole distribution. So each projection here is a probability mass function over
points, built by convolving the individual scoring components.

The model conditions on minutes first, because every other component depends on
them:

    no appearance   ->  0 points, nothing else can happen
    under 60        ->  1 appearance point, no clean sheet, rates scaled down
    60 or more      ->  2 appearance points, clean sheet live

Within each branch, goals, assists, clean sheets, goals conceded, saves, cards,
defensive contribution and bonus are convolved into a single pmf, and the three
branches are mixed by their probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import nbinom, poisson

from ..config import Rules, get_rules
from .minutes import MinutesProjection
from .rates import PlayerRates
from .strength import FixtureForecast

# The points grid every pmf lives on. Nothing in FPL scores below -6 or above 40
# in a single match with any meaningful probability.
POINTS_MIN = -8
POINTS_MAX = 45
GRID = np.arange(POINTS_MIN, POINTS_MAX + 1)
GRID_SIZE = len(GRID)

# Defensive actions are overdispersed relative to Poisson: a centre-back facing
# a long-ball side racks up clearances in bunches. A negative binomial with this
# variance-to-mean ratio fits observed CBIT counts far better than Poisson, and
# it matters because the DefCon threshold sits in the tail.
DEFCON_DISPERSION = 1.6

# Typical share of a team's shots on target that the keeper actually has to
# save, relative to expected goals conceded. Saves scale with opponent quality.
SAVES_PER_XGC = 2.35

# Expected goals for an average Premier League side in an average fixture. Used
# only as the reference point when no caller-supplied multiplier is available.
LEAGUE_AVERAGE_TEAM_XG = 1.40


def _delta(value: int) -> np.ndarray:
    """A pmf placing all mass on a single points value."""
    pmf = np.zeros(GRID_SIZE)
    pmf[int(np.clip(value - POINTS_MIN, 0, GRID_SIZE - 1))] = 1.0
    return pmf


def _from_counts(counts: np.ndarray, probs: np.ndarray, points_per: int) -> np.ndarray:
    """A pmf for ``points_per`` points per event, given a count distribution."""
    pmf = np.zeros(GRID_SIZE)
    for count, prob in zip(counts, probs):
        if prob <= 0:
            continue
        index = int(np.clip(count * points_per - POINTS_MIN, 0, GRID_SIZE - 1))
        pmf[index] += prob
    return pmf


def _poisson_points(rate: float, points_per: int, max_count: int = 8) -> np.ndarray:
    counts = np.arange(max_count + 1)
    probs = poisson.pmf(counts, max(rate, 1e-9))
    probs[-1] += max(0.0, 1.0 - probs.sum())
    return _from_counts(counts, probs, points_per)


def _bernoulli_points(prob: float, points: int) -> np.ndarray:
    pmf = np.zeros(GRID_SIZE)
    prob = float(np.clip(prob, 0.0, 1.0))
    pmf[int(np.clip(0 - POINTS_MIN, 0, GRID_SIZE - 1))] += 1 - prob
    pmf[int(np.clip(points - POINTS_MIN, 0, GRID_SIZE - 1))] += prob
    return pmf


def convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Convolve two pmfs on the shared grid, clipping mass at the edges."""
    full = np.convolve(a, b)
    # Index 0 of `full` corresponds to 2 * POINTS_MIN.
    offset = -POINTS_MIN
    out = np.zeros(GRID_SIZE)
    lo = offset
    hi = offset + GRID_SIZE
    out += full[lo:hi]
    # Fold the tails back onto the edges so probability is conserved.
    if lo > 0:
        out[0] += full[:lo].sum()
    if hi < len(full):
        out[-1] += full[hi:].sum()
    return out


def convolve_all(pmfs: Iterable[np.ndarray]) -> np.ndarray:
    result = _delta(0)
    for pmf in pmfs:
        result = convolve(result, pmf)
    return result


def negbinom_tail(mean: float, threshold: int, dispersion: float = DEFCON_DISPERSION) -> float:
    """P(X >= threshold) for an overdispersed count with the given mean."""
    mean = max(float(mean), 1e-6)
    if dispersion <= 1.0 + 1e-9:
        return float(1.0 - poisson.cdf(threshold - 1, mean))
    # variance = mean * dispersion  =>  n = mean / (dispersion - 1), p = 1 / dispersion
    n = mean / (dispersion - 1.0)
    p = 1.0 / dispersion
    return float(1.0 - nbinom.cdf(threshold - 1, n, p))


@dataclass
class PointsBreakdown:
    """Expected points attributed to each scoring component."""

    appearance: float = 0.0
    goals: float = 0.0
    assists: float = 0.0
    clean_sheet: float = 0.0
    goals_conceded: float = 0.0
    saves: float = 0.0
    defensive_contribution: float = 0.0
    cards: float = 0.0
    bonus: float = 0.0
    penalties: float = 0.0

    def total(self) -> float:
        return (
            self.appearance
            + self.goals
            + self.assists
            + self.clean_sheet
            + self.goals_conceded
            + self.saves
            + self.defensive_contribution
            + self.cards
            + self.bonus
            + self.penalties
        )

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 3) for k, v in self.__dict__.items()}


@dataclass
class PlayerFixtureProjection:
    """A player's projected points for one fixture, as a distribution."""

    element: int
    fixture_id: int
    gameweek: int | None
    team_id: int
    opponent_id: int
    is_home: bool
    pmf: np.ndarray = field(repr=False, default_factory=lambda: _delta(0))
    breakdown: PointsBreakdown = field(default_factory=PointsBreakdown)
    minutes: MinutesProjection | None = None
    clean_sheet_probability: float = 0.0
    defcon_probability: float = 0.0
    anytime_goal_probability: float = 0.0

    @property
    def expected_points(self) -> float:
        return float(np.dot(GRID, self.pmf))

    @property
    def variance(self) -> float:
        mean = self.expected_points
        return float(np.dot((GRID - mean) ** 2, self.pmf))

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.variance, 0.0)))

    def probability_at_least(self, points: int) -> float:
        return float(self.pmf[GRID >= points].sum())

    def probability_at_most(self, points: int) -> float:
        return float(self.pmf[GRID <= points].sum())

    @property
    def p_haul(self) -> float:
        """Probability of a double-figure haul — the captaincy question."""
        return self.probability_at_least(10)

    @property
    def p_blank(self) -> float:
        """Probability of two points or fewer — the bench-order question."""
        return self.probability_at_most(2)

    def quantile(self, q: float) -> int:
        cumulative = np.cumsum(self.pmf)
        idx = int(np.searchsorted(cumulative, q))
        return int(GRID[min(idx, GRID_SIZE - 1)])


@dataclass
class PlayerGameweekProjection:
    """A player's projection for a whole gameweek, across all their fixtures.

    A blank gameweek yields an empty fixture list and zero points; a double
    convolves the two fixtures, which is exactly right — the distributions are
    close to independent once the opponent is fixed.
    """

    element: int
    gameweek: int
    fixtures: list[PlayerFixtureProjection] = field(default_factory=list)

    @property
    def pmf(self) -> np.ndarray:
        if not self.fixtures:
            return _delta(0)
        return convolve_all(f.pmf for f in self.fixtures)

    @property
    def expected_points(self) -> float:
        return float(sum(f.expected_points for f in self.fixtures))

    @property
    def std(self) -> float:
        return float(np.sqrt(sum(f.variance for f in self.fixtures)))

    @property
    def n_fixtures(self) -> int:
        return len(self.fixtures)

    @property
    def p_haul(self) -> float:
        return float(self.pmf[GRID >= 10].sum())

    @property
    def p_blank(self) -> float:
        return float(self.pmf[GRID <= 2].sum())

    @property
    def breakdown(self) -> PointsBreakdown:
        total = PointsBreakdown()
        for fixture in self.fixtures:
            for key, value in fixture.breakdown.__dict__.items():
                setattr(total, key, getattr(total, key) + value)
        return total


class PointsModel:
    """Turns rates, minutes and a fixture forecast into a points distribution."""

    def __init__(self, rules: Rules | None = None):
        self.rules = rules or get_rules()
        scoring = self.rules.scoring
        self.goal_points = scoring["goal"]
        self.assist_points = int(scoring["assist"])
        self.cs_points = scoring["clean_sheet"]
        self.gc_per = int(scoring["goals_conceded"]["per"])
        self.gc_points = int(scoring["goals_conceded"]["points"])
        self.gc_positions = set(scoring["goals_conceded"]["positions"])
        self.saves_per = int(scoring["saves"]["per"])
        self.saves_points = int(scoring["saves"]["points"])
        self.yellow = int(scoring["yellow_card"])
        self.red = int(scoring["red_card"])
        self.defcon_points = self.rules.defcon_points

    # -- component pmfs ------------------------------------------------------

    def _goals_pmf(self, rates: PlayerRates, minutes: float, attack_multiplier: float):
        rate = rates.xg90 * (minutes / 90.0) * attack_multiplier
        return _poisson_points(rate, int(self.goal_points.get(rates.position, 4))), rate

    def _assists_pmf(self, rates: PlayerRates, minutes: float, attack_multiplier: float):
        rate = rates.xa90 * (minutes / 90.0) * attack_multiplier
        return _poisson_points(rate, self.assist_points), rate

    def _clean_sheet_pmf(self, position: str, p_clean_sheet: float):
        points = int(self.cs_points.get(position, 0))
        if points == 0:
            return _delta(0), 0.0
        return _bernoulli_points(p_clean_sheet, points), p_clean_sheet * points

    def _goals_conceded_pmf(self, position: str, conceded_pmf: np.ndarray):
        """-1 per two goals conceded, for keepers and defenders."""
        if position not in self.gc_positions:
            return _delta(0), 0.0
        counts = np.arange(len(conceded_pmf))
        deductions = (counts // self.gc_per) * self.gc_points
        pmf = np.zeros(GRID_SIZE)
        expected = 0.0
        for deduction, prob in zip(deductions, conceded_pmf):
            if prob <= 0:
                continue
            index = int(np.clip(deduction - POINTS_MIN, 0, GRID_SIZE - 1))
            pmf[index] += prob
            expected += deduction * prob
        return pmf, expected

    def _saves_pmf(self, rates: PlayerRates, minutes: float, opponent_xg: float):
        """Save points for a keeper, scaled by how much shooting he will face."""
        if rates.position != "GKP":
            return _delta(0), 0.0
        # A keeper's own save rate carries information about his side's style;
        # anchor it to the opponent's expected goals for this specific fixture.
        expected_saves = max(
            0.1, SAVES_PER_XGC * opponent_xg * (minutes / 90.0)
        )
        if rates.saves90 > 0:
            expected_saves = 0.5 * expected_saves + 0.5 * rates.saves90 * (minutes / 90.0)

        counts = np.arange(0, 16)
        probs = poisson.pmf(counts, expected_saves)
        probs[-1] += max(0.0, 1.0 - probs.sum())
        points = (counts // self.saves_per) * self.saves_points
        pmf = np.zeros(GRID_SIZE)
        expected = 0.0
        for point, prob in zip(points, probs):
            index = int(np.clip(point - POINTS_MIN, 0, GRID_SIZE - 1))
            pmf[index] += prob
            expected += point * prob
        return pmf, expected

    def _defcon_pmf(self, rates: PlayerRates, minutes: float):
        """Defensive contribution: +2 for clearing the positional threshold."""
        rule = self.rules.defcon_rule(rates.position)
        if not rule:
            return _delta(0), 0.0, 0.0
        rate = rates.defcon_rate()
        if rate is None:
            return _delta(0), 0.0, 0.0

        threshold = int(rule["threshold"])
        expected_actions = rate * (minutes / 90.0)
        probability = negbinom_tail(expected_actions, threshold)

        # Where a player has a real track record, trust it over the model. The
        # count model gets the shape right but individual roles vary a lot.
        if rates.defcon_hit_rate is not None and rates.defcon_games >= 4:
            weight = min(rates.defcon_games / (rates.defcon_games + 6.0), 0.75)
            observed = rates.defcon_hit_rate * min(1.0, minutes / 75.0)
            probability = weight * observed + (1 - weight) * probability

        probability = float(np.clip(probability, 0.0, 0.98))
        return (
            _bernoulli_points(probability, self.defcon_points),
            probability * self.defcon_points,
            probability,
        )

    def _cards_pmf(self, rates: PlayerRates, minutes: float):
        scale = minutes / 90.0
        p_yellow = float(np.clip(rates.yellow90 * scale, 0.0, 0.6))
        p_red = float(np.clip(rates.red90 * scale, 0.0, 0.15))
        pmf = np.zeros(GRID_SIZE)
        pmf[int(0 - POINTS_MIN)] += (1 - p_yellow) * (1 - p_red)
        pmf[int(self.yellow - POINTS_MIN)] += p_yellow * (1 - p_red)
        pmf[int(self.red - POINTS_MIN)] += p_red
        expected = p_yellow * self.yellow + p_red * self.red
        return pmf, expected

    # -- assembly ------------------------------------------------------------

    def project(
        self,
        *,
        element: int,
        rates: PlayerRates,
        minutes: MinutesProjection,
        forecast: FixtureForecast,
        team_id: int,
        attack_multiplier: float | None = None,
        bonus_pmf: np.ndarray | None = None,
        expected_bonus: float = 0.0,
        anytime_goal_probability: float | None = None,
    ) -> PlayerFixtureProjection:
        """Build the full points distribution for one player in one fixture.

        ``attack_multiplier`` scales the player's attacking rates for this
        particular fixture. The engine supplies one computed against the club's
        own average output, which is the more accurate reading. When it is
        omitted the multiplier is derived from the fixture against a league
        average instead, so a projection built by hand still responds to whether
        the fixture is a good one.
        """
        is_home = team_id == forecast.home_team
        opponent_id = forecast.away_team if is_home else forecast.home_team
        probs = forecast.probabilities()

        team_xg = forecast.expected_goals_for(team_id)
        opponent_xg = forecast.expected_goals_against(team_id)
        if attack_multiplier is None:
            attack_multiplier = float(np.clip(team_xg / LEAGUE_AVERAGE_TEAM_XG, 0.3, 3.0)) ** 0.85
        p_clean_sheet = probs.clean_sheet(home=is_home)
        conceded_pmf = probs.goals_conceded_pmf(home=is_home)

        # An anytime-goalscorer market, when we have one, overrides the model's
        # own goal rate: it is the sharpest single-player signal available.
        goal_rate_override = None
        if anytime_goal_probability is not None and 0 < anytime_goal_probability < 0.95:
            goal_rate_override = -np.log(1.0 - anytime_goal_probability)

        branches: list[tuple[float, np.ndarray, PointsBreakdown]] = []

        for label, probability, minutes_played, appearance_points, clean_sheet_live in (
            ("none", 1.0 - minutes.p_appear, 0.0, 0, False),
            ("short", max(minutes.p_appear - minutes.p_60, 0.0), 32.0, 1, False),
            ("full", minutes.p_60, 84.0, 2, True),
        ):
            if probability <= 1e-9:
                continue
            if label == "none":
                branches.append((probability, _delta(0), PointsBreakdown()))
                continue

            share = minutes_played / 90.0
            breakdown = PointsBreakdown(appearance=float(appearance_points))
            pmfs = [_delta(appearance_points)]

            goals_pmf, goal_rate = self._goals_pmf(rates, minutes_played, attack_multiplier)
            if goal_rate_override is not None:
                goal_rate = goal_rate_override * share
                goals_pmf = _poisson_points(
                    goal_rate, int(self.goal_points.get(rates.position, 4))
                )
            pmfs.append(goals_pmf)
            breakdown.goals = goal_rate * self.goal_points.get(rates.position, 4)

            assists_pmf, assist_rate = self._assists_pmf(rates, minutes_played, attack_multiplier)
            pmfs.append(assists_pmf)
            breakdown.assists = assist_rate * self.assist_points

            if clean_sheet_live:
                cs_pmf, cs_ev = self._clean_sheet_pmf(rates.position, p_clean_sheet)
                pmfs.append(cs_pmf)
                breakdown.clean_sheet = cs_ev

            gc_pmf, gc_ev = self._goals_conceded_pmf(rates.position, conceded_pmf)
            pmfs.append(gc_pmf)
            breakdown.goals_conceded = gc_ev

            saves_pmf, saves_ev = self._saves_pmf(rates, minutes_played, opponent_xg)
            pmfs.append(saves_pmf)
            breakdown.saves = saves_ev

            defcon_pmf, defcon_ev, _ = self._defcon_pmf(rates, minutes_played)
            pmfs.append(defcon_pmf)
            breakdown.defensive_contribution = defcon_ev

            cards_pmf, cards_ev = self._cards_pmf(rates, minutes_played)
            pmfs.append(cards_pmf)
            breakdown.cards = cards_ev

            if bonus_pmf is not None:
                # Bonus is only realistically available to players who play a
                # meaningful part, so damp it for a short cameo. A player who
                # lasts the match gets the full distribution.
                bonus_share = float(np.clip(minutes_played / 80.0, 0.0, 1.0))
                pmfs.append(_scale_bonus(bonus_pmf, bonus_share))
                breakdown.bonus = expected_bonus * bonus_share

            branches.append((probability, convolve_all(pmfs), breakdown))

        total_probability = sum(p for p, _, _ in branches)
        if total_probability <= 0:
            return PlayerFixtureProjection(
                element=element,
                fixture_id=forecast.fixture_id,
                gameweek=forecast.gameweek,
                team_id=team_id,
                opponent_id=opponent_id,
                is_home=is_home,
                minutes=minutes,
            )

        pmf = np.zeros(GRID_SIZE)
        combined = PointsBreakdown()
        for probability, branch_pmf, breakdown in branches:
            weight = probability / total_probability
            pmf += weight * branch_pmf
            for key, value in breakdown.__dict__.items():
                setattr(combined, key, getattr(combined, key) + weight * value)

        _, _, defcon_probability = self._defcon_pmf(rates, 84.0)

        return PlayerFixtureProjection(
            element=element,
            fixture_id=forecast.fixture_id,
            gameweek=forecast.gameweek,
            team_id=team_id,
            opponent_id=opponent_id,
            is_home=is_home,
            pmf=pmf / pmf.sum(),
            breakdown=combined,
            minutes=minutes,
            clean_sheet_probability=p_clean_sheet * minutes.p_60,
            defcon_probability=defcon_probability * minutes.p_60,
            anytime_goal_probability=(
                anytime_goal_probability
                if anytime_goal_probability is not None
                else float(1.0 - np.exp(-rates.xg90 * minutes.expected_minutes / 90.0))
            ),
        )


def _scale_bonus(bonus_pmf: np.ndarray, share: float) -> np.ndarray:
    """Damp a bonus pmf toward zero for a player who plays only part of a match."""
    share = float(np.clip(share, 0.0, 1.0))
    if share >= 0.999:
        return bonus_pmf
    zero = _delta(0)
    return share * bonus_pmf + (1 - share) * zero


# ---------------------------------------------------------------------------
# Bonus points
# ---------------------------------------------------------------------------


@dataclass
class BpsProfile:
    """Expected BPS and its spread for one player in one fixture."""

    element: int
    mean: float
    std: float
    p_start: float


def expected_bps(
    rates: PlayerRates,
    minutes: MinutesProjection,
    weights: dict,
    *,
    p_clean_sheet: float,
    expected_conceded: float,
    attack_multiplier: float = 1.0,
    opponent_xg: float = 1.3,
) -> BpsProfile:
    """Expected BPS for a player, using the configured BPS weights.

    BPS is a deterministic linear function of match statistics, so this is a
    straightforward expectation over the stats the rate model already produces.
    The standard deviation is dominated by whether the player returns, which is
    what makes bonus so lumpy — that is captured explicitly.
    """
    position = rates.position
    share = minutes.expected_minutes / 90.0

    mean = 0.0
    mean += weights.get("played_60_plus", 6) * minutes.p_60
    mean += weights.get("played_1_to_60", 3) * max(minutes.p_appear - minutes.p_60, 0.0)

    goal_rate = rates.xg90 * share * attack_multiplier
    assist_rate = rates.xa90 * share * attack_multiplier
    goal_weight = float(weights.get("goal", {}).get(position, 18))
    mean += goal_weight * goal_rate
    mean += float(weights.get("assist", 9)) * assist_rate

    cs_weight = float(weights.get("clean_sheet", {}).get(position, 0))
    mean += cs_weight * p_clean_sheet * minutes.p_60

    gc_weight = float(weights.get("goal_conceded", {}).get(position, 0))
    mean += gc_weight * expected_conceded * share

    if position == "GKP":
        saves = SAVES_PER_XGC * opponent_xg * share
        mean += float(weights.get("save", 2)) * saves
        # Roughly two thirds of shots on target come from inside the box.
        mean += float(weights.get("save_inside_box_bonus", 1)) * saves * 0.65

    # Defensive actions: CBI at one BPS per three, recoveries per four, plus the
    # standalone bonus for hitting the defensive-contribution threshold.
    cbi_rate = max(rates.cbit90 - 0.0, 0.0) * share
    mean += float(weights.get("cbi_points", 1)) * cbi_rate / max(
        float(weights.get("cbi_per", 3)), 1.0
    )
    recovery_rate = max(rates.cbirt90 - rates.cbit90, 0.0) * share
    mean += float(weights.get("recovery_points", 1)) * recovery_rate / max(
        float(weights.get("recovery_per", 4)), 1.0
    )

    defcon_rate = rates.defcon_rate()
    if defcon_rate is not None:
        rule = {"DEF": 10, "MID": 12, "FWD": 12}.get(position)
        if rule:
            p_defcon = negbinom_tail(defcon_rate * share, rule)
            mean += float(weights.get("defensive_contribution", 3)) * p_defcon

    mean += float(weights.get("yellow_card", -3)) * rates.yellow90 * share
    mean += float(weights.get("red_card", -9)) * rates.red90 * share

    # Spread: returns are the main driver, so scale the standard deviation with
    # the chance of one. A player with a 30% chance of a goal has a bimodal BPS.
    return_probability = 1.0 - np.exp(-(goal_rate + assist_rate))
    std = 6.0 + 14.0 * float(return_probability) + 4.0 * (1.0 if position == "GKP" else 0.0)

    return BpsProfile(
        element=rates.element, mean=max(mean, 0.0), std=max(std, 3.0), p_start=minutes.p_start
    )


def bonus_distribution(
    profiles: Sequence[BpsProfile],
    bonus_points: Sequence[int] = (3, 2, 1),
    draws: int = 4000,
    seed: int = 12345,
) -> dict[int, np.ndarray]:
    """Monte Carlo the bonus allocation for one fixture.

    Bonus goes to the top three BPS scorers in a match, so it can only be
    computed by comparing all 22 players against each other. Sampling BPS for
    every candidate and counting how often each finishes in the top three gives
    a proper distribution over {0, 1, 2, 3} bonus points per player.
    """
    if not profiles:
        return {}

    rng = np.random.default_rng(seed)
    n = len(profiles)
    means = np.array([p.mean for p in profiles])
    stds = np.array([p.std for p in profiles])
    plays = np.array([p.p_start for p in profiles])

    samples = rng.normal(means[None, :], stds[None, :], size=(draws, n))
    # A player who does not appear cannot accumulate BPS.
    appeared = rng.random(size=(draws, n)) < np.clip(plays, 0.0, 1.0)[None, :]
    samples = np.where(appeared, samples, -1e6)

    order = np.argsort(-samples, axis=1)
    result: dict[int, np.ndarray] = {}
    counts = np.zeros((n, len(bonus_points) + 1))

    for rank, points in enumerate(bonus_points):
        winners = order[:, rank]
        for index in range(n):
            counts[index, rank + 1] += float(np.sum(winners == index))
    counts[:, 0] = draws - counts[:, 1:].sum(axis=1)
    counts /= draws

    for index, profile in enumerate(profiles):
        pmf = np.zeros(GRID_SIZE)
        pmf[int(0 - POINTS_MIN)] += counts[index, 0]
        for rank, points in enumerate(bonus_points):
            pmf[int(points - POINTS_MIN)] += counts[index, rank + 1]
        result[profile.element] = pmf
    return result


def expected_from_pmf(pmf: np.ndarray) -> float:
    return float(np.dot(GRID, pmf))
