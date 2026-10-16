"""Team attack and defence ratings, and the expected goals for each fixture.

Two independent estimates of how many goals a fixture will produce:

    * a **ratings model** fitted to results with exponential time decay, so
      recent form counts more than August;
    * the **betting market**, when odds are available.

The market is the better forecaster, so it wins where it exists. The ratings
model covers fixtures without odds — anything more than a week or two out — and
provides a prior early in the season when few matches have been played.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize

from ..data.entities import Fixture, Team
from ..data.odds import OddsSnapshot, normalise_team
from .match import MatchProbabilities

log = logging.getLogger(__name__)

# Long-run Premier League averages, used as priors and as sanity clamps.
LEAGUE_GOALS_PER_GAME = 2.80
HOME_ADVANTAGE_PRIOR = 0.20  # in log-goal space, ~1.22x


@dataclass
class MatchResult:
    home_team: int
    away_team: int
    home_goals: int
    away_goals: int
    date: datetime | None = None


@dataclass
class TeamRatings:
    """Fitted log-scale attack and defence ratings, one pair per team."""

    attack: dict[int, float] = field(default_factory=dict)
    defence: dict[int, float] = field(default_factory=dict)
    home_advantage: float = HOME_ADVANTAGE_PRIOR
    intercept: float = math.log(LEAGUE_GOALS_PER_GAME / 2.0)
    matches_used: int = 0
    fitted: bool = False

    def expected_goals(self, home_team: int, away_team: int) -> tuple[float, float]:
        """(home xG, away xG) implied by the ratings."""
        att_h = self.attack.get(home_team, 0.0)
        def_h = self.defence.get(home_team, 0.0)
        att_a = self.attack.get(away_team, 0.0)
        def_a = self.defence.get(away_team, 0.0)
        lam_home = math.exp(self.intercept + self.home_advantage + att_h - def_a)
        lam_away = math.exp(self.intercept + att_a - def_h)
        return (
            float(np.clip(lam_home, 0.15, 5.0)),
            float(np.clip(lam_away, 0.15, 5.0)),
        )

    def strength_table(self, team_names: dict[int, str]) -> list[dict]:
        rows = []
        for team_id, name in team_names.items():
            att = self.attack.get(team_id, 0.0)
            dfn = self.defence.get(team_id, 0.0)
            rows.append(
                {
                    "team_id": team_id,
                    "team": name,
                    "attack": round(math.exp(att), 3),
                    "defence": round(math.exp(-dfn), 3),
                    "net": round(att + dfn, 3),
                }
            )
        return sorted(rows, key=lambda r: -r["net"])


def ratings_from_fpl_strength(teams: Sequence[Team]) -> TeamRatings:
    """A prior built from FPL's own published strength numbers.

    FPL grades every club's attack and defence, home and away, on a scale
    centred near 1100. Those grades are set by hand and updated occasionally,
    but before a ball is kicked they are far better than assuming every team is
    average — which matters a lot in gameweek 1.
    """
    ratings = TeamRatings()
    if not teams:
        return ratings

    attack_raw = {
        t.id: 0.5 * (t.strength_attack_home + t.strength_attack_away) for t in teams
    }
    defence_raw = {
        t.id: 0.5 * (t.strength_defence_home + t.strength_defence_away) for t in teams
    }
    mean_attack = float(np.mean(list(attack_raw.values()))) or 1.0
    mean_defence = float(np.mean(list(defence_raw.values()))) or 1.0

    # A club rated 20% above average scores roughly 20% more goals; the log of
    # the ratio, damped, is a reasonable rating. The damping factor keeps the
    # spread realistic (the published numbers are more spread than outcomes).
    damping = 0.75
    for team in teams:
        ratings.attack[team.id] = damping * math.log(max(attack_raw[team.id], 1.0) / mean_attack)
        # Higher defensive strength should *reduce* the opponent's goals, and
        # `expected_goals` subtracts the defender's rating, so a strong defence
        # gets a positive value here.
        ratings.defence[team.id] = damping * math.log(
            max(defence_raw[team.id], 1.0) / mean_defence
        )
    ratings.home_advantage = HOME_ADVANTAGE_PRIOR
    ratings.intercept = math.log(LEAGUE_GOALS_PER_GAME / 2.0)
    ratings.fitted = False
    return ratings


def fit_ratings(
    results: Sequence[MatchResult],
    *,
    team_ids: Sequence[int] | None = None,
    half_life_days: float = 180.0,
    prior: TeamRatings | None = None,
    prior_weight: float = 4.0,
    now: datetime | None = None,
) -> TeamRatings:
    """Fit attack/defence ratings by weighted Poisson maximum likelihood.

    Each match contributes two Poisson observations. Matches are weighted by
    ``0.5 ** (age / half_life)`` so the fit tracks form without throwing away
    history. A ridge penalty pulls every rating toward the prior, which is what
    makes the model usable after three gameweeks instead of thirteen.
    """
    now = now or datetime.now(timezone.utc)
    results = [r for r in results if r.home_goals is not None and r.away_goals is not None]
    if not results:
        return prior or TeamRatings()

    ids = sorted(set(team_ids) if team_ids else {r.home_team for r in results} | {
        r.away_team for r in results
    })
    index = {team_id: i for i, team_id in enumerate(ids)}
    n = len(ids)

    home_idx = np.array([index[r.home_team] for r in results])
    away_idx = np.array([index[r.away_team] for r in results])
    home_goals = np.array([r.home_goals for r in results], dtype=float)
    away_goals = np.array([r.away_goals for r in results], dtype=float)

    weights = np.ones(len(results))
    if half_life_days > 0:
        for i, result in enumerate(results):
            if result.date is None:
                continue
            date = result.date
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            age_days = max(0.0, (now - date).total_seconds() / 86400.0)
            weights[i] = 0.5 ** (age_days / half_life_days)

    prior_attack = np.zeros(n)
    prior_defence = np.zeros(n)
    if prior is not None:
        for team_id, i in index.items():
            prior_attack[i] = prior.attack.get(team_id, 0.0)
            prior_defence[i] = prior.defence.get(team_id, 0.0)

    # Parameters: [attack (n), defence (n), home_advantage, intercept]
    x0 = np.concatenate([prior_attack, prior_defence, [HOME_ADVANTAGE_PRIOR], [
        math.log(max(LEAGUE_GOALS_PER_GAME / 2.0, 0.1))
    ]])

    def unpack(params: np.ndarray):
        attack = params[:n]
        defence = params[n : 2 * n]
        return attack, defence, params[2 * n], params[2 * n + 1]

    def negative_log_likelihood(params: np.ndarray) -> float:
        attack, defence, home_adv, intercept = unpack(params)
        # Identifiability: ratings are only defined up to a shift, so centre them.
        attack = attack - attack.mean()
        defence = defence - defence.mean()

        log_lam_home = intercept + home_adv + attack[home_idx] - defence[away_idx]
        log_lam_away = intercept + attack[away_idx] - defence[home_idx]
        log_lam_home = np.clip(log_lam_home, -3.0, 2.0)
        log_lam_away = np.clip(log_lam_away, -3.0, 2.0)
        lam_home = np.exp(log_lam_home)
        lam_away = np.exp(log_lam_away)

        ll = weights * (home_goals * log_lam_home - lam_home)
        ll += weights * (away_goals * log_lam_away - lam_away)

        penalty = prior_weight * (
            np.sum((attack - prior_attack) ** 2) + np.sum((defence - prior_defence) ** 2)
        )
        return float(-ll.sum() + penalty)

    result = minimize(
        negative_log_likelihood,
        x0,
        method="L-BFGS-B",
        options={"maxiter": 500},
    )
    attack, defence, home_adv, intercept = unpack(result.x)
    attack = attack - attack.mean()
    defence = defence - defence.mean()

    ratings = TeamRatings(
        attack={team_id: float(attack[i]) for team_id, i in index.items()},
        defence={team_id: float(defence[i]) for team_id, i in index.items()},
        home_advantage=float(np.clip(home_adv, -0.1, 0.6)),
        intercept=float(intercept),
        matches_used=len(results),
        fitted=bool(result.success),
    )
    return ratings


def results_from_fixtures(fixtures: Iterable[Fixture]) -> list[MatchResult]:
    """Extract completed results from FPL fixture rows."""
    out = []
    for fixture in fixtures:
        if not fixture.finished:
            continue
        if fixture.team_h_score is None or fixture.team_a_score is None:
            continue
        out.append(
            MatchResult(
                home_team=fixture.team_h,
                away_team=fixture.team_a,
                home_goals=fixture.team_h_score,
                away_goals=fixture.team_a_score,
                date=fixture.kickoff,
            )
        )
    return out


@dataclass
class FixtureForecast:
    """The engine's final view of one fixture."""

    fixture_id: int
    gameweek: int | None
    home_team: int
    away_team: int
    lam_home: float
    lam_away: float
    source: str  # "market", "market+ratings" or "ratings"
    kickoff: datetime | None = None

    def probabilities(self) -> MatchProbabilities:
        return MatchProbabilities(self.lam_home, self.lam_away)

    def expected_goals_for(self, team_id: int) -> float:
        return self.lam_home if team_id == self.home_team else self.lam_away

    def expected_goals_against(self, team_id: int) -> float:
        return self.lam_away if team_id == self.home_team else self.lam_home

    def clean_sheet_probability(self, team_id: int) -> float:
        return self.probabilities().clean_sheet(home=team_id == self.home_team)

    def win_probability(self, team_id: int) -> float:
        home, draw, away = self.probabilities().outcome()
        return home if team_id == self.home_team else away


def build_fixture_forecasts(
    fixtures: Sequence[Fixture],
    ratings: TeamRatings,
    teams: dict[int, Team],
    odds: OddsSnapshot | None = None,
    *,
    market_weight: float = 0.8,
) -> dict[int, FixtureForecast]:
    """Expected goals for every unplayed fixture, market-informed where possible.

    ``market_weight`` is how far to move from the ratings model toward the
    market when both exist. It is deliberately below 1: bookmakers price in
    team news we may already model explicitly, and a small blend guards against
    a stale or mismatched odds row.
    """
    odds_by_key = odds.by_teams() if odds and odds.available else {}
    name_key = {team_id: normalise_team(team.name) for team_id, team in teams.items()}

    forecasts: dict[int, FixtureForecast] = {}
    for fixture in fixtures:
        if fixture.finished:
            continue
        lam_home, lam_away = ratings.expected_goals(fixture.team_h, fixture.team_a)
        source = "ratings"

        key = (name_key.get(fixture.team_h, ""), name_key.get(fixture.team_a, ""))
        market = odds_by_key.get(key)
        if market is not None:
            try:
                m_home, m_away = market.expected_goals()
            except (ValueError, RuntimeError) as exc:
                log.debug("could not solve market odds for fixture %s: %s", fixture.id, exc)
            else:
                lam_home = market_weight * m_home + (1 - market_weight) * lam_home
                lam_away = market_weight * m_away + (1 - market_weight) * lam_away
                source = "market+ratings" if market_weight < 1 else "market"

        forecasts[fixture.id] = FixtureForecast(
            fixture_id=fixture.id,
            gameweek=fixture.gameweek,
            home_team=fixture.team_h,
            away_team=fixture.team_a,
            lam_home=lam_home,
            lam_away=lam_away,
            source=source,
            kickoff=fixture.kickoff,
        )
    return forecasts


def fixtures_by_gameweek(
    forecasts: dict[int, FixtureForecast], gameweeks: Sequence[int]
) -> dict[int, list[FixtureForecast]]:
    """Group forecasts by gameweek, which is how the optimiser consumes them."""
    out: dict[int, list[FixtureForecast]] = {gw: [] for gw in gameweeks}
    for forecast in forecasts.values():
        if forecast.gameweek in out:
            out[forecast.gameweek].append(forecast)
    return out


def team_fixtures(
    forecasts: dict[int, FixtureForecast], team_id: int, gameweek: int
) -> list[FixtureForecast]:
    """Every fixture a club plays in a gameweek — zero for a blank, 2+ for a double."""
    return [
        f
        for f in forecasts.values()
        if f.gameweek == gameweek and team_id in (f.home_team, f.away_team)
    ]
