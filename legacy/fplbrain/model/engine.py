"""The projection engine: everything wired together.

Give it a client and a horizon and it returns a projection for every player in
every gameweek, along with the intermediate objects (team ratings, fixture
forecasts, per-90 rates) so that a report can explain *why* a projection says
what it says.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Sequence

import numpy as np

from ..config import Rules, Settings, get_rules, get_settings
from ..data.entities import Fixture, Player, Team
from ..data.fpl import Bootstrap, FplClient, element_history
from ..data.advanced import BlendReport, blend_rates, get_provider
from ..data.odds import OddsSnapshot, normalise_team
from ..util.http import HttpClient
from .minutes import (
    MinutesProjection,
    congestion_rotation_risk,
    minutes_history_from_summary,
    project_minutes,
)
from .points import (
    BpsProfile,
    PlayerGameweekProjection,
    PointsModel,
    bonus_distribution,
    expected_bps,
    expected_from_pmf,
)
from .rates import PlayerRates, RateEstimator, average_team_xg, team_attack_multiplier
from .strength import (
    FixtureForecast,
    TeamRatings,
    build_fixture_forecasts,
    fit_ratings,
    ratings_from_fpl_strength,
    results_from_fixtures,
)

log = logging.getLogger(__name__)


@dataclass
class ProjectionSet:
    """All projections for a horizon, plus the inputs that produced them."""

    gameweeks: list[int]
    players: dict[int, Player]
    teams: dict[int, Team]
    rates: dict[int, PlayerRates]
    minutes: dict[int, MinutesProjection]
    ratings: TeamRatings
    forecasts: dict[int, FixtureForecast]
    projections: dict[int, dict[int, PlayerGameweekProjection]] = field(default_factory=dict)
    odds: OddsSnapshot | None = None
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    notes: list[str] = field(default_factory=list)
    _display_names: dict[int, str] | None = field(default=None, repr=False)

    def get(self, element: int, gameweek: int) -> PlayerGameweekProjection:
        by_gw = self.projections.get(gameweek, {})
        return by_gw.get(element) or PlayerGameweekProjection(element=element, gameweek=gameweek)

    def expected_points(self, element: int, gameweek: int) -> float:
        return self.get(element, gameweek).expected_points

    def horizon_points(self, element: int, gameweeks: Sequence[int] | None = None) -> float:
        return float(
            sum(self.expected_points(element, gw) for gw in (gameweeks or self.gameweeks))
        )

    def matrix(self, elements: Sequence[int], gameweeks: Sequence[int]) -> np.ndarray:
        """Expected points as a ``len(elements) x len(gameweeks)`` array."""
        return np.array(
            [[self.expected_points(e, gw) for gw in gameweeks] for e in elements], dtype=float
        )

    def price(self, element: int) -> float:
        player = self.players.get(element)
        return player.price if player else 0.0

    def position(self, element: int) -> str:
        player = self.players.get(element)
        return player.position if player else "MID"

    def team_of(self, element: int) -> int:
        player = self.players.get(element)
        return player.team_id if player else 0

    def name(self, element: int) -> str:
        """Display name, guaranteed unique across the player pool.

        FPL's ``web_name`` is usually a surname, and surnames repeat — a
        briefing that reads "out Silva / in Silva" is worse than useless. Names
        are widened only as far as they need to be: the club is added for a
        collision, a first initial if the club is shared too, and the element id
        in the last resort.
        """
        if self._display_names is None:
            self._display_names = self._build_display_names()
        return self._display_names.get(element, str(element))

    def _build_display_names(self) -> dict[int, str]:
        by_name: dict[str, list[int]] = {}
        for element, player in self.players.items():
            by_name.setdefault(player.web_name, []).append(element)

        names: dict[int, str] = {}
        for web_name, elements in by_name.items():
            if len(elements) == 1:
                names[elements[0]] = web_name
                continue

            with_club = {
                element: f"{web_name} ({self.team_name(self.players[element].team_id)})"
                for element in elements
            }
            if len(set(with_club.values())) == len(elements):
                names.update(with_club)
                continue

            with_initial = {}
            for element in elements:
                player = self.players[element]
                initial = (player.raw.get("first_name") or " ")[0]
                club = self.team_name(player.team_id)
                with_initial[element] = f"{initial}. {web_name} ({club})"
            if len(set(with_initial.values())) == len(elements):
                names.update(with_initial)
                continue

            for element in elements:
                names[element] = f"{with_initial[element]} #{element}"
        return names

    def team_name(self, team_id: int) -> str:
        team = self.teams.get(team_id)
        return team.short_name if team else str(team_id)

    def fixture_count(self, element: int, gameweek: int) -> int:
        return self.get(element, gameweek).n_fixtures


class ProjectionEngine:
    """Builds a :class:`ProjectionSet`."""

    def __init__(
        self,
        client: FplClient,
        *,
        settings: Settings | None = None,
        rules: Rules | None = None,
        odds: OddsSnapshot | None = None,
    ):
        self.client = client
        self.settings = settings or get_settings()
        self.rules = rules or get_rules()
        self.points_model = PointsModel(self.rules)
        self.odds = odds

    def run(
        self,
        gameweeks: Sequence[int] | None = None,
        *,
        deep_history: bool = True,
        max_history_players: int = 700,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> ProjectionSet:
        """Project every player over ``gameweeks``.

        ``deep_history`` fetches per-player gameweek history, which costs one
        HTTP request per player but sharpens the minutes model considerably. It
        is cached, so the expensive run happens once per gameweek.

        ``progress`` is called as ``(done, total, label)`` during the slow parts.
        The first run of a season fetches six hundred players behind a polite
        rate limit, which takes minutes — long enough that silence looks like a
        hang.
        """
        boot = self.client.bootstrap()
        fixtures = self.client.fixtures()
        players = {p.id: p for p in boot.players}
        teams = {t.id: t for t in boot.teams}

        gameweeks = list(gameweeks or self._default_horizon(boot))
        notes: list[str] = []

        ratings = self._fit_ratings(fixtures, teams, notes)
        odds = self.odds if self.odds is not None else OddsSnapshot(available=False)
        if not odds.available and odds.reason:
            notes.append(odds.reason)

        forecasts = build_fixture_forecasts(
            [f for f in fixtures if f.gameweek in set(gameweeks)],
            ratings,
            teams,
            odds,
        )
        market_backed = sum(1 for f in forecasts.values() if f.source != "ratings")
        if market_backed:
            notes.append(
                f"{market_backed} of {len(forecasts)} fixtures priced from bookmaker odds"
            )

        histories = self._load_histories(
            players, deep_history, max_history_players, progress=progress
        )
        rates, minutes = self._estimate_players(players, histories)

        blend = self._blend_advanced(rates, players, teams)
        if blend is not None:
            notes.append(blend.summary())

        projections = self._project_gameweeks(
            gameweeks, players, teams, rates, minutes, forecasts
        )

        return ProjectionSet(
            gameweeks=gameweeks,
            players=players,
            teams=teams,
            rates=rates,
            minutes=minutes,
            ratings=ratings,
            forecasts=forecasts,
            projections=projections,
            odds=odds,
            notes=notes,
        )

    # -- steps ---------------------------------------------------------------

    def _blend_advanced(self, rates, players, teams) -> "BlendReport | None":
        """Fold a second expected-goals model into the rates, if one is configured.

        Off unless asked for. A second xG model is a check on the first, not an
        authority over it, so the blend weight is well below one and the report
        says how many players actually matched — a name-matching failure that
        silently blends nothing would be worse than not trying.
        """
        name = str(self.settings.get("advanced.provider", "fpl") or "fpl")
        if name == "fpl":
            return None
        season = str(self.rules.raw.get("season", "2026-27"))

        # Advanced stats come from a different host, so they get their own HTTP
        # client rather than borrowing the FPL one — which a stub client (the
        # synthetic season, for instance) may not have at all.
        http = getattr(self.client, "http", None)
        if http is None:
            http = HttpClient(self.settings.cache_dir / "advanced")

        provider = get_provider(name, http, self.settings)
        if provider is None:
            return BlendReport(source=name, note=f"unknown stats provider '{name}'")
        if not provider.enabled():
            return BlendReport(source=name, note=provider.fetch(season).reason)

        snapshot = provider.fetch(season)
        return blend_rates(
            rates,
            players,
            teams,
            snapshot,
            weight=float(self.settings.get("advanced.weight", 0.35)),
        )

    def _default_horizon(self, boot: Bootstrap) -> list[int]:
        next_gw = boot.next_gameweek()
        start = next_gw.id if next_gw else 1
        horizon = self.settings.planning_horizon
        return [gw for gw in range(start, min(start + horizon, 39))]

    def _fit_ratings(
        self, fixtures: Sequence[Fixture], teams: dict[int, Team], notes: list[str]
    ) -> TeamRatings:
        prior = ratings_from_fpl_strength(list(teams.values()))
        results = results_from_fixtures(fixtures)
        if len(results) < 20:
            notes.append(
                f"Only {len(results)} results this season; team ratings lean on FPL's "
                "published strength grades."
            )
            if not results:
                return prior
        return fit_ratings(
            results,
            team_ids=list(teams),
            half_life_days=float(self.settings.get("model.team_rating_half_life_days", 180)),
            prior=prior,
            prior_weight=max(1.0, 60.0 / max(len(results), 1)),
        )

    def _load_histories(
        self,
        players: dict[int, Player],
        deep: bool,
        limit: int,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> dict[int, dict]:
        """Per-player history, for players plausibly worth owning.

        Fetching all ~700 players is slow on a cold cache, so players who cannot
        realistically be picked (no minutes and unavailable) are skipped.
        """
        if not deep:
            return {}
        # Before a ball is kicked nobody has minutes, and last season's totals are
        # the only thing separating one player from another — so in pre-season
        # everyone's history is worth fetching, cheap enablers most of all.
        preseason = all(p.minutes == 0 for p in players.values())
        if preseason:
            candidates = list(players.values())
        else:
            candidates = [
                p
                for p in players.values()
                if p.minutes > 0 or p.price >= 4.5 or p.selected_by_percent > 0.2
            ]
        candidates.sort(key=lambda p: (-p.minutes, -p.price))
        candidates = candidates[:limit]

        histories: dict[int, dict] = {}
        total = len(candidates)
        for index, player in enumerate(candidates, start=1):
            try:
                histories[player.id] = self.client.element_summary(player.id)
            except Exception as exc:  # noqa: BLE001 - one bad player must not stop a run
                log.debug("history unavailable for %s: %s", player.web_name, exc)
            if progress is not None and (index % 10 == 0 or index == total):
                progress(index, total, "player history")
        return histories

    def _estimate_players(
        self, players: dict[int, Player], histories: dict[int, dict]
    ) -> tuple[dict[int, PlayerRates], dict[int, MinutesProjection]]:
        estimator = RateEstimator(
            shrinkage_90s=float(self.settings.get("model.shrinkage_90s", 6.0))
        )
        half_life = float(self.settings.get("model.form_half_life", 6.0)) / 1.5

        rates: dict[int, PlayerRates] = {}
        minutes: dict[int, MinutesProjection] = {}
        for element, player in players.items():
            summary = histories.get(element, {})
            history_rows = element_history(summary)
            past = list(summary.get("history_past", []) or [])
            rates[element] = estimator.estimate(
                player, past_seasons=past, history_rows=history_rows
            )
            minutes[element] = project_minutes(
                player,
                minutes_history_from_summary(history_rows),
                half_life=half_life,
            )
        return rates, minutes

    def _project_gameweeks(
        self,
        gameweeks: Sequence[int],
        players: dict[int, Player],
        teams: dict[int, Team],
        rates: dict[int, PlayerRates],
        minutes: dict[int, MinutesProjection],
        forecasts: dict[int, FixtureForecast],
    ) -> dict[int, dict[int, PlayerGameweekProjection]]:
        by_team: dict[int, list[Player]] = {}
        for player in players.values():
            by_team.setdefault(player.team_id, []).append(player)

        # Each club's average attacking output over the horizon, so a specific
        # fixture can be expressed as "better or worse than this team's normal".
        team_average_xg: dict[int, float] = {}
        for team_id in teams:
            xgs = [
                f.expected_goals_for(team_id)
                for f in forecasts.values()
                if team_id in (f.home_team, f.away_team)
            ]
            team_average_xg[team_id] = average_team_xg(xgs)

        output: dict[int, dict[int, PlayerGameweekProjection]] = {gw: {} for gw in gameweeks}

        for gameweek in gameweeks:
            gw_fixtures = [f for f in forecasts.values() if f.gameweek == gameweek]
            fixtures_per_team: dict[int, int] = {}
            for fixture in gw_fixtures:
                fixtures_per_team[fixture.home_team] = (
                    fixtures_per_team.get(fixture.home_team, 0) + 1
                )
                fixtures_per_team[fixture.away_team] = (
                    fixtures_per_team.get(fixture.away_team, 0) + 1
                )

            for fixture in gw_fixtures:
                bonus_pmfs = self._bonus_for_fixture(
                    fixture, by_team, rates, minutes, team_average_xg
                )

                for team_id in (fixture.home_team, fixture.away_team):
                    rotation = congestion_rotation_risk(fixtures_per_team.get(team_id, 1))
                    multiplier = team_attack_multiplier(
                        fixture.expected_goals_for(team_id), team_average_xg.get(team_id, 1.4)
                    )
                    # Individual per-90 rates need not add up to what the team is
                    # forecast to score. Rescaling them so they do keeps the
                    # analytic projection consistent with the match model — and
                    # with the simulator, which allocates a drawn scoreline.
                    multiplier *= _goal_conservation_scale(
                        by_team.get(team_id, []),
                        rates,
                        minutes,
                        rotation,
                        multiplier,
                        fixture.expected_goals_for(team_id),
                    )
                    for player in by_team.get(team_id, []):
                        player_rates = rates[player.id]
                        player_minutes = minutes[player.id]
                        if rotation > 0:
                            player_minutes = player_minutes.scaled(1.0 - rotation)
                        if player_minutes.p_appear <= 0.005:
                            continue

                        bonus_pmf = bonus_pmfs.get(player.id)
                        projection = self.points_model.project(
                            element=player.id,
                            rates=player_rates,
                            minutes=player_minutes,
                            forecast=fixture,
                            team_id=team_id,
                            attack_multiplier=multiplier,
                            bonus_pmf=bonus_pmf,
                            expected_bonus=(
                                expected_from_pmf(bonus_pmf) if bonus_pmf is not None else 0.0
                            ),
                        )
                        bucket = output[gameweek].setdefault(
                            player.id,
                            PlayerGameweekProjection(element=player.id, gameweek=gameweek),
                        )
                        bucket.fixtures.append(projection)

            # Players whose club has no fixture this gameweek still need an
            # entry, so that a blank shows up as zero rather than as missing.
            for player in players.values():
                output[gameweek].setdefault(
                    player.id, PlayerGameweekProjection(element=player.id, gameweek=gameweek)
                )

        return output

    def _bonus_for_fixture(
        self,
        fixture: FixtureForecast,
        by_team: dict[int, list[Player]],
        rates: dict[int, PlayerRates],
        minutes: dict[int, MinutesProjection],
        team_average_xg: dict[int, float],
    ) -> dict[int, np.ndarray]:
        """Bonus point distribution for everyone plausibly involved in a fixture."""
        weights = self.rules.bps_weights
        probabilities = fixture.probabilities()
        profiles: list[BpsProfile] = []

        for team_id in (fixture.home_team, fixture.away_team):
            is_home = team_id == fixture.home_team
            p_clean_sheet = probabilities.clean_sheet(home=is_home)
            expected_conceded = fixture.expected_goals_against(team_id)
            multiplier = team_attack_multiplier(
                fixture.expected_goals_for(team_id), team_average_xg.get(team_id, 1.4)
            )
            for player in by_team.get(team_id, []):
                player_minutes = minutes[player.id]
                if player_minutes.p_appear < 0.08:
                    continue
                profiles.append(
                    expected_bps(
                        rates[player.id],
                        player_minutes,
                        weights,
                        p_clean_sheet=p_clean_sheet,
                        expected_conceded=expected_conceded,
                        attack_multiplier=multiplier,
                        opponent_xg=expected_conceded,
                    )
                )

        if not profiles:
            return {}
        return bonus_distribution(
            profiles,
            bonus_points=self.rules.scoring.get("bonus", [3, 2, 1]),
            seed=int(self.settings.get("simulation.seed", 20262027)) + fixture.fixture_id,
        )


def _goal_conservation_scale(
    squad: Sequence[Player],
    rates: dict[int, PlayerRates],
    minutes: dict[int, MinutesProjection],
    rotation: float,
    multiplier: float,
    team_expected_goals: float,
) -> float:
    """Scale factor making a squad's individual xG add up to the team's xG.

    Shrunk per-90 rates are estimated one player at a time, so nothing forces
    them to sum to what the side is expected to score. Left alone, a squad of
    over-estimated individuals quietly inflates every attacking projection for
    that club. The correction is clamped: it is there to fix an accounting
    mismatch, not to override the rate model.
    """
    if team_expected_goals <= 0 or not squad:
        return 1.0
    total = 0.0
    for player in squad:
        player_rates = rates.get(player.id)
        player_minutes = minutes.get(player.id)
        if player_rates is None or player_minutes is None:
            continue
        expected_minutes = player_minutes.expected_minutes * (1.0 - rotation)
        total += player_rates.xg90 * (expected_minutes / 90.0) * multiplier
    if total <= 1e-6:
        return 1.0
    return float(np.clip(team_expected_goals / total, 0.5, 2.0))


def attach_goalscorer_odds(
    projection_set: ProjectionSet, player_goal_probabilities: dict[str, float]
) -> int:
    """Overlay anytime-goalscorer probabilities onto an existing projection set.

    Matching is by normalised surname within the right club, which is the best
    that can be done when bookmakers publish names in their own format.
    Returns the number of players matched.
    """
    by_key: dict[tuple[str, str], int] = {}
    for element, player in projection_set.players.items():
        team = projection_set.teams.get(player.team_id)
        team_key = normalise_team(team.name) if team else ""
        surname = normalise_team(player.raw.get("second_name", "") or player.web_name)
        by_key[(team_key, surname)] = element

    matched = 0
    for name, probability in player_goal_probabilities.items():
        surname = normalise_team(name.split()[-1] if name.split() else name)
        for (team_key, key_surname), element in by_key.items():
            if key_surname == surname:
                for gw_projections in projection_set.projections.values():
                    projection = gw_projections.get(element)
                    if projection:
                        for fixture in projection.fixtures:
                            fixture.anytime_goal_probability = probability
                matched += 1
                break
    return matched


def summarise_projection_set(projection_set: ProjectionSet, top_n: int = 20) -> list[dict]:
    """Top players by expected points over the horizon, for a quick look."""
    rows = []
    for element, player in projection_set.players.items():
        total = projection_set.horizon_points(element)
        if total <= 0:
            continue
        rows.append(
            {
                "element": element,
                "name": player.web_name,
                "team": projection_set.team_name(player.team_id),
                "position": player.position,
                "price": player.price,
                "xpts": round(total, 2),
                "xpts_per_million": round(total / max(player.price, 0.1), 3),
                "minutes_p_start": round(projection_set.minutes[element].p_start, 2),
            }
        )
    rows.sort(key=lambda r: -r["xpts"])
    return rows[:top_n]
