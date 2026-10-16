"""A synthetic Premier League, for testing and for demos.

The engine talks to a live API that cannot be reached from a test runner, needs
no credentials to be useful, and changes shape between seasons. This module
generates payloads with the same schema as the real endpoints, so every layer
above the client — rates, minutes, points, optimiser, simulator, report — can be
exercised end to end without a network connection.

It is also how ``fplbrain demo`` works: a complete worked example on a plausible
fake league, so the tool can be evaluated before wiring up a real team.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .fpl import Bootstrap, FplClient

TEAM_NAMES = [
    ("Arsenal", "ARS"),
    ("Aston Villa", "AVL"),
    ("Bournemouth", "BOU"),
    ("Brentford", "BRE"),
    ("Brighton", "BHA"),
    ("Burnley", "BUR"),
    ("Chelsea", "CHE"),
    ("Crystal Palace", "CRY"),
    ("Everton", "EVE"),
    ("Fulham", "FUL"),
    ("Leeds", "LEE"),
    ("Liverpool", "LIV"),
    ("Man City", "MCI"),
    ("Man Utd", "MUN"),
    ("Newcastle", "NEW"),
    ("Nott'm Forest", "NFO"),
    ("Sunderland", "SUN"),
    ("Spurs", "TOT"),
    ("West Ham", "WHU"),
    ("Wolves", "WOL"),
]

FIRST_NAMES = [
    "Alex", "Ben", "Callum", "Diogo", "Eze", "Finn", "Gabriel", "Harvey", "Ivan",
    "James", "Kai", "Luis", "Mateo", "Noah", "Oscar", "Pedro", "Rafael", "Sam",
    "Tomas", "Viktor",
]
LAST_NAMES = [
    "Adams", "Bailey", "Costa", "Duarte", "Evans", "Fernandes", "Gomez", "Hughes",
    "Iversen", "Jansen", "Kovac", "Lindqvist", "Moreno", "Novak", "Owusu", "Pereira",
    "Quinn", "Rossi", "Silva", "Thomas", "Ubbe", "Vargas", "Walsh", "Yilmaz", "Zanetti",
]


@dataclass
class SyntheticSeason:
    """Generates a coherent fake season."""

    n_teams: int = 20
    players_per_team: int = 22
    gameweeks_played: int = 0
    total_gameweeks: int = 38
    seed: int = 20262027
    season_start: datetime = datetime(2026, 8, 21, 18, 30, tzinfo=timezone.utc)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.teams = self._make_teams()
        self.players = self._make_players()
        self.fixtures = self._make_fixtures()
        self._play_gameweeks()

    # -- construction --------------------------------------------------------

    def _make_teams(self) -> list[dict]:
        teams = []
        for index, (name, short) in enumerate(TEAM_NAMES[: self.n_teams], start=1):
            # A plausible spread of quality, strongest first in the list order
            # after a shuffle so tests do not depend on real-world standings.
            quality = self.rng.uniform(-0.45, 0.45)
            teams.append(
                {
                    "id": index,
                    "code": 100 + index,
                    "name": name,
                    "short_name": short,
                    "strength": 3,
                    "strength_attack_home": 1100 + quality * 260,
                    "strength_attack_away": 1080 + quality * 260,
                    "strength_defence_home": 1100 + quality * 240,
                    "strength_defence_away": 1080 + quality * 240,
                    "_quality": quality,
                }
            )
        return teams

    def _make_players(self) -> list[dict]:
        players: list[dict] = []
        element_id = 1
        shape = [(1, 3), (2, 7), (3, 8), (4, 4)]  # element_type, count
        for team in self.teams:
            # Surnames are unique within a club, as they nearly always are in
            # reality — otherwise the display-name disambiguation dominates
            # every table in the demo output.
            surnames = self.rng.sample(LAST_NAMES, k=sum(count for _, count in shape))
            for element_type, count in shape:
                for slot in range(count):
                    players.append(
                        self._make_player(
                            element_id, team, element_type, slot, surnames.pop()
                        )
                    )
                    element_id += 1
        return players

    def _make_player(
        self, element_id: int, team: dict, element_type: int, slot: int, last: str
    ) -> dict:
        rng = self.rng
        first = rng.choice(FIRST_NAMES)
        quality = team["_quality"]

        # First-choice players are better and more expensive than the reserves.
        seniority = max(0.0, 1.0 - slot * 0.28)
        base_price = {1: 4.5, 2: 4.5, 3: 5.0, 4: 5.5}[element_type]
        price = base_price + seniority * (3.5 + quality * 3.0) + rng.uniform(-0.3, 0.6)
        price = round(max(3.9, min(15.0, price)) * 2) / 2

        minutes_per_game = 90 * seniority * rng.uniform(0.75, 1.05)
        played = self.gameweeks_played
        minutes = int(minutes_per_game * played)

        xg90 = {1: 0.004, 2: 0.05, 3: 0.16, 4: 0.38}[element_type] * (
            0.5 + seniority
        ) * (1 + quality)
        xa90 = {1: 0.01, 2: 0.06, 3: 0.14, 4: 0.12}[element_type] * (
            0.5 + seniority
        ) * (1 + quality)
        cbi90 = {1: 1.2, 2: 6.5, 3: 2.6, 4: 1.2}[element_type] * rng.uniform(0.6, 1.5)
        tackles90 = {1: 0.1, 2: 2.0, 3: 1.9, 4: 0.9}[element_type] * rng.uniform(0.5, 1.6)
        recoveries90 = {1: 2.5, 2: 4.5, 3: 5.5, 4: 3.0}[element_type] * rng.uniform(0.6, 1.5)
        saves90 = 3.1 * rng.uniform(0.8, 1.2) if element_type == 1 else 0.0

        ninetys = minutes / 90.0
        goals = int(xg90 * ninetys * rng.uniform(0.7, 1.4))
        assists = int(xa90 * ninetys * rng.uniform(0.7, 1.4))
        points = int(
            (2 * played * seniority)
            + goals * {1: 6, 2: 6, 3: 5, 4: 4}[element_type]
            + assists * 3
            + rng.uniform(0, 6)
        ) if played else 0

        status = "a"
        news = ""
        chance = None
        roll = rng.random()
        if roll < 0.04:
            status, news, chance = "i", "Hamstring injury - expected back soon", 0
        elif roll < 0.09:
            status, news, chance = "d", "Knock - 75% chance of playing", 75

        return {
            "id": element_id,
            "code": 200000 + element_id,
            "first_name": first,
            "second_name": last,
            "web_name": last,
            "team": team["id"],
            "team_code": team["code"],
            "element_type": element_type,
            "now_cost": int(price * 10),
            "total_points": points,
            "event_points": 0,
            "points_per_game": round(points / played, 1) if played else 0.0,
            "form": round(rng.uniform(0, 6) * seniority, 1),
            "selected_by_percent": round(
                max(0.1, seniority * rng.uniform(0.2, 35) * (1 + quality)), 1
            ),
            "minutes": minutes,
            "starts": int(played * seniority) if played else 0,
            "goals_scored": goals,
            "assists": assists,
            "clean_sheets": int(played * 0.3) if played else 0,
            "goals_conceded": int(played * 1.3) if played else 0,
            "own_goals": 0,
            "penalties_saved": 0,
            "penalties_missed": 0,
            "yellow_cards": int(played * 0.15 * rng.uniform(0, 2)),
            "red_cards": 0,
            "saves": int(saves90 * ninetys),
            "bonus": int(points * 0.08),
            "bps": int(points * 3.2),
            "expected_goals": round(xg90 * ninetys, 2),
            "expected_assists": round(xa90 * ninetys, 2),
            "expected_goals_conceded": round(1.3 * ninetys, 2),
            "clearances_blocks_interceptions": int(cbi90 * ninetys),
            "tackles": int(tackles90 * ninetys),
            "recoveries": int(recoveries90 * ninetys),
            "defensive_contribution": int(played * 0.3) if element_type in (2, 3) else 0,
            "status": status,
            "news": news,
            "chance_of_playing_next_round": chance,
            "chance_of_playing_this_round": chance,
            "ep_next": round(rng.uniform(1, 7) * seniority, 1),
            "cost_change_event": 0,
            "transfers_in_event": 0,
            "transfers_out_event": 0,
            "penalties_order": 1 if (element_type == 4 and slot == 0) else None,
            "corners_and_indirect_freekicks_order": 1 if (element_type == 3 and slot == 0) else None,
            "direct_freekicks_order": None,
            "_seniority": seniority,
            "_xg90": xg90,
            "_xa90": xa90,
        }

    def _make_fixtures(self) -> list[dict]:
        """A double round robin, trimmed to the number of gameweeks requested."""
        teams = [t["id"] for t in self.teams]
        rng = self.rng
        fixtures: list[dict] = []
        fixture_id = 1

        rotation = teams[:]
        for round_index in range(self.total_gameweeks):
            gameweek = round_index + 1
            half = len(rotation) // 2
            pairs = list(zip(rotation[:half], rotation[half:][::-1]))
            kickoff = self.season_start + timedelta(days=7 * round_index)
            for home, away in pairs:
                if round_index % 2 == 1:
                    home, away = away, home
                fixtures.append(
                    {
                        "id": fixture_id,
                        "event": gameweek,
                        "team_h": home,
                        "team_a": away,
                        "team_h_score": None,
                        "team_a_score": None,
                        "finished": False,
                        "kickoff_time": kickoff.isoformat().replace("+00:00", "Z"),
                        "team_h_difficulty": rng.randint(2, 5),
                        "team_a_difficulty": rng.randint(2, 5),
                    }
                )
                fixture_id += 1
            rotation = [rotation[0]] + [rotation[-1]] + rotation[1:-1]
        return fixtures

    def _play_gameweeks(self) -> None:
        """Fill in results for gameweeks that have already happened."""
        rng = self.rng
        quality = {t["id"]: t["_quality"] for t in self.teams}
        for fixture in self.fixtures:
            if fixture["event"] > self.gameweeks_played:
                continue
            home_lambda = max(0.3, 1.55 * (1 + quality[fixture["team_h"]]))
            away_lambda = max(0.3, 1.20 * (1 + quality[fixture["team_a"]]))
            fixture["team_h_score"] = _poisson(rng, home_lambda)
            fixture["team_a_score"] = _poisson(rng, away_lambda)
            fixture["finished"] = True

    # -- payloads ------------------------------------------------------------

    def bootstrap_payload(self) -> dict:
        events = []
        for gameweek in range(1, self.total_gameweeks + 1):
            deadline = self.season_start + timedelta(days=7 * (gameweek - 1))
            events.append(
                {
                    "id": gameweek,
                    "name": f"Gameweek {gameweek}",
                    "deadline_time": deadline.isoformat().replace("+00:00", "Z"),
                    "finished": gameweek <= self.gameweeks_played,
                    "data_checked": gameweek <= self.gameweeks_played,
                    "is_current": gameweek == self.gameweeks_played,
                    "is_next": gameweek == self.gameweeks_played + 1,
                    "average_entry_score": 52 if gameweek <= self.gameweeks_played else 0,
                    "highest_score": 120 if gameweek <= self.gameweeks_played else 0,
                }
            )
        return {
            "events": events,
            "teams": [{k: v for k, v in t.items() if not k.startswith("_")} for t in self.teams],
            "elements": [
                {k: v for k, v in p.items() if not k.startswith("_")} for p in self.players
            ],
            "element_types": [
                {"id": 1, "singular_name_short": "GKP", "squad_select": 2, "squad_min_play": 1,
                 "squad_max_play": 1},
                {"id": 2, "singular_name_short": "DEF", "squad_select": 5, "squad_min_play": 3,
                 "squad_max_play": 5},
                {"id": 3, "singular_name_short": "MID", "squad_select": 5, "squad_min_play": 2,
                 "squad_max_play": 5},
                {"id": 4, "singular_name_short": "FWD", "squad_select": 3, "squad_min_play": 1,
                 "squad_max_play": 3},
            ],
            "total_players": 11_000_000,
            "game_settings": {"squad_squadplay": 11, "squad_total_spend": 1000},
        }

    def _past_season_payload(self, player: dict) -> list[dict]:
        """Last season's totals for a player, as ``history_past`` returns them.

        This matters more than it looks. Before a ball is kicked, last season is
        the *only* thing that separates one midfielder from another — without it
        every player collapses onto the positional prior and the engine cannot
        tell a 20-goal striker from a squad filler. Roughly one player in eight
        is treated as a newcomer with no record, which is about right for a
        Premier League summer.
        """
        rng = random.Random(self.seed * 7 + player["id"])
        if rng.random() < 0.12:
            return []

        seniority = player.get("_seniority", 0.6)
        # Last season's minutes vary a lot even for established players.
        minutes = int(38 * 90 * seniority * rng.uniform(0.45, 0.98))
        if minutes < 450:
            return []
        ninetys = minutes / 90.0

        xg90 = player.get("_xg90", 0.1) * rng.uniform(0.8, 1.25)
        xa90 = player.get("_xa90", 0.1) * rng.uniform(0.8, 1.25)
        element_type = player["element_type"]
        cbi90 = {1: 1.2, 2: 6.5, 3: 2.6, 4: 1.2}[element_type] * rng.uniform(0.6, 1.5)
        tackles90 = {1: 0.1, 2: 2.0, 3: 1.9, 4: 0.9}[element_type] * rng.uniform(0.5, 1.6)
        recoveries90 = {1: 2.5, 2: 4.5, 3: 5.5, 4: 3.0}[element_type] * rng.uniform(0.6, 1.5)

        goals = int(xg90 * ninetys * rng.uniform(0.75, 1.3))
        assists = int(xa90 * ninetys * rng.uniform(0.75, 1.3))
        return [
            {
                "season_name": "2025/26",
                "element_code": player["code"],
                "start_cost": player["now_cost"] - rng.randint(-5, 5),
                "end_cost": player["now_cost"],
                "total_points": int(
                    2 * ninetys
                    + goals * {1: 6, 2: 6, 3: 5, 4: 4}[element_type]
                    + assists * 3
                ),
                "minutes": minutes,
                "starts": int(ninetys * rng.uniform(0.85, 1.0)),
                "goals_scored": goals,
                "assists": assists,
                "clean_sheets": int(ninetys * 0.3),
                "goals_conceded": int(ninetys * 1.3),
                "own_goals": 0,
                "penalties_saved": 0,
                "penalties_missed": 0,
                "yellow_cards": int(ninetys * 0.15 * rng.uniform(0, 2)),
                "red_cards": 0,
                "saves": int(3.1 * ninetys) if element_type == 1 else 0,
                "bonus": int(ninetys * 0.25),
                "bps": int(ninetys * 20),
                "expected_goals": round(xg90 * ninetys, 2),
                "expected_assists": round(xa90 * ninetys, 2),
                "expected_goals_conceded": round(1.3 * ninetys, 2),
                "clearances_blocks_interceptions": int(cbi90 * ninetys),
                "tackles": int(tackles90 * ninetys),
                "recoveries": int(recoveries90 * ninetys),
            }
        ]

    def element_summary_payload(self, element_id: int) -> dict:
        """Per-gameweek history for one player, consistent with their season totals."""
        player = next((p for p in self.players if p["id"] == element_id), None)
        if player is None:
            return {"history": [], "history_past": [], "fixtures": []}

        rng = random.Random(self.seed + element_id)
        seniority = player.get("_seniority", 0.6)
        past = self._past_season_payload(player)
        history = []
        for gameweek in range(1, self.gameweeks_played + 1):
            fixture = next(
                (
                    f
                    for f in self.fixtures
                    if f["event"] == gameweek
                    and player["team"] in (f["team_h"], f["team_a"])
                ),
                None,
            )
            if fixture is None:
                continue
            started = rng.random() < seniority
            minutes = int(rng.uniform(70, 90)) if started else (
                int(rng.uniform(1, 25)) if rng.random() < 0.4 else 0
            )
            goals = 1 if rng.random() < player.get("_xg90", 0.1) * minutes / 90 else 0
            assists = 1 if rng.random() < player.get("_xa90", 0.1) * minutes / 90 else 0
            is_home = player["team"] == fixture["team_h"]
            conceded = (fixture["team_a_score"] or 0) if is_home else (fixture["team_h_score"] or 0)
            clean_sheet = 1 if conceded == 0 and minutes >= 60 else 0
            points = (
                (2 if minutes >= 60 else (1 if minutes else 0))
                + goals * {1: 6, 2: 6, 3: 5, 4: 4}[player["element_type"]]
                + assists * 3
                + clean_sheet * {1: 4, 2: 4, 3: 1, 4: 0}[player["element_type"]]
            )
            history.append(
                {
                    "element": element_id,
                    "fixture": fixture["id"],
                    "opponent_team": fixture["team_a"] if is_home else fixture["team_h"],
                    "round": gameweek,
                    "was_home": is_home,
                    "minutes": minutes,
                    "total_points": points,
                    "goals_scored": goals,
                    "assists": assists,
                    "clean_sheets": clean_sheet,
                    "goals_conceded": conceded if minutes else 0,
                    "saves": int(rng.uniform(0, 5)) if player["element_type"] == 1 else 0,
                    "bonus": 3 if goals and rng.random() < 0.5 else 0,
                    "bps": int(points * 3 + rng.uniform(0, 12)),
                    "yellow_cards": 1 if rng.random() < 0.12 else 0,
                    "red_cards": 0,
                    "own_goals": 0,
                    "penalties_saved": 0,
                    "penalties_missed": 0,
                    "expected_goals": round(player.get("_xg90", 0.1) * minutes / 90, 2),
                    "expected_assists": round(player.get("_xa90", 0.1) * minutes / 90, 2),
                    "expected_goals_conceded": round(1.3 * minutes / 90, 2),
                    "clearances_blocks_interceptions": int(
                        player["clearances_blocks_interceptions"]
                        / max(self.gameweeks_played, 1)
                        * rng.uniform(0.5, 1.6)
                    ),
                    "recoveries": int(
                        player["recoveries"] / max(self.gameweeks_played, 1) * rng.uniform(0.5, 1.6)
                    ),
                    "tackles": int(
                        player["tackles"] / max(self.gameweeks_played, 1) * rng.uniform(0.5, 1.6)
                    ),
                    "defensive_contribution": 2 if rng.random() < 0.28 else 0,
                    "starts": 1 if started else 0,
                    "value": player["now_cost"],
                }
            )
        return {"history": history, "history_past": past, "fixtures": []}


def _poisson(rng: random.Random, lam: float) -> int:
    """Knuth's Poisson sampler — avoids a numpy dependency in the generator."""
    import math

    limit = math.exp(-lam)
    count = 0
    product = rng.random()
    while product > limit and count < 12:
        count += 1
        product *= rng.random()
    return count


class SyntheticClient(FplClient):
    """An :class:`FplClient` that serves a :class:`SyntheticSeason` instead of HTTP."""

    def __init__(self, season: SyntheticSeason | None = None, **kwargs: Any):
        self.season = season or SyntheticSeason()
        self._bootstrap = Bootstrap(self.season.bootstrap_payload())
        # Deliberately skip the parent constructor: there is no HTTP layer here.
        from ..config import get_settings

        self.settings = kwargs.get("settings") or get_settings()
        self.http = None  # type: ignore[assignment]

    def bootstrap(self, force_refresh: bool = False) -> Bootstrap:
        return self._bootstrap

    def fixtures(self, gameweek: int | None = None, force_refresh: bool = False):
        from .entities import Fixture

        rows = self.season.fixtures
        if gameweek is not None:
            rows = [f for f in rows if f["event"] == gameweek]
        return [Fixture(row) for row in rows]

    def element_summary(self, element_id: int, force_refresh: bool = False) -> dict:
        return self.season.element_summary_payload(element_id)

    def classic_league(self, league_id: int, page: int = 1) -> dict:
        """A fake overall league, so crowd sampling can be exercised offline."""
        page_size = 50
        start = (page - 1) * page_size
        results = []
        for offset in range(page_size):
            rank = start + offset + 1
            results.append(
                {
                    "id": 1_000_000 + rank,
                    "entry": 500_000 + rank,
                    "rank": rank,
                    "last_rank": rank,
                    "total": 2000 - rank,
                    "player_name": f"Manager {rank}",
                    "entry_name": f"Team {rank}",
                }
            )
        return {
            "league": {"id": league_id, "name": "Overall"},
            "standings": {"page": page, "results": results, "has_next": page < 20},
        }

    def entry_picks(self, entry_id: int, gameweek: int, force_refresh: bool = False) -> dict:
        """Picks in the API's shape, derived from this entry's synthetic squad.

        Mirrors the real endpoint's most important behaviour: a gameweek that
        has not been played yet has no public picks.
        """
        if gameweek > self.season.gameweeks_played:
            from ..util.http import HttpError

            raise HttpError(f"picks for GW{gameweek} are not public yet")

        squad = self.squad(entry_id)
        return {
            "active_chip": "wildcard" if entry_id % 17 == 0 else None,
            "entry_history": {
                "event": gameweek,
                "bank": int(squad.bank * 10),
                "value": int(squad.squad_value * 10),
            },
            "picks": [
                {
                    "element": pick.element,
                    "position": pick.position,
                    "multiplier": pick.multiplier,
                    "is_captain": pick.is_captain,
                    "is_vice_captain": pick.is_vice_captain,
                }
                for pick in squad.picks
            ],
        }


    @property
    def has_credentials(self) -> bool:
        return False

    def squad(self, entry_id: int, gameweek: int | None = None):
        """A plausible starting squad, so the demo has something to advise on."""
        from .entities import ManagerSquad, Pick

        rng = random.Random(self.season.seed + entry_id)
        needed = {1: 2, 2: 5, 3: 5, 4: 3}
        budget = 1000

        # Fill every slot with the cheapest legal option first, so the squad is
        # always complete, then spend the remaining budget on upgrades.
        club_counts: dict[int, int] = {}
        chosen: list[dict] = []
        for element_type, count in needed.items():
            pool = sorted(
                (p for p in self.season.players if p["element_type"] == element_type),
                key=lambda p: p["now_cost"],
            )
            taken = 0
            for player in pool:
                if taken >= count:
                    break
                if club_counts.get(player["team"], 0) >= 3:
                    continue
                chosen.append(player)
                club_counts[player["team"]] = club_counts.get(player["team"], 0) + 1
                taken += 1

        chosen_ids = {p["id"] for p in chosen}
        for _ in range(400):
            spent = sum(p["now_cost"] for p in chosen)
            slack = budget - spent
            if slack <= 0:
                break
            index = rng.randrange(len(chosen))
            current = chosen[index]
            upgrades = [
                p
                for p in self.season.players
                if p["element_type"] == current["element_type"]
                and p["id"] not in chosen_ids
                and 0 < p["now_cost"] - current["now_cost"] <= slack
                and (
                    p["team"] == current["team"]
                    or club_counts.get(p["team"], 0) < 3
                )
            ]
            if not upgrades:
                continue
            replacement = rng.choice(upgrades)
            club_counts[current["team"]] -= 1
            club_counts[replacement["team"]] = club_counts.get(replacement["team"], 0) + 1
            chosen_ids.discard(current["id"])
            chosen_ids.add(replacement["id"])
            chosen[index] = replacement

        order = {1: 0, 2: 1, 3: 2, 4: 3}
        chosen.sort(key=lambda p: (order[p["element_type"]], -p["now_cost"]))
        picks = []
        for index, player in enumerate(chosen, start=1):
            picks.append(
                Pick(
                    element=player["id"],
                    position=index,
                    multiplier=1 if index <= 11 else 0,
                    is_captain=index == 1,
                    is_vice_captain=index == 2,
                    selling_price=player["now_cost"] / 10.0,
                    purchase_price=player["now_cost"] / 10.0,
                )
            )
        spent = sum(p["now_cost"] for p in chosen) / 10.0
        return ManagerSquad(
            entry_id=entry_id,
            gameweek=max(self.season.gameweeks_played, 1),
            picks=picks,
            bank=round(100.0 - spent, 1),
            squad_value=spent,
            free_transfers=1,
            source="synthetic",
        )
