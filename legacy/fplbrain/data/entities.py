"""Typed views over the FPL API payloads.

The API returns wide, loosely-typed dictionaries whose exact field set drifts
between seasons. Rather than mirror every field, these classes keep the raw dict
and expose the fields the engine actually relies on, each with a safe default.
Anything unusual stays reachable through ``.raw``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..config import ELEMENT_TYPE_TO_POSITION


def _f(value: Any, default: float = 0.0) -> float:
    """Coerce an API value to float. The API sends numbers as strings a lot."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _i(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass
class Team:
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def id(self) -> int:
        return _i(self.raw.get("id"))

    @property
    def name(self) -> str:
        return str(self.raw.get("name", ""))

    @property
    def short_name(self) -> str:
        return str(self.raw.get("short_name", ""))

    @property
    def code(self) -> int:
        return _i(self.raw.get("code"))

    # FPL's own strength ratings, on a 1000-1400ish scale. Useful as a prior
    # before enough of the season has been played to fit our own ratings.
    @property
    def strength_attack_home(self) -> float:
        return _f(self.raw.get("strength_attack_home"), 1000.0)

    @property
    def strength_attack_away(self) -> float:
        return _f(self.raw.get("strength_attack_away"), 1000.0)

    @property
    def strength_defence_home(self) -> float:
        return _f(self.raw.get("strength_defence_home"), 1000.0)

    @property
    def strength_defence_away(self) -> float:
        return _f(self.raw.get("strength_defence_away"), 1000.0)


@dataclass
class Player:
    raw: dict = field(repr=False, default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.web_name} ({self.position})"

    @property
    def id(self) -> int:
        return _i(self.raw.get("id"))

    @property
    def web_name(self) -> str:
        return str(self.raw.get("web_name", ""))

    @property
    def full_name(self) -> str:
        return f"{self.raw.get('first_name', '')} {self.raw.get('second_name', '')}".strip()

    @property
    def team_id(self) -> int:
        return _i(self.raw.get("team"))

    @property
    def position(self) -> str:
        return ELEMENT_TYPE_TO_POSITION.get(_i(self.raw.get("element_type")), "MID")

    @property
    def price(self) -> float:
        """Current price in £m (the API stores tenths)."""
        return _i(self.raw.get("now_cost")) / 10.0

    @property
    def total_points(self) -> int:
        return _i(self.raw.get("total_points"))

    @property
    def minutes(self) -> int:
        return _i(self.raw.get("minutes"))

    @property
    def starts(self) -> int:
        return _i(self.raw.get("starts"))

    @property
    def form(self) -> float:
        return _f(self.raw.get("form"))

    @property
    def points_per_game(self) -> float:
        return _f(self.raw.get("points_per_game"))

    @property
    def selected_by_percent(self) -> float:
        return _f(self.raw.get("selected_by_percent"))

    @property
    def status(self) -> str:
        """One of a(vailable), d(oubtful), i(njured), s(uspended), u(navailable), n(ot in squad)."""
        return str(self.raw.get("status", "a"))

    @property
    def news(self) -> str:
        return str(self.raw.get("news", "") or "")

    @property
    def chance_of_playing_next_round(self) -> float | None:
        value = self.raw.get("chance_of_playing_next_round")
        return None if value is None else _f(value)

    @property
    def expected_goals(self) -> float:
        return _f(self.raw.get("expected_goals"))

    @property
    def expected_assists(self) -> float:
        return _f(self.raw.get("expected_assists"))

    @property
    def expected_goals_conceded(self) -> float:
        return _f(self.raw.get("expected_goals_conceded"))

    @property
    def defensive_contribution(self) -> float:
        """Season total of defensive-contribution points scored, if exposed."""
        return _f(self.raw.get("defensive_contribution"))

    @property
    def clearances_blocks_interceptions(self) -> float:
        return _f(self.raw.get("clearances_blocks_interceptions"))

    @property
    def tackles(self) -> float:
        return _f(self.raw.get("tackles"))

    @property
    def recoveries(self) -> float:
        return _f(self.raw.get("recoveries"))

    @property
    def saves(self) -> float:
        return _f(self.raw.get("saves"))

    @property
    def bps(self) -> float:
        return _f(self.raw.get("bps"))

    @property
    def bonus(self) -> int:
        return _i(self.raw.get("bonus"))

    @property
    def penalties_order(self) -> int | None:
        value = self.raw.get("penalties_order")
        return None if value is None else _i(value)

    @property
    def corners_order(self) -> int | None:
        value = self.raw.get("corners_and_indirect_freekicks_order")
        return None if value is None else _i(value)

    @property
    def freekicks_order(self) -> int | None:
        value = self.raw.get("direct_freekicks_order")
        return None if value is None else _i(value)

    @property
    def ep_next(self) -> float:
        """FPL's own expected points for the next gameweek — a sanity check."""
        return _f(self.raw.get("ep_next"))

    @property
    def cost_change_event(self) -> int:
        return _i(self.raw.get("cost_change_event"))

    @property
    def transfers_in_event(self) -> int:
        return _i(self.raw.get("transfers_in_event"))

    @property
    def transfers_out_event(self) -> int:
        return _i(self.raw.get("transfers_out_event"))

    @property
    def ninetys(self) -> float:
        return self.minutes / 90.0

    def per90(self, stat: str) -> float:
        """A season per-90 rate for any numeric field, guarding against 0 minutes."""
        if self.minutes <= 0:
            return 0.0
        return _f(self.raw.get(stat)) * 90.0 / self.minutes

    @property
    def is_available(self) -> bool:
        return self.status == "a"

    @property
    def availability(self) -> float:
        """Probability the player is fit and selectable, in [0, 1].

        FPL publishes ``chance_of_playing_next_round`` when a player is flagged.
        When it is absent the status letter is the only signal: suspended and
        unavailable players cannot play at all, and a doubt with no percentage
        is treated as a serious one.
        """
        chance = self.chance_of_playing_next_round
        if chance is not None:
            return max(0.0, min(1.0, chance / 100.0))
        return {"a": 1.0, "d": 0.5, "i": 0.0, "s": 0.0, "u": 0.0, "n": 0.0}.get(self.status, 1.0)


@dataclass
class Gameweek:
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def id(self) -> int:
        return _i(self.raw.get("id"))

    @property
    def name(self) -> str:
        return str(self.raw.get("name", f"Gameweek {self.id}"))

    @property
    def deadline(self) -> datetime | None:
        return parse_dt(self.raw.get("deadline_time"))

    @property
    def finished(self) -> bool:
        return bool(self.raw.get("finished"))

    @property
    def data_checked(self) -> bool:
        return bool(self.raw.get("data_checked"))

    @property
    def is_next(self) -> bool:
        return bool(self.raw.get("is_next"))

    @property
    def is_current(self) -> bool:
        return bool(self.raw.get("is_current"))

    @property
    def average_score(self) -> float:
        return _f(self.raw.get("average_entry_score"))

    @property
    def highest_score(self) -> float:
        return _f(self.raw.get("highest_score"))

    @property
    def seconds_to_deadline(self) -> float | None:
        deadline = self.deadline
        if deadline is None:
            return None
        return (deadline - datetime.now(timezone.utc)).total_seconds()


@dataclass
class Fixture:
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def id(self) -> int:
        return _i(self.raw.get("id"))

    @property
    def gameweek(self) -> int | None:
        value = self.raw.get("event")
        return None if value is None else _i(value)

    @property
    def team_h(self) -> int:
        return _i(self.raw.get("team_h"))

    @property
    def team_a(self) -> int:
        return _i(self.raw.get("team_a"))

    @property
    def finished(self) -> bool:
        return bool(self.raw.get("finished"))

    @property
    def kickoff(self) -> datetime | None:
        return parse_dt(self.raw.get("kickoff_time"))

    @property
    def team_h_score(self) -> int | None:
        value = self.raw.get("team_h_score")
        return None if value is None else _i(value)

    @property
    def team_a_score(self) -> int | None:
        value = self.raw.get("team_a_score")
        return None if value is None else _i(value)

    @property
    def difficulty_h(self) -> int:
        return _i(self.raw.get("team_h_difficulty"), 3)

    @property
    def difficulty_a(self) -> int:
        return _i(self.raw.get("team_a_difficulty"), 3)

    def opponent_of(self, team_id: int) -> int | None:
        if team_id == self.team_h:
            return self.team_a
        if team_id == self.team_a:
            return self.team_h
        return None

    def is_home(self, team_id: int) -> bool:
        return team_id == self.team_h

    def involves(self, team_id: int) -> bool:
        return team_id in (self.team_h, self.team_a)


@dataclass
class Pick:
    """One row of a manager's squad for a gameweek."""

    element: int
    position: int  # 1-11 start, 12-15 bench, in bench order
    multiplier: int
    is_captain: bool
    is_vice_captain: bool
    selling_price: float | None = None
    purchase_price: float | None = None

    @classmethod
    def from_api(cls, raw: dict) -> "Pick":
        return cls(
            element=_i(raw.get("element")),
            position=_i(raw.get("position")),
            multiplier=_i(raw.get("multiplier")),
            is_captain=bool(raw.get("is_captain")),
            is_vice_captain=bool(raw.get("is_vice_captain")),
            selling_price=(
                _i(raw["selling_price"]) / 10.0 if raw.get("selling_price") is not None else None
            ),
            purchase_price=(
                _i(raw["purchase_price"]) / 10.0 if raw.get("purchase_price") is not None else None
            ),
        )

    @property
    def is_starter(self) -> bool:
        return self.position <= 11


@dataclass
class ManagerSquad:
    """A manager's squad plus the state needed to plan transfers from it."""

    entry_id: int
    gameweek: int
    picks: list[Pick]
    bank: float = 0.0
    squad_value: float = 0.0
    free_transfers: int = 1
    chips_used: list[str] = field(default_factory=list)
    chip_active: str | None = None
    source: str = "public"  # "authenticated" when read from /my-team/
    total_points: int = 0
    overall_rank: int | None = None

    @property
    def element_ids(self) -> list[int]:
        return [p.element for p in self.picks]

    @property
    def captain(self) -> int | None:
        for pick in self.picks:
            if pick.is_captain:
                return pick.element
        return None

    @property
    def vice_captain(self) -> int | None:
        for pick in self.picks:
            if pick.is_vice_captain:
                return pick.element
        return None

    def selling_prices(self, fallback: dict[int, float]) -> dict[int, float]:
        """Selling price per player, falling back to current price when unknown.

        Only the authenticated endpoint exposes true selling prices, which matter
        because profit is halved on sale. Without it the engine assumes you can
        sell at market price, which slightly overstates your budget.
        """
        out: dict[int, float] = {}
        for pick in self.picks:
            if pick.selling_price is not None:
                out[pick.element] = pick.selling_price
            else:
                out[pick.element] = fallback.get(pick.element, 0.0)
        return out
