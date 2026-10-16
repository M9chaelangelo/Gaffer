"""Client for the official Fantasy Premier League API.

Everything here is read-only. The public endpoints need no credentials; the one
exception is ``/api/my-team/{id}/``, which is the only way to see your squad
*before* a deadline together with your bank balance and free-transfer count.
That endpoint needs a logged-in session cookie — see :meth:`FplClient.my_team`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..config import Settings, get_settings
from ..util.http import HttpClient, HttpError
from .entities import Fixture, Gameweek, ManagerSquad, Pick, Player, Team

log = logging.getLogger(__name__)

BASE = "https://fantasy.premierleague.com/api"


class AuthRequired(RuntimeError):
    """Raised when an endpoint needs a session cookie that we do not have."""


@dataclass
class Bootstrap:
    """The ``bootstrap-static`` payload: players, teams, gameweeks, settings."""

    raw: dict

    @property
    def players(self) -> list[Player]:
        return [Player(row) for row in self.raw.get("elements", [])]

    @property
    def teams(self) -> list[Team]:
        return [Team(row) for row in self.raw.get("teams", [])]

    @property
    def gameweeks(self) -> list[Gameweek]:
        return [Gameweek(row) for row in self.raw.get("events", [])]

    @property
    def total_players(self) -> int:
        """Number of managers playing — the denominator for rank targets."""
        return int(self.raw.get("total_players", 0) or 0)

    @property
    def game_settings(self) -> dict:
        return self.raw.get("game_settings", {}) or {}

    def player_map(self) -> dict[int, Player]:
        return {p.id: p for p in self.players}

    def team_map(self) -> dict[int, Team]:
        return {t.id: t for t in self.teams}

    def current_gameweek(self) -> Gameweek | None:
        for gw in self.gameweeks:
            if gw.is_current:
                return gw
        return None

    def next_gameweek(self) -> Gameweek | None:
        for gw in self.gameweeks:
            if gw.is_next:
                return gw
        # Before the season opens nothing is flagged current or next, so fall
        # back to the first gameweek that has not finished.
        for gw in self.gameweeks:
            if not gw.finished:
                return gw
        return None

    def finished_gameweeks(self) -> list[int]:
        return [gw.id for gw in self.gameweeks if gw.finished]


class FplClient:
    """Thin, cached wrapper over the FPL endpoints the engine needs."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        cache_dir: Path | None = None,
        offline: bool = False,
        http: HttpClient | None = None,
    ):
        self.settings = settings or get_settings()
        self.http = http or HttpClient(
            cache_dir or self.settings.cache_dir / "fpl", offline=offline
        )

    # -- public endpoints ----------------------------------------------------

    def bootstrap(self, force_refresh: bool = False) -> Bootstrap:
        payload = self.http.get_json(
            f"{BASE}/bootstrap-static/",
            ttl=self.settings.cache_ttl("bootstrap", 3600),
            force_refresh=force_refresh,
        )
        return Bootstrap(payload)

    def fixtures(self, gameweek: int | None = None, force_refresh: bool = False) -> list[Fixture]:
        params = {"event": gameweek} if gameweek is not None else None
        payload = self.http.get_json(
            f"{BASE}/fixtures/",
            params=params,
            ttl=self.settings.cache_ttl("fixtures", 3600),
            force_refresh=force_refresh,
        )
        return [Fixture(row) for row in payload]

    def element_summary(self, element_id: int, force_refresh: bool = False) -> dict:
        """Per-player history: this season's gameweeks, past seasons, fixtures."""
        return self.http.get_json(
            f"{BASE}/element-summary/{element_id}/",
            ttl=self.settings.cache_ttl("element_summary", 21600),
            force_refresh=force_refresh,
        )

    def live(self, gameweek: int, force_refresh: bool = False) -> dict:
        """Per-player stats for a gameweek, updating live during matches."""
        return self.http.get_json(
            f"{BASE}/event/{gameweek}/live/",
            ttl=self.settings.cache_ttl("live", 60),
            force_refresh=force_refresh,
        )

    def event_status(self) -> dict:
        """Whether bonus points and league tables have been processed yet."""
        return self.http.get_json(f"{BASE}/event-status/", ttl=300)

    def entry(self, entry_id: int, force_refresh: bool = False) -> dict:
        return self.http.get_json(
            f"{BASE}/entry/{entry_id}/",
            ttl=self.settings.cache_ttl("entry", 900),
            force_refresh=force_refresh,
        )

    def entry_history(self, entry_id: int, force_refresh: bool = False) -> dict:
        """Season-by-season and gameweek-by-gameweek history, including chips."""
        return self.http.get_json(
            f"{BASE}/entry/{entry_id}/history/",
            ttl=self.settings.cache_ttl("entry", 900),
            force_refresh=force_refresh,
        )

    def entry_picks(self, entry_id: int, gameweek: int, force_refresh: bool = False) -> dict:
        """A manager's picks for a gameweek. Public only after the deadline."""
        return self.http.get_json(
            f"{BASE}/entry/{entry_id}/event/{gameweek}/picks/",
            ttl=self.settings.cache_ttl("entry", 900),
            force_refresh=force_refresh,
        )

    def entry_transfers(self, entry_id: int) -> list[dict]:
        return self.http.get_json(
            f"{BASE}/entry/{entry_id}/transfers/", ttl=self.settings.cache_ttl("entry", 900)
        )

    def classic_league(self, league_id: int, page: int = 1) -> dict:
        return self.http.get_json(
            f"{BASE}/leagues-classic/{league_id}/standings/",
            params={"page_standings": page},
            ttl=1800,
        )

    # -- authenticated -------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        cookie = self.settings.cookie
        if not cookie:
            raise AuthRequired(
                "No FPL session cookie found. Set FPL_COOKIE, or paste the cookie "
                "into config/cookie.txt. See docs/AUTH.md for how to copy it."
            )
        # Accept either a bare token or a full Cookie header pasted from devtools.
        value = cookie if "=" in cookie else f"pl_profile={cookie}"
        return {
            "Cookie": value,
            "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://fantasy.premierleague.com/my-team",
        }

    def my_team(self, entry_id: int, force_refresh: bool = True) -> dict:
        """Your live squad *before* the deadline: picks, bank, free transfers.

        Requires a session cookie. FPL puts a bot-protection layer in front of
        the login form, so the supported path is copying the cookie out of a
        browser you have already logged into rather than posting credentials.
        """
        return self.http.get_json(
            f"{BASE}/my-team/{entry_id}/",
            ttl=60,
            headers=self._auth_headers(),
            force_refresh=force_refresh,
        )

    @property
    def has_credentials(self) -> bool:
        return bool(self.settings.cookie)

    # -- composed views ------------------------------------------------------

    def squad(self, entry_id: int, gameweek: int | None = None) -> ManagerSquad:
        """Best available view of a manager's squad.

        Prefers the authenticated pre-deadline view, which is the only one that
        knows your bank, your free transfers and your true selling prices. Falls
        back to the last published gameweek's picks, and says which it used via
        :attr:`ManagerSquad.source`.
        """
        if self.has_credentials:
            try:
                return self._squad_from_my_team(entry_id)
            except (AuthRequired, HttpError) as exc:
                log.warning("authenticated squad fetch failed (%s); using public data", exc)

        return self._squad_from_public(entry_id, gameweek)

    def _squad_from_my_team(self, entry_id: int) -> ManagerSquad:
        payload = self.my_team(entry_id)
        transfers = payload.get("transfers", {}) or {}
        chips = payload.get("chips", []) or []
        used = [c.get("name") for c in chips if c.get("status_for_entry") == "played"]
        active = next(
            (c.get("name") for c in chips if c.get("status_for_entry") == "active"), None
        )
        return ManagerSquad(
            entry_id=entry_id,
            gameweek=int(transfers.get("event", 0) or 0),
            picks=[Pick.from_api(row) for row in payload.get("picks", [])],
            bank=float(transfers.get("bank", 0) or 0) / 10.0,
            squad_value=float(transfers.get("value", 0) or 0) / 10.0,
            free_transfers=int(transfers.get("limit") or transfers.get("made", 0) or 1),
            chips_used=[c for c in used if c],
            chip_active=active,
            source="authenticated",
        )

    def _squad_from_public(self, entry_id: int, gameweek: int | None) -> ManagerSquad:
        boot = self.bootstrap()
        if gameweek is None:
            finished = boot.finished_gameweeks()
            if not finished:
                raise HttpError(
                    "No gameweek has finished yet, so your squad is not public. "
                    "Add an FPL session cookie to read it before the deadline, or "
                    "enter your squad manually with `fplbrain squad --from-file`."
                )
            gameweek = max(finished)

        picks_payload = self.entry_picks(entry_id, gameweek)
        entry_payload = self.entry(entry_id)
        history = self.entry_history(entry_id)

        chips_used = [c.get("name") for c in history.get("chips", []) if c.get("name")]
        current = history.get("current", []) or []

        entry_history = picks_payload.get("entry_history", {}) or {}
        bank = float(entry_history.get("bank", 0) or 0) / 10.0
        value = float(entry_history.get("value", 0) or 0) / 10.0

        return ManagerSquad(
            entry_id=entry_id,
            gameweek=gameweek,
            picks=[Pick.from_api(row) for row in picks_payload.get("picks", [])],
            bank=bank,
            squad_value=value,
            free_transfers=self._infer_free_transfers(current),
            chips_used=chips_used,
            chip_active=picks_payload.get("active_chip"),
            source="public",
            total_points=int(entry_payload.get("summary_overall_points", 0) or 0),
            overall_rank=entry_payload.get("summary_overall_rank"),
        )

    @staticmethod
    def _infer_free_transfers(current: list[dict], max_banked: int = 5) -> int:
        """Reconstruct free transfers from gameweek history.

        The public API never states how many free transfers you are carrying, but
        it does record transfers made per gameweek. Starting from one free
        transfer, each gameweek adds one and spends what was used, capped at the
        banking limit. Wildcards and free hits do not consume free transfers.
        """
        free = 1
        for row in current:
            made = int(row.get("event_transfers", 0) or 0)
            cost = int(row.get("event_transfers_cost", 0) or 0)
            if cost == 0 and made > free:
                # More transfers than we thought were free and yet no hit: the
                # gameweek must have been a wildcard or free hit.
                free = min(max_banked, free + 1)
                continue
            free = min(max_banked, max(0, free - made) + 1)
        return max(1, min(max_banked, free))


def element_history(summary: dict) -> list[dict]:
    """Rows for the current season out of an ``element-summary`` payload."""
    return list(summary.get("history", []) or [])


def element_fixtures(summary: dict) -> list[dict]:
    return list(summary.get("fixtures", []) or [])


def live_stats(live_payload: dict) -> dict[int, dict]:
    """Map element id to its stat block from an ``event/{gw}/live`` payload."""
    out: dict[int, dict] = {}
    for row in live_payload.get("elements", []) or []:
        try:
            out[int(row["id"])] = row.get("stats", {}) or {}
        except (KeyError, TypeError, ValueError):
            continue
    return out


def iter_player_gameweeks(client: FplClient, element_ids: Iterable[int]) -> Iterable[dict]:
    """Yield every per-gameweek history row for the given players.

    One HTTP call per player, so this is cached aggressively and meant to be run
    once after each gameweek rather than on every command.
    """
    for element_id in element_ids:
        try:
            summary = client.element_summary(element_id)
        except HttpError as exc:  # pragma: no cover - network dependent
            log.warning("could not fetch history for element %s: %s", element_id, exc)
            continue
        for row in element_history(summary):
            row = dict(row)
            row.setdefault("element", element_id)
            yield row


def summarise_availability(player: Player) -> str:
    """A short human explanation of why a player may not play."""
    if player.status == "a" and not player.news:
        return ""
    label = {
        "d": "doubt",
        "i": "injured",
        "s": "suspended",
        "u": "unavailable",
        "n": "not in squad",
    }.get(player.status, player.status)
    chance = player.chance_of_playing_next_round
    parts: list[Any] = [label]
    if chance is not None:
        parts.append(f"{chance:.0f}% chance")
    if player.news:
        parts.append(player.news)
    return " — ".join(str(p) for p in parts)
