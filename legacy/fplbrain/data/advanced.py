"""Advanced per-player metrics from sources other than the FPL API.

The FPL API already publishes expected goals, expected assists, expected goals
conceded and the full defensive counts, and the engine uses all of them. This
module exists so a *second* opinion can be brought in — a different xG model
built on different event data — and blended against the default.

Three providers, deliberately behind one interface so the rest of the engine
does not care which is in use:

``fpl``        the default. No extra fetch: the numbers are already loaded.
``understat``  free, no key, a genuinely independent shot-level xG model.
``opta``       Opta's SDAPI. Requires a licensed outlet key — see below.

**On Opta.** The endpoint shape is public and implemented here in full:

    https://api.performfeeds.com/soccerdata/{feed}/{outletAuthKey}?_rt=b&_fmt=json

What is not public is the ``outletAuthKey``. Opta is a Stats Perform product
sold under a commercial contract; there is no free tier and no self-service
signup. So this client is complete and will work the moment a key is supplied,
and says plainly that it cannot run without one rather than silently returning
nothing. It has never been exercised against the live service, because doing so
requires a licence.

Blending is deliberately conservative. A second xG model disagreeing with the
first is information, but it is not authority: the default weight moves the rate
part of the way toward the external number rather than replacing it.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

from ..config import Settings, get_settings
from ..util.http import HttpClient, HttpError
from .odds import normalise_team

log = logging.getLogger(__name__)

OPTA_BASE = "https://api.performfeeds.com/soccerdata"
UNDERSTAT_LEAGUE_URL = "https://understat.com/league/EPL/{season}"


def normalise_person(name: str) -> str:
    """Reduce a player name to a key that survives crossing data sources.

    Accents, punctuation and word order all differ between providers — Opta's
    "Gabriel Fernando de Jesus" and Understat's "Gabriel Jesus" have to land on
    something comparable, so the key is the alphabetic characters of the last
    token, lowercased.
    """
    cleaned = _strip_accents(name or "").strip()
    if not cleaned:
        return ""
    tokens = [t for t in re.split(r"[\s\-']+", cleaned) if t]
    if not tokens:
        return ""
    return re.sub(r"[^a-z]", "", tokens[-1].lower())


def _strip_accents(text: str) -> str:
    import unicodedata

    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


@dataclass
class AdvancedStats:
    """One player's season totals from an external provider."""

    name: str
    team: str
    minutes: float
    xg: float = 0.0
    xa: float = 0.0
    shots: float = 0.0
    key_passes: float = 0.0
    goals: float = 0.0
    assists: float = 0.0
    source: str = ""

    @property
    def key(self) -> str:
        return normalise_person(self.name)

    @property
    def team_key(self) -> str:
        return normalise_team(self.team)

    @property
    def ninetys(self) -> float:
        return self.minutes / 90.0

    def xg90(self) -> float:
        return self.xg / self.ninetys if self.ninetys > 0 else 0.0

    def xa90(self) -> float:
        return self.xa / self.ninetys if self.ninetys > 0 else 0.0


@dataclass
class StatsSnapshot:
    players: list[AdvancedStats] = field(default_factory=list)
    source: str = ""
    available: bool = True
    reason: str = ""

    def __len__(self) -> int:
        return len(self.players)

    def by_key(self) -> dict[tuple[str, str], AdvancedStats]:
        return {(p.team_key, p.key): p for p in self.players}


class StatsProvider(Protocol):
    """Anything that can supply season-level advanced stats."""

    name: str

    def enabled(self) -> bool: ...

    def fetch(self, season: str) -> StatsSnapshot: ...


# ---------------------------------------------------------------------------
# Understat
# ---------------------------------------------------------------------------


class UnderstatProvider:
    """Understat's shot-level xG model. Free, no key, no account.

    The league page embeds its data as a JSON string inside a script tag, which
    is the only interface it offers. That is fragile by nature, so a parse
    failure is reported as unavailable rather than raised.
    """

    name = "understat"

    def __init__(self, http: HttpClient, settings: Settings | None = None):
        self.http = http
        self.settings = settings or get_settings()

    def enabled(self) -> bool:
        return True

    def fetch(self, season: str) -> StatsSnapshot:
        # Understat labels a season by its starting year: 2026/27 is "2026".
        year = season.split("-")[0]
        url = UNDERSTAT_LEAGUE_URL.format(season=year)
        try:
            body = self.http.get(url, ttl=self.settings.cache_ttl("understat", 43200))
        except HttpError as exc:
            return StatsSnapshot(
                source=self.name, available=False, reason=f"understat unreachable: {exc}"
            )

        payload = _extract_understat_json(body, "playersData")
        if payload is None:
            return StatsSnapshot(
                source=self.name,
                available=False,
                reason="could not find playersData on the Understat page (layout may have changed)",
            )

        players = []
        for row in payload:
            try:
                players.append(
                    AdvancedStats(
                        name=str(row.get("player_name", "")),
                        team=str(row.get("team_title", "")),
                        minutes=float(row.get("time", 0) or 0),
                        xg=float(row.get("xG", 0) or 0),
                        xa=float(row.get("xA", 0) or 0),
                        shots=float(row.get("shots", 0) or 0),
                        key_passes=float(row.get("key_passes", 0) or 0),
                        goals=float(row.get("goals", 0) or 0),
                        assists=float(row.get("assists", 0) or 0),
                        source=self.name,
                    )
                )
            except (TypeError, ValueError):
                continue
        return StatsSnapshot(players=players, source=self.name)


def _extract_understat_json(body: str, variable: str) -> list[dict] | None:
    """Pull one of Understat's embedded ``JSON.parse('...')`` blobs out of a page."""
    match = re.search(
        variable + r"\s*=\s*JSON\.parse\('(?P<payload>.*?)'\)", body, re.DOTALL
    )
    if not match:
        return None
    raw = match.group("payload")
    # The payload is hex-escaped inside a JavaScript string literal.
    try:
        decoded = raw.encode().decode("unicode_escape")
        return json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        log.debug("could not decode Understat payload: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Opta
# ---------------------------------------------------------------------------


class OptaProvider:
    """Opta SDAPI client.

    Requires ``OPTA_OUTLET_KEY`` and a tournament calendar id. Opta is sold
    under a commercial contract — there is no free tier — so without a key this
    reports why it cannot run rather than failing obscurely.

    The response shape below follows Opta's documented ``seasonstats`` feed,
    where each player carries a list of ``{type, value}`` stat pairs. Because
    the feed cannot be exercised without a licence, parsing is written
    defensively and every field is optional.
    """

    name = "opta"

    def __init__(self, http: HttpClient, settings: Settings | None = None):
        self.http = http
        self.settings = settings or get_settings()

    @property
    def outlet_key(self) -> str | None:
        import os

        return os.environ.get("OPTA_OUTLET_KEY") or None

    @property
    def tournament_calendar(self) -> str | None:
        import os

        return (
            os.environ.get("OPTA_TOURNAMENT_CALENDAR")
            or self.settings.get("advanced.opta_tournament_calendar")
        )

    def enabled(self) -> bool:
        return bool(self.outlet_key)

    def fetch(self, season: str) -> StatsSnapshot:
        if not self.outlet_key:
            return StatsSnapshot(
                source=self.name,
                available=False,
                reason=(
                    "No OPTA_OUTLET_KEY. Opta is a Stats Perform product sold under a "
                    "commercial licence — there is no free tier or self-service signup. "
                    "With a key this client works as-is."
                ),
            )
        if not self.tournament_calendar:
            return StatsSnapshot(
                source=self.name,
                available=False,
                reason=(
                    "No OPTA_TOURNAMENT_CALENDAR. Opta identifies a season by a "
                    "tournament calendar id, which comes from the tournamentcalendar feed."
                ),
            )

        url = f"{OPTA_BASE}/seasonstats/{self.outlet_key}"
        try:
            payload = self.http.get_json(
                url,
                params={"_rt": "b", "_fmt": "json", "tmcl": self.tournament_calendar},
                ttl=self.settings.cache_ttl("understat", 43200),
            )
        except HttpError as exc:
            return StatsSnapshot(
                source=self.name, available=False, reason=f"Opta request failed: {exc}"
            )

        players = _parse_opta_seasonstats(payload)
        if not players:
            return StatsSnapshot(
                source=self.name,
                available=False,
                reason="Opta responded but no player stats were found in the payload",
            )
        return StatsSnapshot(players=players, source=self.name)


# Opta stat names differ from everyone else's; this is the mapping the parser uses.
_OPTA_STAT_MAP = {
    "expectedGoals": "xg",
    "expectedAssists": "xa",
    "totalScoringAtt": "shots",
    "goalAssist": "assists",
    "goals": "goals",
    "minsPlayed": "minutes",
    "bigChanceCreated": "key_passes",
}


def _parse_opta_seasonstats(payload: object) -> list[AdvancedStats]:
    """Read Opta's nested contestant/player/stat structure.

    Written to tolerate shape drift: any level may be missing, and unknown stat
    names are ignored rather than being an error.
    """
    out: list[AdvancedStats] = []
    if not isinstance(payload, dict):
        return out

    contestants = payload.get("contestant") or payload.get("contestants") or []
    if isinstance(contestants, dict):
        contestants = [contestants]

    for contestant in contestants:
        if not isinstance(contestant, dict):
            continue
        team = str(contestant.get("name") or contestant.get("shortName") or "")
        people = contestant.get("player") or contestant.get("players") or []
        if isinstance(people, dict):
            people = [people]

        for person in people:
            if not isinstance(person, dict):
                continue
            values: dict[str, float] = {}
            for stat in person.get("stat") or person.get("stats") or []:
                if not isinstance(stat, dict):
                    continue
                field_name = _OPTA_STAT_MAP.get(str(stat.get("type", "")))
                if field_name is None:
                    continue
                try:
                    values[field_name] = float(stat.get("value", 0) or 0)
                except (TypeError, ValueError):
                    continue
            if not values:
                continue
            out.append(
                AdvancedStats(
                    name=str(person.get("matchName") or person.get("knownName") or ""),
                    team=team,
                    minutes=values.get("minutes", 0.0),
                    xg=values.get("xg", 0.0),
                    xa=values.get("xa", 0.0),
                    shots=values.get("shots", 0.0),
                    key_passes=values.get("key_passes", 0.0),
                    goals=values.get("goals", 0.0),
                    assists=values.get("assists", 0.0),
                    source="opta",
                )
            )
    return out


# ---------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------


def get_provider(name: str, http: HttpClient, settings: Settings | None = None):
    """Build a provider by name, or None for the FPL default."""
    settings = settings or get_settings()
    if name == "understat":
        return UnderstatProvider(http, settings)
    if name == "opta":
        return OptaProvider(http, settings)
    return None


@dataclass
class BlendReport:
    matched: int = 0
    total: int = 0
    source: str = ""
    note: str = ""

    def summary(self) -> str:
        if not self.source:
            return "FPL's own expected-goals data only."
        if not self.total:
            return self.note or f"{self.source}: nothing to blend."
        return (
            f"{self.source}: matched {self.matched} of {self.total} players "
            f"({self.matched / self.total:.0%})."
        )


def blend_rates(
    rates: dict,
    players: dict,
    teams: dict,
    snapshot: StatsSnapshot,
    *,
    weight: float = 0.35,
    min_minutes: float = 270.0,
) -> BlendReport:
    """Move each player's xG and xA rate part of the way toward an external model.

    Mutates ``rates`` in place. Only players with a real sample on *both* sides
    are touched: an external source with 90 minutes of data would otherwise drag
    a well-established rate around for no reason.

    The weight is well below 1 on purpose. A second xG model is a useful check
    on the first, not a replacement for it — they disagree because they are
    built on different definitions, and neither is the truth.
    """
    if not snapshot.available or not snapshot.players:
        return BlendReport(source=snapshot.source, note=snapshot.reason)

    lookup = snapshot.by_key()
    team_key = {team_id: normalise_team(team.name) for team_id, team in teams.items()}

    matched = 0
    eligible = 0
    for element, player_rates in rates.items():
        player = players.get(element)
        if player is None or player.minutes < min_minutes:
            continue
        eligible += 1

        key = (team_key.get(player.team_id, ""), normalise_person(
            player.raw.get("second_name") or player.web_name
        ))
        external = lookup.get(key)
        if external is None or external.minutes < min_minutes:
            continue

        player_rates.xg90 = (1 - weight) * player_rates.xg90 + weight * external.xg90()
        player_rates.xa90 = (1 - weight) * player_rates.xa90 + weight * external.xa90()
        matched += 1

    return BlendReport(matched=matched, total=eligible, source=snapshot.source)
