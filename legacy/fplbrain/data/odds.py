"""Bookmaker odds, converted into the quantities the points model needs.

Betting markets are the sharpest publicly available forecast of a football
match, so the engine prefers them over its own ratings whenever they exist. Two
markets carry almost all the signal:

    1X2 + over/under total goals -> each side's expected goals, and from that
        clean sheet probability and goals-conceded risk
    anytime goalscorer           -> a direct read on an individual's goal
        probability, which is the single best captaincy input there is

Odds come from The Odds API (free tier: 500 requests a month, which covers a
weekly pull comfortably). Without a key the engine falls back to its own team
ratings and marks the projection as model-only rather than market-informed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..config import Settings, get_settings
from ..model.match import devig, solve_lambdas_from_market
from ..util.http import HttpClient, HttpError

log = logging.getLogger(__name__)

ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Bookmakers and the FPL disagree about club names. Normalising to a compact key
# handles almost everything; the alias table covers the rest.
TEAM_ALIASES = {
    "manchesterunited": "manutd",
    "manutd": "manutd",
    "manchesterutd": "manutd",
    "manchestercity": "mancity",
    "mancity": "mancity",
    "tottenhamhotspur": "spurs",
    "tottenham": "spurs",
    "spurs": "spurs",
    "wolverhamptonwanderers": "wolves",
    "wolverhampton": "wolves",
    "wolves": "wolves",
    "brightonandhovealbion": "brighton",
    "brightonhovealbion": "brighton",
    "brighton": "brighton",
    "nottinghamforest": "forest",
    "nottmforest": "forest",
    "nottinghamfor": "forest",
    "forest": "forest",
    "newcastleunited": "newcastle",
    "newcastle": "newcastle",
    "westhamunited": "westham",
    "westham": "westham",
    "leedsunited": "leeds",
    "leeds": "leeds",
    "leicestercity": "leicester",
    "leicester": "leicester",
    "afcbournemouth": "bournemouth",
    "bournemouth": "bournemouth",
    "crystalpalace": "palace",
    "palace": "palace",
    "sheffieldunited": "sheffieldutd",
    "sheffieldutd": "sheffieldutd",
    "ipswichtown": "ipswich",
    "ipswich": "ipswich",
    "lutontown": "luton",
    "norwichcity": "norwich",
    "stokecity": "stoke",
    "hullcity": "hull",
    "cardiffcity": "cardiff",
    "swanseacity": "swansea",
    "birminghamcity": "birmingham",
    "coventrycity": "coventry",
    "westbromwichalbion": "westbrom",
    "westbrom": "westbrom",
    "sunderlandafc": "sunderland",
}


def normalise_team(name: str) -> str:
    """Reduce a club name to a stable key for cross-source matching."""
    key = re.sub(r"[^a-z]", "", (name or "").lower())
    for suffix in ("football club", "fc", "afc"):
        key = key.replace(re.sub(r"[^a-z]", "", suffix), "")
    return TEAM_ALIASES.get(key, key)


@dataclass
class MatchOdds:
    """Devigged market view of a single fixture."""

    home_team: str
    away_team: str
    kickoff: datetime | None
    p_home: float
    p_draw: float
    p_away: float
    p_over: float | None = None
    total_line: float = 2.5
    bookmakers: int = 0
    source: str = "the_odds_api"

    @property
    def home_key(self) -> str:
        return normalise_team(self.home_team)

    @property
    def away_key(self) -> str:
        return normalise_team(self.away_team)

    def expected_goals(self) -> tuple[float, float]:
        """Solve the market back into (home xG, away xG)."""
        return solve_lambdas_from_market(
            self.p_home, self.p_draw, self.p_away, self.p_over, self.total_line
        )


@dataclass
class PlayerGoalOdds:
    """Anytime-goalscorer probability for one player in one fixture."""

    player_name: str
    team_key: str
    probability: float
    bookmakers: int = 0


@dataclass
class OddsSnapshot:
    matches: list[MatchOdds] = field(default_factory=list)
    player_goals: list[PlayerGoalOdds] = field(default_factory=list)
    fetched_at: datetime | None = None
    available: bool = True
    reason: str = ""

    def by_teams(self) -> dict[tuple[str, str], MatchOdds]:
        return {(m.home_key, m.away_key): m for m in self.matches}

    def find(self, home_team_name: str, away_team_name: str) -> MatchOdds | None:
        return self.by_teams().get((normalise_team(home_team_name), normalise_team(away_team_name)))


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("empty")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _consensus_h2h(
    bookmakers: Iterable[dict], method: str
) -> tuple[tuple[float, float, float, int], list[str]]:
    """Median devigged 1X2 probabilities across bookmakers.

    Taking the median of devigged books rather than the best price is
    deliberate: we want the market's central estimate, not the outlier that a
    bettor would shop for.

    Returns ``((p_first, p_draw, p_second, book_count), [first, second])`` where
    the two names let the caller work out which side is home.
    """
    rows: list[dict[str, float]] = []
    for book in bookmakers:
        for market in book.get("markets", []) or []:
            if market.get("key") != "h2h":
                continue
            prices: dict[str, float] = {}
            for outcome in market.get("outcomes", []) or []:
                name = outcome.get("name")
                price = outcome.get("price")
                if name and price:
                    prices[str(name)] = float(price)
            if len(prices) >= 3:
                try:
                    rows.append(devig(prices, method))
                except ValueError:
                    continue
    if not rows:
        raise ValueError("no h2h market")

    # Outcome names are the club names plus "Draw"; align them by position.
    keys = list(rows[0])
    draw_key = next((k for k in keys if k.lower() == "draw"), None)
    others = [k for k in keys if k != draw_key]
    if draw_key is None or len(others) != 2:
        raise ValueError("unexpected h2h outcome shape")
    consensus = (
        _median([r.get(others[0], 0.0) for r in rows]),
        _median([r.get(draw_key, 0.0) for r in rows]),
        _median([r.get(others[1], 0.0) for r in rows]),
        len(rows),
    )
    return consensus, others


def _consensus_totals(bookmakers: Iterable[dict], method: str) -> tuple[float, float] | None:
    """Median P(over) and the line it refers to, preferring the 2.5 line."""
    by_line: dict[float, list[float]] = {}
    for book in bookmakers:
        for market in book.get("markets", []) or []:
            if market.get("key") != "totals":
                continue
            prices: dict[str, float] = {}
            line = None
            for outcome in market.get("outcomes", []) or []:
                name = str(outcome.get("name", "")).lower()
                price = outcome.get("price")
                point = outcome.get("point")
                if name in ("over", "under") and price:
                    prices[name] = float(price)
                    if point is not None:
                        line = float(point)
            if len(prices) == 2 and line is not None:
                try:
                    fair = devig(prices, method)
                except ValueError:
                    continue
                by_line.setdefault(line, []).append(fair["over"])
    if not by_line:
        return None
    line = 2.5 if 2.5 in by_line else max(by_line, key=lambda k: len(by_line[k]))
    return _median(by_line[line]), line


class OddsClient:
    """Fetches and devigs odds, with a manual CSV fallback."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        http: HttpClient | None = None,
        offline: bool = False,
    ):
        self.settings = settings or get_settings()
        self.http = http or HttpClient(self.settings.cache_dir / "odds", offline=offline)

    @property
    def enabled(self) -> bool:
        return bool(self.settings.odds_api_key)

    def fetch(self, force_refresh: bool = False) -> OddsSnapshot:
        """Market odds for upcoming fixtures, or an empty snapshot with a reason."""
        manual = self._manual_path()
        if manual.exists():
            log.info("using manual odds from %s", manual)
            return self._load_manual(manual)

        if not self.enabled:
            return OddsSnapshot(
                available=False,
                reason=(
                    "No ODDS_API_KEY set. Projections use the engine's own team "
                    "ratings; add a free key from the-odds-api.com to sharpen them."
                ),
            )

        cfg = self.settings.odds
        url = f"{ODDS_API_BASE}/sports/{cfg.get('sport_key', 'soccer_epl')}/odds/"
        params = {
            "apiKey": self.settings.odds_api_key,
            "regions": cfg.get("regions", "uk"),
            "markets": ",".join(cfg.get("markets", ["h2h", "totals"])),
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        try:
            payload = self.http.get_json(
                url,
                params=params,
                ttl=self.settings.cache_ttl("odds", 1800),
                force_refresh=force_refresh,
            )
        except HttpError as exc:
            return OddsSnapshot(available=False, reason=f"odds fetch failed: {exc}")

        method = str(cfg.get("devig", "shin"))
        matches: list[MatchOdds] = []
        for event in payload or []:
            books = event.get("bookmakers", []) or []
            if not books:
                continue
            try:
                (p_home, p_draw, p_away, count), names = _consensus_h2h(books, method)
            except ValueError:
                continue

            home_team = str(event.get("home_team", ""))
            away_team = str(event.get("away_team", ""))
            # `_consensus_h2h` returns outcomes in market order, which is usually
            # but not always home-first. Re-align against the event's own labels.
            if names and normalise_team(names[0]) != normalise_team(home_team):
                p_home, p_away = p_away, p_home

            totals = _consensus_totals(books, method)
            matches.append(
                MatchOdds(
                    home_team=home_team,
                    away_team=away_team,
                    kickoff=_parse_iso(event.get("commence_time")),
                    p_home=p_home,
                    p_draw=p_draw,
                    p_away=p_away,
                    p_over=totals[0] if totals else None,
                    total_line=totals[1] if totals else 2.5,
                    bookmakers=count,
                )
            )

        return OddsSnapshot(matches=matches, fetched_at=datetime.now(timezone.utc))

    def fetch_goalscorer_odds(self, event_ids: list[str]) -> list[PlayerGoalOdds]:
        """Anytime-goalscorer probabilities, when the API plan exposes them.

        This market is not on every plan, so a failure here is logged and
        swallowed: the engine simply falls back to its own goal model.
        """
        if not self.enabled:
            return []
        cfg = self.settings.odds
        out: list[PlayerGoalOdds] = []
        for event_id in event_ids:
            url = (
                f"{ODDS_API_BASE}/sports/{cfg.get('sport_key', 'soccer_epl')}"
                f"/events/{event_id}/odds/"
            )
            params = {
                "apiKey": self.settings.odds_api_key,
                "regions": cfg.get("regions", "uk"),
                "markets": "player_goal_scorer_anytime",
                "oddsFormat": "decimal",
            }
            try:
                payload = self.http.get_json(
                    url, params=params, ttl=self.settings.cache_ttl("odds", 1800)
                )
            except HttpError as exc:
                log.info("goalscorer odds unavailable for %s: %s", event_id, exc)
                continue
            out.extend(_parse_goalscorer_payload(payload))
        return out

    # -- manual fallback -----------------------------------------------------

    def _manual_path(self) -> Path:
        return self.settings.repo_root / "data" / "manual_odds.csv"

    def _load_manual(self, path: Path) -> OddsSnapshot:
        """Read odds typed in by hand.

        Columns: home,away,odds_home,odds_draw,odds_away[,odds_over,odds_under,line]
        Decimal odds, with the bookmaker margin left in — it is removed here.
        """
        import csv

        matches: list[MatchOdds] = []
        method = str(self.settings.odds.get("devig", "shin"))
        with path.open() as fh:
            for row in csv.DictReader(fh):
                try:
                    fair = devig(
                        {
                            "home": float(row["odds_home"]),
                            "draw": float(row["odds_draw"]),
                            "away": float(row["odds_away"]),
                        },
                        method,
                    )
                except (KeyError, ValueError):
                    continue
                p_over = None
                line = 2.5
                if row.get("odds_over") and row.get("odds_under"):
                    try:
                        totals = devig(
                            {"over": float(row["odds_over"]), "under": float(row["odds_under"])},
                            method,
                        )
                        p_over = totals["over"]
                        line = float(row.get("line") or 2.5)
                    except ValueError:
                        pass
                matches.append(
                    MatchOdds(
                        home_team=row["home"],
                        away_team=row["away"],
                        kickoff=None,
                        p_home=fair["home"],
                        p_draw=fair["draw"],
                        p_away=fair["away"],
                        p_over=p_over,
                        total_line=line,
                        source="manual",
                    )
                )
        return OddsSnapshot(
            matches=matches, fetched_at=datetime.now(timezone.utc), reason="manual odds file"
        )


def _parse_goalscorer_payload(payload: Any) -> list[PlayerGoalOdds]:
    out: list[PlayerGoalOdds] = []
    by_player: dict[str, list[float]] = {}
    for book in (payload or {}).get("bookmakers", []) or []:
        for market in book.get("markets", []) or []:
            if market.get("key") != "player_goal_scorer_anytime":
                continue
            for outcome in market.get("outcomes", []) or []:
                name = outcome.get("description") or outcome.get("name")
                price = outcome.get("price")
                if name and price and float(price) > 1.0:
                    by_player.setdefault(str(name), []).append(1.0 / float(price))
    for name, probs in by_player.items():
        # Anytime-scorer books carry a large overround; scaling by a typical
        # single-market margin is closer to fair than using the raw implied
        # probability, without pretending to a precise devig on one leg.
        out.append(
            PlayerGoalOdds(
                player_name=name,
                team_key="",
                probability=min(0.95, _median(probs) / 1.12),
                bookmakers=len(probs),
            )
        )
    return out


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
