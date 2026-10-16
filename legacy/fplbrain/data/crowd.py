"""What everyone else is doing.

Overall ownership is published by the API and is already used throughout the
engine, but it is a blunt instrument: it averages ten million managers, most of
whom set a squad in August and barely touch it. Two sharper signals are
available, and both come from the same public endpoint — a manager's picks
become readable once a deadline has passed.

**Elite ownership** samples the top of the overall league. When a player is on
8% of all teams but 40% of the top thousand, that gap is information: the people
who have been right so far are positioned differently from the crowd.

**A watchlist** tracks specific managers by entry id — content creators, a rival
in your mini-league, anyone whose thinking you want to see. There is no API for
"what are the creators picking", but creators publish their team ids, and a team
id is all this needs.

Both produce the same :class:`CrowdSnapshot`, so the rest of the engine does not
care which one it is looking at.

One hard limitation, stated up front: **picks are only public after a deadline
has passed.** Before the first deadline of a season there is nothing to read at
all, and mid-season you are always looking at the previous gameweek's squads,
not the one being set right now.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..config import Settings, get_settings, load_yaml_with_local
from .fpl import FplClient

log = logging.getLogger(__name__)

# The "Overall" classic league contains every manager, ranked. Its id has been
# stable for the life of the game.
OVERALL_LEAGUE_ID = 314

# The standings endpoint pages fifty managers at a time.
PAGE_SIZE = 50


@dataclass
class ManagerPicks:
    """One manager's squad for one gameweek."""

    entry_id: int
    gameweek: int
    elements: list[int]
    starters: list[int] = field(default_factory=list)
    captain: int | None = None
    vice_captain: int | None = None
    chip: str | None = None
    label: str = ""
    rank: int | None = None

    @property
    def name(self) -> str:
        return self.label or f"entry {self.entry_id}"


@dataclass
class CrowdSnapshot:
    """Ownership and captaincy across a sample of managers."""

    gameweek: int
    managers: list[ManagerPicks] = field(default_factory=list)
    source: str = "elite"
    requested: int = 0
    note: str = ""

    def __len__(self) -> int:
        return len(self.managers)

    @property
    def available(self) -> bool:
        return bool(self.managers)

    # -- aggregates ----------------------------------------------------------

    def ownership(self) -> dict[int, float]:
        """Percent of the sample owning each player."""
        if not self.managers:
            return {}
        counts = Counter(e for m in self.managers for e in set(m.elements))
        return {e: 100.0 * n / len(self.managers) for e, n in counts.items()}

    def start_share(self) -> dict[int, float]:
        """Percent starting each player, which is not the same as owning him."""
        if not self.managers:
            return {}
        counts = Counter(e for m in self.managers for e in set(m.starters))
        return {e: 100.0 * n / len(self.managers) for e, n in counts.items()}

    def captaincy(self) -> dict[int, float]:
        if not self.managers:
            return {}
        counts = Counter(m.captain for m in self.managers if m.captain is not None)
        return {e: 100.0 * n / len(self.managers) for e, n in counts.items()}

    def effective_ownership(self) -> dict[int, float]:
        """Ownership plus captaincy — captains count twice, so they count twice."""
        owned = self.ownership()
        captained = self.captaincy()
        return {e: share + captained.get(e, 0.0) for e, share in owned.items()}

    def chip_usage(self) -> dict[str, float]:
        """Percent of the sample that played each chip this gameweek."""
        if not self.managers:
            return {}
        counts = Counter(m.chip for m in self.managers if m.chip)
        return {chip: 100.0 * n / len(self.managers) for chip, n in counts.items()}

    def margin_of_error(self, confidence_z: float = 1.96) -> float:
        """Worst-case 95% margin on an ownership estimate, in percentage points.

        A sample is not a census. Reporting "43% of the top 1000 own him" from a
        sample of 100 without saying it is ±10 points would be false precision,
        and the whole value of elite ownership is in reading small gaps.
        """
        n = len(self.managers)
        if n <= 1:
            return 100.0
        # p = 0.5 maximises the variance, so this bounds every estimate.
        return 100.0 * confidence_z * math.sqrt(0.25 / n)


@dataclass
class CrowdComparison:
    """One player, seen from the crowd's point of view and from yours."""

    element: int
    name: str
    position: str
    price: float
    overall_ownership: float
    crowd_ownership: float
    crowd_captaincy: float
    owned_by_you: bool
    expected_points: float = 0.0

    @property
    def edge(self) -> float:
        """How much more the sampled crowd backs him than the field does."""
        return self.crowd_ownership - self.overall_ownership

    @property
    def verdict(self) -> str:
        if self.owned_by_you:
            return "held"
        if self.edge >= 10:
            return "crowd is ahead of the field"
        if self.crowd_ownership >= 40:
            return "template"
        return ""


class CrowdClient:
    """Reads squads for a sample of managers."""

    def __init__(self, client: FplClient, settings: Settings | None = None):
        self.client = client
        self.settings = settings or get_settings()

    # -- sources -------------------------------------------------------------

    def top_entry_ids(self, sample_size: int = 100) -> list[tuple[int, int, str]]:
        """(entry id, rank, name) for the best-ranked managers.

        Each request returns fifty, so a sample of a few hundred is a handful of
        calls. Going deeper costs one request per fifty and buys accuracy that
        rarely changes a decision — see :meth:`CrowdSnapshot.margin_of_error`.
        """
        wanted = max(1, sample_size)
        out: list[tuple[int, int, str]] = []
        page = 1
        while len(out) < wanted:
            try:
                payload = self.client.classic_league(OVERALL_LEAGUE_ID, page=page)
            except Exception as exc:  # noqa: BLE001 - report the gap, do not crash
                log.warning("could not read overall league page %s: %s", page, exc)
                break
            standings = (payload.get("standings") or {}).get("results") or []
            if not standings:
                break
            for row in standings:
                entry = row.get("entry")
                if entry is None:
                    continue
                out.append(
                    (int(entry), int(row.get("rank", 0) or 0), str(row.get("entry_name", "")))
                )
                if len(out) >= wanted:
                    break
            if not (payload.get("standings") or {}).get("has_next"):
                break
            page += 1
        return out

    def watchlist_entries(self) -> list[tuple[int, str]]:
        """(entry id, label) pairs from ``config/watchlist.yaml``."""
        config = load_yaml_with_local("watchlist")
        entries: list[tuple[int, str]] = []
        for row in config.get("managers", []) or []:
            if not isinstance(row, dict):
                continue
            entry = row.get("entry_id")
            if not entry:
                continue
            entries.append((int(entry), str(row.get("name", "")) or f"entry {entry}"))
        return entries

    # -- fetching ------------------------------------------------------------

    def picks_for(
        self, entries: Sequence[tuple[int, str, int | None]], gameweek: int
    ) -> list[ManagerPicks]:
        """Fetch squads, skipping any manager whose picks are not public."""
        out: list[ManagerPicks] = []
        for entry_id, label, rank in entries:
            try:
                payload = self.client.entry_picks(entry_id, gameweek)
            except Exception as exc:  # noqa: BLE001 - one manager must not stop the sample
                log.debug("picks unavailable for entry %s: %s", entry_id, exc)
                continue
            picks = payload.get("picks") or []
            if not picks:
                continue
            elements = [int(p["element"]) for p in picks if p.get("element")]
            starters = [
                int(p["element"])
                for p in picks
                if p.get("element") and int(p.get("position", 99)) <= 11
            ]
            captain = next(
                (int(p["element"]) for p in picks if p.get("is_captain")), None
            )
            vice = next(
                (int(p["element"]) for p in picks if p.get("is_vice_captain")), None
            )
            out.append(
                ManagerPicks(
                    entry_id=entry_id,
                    gameweek=gameweek,
                    elements=elements,
                    starters=starters,
                    captain=captain,
                    vice_captain=vice,
                    chip=payload.get("active_chip"),
                    label=label,
                    rank=rank,
                )
            )
        return out

    def entries_in_band(self, top: int, sample_size: int) -> list[tuple[int, str, int | None]]:
        """Sample managers spread evenly across ranks 1..top.

        Taking the first N would measure the top N, not the band. To estimate
        what the top 10k own you have to sample *across* the 10k — so this walks
        evenly spaced pages and takes a couple of managers from each.
        """
        if top <= sample_size:
            return [(e, name, rank) for e, rank, name in self.top_entry_ids(top)]

        total_pages = max(1, math.ceil(top / PAGE_SIZE))
        pages_to_read = min(total_pages, max(1, sample_size // 2))
        step = max(1, total_pages // pages_to_read)
        per_page = max(1, math.ceil(sample_size / pages_to_read))

        out: list[tuple[int, str, int | None]] = []
        for index in range(pages_to_read):
            page = 1 + index * step
            try:
                payload = self.client.classic_league(OVERALL_LEAGUE_ID, page=page)
            except Exception as exc:  # noqa: BLE001 - a gap is not a crash
                log.warning("could not read league page %s: %s", page, exc)
                continue
            rows = (payload.get("standings") or {}).get("results") or []
            for row in rows[:per_page]:
                entry = row.get("entry")
                if entry is None:
                    continue
                out.append(
                    (int(entry), str(row.get("entry_name", "")), int(row.get("rank", 0) or 0))
                )
            if len(out) >= sample_size:
                break
        return out[:sample_size]

    def entries_in_league(self, league_id: int, limit: int) -> list[tuple[int, str, int | None]]:
        """Every manager in a classic league, up to ``limit``.

        This is how a group of experts or creators is usually published: one
        mini-league everyone is in, rather than a list of ids.
        """
        out: list[tuple[int, str, int | None]] = []
        page = 1
        while len(out) < limit:
            try:
                payload = self.client.classic_league(league_id, page=page)
            except Exception as exc:  # noqa: BLE001
                log.warning("could not read league %s page %s: %s", league_id, page, exc)
                break
            standings = payload.get("standings") or {}
            rows = standings.get("results") or []
            if not rows:
                break
            for row in rows:
                entry = row.get("entry")
                if entry is None:
                    continue
                label = str(row.get("player_name") or row.get("entry_name") or f"entry {entry}")
                out.append((int(entry), label, int(row.get("rank", 0) or 0)))
                if len(out) >= limit:
                    break
            if not standings.get("has_next"):
                break
            page += 1
        return out

    def cohort(
        self, spec: "CohortSpec", gameweek: int, sample_size: int = 100, progress=None
    ) -> CrowdSnapshot:
        """Sample one named cohort."""
        if spec.kind == "managers":
            entries = [(e, label, None) for e, label in spec.managers]
        elif spec.kind == "league" and spec.league_id:
            entries = self.entries_in_league(spec.league_id, sample_size)
        else:
            entries = self.entries_in_band(spec.top, sample_size)

        if not entries:
            return CrowdSnapshot(
                gameweek=gameweek,
                source=spec.name,
                requested=sample_size,
                note=f"no managers found for cohort '{spec.name}'",
            )

        managers = self._fetch_with_progress(entries, gameweek, progress)
        note = ""
        if not managers:
            note = (
                f"No picks are public for GW{gameweek}. Squads become readable only "
                "after a deadline has passed."
            )
        return CrowdSnapshot(
            gameweek=gameweek,
            managers=managers,
            source=spec.name,
            requested=len(entries),
            note=note,
        )

    def elite(
        self, gameweek: int, sample_size: int = 100, progress=None
    ) -> CrowdSnapshot:
        """Ownership among the top-ranked managers."""
        top = self.top_entry_ids(sample_size)
        if not top:
            return CrowdSnapshot(
                gameweek=gameweek,
                source="elite",
                requested=sample_size,
                note="could not read the overall league standings",
            )
        entries = [(entry, name, rank) for entry, rank, name in top]
        managers = self._fetch_with_progress(entries, gameweek, progress)
        note = ""
        if not managers:
            note = (
                f"No picks are public for GW{gameweek}. Squads become readable only "
                "after a deadline has passed."
            )
        return CrowdSnapshot(
            gameweek=gameweek,
            managers=managers,
            source="elite",
            requested=sample_size,
            note=note,
        )

    def watchlist(self, gameweek: int, progress=None) -> CrowdSnapshot:
        """Squads of the specific managers you are tracking."""
        entries = self.watchlist_entries()
        if not entries:
            return CrowdSnapshot(
                gameweek=gameweek,
                source="watchlist",
                note=(
                    "config/watchlist.yaml has no managers in it. Add entry ids for "
                    "the people you want to follow — see the comments in that file."
                ),
            )
        managers = self._fetch_with_progress(
            [(entry, label, None) for entry, label in entries], gameweek, progress
        )
        note = ""
        if not managers:
            note = f"None of the tracked managers have public picks for GW{gameweek}."
        return CrowdSnapshot(
            gameweek=gameweek,
            managers=managers,
            source="watchlist",
            requested=len(entries),
            note=note,
        )

    def _fetch_with_progress(self, entries, gameweek: int, progress) -> list[ManagerPicks]:
        managers: list[ManagerPicks] = []
        total = len(entries)
        for index, entry in enumerate(entries, start=1):
            managers.extend(self.picks_for([entry], gameweek))
            if progress is not None and (index % 5 == 0 or index == total):
                progress(index, total, "manager squads")
        return managers


def compare_to_crowd(
    snapshot: CrowdSnapshot,
    projections,
    your_squad: Iterable[int] = (),
    *,
    top_n: int = 25,
) -> list[CrowdComparison]:
    """Rank players by how strongly the sampled crowd backs them."""
    owned = set(your_squad)
    crowd_ownership = snapshot.ownership()
    crowd_captaincy = snapshot.captaincy()

    rows: list[CrowdComparison] = []
    for element, share in crowd_ownership.items():
        player = projections.players.get(element)
        if player is None:
            continue
        rows.append(
            CrowdComparison(
                element=element,
                name=projections.name(element),
                position=projections.position(element),
                price=projections.price(element),
                overall_ownership=player.selected_by_percent,
                crowd_ownership=share,
                crowd_captaincy=crowd_captaincy.get(element, 0.0),
                owned_by_you=element in owned,
                expected_points=projections.horizon_points(element),
            )
        )
    rows.sort(key=lambda r: -r.crowd_ownership)
    return rows[:top_n]


def template_squad(snapshot: CrowdSnapshot, size: int = 15) -> list[int]:
    """The most-owned squad in the sample — what "the template" actually is."""
    ownership = snapshot.ownership()
    return [e for e, _ in sorted(ownership.items(), key=lambda kv: -kv[1])[:size]]


def differentials(
    snapshot: CrowdSnapshot, your_squad: Iterable[int], threshold: float = 15.0
) -> tuple[list[int], list[int]]:
    """(players you own that the crowd does not, players it owns that you do not).

    The first list is where your rank can move away from the field; the second
    is where it can move against you.
    """
    ownership = snapshot.ownership()
    owned = set(your_squad)
    yours_only = [e for e in owned if ownership.get(e, 0.0) < threshold]
    theirs_only = [
        e for e, share in ownership.items() if share >= 100 - threshold and e not in owned
    ]
    return sorted(yours_only), sorted(theirs_only)


# ---------------------------------------------------------------------------
# Cohorts
# ---------------------------------------------------------------------------


@dataclass
class CohortSpec:
    """A named group of managers to sample.

    Three kinds, all resolving to a list of entry ids:

    ``overall``   the top N of the overall league — "what is the top 1k doing"
    ``league``    every manager in a classic league, which is how a group of
                  experts or creators is usually published
    ``managers``  an explicit list of entry ids
    """

    name: str
    kind: str = "overall"
    top: int = 100
    league_id: int | None = None
    managers: list[tuple[int, str]] = field(default_factory=list)

    @classmethod
    def from_config(cls, row: dict) -> "CohortSpec | None":
        if not isinstance(row, dict) or not row.get("name"):
            return None
        kind = str(row.get("type", "overall")).lower()
        managers = []
        for entry in row.get("managers", []) or []:
            if isinstance(entry, dict) and entry.get("entry_id"):
                managers.append(
                    (int(entry["entry_id"]), str(entry.get("name", "")) or "manager")
                )
            elif isinstance(entry, int):
                managers.append((entry, f"entry {entry}"))
        return cls(
            name=str(row["name"]),
            kind=kind,
            top=int(row.get("top", 100) or 100),
            league_id=int(row["league_id"]) if row.get("league_id") else None,
            managers=managers,
        )


def load_cohorts() -> list[CohortSpec]:
    """Cohorts from ``config/watchlist.yaml``, plus the standard rank bands.

    The rank bands are always available because they need no configuration: the
    overall league exists for everyone. Anything else has to be pointed at.
    """
    config = load_yaml_with_local("watchlist")
    specs: list[CohortSpec] = []
    for row in config.get("cohorts", []) or []:
        spec = CohortSpec.from_config(row)
        if spec is not None:
            specs.append(spec)

    # Backwards compatibility with the flat `managers:` list.
    flat = [
        (int(r["entry_id"]), str(r.get("name", "")) or f"entry {r['entry_id']}")
        for r in (config.get("managers") or [])
        if isinstance(r, dict) and r.get("entry_id")
    ]
    if flat:
        specs.append(CohortSpec(name="Watchlist", kind="managers", managers=flat))
    return specs


DEFAULT_RANK_BANDS = [
    CohortSpec(name="Top 100", kind="overall", top=100),
    CohortSpec(name="Top 1k", kind="overall", top=1000),
    CohortSpec(name="Top 10k", kind="overall", top=10000),
]
