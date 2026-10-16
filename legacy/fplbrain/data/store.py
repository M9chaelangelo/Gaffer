"""Local warehouse of everything the engine has seen and predicted.

Two purposes. First, a record of actual results per player per gameweek, which
is the training set for the learned model. Second, a record of what the engine
*predicted* before each deadline, which is the only way to know whether it is
any good — a projection compared against results it was fitted on tells you
nothing.

SQLite is deliberate: the whole history of a season is a few megabytes, and a
single file that can be copied, inspected in any client and thrown away is worth
more here than a database server.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS player_gameweek (
    season       TEXT NOT NULL,
    gameweek     INTEGER NOT NULL,
    element      INTEGER NOT NULL,
    fixture      INTEGER,
    opponent     INTEGER,
    was_home     INTEGER,
    minutes      INTEGER,
    total_points INTEGER,
    goals        INTEGER,
    assists      INTEGER,
    clean_sheet  INTEGER,
    goals_conceded INTEGER,
    saves        INTEGER,
    bonus        INTEGER,
    bps          INTEGER,
    yellow_cards INTEGER,
    red_cards    INTEGER,
    own_goals    INTEGER,
    penalties_saved INTEGER,
    penalties_missed INTEGER,
    expected_goals REAL,
    expected_assists REAL,
    expected_goals_conceded REAL,
    clearances_blocks_interceptions INTEGER,
    recoveries   INTEGER,
    tackles      INTEGER,
    defensive_contribution INTEGER,
    starts       INTEGER,
    value        INTEGER,
    raw          TEXT,
    PRIMARY KEY (season, gameweek, element, fixture)
);

CREATE TABLE IF NOT EXISTS projection_snapshot (
    season       TEXT NOT NULL,
    made_at      TEXT NOT NULL,
    for_gameweek INTEGER NOT NULL,
    element      INTEGER NOT NULL,
    expected_points REAL,
    p_start      REAL,
    p_60         REAL,
    p_haul       REAL,
    p_blank      REAL,
    clean_sheet_probability REAL,
    defcon_probability REAL,
    n_fixtures   INTEGER,
    price        REAL,
    breakdown    TEXT,
    PRIMARY KEY (season, for_gameweek, element, made_at)
);

CREATE TABLE IF NOT EXISTS decision_log (
    season       TEXT NOT NULL,
    made_at      TEXT NOT NULL,
    gameweek     INTEGER NOT NULL,
    kind         TEXT NOT NULL,
    payload      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pgw_element ON player_gameweek (element, season);
CREATE INDEX IF NOT EXISTS idx_pgw_gameweek ON player_gameweek (season, gameweek);
CREATE INDEX IF NOT EXISTS idx_snapshot_gw ON projection_snapshot (season, for_gameweek);
"""

_HISTORY_FIELDS = (
    "minutes",
    "total_points",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "bonus",
    "bps",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "expected_goals",
    "expected_assists",
    "expected_goals_conceded",
    "clearances_blocks_interceptions",
    "recoveries",
    "tackles",
    "defensive_contribution",
    "starts",
    "value",
)


@dataclass
class Store:
    """A thin wrapper over the SQLite file."""

    path: Path
    season: str = "2026-27"

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- writes --------------------------------------------------------------

    def record_history(self, rows: Iterable[dict], season: str | None = None) -> int:
        """Store per-player gameweek rows from ``element-summary`` payloads."""
        season = season or self.season
        prepared = []
        for row in rows:
            try:
                element = int(row["element"])
                gameweek = int(row.get("round", 0) or 0)
            except (KeyError, TypeError, ValueError):
                continue
            if gameweek <= 0:
                continue
            values = {key: row.get(key) for key in _HISTORY_FIELDS}
            prepared.append(
                (
                    season,
                    gameweek,
                    element,
                    row.get("fixture"),
                    row.get("opponent_team"),
                    1 if row.get("was_home") else 0,
                    _int(values["minutes"]),
                    _int(values["total_points"]),
                    _int(values["goals_scored"]),
                    _int(values["assists"]),
                    _int(values["clean_sheets"]),
                    _int(values["goals_conceded"]),
                    _int(values["saves"]),
                    _int(values["bonus"]),
                    _int(values["bps"]),
                    _int(values["yellow_cards"]),
                    _int(values["red_cards"]),
                    _int(values["own_goals"]),
                    _int(values["penalties_saved"]),
                    _int(values["penalties_missed"]),
                    _float(values["expected_goals"]),
                    _float(values["expected_assists"]),
                    _float(values["expected_goals_conceded"]),
                    _int(values["clearances_blocks_interceptions"]),
                    _int(values["recoveries"]),
                    _int(values["tackles"]),
                    _int(values["defensive_contribution"]),
                    _int(values["starts"]),
                    _int(values["value"]),
                    json.dumps(row),
                )
            )
        if not prepared:
            return 0
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO player_gameweek VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                prepared,
            )
        return len(prepared)

    def record_projections(self, projection_set, gameweeks: Sequence[int] | None = None) -> int:
        """Snapshot the current projections so they can be scored later."""
        made_at = datetime.now(timezone.utc).isoformat()
        gameweeks = list(gameweeks or projection_set.gameweeks)
        rows = []
        for gameweek in gameweeks:
            for element in projection_set.players:
                projection = projection_set.get(element, gameweek)
                if projection.n_fixtures == 0 and projection.expected_points == 0:
                    continue
                minutes = projection_set.minutes.get(element)
                rows.append(
                    (
                        self.season,
                        made_at,
                        gameweek,
                        element,
                        projection.expected_points,
                        minutes.p_start if minutes else None,
                        minutes.p_60 if minutes else None,
                        projection.p_haul,
                        projection.p_blank,
                        max(
                            (f.clean_sheet_probability for f in projection.fixtures), default=0.0
                        ),
                        max((f.defcon_probability for f in projection.fixtures), default=0.0),
                        projection.n_fixtures,
                        projection_set.price(element),
                        json.dumps(projection.breakdown.as_dict()),
                    )
                )
        if not rows:
            return 0
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO projection_snapshot VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
        return len(rows)

    def log_decision(self, gameweek: int, kind: str, payload: dict) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO decision_log VALUES (?,?,?,?,?)",
                (
                    self.season,
                    datetime.now(timezone.utc).isoformat(),
                    gameweek,
                    kind,
                    json.dumps(payload, default=str),
                ),
            )

    # -- reads ---------------------------------------------------------------

    def history(self, season: str | None = None) -> list[dict]:
        with self.connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM player_gameweek WHERE season = ? ORDER BY gameweek, element",
                (season or self.season,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def latest_snapshots(self, gameweek: int, season: str | None = None) -> list[dict]:
        """The most recent projection made for a gameweek."""
        with self.connect() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM projection_snapshot
                WHERE season = ? AND for_gameweek = ?
                  AND made_at = (
                      SELECT MAX(made_at) FROM projection_snapshot
                      WHERE season = ? AND for_gameweek = ?
                  )
                """,
                (season or self.season, gameweek, season or self.season, gameweek),
            )
            return [dict(row) for row in cursor.fetchall()]

    def gameweeks_recorded(self, season: str | None = None) -> list[int]:
        with self.connect() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT gameweek FROM player_gameweek WHERE season = ? ORDER BY gameweek",
                (season or self.season,),
            )
            return [int(row[0]) for row in cursor.fetchall()]

    def counts(self) -> dict[str, int]:
        with self.connect() as conn:
            out = {}
            for table in ("player_gameweek", "projection_snapshot", "decision_log"):
                out[table] = int(
                    conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608
                )
            return out


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
