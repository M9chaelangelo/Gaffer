"""Pull UEFA's UCL Fantasy feed and normalise it.

One endpoint carries everything:

    https://gaming.uefa.com/en/uclfantasy/services/api/Feed/players
        ?gamedayId={gd}&language=en

1,163 players with price, ownership, availability, season totals — and, per
player, `upcomingMatchesList`, which carries the opponent, home or away, and
the kick-off time. So the fixture list is derived by aggregating across players
rather than fetched: there is no fixtures endpoint, and sixteen guesses at one
all returned 404.

UEFA publishes no developer API and promises nothing about these shapes, so
every field is read defensively and `verify()` says plainly when the feed stops
looking like what this was written against.
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime

from .rules import POS

BASE = "https://gaming.uefa.com/en/uclfantasy/services/api"
UA = {"User-Agent": "Mozilla/5.0 (fpl-gaffer)",
      "Accept": "application/json, text/plain, */*"}

# Fields this module was written against. If UEFA drops one, say so loudly
# rather than silently projecting zeros.
REQUIRED = ("id", "pDName", "skill", "value", "tName", "tId", "selPer",
            "totPts", "minsPlyd", "gS", "assist", "bR", "upcomingMatchesList")


def _get(url, tries=3, timeout=45):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())      # read in full — a cap here
        except Exception:                        # silently truncates the body
            if i == tries - 1:                   # and corrupts the parse
                raise
            time.sleep(2 * (i + 1))


def raw_players(gameday):
    d = _get(f"{BASE}/Feed/players?gamedayId={gameday}&language=en")
    value = (d.get("data") or {}).get("value") or {}
    return value.get("playerList") or []


def verify(rows):
    """What this feed is missing relative to what the code expects."""
    if not rows:
        return ["the player list came back empty"]
    missing = [f for f in REQUIRED if f not in rows[0]]
    return [f"field '{f}' is gone from UEFA's feed" for f in missing]


def _num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def players(gameday):
    """Normalised player records, one per player."""
    out = []
    for r in raw_players(gameday):
        pos = POS.get(int(_num(r.get("skill"), 0)))
        if not pos:
            continue
        mins = _num(r.get("minsPlyd"))
        nineties = max(mins / 90.0, 1e-9)
        out.append({
            "id": str(r.get("id")),
            "name": r.get("pDName") or r.get("latinName") or "?",
            "full_name": r.get("pFName") or r.get("latinName") or "",
            "pos": pos,
            "team": r.get("tName") or "?",
            "team_id": str(r.get("tId")),
            "code": r.get("cCode") or "",
            "price": _num(r.get("value")),
            "own": _num(r.get("selPer")),           # per cent
            "transfers_in": _num(r.get("mTransferIn")),
            "transfers_out": _num(r.get("mTransferOut")),
            "total_points": _num(r.get("totPts")),
            "avg_points": _num(r.get("avgPlayerPts")),
            "minutes": mins,
            "team_played": int(_num(r.get("teamPlayed"))),
            # season counting stats, and the per-90 rates the projection needs
            "goals": _num(r.get("gS")),
            "assists": _num(r.get("assist")),
            "recoveries": _num(r.get("bR")),
            "saves": _num(r.get("saves")),
            "clean_sheets": _num(r.get("cS")),
            "conceded": _num(r.get("gC")),
            "yellow": _num(r.get("yC")),
            "red": _num(r.get("rC")),
            "motm": _num(r.get("mOM")),
            "goals90": _num(r.get("gS")) / nineties,
            "assists90": _num(r.get("assist")) / nineties,
            "recoveries90": _num(r.get("bR")) / nineties,
            "saves90": _num(r.get("saves")) / nineties,
            # availability. UEFA writes prose, not a flag.
            "trained": (r.get("trained") or "").strip(),
            "status": (r.get("pStatus") or "").strip(),
            "fixtures": _fixtures_of(r),
        })
    return out


def _fixtures_of(r):
    out = []
    for m in (r.get("upcomingMatchesList") or []):
        out.append({
            "matchday": str(m.get("mdId")),
            "home": (m.get("tLoc") or "").upper() == "H",
            "opponent": m.get("vsTSCode") or m.get("vsCCode") or "?",
            "opponent_id": str(m.get("vsTID")),
            "kickoff": _when(m.get("matchDate")),
        })
    return out


def _when(s):
    """UEFA sends 'MM/DD/YYYY HH:MM:SS'. Returns ISO, or the raw string."""
    try:
        return datetime.strptime(s, "%m/%d/%Y %H:%M:%S").isoformat()
    except Exception:
        return s or ""


def fixtures(rows, matchday):
    """The matchday's fixture list, aggregated out of the players.

    Each fixture appears once per side in the feed, so the pair is collapsed on
    a frozenset of the two team ids. Kick-off time matters more here than in
    FPL: the armband can be moved mid-matchday to a player who has not yet
    played, so which day a team plays is worth points.
    """
    seen = {}
    for p in rows:
        for f in p["fixtures"]:
            if f["matchday"] != str(matchday):
                continue
            key = frozenset((p["team_id"], f["opponent_id"]))
            if key in seen:
                continue
            seen[key] = {
                "matchday": f["matchday"],
                "home": p["team"] if f["home"] else f["opponent"],
                "away": f["opponent"] if f["home"] else p["team"],
                "kickoff": f["kickoff"],
            }
    return sorted(seen.values(), key=lambda x: x["kickoff"])
