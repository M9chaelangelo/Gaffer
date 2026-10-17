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
UA = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/140.0.0.0 Safari/537.36"),
      "Accept": "application/json, text/plain, */*",
      "Accept-Language": "en-GB,en;q=0.9",
      "Referer": "https://gaming.uefa.com/en/uclfantasy/",
      "Origin": "https://gaming.uefa.com",
      "Connection": "close"}

# Fields this module was written against. If UEFA drops one, say so loudly
# rather than silently projecting zeros.
REQUIRED = ("id", "pDName", "skill", "value", "tName", "tId", "selPer",
            "totPts", "minsPlyd", "gS", "assist", "bR", "upcomingMatchesList")


def _get(url, tries=4, timeout=45):
    """Fetch and parse, or raise with every attempt's actual error.

    The first version re-raised only the last exception, and the caller
    turned that into "no live feed". Six minutes of timeouts became one
    sentence that did not say what had happened — the diagnosis was
    collected and discarded, which is the same bug as printing a profile
    you then throw away.

    Shorter timeouts and one more try: a two-megabyte body that has not
    begun after forty-five seconds is not slow, it is blocked, and four
    quick attempts learn more than three long ones.
    """
    errors = []
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()                  # read in full — a cap here
            return json.loads(body)              # silently truncates the body
        except Exception as e:                   # and corrupts the parse
            detail = getattr(e, "code", None) or getattr(e, "reason", None)
            errors.append(f"attempt {i + 1}: {e.__class__.__name__}"
                          + (f" ({detail})" if detail else ""))
            if i < tries - 1:
                time.sleep(2 * (i + 1))
    raise RuntimeError(f"GET {url} failed — " + "; ".join(errors))


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


def players(gameday=None, rows=None):
    """Normalised player records, one per player.

    Pass `rows` to reuse a payload already fetched. The feed is 2MB and the
    first version of this pulled it four times in one run, which is how the
    snapshot step ended in a read timeout.
    """
    if rows is None:
        rows = raw_players(gameday)
    out = []
    for r in rows:
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
