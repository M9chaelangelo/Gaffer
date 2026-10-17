"""Turns your mini-league into ownership numbers you can act on.

Global ownership tells you what the world is doing. It does not tell you how to
beat 41 specific people. This does."""
import time
from collections import Counter

from . import cache, fetch


# Bumped when the shape of a snapshot changes, so a cached one written by
# an older version is refetched rather than silently read as missing data.
SNAPSHOT_VERSION = 2


def snapshot(league_id, entry_id, gw, rival_depth=10, pages=2):
    raw = cache.cached(f"league:v{SNAPSHOT_VERSION}:{league_id}:{gw}:{rival_depth}",
                       lambda: _snapshot(league_id, entry_id, gw, rival_depth, pages))
    raw["my_ids"] = set(raw["my_ids"])
    raw["league_own"] = {int(k): v for k, v in raw["league_own"].items()}
    raw["pack_own"] = {int(k): v for k, v in raw["pack_own"].items()}
    raw.setdefault("squads", [])
    return raw


def _snapshot(league_id, entry_id, gw, rival_depth=10, pages=2):
    rows = fetch.league(league_id, pages=pages)
    rows.sort(key=lambda r: r["rank"])
    me = next((r for r in rows if r["entry"] == entry_id), None)

    all_own, pack_own, cap = Counter(), Counter(), Counter()
    my_ids = set()
    squads = []
    n_all = n_pack = 0

    my_rank = me["rank"] if me else len(rows)
    pack = [r for r in rows if r["rank"] < my_rank][:rival_depth] or rows[:rival_depth]
    pack_ids = {r["entry"] for r in pack}

    for r in rows:
        try:
            p = fetch.picks(r["entry"], gw)
        except Exception:
            continue
        ids = [x["element"] for x in p["picks"]]
        if r["entry"] == entry_id:
            my_ids = set(ids)
        n_all += 1
        for i in ids:
            all_own[i] += 1
        if r["entry"] in pack_ids:
            n_pack += 1
            for i in ids:
                pack_own[i] += 1
        c = [x["element"] for x in p["picks"] if x["is_captain"]]
        if c:
            cap[c[0]] += 1
        # Keep the squad itself, not just the counters it contributes to.
        # Ownership says what the league holds; only the individual teams
        # say who finishes above whom, and the whole point of the league is
        # the second question. `multiplier` carries their own lineup and
        # armband, so replaying it needs no assumption about either.
        squads.append({
            "entry": r["entry"],
            "name": r.get("entry_name"),
            "manager": r.get("player_name"),
            "rank": r.get("rank"),
            "total": r.get("total"),
            "captain": c[0] if c else None,
            "picks": [{"element": x["element"],
                       "multiplier": x.get("multiplier", 0),
                       "position": x.get("position")} for x in p["picks"]],
        })
        time.sleep(0.08)

    return {
        "rows": rows,
        "me": me,
        "my_ids": list(my_ids),
        "squads": squads,
        "league_own": {i: c / max(n_all, 1) for i, c in all_own.items()},
        "pack_own": {i: c / max(n_pack, 1) for i, c in pack_own.items()},
        "captains": dict(cap),
        "n_all": n_all,
        "n_pack": n_pack,
        "points_behind": (rows[0]["total"] - me["total"]) if me else None,
    }
