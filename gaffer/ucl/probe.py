"""Map UEFA's UCL Fantasy feeds from a machine with network.

Round one found the player feed and then misreported it: the body was capped at
200KB, so json.loads choked on a truncated object and the endpoint was recorded
as "not JSON" when its content-type had said application/json all along. Read
the whole body or do not claim to have parsed it.

Round two reads fully, dumps the player schema, and hunts the remaining feeds.
"""
from __future__ import annotations

import json
import sys
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (fpl-gaffer ucl probe)",
      "Accept": "application/json, text/plain, */*"}

BASE = "https://gaming.uefa.com/en/uclfantasy/services/api"
GD = 2   # the live matchday


def get(url, timeout=20):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.headers.get("Content-Type", ""), r.read()


def unwrap(obj):
    """UEFA wraps payloads as {"data": {"value": ...}}."""
    if isinstance(obj, dict) and "data" in obj:
        d = obj["data"]
        if isinstance(d, dict) and "value" in d:
            return d["value"]
        return d
    return obj


def describe(name, url):
    try:
        status, ctype, raw = get(url)
    except Exception as e:
        print(f"  ERR  {name:26} {type(e).__name__}: {str(e)[:70]}")
        return None
    if "json" not in ctype:
        print(f"  html {name:26} {status} {ctype} ({len(raw)}b) — SPA shell, not an API")
        return None
    try:
        data = unwrap(json.loads(raw))
    except Exception as e:
        print(f"  BAD  {name:26} {status} json parse failed: {e}")
        return None
    shape = (f"dict keys={list(data)[:10]}" if isinstance(data, dict)
             else f"list len={len(data)}" if isinstance(data, list) else type(data).__name__)
    print(f"  OK   {name:26} {status} {len(raw):>9,}b  {shape}")
    return data


def main():
    found = {}

    print("\n--- player feed, read in full ---")
    players = describe("players", f"{BASE}/Feed/players?gamedayId={GD}&language=en")
    plist = None
    if isinstance(players, dict):
        for k, v in players.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                plist = v
                print(f"       list key '{k}', {len(v)} players")
                break
    if plist:
        found["players"] = f"{BASE}/Feed/players?gamedayId={{gd}}&language=en"
        print("\n       FULL FIELD LIST of one player:")
        for k, v in sorted(plist[0].items()):
            print(f"         {k:22} = {json.dumps(v, ensure_ascii=False)[:70]}")
        print("\n       three samples:")
        for p in plist[:3]:
            keep = {k: p.get(k) for k in
                    ("id", "pDName", "tName", "skill", "value", "totPts", "minsPlyd", "gpIndex")
                    if k in p}
            print(f"         {json.dumps(keep, ensure_ascii=False)}")

    print("\n--- hunting the other feeds ---")
    for name, path in [
        ("matches",       f"/Feed/matches?gamedayId={GD}&language=en"),
        ("fixtures",      f"/Feed/fixtures?gamedayId={GD}&language=en"),
        ("match",         f"/Feed/match?gamedayId={GD}&language=en"),
        ("gamedays",      "/Feed/gamedays?language=en"),
        ("gameday",       f"/Feed/gameday?gamedayId={GD}&language=en"),
        ("teams",         "/Feed/teams?language=en"),
        ("team",          "/Feed/team?language=en"),
        ("constraints",   "/Feed/constraints?language=en"),
        ("settings",      "/Feed/settings?language=en"),
        ("playerstats",   f"/Feed/playerstats?gamedayId={GD}&language=en"),
        ("matchstats",    f"/Feed/matchstats?gamedayId={GD}&language=en"),
        ("competition",   "/Feed/competition?language=en"),
        ("phases",        "/Feed/phases?language=en"),
        ("playersGd1",    "/Feed/players?gamedayId=1&language=en"),
        ("statistics",    f"/Statistics/players?gamedayId={GD}&language=en"),
        ("leagues",       "/Leagues/leagues?language=en"),
    ]:
        d = describe(name, BASE + path)
        if d is not None:
            found[name] = BASE + path

    print(f"\n--- {len(found)} usable feeds ---")
    for k, v in found.items():
        print(f"  {k:14} {v}")

    with open("probe.json", "w") as f:
        json.dump({"found": found,
                   "player_fields": sorted(plist[0]) if plist else [],
                   "player_count": len(plist) if plist else 0,
                   "player_sample": plist[:2] if plist else []}, f, indent=1,
                  ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
