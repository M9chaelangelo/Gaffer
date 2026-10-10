"""Pulls everything the gaffer needs. All sources are free and need no login."""
import json, time
import requests

FPL = "https://fantasy.premierleague.com/api"
UA = {"User-Agent": "Mozilla/5.0 (fpl-gaffer)"}


def _get(url, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def bootstrap():
    """Players, teams, prices, ownership, transfer flow, gameweek calendar."""
    return _get(f"{FPL}/bootstrap-static/")


def fixtures():
    return _get(f"{FPL}/fixtures/")


def history(entry_id):
    """Season history for an entry — and, in `chips`, every chip it has played.

    This is public. `chips_used` in config exists because nothing read it.
    """
    return _get(f"{FPL}/entry/{entry_id}/history/")


def entry(entry_id):
    return _get(f"{FPL}/entry/{entry_id}/")


def picks(entry_id, gw):
    return _get(f"{FPL}/entry/{entry_id}/event/{gw}/picks/")


def transfers(entry_id):
    """Needed to reconstruct purchase prices, which drive selling prices."""
    return _get(f"{FPL}/entry/{entry_id}/transfers/")


def league(league_id, pages=2):
    out = []
    for p in range(1, pages + 1):
        d = _get(f"{FPL}/leagues-classic/{league_id}/standings/?page_standings={p}")
        out += d["standings"]["results"]
        if not d["standings"]["has_next"]:
            break
    return out


def solio(url):
    """Solio's public projection feed. Free, no auth, refreshed every 4 hours.
    Only publishes leaders per category, so treat it as a partial override."""
    try:
        return _get(url)
    except Exception as e:
        print(f"  solio unavailable ({e}); falling back to local model only")
        return None


def current_gw(boot):
    nxt = [e for e in boot["events"] if e["is_next"]]
    if nxt:
        return nxt[0]["id"]
    return max(e["id"] for e in boot["events"] if e["finished"]) + 1
