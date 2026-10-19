"""Injury and suspension flags for the UCL pool, taken from FPL where it can.

UEFA's own availability prose lags. On 19 Oct 2026 FPL carried Haaland as
"Unspecified injury - 75% chance of playing" at 13:00; four hours later the
UEFA feed still said "In contention to start next game", with the matchday
deadline the following afternoon. The engine captained him.

Five of the thirty-six clubs in the league phase are Premier League sides, and
for those five FPL publishes a numeric `chance_of_playing_next_round` that is
updated several times a day. That covers the players most likely to be in an
English manager's squad. For everyone else the only honest answer is a
`doubts:` line in config/ucl.yaml, typed by hand.

UEFA's three-letter club code is the same string as FPL's `short_name` for all
five, so clubs join exactly and matching only has to resolve names within one
club — which keeps the two Gomezes and the two Hendersons apart.
"""
from __future__ import annotations

import unicodedata

# Written in full rather than derived, so that a code changing upstream shows
# up as a club that stops matching instead of as silently absent flags.
ENGLISH = ("ARS", "AVL", "LIV", "MCI", "MUN", "NEW", "TOT", "CHE", "NFO")

_PUNCT = str.maketrans({c: " " for c in ".'’-_"})


def _fold(s):
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.translate(_PUNCT).lower().split())


def _keys(full, short=""):
    """The forms a name might be written in, most specific first."""
    f, s = _fold(full), _fold(short)
    out = []
    if f:
        out.append(f)
        parts = f.split()
        if len(parts) > 1:
            out.append(parts[-1])              # surname alone
            # "B. Mbeumo" against "Bryan Mbeumo": initial plus surname.
            out.append(f"{parts[0][0]} {parts[-1]}")
    if s and s != f:
        out.append(s)
    return out


def chances(pool, boot):
    """{ucl player id: P(available)} for the clubs FPL also covers.

    Only players FPL has actually flagged appear. An unflagged player is not
    "100% fit" — he is "not news", and the caller must leave him alone rather
    than overwrite a UEFA note with a 1.0.
    """
    if not boot:
        return {}, []
    short = {t["id"]: t["short_name"] for t in boot["teams"]}
    by_club = {}
    for e in boot["elements"]:
        code = short.get(e["team"])
        if code not in ENGLISH:
            continue
        full = f"{e.get('first_name', '')} {e.get('second_name', '')}"
        for k in _keys(full, e.get("web_name")):
            # First writing wins, so a later namesake cannot steal the key;
            # within one club a collision is rare and a miss is safe.
            by_club.setdefault(code, {}).setdefault(k, e)

    out, unmatched = {}, []
    for p in pool:
        code = p.get("code")
        if code not in by_club:
            continue
        e = None
        for k in _keys(p.get("full_name"), p.get("name")):
            e = by_club[code].get(k)
            if e is not None:
                break
        if e is None:
            unmatched.append(f"{p.get('name')} ({code})")
            continue
        if e["status"] in ("i", "u", "s", "n"):
            out[p["id"]] = 0.0
            continue
        c = e.get("chance_of_playing_next_round")
        if c is None or c >= 100:
            continue                           # not news, so say nothing
        out[p["id"]] = max(0.0, min(1.0, c / 100.0))
    return out, unmatched


def apply(pool, flags, doubts=None):
    """Write P(available) onto the pool rows. Returns what it changed.

    A hand-typed `doubts` entry wins over a fetched flag: it is the later
    information by definition, since someone read the team news to write it.
    """
    # Match a hand-typed name the way a fetched one is matched, so "Barkley"
    # finds "R. Barkley". Having to type the exact feed spelling to set a doubt
    # is the friction that leaves doubts untyped before a deadline.
    forms = {p["id"]: set(_keys(p.get("full_name"), p.get("name")))
             for p in pool}
    named = {}
    for key, val in (doubts or {}).items():
        k = _fold(key)
        for p in pool:
            if k in forms[p["id"]]:
                named[p["id"]] = float(val)
    changed = []
    for p in pool:
        v = named.get(p["id"], flags.get(p["id"]))
        if v is None:
            continue
        p["availability"] = v
        changed.append((p.get("name"), p.get("code"), v,
                        "config" if p["id"] in named else "fpl"))
    missed = [k for k in (doubts or {})
              if not any(_fold(k) in forms[p["id"]] for p in pool)]
    return changed, missed
