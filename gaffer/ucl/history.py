"""A log of cumulative minutes, so future matchdays have per-match evidence.

UEFA's feed serves season totals and nothing per match: asking for matchday 1
and matchday 2 returns the same 106 minutes for Barkley. So the only thing a
single fetch can measure is the SHARE of his club's minutes a player has taken,
and that cannot tell a man who starts and is replaced on sixty from a man who
starts one game and watches the next. Guirassy's 120 of 180 minutes is either.

Differencing fixes it, and differencing needs a record. Each run writes the
cumulative minutes it saw, filed under the matchday it was fetched for — a read
taken before matchday N covers matchdays 1 to N-1 — and once two consecutive
reads exist, the gap between them is one matchday's minutes.

This starts paying at matchday four. There is no pre-matchday-two read in the
repository's history to recover, which is itself the argument for keeping one.
"""
from __future__ import annotations

import json
import os

PATH = "gaffer/ucl/minutes_log.json"


def load(path=PATH):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:                                       # noqa: BLE001
        return {}
    return d if isinstance(d, dict) else {}


def record(rows, matchday, path=PATH, log=None):
    """File this read's cumulative minutes under `matchday`.

    The first read for a matchday wins. A later one may have been taken while
    the matchday was in progress, in which case its totals are part-way through
    a match and differencing them would attribute half a game to the wrong
    week. An earlier read is never worse: before the deadline, nothing of this
    matchday has been played yet.
    """
    log = load(path) if log is None else log
    key = str(int(matchday))
    if key in log:
        return log, False
    log[key] = {r["id"]: float(r.get("minutes") or 0.0) for r in rows}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(log, f, separators=(",", ":"), sort_keys=True)
    return log, True


def per_matchday(log, upto):
    """{player id: [(matchday, minutes)]} for every matchday we can difference.

    A read filed under matchday N holds the totals after matchday N-1, so
    consecutive reads N and N+1 bracket matchday N exactly.
    """
    keys = sorted(int(k) for k in log)
    out = {}
    for a, b in zip(keys, keys[1:]):
        if b != a + 1 or a > upto:
            continue                       # a gap cannot be attributed
        before, after = log[str(a)], log[str(b)]
        for pid, mins in after.items():
            delta = mins - float(before.get(pid, 0.0))
            if delta < -1e-9:
                continue                   # totals only grow; this is not real
            out.setdefault(pid, []).append((a, max(0.0, delta)))
    return out


def attach(rows, played, cap=90.0):
    """Write each player's per-matchday minutes onto his row.

    `cap` keeps a double-header — which the league phase does not have, but the
    knockout rounds do — from reading as a 180-minute appearance.
    """
    n = 0
    for r in rows:
        log = played.get(r["id"])
        if not log:
            continue
        r["match_minutes"] = [min(cap, m) for _, m in sorted(log)]
        n += 1
    return n
