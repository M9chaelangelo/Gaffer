"""Reading a raw projection row without guessing its key names.

`projections.build` returns rows keyed one way; `explain.player_card`
renames them for the web output. Three separate modules written against the
*card* names while holding *raw* rows, and each failed silently:

  imminent.py   read `ep_next`  -> printed 0.00 for every player
  crowd.py      read `ep_next`  -> elite captains table with no numbers
  template.py   read `start_prob` -> buy_leverage returned an empty list

None of them raised. `dict.get` on a missing key is a None or a default,
and a column of zeros looks like data. The mapping is small and it belongs
in one place, so a rename on either side breaks a test instead of a report.

    card name      raw name
    ep_next        ep[gw]
    ep_horizon     sum(ep[g] for g in gws)
    start_prob     p60
    exp_minutes    exp_min
"""
from __future__ import annotations

# The raw keys this module depends on. `test_projview` asserts a real row
# still carries them, so renaming one fails loudly rather than quietly.
RAW_KEYS = ("ep", "p60", "exp_min", "pos", "price", "name", "team", "status")


def ep(p, gw):
    """Projected points for one gameweek, from a raw row or a card row."""
    e = (p or {}).get("ep")
    if isinstance(e, dict):
        if gw in e:
            return float(e[gw] or 0.0)
        return float(e[min(e)] or 0.0) if e else 0.0
    if isinstance(e, list):
        return float(e[0] or 0.0) if e else 0.0
    if e is not None:
        return float(e)
    return float((p or {}).get("ep_next") or 0.0)


def ep_next(p, gws=None):
    return ep(p, (gws or [None])[0])


def horizon(p, gws):
    """Projected points over the whole horizon, raw or card."""
    e = (p or {}).get("ep")
    if isinstance(e, dict):
        return float(sum(e.get(g, 0.0) or 0.0 for g in gws))
    v = (p or {}).get("ep_horizon")
    return float(v) if v is not None else ep_next(p, gws) * len(gws)


def start_prob(p, default=None):
    """P(60+ minutes). `p60` on a raw row, `start_prob` on a card."""
    p = p or {}
    v = p.get("p60")
    if v is None:
        v = p.get("start_prob")
    return default if v is None else float(v)


def exp_minutes(p):
    p = p or {}
    v = p.get("exp_min")
    return v if v is not None else p.get("exp_minutes")
