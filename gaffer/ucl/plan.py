"""Pick the squad, the eleven and the armband for one UCL matchday.

Two things here have no FPL analogue and both are worth points.

**The armband moves.** There is no vice-captain; instead you may move the
captaincy during the matchday to any player who has not yet kicked off. So
captaining a Tuesday player is not the same decision as captaining an
equally good Wednesday one: after Tuesday you have seen his score and may
still change your mind. Writing `s` for the Tuesday captain's realised
points and `f` for the best Wednesday fallback's expectation, you keep the
armband exactly when `s > E[f]`, so the captaincy is worth

    E[ max(s, E[f]) ]  >=  max(E[s], E[f])

and the gap is the option. It is computed from the simulated samples, not
asserted, and it is why the plan names both a captain and a fallback.

**Transfers are two a matchday, free.** The budget is the squad's current
value plus the bank, because UEFA sells at the live price with no FPL-style
selling-price rule.
"""
from __future__ import annotations

import numpy as np
import pulp

from .. import solver
from . import bench as benchmod
from .rules import GAME


def _index(rows, key="id"):
    return {str(r[key]): i for i, r in enumerate(rows)}


def squad_value(rows, owned_ids):
    idx = _index(rows)
    return sum(rows[idx[i]]["price"] for i in owned_ids if i in idx)


def _solve(rows, owned_ids, bank, free_transfers, max_extra_transfers,
           locked, banned, time_limit, bench_w):
    g = GAME
    ft = g.free_transfers_per_round if free_transfers is None else free_transfers
    idx = _index(rows)
    P = range(len(rows))
    owned = {idx[i] for i in map(str, owned_ids) if i in idx}
    lock = {idx[i] for i in map(str, locked) if i in idx}
    ban = {idx[i] for i in map(str, banned) if i in idx}
    budget = (squad_value(rows, [str(i) for i in owned_ids]) + bank
              if owned else g.budget)

    m = pulp.LpProblem("ucl", pulp.LpMaximize)
    sq = {i: m.add_variable(f"sq{i}", cat="Binary") for i in P}
    xi = {i: m.add_variable(f"xi{i}", cat="Binary") for i in P}
    cap = {i: m.add_variable(f"cap{i}", cat="Binary") for i in P}
    tin = {i: m.add_variable(f"in{i}", cat="Binary") for i in P}
    tout = {i: m.add_variable(f"out{i}", cat="Binary") for i in P}
    extra = m.add_variable("extra", lowBound=0,
                           upBound=max_extra_transfers, cat="Integer")

    m += pulp.lpSum(sq[i] for i in P) == g.squad_size
    for pos, k in g.squad.items():
        m += pulp.lpSum(sq[i] for i in P if rows[i]["pos"] == pos) == k
    for team in {r["team"] for r in rows}:
        m += pulp.lpSum(sq[i] for i in P
                        if rows[i]["team"] == team) <= g.max_per_club
    m += pulp.lpSum(rows[i]["price"] * sq[i] for i in P) <= budget + 1e-6

    m += pulp.lpSum(xi[i] for i in P) == 11
    for pos in g.positions:
        inpos = [i for i in P if rows[i]["pos"] == pos]
        m += pulp.lpSum(xi[i] for i in inpos) >= g.xi_min[pos]
        m += pulp.lpSum(xi[i] for i in inpos) <= g.xi_max[pos]
    for i in P:
        m += xi[i] <= sq[i]
        m += cap[i] <= xi[i]
    m += pulp.lpSum(cap[i] for i in P) == 1

    for i in P:
        was = 1 if i in owned else 0
        m += sq[i] == was + tin[i] - tout[i]
        if was:
            m += tin[i] == 0
        else:
            m += tout[i] == 0
        if i in lock:
            m += sq[i] == 1
        if i in ban:
            m += sq[i] == 0
    if owned:
        m += pulp.lpSum(tin[i] for i in P) == pulp.lpSum(tout[i] for i in P)
        m += pulp.lpSum(tin[i] for i in P) <= ft + extra

    # The captain term here is the plain doubling. The armband's option
    # value depends on which other players are in the eleven, which the MILP
    # cannot see while it is still choosing them, so the captain is re-picked
    # exactly by `armband()` once the eleven is known. That re-pick can only
    # raise the total; what it cannot do is reach back and change the eleven,
    # and the error from that is second order.
    #
    # The bench is not scenery either. A starter who takes no part is replaced
    # automatically, so the four men behind the eleven carry an expectation of
    # their own: the chance their slot is called on, times what they score.
    # Scoring them at zero bought the cheapest legal bodies, which is the one
    # bench guaranteed to be worth nothing. `bench.weights` reads the
    # coefficient off a real eleven rather than asserting one.
    ep = [float(r["ep"]) for r in rows]
    bw = [bench_w["GK"] if rows[i]["pos"] == "GK" else bench_w["OUT"]
          for i in P]
    m += (pulp.lpSum(ep[i] * xi[i] for i in P)
          + pulp.lpSum(ep[i] * (g.captain_multiplier - 1) * cap[i] for i in P)
          + pulp.lpSum(ep[i] * bw[i] * (sq[i] - xi[i]) for i in P)
          - g.extra_transfer_cost * extra)

    # Ask CBC to prove optimality rather than stop at its build's default
    # gap: fifteen from a thousand is a small model, and "Gap Limit" reads to
    # a caller comparing the word exactly like a solve that failed.
    status = solver.solve(m, time_limit, gap=0)
    chosen = [i for i in P if sq[i].value() and sq[i].value() > 0.5]
    start = [i for i in chosen if xi[i].value() and xi[i].value() > 0.5]
    return {
        # The word is for the reader; `optimal` is what code should test.
        # A build that stops on a zero gap has proved optimality and says
        # "Gap Limit" doing it.
        "status": status,
        "optimal": solver.proven(status, gap=0),
        "squad": [rows[i] for i in chosen],
        "xi": [rows[i] for i in start],
        "bench": [rows[i] for i in chosen if i not in set(start)],
        "captain": next((rows[i] for i in P
                         if cap[i].value() and cap[i].value() > 0.5), None),
        # Drafting from scratch, every pick satisfies `tin`, which is not a
        # transfer in any sense a reader means. A draft reports no transfers.
        "draft": not owned,
        "out": ([rows[i] for i in P
                 if tout[i].value() and tout[i].value() > 0.5] if owned else []),
        "in": ([rows[i] for i in P
                if tin[i].value() and tin[i].value() > 0.5] if owned else []),
        "hits": int(round(extra.value() or 0)),
        "budget": budget,
        "spend": sum(rows[i]["price"] for i in chosen),
        "bench_weight": dict(bench_w),
    }


ZERO_BENCH = {"GK": 0.0, "OUT": 0.0}


def optimise(rows, owned_ids=(), bank=0.0, free_transfers=None,
             max_extra_transfers=0, locked=(), banned=(), time_limit=120,
             bench_weight=None):
    """Choose fifteen, start eleven, name a captain.

    `rows` carry an `ep` each (see `project.summarise`). With `owned_ids`
    empty this drafts a squad from scratch; with a squad it plans transfers
    against it. `locked` are ids that must stay (a suspended player you
    cannot sell for value, say); `banned` are ids that must not be bought.

    `bench_weight` is how much of a bench player's expectation counts, as
    {"GK": w, "OUT": w} or one number for both. Left alone it is derived
    rather than assumed: solve once with the bench at zero, then ask how
    often THAT eleven would leave somebody out, which is the only thing the
    weight depends on. Passing 0 reproduces the old eleven-only solve.
    """
    args = (rows, owned_ids, bank, free_transfers, max_extra_transfers,
            locked, banned, time_limit)
    first = _solve(*args, bench_w=ZERO_BENCH)
    if isinstance(bench_weight, (int, float)):
        bench_weight = {"GK": float(bench_weight), "OUT": float(bench_weight)}
    if bench_weight is None:
        if not first["xi"]:
            return first                  # infeasible; nothing to read it off
        w = benchmod.weights(first["xi"])
        bench_weight = {"GK": w["GK"], "OUT": w["OUT"]}
    if max(bench_weight.values()) <= 0:
        return first
    out = _solve(*args, bench_w=bench_weight)
    # A second solve that fails where the first succeeded is a solver
    # problem, not a plan: keep the one that worked.
    return out if out["xi"] else first


# --- the armband -------------------------------------------------------------

def day_of(player, fixtures):
    """Which calendar day this player's club plays on, or None."""
    for f in fixtures:
        if player["team"] in (f["home"], f["away"]):
            return (f.get("kickoff") or "")[:10] or None
    return None


def armband(xi_rows, samples, columns, fixtures, switch_cost=None):
    """Best captain, pricing the mid-matchday move.

    `columns` maps a player id to its column in `samples`. Returns one entry
    per candidate, best first, each carrying the plain captaincy value and
    the value with the option to switch to the best later-day fallback.

    `switch_cost` is what moving the armband costs, as one number or a
    mapping from the day you would move it on. It is not zero: any manual
    change disables automatic substitutions for the rest of the matchday, so
    moving the armband after Tuesday means nobody covers an absent Wednesday
    starter. That changes the rule, not just the total — you keep the armband
    unless the fallback beats it by MORE than the cover you are giving up —
    which is why the cost belongs in here and not in a subtraction afterwards.
    See `bench.forfeit`.
    """
    days = {r["id"]: day_of(r, fixtures) for r in xi_rows}
    playing = [r for r in xi_rows if days[r["id"]]]
    if not playing:
        return []
    order = sorted({d for d in days.values() if d})

    out = []
    for r in playing:
        d = days[r["id"]]
        cost = _cost(switch_cost, d)
        later = [q for q in playing
                 if days[q["id"]] and days[q["id"]] > d]
        fallback = max(later, key=lambda q: q["ep"], default=None)
        s = samples[:, columns[r["id"]]]
        if fallback is None:
            val, gain = float(s.mean()), 0.0
        else:
            # Keep the armband when the Tuesday score already beats what the
            # Wednesday man is worth NET of the cover the switch forfeits;
            # otherwise move it. The expectation of that rule is the
            # captaincy's true value.
            f_ep = float(fallback["ep"]) - cost
            val = float(np.maximum(s, f_ep).mean())
            gain = val - float(s.mean())
        out.append({
            "player": r, "day": d, "days": order,
            "captain_ep": float(s.mean()),
            "armband_ep": val,
            "option_gain": gain,
            "switch_cost": cost,
            "fallback": fallback,
        })
    return sorted(out, key=lambda x: -x["armband_ep"])


def _cost(switch_cost, day):
    if switch_cost is None:
        return 0.0
    if isinstance(switch_cost, (int, float)):
        return float(switch_cost)
    return float(switch_cost.get(day, 0.0))
