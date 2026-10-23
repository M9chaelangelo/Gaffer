"""What the bench is worth, and what it costs to touch it.

The solver maximised the eleven plus the captain and gave the four men on the
bench a coefficient of zero. Two things make that wrong, and they pull in
opposite directions.

**The bench is an option on your own team news.** A starter who takes no part
is replaced automatically by the first man on the bench who played and whose
position leaves the formation legal. With an eleven of nailed starters that
still fires rather more than once a matchday, so a fourth defender at 4.0 who
never plays is not the same purchase as one at 4.0 who starts every week — and
to a solver that scores only the eleven, they are identical.

**Touching the team kills it.** Any manual change during a matchday — a
substitution, moving the armband, even reordering the bench — disables
automatic substitutions for the rest of that matchday, and switching back does
not restore them. So the armband's option value and the bench's insurance
value are not additive. The engine priced both and added them, which flattered
every plan that used the Tuesday-to-Wednesday switch.

Only positions matter to the substitution cascade, never identities: swapping
any absent midfielder for a bench defender has the same consequences as
swapping any other. That collapses the state space to a few hundred cases,
which is why the exact value here is a lookup rather than a second simulation.
"""
from __future__ import annotations

from functools import lru_cache
from math import comb

import numpy as np

from . import project
from .rules import MAX_STARTING, MIN_STARTING

POSITIONS = ("GK", "DEF", "MID", "FWD")


# --- what the solver is told -------------------------------------------------

def nonstart(row):
    """P(this man takes no part), which is what calls the bench into play."""
    return 1.0 - project.start_prob(row)


def slot_odds(q, slots=3, starters=10):
    """P(the kth outfield bench slot is called on), for k = 1..slots.

    The bench is used in order, so slot k comes on exactly when at least k
    starters are absent. Treating absences as independent is the same
    assumption the simulator already makes when it draws them.
    """
    pmf = [comb(starters, i) * q ** i * (1 - q) ** (starters - i)
           for i in range(starters + 1)]
    return [float(sum(pmf[k:])) for k in range(1, slots + 1)]


def weights(xi_rows, bench_slots=3):
    """Linear bench coefficients for the solver, read off a real eleven.

    A bench player's contribution is P(his slot is called on) times his
    expected points: the chance he himself plays appears in both the
    probability of being used and in his expectation, and cancels.

    The outfield slots share one weight because the solver chooses the bench
    but not its order, so it cannot be told which slot a man would occupy.
    Averaging understates a bench whose best man is much better than its
    worst, which is the safe direction — `value()` prices the real thing once
    the squad is known.
    """
    out = [r for r in xi_rows if r["pos"] != "GK"]
    gk = [r for r in xi_rows if r["pos"] == "GK"]
    if not out:
        return {"GK": 0.0, "OUT": 0.0, "q": 0.0, "slot_odds": []}
    q = sum(nonstart(r) for r in out) / len(out)
    odds = slot_odds(q, slots=bench_slots, starters=len(out))
    return {
        # The reserve keeper comes on for one man and one man only.
        "GK": float(nonstart(gk[0])) if gk else 0.0,
        "OUT": float(sum(odds) / len(odds)) if odds else 0.0,
        "q": float(q),
        "slot_odds": odds,
    }


def order(bench_rows):
    """The order to enter on the site: outfielders best first, keeper apart.

    Formation can block a substitution, but it cannot make a worse player
    preferable to a better one at the same slot, so expectation descending is
    the order for the three outfielders.

    The reserve keeper is listed last whatever he projects, because his place
    in the queue cannot matter: a squad holds two keepers, so he is the only
    man who can replace the starting one and the only man who cannot replace
    anybody else. Ranking him on expected points put him at the head of the
    bench, which reads as a recommendation and is not one.
    """
    out = [r for r in bench_rows if r["pos"] != "GK"]
    gk = [r for r in bench_rows if r["pos"] == "GK"]
    return sorted(out, key=lambda r: -float(r["ep"])) + gk


# --- what actually happens ---------------------------------------------------

def _legal(comp):
    return all(MIN_STARTING[p] <= comp[p] <= MAX_STARTING[p] for p in POSITIONS)


def _cascade(comp, missing, bench_pos, mask, k=0):
    """Which bench slots come on. Returns a tuple of slot indices.

    The bench is used strictly in order, so a man who legally can come on
    does — skipping him to free a slot for someone behind him is not
    something the game would do, and an earlier version that maximised the
    count alone happily did it.

    Which absentee he replaces is the one free choice, and it is resolved by
    looking ahead: when two positions are both short, vacating one of them
    could in principle leave the rest of the bench unusable. It never does —
    `test_ucl_bench` sweeps every legal eleven, bench and pattern of
    absences and finds the look-ahead and the obvious first-vacancy rule
    agree everywhere — so this costs nothing and removes the question.
    """
    if k >= len(bench_pos):
        return ()
    if mask >> k & 1:
        best = None
        for out_pos in POSITIONS:
            if missing[out_pos] <= 0:
                continue
            trial = dict(comp)
            trial[out_pos] -= 1
            trial[bench_pos[k]] += 1
            if not _legal(trial):
                continue
            nxt = dict(missing)
            nxt[out_pos] -= 1
            cand = (k,) + _cascade(trial, nxt, bench_pos, mask, k + 1)
            if best is None or len(cand) > len(best):
                best = cand
        if best is not None:
            return best
    return _cascade(comp, missing, bench_pos, mask, k + 1)


@lru_cache(maxsize=64)
def _table(comp_items, cover_items, bench_pos):
    """Every reachable (absences, bench available) case, solved once.

    `cover` counts, per position, the starters whose absence is still coverable
    — see `value`'s `eligible`. Cached because the same eleven is priced two
    or three times over in one report and the sweep is a few thousand cases.
    """
    comp, cover = dict(comp_items), dict(cover_items)
    sizes = [cover[p] + 1 for p in POSITIONS]
    slots = len(bench_pos)
    rows = int(np.prod(sizes)) * (1 << slots)
    used = np.zeros((rows, slots), dtype=float)
    for state in range(rows):
        mask, rest = state % (1 << slots), state // (1 << slots)
        missing, ok = {}, True
        for p in reversed(POSITIONS):
            size = cover[p] + 1
            missing[p], rest = rest % size, rest // size
        for p in POSITIONS:
            if missing[p] > comp[p]:
                ok = False                 # cannot lose more than you started
        if not ok:
            continue
        for k in _cascade(comp, missing, bench_pos, mask):
            used[state, k] = 1.0
    return used


def value(xi_rows, bench_rows, samples, played, columns, eligible=None):
    """Expected points the bench adds, by replaying the simulated matchday.

    `played` is the boolean (sample, player) matrix from
    `project.simulate(..., with_minutes=True)`; a starter counts as absent
    when it is False, which includes a club with no fixture.

    `eligible` is the set of starter ids whose absence can still be covered.
    Pass it to price a matchday in which automatic substitutions were switched
    off part way through: the men who had already played are covered, the rest
    are not.
    """
    bench_rows = list(bench_rows)
    if not bench_rows or not xi_rows:
        return {"ep": 0.0, "by_slot": [], "expected_subs": 0.0}

    comp = {p: sum(1 for r in xi_rows if r["pos"] == p) for p in POSITIONS}
    covered = [r for r in xi_rows
               if eligible is None or r["id"] in eligible]
    cover = {p: sum(1 for r in covered if r["pos"] == p) for p in POSITIONS}
    bench_pos = [r["pos"] for r in bench_rows]
    bcols = [columns[r["id"]] for r in bench_rows]

    n = samples.shape[0]
    state = np.zeros(n, dtype=np.int64)
    for p in POSITIONS:
        cols = [columns[r["id"]] for r in covered if r["pos"] == p]
        absent = (~played[:, cols]).sum(1) if cols else np.zeros(n, dtype=int)
        state = state * (cover[p] + 1) + absent
    mask = np.zeros(n, dtype=np.int64)
    for k, c in enumerate(bcols):
        mask |= played[:, c].astype(np.int64) << k
    state = state * (1 << len(bench_rows)) + mask

    used = _table(tuple(comp.items()), tuple(cover.items()),
                  tuple(bench_pos))[state]              # (n, slots) of 0/1
    gained = used * samples[:, bcols]
    return {
        "ep": float(gained.sum(1).mean()),
        "by_slot": [{"name": r.get("name"), "pos": r["pos"],
                     "p_used": float(used[:, k].mean()),
                     "ep": float(gained[:, k].mean())}
                    for k, r in enumerate(bench_rows)],
        "expected_subs": float(used.sum(1).mean()),
    }


def forfeit(xi_rows, bench_rows, samples, played, columns, fixtures, day):
    """What switching the autosubs off at the end of `day` costs.

    Two numbers, because UEFA's rules say automatic substitutions stop for
    "the rest of the matchday" without saying when they are applied. If they
    settle at the end of each day, only the later matches lose their cover;
    if they settle once at the end, the whole bench is forfeited. The decision
    should be made on `most`, which is the number you cannot be wrong about by
    assuming it.
    """
    from .plan import day_of

    later = {r["id"] for r in xi_rows
             if (day_of(r, fixtures) or "") > day}
    whole = value(xi_rows, bench_rows, samples, played, columns)
    rest = value(xi_rows, bench_rows, samples, played, columns, eligible=later)
    return {"least": rest["ep"], "most": whole["ep"], "day": day,
            "starters_after": len(later)}
