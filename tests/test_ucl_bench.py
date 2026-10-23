"""The bench is an option, and touching the team cancels it.

The solver used to score the eleven and the captain and nothing else, so it
bought the cheapest legal bodies to sit behind them. These tests pin down what
the bench is actually worth: who comes on, when the formation stops them, and
what a mid-matchday change forfeits.
"""
from __future__ import annotations

import itertools

import numpy as np

from gaffer.ucl import bench, plan
from gaffer.ucl.rules import MAX_STARTING, MIN_STARTING

POSITIONS = bench.POSITIONS


def _row(pid, pos, ep=4.0, **kw):
    return dict(id=str(pid), name=f"P{pid}", pos=pos, ep=float(ep),
                price=5.0, **kw)


def _squad(xi_shape, bench_shape=("GK", "DEF", "MID", "FWD")):
    """An eleven of the given shape plus a bench, with ids 0..14."""
    xi, n = [], 0
    for pos, k in xi_shape.items():
        for _ in range(k):
            xi.append(_row(n, pos))
            n += 1
    ben = []
    for pos in bench_shape:
        ben.append(_row(n, pos, ep=6.0))
        n += 1
    cols = {r["id"]: i for i, r in enumerate(xi + ben)}
    return xi, ben, cols


def _matrices(xi, ben, absent=(), bench_out=(), points=6.0, n=50):
    """Points and played matrices with a fixed, hand-chosen outcome."""
    total = len(xi) + len(ben)
    played = np.ones((n, total), dtype=bool)
    samples = np.zeros((n, total))
    for i, r in enumerate(xi):
        samples[:, i] = 2.0
    for k in range(len(ben)):
        samples[:, len(xi) + k] = points
    for i in absent:
        played[:, i] = False
        samples[:, i] = 0.0
    for k in bench_out:
        played[:, len(xi) + k] = False
        samples[:, len(xi) + k] = 0.0
    return samples, played


# --- what the solver is told -------------------------------------------------

def test_slot_odds_fall_away_and_match_the_binomial():
    odds = bench.slot_odds(0.1, slots=3, starters=10)
    assert odds[0] > odds[1] > odds[2]
    assert abs(odds[0] - (1 - 0.9 ** 10)) < 1e-12
    assert abs(odds[2] - sum(
        __import__("math").comb(10, i) * 0.1 ** i * 0.9 ** (10 - i)
        for i in range(3, 11))) < 1e-12


def test_a_nailed_eleven_needs_less_bench_than_a_rotated_one():
    """The weight is a property of the eleven, which is why it is derived."""
    nailed = [_row(i, "MID", match_minutes=[90, 90, 90]) for i in range(10)]
    flaky = [_row(i, "MID", match_minutes=[90, 0, 0]) for i in range(10)]
    gk = [_row(99, "GK", match_minutes=[90, 90, 90])]
    a = bench.weights(nailed + gk)
    b = bench.weights(flaky + gk)
    assert 0 < a["OUT"] < b["OUT"]
    assert a["q"] < b["q"]


def test_the_reserve_keeper_is_priced_off_the_starting_keeper_alone():
    """He can come on for one man, so one man's absence is his whole value."""
    out = [_row(i, "MID", match_minutes=[90, 90, 90]) for i in range(10)]
    shaky = _row(99, "GK", match_minutes=[90, 0, 0])
    w = bench.weights(out + [shaky])
    assert abs(w["GK"] - bench.nonstart(shaky)) < 1e-12
    assert w["GK"] > w["OUT"]


def test_an_eleven_with_no_outfielders_prices_no_bench():
    assert bench.weights([])["OUT"] == 0.0


def test_the_outfield_bench_is_ordered_best_first():
    rows = [_row(1, "DEF", ep=2.0), _row(2, "MID", ep=5.0),
            _row(4, "FWD", ep=4.0)]
    assert [r["ep"] for r in bench.order(rows)] == [5.0, 4.0, 2.0]


def test_the_reserve_keeper_goes_last_however_well_he_projects():
    """His place in the queue cannot matter — he is the only man who can
    replace the keeper and the only one who can replace nobody else — so
    putting him at the head of the bench only reads as advice."""
    rows = [_row(1, "DEF", ep=2.0), _row(3, "GK", ep=9.0)]
    assert [r["pos"] for r in bench.order(rows)] == ["DEF", "GK"]


# --- what actually happens ---------------------------------------------------

def test_a_full_eleven_leaves_the_bench_worth_nothing():
    xi, ben, cols = _squad({"GK": 1, "DEF": 4, "MID": 4, "FWD": 2})
    samples, played = _matrices(xi, ben)
    v = bench.value(xi, bench.order(ben), samples, played, cols)
    assert v["ep"] == 0.0 and v["expected_subs"] == 0.0


def test_one_absent_starter_brings_on_the_best_legal_man():
    xi, ben, cols = _squad({"GK": 1, "DEF": 4, "MID": 4, "FWD": 2})
    # Starter 1 is a defender; the bench's best is the midfielder, and
    # four defenders can become three.
    samples, played = _matrices(xi, ben, absent=[1], points=6.0)
    ordered = bench.order(ben)
    v = bench.value(xi, ordered, samples, played, cols)
    assert v["expected_subs"] == 1.0
    assert v["ep"] == 6.0
    on = [s for s in v["by_slot"] if s["p_used"] > 0]
    assert len(on) == 1


def test_a_bench_player_who_did_not_play_himself_cannot_come_on():
    xi, ben, cols = _squad({"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
                           bench_shape=("MID",))
    samples, played = _matrices(xi, ben, absent=[1], bench_out=[0])
    v = bench.value(xi, ben, samples, played, cols)
    assert v["ep"] == 0.0


def test_the_formation_floor_blocks_a_substitution_that_would_break_it():
    """Three at the back is the floor, so the fourth defender has no cover.

    The eleven is 1-3-5-2 and a defender is out. The bench holds a midfielder
    — but the eleven already fields the maximum five, and dropping to two
    defenders is illegal, so nobody comes on and the bench scores nothing.
    """
    xi, ben, cols = _squad({"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},
                           bench_shape=("MID",))
    samples, played = _matrices(xi, ben, absent=[1])
    assert bench.value(xi, ben, samples, played, cols)["ep"] == 0.0
    # Same absence, a defender on the bench: he comes straight in.
    xi, ben, cols = _squad({"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},
                           bench_shape=("DEF",))
    samples, played = _matrices(xi, ben, absent=[1])
    assert bench.value(xi, ben, samples, played, cols)["ep"] == 6.0


def test_the_reserve_keeper_only_ever_replaces_the_keeper():
    xi, ben, cols = _squad({"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
                           bench_shape=("GK",))
    # An outfielder is out: the keeper cannot take his place, two keepers
    # being an illegal eleven.
    samples, played = _matrices(xi, ben, absent=[5])
    assert bench.value(xi, ben, samples, played, cols)["ep"] == 0.0
    # The keeper is out: now he can.
    samples, played = _matrices(xi, ben, absent=[0])
    assert bench.value(xi, ben, samples, played, cols)["ep"] == 6.0


def test_three_absences_use_three_bench_slots_and_no_more():
    xi, ben, cols = _squad({"GK": 1, "DEF": 4, "MID": 4, "FWD": 2})
    samples, played = _matrices(xi, ben, absent=[1, 5, 9])
    v = bench.value(xi, bench.order(ben), samples, played, cols)
    assert v["expected_subs"] == 3.0
    assert v["ep"] == 18.0


def test_a_club_with_no_fixture_counts_as_an_absence():
    """`played` is False for a man whose club does not play, and the engine
    must read that as the autosub-triggering absence it is, not as an error."""
    xi, ben, cols = _squad({"GK": 1, "DEF": 4, "MID": 4, "FWD": 2})
    samples, played = _matrices(xi, ben, absent=[3])
    assert bench.value(xi, bench.order(ben), samples, played,
                       cols)["expected_subs"] == 1.0


def test_narrowing_the_cover_can_only_reduce_the_bench_value():
    xi, ben, cols = _squad({"GK": 1, "DEF": 4, "MID": 4, "FWD": 2})
    samples, played = _matrices(xi, ben, absent=[1, 5])
    ordered = bench.order(ben)
    whole = bench.value(xi, ordered, samples, played, cols)
    part = bench.value(xi, ordered, samples, played, cols,
                       eligible={xi[1]["id"]})
    assert part["ep"] < whole["ep"]
    none = bench.value(xi, ordered, samples, played, cols, eligible=set())
    assert none["ep"] == 0.0


def test_the_value_is_a_probability_weighted_average_not_a_best_case():
    """Half the draws lose a starter, half do not: the bench is worth half."""
    xi, ben, cols = _squad({"GK": 1, "DEF": 4, "MID": 4, "FWD": 2})
    n = 100
    samples, played = _matrices(xi, ben, points=6.0, n=n)
    played[: n // 2, 1] = False
    samples[: n // 2, 1] = 0.0
    v = bench.value(xi, bench.order(ben), samples, played, cols)
    assert abs(v["ep"] - 3.0) < 1e-9
    assert abs(v["expected_subs"] - 0.5) < 1e-9


# --- the modelling assumption, measured rather than asserted -----------------

def _greedy(comp, missing, bench_pos, mask):
    """First legal vacancy, in position order — the obvious implementation."""
    comp, missing, used = dict(comp), dict(missing), []
    for k, pos in enumerate(bench_pos):
        if not (mask >> k & 1):
            continue
        for out_pos in POSITIONS:
            if missing[out_pos] <= 0:
                continue
            trial = dict(comp)
            trial[out_pos] -= 1
            trial[pos] = trial.get(pos, 0) + 1
            if all(MIN_STARTING[q] <= trial[q] <= MAX_STARTING[q]
                   for q in POSITIONS):
                comp, used = trial, used + [k]
                missing[out_pos] -= 1
                break
    return used


def test_taking_as_many_subs_as_the_formation_allows_is_the_greedy_rule():
    """`_cascade` searches exhaustively because a greedy pick could in
    principle vacate the position a later substitution needed. This sweeps
    every legal eleven, every pattern of absences and every bench the squad
    rules permit, and finds the two never disagree — so the exhaustive
    version is a bound that costs nothing and the greedy one is safe.
    """
    squad = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    shapes = [s for s in (
        {"GK": g, "DEF": d, "MID": m, "FWD": f}
        for g in (1,) for d in range(3, 6) for m in range(2, 6)
        for f in range(1, 4))
        if sum(s.values()) == 11]
    assert shapes
    checked = 0
    for comp in shapes:
        bench_pool = [p for p in POSITIONS
                      for _ in range(squad[p] - comp[p])]
        assert len(bench_pool) == 4
        for bench_pos in set(itertools.permutations(bench_pool)):
            for missing in itertools.product(
                    *[range(comp[p] + 1) for p in POSITIONS]):
                miss = dict(zip(POSITIONS, missing))
                if sum(missing) > 4:
                    continue          # more than the bench could ever cover
                for mask in range(16):
                    a = sorted(bench._cascade(comp, miss, bench_pos, mask))
                    b = _greedy(comp, miss, bench_pos, mask)
                    assert a == b, (comp, bench_pos, miss, mask, a, b)
                    checked += 1
    assert checked > 10000


# --- what the solver does with it --------------------------------------------

def _universe():
    """A pool where the bench is a real choice, not a rounding error.

    Six clubs, a full complement each, and at the bottom of the market two
    kinds of cheap player: a dud at 4.0 who projects half a point, and a
    squad man at 4.5 who projects two and a half. A solver that scores only
    the eleven prefers the dud every time, because he is 0.5m cheaper and
    both are worth nothing where it is looking.
    """
    rows, n = [], 0
    for club in "ABCDEF":
        for pos, k in (("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
            for i in range(k):
                n += 1
                if i == 0:
                    price, ep = 9.5, 6.0          # a star to spend on
                elif i == 1:
                    price, ep = 7.0, 4.0
                elif i % 2:
                    price, ep = 4.0, 0.5          # the dud
                else:
                    price, ep = 4.5, 2.5          # the squad man
                rows.append(dict(id=str(n), name=f"{club}{pos}{i}", pos=pos,
                                 team=club, price=price, ep=ep,
                                 match_minutes=[90, 90, 0]))
    return rows


def test_the_bench_is_not_bought_as_the_cheapest_legal_bodies():
    rows = _universe()
    blind = plan.optimise(rows, bench_weight=0, time_limit=60)
    priced = plan.optimise(rows, time_limit=60)
    assert blind["optimal"] and priced["optimal"]
    assert priced["bench_weight"]["OUT"] > 0
    ep = lambda out: sum(r["ep"] for r in out["bench"])
    assert ep(priced) > ep(blind)
    assert len(priced["squad"]) == 15 and len(priced["xi"]) == 11
    assert priced["spend"] <= 100.0 + 1e-6


def test_a_bench_weight_of_zero_reproduces_the_eleven_only_solve():
    """The old behaviour has to stay reachable, or a regression cannot be
    measured against it."""
    rows = _universe()
    a = plan.optimise(rows, bench_weight=0, time_limit=60)
    b = plan.optimise(rows, bench_weight={"GK": 0.0, "OUT": 0.0},
                      time_limit=60)
    assert sorted(r["id"] for r in a["squad"]) == \
        sorted(r["id"] for r in b["squad"])


def test_the_eleven_is_never_sold_off_to_buy_a_bench():
    rows = _universe()
    blind = plan.optimise(rows, bench_weight=0, time_limit=60)
    priced = plan.optimise(rows, time_limit=60)
    xi = lambda out: sum(r["ep"] for r in out["xi"])
    w = priced["bench_weight"]
    weigh = lambda out: sum(
        r["ep"] * (w["GK"] if r["pos"] == "GK" else w["OUT"])
        for r in out["bench"])
    assert weigh(priced) - weigh(blind) >= xi(blind) - xi(priced) - 1e-6
