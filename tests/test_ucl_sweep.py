"""Chips are worth what they beat, and the thing they beat is not nothing.

The failure these guard against is a baseline that quietly flatters the chip.
An early version measured a Wildcard against a squad frozen for six matchdays
and put it at +54 points, which measured the cost of paralysis. The fix moved
the error rather than removing it: measuring a rebuild at matchday seven
against the squad you hold TODAY credits the chip with three matchdays of
ordinary transfers you were going to make anyway. It showed up as a Wildcard
worth more over a two-matchday window than over a five-matchday one, which
cannot be true of the same chip.
"""
from __future__ import annotations

import pytest

from gaffer.ucl import project, sweep
from test_ucl_bench import _universe


SCHED = {
    1: [{"home": "A", "away": "B", "date": "2026-09-08"},
        {"home": "C", "away": "D", "date": "2026-09-09"}],
    2: [{"home": "B", "away": "C", "date": "2026-10-13"},
        {"home": "D", "away": "A", "date": "2026-10-14"}],
}


def _cheap_squad(rows):
    """The cheapest LEGAL fifteen: 2/5/5/3 and no more than three per club.

    Taking the fifteen cheapest rows instead gives eight defenders and seven
    midfielders, which is not a squad — the solver says Infeasible and the
    test that built it reads like a bug in the engine.
    """
    shape, perclub, out = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}, {}, []
    for r in sorted(rows, key=lambda r: r["price"]):
        if shape.get(r["pos"], 0) and perclub.get(r["team"], 0) < 3:
            out.append(r["id"])
            shape[r["pos"]] -= 1
            perclub[r["team"]] = perclub.get(r["team"], 0) + 1
    assert len(out) == 15
    return out


def _by_md(scale):
    """{matchday: rows} off one universe, scaled per matchday per club.

    `scale` is {matchday: {club letter: multiplier}}; a club absent from a
    matchday keeps its expectation.
    """
    base = _universe()
    out = {}
    for md, by_club in scale.items():
        out[md] = [dict(r, ep=r["ep"] * by_club.get(r["team"], 1.0))
                   for r in base]
    return out


# --- fixtures ----------------------------------------------------------------

def test_fixtures_come_out_in_the_shape_the_simulator_wants():
    fx = sweep.fixtures_for(SCHED, 2)
    assert [f["home"] for f in fx] == ["B", "D"]
    assert all(f["matchday"] == "2" for f in fx)
    # The day is what the armband cares about, and it has to survive.
    assert [f["kickoff"][:10] for f in fx] == ["2026-10-13", "2026-10-14"]


def test_the_same_player_is_projected_differently_in_different_matchdays():
    """If he were not, there would be no horizon — only a repeated matchday."""
    import sys
    sys.path.insert(0, "tests")
    from test_ucl import _pool
    pool = [p for p in _pool() if p["team"] in ("Alpha", "Beta")]
    sched = {1: [{"home": "Alpha", "away": "Beta", "date": "2026-09-08"}],
             2: [{"home": "Beta", "away": "Alpha", "date": "2026-10-13"}]}
    by_md = sweep.project_window(pool, sched, [1, 2], n=3000)
    one = {r["id"]: r["ep"] for r in by_md[1]}
    two = {r["id"]: r["ep"] for r in by_md[2]}
    moved = [k for k in one if abs(one[k] - two[k]) > 0.05]
    assert moved, "home and away gave the same answer"


# --- the roll ----------------------------------------------------------------

def test_the_roll_carries_the_squad_it_built_into_the_next_matchday():
    by_md = _by_md({1: {"A": 0.1}, 2: {"A": 0.1, "F": 1.5}})
    start = _cheap_squad(by_md[1])
    w = sweep.roll(by_md, start, bank=60.0, time_limit=60)
    assert w["complete"] and len(w["legs"]) == 2
    assert w["legs"][0]["held_before"] == start
    assert w["legs"][1]["held_before"] == w["legs"][0]["squad_ids"]
    assert w["total"] == pytest.approx(sum(l["worth"] for l in w["legs"]))


def test_holdings_is_what_you_hold_going_in_not_what_you_hold_today():
    by_md = _by_md({1: {}, 2: {"F": 2.0}})
    start = _cheap_squad(by_md[1])
    w = sweep.roll(by_md, start, bank=60.0, time_limit=60)
    held = sweep.holdings(w)
    assert held[1][0] == start
    assert held[2][0] != start, "the roll made transfers and holdings lost them"
    assert held[2][0] == w["legs"][0]["squad_ids"]


def test_tails_counts_from_each_matchday_to_the_end():
    by_md = _by_md({1: {}, 2: {}})
    start = _cheap_squad(by_md[1])
    w = sweep.roll(by_md, start, bank=60.0, time_limit=60)
    t = sweep.tails(w)
    assert t[1] == pytest.approx(w["total"])
    assert t[2] == pytest.approx(w["legs"][1]["worth"])
    assert t[1] > t[2]


def test_a_matchday_the_solver_cannot_finish_stops_the_roll():
    """A total that silently misses a matchday is worse than a short one."""
    by_md = _by_md({1: {}, 2: {}})
    start = _cheap_squad(by_md[1])
    every_keeper = [r["id"] for r in by_md[1] if r["pos"] == "GK"]
    w = sweep.roll(by_md, start, bank=60.0, banned=every_keeper,
                   time_limit=30)
    assert not w["complete"]
    assert w["legs"] and w["legs"][0]["worth"] is None


def test_an_unknown_squad_id_is_not_quietly_read_as_a_draft():
    """`optimise` drops ids it cannot find, so a squad of fifteen typos reads
    as nobody owning anything and the engine drafts from scratch on the full
    budget. That is a plausible answer to the wrong question, which is why
    `main.run` refuses to plan when a configured name does not match — this
    records the behaviour the refusal is there to catch."""
    by_md = _by_md({1: {}, 2: {}})
    w = sweep.roll(by_md, [f"nope{i}" for i in range(15)], bank=0.0,
                   time_limit=30)
    assert w["complete"]
    assert w["legs"][0]["spend"] > 90.0            # it spent the whole budget


# --- the chips ---------------------------------------------------------------

def test_limitless_is_measured_against_the_squad_the_roll_leaves_you():
    by_md = _by_md({1: {}, 2: {"F": 1.4}})
    start = _cheap_squad(by_md[1])
    w = sweep.roll(by_md, start, bank=60.0, time_limit=60)
    lim = sweep.limitless(by_md, w, time_limit=60)
    for md, r in lim.items():
        assert r["mine"] == pytest.approx(
            next(l["worth"] for l in w["legs"] if l["matchday"] == md))
        assert r["gain"] >= -1e-6, "money cannot make your eleven worse"


def test_limitless_actually_spends_more_than_the_budget_allows():
    by_md = _by_md({1: {}, 2: {}})
    start = _cheap_squad(by_md[1])
    w = sweep.roll(by_md, start, bank=60.0, time_limit=60)
    lim = sweep.limitless(by_md, w, time_limit=60)
    assert max(r["spend"] for r in lim.values()) > 100.0


def test_the_wildcard_baseline_is_the_tail_of_the_roll():
    """The whole bug, in one assertion."""
    by_md = _by_md({1: {}, 2: {"F": 1.5}})
    start = _cheap_squad(by_md[1])
    w = sweep.roll(by_md, start, bank=60.0, time_limit=60)
    tails = sweep.tails(w)
    wc = sweep.wildcard(by_md, w, window_time_limit=90, roll_time_limit=60,
                        prune=None)
    for md, r in wc.items():
        assert r["baseline"] == pytest.approx(tails[md])
        assert r["gain"] >= -1e-6


def test_a_rebuild_on_a_squad_that_is_already_optimal_gains_nothing():
    """And so the chip is not recommended out of habit."""
    by_md = _by_md({1: {}, 2: {}})
    from gaffer.ucl import plan
    best = plan.optimise(by_md[1], time_limit=60)
    ids = [r["id"] for r in best["squad"]]
    w = sweep.roll(by_md, ids, bank=100.0 - best["spend"], time_limit=60)
    wc = sweep.wildcard(by_md, w, window_time_limit=90, roll_time_limit=60,
                        prune=None)
    assert wc[1]["gain"] < 1.0


def test_the_table_names_a_blocked_matchday_and_a_simultaneous_one():
    """Matchdays 1, 9 and 11 give everyone unlimited transfers, so a chip
    there buys nothing; 8 and 17 kick off at once, so the in-play option is
    gone. Both are the sort of thing a number alone will not tell you."""
    lim = {1: {"matchday": 1, "blocked": True, "simultaneous": False,
               "mine": 50.0, "best": 60.0, "gain": 10.0, "spend": 150.0,
               "optimal": True},
           8: {"matchday": 8, "blocked": False, "simultaneous": True,
               "mine": 50.0, "best": 60.0, "gain": 10.0, "spend": 150.0,
               "optimal": True}}
    text = sweep.table(lim=lim)
    assert "CHIP BLOCKED" in text
    assert "all kick off at once" in text


# --- one squad, many matchdays -----------------------------------------------

def _over_window(squad_ids, ep_by_md, rows):
    """What a fixed squad scores over the window, picking the best legal
    eleven and armband each matchday. The comparison a horizon has to win."""
    from gaffer.ucl import plan
    owned = set(map(str, squad_ids))
    total = 0.0
    for md, ep in ep_by_md.items():
        pool = [dict(r, ep=ep[r["id"]]) for r in rows if r["id"] in owned]
        out = plan.optimise(pool, owned_ids=squad_ids, free_transfers=0,
                            time_limit=60)
        total += (sum(r["ep"] for r in out["xi"])
                  + max((r["ep"] for r in out["xi"]), default=0.0))
    return total


def _two_specials():
    """A pool holding exactly one real decision, so the answer is checkable.

    Eight identical clubs of filler at 6.0, and two expensive midfielders at
    12.0: the HOP man scores nine in matchday one and nothing in matchday
    two, the KEEP man scores six in both. A solve that sees one matchday
    takes the nine; a solve that sees both takes the twelve. Six and not
    five: the armband doubles whoever is best, and at five the two answers
    came out dead level at 40.0 apiece.

    The budget is what makes it a choice. Fifteen filler costs 90.0, so one
    special fits at 96.0 and two do not at 102.0. Without that the solver
    simply bought both and the two answers came out identical — as they did
    on `_universe()`, whose six clubs of identical players leave the
    three-per-club cap to force the squad across five of them, with nothing
    left to get wrong.
    """
    rows, n = [], 0
    for club in "ABCDEFGH":
        for pos, k in (("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
            for _ in range(k):
                n += 1
                rows.append(dict(id=str(n), name=f"{club}{pos}{n}", pos=pos,
                                 team=club, price=6.0, ep=1.0,
                                 match_minutes=[90, 90, 90]))
    hop = dict(id="hop", name="HOP", pos="MID", team="G", price=12.0, ep=9.0,
               match_minutes=[90, 90, 90])
    keep = dict(id="keep", name="KEEP", pos="MID", team="H", price=12.0,
                ep=6.0, match_minutes=[90, 90, 90])
    rows += [hop, keep]
    ep1 = {r["id"]: r["ep"] for r in rows}
    ep2 = {r["id"]: (0.0 if r["id"] == "hop" else r["ep"]) for r in rows}
    return rows, {1: ep1, 2: ep2}


def test_a_player_bought_for_one_matchday_is_the_wrong_buy_for_the_window():
    """The whole point of a horizon."""
    from gaffer.ucl import plan
    rows, by_md = _two_specials()
    single = plan.optimise([dict(r, ep=by_md[1][r["id"]]) for r in rows],
                           time_limit=60)
    window = plan.optimise_window(rows, by_md, time_limit=120, prune=None)
    assert window["optimal"]
    assert {r["id"] for r in single["squad"]} & {"hop", "keep"} == {"hop"}
    assert {r["id"] for r in window["squad"]} & {"hop", "keep"} == {"keep"}
    mine = sum(window["by_matchday"].values())
    theirs = _over_window([r["id"] for r in single["squad"]], by_md, rows)
    assert mine > theirs


def test_the_eleven_moves_between_matchdays_and_the_fifteen_does_not():
    from gaffer.ucl import plan
    base = _universe()
    ep1 = {r["id"]: r["ep"] * (1.6 if r["team"] == "B" else 1.0) for r in base}
    ep2 = {r["id"]: r["ep"] * (1.6 if r["team"] == "E" else 1.0) for r in base}
    out = plan.optimise_window(base, {1: ep1, 2: ep2}, time_limit=90,
                               prune=None)
    assert len(out["squad"]) == 15
    assert len(out["xi"][1]) == len(out["xi"][2]) == 11
    assert ({r["id"] for r in out["xi"][1]}
            != {r["id"] for r in out["xi"][2]})
    assert out["captain"][1] is not None and out["captain"][2] is not None


def test_the_club_cap_is_taken_from_the_earliest_matchday_in_the_window():
    """It only rises as the field narrows, so the earliest one binds and a
    squad legal then stays legal. Reading it off the last matchday would
    build a knockout squad and play it in the league phase."""
    from gaffer.ucl import plan
    base = _universe()
    ep = {r["id"]: r["ep"] for r in base}
    league = plan.optimise_window(base, {7: ep, 8: ep}, time_limit=90,
                                  prune=None)
    assert league["club_cap"] == 3
    knockout = plan.optimise_window(base, {13: ep, 14: ep}, time_limit=90,
                                    prune=None)
    assert knockout["club_cap"] == 5
    perclub = {}
    for r in league["squad"]:
        perclub[r["team"]] = perclub.get(r["team"], 0) + 1
    assert max(perclub.values()) <= 3


def test_pruning_never_throws_away_a_player_you_own_or_have_locked():
    """A pruned squad member reads as a player who does not exist, and the
    transfer count then counts a sale the engine never made."""
    from gaffer.ucl import plan
    base = _universe()
    ep = {r["id"]: r["ep"] for r in base}
    worst = [r["id"] for r in sorted(base, key=lambda r: r["ep"])[:3]]
    out = plan.optimise_window(base, {1: ep, 2: ep},
                               owned_ids=_cheap_squad(base), bank=60.0,
                               locked=worst[:1], time_limit=90, prune=20)
    assert out["universe"] >= 15
    assert worst[0] in {r["id"] for r in out["squad"]}


def test_a_window_bench_weight_of_zero_reproduces_the_eleven_only_solve():
    from gaffer.ucl import plan
    base = _universe()
    ep = {r["id"]: r["ep"] for r in base}
    a = plan.optimise_window(base, {1: ep}, bench_weight=0, time_limit=60,
                             prune=None)
    b = plan.optimise_window(base, {1: ep},
                             bench_weight={"GK": 0.0, "OUT": 0.0},
                             time_limit=60, prune=None)
    assert sorted(r["id"] for r in a["squad"]) == \
        sorted(r["id"] for r in b["squad"])
