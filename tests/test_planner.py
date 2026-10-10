"""Tests for the plan evaluator.

Everything here runs on a synthetic `proj` dict — no network, no FPL API, no
solver. That is deliberate: the rules the evaluator enforces are exactly the
ones that are expensive to get wrong and impossible to check by eye at 19:29
on a Friday.
"""
import pytest

from gaffer.planner import (
    CHIP_NAMES, MAX_BANKED_FT, SQUAD,
    best_xi, chip_windows, evaluate, from_config, grade, resolve, sweep_chip,
)

# The eleven that should start in the synthetic world below is 1 GKP, 5 DEF,
# 4 MID, 1 FWD — so ids 2 (second keeper), 12 (fifth mid) and two forwards sit.
STARTERS = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]
BENCH = [2, 12, 14, 15]


def make_proj(gws=(5, 6, 7), starter_ep=4.0, bench_ep=4.0, market_ep=4.0):
    """Fifteen owned players (ids 1-15) plus a market (ids 100+)."""
    proj = {}
    pid = 1
    for pos, n in SQUAD.items():
        for _ in range(n):
            ep = bench_ep if pid in BENCH else starter_ep
            proj[pid] = {
                "id": pid, "name": f"P{pid}", "pos": pos,
                "team": f"T{pid}", "price": 5.0,
                "ep": {g: ep for g in gws},
                "exp_min": 85, "status": "a", "own": 5.0, "news": "",
            }
            pid += 1

    mid = 100
    for pos in SQUAD:
        for k in range(4):
            proj[mid] = {
                "id": mid, "name": f"M{mid}", "pos": pos,
                "team": f"X{k}", "price": 5.0,
                "ep": {g: market_ep for g in gws},
                "exp_min": 85, "status": "a", "own": 5.0, "news": "",
            }
            mid += 1
    return proj


def squad_ids():
    return list(range(1, 16))


def steps(gws, per_gw=None):
    """A plan skeleton. Integer gameweeks cannot be keyword arguments, so the
    per-week overrides come in as a plain dict."""
    per_gw = per_gw or {}
    return {g: per_gw.get(g, {}) for g in gws}


# --- the eleven ---------------------------------------------------------


def test_best_xi_picks_a_legal_shape():
    proj = make_proj()
    r = best_xi(squad_ids(), proj, 5)
    assert len(r["xi"]) == 11
    assert len(r["bench"]) == 4
    shape = {}
    for pid in r["xi"]:
        shape[proj[pid]["pos"]] = shape.get(proj[pid]["pos"], 0) + 1
    assert shape["GKP"] == 1
    assert shape["DEF"] >= 3
    assert shape["MID"] >= 2
    assert shape["FWD"] >= 1


def test_best_xi_benches_the_reserve_keeper_last():
    proj = make_proj()
    r = best_xi(squad_ids(), proj, 5)
    assert proj[r["bench"][-1]]["pos"] == "GKP"


def test_best_xi_beats_greedy_when_greedy_would_corner_itself():
    # Five brilliant forwards, dreadful defenders. A greedy "take the best
    # eleven" grabs the forwards and then cannot field three defenders.
    proj = make_proj()
    for pid, p in proj.items():
        if p["pos"] == "FWD":
            p["ep"] = {g: 12.0 for g in p["ep"]}
        if p["pos"] == "DEF":
            p["ep"] = {g: 0.5 for g in p["ep"]}
    r = best_xi(squad_ids(), proj, 5)
    shape = {}
    for pid in r["xi"]:
        shape[proj[pid]["pos"]] = shape.get(proj[pid]["pos"], 0) + 1
    assert shape["DEF"] >= 3
    assert shape["FWD"] <= 3


# --- a clean plan -------------------------------------------------------


def test_empty_plan_is_legal_and_scores():
    gws = [5, 6, 7]
    proj = make_proj(gws)
    r = evaluate(steps(gws), proj, squad_ids(), bank=0.0, first_ft=1, gws=gws)
    assert r.legal
    assert len(r.weeks) == 3
    # Eleven starters at 4.0 plus the captain counted twice.
    assert r.weeks[0].ep == pytest.approx(48.0)
    assert r.total_hits == 0


def test_free_transfers_stack_to_five_and_stop():
    gws = list(range(5, 13))
    proj = make_proj(gws)
    r = evaluate(steps(gws), proj, squad_ids(), 0.0, 1, gws)
    free = [w.free_transfers for w in r.weeks]
    assert free[0] == 1
    assert free[1] == 2
    assert max(free) <= MAX_BANKED_FT
    assert free[-1] == MAX_BANKED_FT


def test_a_transfer_spends_a_free_one():
    gws = [5, 6, 7]
    proj = make_proj(gws)
    plan = steps(gws, {5: {"out": [8], "in": [200], "chip": None}})
    proj[200] = dict(proj[108], id=200, name="New", pos="MID")
    r = evaluate(plan, proj, squad_ids(), 0.0, 1, gws)
    assert r.weeks[0].transfers == 1
    assert r.weeks[0].hit == 0
    # Spent the one, earned one back.
    assert r.weeks[1].free_transfers == 1


def test_second_transfer_in_a_week_costs_four():
    gws = [5, 6, 7]
    proj = make_proj(gws)
    plan = steps(gws, {5: {"out": [8, 9], "in": [108, 109], "chip": None}})
    r = evaluate(plan, proj, squad_ids(), 0.0, 1, gws)
    assert r.weeks[0].transfers == 2
    assert r.weeks[0].hit == -4
    assert r.total_hits == -4


def test_selling_price_is_used_not_list_price():
    gws = [5, 6, 7]
    proj = make_proj(gws)
    proj[8]["price"] = 8.0
    # He rose; FPL lets you keep half, so he sells for 7.6 not 8.0.
    plan = steps(gws, {5: {"out": [8], "in": [108], "chip": None}})
    r = evaluate(plan, proj, squad_ids(), 0.0, 1, gws, sell={8: 7.6})
    # Sold for 7.6, bought at 5.0.
    assert r.weeks[0].bank == pytest.approx(2.6)


# --- the chips ----------------------------------------------------------


def test_wildcard_makes_transfers_free_and_keeps_the_squad():
    gws = [5, 6, 7]
    proj = make_proj(gws)
    plan = steps(gws, {5: {
        "out": [3, 4, 5], "in": [104, 105, 106], "chip": "wc"}})
    r = evaluate(plan, proj, squad_ids(), 0.0, 1, gws)
    assert r.weeks[0].hit == 0
    assert r.weeks[0].transfers == 3
    # Kept, not rented.
    assert 104 in r.weeks[1].squad


def test_wildcard_neither_spends_nor_earns_a_free_transfer():
    gws = [5, 6, 7]
    proj = make_proj(gws)
    plan = steps(gws, {5: {"out": [3], "in": [104], "chip": "wc"}})
    r = evaluate(plan, proj, squad_ids(), 0.0, 3, gws)
    # Held three going in, still three coming out.
    assert r.weeks[1].free_transfers == 3


def test_free_hit_reverts_the_squad():
    gws = [5, 6, 7]
    proj = make_proj(gws)
    plan = steps(gws, {5: {"out": [8], "in": [108], "chip": "fh"}})
    r = evaluate(plan, proj, squad_ids(), 0.0, 1, gws)
    assert 108 in r.weeks[0].squad
    assert 108 not in r.weeks[1].squad
    assert 8 in r.weeks[1].squad


def test_bench_boost_adds_exactly_the_bench():
    gws = [5, 6, 7]
    proj = make_proj(gws, starter_ep=4.0, bench_ep=3.0)
    base = evaluate(steps(gws), proj, squad_ids(), 0.0, 1, gws)
    assert sorted(base.weeks[0].bench) == sorted(BENCH)

    plan = steps(gws, {5: {"chip": "bb"}})
    boosted = evaluate(plan, proj, squad_ids(), 0.0, 1, gws)
    assert boosted.weeks[0].ep - base.weeks[0].ep == pytest.approx(12.0)


def test_triple_captain_adds_one_more_helping():
    gws = [5, 6, 7]
    proj = make_proj(gws)
    base = evaluate(steps(gws), proj, squad_ids(), 0.0, 1, gws)
    plan = steps(gws, {5: {"chip": "tc"}})
    tripled = evaluate(plan, proj, squad_ids(), 0.0, 1, gws)
    assert tripled.weeks[0].ep - base.weeks[0].ep == pytest.approx(4.0)


def test_a_chip_cannot_be_played_twice():
    gws = [5, 6, 7]
    proj = make_proj(gws)
    plan = steps(gws, {5: {"chip": "bb"}, 6: {"chip": "bb"}})
    r = evaluate(plan, proj, squad_ids(), 0.0, 1, gws)
    assert not r.legal
    v = [x for x in r.violations if x.code == "chip-spent"]
    assert v and v[0].gw == 6


def test_an_unknown_chip_is_rejected_by_name():
    gws = [5, 6, 7]
    proj = make_proj(gws)
    plan = steps(gws, {5: {"chip": "limitless"}})
    r = evaluate(plan, proj, squad_ids(), 0.0, 1, gws)
    v = [x for x in r.violations if x.code == "chip-unknown"]
    assert v
    assert "limitless" in v[0].message


def test_first_half_chips_expire_at_gw19():
    windows = chip_windows(list(range(15, 25)))
    for chip in CHIP_NAMES:
        assert max(windows[chip]) == 19


# --- the rules that bite ------------------------------------------------


def test_selling_someone_you_do_not_own():
    gws = [5, 6, 7]
    proj = make_proj(gws)
    plan = steps(gws, {5: {"out": [108], "in": [109], "chip": None}})
    r = evaluate(plan, proj, squad_ids(), 0.0, 1, gws)
    v = [x for x in r.violations if x.code == "not-owned"]
    assert v and v[0].gw == 5


def test_buying_someone_you_already_own():
    gws = [5, 6, 7]
    proj = make_proj(gws)
    plan = steps(gws, {5: {"out": [8], "in": [9], "chip": None}})
    r = evaluate(plan, proj, squad_ids(), 0.0, 1, gws)
    assert [x for x in r.violations if x.code == "already-owned"]


def test_swapping_across_positions_is_refused():
    gws = [5, 6, 7]
    proj = make_proj(gws)
    # id 1 is a keeper, id 112 a forward.
    fwd = next(i for i, p in proj.items() if i >= 100 and p["pos"] == "FWD")
    plan = steps(gws, {5: {"out": [1], "in": [fwd], "chip": None}})
    r = evaluate(plan, proj, squad_ids(), 0.0, 1, gws)
    v = [x for x in r.violations if x.code == "position"]
    assert v
    assert "GKP" in v[0].message and "FWD" in v[0].message


def test_going_over_budget_says_by_how_much():
    gws = [5, 6, 7]
    proj = make_proj(gws)
    rich = next(i for i, p in proj.items() if i >= 100 and p["pos"] == "MID")
    proj[rich]["price"] = 15.0
    plan = steps(gws, {5: {"out": [8], "in": [rich], "chip": None}})
    r = evaluate(plan, proj, squad_ids(), 0.0, 1, gws)
    v = [x for x in r.violations if x.code == "over-budget"]
    assert v
    assert "10.0m over budget" in v[0].message


def test_four_from_one_club_is_refused():
    gws = [5, 6, 7]
    proj = make_proj(gws)
    for pid in (3, 4, 5):
        proj[pid]["team"] = "ARS"
    target = next(i for i, p in proj.items() if i >= 100 and p["pos"] == "DEF")
    proj[target]["team"] = "ARS"
    plan = steps(gws, {5: {"out": [6], "in": [target], "chip": None}})
    r = evaluate(plan, proj, squad_ids(), 0.0, 1, gws)
    v = [x for x in r.violations if x.code == "club-limit"]
    assert v
    assert "4 players from ARS" in v[0].message


def test_unbalanced_transfer_lists_are_caught():
    gws = [5, 6, 7]
    proj = make_proj(gws)
    plan = steps(gws, {5: {"out": [8, 9], "in": [108], "chip": None}})
    r = evaluate(plan, proj, squad_ids(), 0.0, 1, gws)
    assert [x for x in r.violations if x.code == "unbalanced"]


def test_every_violation_is_reported_not_just_the_first():
    gws = [5, 6, 7]
    proj = make_proj(gws)
    plan = steps(gws, {5: {"out": [999], "in": [998], "chip": None},
                    6: {"chip": "bb"},
                    7: {"chip": "bb"}})
    r = evaluate(plan, proj, squad_ids(), 0.0, 1, gws)
    assert len(r.violations) >= 2
    # And it keeps going, so the rest of the plan is still readable.
    assert len(r.weeks) == 3


def test_a_broken_plan_still_reports_weeks():
    gws = [5, 6, 7]
    proj = make_proj(gws)
    plan = steps(gws, {5: {"out": [8], "in": [9], "chip": None}})
    r = evaluate(plan, proj, squad_ids(), 0.0, 1, gws)
    assert not r.legal
    assert len(r.weeks) == 3
    assert all(w.ep > 0 for w in r.weeks)


# --- the sweep ----------------------------------------------------------


def test_sweep_finds_the_week_the_bench_is_best():
    gws = [5, 6, 7]
    proj = make_proj(gws, starter_ep=4.0, bench_ep=1.0)
    for pid in BENCH:
        proj[pid]["ep"][6] = 9.0
    ranked = sweep_chip("bb", steps(gws), proj, squad_ids(), 0.0, 1, gws)
    assert ranked[0]["gw"] == 6
    assert ranked[0]["gain"] > ranked[1]["gain"]


def test_sweep_skips_weeks_another_chip_holds():
    gws = [5, 6, 7]
    proj = make_proj(gws, bench_ep=1.0)
    plan = steps(gws, {6: {"chip": "tc"}})
    ranked = sweep_chip("bb", plan, proj, squad_ids(), 0.0, 1, gws)
    assert 6 not in [r["gw"] for r in ranked]


def test_sweep_reports_a_worthless_chip_as_worthless():
    gws = [5, 6, 7]
    proj = make_proj(gws, bench_ep=0.0)
    ranked = sweep_chip("bb", steps(gws), proj, squad_ids(), 0.0, 1, gws)
    assert all(r["gain"] == 0 for r in ranked)


def test_sweep_will_not_place_a_first_half_chip_after_gw19():
    gws = [18, 19, 20, 21]
    proj = make_proj(gws, bench_ep=8.0)
    ranked = sweep_chip("bb", steps(gws), proj, squad_ids(), 0.0, 1, gws)
    assert [r["gw"] for r in ranked] and max(r["gw"] for r in ranked) <= 19


# --- reading a plan out of config ---------------------------------------


def test_resolve_maps_names_to_ids():
    proj = make_proj()
    ids, missing = resolve(["P1", "P8"], proj)
    assert ids == [1, 8]
    assert missing == []


def test_resolve_reports_a_name_it_cannot_find():
    proj = make_proj()
    ids, missing = resolve(["P1", "Haaland"], proj)
    assert ids == [1]
    assert missing == ["Haaland"]


def test_resolve_flags_ambiguity_rather_than_guessing_silently():
    proj = make_proj()
    proj[500] = dict(proj[8], id=500, name="P8")
    ids, missing = resolve(["P8"], proj)
    assert len(ids) == 1
    assert missing and "ambiguous" in missing[0]


def test_from_config_folds_in_the_locked_chip_plan():
    gws = [5, 6, 7]
    proj = make_proj(gws)
    cfg = {"chip_plan": {"tc": 7}, "my_plan": {5: {"out": ["P8"], "in": ["M108"]}}}
    steps_out, missing = from_config(cfg, proj, gws)
    assert steps_out[5]["out"] == [8]
    assert steps_out[7]["chip"] == "tc"
    assert missing == []


def test_from_config_lets_my_plan_win_over_chip_plan():
    gws = [5, 6, 7]
    proj = make_proj(gws)
    cfg = {"chip_plan": {"bb": 5}, "my_plan": {5: {"chip": "tc"}}}
    steps_out, _ = from_config(cfg, proj, gws)
    assert steps_out[5]["chip"] == "tc"


def test_grade_prices_the_gap_against_the_solver():
    gws = [5, 6, 7]
    proj = make_proj(gws)
    cfg = {"my_plan": {}}
    g = grade(cfg, proj, squad_ids(), 0.0, 1, gws, solver_total=200.0)
    assert g["plan"]["legal"]
    assert g["horizon"] == {"from": 5, "to": 7}
    assert g["gap"] == pytest.approx(g["plan"]["total_ep"] - 200.0)


def test_grade_says_where_a_chip_should_move_to():
    gws = [5, 6, 7]
    proj = make_proj(gws, starter_ep=4.0, bench_ep=1.0)
    for pid in BENCH:
        proj[pid]["ep"][7] = 9.0
    cfg = {"chip_plan": {"bb": 5}}
    g = grade(cfg, proj, squad_ids(), 0.0, 1, gws)
    bb = g["chips"]["bb"]
    assert bb["current_gw"] == 5
    assert bb["best_gw"] == 7
    assert bb["move_gain"] > 0


def test_grade_surfaces_names_it_could_not_resolve():
    gws = [5, 6, 7]
    proj = make_proj(gws)
    cfg = {"my_plan": {5: {"out": ["Nobody"], "in": ["M108"]}}}
    g = grade(cfg, proj, squad_ids(), 0.0, 1, gws)
    assert any("Nobody" in u for u in g["unresolved"])


# --- chips already played ------------------------------------------------
#
# FPL's public API does not expose which chips an entry has spent, so it is
# declared in config. Before `chips_used` was wired through, the sweep offered
# every chip every week regardless — including ones already played, which is a
# recommendation the manager cannot act on.


def test_a_spent_chip_is_not_swept():
    gws = [5, 6, 7]
    proj = make_proj(gws)
    cfg = {"chips_used": ["tc"]}
    g = grade(cfg, proj, squad_ids(), 0.0, 1, gws)
    assert "tc" not in g["chips"], "a chip already played is not a decision"
    assert "bb" in g["chips"], "the chips still held are still swept"
    assert g["chips_used"] == ["tc"]


def test_planning_a_spent_chip_is_a_violation():
    gws = [5, 6, 7]
    proj = make_proj(gws)
    cfg = {"chips_used": ["tc"], "chip_plan": {"tc": 6}}
    g = grade(cfg, proj, squad_ids(), 0.0, 1, gws)
    codes = [v["code"] for v in g["plan"]["violations"]]
    assert "chip-spent" in codes, "a stale chip_plan entry must be reported"


def test_a_typo_in_chips_used_is_reported_not_ignored():
    gws = [5, 6, 7]
    proj = make_proj(gws)
    cfg = {"chips_used": ["triple_captain"]}
    g = grade(cfg, proj, squad_ids(), 0.0, 1, gws)
    assert any("triple_captain" in u for u in g["unresolved"])
    # Silently reading it as "nothing spent" is the failure mode: the sweep
    # would then offer a chip that is gone.
    assert g["chips_used"] == []


def test_a_spent_chip_has_no_window():
    windows = chip_windows(list(range(15, 25)), spent=("wc",))
    assert "wc" not in windows
    assert max(windows["bb"]) == 19


def test_wildcard_and_free_hit_are_declared_unvalued_not_zero():
    """The sweep moves a chip's label, never its transfers.

    So wc and fh gain exactly nothing in every week, and printing that as
    `best_gw` with gain 0.0 turns the first week of a tie into a confident
    recommendation about the single biggest decision in the game.
    """
    gws = [5, 6, 7]
    proj = make_proj(gws)
    g = grade({}, proj, squad_ids(), 0.0, 1, gws)

    for chip in ("wc", "fh"):
        entry = g["chips"][chip]
        assert entry["valued"] is False
        assert entry["best_gw"] is None, "an unvalued chip names no best week"
        assert entry["best_gain"] is None
        assert all(r["gain"] is None for r in entry["ranked"])
        assert "squad" in entry["note"]

    bb = g["chips"]["bb"]
    assert bb["valued"] is True, "bench boost is genuinely priced"
    assert bb["best_gw"] in gws


# --- chips read from the entry's own history ------------------------------


def test_chips_played_maps_fpl_names():
    from gaffer.planner import chips_played
    hist = {"chips": [{"name": "3xc", "event": 4}, {"name": "bboost", "event": 2}]}
    assert chips_played(hist) == ("tc", "bb")


def test_chips_played_drops_names_it_cannot_map():
    from gaffer.planner import chips_played
    # Silently reading an unknown chip as "nothing played" is the dangerous
    # direction, so it must not appear as a known chip either.
    assert chips_played({"chips": [{"name": "manager"}]}) == ()
    assert chips_played({}) == ()
    assert chips_played(None) == ()


def test_reconcile_chips_finds_the_one_you_forgot_to_declare():
    from gaffer.planner import reconcile_chips
    missing, surplus = reconcile_chips(declared=(), played=("tc",))
    assert missing == ("tc",) and surplus == ()

    missing, surplus = reconcile_chips(declared=("tc", "wc"), played=("tc",))
    assert missing == () and surplus == ("wc",)

    assert reconcile_chips(("tc",), ("tc",)) == ((), ())
