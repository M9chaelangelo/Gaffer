"""Finishing first is not the same question as scoring most."""
from __future__ import annotations

import numpy as np
import pytest

from gaffer import position


def test_a_squad_is_replayed_through_its_own_multipliers():
    """Bench scores nothing, the captain scores twice, and a Triple Captain
    three times — their lineup, not an assumption about it."""
    samples = np.array([[10.0, 4.0, 2.0]])          # one draw, three players
    columns = {"1": 0, "2": 1, "3": 2}
    squads = [{"picks": [{"element": 1, "multiplier": 2},
                         {"element": 2, "multiplier": 1},
                         {"element": 3, "multiplier": 0}]}]
    got, missing = position.squad_scores(squads, samples, columns)
    assert got[0, 0] == pytest.approx(10 * 2 + 4 * 1)
    assert missing == 0

    squads[0]["picks"][0]["multiplier"] = 3
    got, _ = position.squad_scores(squads, samples, columns)
    assert got[0, 0] == pytest.approx(10 * 3 + 4)


def test_a_pick_we_did_not_simulate_is_counted_not_concealed():
    samples = np.array([[5.0]])
    squads = [{"picks": [{"element": 1, "multiplier": 1},
                         {"element": 99, "multiplier": 1}]}]
    got, missing = position.squad_scores(squads, samples, {"1": 0})
    assert got[0, 0] == pytest.approx(5.0)
    assert missing == 1, "a silently dropped pick is a silently wrong rival"


def test_a_tie_does_not_count_as_finishing_first():
    mine = np.array([50.0, 50.0])
    theirs = np.array([[50.0], [49.0]])
    got = position.finishing_order(mine, theirs, 0, [0])
    assert got["p_first"] == pytest.approx(0.5)


def test_being_behind_makes_first_place_unlikely_and_the_sim_says_so():
    rng = np.random.default_rng(0)
    n = 4000
    mine = rng.normal(55, 18, n)
    theirs = rng.normal(55, 18, (n, 1))
    far = position.finishing_order(mine, theirs, 0, [132])      # 132 behind
    level = position.finishing_order(mine, theirs, 0, [0])
    assert far["p_first"] < 0.01
    assert level["p_first"] == pytest.approx(0.5, abs=0.05)


# --- the point of the module -------------------------------------------------

def test_a_differential_can_lose_points_and_still_win_more_leagues():
    """The case the points objective cannot see.

    Template: the same captain as everyone, so his hauls move the whole
    league and change nothing. Differential: fewer expected points, but
    uncorrelated with the rivals — which is the only thing that reorders a
    table you are bottom of.
    """
    rng = np.random.default_rng(3)
    n = 20000
    common = rng.normal(50, 20, n)                 # the league's shared week

    rival_gw = common[:, None] + rng.normal(0, 3, (n, 5))
    template = common + rng.normal(0, 3, n) + 2.0  # +2 points, fully shared
    differential = rng.normal(50, 20, n)           # same mean, independent

    rows = position.compare(
        {"template": template, "differential": differential},
        rival_gw, my_total=0, rival_totals=[40] * 5)

    by = {r["option"]: r for r in rows}
    assert by["template"]["mean_points"] > by["differential"]["mean_points"]
    assert by["differential"]["p_first_now"] > by["template"]["p_first_now"]
    assert rows[0]["option"] == "differential", "ranked by place, not points"

    d = position.disagreement(rows)
    assert d is not None
    assert d["points_pick"] == "template" and d["place_pick"] == "differential"
    assert d["points_cost"] > 0 and d["place_gain"] > 0


def test_no_disagreement_is_reported_when_the_two_objectives_agree():
    rng = np.random.default_rng(5)
    n = 3000
    rival_gw = rng.normal(50, 15, (n, 4))
    good = rng.normal(70, 15, n)
    bad = rng.normal(40, 15, n)
    rows = position.compare({"good": good, "bad": bad}, rival_gw, 0, [0] * 4)
    assert rows[0]["option"] == "good"
    assert position.disagreement(rows) is None


# --- the season ---------------------------------------------------------------

def test_a_deficit_matters_less_the_more_gameweeks_remain():
    rng = np.random.default_rng(11)
    n = 6000
    mine = rng.normal(55, 18, n)
    theirs = rng.normal(55, 18, (n, 3))
    short = position.project_season(mine, theirs, 0, [60] * 3, remaining=1)
    long = position.project_season(mine, theirs, 0, [60] * 3, remaining=25)
    assert long["p_first"] > short["p_first"]
    assert long["remaining"] == 25


def test_shared_variance_is_what_stops_a_deficit_being_clawed_back():
    """Treating gameweeks as independent across managers would say a
    132-point gap closes easily. It does not, because most of a gameweek's
    movement is the whole league rising together on the same captain."""
    rng = np.random.default_rng(13)
    n = 6000
    mine = rng.normal(55, 18, n)
    theirs = rng.normal(55, 18, (n, 3))
    correlated = position.project_season(mine, theirs, 0, [100] * 3,
                                         remaining=20, shared=0.9)
    independent = position.project_season(mine, theirs, 0, [100] * 3,
                                          remaining=20, shared=0.0)
    assert independent["p_first"] > correlated["p_first"]


def test_from_far_behind_variance_helps_and_from_far_ahead_it_hurts():
    """The crossover is the whole finding, and it is not a slogan.

    Measured on league 88825 with 132 points to make up and 31 gameweeks
    left: with no edge at all, going from an 18-point weekly spread to a
    40-point one takes P(first) from 0.09% to 9.93%. At +8 a week it runs
    the other way, 58.97% down to 53.57% — once you are good enough to win
    on merit, noise is the enemy.
    """
    table = position.edge_variance_table(
        my_total=320, rival_totals=[452] + [400] * 40, remaining=31,
        edges=(0, 8), sds=(18, 40), draws=8000)
    behind = next(r for r in table if r["edge"] == 0)
    ahead = next(r for r in table if r["edge"] == 8)
    assert behind["sd40"] > behind["sd18"] * 5, "variance must pay from behind"
    assert ahead["sd40"] < ahead["sd18"], "and cost from in front"


def test_the_table_is_shaped_the_way_it_is_printed():
    table = position.edge_variance_table(
        my_total=320, rival_totals=[452] * 5, remaining=5,
        edges=(0, 4), sds=(18, 32), draws=2000)
    assert [r["edge"] for r in table] == [0, 4]
    assert set(table[0]) == {"edge", "sd18", "sd32"}
    assert all(0.0 <= v <= 1.0 for r in table for k, v in r.items() if k != "edge")
