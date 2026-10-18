"""Correlated sampling: exact marginals, real dependence between squads."""
from __future__ import annotations

import numpy as np
import pytest

from gaffer import decide, distribution, position


def _card(pid, team, pos="MID", ep=5.0, home=True, **kw):
    c = {"id": pid, "name": f"P{pid}", "team": team, "pos": pos, "home": home,
         "fixture_key": kw.pop("fixture_key", team),
         "ep_next": ep, "start_prob": 0.9, "exp_minutes": 85,
         "xg90": 0.35, "xa90": 0.2, "cs_prob": 0.3, "defcon_prob": 0.2,
         "price": 6.0}
    c.update(kw)
    return c


def test_the_marginal_distribution_is_unchanged_by_the_copula():
    """The whole point of inverting each player's own PMF: correlation is
    added without touching what any single player is expected to score, so
    the simulator cannot disagree with the number the rest of the app
    quotes."""
    cards = [_card(1, "BHA"), _card(2, "LIV", home=False, fixture_key="BHA")]
    s = decide.sample_players(cards, n=40000, rho=0.6,
                              rng=np.random.default_rng(1))
    for j, c in enumerate(cards):
        pmf = distribution.player_points(c)
        assert s[:, j].mean() == pytest.approx(distribution.mean(pmf), abs=0.15)
        assert s[:, j].std() == pytest.approx(distribution.sd(pmf), abs=0.15)


def test_rho_zero_reproduces_independence():
    cards = [_card(1, "BHA"), _card(2, "BHA")]
    s = decide.sample_players(cards, n=30000, rho=0.0,
                              rng=np.random.default_rng(2))
    assert abs(np.corrcoef(s[:, 0], s[:, 1])[0, 1]) < 0.05


def test_team_mates_move_together_and_opponents_move_apart():
    cards = [_card(1, "BHA", fixture_key="BHA-LIV", home=True),
             _card(2, "BHA", fixture_key="BHA-LIV", home=True),
             _card(3, "LIV", fixture_key="BHA-LIV", home=False)]
    s = decide.sample_players(cards, n=40000, rho=0.6,
                              rng=np.random.default_rng(3))
    mates = np.corrcoef(s[:, 0], s[:, 1])[0, 1]
    foes = np.corrcoef(s[:, 0], s[:, 2])[0, 1]
    assert mates > 0.15, "two Brighton players share a Brighton afternoon"
    assert foes < -0.05, "the week Brighton keep a clean sheet is not Liverpool's"


def test_players_in_unrelated_fixtures_are_uncorrelated():
    cards = [_card(1, "BHA", fixture_key="BHA-LIV"),
             _card(2, "ARS", fixture_key="ARS-EVE")]
    s = decide.sample_players(cards, n=30000, rho=0.6,
                              rng=np.random.default_rng(4))
    assert abs(np.corrcoef(s[:, 0], s[:, 1])[0, 1]) < 0.05


def test_a_player_with_no_distribution_scores_zero_and_keeps_his_column():
    """Dropping him would shift every column after him, and a squad indexed
    by position would then be someone else's squad."""
    broken = {"id": 9, "team": "X", "home": True}
    cards = [_card(1, "BHA"), broken, _card(3, "ARS")]
    s = decide.sample_players(cards, n=500, rng=np.random.default_rng(5))
    assert s.shape[1] == 3
    assert s[:, 1].max() == 0
    assert s[:, 0].mean() > 0 and s[:, 2].mean() > 0


# --- squads ------------------------------------------------------------------

def test_a_captain_counts_twice_and_a_bench_player_not_at_all():
    s = np.array([[4.0, 7.0, 2.0]])
    cols = {"1": 0, "2": 1, "3": 2}
    assert decide.squad_totals(s, cols, [1, 2], captain=2)[0] == pytest.approx(4 + 14)
    assert decide.squad_totals(s, cols, {1: 1, 2: 2, 3: 0})[0] == pytest.approx(4 + 14)


def test_a_pick_outside_the_sample_is_skipped_not_crashed():
    s = np.array([[4.0]])
    assert decide.squad_totals(s, {"1": 0}, [1, 99])[0] == pytest.approx(4.0)


# --- the point ----------------------------------------------------------------

def test_correlation_is_what_decides_whether_a_differential_is_worth_it():
    """With everyone's week independent, a template pick and a differential
    of equal mean look identical. They are not, and the copula is what makes
    the difference visible."""
    rng = np.random.default_rng(7)
    n = 20000
    shared = rng.normal(50, 18, n)
    rival_gw = shared[:, None] + rng.normal(0, 3, (n, 6))

    template = shared + rng.normal(0, 3, n) + 1.5     # better mean, no leverage
    differential = rng.normal(51.5 - 1.5, 18, n)      # same mean, own week

    rows = position.compare({"template": template, "differential": differential},
                            rival_gw, my_total=0, rival_totals=[35] * 6)
    by = {r["option"]: r for r in rows}
    assert by["template"]["mean_points"] > by["differential"]["mean_points"]
    assert by["differential"]["p_first_now"] > by["template"]["p_first_now"]


def test_sensitivity_reports_whether_the_answer_turns_on_rho():
    rng = np.random.default_rng(11)
    n = 4000
    rival_gw = rng.normal(50, 15, (n, 5))

    def build(rho):
        r = np.random.default_rng(int(rho * 100) + 1)
        return {"safe": r.normal(52, 10, n), "wild": r.normal(50, 30, n)}

    # A deficit one gameweek can plausibly bridge. The first version of
    # this test used 120 points, where both options are ~0% and "best" is
    # whichever way the noise fell — a test that asserts on a coin toss.
    got = decide.sensitivity(build, rival_gw, 380, [420] * 5,
                             rhos=(0.0, 0.45, 0.8))
    assert [g["rho"] for g in got] == [0.0, 0.45, 0.8]
    assert all(g["best"] == "wild" for g in got), (
        "40 behind with one week left, the wide option is the only one that "
        "ever gets there")
    assert all(g["p_first"] > 0.02 for g in got), "and it is a live chance"
