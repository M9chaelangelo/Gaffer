"""Score the strength model against the only thing that can score it.

`gaffer/ucl/strength.py` carries every lambda in the UCL projection: the
runner cannot reach a bookmaker, so the fallback is not a fallback, it is the
model. Nothing had ever measured it. These tests do, against the devigged
sharp market for all eighteen matchday-3 fixtures.

The panel is a fixture and not a fit. Nothing in strength.py was tuned on it
— a sweep of the clamps and the shrinkage did improve the in-sample error and
then FAILED leave-one-fixture-out, which is what overfitting thirty-six points
with four dials looks like. The one change that survives is the league mean,
and its justification comes from outside the panel entirely.
"""
from __future__ import annotations

import json
import math
import os

import pytest

from gaffer.ucl import main as uclmain, strength

PANEL = os.path.join(os.path.dirname(__file__), "fixtures",
                     "ucl_market_md3.json")


@pytest.fixture(scope="module")
def market():
    with open(PANEL, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def pool():
    rows, _, _, _ = uclmain.load_snapshot(3, offline=True)
    return rows


def _errors(pool, fixtures, **kw):
    rate, mu = strength.ratings(pool)
    if kw:
        live = [t for t in strength.aggregate(pool).values() if t["played"]]
        mu = strength.league_mean(live, **kw)
    out = []
    for f in fixtures:
        lh, la = strength.lambdas(rate, f["home"], f["away"], mu)
        out += [(f["home_xg"], lh), (f["away_xg"], la)]
    return out


def _mae(pairs):
    return sum(abs(b - a) for a, b in pairs) / len(pairs)


def _bias(pairs):
    return sum(b - a for a, b in pairs) / len(pairs)


# --- the panel itself --------------------------------------------------------

def test_the_panel_covers_the_whole_matchday_and_joins_to_the_feed(market, pool):
    assert len(market["fixtures"]) == 18
    teams = {r["team"] for r in pool}
    for f in market["fixtures"]:
        assert f["home"] in teams, f["home"]
        assert f["away"] in teams, f["away"]
    assert "Pinnacle" in market["_source"]


def test_the_market_expects_fewer_goals_than_the_season_has_produced(market, pool):
    """The fact the fix rests on. Matchdays 1-2 produced 1.97 goals a team;
    the market's forward view for matchday 3 is 1.78."""
    live = [t for t in strength.aggregate(pool).values() if t["played"]]
    raw = strength.league_mean(live, prior_games=0)
    mkt = sum(v for f in market["fixtures"]
              for v in (f["home_xg"], f["away_xg"])) / 36
    assert raw > mkt + 0.1
    assert 1.7 < mkt < 1.9


# --- the model ---------------------------------------------------------------

def test_the_engine_tracks_the_market_closely_enough_to_plan_on(market, pool):
    pairs = _errors(pool, market["fixtures"])
    assert _mae(pairs) < 0.36, (
        "strength.py has drifted away from the sharp market; the whole UCL "
        "projection rests on it")


def test_it_beats_giving_every_team_the_league_average(market, pool):
    """The bar a team-strength model has to clear to be worth having."""
    pairs = _errors(pool, market["fixtures"])
    _, mu = strength.ratings(pool)
    flat = sum(abs(a - mu) for a, _ in pairs) / len(pairs)
    assert _mae(pairs) < 0.6 * flat


def test_it_ranks_fixtures_the_way_the_market_does(market, pool):
    pairs = _errors(pool, market["fixtures"])
    ma = sum(a for a, _ in pairs) / len(pairs)
    mb = sum(b for _, b in pairs) / len(pairs)
    num = sum((a - ma) * (b - mb) for a, b in pairs)
    den = math.sqrt(sum((a - ma) ** 2 for a, _ in pairs)
                    * sum((b - mb) ** 2 for _, b in pairs))
    assert num / den > 0.85


def test_shrinking_the_league_mean_is_what_took_the_bias_out(market, pool):
    """The specific change, measured against the panel it was not fitted to."""
    now = _errors(pool, market["fixtures"])
    before = _errors(pool, market["fixtures"], prior_games=0)
    assert _bias(before) > 0.2
    assert 0 < _bias(now) < _bias(before)
    assert _mae(now) < _mae(before)


def test_the_old_behaviour_is_still_reachable_for_comparison(pool):
    live = [t for t in strength.aggregate(pool).values() if t["played"]]
    raw = sum(t["scored"] for t in live) / sum(t["played"] for t in live)
    assert strength.league_mean(live, prior_games=0) == pytest.approx(raw)
    assert strength.league_mean(live) < raw


def test_an_empty_season_falls_back_to_the_prior_rather_than_dividing_by_zero():
    assert strength.league_mean([]) == pytest.approx(strength.MU_PRIOR)


# --- what is still wrong, recorded rather than hidden -----------------------

def test_the_engine_cannot_yet_express_a_genuinely_weak_away_side(market, pool):
    """A known, measured shortfall, pinned so that fixing it is visible.

    The market puts AEK Athens at Manchester City on 0.6 goals. The engine
    cannot go below about 1.0, because two matchdays of a team's own scoring
    carry only a quarter of the weight and the rest comes from a squad-value
    prior that is clamped. Lowering the shrinkage improves the in-sample
    error and fails leave-one-fixture-out, so it is not changed here.
    """
    rate, mu = strength.ratings(pool)
    lows = sorted(strength.lambdas(rate, f["home"], f["away"], mu)[1]
                  for f in market["fixtures"])
    assert min(lows) > 0.8, "fixed? then the comment above is out of date"
    market_low = min(f["away_xg"] for f in market["fixtures"])
    assert market_low < min(lows) - 0.2


# --- clean sheets: the other half of the panel ------------------------------

def _cs_pairs(pool, fixtures, use_market_lambdas=False):
    """(market clean-sheet odds, modelled) for both sides of every fixture.

    A side keeps a clean sheet when the OTHER side fails to score, so the
    modelled figure is exp(-lambda) on the opponent's expected goals.
    """
    rate, mu = strength.ratings(pool)
    out = []
    for f in fixtures:
        lh, la = strength.lambdas(rate, f["home"], f["away"], mu)
        if use_market_lambdas:
            lh, la = f["home_xg"], f["away_xg"]
        out += [(f["home_cs"], math.exp(-la)), (f["away_cs"], math.exp(-lh))]
    return out


def test_the_clean_sheet_model_is_not_the_problem(market, pool):
    """A negative result worth keeping, because it says where NOT to look.

    Feed the market's own expected goals into exp(-lambda) and the market's
    own clean-sheet percentages come back to within half a point: MAE 0.005,
    bias -0.0001 over thirty-six sides. The independent Poisson is exactly
    the relationship the market is using, so there is no Dixon-Coles
    correction owed here and no reason to touch the simulator. Every point
    of clean-sheet error the engine has is inherited from its lambdas.
    """
    pairs = _cs_pairs(pool, market["fixtures"], use_market_lambdas=True)
    assert _mae(pairs) < 0.01
    assert abs(_bias(pairs)) < 0.005


def test_clean_sheets_are_understated_by_exactly_as_much_as_goals_are_over(
        market, pool):
    """And so they are one defect, not two. Every defender and goalkeeper
    pick rests on this number, and the engine has it 4.5 points low because
    it still expects 0.18 goals a side too many."""
    cs = _cs_pairs(pool, market["fixtures"])
    goals = _errors(pool, market["fixtures"])
    assert _bias(goals) > 0, "goals over-predicted"
    assert _bias(cs) < 0, "clean sheets under-predicted"
    assert _mae(cs) < 0.08
    # Same direction, same cause: fixing the lambdas fixes both.
    assert abs(_bias(cs)) < 0.07


def test_the_engine_ranks_clean_sheets_the_way_the_market_does(market, pool):
    pairs = _cs_pairs(pool, market["fixtures"])
    ma = sum(a for a, _ in pairs) / len(pairs)
    mb = sum(b for _, b in pairs) / len(pairs)
    num = sum((a - ma) * (b - mb) for a, b in pairs)
    den = math.sqrt(sum((a - ma) ** 2 for a, _ in pairs)
                    * sum((b - mb) ** 2 for _, b in pairs))
    assert num / den > 0.85
    flat = sum(abs(a - ma) for a, _ in pairs) / len(pairs)
    assert _mae(pairs) < 0.6 * flat


# --- typing the card in -----------------------------------------------------

def test_expected_goals_typed_from_a_card_beat_the_model(market, pool):
    """The whole point. Scored against the market the model is 0.33 goals
    out; the card is the market, so a fixture with a card entry should come
    back exactly as typed."""
    from gaffer.ucl import project
    f = market["fixtures"][3]
    fx = [{"home": f["home"], "away": f["away"], "kickoff": "2026-10-20T20:00:00Z"}]
    table = {f"{f['home']} v {f['away']}": [f["home_xg"], f["away_xg"]]}
    rows, _, _ = project.fixture_rates(fx, pool, xg=table)
    assert rows[0]["source"] == "market xg"
    assert rows[0]["lh"] == pytest.approx(f["home_xg"])
    assert rows[0]["la"] == pytest.approx(f["away_xg"])


def test_a_fixture_typed_the_wrong_way_round_is_turned_around(market, pool):
    from gaffer.ucl import project
    f = market["fixtures"][3]
    fx = [{"home": f["home"], "away": f["away"], "kickoff": "2026-10-20T20:00:00Z"}]
    table = {f"{f['away']} v {f['home']}": [f["away_xg"], f["home_xg"]]}
    rows, _, _ = project.fixture_rates(fx, pool, xg=table)
    assert rows[0]["lh"] == pytest.approx(f["home_xg"])
    assert rows[0]["la"] == pytest.approx(f["away_xg"])


def test_a_fixture_with_no_card_entry_still_falls_back_to_the_model(market, pool):
    from gaffer.ucl import project
    fx = [{"home": f["home"], "away": f["away"],
           "kickoff": "2026-10-20T20:00:00Z"} for f in market["fixtures"][:3]]
    one = market["fixtures"][0]
    table = {f"{one['home']} v {one['away']}": [one["home_xg"], one["away_xg"]]}
    rows, _, _ = project.fixture_rates(fx, pool, xg=table)
    assert [r["source"] for r in rows] == ["market xg", "model", "model"]


def test_a_mistyped_fixture_name_is_reported_rather_than_ignored(market, pool):
    """Silently dropping it would leave the model in charge of a fixture the
    manager believes he has priced."""
    from gaffer.ucl import project
    fx = [{"home": f["home"], "away": f["away"],
           "kickoff": "2026-10-20T20:00:00Z"} for f in market["fixtures"]]
    table = {"Man Shitty v AEK Athens": [3.3, 0.6],
             "Man City v AEK Athens": [3.3, 0.6]}
    assert project.unmatched_xg(table, fx) == ["Man Shitty v AEK Athens"]


def test_the_whole_card_drives_every_fixture_when_it_is_typed_in(market, pool):
    """With all eighteen entered the engine reproduces the market exactly,
    which is the error floor nothing else can reach."""
    from gaffer.ucl import project
    fx = [{"home": f["home"], "away": f["away"],
           "kickoff": "2026-10-20T20:00:00Z"} for f in market["fixtures"]]
    table = {f"{f['home']} v {f['away']}": [f["home_xg"], f["away_xg"]]
             for f in market["fixtures"]}
    assert project.unmatched_xg(table, fx) == []
    rows, _, _ = project.fixture_rates(fx, pool, xg=table)
    assert all(r["source"] == "market xg" for r in rows)
    err = [abs(r["lh"] - f["home_xg"]) + abs(r["la"] - f["away_xg"])
           for r, f in zip(rows, market["fixtures"])]
    assert max(err) < 1e-9
