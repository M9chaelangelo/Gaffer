"""Tests for the ceiling model wiring.

The model was trained, scored and committed months ago and then loaded by
nothing. These tests exist so that cannot happen again quietly: they assert the
model is reachable, that its prediction survives into the projection, and that
it reaches the two places that consume it — the Triple Captain armband and the
page.
"""
import json

import pytest

from gaffer import explain, learn, projections, webdata


# --- the model itself ---------------------------------------------------


def test_the_trained_ceiling_model_is_committed_and_loadable():
    """If this fails the repo is shipping a model nobody can use."""
    bundle = learn.load("ceiling")
    assert bundle is not None, "models/ceiling.joblib missing"
    assert "model" in bundle and "features" in bundle


def test_its_features_match_what_training_declared():
    bundle = learn.load("ceiling")
    assert list(bundle["features"]) == list(learn.CEILING_X)


def test_it_scored_well_enough_to_be_worth_trusting():
    scores = json.load(open("models/scores.json", encoding="utf-8"))
    assert scores["ceiling"]["metric"] == "AUC"
    # Below chance-plus-a-bit it should not be steering an armband.
    assert scores["ceiling"]["value"] > 0.70


# --- the prediction pass ------------------------------------------------


def _boot(n=6):
    """Players 1..n, odd ids on team 1 and even on team 2.

    Spelled out rather than computed: an expression like `1 + (i % 2)` reads as
    "alternating" and actually puts player 1 on team 2, which is exactly the
    kind of thing a test fixture should not make you work out.
    """
    els = []
    for i in range(1, n + 1):
        els.append({
            "id": i, "web_name": f"P{i}", "team": 1 if i % 2 else 2,
            "element_type": 3, "now_cost": 50 + i,
            "minutes": 400, "starts": 4, "total_points": 20 + i,
            "bps": 100 + i, "threat": str(50 + i), "ict_index": str(40 + i),
            "expected_goal_involvements": str(1.0 + i / 10),
            "status": "a", "news": "", "selected_by_percent": "5.0",
        })
    return {"elements": els,
            "teams": [{"id": 1, "short_name": "ARS"}, {"id": 2, "short_name": "LIV"}],
            "element_types": [{"id": 3, "singular_name_short": "MID"}]}


def _fx(gws):
    # Team 1 home, team 2 away, every gameweek.
    return {1: {g: [(2, True, 2)] for g in gws},
            2: {g: [(2, False, 1)] for g in gws}}


def test_it_predicts_a_probability_for_every_player_and_gameweek():
    gws = [5, 6]
    boot = _boot()
    lams = {1: {g: 1.2 for g in gws}, 2: {g: 1.6 for g in gws}}
    out = projections._ceiling_model(boot, 4, _fx(gws), lams, gws)
    assert out is not None, "model present but produced nothing"
    for e in boot["elements"]:
        for g in gws:
            p = out[e["id"]][g]
            assert 0.0 <= p <= 1.0


def test_a_double_gameweek_combines_the_two_chances():
    """Two fixtures in one week is P(haul in either), not the last one seen."""
    gws = [5]
    boot = _boot(2)
    fx = {1: {5: [(2, True, 2), (3, False, 2)]}, 2: {5: [(2, False, 1)]}}
    lams = {1: {5: 1.2}, 2: {5: 1.4}}
    double = projections._ceiling_model(boot, 4, fx, lams, gws)

    single = projections._ceiling_model(
        boot, 4, {1: {5: [(2, True, 2)]}, 2: {5: [(2, False, 1)]}}, lams, gws)

    # Player 1 is on team 1 and plays twice; his haul chance must rise.
    assert double[1][5] > single[1][5]
    assert double[1][5] <= 1.0


def test_a_player_with_no_fixture_gets_no_entry():
    gws = [5]
    boot = _boot(2)
    fx = {1: {5: [(2, True, 2)]}, 2: {5: []}}      # team 2 blanks
    lams = {1: {5: 1.2}, 2: {5: 1.4}}
    out = projections._ceiling_model(boot, 4, fx, lams, gws)
    blank = [e["id"] for e in boot["elements"] if e["team"] == 2]
    for pid in blank:
        assert 5 not in out.get(pid, {})


def test_missing_model_degrades_instead_of_failing(monkeypatch):
    """An untrained repo must still solve, just without the tilt."""
    monkeypatch.setattr(learn, "load", lambda name: None)
    out = projections._ceiling_model(_boot(), 4, _fx([5]), {1: {5: 1.0}, 2: {5: 1.0}}, [5])
    assert out is None


# --- what consumes it ---------------------------------------------------


def test_the_player_card_carries_the_ceiling():
    proj = {1: {"name": "X", "team": "ARS", "pos": "MID", "price": 7.0,
                "ep": {5: 6.0}, "opp": {5: "BOU(H)"}, "p60": 0.9, "exp_min": 85,
                "ceiling_next": 0.23, "own": 10.0, "status": "a", "news": "",
                "defcon_prob": 0.2, "cs_prob": 0.3, "setpiece_bonus": 0.1}}
    card = explain.player_card(1, proj, {}, {}, {}, {}, {}, [5])
    assert card["ceiling"] == 0.23


def test_the_compare_view_plots_it():
    keys = [m["key"] for m in webdata.METRICS]
    assert "ceiling" in keys, "ceiling is exported but never charted"
    assert "ceiling" in webdata.CARD_FIELDS


def test_the_captain_rationale_states_the_tail():
    proj = {
        1: {"id": 1, "name": "Haaland", "ep": {5: 9.0}, "opp": {5: "SUN(H)"},
            "ceiling": {5: 0.31}},
        2: {"id": 2, "name": "Spiky", "ep": {5: 6.2}, "opp": {5: "BOU(A)"},
            "ceiling": {5: 0.44}},
    }
    week = {"gw": 5, "captain": dict(proj[1]),
            "xi": [{"id": 1}, {"id": 2}]}
    out = explain.captain(week, proj, {}, {}, {}, [5])
    text = " ".join(out["reasons"])
    assert "31%" in text
    # And it names the bigger tail rather than hiding it.
    assert "Spiky" in text and "44%" in text


# --- end to end through build() -----------------------------------------


def _element(i, team, pos_id=3):
    """A bootstrap element with every field build() reads."""
    return {
        "id": i, "web_name": f"P{i}", "team": team, "element_type": pos_id,
        "now_cost": 55 + i, "minutes": 400, "starts": 4,
        "total_points": 24 + i, "bps": 120 + i,
        "threat": str(45 + i), "ict_index": str(38 + i),
        "creativity": "30.0", "influence": "40.0", "form": "4.0",
        "expected_goals": "1.2", "expected_assists": "0.8",
        "expected_goal_involvements": "2.0",
        "expected_goals_per_90": "0.30", "expected_assists_per_90": "0.20",
        "expected_goal_involvements_per_90": "0.50",
        "expected_goals_conceded_per_90": "1.10",
        "defensive_contribution_per_90": "4.0",
        "saves_per_90": "0.0", "clean_sheets": 1, "goals_conceded": 4,
        "clearances_blocks_interceptions": 12, "tackles": 6, "recoveries": 20,
        "saves": 0, "defensive_contribution": 18,
        "status": "a", "news": "", "selected_by_percent": "8.0",
        "chance_of_playing_next_round": None,
        "penalties_order": None, "corners_and_indirect_freekicks_order": None,
        "direct_freekicks_order": None,
    }


def _full_boot():
    return {
        "elements": [_element(i, 1 if i % 2 else 2) for i in range(1, 9)],
        "teams": [{"id": 1, "short_name": "ARS"}, {"id": 2, "short_name": "LIV"}],
        "element_types": [
            {"id": 1, "singular_name_short": "GKP"},
            {"id": 2, "singular_name_short": "DEF"},
            {"id": 3, "singular_name_short": "MID"},
            {"id": 4, "singular_name_short": "FWD"},
        ],
    }


def test_build_puts_the_ceiling_on_every_player(monkeypatch):
    """The wiring test. _ceiling_model working in isolation proves nothing if
    build() computes it before `lams` exists or drops it from the record."""
    gws = [5, 6]
    boot = _full_boot()
    fixtures = [
        {"event": g, "team_h": 1, "team_a": 2,
         "team_h_difficulty": 3, "team_a_difficulty": 2}
        for g in gws
    ]
    # defence.lambdas reaches for Solio and match odds; the ceiling wiring is
    # what is under test, so the expected-goals table is stubbed.
    monkeypatch.setattr(
        projections.defence, "lambdas",
        lambda *a, **k: ({1: {g: 1.25 for g in gws}, 2: {g: 1.60 for g in gws}}, {}),
    )

    out = projections.build(boot, fixtures, gws)

    assert out, "build produced no players"
    for pid, p in out.items():
        assert "ceiling" in p, f"player {pid} has no ceiling"
        assert "ceiling_next" in p
        for g in gws:
            assert 0.0 <= p["ceiling"][g] <= 1.0
        assert p["ceiling_next"] == p["ceiling"][gws[0]]


def test_build_survives_without_a_trained_ceiling(monkeypatch):
    """An untrained repo still solves; the field is simply empty."""
    gws = [5]
    boot = _full_boot()
    fixtures = [{"event": 5, "team_h": 1, "team_a": 2,
                 "team_h_difficulty": 3, "team_a_difficulty": 2}]
    monkeypatch.setattr(
        projections.defence, "lambdas",
        lambda *a, **k: ({1: {5: 1.25}, 2: {5: 1.60}}, {}))
    real_load = learn.load          # bind before patching, or the lambda recurses
    monkeypatch.setattr(projections.learn, "load",
                        lambda name: None if name == "ceiling" else real_load(name))

    out = projections.build(boot, fixtures, gws)
    for p in out.values():
        assert p["ceiling"] == {}
        assert p["ceiling_next"] is None


# --- the probability has to be a probability ----------------------------
#
# AUC is a ranking metric, and the only test above was AUC. Over 82,075
# player-gameweeks the model scored 0.85 and was over-confident by a factor
# of nearly three in the only range anyone reads it in: the band it predicted
# at 0.483 came in at 0.167. For GW8 it put Bruno Fernandes at 61.5%, the
# highest of 668 players and five times the next, and the captaincy note
# offered that to a manager who had asked for variance.


def test_the_model_carries_the_calibration_it_was_fitted_with():
    bundle = learn.load("ceiling")
    assert bundle.get("calibrator") is not None, (
        "models/ceiling.joblib predates calibration — retrain with "
        "`python -m gaffer.learn ceiling`")


def test_no_player_can_be_quoted_a_haul_rate_football_does_not_produce():
    """The most anyone hauls is about a third of the time. The uncalibrated
    model's ceiling was 0.636, and it reached 0.746 on an earlier fit."""
    import numpy as np
    bundle = learn.load("ceiling")
    grid = np.linspace(0.0, 1.0, 201)
    out = bundle["calibrator"].predict(grid)
    assert out.max() <= 0.5
    assert out.min() >= 0.0


def test_calibration_cannot_reorder_two_players():
    """Isotonic is monotone, which is the reason it was chosen: the ranking
    AUC measures survives untouched while the level is corrected. A fix that
    reordered players would be trading one error for another."""
    import numpy as np
    bundle = learn.load("ceiling")
    grid = np.linspace(0.0, 1.0, 201)
    out = bundle["calibrator"].predict(grid)
    assert np.all(np.diff(out) >= -1e-12)


def test_the_reliability_table_is_recorded_so_this_cannot_regress_quietly():
    scores = json.load(open("models/scores.json", encoding="utf-8"))
    c = scores["ceiling"]
    assert c["reliability"], "no reliability table recorded"
    assert c["predicted_max"] > 0
    assert c["brier"] < c["brier_base_rate"], (
        "the model is worse than quoting the base rate to everybody")
    after = c["calibrated"]
    assert after["brier"] <= c["brier"]
    assert after["predicted_max"] < c["predicted_max"]
    # Every band anyone acts on has to be within a quarter of the truth.
    worst = after["worst_band"]
    assert worst is None or abs(worst["actual"] / worst["predicted"] - 1) < 0.25


def test_the_engine_reads_the_model_through_the_calibrated_path():
    """`projections` must not take probabilities off the model itself: read
    raw, this one says 61% where the truth is 20%.

    The check is on the CALL, `predict_proba(X)`, and not on the name — the
    bonus model is still tested for the attribute, by design, and a check on
    the bare name fails on that and on a docstring.
    """
    import inspect
    src = inspect.getsource(projections)
    assert "learn.probability(" in src
    assert "predict_proba(X)" not in src


def test_reliability_bands_report_predicted_against_realised():
    rel = learn.reliability([0, 0, 1, 1], [0.01, 0.03, 0.03, 0.5])
    bands = {(b["from"], b["to"]): b for b in rel}
    assert bands[(0.0, 0.02)]["rows"] == 1
    assert bands[(0.02, 0.05)]["actual"] == 0.5
    assert bands[(0.4, 0.6)]["actual"] == 1.0


def test_the_worst_band_ignores_the_ones_nobody_acts_on():
    """A 2% forecast that comes in at 4% is not a problem worth naming, and
    three rows is not a rate."""
    rel = [{"from": 0.0, "to": 0.02, "rows": 50000, "predicted": 0.004,
            "actual": 0.02},
           {"from": 0.1, "to": 0.2, "rows": 1000, "predicted": 0.13,
            "actual": 0.10},
           {"from": 0.6, "to": 1.01, "rows": 3, "predicted": 0.7,
            "actual": 0.0}]
    w = learn.worst_band(rel)
    assert w["from"] == 0.1


def test_probability_passes_a_bundle_without_a_calibrator_straight_through():
    class Raw:
        def predict_proba(self, X):
            import numpy as np
            return np.array([[0.4, 0.6], [0.9, 0.1]])
    assert list(learn.probability({"model": Raw()}, None)) == [0.6, 0.1]


def test_probability_applies_the_calibrator_and_clips_it():
    class Raw:
        def predict_proba(self, X):
            import numpy as np
            return np.array([[0.0, 0.6]])

    class Silly:
        def predict(self, p):
            return p * 10          # deliberately out of range

    out = learn.probability({"model": Raw(), "calibrator": Silly()}, None)
    assert list(out) == [1.0]


def test_the_top_of_the_curve_is_estimated_from_enough_rows():
    """Isotonic puts a step wherever the data lets it, and the top step of a
    rare-event model is fitted on a handful of rows. Held out properly, one
    fold's curve mapped a prediction of 0.72 to 1.0 on the strength of four
    rows, none of which hauled. The fix collapses everything above the
    hundredth-largest prediction onto one input, so that step is a rate."""
    bundle = learn.load("ceiling")
    hi = bundle.get("calibrator_input_max")
    assert hi is not None, "retrain: the curve has no input cap"
    assert 0.0 < hi < 1.0
    scores = json.load(open("models/scores.json", encoding="utf-8"))
    assert scores["ceiling"]["calibrated"]["top_rows"] >= 100


def test_a_prediction_past_the_cap_is_not_promised_more_than_the_cap():
    import numpy as np
    bundle = learn.load("ceiling")
    hi = bundle["calibrator_input_max"]
    at = learn.train._calibrated(bundle, np.array([hi]))
    beyond = learn.train._calibrated(bundle, np.array([1.0]))
    assert beyond[0] == pytest.approx(at[0])


def test_a_bundle_with_no_calibration_is_left_exactly_as_it_is():
    import numpy as np
    raw = np.array([0.1, 0.9])
    assert list(learn.train._calibrated({}, raw)) == [0.1, 0.9]


def test_refitting_one_model_does_not_touch_another():
    """`python -m gaffer.learn ceiling` has to leave the other nine alone, or
    a one-model change moves every projection and nothing says which model
    moved it."""
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    df = pd.DataFrame({"a": [0.0, 1.0, 0.0, 1.0], "y": [0, 1, 0, 1],
                       "season": ["x"] * 4, "GW": [1, 2, 3, 4]})
    score, rows = learn.train._fit(
        HistGradientBoostingClassifier(max_iter=5), df, ["a"], "y",
        "definitely_not_a_real_model", lambda y, m, X: 0.0,
        only={"ceiling"})
    assert rows == 0
    import os
    assert not os.path.exists("models/definitely_not_a_real_model.joblib")


# --- what else is stale in models/ --------------------------------------


def test_the_only_committed_model_that_no_longer_matches_training_is_bonus():
    """A pinned finding, not a passing grade.

    `models/bonus.joblib` is a six-feature REGRESSOR, from before the model
    was rewritten as a classifier over the four bonus classes.
    `projections._bonus_model` checks for predict_proba and returns None
    without it, BY DESIGN — so the rewrite has never been in effect and the
    engine has been using the bps90 heuristic the rewrite was meant to
    replace. Nothing failed; it just quietly did not happen.

    This test fails when bonus is refitted, which is the point: the fix is
    `python -m gaffer.learn bonus`, and whoever runs it should come here and
    delete the exception rather than leave a passing test that expects a
    broken model.
    """
    stale = learn.audit()
    assert set(stale) == {"bonus"}, (
        "models/ no longer matches training for: "
        + ", ".join(f"{k} ({v})" for k, v in stale.items()))


def test_every_other_model_is_the_model_training_declares():
    stale = learn.audit()
    for name in learn.train.DECLARED:
        if name == "bonus":
            continue
        assert name not in stale, f"{name}: {stale.get(name)}"
