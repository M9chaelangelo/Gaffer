"""How two sides match up: what a defence concedes, and to whom."""
from gaffer import matchup

BOOT = {
    "teams": [{"id": 1, "short_name": "AAA"}, {"id": 2, "short_name": "BBB"}],
    "elements": [
        {"id": 10, "team": 1, "element_type": 3},   # AAA midfielder
        {"id": 11, "team": 1, "element_type": 4},   # AAA forward
        {"id": 20, "team": 2, "element_type": 3},   # BBB midfielder
    ],
}
FX = [{"event": 1, "team_h": 1, "team_a": 2, "finished": True,
       "team_h_score": 3, "team_a_score": 0},
      {"event": 2, "team_h": 2, "team_a": 1, "finished": True,
       "team_h_score": 1, "team_a_score": 2},
      {"event": 3, "team_h": 1, "team_a": 2, "finished": False,
       "team_h_score": None, "team_a_score": None}]


def live(gw, scorers):
    return {gw: {"elements": [{"id": i, "stats": {"goals_scored": g, "assists": 0}}
                              for i, g in scorers.items()]}}


def test_an_unfinished_match_is_not_counted():
    """A live score is not a result. Reading one is wishful thinking."""
    v = matchup.venue_splits(FX, [1, 2])
    assert v[1]["home"]["games"] == 1       # not 2, despite the GW3 fixture
    assert v[1]["home"]["for"] == 3


def test_venue_splits_separate_home_from_away():
    v = matchup.venue_splits(FX, [1, 2])
    assert v[1]["home"]["for_per_game"] == 3.0
    assert v[1]["away"]["for_per_game"] == 2.0
    assert v[2]["home"]["against_per_game"] == 2.0


def test_concession_is_attributed_to_the_position_that_scored():
    lv = {}
    lv.update(live(1, {10: 2, 11: 1}))       # AAA mid x2, fwd x1, all vs BBB
    prof, league = matchup.concession_by_position(BOOT, FX[:1], lv)
    assert prof[2]["raw"]["MID"] == 2.0
    assert prof[2]["raw"]["FWD"] == 1.0
    assert prof[1]["raw"]["MID"] == 0.0, "AAA conceded nothing"


def test_the_index_is_shrunk_hard_toward_average():
    """Two or three goals per position is not a rate. One game must not
    produce a 'weak to midfielders' verdict."""
    lv = live(1, {10: 3})
    prof, _ = matchup.concession_by_position(BOOT, FX[:1], lv)
    idx = prof[2]["index"]["MID"]
    assert matchup.CLAMP[0] <= idx <= matchup.CLAMP[1]
    assert abs(idx - 1.0) < 0.25, "one game may only nudge, never decide"


def test_weight_zero_means_the_signal_cannot_move_a_projection():
    """It ships measured and inert, unlike the ceiling model."""
    prof = {2: {"index": {"MID": 1.25}}}
    assert matchup.multiplier(prof, 2, "MID", weight=0.0) == 1.0
    assert matchup.multiplier(prof, 2, "MID", weight=1.0) > 1.0


def test_the_multiplier_is_always_clamped():
    prof = {2: {"index": {"MID": 99.0}}}
    assert matchup.multiplier(prof, 2, "MID", weight=5.0) == matchup.CLAMP[1]


def test_an_unknown_opponent_is_neutral_rather_than_an_error():
    assert matchup.multiplier({}, 999, "MID", weight=1.0) == 1.0


def test_build_reports_by_team_name_and_states_its_own_weight():
    out = matchup.build(BOOT, FX, live(1, {10: 2}), weight=0.0)
    assert out["weight"] == 0.0
    assert "AAA" in out["by_team"] and "BBB" in out["by_team"]
    assert out["by_team"]["BBB"]["venue"] is not None


def test_an_empty_live_window_still_returns_a_neutral_profile():
    """The first wiring borrowed the ledger's payloads, which are only
    fetched for gameweeks it has not already settled — so from the second
    run on there was no data and the whole block vanished. Whatever the
    cause, no data must mean a neutral index, not a missing section."""
    out = matchup.build(BOOT, FX, {}, weight=0.0)
    assert out["by_team"], "a profile is still produced"
    for row in out["by_team"].values():
        # With no goals at all every league baseline is zero, so every
        # position is unmeasurable. None is the honest answer; 1.0 would
        # claim "exactly average" on no evidence whatsoever.
        assert all(v is None for v in row["index"].values())
        assert sorted(row["unmeasured"]) == ["DEF", "FWD", "GKP", "MID"]


# --- the goalkeeper floor ---------------------------------------------------

def test_a_position_with_no_league_baseline_reports_null_not_a_number():
    """Goalkeepers score ~0.003 goals a game across the whole league.

    One freak keeper goal gave that club a rate of ~0.17, which divided by
    0.003 is an index of 55 — and the clamp dutifully printed 1.25, reading
    on the page as 'weak to goalkeepers' and indistinguishable from a real
    finding. A ratio is only meaningful when its denominator is.
    """
    boot = {"teams": [{"id": 1, "short_name": "AAA"}, {"id": 2, "short_name": "BBB"}],
            "elements": [{"id": 10, "team": 1, "element_type": 1},    # a keeper
                         {"id": 11, "team": 1, "element_type": 4}]}   # a forward
    # Fourteen games so the one keeper goal is ~0.04 per game, which is the
    # real shape: across a league, keepers score about 0.003.
    fx = [{"event": g, "team_h": 1, "team_a": 2, "finished": True,
           "team_h_score": 1, "team_a_score": 0} for g in range(1, 15)]
    lv = {g: {"elements": [{"id": 11, "stats": {"goals_scored": 1, "assists": 0}}]}
          for g in range(1, 15)}
    lv[1] = {"elements": [{"id": 10, "stats": {"goals_scored": 1, "assists": 0}}]}
    prof, league = matchup.concession_by_position(boot, fx, lv)
    assert league["GKP"] < matchup.MIN_BASELINE
    assert prof[2]["index"]["GKP"] is None
    assert "GKP" in prof[2]["unmeasured"]


def test_an_unmeasured_position_is_neutral_in_the_multiplier():
    assert matchup.multiplier({2: {"index": {"GKP": None}}}, 2, "GKP",
                              weight=1.0) == 1.0


# --- last season as the prior -----------------------------------------------

import pandas as pd


def _hist():
    rows = []
    for fixture, (home, away) in enumerate([("AAA", "BBB")] * 10):
        # BBB concede two to midfielders every week; AAA concede nothing
        rows.append(dict(season="2025-26", fixture=fixture, was_home=True,
                         team=home, position="MID", goals_scored=2, assists=0))
        rows.append(dict(season="2025-26", fixture=fixture, was_home=False,
                         team=away, position="MID", goals_scored=0, assists=0))
    return pd.DataFrame(rows)


def test_history_attributes_concession_to_the_opposing_club():
    h = matchup.historical(_hist())
    assert h["BBB"]["MID"] > h.get("AAA", {}).get("MID", 0.0)


def test_history_is_empty_rather_than_wrong_when_columns_are_missing():
    assert matchup.historical(pd.DataFrame({"season": ["x"]})) == {}
    assert matchup.historical(None) == {}


def test_a_club_with_history_starts_from_it_not_from_average():
    """A promoted side has no history and must fall back to the mean."""
    boot = {"teams": [{"id": 1, "short_name": "AAA"}, {"id": 2, "short_name": "BBB"}],
            "elements": [{"id": 10, "team": 1, "element_type": 3}]}
    fx = [{"event": 1, "team_h": 1, "team_a": 2, "finished": True,
           "team_h_score": 1, "team_a_score": 0}]
    lv = {1: {"elements": [{"id": 10, "stats": {"goals_scored": 1, "assists": 0}}]}}
    leaky = matchup.concession_by_position(boot, fx, lv, prior={"BBB": {"MID": 1.25}})[0]
    tight = matchup.concession_by_position(boot, fx, lv, prior={"BBB": {"MID": 0.80}})[0]
    assert leaky[2]["index"]["MID"] > tight[2]["index"]["MID"]
