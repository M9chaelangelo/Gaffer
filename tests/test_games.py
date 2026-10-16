"""One engine, two games: the differences that change strategy."""
import pytest

from gaffer import games


def test_both_games_resolve():
    assert games.get("fpl").key == "fpl"
    assert games.get("ucl").key == "ucl"
    assert games.get("FPL").key == "fpl"


def test_an_unknown_game_names_the_ones_that_exist():
    with pytest.raises(KeyError) as e:
        games.get("nfl")
    assert "fpl" in str(e.value) and "ucl" in str(e.value)


def test_ucl_has_no_vice_captain_but_a_movable_armband():
    """The single biggest strategic difference, and it has no FPL analogue."""
    u, f = games.get("ucl"), games.get("fpl")
    assert u.has_vice_captain is False
    assert u.armband_movable_midround is True
    assert f.has_vice_captain is True
    assert f.armband_movable_midround is False


def test_ucl_gives_two_free_transfers_a_round_and_fpl_one():
    assert games.get("ucl").free_transfers_per_round == 2
    assert games.get("fpl").free_transfers_per_round == 1


def test_both_pick_fifteen_and_start_eleven():
    for k in ("fpl", "ucl"):
        g = games.get(k)
        assert g.squad_size == 15
        assert sum(g.xi_min.values()) <= 11 <= sum(g.xi_max.values())


def test_recoveries_pay_from_the_first_in_ucl_and_not_at_all_in_fpl():
    """FPL's defensive contribution is a cliff; UCL's is a rate.

    A midfielder with nine recoveries and nothing else: UCL pays three
    points for them, FPL pays none, because nine is under the threshold of
    twelve. That inverts how a holding midfielder should be priced.
    """
    ucl = games.get("ucl")
    line = {"minutes": 90, "recoveries": 9, "goals_conceded": 2}
    assert ucl.score(line, "MID") == 2 + 3          # appearance 2, recoveries 3

    fpl = games.get("fpl")
    assert fpl.score({"minutes": 90, "defensive_contribution": 9}, "MID") == 2
    assert fpl.score({"minutes": 90, "defensive_contribution": 12}, "MID") == 4


def test_a_defender_goal_is_worth_six_in_both():
    assert games.get("fpl").score(
        {"minutes": 90, "goals_scored": 1, "goals_conceded": 1}, "DEF") == 8
    assert games.get("ucl").score(
        {"minutes": 90, "goals": 1, "goals_conceded": 1}, "DEF") == 8


def test_a_missing_goals_conceded_is_not_a_clean_sheet():
    """Absent data must never pay. `g()` returns 0 for a missing key, so
    deriving the clean sheet from `goals_conceded == 0` handed one to every
    line that simply did not carry the field."""
    ucl = games.get("ucl")
    assert ucl.score({"minutes": 90}, "DEF") == 2                  # nothing known
    assert ucl.score({"minutes": 90, "goals_conceded": 0}, "DEF") == 6
    assert ucl.score({"minutes": 90, "clean_sheet": True}, "DEF") == 6


def test_ucl_player_of_the_match_is_worth_three():
    assert games.get("ucl").score(
        {"minutes": 90, "motm": 1, "goals_conceded": 1}, "MID") == 5


def test_a_player_who_did_not_play_scores_nothing():
    for k, goal_key in (("fpl", "goals_scored"), ("ucl", "goals")):
        assert games.get(k).score({"minutes": 0, goal_key: 1}, "MID") == 0
