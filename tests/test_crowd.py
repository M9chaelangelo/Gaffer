"""Crowd signals — three of which were computed and discarded for weeks."""
from gaffer import crowd


def boot():
    return {
        "total_players": 1_000_000,
        "events": [
            {"id": 1, "finished": True, "chip_plays": [
                {"chip_name": "bboost", "num_played": 10_000},
                {"chip_name": "wildcard", "num_played": 50_000}]},
            {"id": 2, "finished": True, "chip_plays": [
                {"chip_name": "bboost", "num_played": 40_000}]},
            {"id": 3, "finished": False, "chip_plays": []},
        ],
        "teams": [{"id": 1, "short_name": "AAA"}],
        "elements": [],
    }


PROJ = {11: {"name": "Haaland", "team": "MCI", "pos": "FWD", "price": 15.6,
             "ep_next": 8.9, "ep_horizon": 46.4},
        22: {"name": "Saka", "team": "ARS", "pos": "MID", "price": 9.6,
             "ep_next": 5.7, "ep_horizon": 29.0},
        33: {"name": "Palmer", "team": "CHE", "pos": "MID", "price": 10.0,
             "ep_next": 6.0, "ep_horizon": 30.0}}


def test_chip_appetite_reads_the_whole_game_not_a_sample():
    a = crowd.chip_appetite(boot(), "Bench Boost")
    assert a["by_gw"] == [(1, 0.01), (2, 0.04)]
    assert a["share_spent"] == 0.05
    assert a["share_remaining"] == 0.95
    assert a["busiest_gw"] == 2


def test_an_unfinished_gameweek_is_not_counted():
    """chip_plays fills in as a gameweek is played, so a live week understates."""
    b = boot()
    b["events"][2] = {"id": 3, "finished": False,
                      "chip_plays": [{"chip_name": "bboost", "num_played": 900_000}]}
    assert crowd.chip_appetite(b, "Bench Boost")["share_spent"] == 0.05


def test_elite_chips_names_them_in_plain_english():
    el = {"n": 100, "chips": {"bboost": 0.12, "3xc": 0.03}}
    out = crowd.elite_chips(el)
    assert out["played"]["Bench Boost"] == 0.12
    assert out["played"]["Triple Captain"] == 0.03


def test_no_elite_chip_says_so_rather_than_looking_like_zero_data():
    out = crowd.elite_chips({"n": 100, "chips": {}})
    assert out["played"] == {} and "no chip" in out["note"]


def test_elite_captains_are_ranked_and_named():
    el = {"n": 100, "captains": {11: 0.74, 22: 0.10}}
    rows = crowd.elite_captains(el, PROJ)
    assert [r["name"] for r in rows] == ["Haaland", "Saka"]
    assert rows[0]["share"] == 0.74


def test_versus_elite_splits_what_they_hold_from_what_you_hold_alone():
    el = {"n": 200, "own": {11: 0.99, 33: 0.60, 22: 0.04}}
    out = crowd.versus_elite([22, 11], el, PROJ)
    assert [r["name"] for r in out["they_own_you_do_not"]] == ["Palmer"]
    assert [r["name"] for r in out["you_own_they_mostly_do_not"]] == ["Saka"]


def test_a_missing_elite_sample_does_not_crash_the_block():
    out = crowd.build(boot(), None, PROJ, [11])
    assert out["elite_sample"] == 0
    assert out["bench_boost_appetite"]["share_spent"] == 0.05


def test_the_block_states_that_upcoming_picks_are_not_public():
    """The honest constraint: nobody can see this week before the deadline."""
    out = crowd.build(boot(), {"n": 100, "own": {}}, PROJ, [11])
    assert "last finished gameweek" in out["note"]
