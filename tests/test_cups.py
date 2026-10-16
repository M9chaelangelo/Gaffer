"""Predicting blanks and doubles from cup progression, the Crellin way.

The calendar is published a year ahead; the only unknown is who survives
each round, and a knockout bracket is exactly what a simulation is for. So
this produces a probability per club rather than a hand-made point call.
"""
from gaffer import cups

STR = {"ARS": {"overall": 1.85}, "MCI": {"overall": 1.70},
       "LIV": {"overall": 1.40}, "BHA": {"overall": 1.50},
       "HUL": {"overall": 0.45}, "COV": {"overall": 0.50}}
TEAMS = list(STR)


def test_a_stronger_club_is_likelier_to_go_through_but_never_certain():
    assert cups.win_prob(2.0, 0.5) > cups.win_prob(1.0, 1.0) > cups.win_prob(0.5, 2.0)
    assert cups.win_prob(1.0, 1.0) == 0.5
    assert cups.win_prob(100.0, 0.01) < 0.95, "knockout football is never a lock"


def test_the_upset_term_pulls_toward_a_coin_flip():
    lopsided = (5.0, 0.2)
    assert cups.win_prob(*lopsided, upset=0.0) > cups.win_prob(*lopsided, upset=0.5)
    assert abs(cups.win_prob(*lopsided, upset=1.0) - 0.5) < 1e-9


def test_survival_falls_every_round_and_favours_the_strong():
    s = cups.simulate_rounds(TEAMS, STR, rounds=3, field_size=16, n=3000)
    assert all(abs(v - 1.0) < 1e-9 for v in s[0].values()), "everyone starts alive"
    for r in range(3):
        assert sum(s[r + 1].values()) < sum(s[r].values())
    assert s[3]["ARS"] > s[3]["HUL"]


def test_the_model_is_calibrated_to_the_historical_semi_final_mix():
    """The one number this can be held to without waiting a season.

    About 3.4 of the four FA Cup semi-finalists are Premier League clubs.
    A first pass produced 2.59, which would have understated every blank
    probability by a quarter.
    """
    teams = [f"T{i}" for i in range(20)]
    strengths = {t: {"overall": 1.0} for t in teams}
    r = cups.fa_cup(strengths, teams, n=6000)
    assert 3.0 <= sum(r["semi_final"].values()) <= 3.8


def test_an_undeclared_cup_says_so_instead_of_guessing():
    """The EFL Cup draw is in no feed here. Silence beats two invented finalists."""
    cfg = {"cup_blanks": {30: {"cup": "efl", "round": "final"}}}
    out = cups.blanks(cfg, STR, TEAMS, n=500)
    assert out[30]["known"] is False
    assert "clubs" not in out[30]
    assert "not declared" in out[30]["note"]


def test_doubles_are_carried_from_the_blank_that_caused_them():
    cfg = {"cup_blanks": {33: {"cup": "fa", "round": "semi_final"}},
           "cup_doubles": {35: {"from_blank_gw": 33, "share": 0.5}}}
    out = cups.build(cfg, STR, TEAMS, n=2000)
    b = {r["team"]: r["p_blank"] for r in out["blanks"][33]["clubs"]}
    for r in out["doubles"][35]["clubs"]:
        # both sides are rounded to three places, so compare at that scale
        assert abs(r["p_double"] - b[r["team"]] * 0.5) < 1e-3


def test_a_double_whose_blank_is_unknown_is_also_unknown():
    cfg = {"cup_blanks": {30: {"cup": "efl", "round": "final"}},
           "cup_doubles": {32: {"from_blank_gw": 30, "share": 0.7}}}
    out = cups.build(cfg, STR, TEAMS, n=500)
    assert out["doubles"][32]["known"] is False
