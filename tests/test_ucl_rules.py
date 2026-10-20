"""The ruleset, checked against UEFA's published 2026/27 rules.

Every number here was read off uefa.com rather than remembered. The scoring
table was already right; the structural rules around it were not, and the
engine had been treating a competition whose constraints change six times as
though they were constant.
"""
from gaffer.ucl import rules


def test_the_scoring_table_matches_the_published_one():
    assert rules.APPEARANCE == 1 and rules.APPEARANCE_60 == 1
    assert rules.ASSIST == 3
    assert rules.GOAL == {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
    assert rules.CLEAN_SHEET == {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
    assert rules.RECOVERIES_PER_POINT == 3      # no threshold, unlike FPL
    assert rules.SAVES_PER_POINT == 3
    assert rules.CONCEDED_PER_POINT == 2
    assert rules.PLAYER_OF_THE_MATCH == 3
    assert rules.GOAL_OUTSIDE_BOX == 1
    assert (rules.PENALTY_SAVE, rules.PENALTY_MISS) == (5, -2)
    assert (rules.PENALTY_WON, rules.PENALTY_CONCEDED) == (2, -1)
    assert (rules.YELLOW, rules.RED, rules.OWN_GOAL) == (-1, -3, -2)


def test_the_squad_and_formation_rules():
    assert rules.SQUAD_COMPOSITION == {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    assert sum(rules.SQUAD_COMPOSITION.values()) == rules.SQUAD_SIZE == 15
    assert rules.MIN_STARTING == {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
    assert sum(rules.MIN_STARTING.values()) <= rules.STARTING_SIZE == 11


def test_the_club_limit_rises_as_the_field_narrows():
    """Three in the league phase, eight in the final. A chip held back for the
    knockouts is a different instrument from one played now."""
    m = rules.MAX_PER_CLUB_BY_MATCHDAY
    assert all(m[md] == 3 for md in range(1, 9))
    assert m[9] == m[10] == m[11] == m[12] == 4
    assert m[13] == m[14] == 5
    assert m[15] == m[16] == 6
    assert m[17] == 8
    assert rules.MAX_PER_CLUB == m[3]


def test_the_budget_rises_after_the_league_phase():
    assert rules.BUDGET == 100.0
    assert rules.BUDGET_KNOCKOUT == 105.0


def test_free_transfers_by_matchday():
    f = rules.FREE_TRANSFERS_BY_MATCHDAY
    assert all(f[md] == 2 for md in range(2, 9))
    assert f[13] == f[15] == f[17] == 5
    assert f[12] == f[14] == f[16] == 3
    assert len(f) == 17


def test_chips_are_blocked_exactly_where_transfers_are_already_unlimited():
    """That is the reason for the blackout, so the two must agree."""
    unlimited = {md for md, n in rules.FREE_TRANSFERS_BY_MATCHDAY.items()
                 if n is None}
    assert unlimited == set(rules.CHIP_BLOCKED_MATCHDAYS) == {1, 9, 11}


def test_there_are_two_chips_not_three():
    """`allin` was declared and does not exist in this competition."""
    assert rules.GAME.chips == ("wc", "lim")


def test_the_matchdays_with_no_in_play_options():
    """Every match kicks off together on these, so there is no armband move
    and no substitution window — a chip played there is worth less."""
    assert rules.SIMULTANEOUS_MATCHDAYS == (8, 17)
    assert rules.MAX_SUBS_PER_DAY == 4


def test_a_manual_change_costs_the_safety_net():
    """Touching anything disables automatic substitutions for the rest of the
    matchday, and a captain who then does not play scores nobody double."""
    assert rules.MANUAL_CHANGE_DISABLES_AUTOSUBS is True


def test_the_game_record_agrees_with_the_constants():
    g = rules.GAME
    assert g.squad == rules.SQUAD_COMPOSITION
    assert g.max_per_club == rules.MAX_PER_CLUB
    assert g.budget == rules.BUDGET
    assert g.captain_multiplier == 2
    assert g.has_vice_captain is False
    assert g.armband_movable_midround is True
    assert g.free_transfers_per_round == 2
    assert g.extra_transfer_cost == 4
