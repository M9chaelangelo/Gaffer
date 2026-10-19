"""Expected minutes must follow the match log, not the season's round count."""
from gaffer import playtime


def log(mins, starts=None, first_gw=1):
    """[(gw, minutes, started, slots)] from a list of minutes."""
    if starts is None:
        starts = [1.0 if m >= 45 else 0.0 for m in mins]
    return [(first_gw + i, float(m), float(s), 1)
            for i, (m, s) in enumerate(zip(mins, starts))]


def test_every_minute_starter_reads_as_one():
    r = playtime.estimate(log([90] * 7), 7)
    assert r["p_start"] > 0.97
    assert r["exp_min"] > 88


def test_a_dropped_starter_falls_to_a_substitute():
    """Barkley: started GW1-4, then 6, 4 and 18 minutes off the bench.

    The season average put him at 47 minutes a game and the engine's blend at
    57 expected minutes. He is a substitute, and the estimate has to say so.
    """
    r = playtime.estimate(log([73, 77, 90, 62, 6, 4, 18]), 7)
    assert r["p_start"] < 0.3
    assert r["exp_min"] < 35
    # His minutes WHEN he starts are still a starter's minutes — the two facts
    # are kept apart on purpose.
    assert r["mins_if_start"] > 60


def test_a_new_starter_is_not_charged_for_the_weeks_he_was_not_picked():
    """Barcola: nothing until GW3, then 26, 71, 71, 87, 90.

    345 minutes over seven rounds reads as 49 a game, which is how he came to
    be ranked 40th of 119 midfielders while starting every week.
    """
    r = playtime.estimate(log([0, 0, 26, 71, 71, 87, 90]), 7)
    assert r["p_start"] > 0.85
    # The property, not a number tuned to one setting of the dials: he reads as
    # a starter rather than the half-and-half the season average made of him.
    assert r["exp_min"] > 70
    season_average = sum(m for m in [0, 0, 26, 71, 71, 87, 90]) / 7
    assert r["exp_min"] > season_average + 20


def test_the_prior_is_the_player_own_rate_not_a_coin_toss():
    """One start is not a certainty, but it is not one-in-two either."""
    never = playtime.estimate(log([0, 0, 0, 12], starts=[0, 0, 0, 0]), 4)
    always = playtime.estimate(log([90], starts=[1]), 1)
    assert never["p_start"] < 0.1
    assert always["p_start"] > 0.9


def test_an_early_substitution_does_not_make_a_starter_a_cameo_risk():
    """Off injured on eight minutes, then back to normal."""
    r = playtime.estimate(log([90, 90, 8, 90, 90], starts=[1, 1, 1, 1, 1]), 5)
    assert r["mins_if_start"] >= playtime.MIN_START_MINUTES
    assert r["exp_min"] > 70


def test_a_blank_gameweek_is_not_evidence_that_he_was_dropped():
    """A club with no fixture contributes no row at all, so the estimate is
    unchanged — the alternative reads a blank as a benching."""
    played = playtime.estimate(log([90, 90, 90]), 3)
    blanked = playtime.estimate(log([90, 90]) + [(4, 90.0, 1.0, 1)], 4)
    assert abs(played["exp_min"] - blanked["exp_min"]) < 1.0


def test_a_double_gameweek_is_read_per_fixture():
    """180 minutes across two fixtures is a 90-minute player, not a 180."""
    r = playtime.estimate([(1, 90.0, 1.0, 1), (2, 180.0, 2.0, 2)], 2)
    assert r["exp_min"] <= 90.0
    assert r["mins_if_start"] > 85


def test_club_slots_counts_fixtures_per_gameweek():
    fixtures = [{"event": 1, "team_h": 1, "team_a": 2},
                {"event": 2, "team_h": 1, "team_a": 3},
                {"event": 2, "team_h": 4, "team_a": 1},
                {"event": None, "team_h": 1, "team_a": 5}]
    slots = playtime.club_slots(fixtures, {1, 2})
    assert slots[1] == {1: 1, 2: 2}
    assert slots[2] == {1: 1}
    assert 5 not in slots


def test_table_skips_gameweeks_a_club_did_not_play():
    boot = {"elements": [{"id": 7, "team": 1}]}
    fixtures = [{"event": 1, "team_h": 1, "team_a": 2}]
    live = {1: {"elements": [{"id": 7, "stats": {"minutes": 90, "starts": 1}}]},
            2: {"elements": [{"id": 7, "stats": {"minutes": 0, "starts": 0}}]}}
    t = playtime.table(boot, fixtures, live, now=2)
    # Gameweek 2 has no fixture for team 1, so the zero is not counted.
    assert t[7]["p_start"] > 0.9


def test_a_single_missed_week_does_not_halve_a_settled_starter():
    """Thomas started six in a row at ninety minutes, then missed one game
    carrying no injury flag. Counting that nought at full weight read him as
    47% likely to start and 43 expected minutes, which is a wrong transfer."""
    r = playtime.estimate(log([90, 77, 90, 90, 90, 90, 0]), 7)
    assert r["p_start"] > 0.55
    assert r["exp_min"] > 50


def test_cameos_are_not_discounted_the_way_noughts_are():
    """A nought is ambiguous — dropped, injured, suspended, unused — and the
    feed does not say which. A cameo says plainly that he was available and
    not picked, so Barkley's 6, 4 and 18 must keep their full weight."""
    dropped = playtime.estimate(log([73, 77, 90, 62, 6, 4, 18]), 7)
    absent = playtime.estimate(log([73, 77, 90, 62, 0, 0, 0]), 7)
    assert dropped["p_start"] < 0.3
    assert absent["p_start"] > dropped["p_start"]


def test_minutes_when_starting_use_a_longer_window_than_selection():
    """Groß has played 90 six times and then 70. Measured over the selection
    half-life his last match alone is most of the evidence and he reads as a
    78-minute starter; whether a man is picked moves faster than how long he
    lasts."""
    r = playtime.estimate(log([90, 90, 90, 90, 90, 90, 70]), 7)
    assert r["mins_if_start"] > 82
    short = playtime.estimate(log([90, 90, 90, 90, 90, 90, 70]), 7,
                              duration_half_life=0.8)
    assert short["mins_if_start"] < r["mins_if_start"]
