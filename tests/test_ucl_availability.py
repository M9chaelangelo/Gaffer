"""Start probability must follow minutes played, and flags must beat prose."""
from gaffer.ucl import availability, project


def row(**kw):
    base = dict(id="1", name="E. Haaland", full_name="Erling Haaland",
                code="MCI", team="Man City", pos="FWD", price=11.1,
                minutes=180.0, matches=2, trained="In contention to start next game",
                status="")
    base.update(kw)
    return base


# --- start_prob --------------------------------------------------------------

def test_an_every_minute_player_and_a_bench_player_are_not_the_same_number():
    """The old heuristic returned 0.93 for both of these."""
    nailed = project.start_prob(row(minutes=180.0, matches=2))
    benched = project.start_prob(row(minutes=24.0, matches=2))
    assert nailed > 0.85
    assert benched < 0.2


def test_a_part_time_player_lands_in_between():
    """Barkley: 106 of 180 minutes, one start in two."""
    p = project.start_prob(row(minutes=106.0, matches=2))
    assert 0.4 < p < 0.75


def test_a_suspension_still_beats_a_full_minutes_record():
    assert project.start_prob(row(minutes=180.0, matches=2,
                                  trained="Suspended")) == 0.0
    assert project.start_prob(row(minutes=180.0, matches=2,
                                  forced_out=True)) == 0.0


def test_an_availability_flag_scales_the_minutes_estimate():
    full = project.start_prob(row(minutes=180.0, matches=2))
    flagged = project.start_prob(row(minutes=180.0, matches=2, availability=0.75))
    assert abs(flagged - full * 0.75) < 1e-9


def test_a_flag_of_zero_removes_him_however_much_he_has_played():
    assert project.start_prob(row(minutes=180.0, matches=2,
                                  availability=0.0)) == 0.0


def test_a_flag_cannot_raise_a_doubt_recorded_in_the_prose():
    """UEFA saying "knock" and FPL saying 90% must not become 90%."""
    p = project.start_prob(row(trained="Knock", availability=0.9))
    assert p <= project.DOUBT_START


def test_no_football_played_anywhere_falls_back_to_the_prose():
    assert project.start_prob(row(minutes=0.0, matches=0)) > 0.5
    assert project.minutes_share(row(matches=0)) is None


# --- the FPL bridge ----------------------------------------------------------

def boot(status="a", chance=None, first="Erling", second="Haaland",
         web="Haaland", short="MCI"):
    return {"teams": [{"id": 1, "short_name": short}],
            "elements": [{"id": 9, "team": 1, "status": status,
                          "chance_of_playing_next_round": chance,
                          "first_name": first, "second_name": second,
                          "web_name": web}]}


def test_a_flagged_player_is_matched_through_his_club_code():
    flags, unmatched = availability.chances([row()], boot(status="d", chance=75))
    assert flags == {"1": 0.75}
    assert unmatched == []


def test_an_unflagged_player_is_left_alone_rather_than_set_to_one():
    """Absence of news is not a clean bill of health: overwriting UEFA's own
    note with 1.0 would lose information rather than add it."""
    flags, _ = availability.chances([row()], boot(status="a", chance=None))
    assert flags == {}
    flags, _ = availability.chances([row()], boot(status="a", chance=100))
    assert flags == {}


def test_an_injured_player_reads_as_out():
    for s in ("i", "u", "s", "n"):
        flags, _ = availability.chances([row()], boot(status=s))
        assert flags == {"1": 0.0}, s


def test_a_club_outside_the_premier_league_is_not_matched():
    flags, unmatched = availability.chances(
        [row(code="BAR", team="Barcelona")], boot(status="d", chance=50))
    assert flags == {}
    assert unmatched == []          # not a miss; FPL simply does not cover him


def test_a_namesake_at_another_club_cannot_supply_the_flag():
    """Two Gomezes and two Hendersons exist. Clubs join first, for that
    reason."""
    b = boot(status="i", first="Joe", second="Gomez", web="Gomez", short="LIV")
    flags, _ = availability.chances([row(code="MCI")], b)
    assert flags == {}


def test_a_surname_in_the_config_is_enough_to_set_a_doubt():
    pool = [row()]
    changed, missed = availability.apply(pool, {}, {"Haaland": 0.5})
    assert pool[0]["availability"] == 0.5
    assert missed == []
    assert changed[0][3] == "config"


def test_a_typed_doubt_overrides_a_fetched_flag():
    pool = [row()]
    availability.apply(pool, {"1": 0.75}, {"Haaland": 0.2})
    assert pool[0]["availability"] == 0.2


def test_a_typed_name_that_matches_nobody_is_reported():
    _, missed = availability.apply([row()], {}, {"Nobody At All": 0.5})
    assert missed == ["Nobody At All"]
