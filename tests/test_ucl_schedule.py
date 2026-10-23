"""The league phase is eight matchdays, and the engine could only see one.

The schedule is committed rather than fetched, so the thing that can go wrong
is a bad paste: a fixture read off the wrong row gives one club nine games and
another seven, and no plan would ever reveal it — it would quietly misprice
two clubs for the rest of the phase. These tests are mostly about that.
"""
from __future__ import annotations

import copy

import pytest

from gaffer.ucl import schedule


@pytest.fixture(scope="module")
def sched():
    return schedule.load()


def _pool(clubs, scored=12.0, conceded=12.0, price=6.0, played=8):
    """A squad whose TEAM totals are the numbers given.

    `strength.aggregate` sums goals over every row and takes goals conceded
    from the keeper alone, so a per-player figure has to be divided by the
    squad and a per-team one must not be. Handing it three goals a player
    gave every club fifteen a game and every fixture ran into the clamp,
    which made three of these tests pass on a constant.
    """
    out = []
    for c in clubs:
        out.append(dict(team=c, team_id=c, pos="GK", goals=0.0,
                        conceded=conceded, matches=played,
                        team_played=played, price=price))
        for _ in range(14):
            out.append(dict(team=c, team_id=c, pos="MID",
                            goals=scored / 14.0, conceded=0.0,
                            matches=played, team_played=played, price=price))
    return out


# --- the committed data ------------------------------------------------------

def test_the_committed_schedule_obeys_the_competition_format(sched):
    """Thirty-six clubs, eight distinct opponents each, four home and four
    away, once per matchday. The format guarantees all of it, so anything
    else is a transcription error and not a surprise about the draw."""
    assert schedule.validate(sched) == []
    assert sorted(sched) == list(range(1, 9))
    assert sum(len(fx) for fx in sched.values()) == 144


def test_the_last_matchday_is_the_one_the_rules_call_simultaneous(sched):
    """`rules.SIMULTANEOUS_MATCHDAYS` says matchday eight kicks off at once,
    which is why a chip played there loses the whole in-play option. The
    schedule agrees, and it was typed from a different source."""
    assert len({f["date"] for f in sched[8]}) == 1
    for md in range(1, 8):
        assert len({f["date"] for f in sched[md]}) > 1


def test_the_schedule_carries_its_own_provenance():
    p = schedule.provenance()
    assert "Wikipedia" in p["source"] and "cross-checked" in p["source"]


def test_every_alias_names_a_club_that_is_actually_in_the_schedule(sched):
    """A stale alias is invisible: the club it used to name resolves by its
    own spelling and the dead entry sits there looking authoritative."""
    clubs = {f[end] for fx in sched.values() for f in fx
             for end in ("home", "away")}
    assert set(schedule.ALIASES) <= clubs


# --- what validate() is for --------------------------------------------------

def test_a_club_given_a_ninth_fixture_is_caught(sched):
    bad = copy.deepcopy(sched)
    # Liverpool's matchday 8 opponent mistyped as a club they already play.
    bad[8] = [dict(f) for f in bad[8]]
    bad[8][0] = {**bad[8][0], "home": "Liverpool", "away": "Arsenal"}
    problems = schedule.validate(bad)
    assert problems
    assert any("Liverpool" in p for p in problems)


def test_a_club_playing_twice_in_one_matchday_is_caught(sched):
    bad = copy.deepcopy(sched)
    first = bad[4][0]
    bad[4] = [dict(f) for f in bad[4]]
    bad[4][1] = {**bad[4][1], "home": first["home"]}
    problems = schedule.validate(bad)
    assert any("play twice" in p for p in problems)


def test_a_repeated_opponent_is_caught(sched):
    bad = {1: [{"home": "A", "away": "B", "date": "2026-09-08"}],
           2: [{"home": "A", "away": "B", "date": "2026-10-13"}]}
    problems = schedule.validate(bad)
    assert any("twice" in p for p in problems)


def test_a_short_matchday_is_caught(sched):
    bad = copy.deepcopy(sched)
    bad[5] = bad[5][:-1]
    assert any("17 fixtures" in p for p in schedule.validate(bad))


# --- joining to the feed -----------------------------------------------------

def test_the_alias_table_places_every_club_on_the_feeds_spelling(sched):
    """Built from the schedule's own names put through the table, so this
    tests the mechanism and the table's self-consistency without needing a
    live feed. Accents are folded; nothing else is guessed."""
    clubs = {f[end] for fx in sched.values() for f in fx
             for end in ("home", "away")}
    feed = {schedule.ALIASES.get(c, c) for c in clubs}
    out, missing = schedule.resolve(sched, feed)
    assert missing == []
    names = {f[end] for fx in out.values() for f in fx
             for end in ("home", "away")}
    assert names == feed
    assert schedule.validate(out) == []


def test_a_club_the_table_cannot_place_is_reported_not_dropped():
    """One renamed club must not cost the other thirty-five their horizon."""
    s = {1: [{"home": "Inter Milan", "away": "Narnia FC",
              "date": "2026-09-08"}]}
    out, missing = schedule.resolve(s, {"Inter", "Somewhere Else"})
    assert missing == ["Narnia FC"]
    assert out[1][0]["home"] == "Inter"
    assert out[1][0]["away"] == "Narnia FC"       # left as written


# --- the run -----------------------------------------------------------------

def test_every_club_has_exactly_one_fixture_in_each_matchday(sched):
    r = schedule.runs(sched, matchdays=[4, 5, 6])
    assert len(r) == 36
    for club, by_md in r.items():
        assert sorted(by_md) == [4, 5, 6]


def test_home_advantage_shows_up_in_the_same_pairing_both_ways():
    s = {1: [{"home": "A", "away": "B", "date": "2026-09-08"}],
         2: [{"home": "B", "away": "A", "date": "2026-10-13"}]}
    diff = schedule.difficulty(s, _pool(["A", "B"]))
    assert diff["A"][1]["scored"] > diff["A"][2]["scored"]
    assert diff["A"][1]["home"] is True and diff["A"][2]["home"] is False
    # One side's goals are the other side's concessions, same fixture.
    assert diff["A"][1]["scored"] == pytest.approx(diff["B"][1]["conceded"])


def test_a_stronger_attack_is_priced_above_a_weaker_one():
    s = {1: [{"home": "Strong", "away": "Weak", "date": "2026-09-08"}]}
    pool = (_pool(["Strong"], scored=20.0, conceded=4.0, price=11.0)
            + _pool(["Weak"], scored=4.0, conceded=20.0, price=4.0))
    diff = schedule.difficulty(s, pool)
    assert diff["Strong"][1]["scored"] > diff["Weak"][1]["scored"]
    assert diff["Strong"][1]["cs"] > diff["Weak"][1]["cs"]
    # Not pinned to a clamp: a test that passes on MAX_LAMBDA passes on
    # anything.
    from gaffer.ucl import strength
    assert diff["Strong"][1]["scored"] < strength.MAX_LAMBDA - 1e-6


def test_an_unchanging_run_does_not_read_as_back_loaded():
    """The halves are averages. Summing them subtracted a two-matchday total
    from a three-matchday one, which made thirty-four of thirty-six clubs
    look like clubs to sell."""
    s = {md: [{"home": "A", "away": f"B{md}", "date": f"2026-0{md}-08"}]
         for md in (1, 2, 3)}
    diff = schedule.difficulty(s, _pool(["A", "B1", "B2", "B3"]))
    rows = {r["club"]: r for r in schedule.summarise(diff)}
    turn = rows["A"]["back_scored"] - rows["A"]["front_scored"]
    assert abs(turn) < 1e-9
    assert rows["A"]["scored"] == pytest.approx(
        3 * diff["A"][1]["scored"])


def test_a_run_that_improves_reads_as_a_club_to_buy_later():
    s = {1: [{"home": "Hard", "away": "Mid", "date": "2026-09-08"}],
         2: [{"home": "Mid", "away": "Hard", "date": "2026-10-13"}],
         3: [{"home": "Mid", "away": "Soft", "date": "2026-10-20"}]}
    pool = (_pool(["Hard"], scored=16.0, conceded=2.0, price=11.0)
            + _pool(["Mid"], scored=12.0, conceded=12.0)
            + _pool(["Soft"], scored=4.0, conceded=24.0, price=4.0))
    rows = {r["club"]: r
            for r in schedule.summarise(schedule.difficulty(s, pool))}
    assert rows["Mid"]["back_scored"] > rows["Mid"]["front_scored"]


def test_the_table_renders_every_club_once(sched):
    sch, _ = schedule.resolve(sched, {f[e] for fx in sched.values()
                                      for f in fx for e in ("home", "away")})
    diff = schedule.difficulty(sch, _pool(sorted(
        {f[e] for fx in sch.values() for f in fx for e in ("home", "away")})))
    text = schedule.table(diff, matchdays=[4, 5, 6, 7, 8])
    body = [ln for ln in text.splitlines() if ln.startswith("  ")]
    assert len(body) == 37                        # header plus thirty-six
    assert "MD4" in text and "MD8" in text
