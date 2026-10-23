"""A plan is a photograph, and team news moves.

Gameweek 8 was solved at 09:29 with Stach at 5.13 expected points and Saka a
fifty-fifty. By 13:53 Stach had a foot injury and was worth 2.88, and Saka
had gone to nought with no return date. The arithmetic was never wrong; the
paper had aged, and nothing in the output said so.
"""
from __future__ import annotations

import datetime

import pytest

from gaffer import freshness


def boot(*players):
    return {"elements": [dict({"id": i, "web_name": n, "status": "a",
                               "chance_of_playing_next_round": None,
                               "news": ""}, **kw)
                         for i, (n, kw) in enumerate(players, 1)]}


FIT = ("fit", {})
DOUBT = ("doubt", {"status": "d", "chance_of_playing_next_round": 50,
                   "news": "Foot injury - 50% chance of playing"})
OUT = ("out", {"status": "i", "chance_of_playing_next_round": 0,
               "news": "Thigh injury - Unknown return date"})


# --- the fingerprint ---------------------------------------------------------

def test_it_records_only_the_players_the_plan_rests_on():
    b = boot(FIT, DOUBT, OUT)
    f = freshness.fingerprint(b, ["1", "3"])
    assert sorted(f) == ["1", "3"]
    assert f["3"]["status"] == "i"


def test_the_decision_set_is_your_squad_plus_everyone_the_plan_trades():
    plan = {"weeks": [
        {"in_players": [{"id": 77}], "out_players": [{"id": 88}]},
        {"in_players": [{"id": 99}], "out_players": []},
    ]}
    ids = freshness.decision_set(plan, [1, 2, 3])
    assert ids == {"1", "2", "3", "77", "88", "99"}


def test_a_plan_with_no_transfers_still_watches_the_squad():
    assert freshness.decision_set({"weeks": []}, [4, 5]) == {"4", "5"}


# --- what changed ------------------------------------------------------------

def test_nothing_moving_reads_as_fresh():
    b = boot(FIT, FIT)
    before = freshness.fingerprint(b, ["1", "2"])
    assert freshness.compare(before, before) == []
    assert freshness.verdict([])[0] == "fresh"


def test_the_stach_case_is_caught():
    """Fit at the solve, a fifty-fifty foot injury by the deadline."""
    before = freshness.fingerprint(boot(FIT), ["1"])
    after = freshness.fingerprint(boot(DOUBT), ["1"])
    changes = freshness.compare(before, after)
    assert len(changes) == 1
    assert changes[0]["kind"] == "status" and changes[0]["severity"] > 0
    assert "Foot injury" in changes[0]["news"]
    code, why = freshness.verdict(changes)
    assert code == "stale"
    assert "re-solve" in why


def test_the_saka_case_is_caught_even_though_he_was_already_flagged():
    """He was a 50% doubt at the solve and a 0% unknown-return by the
    deadline. The status letter alone does not move far enough to notice —
    the chance does, and getting worse while already flagged is the case
    most likely to be waved through."""
    was = ("saka", {"status": "d", "chance_of_playing_next_round": 50,
                    "news": "Thigh injury - 50% chance of playing"})
    before = freshness.fingerprint(boot(was), ["1"])
    after = freshness.fingerprint(boot(OUT), ["1"])
    changes = freshness.compare(before, after)
    assert changes and changes[0]["severity"] > 0
    assert freshness.verdict(changes)[0] == "stale"


def test_a_player_getting_better_does_not_make_the_plan_stale():
    """It may no longer be optimal, but it is still safe to act on, and
    crying wolf on good news is how a check gets ignored."""
    before = freshness.fingerprint(boot(DOUBT), ["1"])
    after = freshness.fingerprint(boot(FIT), ["1"])
    changes = freshness.compare(before, after)
    assert changes and changes[0]["severity"] < 0
    assert freshness.verdict(changes)[0] == "check"


def test_news_appearing_on_a_fit_player_is_noticed_but_not_fatal():
    knock = ("knock", {"news": "Knock - expected back soon"})
    before = freshness.fingerprint(boot(FIT), ["1"])
    after = freshness.fingerprint(boot(knock), ["1"])
    changes = freshness.compare(before, after)
    assert changes and changes[0]["kind"] == "news"
    assert freshness.verdict(changes)[0] == "stale"


def test_reworded_news_on_an_unchanged_flag_is_not_an_alarm():
    """The wording is rewritten constantly without the decision changing,
    and a check that fires on prose is a check nobody reads."""
    a = ("x", {"status": "d", "chance_of_playing_next_round": 75,
               "news": "Knock - 75% chance of playing"})
    b = ("x", {"status": "d", "chance_of_playing_next_round": 75,
               "news": "Minor knock - 75% chance of playing"})
    before = freshness.fingerprint(boot(a), ["1"])
    after = freshness.fingerprint(boot(b), ["1"])
    assert freshness.compare(before, after) == []


def test_the_worst_change_is_reported_first():
    before = freshness.fingerprint(boot(FIT, FIT), ["1", "2"])
    after = freshness.fingerprint(
        boot(("a", {"news": "Knock"}), OUT), ["1", "2"])
    changes = freshness.compare(before, after)
    assert changes[0]["name"] == "out"


def test_a_player_who_vanished_from_the_feed_is_skipped_not_crashed_on():
    before = freshness.fingerprint(boot(FIT, FIT), ["1", "2"])
    after = freshness.fingerprint(boot(FIT), ["1"])
    assert freshness.compare(before, after) == []


# --- age and the whole report -----------------------------------------------

def test_age_is_measured_in_hours():
    now = datetime.datetime(2026, 10, 23, 13, 53, tzinfo=datetime.timezone.utc)
    assert freshness.age("2026-10-23T09:29:00+00:00", now) == pytest.approx(4.4, abs=0.05)
    assert freshness.age("2026-10-23T09:29:00Z", now) == pytest.approx(4.4, abs=0.05)
    assert freshness.age(None, now) is None
    assert freshness.age("not a time", now) is None


def test_a_naive_timestamp_is_read_as_utc_rather_than_crashing():
    now = datetime.datetime(2026, 10, 23, 12, 0, tzinfo=datetime.timezone.utc)
    assert freshness.age("2026-10-23T10:00:00", now) == pytest.approx(2.0)


def test_a_plan_from_before_this_check_existed_says_so():
    rep = freshness.report({"solved_at": "2026-10-23T09:29:00Z"}, boot(FIT))
    assert rep["code"] == "unknown"
    assert "predates" in rep["why"]


def test_the_report_reads_as_something_a_person_can_act_on():
    state = {"solved_at": "2026-10-23T09:29:00+00:00",
             "availability": freshness.fingerprint(boot(FIT), ["1"])}
    rep = freshness.report(state, boot(DOUBT),
                           now=datetime.datetime(2026, 10, 23, 13, 53,
                                                 tzinfo=datetime.timezone.utc))
    text = "\n".join(freshness.lines(rep))
    assert "stale" in text and "4.4h old" in text
    assert "Foot injury" in text
