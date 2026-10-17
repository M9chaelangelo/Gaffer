"""The UCL engine: scoring, strength, projection, optimiser, armband."""
from __future__ import annotations

import numpy as np
import pytest

from gaffer import games
from gaffer.ucl import main as uclmain
from gaffer.ucl import plan, project, strength
from gaffer.ucl.rules import score


# --- scoring -----------------------------------------------------------------

def test_recoveries_pay_from_the_first_third_with_no_threshold():
    # FPL's defensive contribution is a cliff at 10; UCL's is a staircase.
    assert score({"minutes": 90, "recoveries": 2}, "MID") == 2      # 1 + 1
    assert score({"minutes": 90, "recoveries": 3}, "MID") == 3
    assert score({"minutes": 90, "recoveries": 9}, "MID") == 5


def test_a_keeper_who_scores_gets_six_not_four():
    assert score({"minutes": 90, "goals": 1}, "GK") == 8            # 2 + 6
    assert score({"minutes": 90, "goals": 1}, "FWD") == 6           # 2 + 4


def test_a_missing_clean_sheet_field_is_not_a_clean_sheet():
    assert score({"minutes": 90}, "DEF") == 2
    assert score({"minutes": 90, "goals_conceded": 0}, "DEF") == 6
    assert score({"minutes": 90, "clean_sheet": True}, "DEF") == 6


def test_the_game_registry_knows_ucl_has_no_vice_captain():
    g = games.get("ucl")
    assert g.has_vice_captain is False
    assert g.armband_movable_midround is True
    assert g.free_transfers_per_round == 2
    assert g.squad_size == 15


# --- fixtures ----------------------------------------------------------------

def _player(pid, name, pos, team, price, **kw):
    base = dict(
        id=str(pid), name=name, full_name=name, pos=pos, team=team,
        team_id=team, code=team[:3].upper(), price=price, own=5.0,
        transfers_in=0, transfers_out=0, total_points=10, avg_points=5,
        minutes=180, team_played=2, goals=0, assists=0, recoveries=6,
        saves=0, clean_sheets=1, conceded=2, yellow=0, red=0, motm=0,
        goals90=0.5, assists90=0.5, recoveries90=3.0, saves90=0.0,
        trained="In contention to start next game", status="", fixtures=[])
    base.update(kw)
    return base


def _pool():
    """Six clubs, a full complement each, three fixtures over two days.

    Six is the smallest number that can field a legal fifteen under the
    three-per-club cap with room to choose.
    """
    rows = []
    n = 0
    for team in ("Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"):
        for pos, count, price in (("GK", 3, 5.0), ("DEF", 6, 5.5),
                                  ("MID", 6, 7.0), ("FWD", 4, 8.0)):
            for k in range(count):
                n += 1
                # Three goals a side over two games, scored by the front two:
                # a Champions League scoreline, not a cricket one. A pool
                # where everyone scores makes every goal rate hit the clamp
                # and home advantage then cannot show at all.
                goals = 2 if (pos == "FWD" and k == 0) else (
                    1 if (pos == "MID" and k == 0) else 0)
                rows.append(_player(n, f"{team[0]}{pos}{k}", pos, team,
                                    price + k * 0.5, goals=goals,
                                    goals90=goals / 2.0,
                                    assists90=0.25 if pos in ("MID", "FWD") else 0.05,
                                    saves90=2.5 if pos == "GK" else 0.0))
    return rows


FIXTURES = [
    {"matchday": "3", "home": "Alpha", "away": "Beta",
     "kickoff": "2026-10-20T19:00:00"},
    {"matchday": "3", "home": "Epsilon", "away": "Zeta",
     "kickoff": "2026-10-20T19:00:00"},
    {"matchday": "3", "home": "Gamma", "away": "Delta",
     "kickoff": "2026-10-21T19:00:00"},
]


# --- strength ----------------------------------------------------------------

def test_goals_against_are_counted_from_the_keepers_only():
    # Every outfield row carries `gC` too. Summing the column over a squad
    # counts each goal conceded about elevenfold.
    rows = _pool()
    agg = strength.aggregate(rows)
    assert agg["Alpha"]["conceded"] == pytest.approx(3 * 2)   # three keepers
    assert agg["Alpha"]["scored"] == pytest.approx(3)         # the front two


def test_ratings_stay_inside_the_clamp_and_a_fixture_gets_two_rates():
    rate, mu = strength.ratings(_pool())
    for r in rate.values():
        assert strength.MIN_RATIO <= r["attack"] <= strength.MAX_RATIO
        assert strength.MIN_RATIO <= r["defence"] <= strength.MAX_RATIO
    lh, la = strength.lambdas(rate, "Alpha", "Beta", mu)
    assert lh > la          # same ratings either side, so home advantage shows


# --- projection --------------------------------------------------------------

def test_configured_odds_are_read_in_either_orientation():
    assert project._odds_for({"Alpha v Beta": [0.5, 0.25, 0.25]},
                             "Alpha", "Beta") == (0.5, 0.25, 0.25)
    # the same line, with the fixture the other way round
    assert project._odds_for({"Alpha v Beta": [0.5, 0.25, 0.25]},
                             "Beta", "Alpha") == (0.25, 0.25, 0.5)
    assert project._odds_for({"Alpha v Beta": [0.5, 0.25, 0.25]},
                             "Gamma", "Delta") is None


def test_odds_beat_the_model_and_the_source_says_which():
    rows = _pool()
    fx, _, _ = project.fixture_rates(FIXTURES, rows,
                                     odds={"Alpha v Beta": [0.80, 0.12, 0.08]})
    byname = {f["home"]: f for f in fx}
    assert byname["Alpha"]["source"] == "odds"
    assert byname["Gamma"]["source"] == "model"
    # a heavy home favourite must come out with the bigger goal rate
    assert byname["Alpha"]["lh"] > byname["Alpha"]["la"] + 0.5


def test_a_player_whose_club_has_no_fixture_scores_nothing():
    rows = _pool() + [_player(999, "Nobody", "MID", "Omega", 7.0)]
    samples, _ = project.simulate(rows, FIXTURES, n=2000)
    assert samples[:, -1].max() == 0


def test_a_suspended_player_scores_exactly_nothing():
    # A suspension is a certainty, not a doubt. Giving Olise a one-in-ten
    # chance of playing also gives him a tenth of a haul, which is enough to
    # keep a banned player in an eleven.
    rows = _pool()
    rows[0]["trained"] = "Suspended"
    samples, _ = project.simulate(rows, FIXTURES, n=4000)
    assert project.start_prob(rows[0]) == 0.0
    assert samples[:, 0].max() == 0


def test_a_doubt_is_a_coin_toss_not_a_certainty_either_way():
    p = _player(1, "Doubtful", "MID", "Alpha", 7.0,
                trained="Doubt over fitness")
    assert project.start_prob(p) == project.DOUBT_START
    assert 0.0 < project.start_prob(p) < 0.5


def test_forcing_a_player_out_overrides_a_clean_feed_row():
    p = _player(1, "Clean", "MID", "Alpha", 7.0)
    assert project.start_prob(p) > 0.5
    p["forced_out"] = True
    assert project.start_prob(p) == 0.0


# --- optimiser ---------------------------------------------------------------

def _projected(n=4000):
    rows = _pool()
    samples, _ = project.simulate(rows, FIXTURES, n=n)
    return project.summarise(rows, samples), samples, rows


def test_a_drafted_squad_obeys_shape_budget_and_the_club_cap():
    rows, _, _ = _projected()
    out = plan.optimise(rows, time_limit=60)
    assert out["optimal"], out["status"]
    assert len(out["squad"]) == 15 and len(out["xi"]) == 11
    shape = {}
    for r in out["squad"]:
        shape[r["pos"]] = shape.get(r["pos"], 0) + 1
    assert shape == {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    perclub = {}
    for r in out["squad"]:
        perclub[r["team"]] = perclub.get(r["team"], 0) + 1
    assert max(perclub.values()) <= 3
    assert out["spend"] <= 100.0 + 1e-6


def test_transfers_are_capped_at_the_free_allowance():
    rows, _, _ = _projected()
    # a deliberately poor starting squad: the cheapest legal fifteen
    cheap = sorted(rows, key=lambda r: r["price"])
    owned, shape = [], {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    perclub = {}
    for r in cheap:
        if shape.get(r["pos"], 0) and perclub.get(r["team"], 0) < 3:
            owned.append(r["id"])
            shape[r["pos"]] -= 1
            perclub[r["team"]] = perclub.get(r["team"], 0) + 1
    assert len(owned) == 15
    out = plan.optimise(rows, owned_ids=owned, bank=50.0, free_transfers=2,
                        time_limit=60)
    assert out["optimal"], out["status"]
    assert len(out["in"]) == len(out["out"]) <= 2


def test_a_locked_player_survives_a_transfer_plan_that_would_sell_him():
    rows, _, _ = _projected()
    cheap = sorted(rows, key=lambda r: r["price"])
    owned, shape, perclub = [], {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}, {}
    for r in cheap:
        if shape.get(r["pos"], 0) and perclub.get(r["team"], 0) < 3:
            owned.append(r["id"])
            shape[r["pos"]] -= 1
            perclub[r["team"]] = perclub.get(r["team"], 0) + 1
    keep = owned[0]
    out = plan.optimise(rows, owned_ids=owned, bank=50.0, locked=[keep],
                        time_limit=60)
    assert keep in {r["id"] for r in out["squad"]}


# --- the armband -------------------------------------------------------------

def test_moving_the_armband_is_worth_something_on_day_one_and_nothing_on_day_two():
    rows, samples, pool = _projected(8000)
    columns = {r["id"]: j for j, r in enumerate(pool)}
    out = plan.optimise(rows, time_limit=60)
    band = plan.armband(out["xi"], samples, columns, FIXTURES)
    assert band
    by_day = {}
    for a in band:
        by_day.setdefault(a["day"], []).append(a)
    early, late = min(by_day), max(by_day)
    # The option is real on the earlier day and absent on the last one, and
    # it can never be negative: you only move when moving is better.
    assert all(a["option_gain"] > 0 for a in by_day[early])
    assert all(a["option_gain"] == 0 for a in by_day[late])
    assert all(a["armband_ep"] >= a["captain_ep"] - 1e-9 for a in band)


def test_the_armband_is_never_worth_less_than_the_plain_captaincy():
    # E[max(s, c)] >= max(E[s], c) by Jensen; this is the arithmetic that
    # makes a day-one captain worth more than an equal day-two one.
    s = np.array([0.0, 2.0, 4.0, 20.0])
    assert np.maximum(s, 5.0).mean() >= max(s.mean(), 5.0)


# --- name matching -----------------------------------------------------------

def test_names_match_across_accents_initials_and_spacing():
    rows = [_player(1, "B. Fernandes", "MID", "Alpha", 8.0),
            _player(2, "K. Mbappé", "FWD", "Beta", 11.0)]
    rows[0]["full_name"] = "Bruno Fernandes"
    assert uclmain._match(rows, "B.Fernandes")["id"] == "1"
    assert uclmain._match(rows, "Bruno Fernandes")["id"] == "1"
    assert uclmain._match(rows, "Mbappe")["id"] == "2"


def test_an_ambiguous_name_matches_nothing_rather_than_the_wrong_man():
    rows = [_player(1, "B. Silva", "MID", "Alpha", 8.0),
            _player(2, "T. Silva", "DEF", "Beta", 6.0)]
    assert uclmain._match(rows, "Silva") is None
    assert uclmain._match(rows, "B. Silva")["id"] == "1"


def test_unmatched_squad_names_are_reported_not_dropped_silently():
    rows = [_player(1, "B. Silva", "MID", "Alpha", 8.0)]
    got = uclmain.resolve(rows, ["B. Silva", "Nobody At All"])
    assert got["ids"] == ["1"]
    assert got["named"] == ["B. Silva"]


# --- end to end --------------------------------------------------------------

def test_the_engine_produces_a_whole_plan_from_a_snapshot(tmp_path, monkeypatch):
    """The thing that had never happened: a UCL plan, start to finish.

    The UCL side sat half-wired for weeks and the advice actually given came
    from a standalone script. A test that only exercises the pieces would
    have passed throughout.
    """
    import json

    pool = _pool()
    snap = tmp_path / "snapshot.json"
    snap.write_text(json.dumps({"matchday": 3, "fixtures": FIXTURES,
                                "players": pool, "problems": []}))
    monkeypatch.setattr(uclmain, "SNAPSHOT", str(snap))

    owned = [r["name"] for r in pool[:2] if r["pos"] == "GK"]
    cfg = {"squad": [], "bank": 0.0, "out": [pool[5]["name"]],
           "odds": {"Alpha v Beta": [0.6, 0.22, 0.18]}}
    out = uclmain.run(3, cfg, offline=True, n=3000, time_limit=60)

    assert out["source"] == "snapshot"
    assert out["plan"]["optimal"], out["plan"]["status"]
    assert len(out["plan"]["xi"]) == 11
    assert len(out["plan"]["squad"]) == 15
    assert out["armband"], "a plan with no armband is not a UCL plan"
    assert out["armband"][0]["armband_ep"] >= out["armband"][0]["captain_ep"]
    # the player marked out is projected at zero and so cannot be started
    assert pool[5]["id"] in out["forced_out"]
    assert pool[5]["id"] not in {r["id"] for r in out["plan"]["xi"]}
    # and the configured odds were used for the fixture they name
    assert [f for f in out["fixtures"] if f["home"] == "Alpha"][0]["source"] == "odds"
    assert owned is not None


def test_a_report_of_a_whole_plan_prints_without_raising(tmp_path, monkeypatch, capsys):
    # Four separate times on the FPL side the computation was right and the
    # carriage was wrong. A report is code too.
    import json

    pool = _pool()
    snap = tmp_path / "snapshot.json"
    snap.write_text(json.dumps({"matchday": 3, "fixtures": FIXTURES,
                                "players": pool, "problems": []}))
    monkeypatch.setattr(uclmain, "SNAPSHOT", str(snap))
    out = uclmain.run(3, {"squad": []}, offline=True, n=1500, time_limit=60)
    uclmain.report(out)
    text = capsys.readouterr().out
    assert "starting eleven" in text and "armband" in text
    assert "None" not in text
    # With no squad configured this is a draft, and calling fifteen picks
    # "transfers in" is how a reader ends up taking fifteen hits.
    assert "draft of the best fifteen" in text
    assert "OUT " not in text


def test_a_failed_fetch_names_every_attempt_rather_than_the_last(monkeypatch):
    """Six minutes of timeouts arrived as one sentence naming no cause.

    The first version re-raised only the last exception and the caller
    turned it into "no live feed", so the diagnosis was collected and
    discarded — the same shape as a print that deletes its own data.
    """
    import urllib.error

    from gaffer.ucl import fetch

    monkeypatch.setattr(fetch.time, "sleep", lambda _s: None)

    def boom(*_a, **_k):
        raise urllib.error.URLError("timed out")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError) as e:
        fetch._get("https://example.invalid/feed")
    msg = str(e.value)
    assert "URLError" in msg and "timed out" in msg
    # every header set it tried is named, not just the last
    for h in fetch.HEADER_SETS:
        assert h.get("User-Agent", "(default)") in msg


def test_the_feed_is_never_asked_for_while_pretending_to_be_a_browser():
    """Measured, not styled. Same URL, same second, on a runner:

        curl's own headers       http 200, 0.79s to first byte, 2.06 MB
        Chrome UA over HTTP/2    stream reset
        Chrome UA over HTTP/1.1  45-second timeout

    The edge fingerprints the mismatch. Anyone who "fixes" this file by
    adding a realistic User-Agent will break it, so the rule is a test.
    """
    from gaffer.ucl import fetch

    for headers in fetch.HEADER_SETS:
        blob = " ".join(headers.values()).lower()
        assert "mozilla" not in blob and "chrome" not in blob
        assert "referer" not in {k.lower() for k in headers}
        assert "origin" not in {k.lower() for k in headers}


def test_the_header_set_that_worked_is_recorded(monkeypatch):
    """So a change at UEFA's edge shows up as data, not as a guess."""
    import contextlib
    import io

    from gaffer.ucl import fetch

    monkeypatch.setattr(fetch, "LAST_HEADERS", None)
    monkeypatch.setattr(fetch.time, "sleep", lambda _s: None)
    tried = []

    @contextlib.contextmanager
    def fake(req, timeout=None):
        tried.append(req.get_header("User-agent"))
        if len(tried) <= 2:            # the curl set fails both its attempts
            raise OSError("refused")
        yield io.BytesIO(b'{"ok": true}')

    monkeypatch.setattr(fetch.urllib.request, "urlopen", fake)

    assert fetch._get("https://example.invalid/feed") == {"ok": True}
    assert tried[0] == "curl/8.5.0"
    assert fetch.LAST_HEADERS == "(default)"
