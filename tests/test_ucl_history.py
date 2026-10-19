"""Differencing successive reads must give one matchday's minutes, or nothing."""
import json
from gaffer.ucl import history, project


def rows(**mins):
    return [{"id": k, "minutes": v} for k, v in mins.items()]


def test_the_first_read_for_a_matchday_wins(tmp_path):
    p = tmp_path / "log.json"
    log, wrote = history.record(rows(a=90.0), 3, path=str(p))
    assert wrote
    # A later read may be taken mid-matchday, when its totals are part-way
    # through a match; attributing those to the previous week is wrong.
    log, wrote = history.record(rows(a=140.0), 3, path=str(p))
    assert not wrote
    assert json.load(open(p))["3"]["a"] == 90.0


def test_consecutive_reads_bracket_one_matchday():
    log = {"3": {"a": 106.0, "b": 180.0}, "4": {"a": 112.0, "b": 270.0}}
    got = history.per_matchday(log, upto=3)
    assert got["a"] == [(3, 6.0)]
    assert got["b"] == [(3, 90.0)]


def test_a_gap_in_the_log_is_not_attributed():
    """Reads for matchdays 2 and 5 say nothing about matchday 2 alone."""
    log = {"2": {"a": 0.0}, "5": {"a": 270.0}}
    assert history.per_matchday(log, upto=4) == {}


def test_minutes_only_grow_so_a_fall_is_discarded():
    log = {"3": {"a": 180.0}, "4": {"a": 90.0}}
    assert history.per_matchday(log, upto=3) == {}


def test_a_player_absent_from_the_earlier_read_starts_from_zero():
    log = {"3": {}, "4": {"new": 90.0}}
    assert history.per_matchday(log, upto=3)["new"] == [(3, 90.0)]


def test_attach_only_touches_players_it_has_a_log_for():
    pool = [{"id": "a", "minutes": 112.0}, {"id": "b", "minutes": 50.0}]
    n = history.attach(pool, {"a": [(3, 6.0)]})
    assert n == 1
    assert pool[0]["match_minutes"] == [6.0]
    assert "match_minutes" not in pool[1]


def test_per_match_minutes_beat_the_minutes_share():
    """Guirassy's 120 of 180 reads as 0.67 either way; the log says which.

    One start and one blank is a rotation risk. Two starts cut short on sixty
    is a first-choice forward, and the share cannot tell them apart.
    """
    base = dict(id="1", minutes=120.0, matches=2, trained="In contention",
                status="")
    rotated = project.start_prob(dict(base, match_minutes=[90.0, 30.0]))
    nailed = project.start_prob(dict(base, match_minutes=[60.0, 60.0]))
    assert nailed > rotated
    assert nailed > 0.75
    assert rotated < 0.75


def test_a_suspension_still_overrides_a_perfect_log():
    p = {"id": "1", "minutes": 180.0, "matches": 2, "trained": "Suspended",
         "status": "", "match_minutes": [90.0, 90.0]}
    assert project.start_prob(p) == 0.0


def test_a_flag_scales_a_log_based_probability_too():
    p = {"id": "1", "minutes": 180.0, "matches": 2,
         "trained": "In contention", "status": "",
         "match_minutes": [90.0, 90.0]}
    full = project.start_prob(p)
    assert abs(project.start_prob(dict(p, availability=0.5)) - full * 0.5) < 1e-9


def test_an_offline_replay_does_not_write_the_log(tmp_path, monkeypatch):
    """The test suite drives run() with a synthetic pool. The first version of
    this recorded on every run, so pytest filed 114 invented players under
    matchday 3 in the repository's own log — and first-write-wins would have
    kept them and differenced them at matchday 4."""
    import os
    from gaffer.ucl import main as uclmain
    path = tmp_path / "minutes_log.json"
    monkeypatch.setattr(history, "PATH", str(path))
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        uclmain.run(3, {"squad": [], "use_fpl_flags": False},
                    offline=True, n=200, time_limit=10)
    except SystemExit:
        pass
    except Exception:
        pass
    finally:
        os.chdir(cwd)
    assert not path.exists()
