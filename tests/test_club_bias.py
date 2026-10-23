"""A manager's own view of a club, applied on top of the measurement.

The model sees what FPL publishes, which is the Premier League and nothing
else. On 21 October 2026 Manchester United lost 4-1 at Como, 2.04 expected
goals against 1.05, extending a winless run to seven across all competitions
— and the engine's view of them did not move by a thousandth, because none of
that happened in a competition it can see.
"""
from __future__ import annotations

import pytest

from gaffer import projections, rotation

CODES = {1: "ARS", 2: "MUN", 3: "LIV"}
ATK = {1: 1.10, 2: 0.86, 3: 0.95}
DFN = {1: 0.90, 2: 0.935, 3: 1.00}


def test_no_bias_changes_nothing():
    a, d, changed = projections.apply_club_bias(ATK, DFN, CODES, None)
    assert a == ATK and d == DFN and changed == []
    a, d, changed = projections.apply_club_bias(ATK, DFN, CODES, {})
    assert a == ATK and d == DFN and changed == []


def test_a_club_marked_down_scores_less_and_concedes_more():
    bias = {"MUN": {"attack": 0.98, "defence": 1.06}}
    a, d, _ = projections.apply_club_bias(ATK, DFN, CODES, bias)
    assert a[2] == pytest.approx(0.86 * 0.98)
    assert d[2] == pytest.approx(0.935 * 1.06)


def test_the_other_clubs_are_left_exactly_alone():
    """It is NOT re-centred. Re-centring would spread an opinion about one
    club across the other nineteen, which is not what anyone means by it."""
    bias = {"MUN": {"attack": 0.85, "defence": 1.15}}
    a, d, _ = projections.apply_club_bias(ATK, DFN, CODES, bias)
    assert a[1] == ATK[1] and a[3] == ATK[3]
    assert d[1] == DFN[1] and d[3] == DFN[3]


def test_half_an_entry_is_read_as_one_sided_not_as_zero():
    a, d, _ = projections.apply_club_bias(ATK, DFN, CODES,
                                          {"MUN": {"attack": 0.9}})
    assert a[2] == pytest.approx(0.86 * 0.9)
    assert d[2] == pytest.approx(0.935)


def test_it_reports_what_it_changed_so_the_run_can_say_so():
    """An override nobody is reminded of is indistinguishable from a model
    that agrees with you."""
    bias = {"MUN": {"attack": 0.98, "defence": 1.06}}
    _, _, changed = projections.apply_club_bias(ATK, DFN, CODES, bias)
    assert len(changed) == 1
    code, adj, before, after = changed[0]
    assert code == "MUN"
    assert adj == (0.98, 1.06)
    assert before == (0.86, 0.935)
    assert after[0] < before[0] and after[1] > before[1]


def test_a_club_that_does_not_exist_is_reported_rather_than_ignored():
    _, _, changed = projections.apply_club_bias(ATK, DFN, CODES,
                                                {"MANU": {"attack": 0.5}})
    assert changed == [("MANU", None, None, "no such club")]


def test_the_inputs_are_not_mutated():
    before = dict(ATK)
    projections.apply_club_bias(ATK, DFN, CODES, {"MUN": {"attack": 0.5}})
    assert ATK == before


# --- the list that went stale ------------------------------------------------

def _boot(codes):
    return {"teams": [{"id": i, "short_name": c} for i, c in enumerate(codes, 1)],
            "events": [{"id": 8, "deadline_time": "2026-10-23T17:30:00Z"}]}


def test_a_club_in_europe_but_missing_from_the_list_is_called_out(capsys):
    """The symptom of the real bug was nothing at all: Manchester United
    played eight European weeks projected as though the midweek were free."""
    boot = _boot(["ARS", "MUN", "LIV"])
    rotation.congestion(boot, european_teams=("ARS", "LIV"),
                        european_weeks=(8,), must_include=("ARS", "MUN", "LIV"))
    assert "MUN" in capsys.readouterr().out


def test_a_complete_list_says_nothing(capsys):
    boot = _boot(["ARS", "MUN", "LIV"])
    rotation.congestion(boot, european_teams=("ARS", "MUN", "LIV"),
                        european_weeks=(8,), must_include=("ARS", "MUN"))
    assert "missing from european_teams" not in capsys.readouterr().out


def test_the_haircut_actually_reaches_the_club_once_it_is_listed():
    boot = _boot(["ARS", "MUN"])
    without = rotation.congestion(boot, european_teams=("ARS",),
                                  european_weeks=(8,), haircut=0.12)
    with_mun = rotation.congestion(boot, european_teams=("ARS", "MUN"),
                                   european_weeks=(8,), haircut=0.12)
    assert without[2][8] == pytest.approx(1.0)
    assert with_mun[2][8] == pytest.approx(0.88)


def test_the_shipped_config_lists_every_english_club_in_the_champions_league():
    """The regression itself. A hand-typed list of eight goes stale the
    moment a ninth qualifies."""
    import yaml
    from gaffer.ucl.availability import ENGLISH
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    listed = set(cfg["european_teams"])
    assert set(ENGLISH) <= listed, sorted(set(ENGLISH) - listed)


def test_the_bias_reaches_the_projections_the_solver_reads():
    """The bug this nearly shipped as. `build` computes its own strengths, so
    biasing the copy main.py holds moved the number on the page and left
    every projection the solver reads exactly as it was. An override that
    looks applied and is not is worse than no override at all."""
    import inspect
    src = inspect.getsource(projections.build)
    assert "apply_club_bias" in src, (
        "build must bias the strengths it computes itself")
    assert "club_bias" in inspect.signature(projections.build).parameters
