"""The solver has to exist before the model is worth writing.

PuLP 4.0 removed both halves of how this project talked to it. The variable
constructor was a rename and the call sites moved. The solver was not: the CBC
that used to ship inside the wheel now has to be installed separately, and the
replacement class has nothing to fall back on when it isn't.

That is the failure worth a test. A missing solver does not raise — it returns
no answer, and the wildcard drafter then reports "no legal squad for 100.0m",
which reads exactly like a budget that doesn't work. A dependency change would
arrive disguised as a football problem.
"""
import pytest

import pulp

from gaffer import solver


@pytest.fixture(autouse=True)
def _forget_the_choice():
    """The backend is cached on purpose; these tests all change the answer."""
    solver._chosen = None
    yield
    solver._chosen = None


def test_it_finds_a_solver_here():
    assert solver.backend().__name__ in solver.BACKENDS


def test_it_prefers_the_backend_that_survives_pulp_4(monkeypatch):
    """COIN_CMD first, whenever it can actually reach a binary."""
    monkeypatch.setattr(solver, "_probe", lambda cls: True)
    assert solver.backend() is pulp.COIN_CMD


def test_it_falls_back_when_coin_cmd_has_no_binary(monkeypatch):
    """PuLP 3 without the extra: COIN_CMD is importable but finds nothing, and
    the bundled binary is the only one there is.

    `PULP_CBC_CMD` is set here rather than assumed, because under PuLP 4 it is
    gone and this test would otherwise be asserting the installed version
    instead of the fallback."""
    bundled = type("PULP_CBC_CMD", (), {})
    monkeypatch.setattr(pulp, "PULP_CBC_CMD", bundled, raising=False)
    monkeypatch.setattr(solver, "_probe",
                        lambda cls: cls is not pulp.COIN_CMD)
    assert solver.backend() is bundled


def test_a_backend_pulp_no_longer_has_is_skipped(monkeypatch):
    """PULP_CBC_CMD is gone in 4.0. `getattr` must not be an AttributeError."""
    monkeypatch.delattr(pulp, "PULP_CBC_CMD", raising=False)
    monkeypatch.setattr(solver, "_probe", lambda cls: True)
    assert solver.backend() is pulp.COIN_CMD


def test_no_solver_at_all_says_so_out_loud(monkeypatch):
    """The whole point. Not a wrong answer, not an empty squad — a sentence
    naming the install that fixes it."""
    monkeypatch.setattr(solver, "_probe", lambda cls: False)
    with pytest.raises(RuntimeError, match=r"pulp\[cbc\]"):
        solver.backend()


def test_probe_is_not_fooled_by_a_class_that_constructs(monkeypatch):
    """`COIN_CMD(msg=0)` builds happily with no cbc anywhere. Availability is
    the question, not construction."""
    class Constructs:
        def __init__(self, **kw):
            pass

        def available(self):
            return None            # what COIN_CMD really returns with no cbc

    assert solver._probe(Constructs) is False


def test_probe_survives_a_backend_that_cannot_be_built():
    class Explodes:
        def __init__(self, **kw):
            raise OSError("no solver directory")

    assert solver._probe(Explodes) is False


def test_the_status_comes_off_the_return_value_not_the_model():
    """PuLP 4 removed `LpProblem.status` and returns a stats object instead.
    PuLP 3 returns the integer. Reading the return value is the only thing that
    works on both."""
    class Stats:
        status = 1

    class Model:
        def solve(self, s):
            return Stats()

    assert solver.status_name(getattr(Stats(), "status", Stats())) == "Optimal"
    assert solver.status_name(getattr(1, "status", 1)) == "Optimal"


def test_solve_returns_the_status_name_and_the_answer():
    m = pulp.LpProblem("t", pulp.LpMaximize)
    a = m.add_variable("a", cat="Binary")
    b = m.add_variable("b", cat="Binary")
    m += 3 * a + 2 * b
    m += a + b <= 1
    assert solver.solve(m, 10) == "Optimal"
    assert a.value() == 1 and b.value() == 0


def test_the_choice_is_made_once(monkeypatch):
    """Probing shells out to look for a binary. Doing that per solve would cost
    more than some of the solves."""
    calls = []
    monkeypatch.setattr(solver, "_probe",
                        lambda cls: (calls.append(cls), True)[1])
    solver.backend()
    solver.backend()
    assert len(calls) == 1


def test_the_status_is_a_word_on_pulp_3():
    """An int and a lookup table."""
    assert solver.status_name(1) == "Optimal"
    assert solver.status_name(-1) == "Infeasible"
    assert solver.status_name(0) == "Not Solved"


def test_the_status_is_the_same_word_on_pulp_4s_enum():
    """The real shape: PuLP 4 returns `LpSolveStatus.NotSolved`, where PuLP 3's
    table said "Not Solved". Spelled `Notsolved`, the wildcard drafter's
    `status == "Optimal"` still works — but every other verdict a human reads
    in an error message is subtly wrong, and `TimeLimit` is one of them."""
    import enum

    class LpSolveStatus(enum.Enum):          # as PuLP 4.0 declares it
        NotSolved = 0
        Optimal = 1
        Infeasible = -1
        TimeLimit = -4

    assert solver.status_name(LpSolveStatus.Optimal) == "Optimal"
    assert solver.status_name(LpSolveStatus.NotSolved) == "Not Solved"
    assert solver.status_name(LpSolveStatus.Infeasible) == "Infeasible"
    assert solver.status_name(LpSolveStatus.TimeLimit) == "Time Limit"


def test_a_screaming_enum_spells_it_the_same_way():
    """Belt and braces: if the names are ever shouted, the words still match."""
    import enum

    class Shouted(enum.Enum):
        OPTIMAL = 1
        NOT_SOLVED = 0

    assert solver.status_name(Shouted.OPTIMAL) == "Optimal"
    assert solver.status_name(Shouted.NOT_SOLVED) == "Not Solved"


def test_an_unknown_status_is_shown_rather_than_swallowed():
    """A number nobody recognises still has to reach the error message."""
    assert solver.status_name(-99) == "-99"


def test_a_real_solve_survives_pulp_4(monkeypatch):
    """Everything 4.0 removed, removed — and a model solved anyway.

    This is the test that would have caught the break. All three CI failures
    were the same shape: a name that only exists on PuLP 3 and that no normal
    run on PuLP 3 can miss. `cat=` on the constructor, then `pulp.LpStatus`,
    then `LpProblem.status` — each one found by a runner on the new major
    rather than by anything here.
    """
    monkeypatch.delattr(pulp, "LpStatus", raising=False)
    monkeypatch.delattr(pulp, "PULP_CBC_CMD", raising=False)
    monkeypatch.delattr(type(pulp.LpProblem("x", pulp.LpMaximize)), "status",
                        raising=False)
    solver._chosen = None

    m = pulp.LpProblem("pulp4", pulp.LpMaximize)
    a = m.add_variable("a", cat="Binary")
    b = m.add_variable("b", lowBound=0, upBound=4, cat="Integer")
    m += 5 * a + b
    m += a + b <= 3

    assert solver.solve(m, 10) == "Optimal"
    assert a.value() == 1 and b.value() == 2


def test_solver_never_plans_a_chip_already_played():
    """The sweep honoured chips_used; the solver did not.

    On 13 Oct it planned a Wildcard in GW8 for an entry that had played it in
    GW6 — and a week the solver reserves for a chip that cannot be played is a
    week it declines to use for anything real.
    """
    import inspect
    from gaffer import optimise

    sig = inspect.signature(optimise.plan)
    assert sig.parameters["chips_available"].default is None, (
        "chips_available must default to config, not to all four chips"
    )

    src = inspect.getsource(optimise.plan)
    assert "planner.chips_used(cfg)" in src, \
        "plan() must read the spent chips from config"
    assert "planner.chips_used_second(cfg)" in src, \
        "and must read the second-half set separately — they are two sets"


def test_bench_boost_week_reports_all_fifteen_and_a_real_eleven():
    """Two defects that appeared together in the GW7 solve of 15 Oct 2026.

    The week's `ep` summed the eleven only, so the Bench Boost week printed
    60.2 against a true 82.5 and read as the WORST of the five weeks when it
    was the best. And because every one of the fifteen scores under the chip,
    the solver was indifferent to the split and returned an arbitrary one —
    it benched Gross (7.42) and B.Fernandes (5.92) to start Schade (2.64, a
    shoulder injury). Harmless to the model, ruinous to a reader who takes the
    row for a team sheet.
    """
    import inspect
    from gaffer import optimise

    src = inspect.getsource(optimise.plan)
    assert "xi + bench if boosted else xi" in src, \
        "a Bench Boost week must score the bench too"
    assert "planner.best_xi(squad_g" in src, \
        "a Bench Boost week must still report a real eleven"


def test_the_solver_may_plan_a_second_half_chip_after_spending_the_first():
    """Two sets of chips, one per half — the solver constrained one per horizon.

    With a horizon straddling GW19 and the Wildcard spent in the first half,
    the old constraint `sum(wc) <= 0` forbade the second-half Wildcard the
    manager actually holds.
    """
    import inspect
    from gaffer import optimise

    src = inspect.getsource(optimise.plan)
    assert "planner.half_of(g)" in src, \
        "chip limits must be applied per half, not across the horizon"
    assert "spent_second" in src


def test_the_solver_is_penalised_for_owning_a_player_for_one_week():
    """At an eight-week horizon the line bought Dewsbury-Hall in GW10 and
    sold him in GW11, then bought Cherki in GW11 and sold him in GW12 —
    each move optimal alone, together two transfers spent for one week of
    ownership each. A transfer's gain was priced and its holding period
    was not.
    """
    import inspect

    from gaffer import optimise

    src = inspect.getsource(optimise.plan)
    assert "churn" in src, "no churn penalty in the solver"
    # bought AND sold inside the horizon is what must be caught
    assert 'pulp.lpSum(tin[(i, g)] for g in gws)\n                          + pulp.lpSum(tout[(i, g)] for g in gws) - 1' in src \
        or ("tin" in src and "tout" in src and "churn[i] >=" in src)
    assert 'cfg.get("churn_cost"' in src, "churn_cost must be configurable"
