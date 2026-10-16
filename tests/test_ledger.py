"""The prediction ledger — the engine's only way of being scored."""
import tempfile

from gaffer.learn import ledger


def _dir():
    return tempfile.mkdtemp()


def test_records_and_settles_then_scores():
    d = _dir()
    ledger.record(d, "fpl", 7, [
        {"id": 1, "name": "A", "ep": 6.0, "role": "xi", "captain": True},
        {"id": 2, "name": "B", "ep": 2.0, "role": "bench"},
    ])
    ledger.settle(d, "fpl", 7, {1: 9, 2: 1})
    s = ledger.score(d, "fpl")
    assert s["n"] == 2
    assert s["bias"] == round(((9 - 6.0) + (1 - 2.0)) / 2, 3)
    assert s["weeks"][0]["xi_projected"] == 6.0
    assert s["weeks"][0]["xi_actual"] == 9


def test_a_resolve_does_not_overwrite_history_but_the_last_word_counts():
    """Re-solving the same gameweek appends; scoring uses the latest.

    Overwriting would let the engine quietly replace a prediction after
    seeing the result, which is how a model marks its own homework.
    """
    d = _dir()
    ledger.record(d, "fpl", 7, [{"id": 1, "ep": 2.0, "role": "xi"}],
                  {"note": "first"})
    ledger.record(d, "fpl", 7, [{"id": 1, "ep": 8.0, "role": "xi"}],
                  {"note": "second"})
    raw = ledger._read(f"{d}/{ledger.PREDICTIONS}")
    assert len(raw) == 2, "both attempts are kept on disk"
    latest = ledger.latest_predictions(d, "fpl")
    assert latest[("fpl", 7, 1)]["ep"] == 8.0


def test_unsettled_gameweeks_are_not_scored():
    d = _dir()
    ledger.record(d, "fpl", 7, [{"id": 1, "ep": 5.0, "role": "xi"}])
    ledger.record(d, "fpl", 8, [{"id": 1, "ep": 5.0, "role": "xi"}])
    ledger.settle(d, "fpl", 7, {1: 5})
    s = ledger.score(d, "fpl")
    assert s["n"] == 1 and s["weeks"][0]["gw"] == 7


def test_empty_ledger_says_so_rather_than_dividing_by_zero():
    assert ledger.score(_dir(), "fpl")["n"] == 0


def test_a_corrupt_line_costs_one_row_not_the_whole_history():
    d = _dir()
    ledger.record(d, "fpl", 7, [{"id": 1, "ep": 5.0, "role": "xi"}])
    with open(f"{d}/{ledger.PREDICTIONS}", "a") as fh:
        fh.write("{not json\n")
    ledger.record(d, "fpl", 7, [{"id": 2, "ep": 3.0, "role": "xi"}])
    assert len(ledger.latest_predictions(d, "fpl")) == 2


def test_games_are_kept_apart():
    d = _dir()
    ledger.record(d, "fpl", 7, [{"id": 1, "ep": 5.0, "role": "xi"}])
    ledger.record(d, "ucl", 7, [{"id": 1, "ep": 9.0, "role": "xi"}])
    ledger.settle(d, "fpl", 7, {1: 5})
    ledger.settle(d, "ucl", 7, {1: 1})
    assert ledger.score(d, "fpl")["bias"] == 0.0
    assert ledger.score(d, "ucl")["bias"] == -8.0
