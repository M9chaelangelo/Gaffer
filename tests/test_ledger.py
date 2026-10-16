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


# --- price forecasts --------------------------------------------------------

def _snap(d, rows):
    ledger.record_prices(d, rows)


def test_a_price_forecast_is_scored_against_the_next_snapshot():
    """The file settles itself: tonight's price answers last night's question."""
    d = _dir()
    _snap(d, [{"id": 1, "name": "Riser", "price": 6.0, "price_direction": "rise",
               "net_transfers": 400_000},
              {"id": 2, "name": "Faller", "price": 8.0, "price_direction": "fall",
               "net_transfers": -300_000}])
    _snap(d, [{"id": 1, "name": "Riser", "price": 6.1, "price_direction": "hold",
               "net_transfers": 10},
              {"id": 2, "name": "Faller", "price": 8.0, "price_direction": "hold",
               "net_transfers": 10}])
    s = ledger.score_prices(d)
    assert s["n"] == 2
    assert s["matrix"]["rise"]["correct"] == 1          # called it, it rose
    assert s["matrix"]["fall"]["correct"] == 0          # called it, it held
    assert s["matrix"]["fall"]["breakdown"]["hold"] == 1
    assert s["accuracy"] == 0.5


def test_one_snapshot_cannot_be_scored_and_says_so():
    d = _dir()
    _snap(d, [{"id": 1, "name": "A", "price": 6.0, "price_direction": "rise"}])
    assert ledger.score_prices(d)["n"] == 0


def test_the_latest_snapshot_is_never_marked_against_itself():
    """Three snapshots give two scored transitions, not three."""
    d = _dir()
    for pr in (6.0, 6.1, 6.2):
        _snap(d, [{"id": 1, "name": "A", "price": pr, "price_direction": "rise"}])
    assert ledger.score_prices(d)["n"] == 2


def test_the_actual_move_is_recorded_with_its_size():
    d = _dir()
    _snap(d, [{"id": 1, "name": "A", "price": 6.0, "price_direction": "rise"}])
    _snap(d, [{"id": 1, "name": "A", "price": 5.9, "price_direction": "hold"}])
    m = ledger.score_prices(d)["moves"]
    assert m and m[0]["actual"] == "fall" and m[0]["delta"] == -0.1
    assert m[0]["said"] == "rise", "a wrong call is kept, not quietly dropped"


def test_a_player_without_a_price_is_skipped_not_recorded_as_zero():
    d = _dir()
    n = ledger.record_prices(d, [{"id": 1, "name": "A", "price": None}])
    assert n == 0
