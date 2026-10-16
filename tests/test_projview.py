"""One place that knows how to read a projection row.

Three modules were written against the web-card field names while holding
raw rows from projections.build, and all three failed silently: a column of
0.00, a captains table of None, and an empty leverage list. dict.get on a
missing key returns a default, and a default looks like data.
"""
import inspect

from gaffer import crowd, imminent, projections, projview, template

RAW = {"ep": {7: 5.0, 8: 4.0, 9: 3.0}, "p60": 0.9, "exp_min": 85,
       "pos": "MID", "price": 6.0, "name": "X", "team": "AAA", "status": "a"}
CARD = {"ep_next": 5.0, "ep_horizon": 12.0, "start_prob": 0.9,
        "exp_minutes": 85, "pos": "MID", "price": 6.0, "name": "X"}


def test_reads_a_raw_row():
    assert projview.ep(RAW, 7) == 5.0
    assert projview.horizon(RAW, [7, 8, 9]) == 12.0
    assert projview.start_prob(RAW) == 0.9
    assert projview.exp_minutes(RAW) == 85


def test_reads_a_card_row_too_so_either_may_be_passed():
    assert projview.ep_next(CARD) == 5.0
    assert projview.horizon(CARD, [7, 8, 9]) == 12.0
    assert projview.start_prob(CARD) == 0.9


def test_a_missing_gameweek_falls_back_to_the_first_rather_than_zero():
    """Returning 0.0 for an out-of-range gameweek reads as 'will not score'."""
    assert projview.ep(RAW, 99) == 5.0


def test_an_absent_start_probability_uses_the_caller_s_default():
    """None must not silently become 0 — that filters out every player."""
    assert projview.start_prob({}, 1.0) == 1.0
    assert projview.start_prob({}) is None


def test_the_raw_row_really_has_these_keys():
    """If projections.build renames a field, this fails instead of a report."""
    src = inspect.getsource(projections.build)
    for k in projview.RAW_KEYS:
        assert f'"{k}":' in src, f"projections.build no longer emits {k!r}"


def test_no_module_reads_a_card_field_off_a_raw_row_any_more():
    for mod in (crowd, imminent, template):
        src = inspect.getsource(mod)
        for bad in ('.get("ep_next")', '.get("start_prob")', '.get("ep_horizon")'):
            assert bad not in src, f"{mod.__name__} still reads {bad} directly"


def test_horizon_without_a_gameweek_list_sums_what_the_row_has():
    """A caller that does not know the horizon must not get 0.0.

    Passing a placeholder [None] made `e.get(None)` miss every key and
    return 0.0 — which reads as 'will score nothing', not 'unspecified'.
    """
    assert projview.horizon(RAW) == 12.0
    assert projview.horizon(RAW, [None]) == 12.0
    assert projview.horizon(RAW, [7, 8]) == 9.0
