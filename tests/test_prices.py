"""Price forecasting: flow measured against the people already holding him."""
from gaffer import prices

TOTAL = 10_000_000


def el(i, pct, tin, tout=0, moved=0):
    return {"id": i, "selected_by_percent": str(pct), "transfers_in_event": tin,
            "transfers_out_event": tout, "cost_change_event": moved}


def boot(elements):
    return {"total_players": TOTAL, "elements": elements}


def test_the_same_flow_means_less_for_a_widely_owned_player():
    """Half a million into a player owned by seven million is a smaller
    event than into one owned by one million, and a single global threshold
    cannot tell them apart. A price site disagreed with the old version on
    exactly this case."""
    small = prices.net_share(el(1, 10.0, 500_000), TOTAL)[0]
    large = prices.net_share(el(2, 70.0, 500_000), TOTAL)[0]
    assert small > large * 5


def test_calibration_is_done_on_the_same_scale_it_forecasts_on():
    b = boot([el(i, 20.0, 300_000, moved=1) for i in range(1, 11)]
             + [el(99, 20.0, 0)])
    up = prices.calibrate(b)
    assert 0 < up < 1, "a share, not a raw transfer count"


def test_a_big_relative_inflow_is_called_a_rise():
    b = boot([el(i, 20.0, 260_000, moved=1) for i in range(1, 11)]
             + [el(50, 5.0, 200_000)])
    assert prices.forecast(b)[50]["direction"] == "rise"


def test_a_big_absolute_inflow_on_a_huge_base_is_not():
    b = boot([el(i, 20.0, 260_000, moved=1) for i in range(1, 11)]
             + [el(51, 70.0, 300_000)])
    assert prices.forecast(b)[51]["direction"] == "hold"


def test_falls_need_more_flow_than_rises():
    b = boot([el(i, 20.0, 260_000, moved=1) for i in range(1, 11)]
             + [el(60, 20.0, 0, tout=270_000)])
    assert prices.forecast(b)[60]["direction"] == "hold", "just short of the fall bar"
    b2 = boot([el(i, 20.0, 260_000, moved=1) for i in range(1, 11)]
              + [el(61, 20.0, 0, tout=600_000)])
    assert prices.forecast(b2)[61]["direction"] == "fall"


def test_a_tiny_ownership_cannot_produce_an_enormous_share():
    """Owned by two hundred people, he hits 100% of owners on a quiet
    afternoon. Without a floor this division blew up: one player in the
    live data read a share of 1339."""
    s, _ = prices.net_share(el(1, 0.0, 1_339), TOTAL)
    assert abs(s) < 0.1
    assert prices.owners(el(1, 0.0, 0), TOTAL) == prices.MIN_OWNERS


def test_a_missing_or_broken_ownership_string_does_not_raise():
    assert prices.owners({"selected_by_percent": None}, TOTAL) == prices.MIN_OWNERS
    assert prices.owners({"selected_by_percent": "n/a"}, TOTAL) == prices.MIN_OWNERS


def test_the_raw_net_is_still_reported_for_reading():
    b = boot([el(1, 10.0, 123_456)])
    assert b and prices.forecast(b)[1]["net"] == 123_456


def test_squad_drift_counts_risers_minus_fallers():
    f = {1: {"direction": "rise"}, 2: {"direction": "rise"}, 3: {"direction": "fall"}}
    assert prices.squad_drift(f, [1, 2, 3]) == 1
