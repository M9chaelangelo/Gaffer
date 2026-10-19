"""A swap is measured by when its gain arrives, not only by the endpoint."""
from gaffer import horizon


def prof(out_vals, in_vals, first=8):
    gws = list(range(first, first + len(out_vals)))
    out_ep = dict(zip(gws, out_vals))
    in_ep = dict(zip(gws, in_vals))
    return horizon.profile(out_ep, in_ep, gws)


def test_cumulative_is_the_running_sum_of_the_weekly_difference():
    rows = prof([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
    assert [r["week"] for r in rows] == [1.0, 1.0, 1.0]
    assert [r["cumulative"] for r in rows] == [1.0, 2.0, 3.0]


def test_a_swap_that_decays_is_flagged_and_its_peak_named():
    """Saka -> Palmer: ahead by 4.8 at GW13, only 1.8 by GW15."""
    rows = prof([5.0] * 8, [5.1, 6.0, 5.1, 5.0, 5.3, 8.4, 3.9, 3.1])
    s = horizon.summarise(rows)
    assert s["peak_gw"] == 13
    assert s["decays"]
    assert s["peak"] > s["end"]
    assert "move on" in horizon.verdict(rows)


def test_a_swap_whose_loss_is_all_at_the_end_is_not_reported_as_uniformly_bad():
    """Bruno -> Palmer is level through GW13 and loses it in the last two."""
    rows = prof([5.0] * 8, [4.7, 5.1, 4.3, 6.9, 2.9, 7.3, 1.3, 1.0])
    s = horizon.summarise(rows)
    assert s["end"] < 0
    assert s["peak"] >= 0
    assert s["front_loaded"]
    assert "then" in horizon.verdict(rows)


def test_a_swap_that_keeps_gaining_says_hold_it():
    rows = prof([2.0] * 8, [4.0] * 8)
    s = horizon.summarise(rows)
    assert not s["decays"]
    assert s["peak_gw"] == s["end_gw"]
    assert "hold it" in horizon.verdict(rows)


def test_a_swap_that_never_leads_is_a_plain_no():
    rows = prof([5.0] * 6, [4.0] * 6)
    assert "no" in horizon.verdict(rows)
    assert horizon.summarise(rows)["peak"] < 0


def test_hold_weeks_counts_gameweeks_not_index():
    rows = prof([1.0] * 5, [2.0, 2.0, 0.0, 0.0, 0.0])
    s = horizon.summarise(rows)
    assert s["hold_weeks"] == 2
    assert s["peak_gw"] == 9


def test_decay_is_off_by_default_so_the_profile_is_points_as_scored():
    plain = prof([0.0] * 3, [1.0] * 3)
    assert plain[-1]["cumulative"] == 3.0
    gws = [8, 9, 10]
    discounted = horizon.profile({g: 0.0 for g in gws}, {g: 1.0 for g in gws},
                                 gws, decay=0.5)
    assert discounted[-1]["cumulative"] == 1.0 + 0.5 + 0.25


def test_an_empty_horizon_does_not_raise():
    assert horizon.summarise([]) == {}
    assert "no projection" in horizon.verdict([])


def test_the_table_lines_up_with_the_rows():
    rows = prof([1.0, 1.0], [2.0, 3.0])
    t = horizon.table(rows).splitlines()
    assert t[0].split()[1:] == ["8", "9"]
    assert t[2].startswith("  cum")
