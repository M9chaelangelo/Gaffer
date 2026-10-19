"""Finishing luck must come back out of the scoring rate before projecting."""
from gaffer import projections


def player(points, goals, assists, xg, xa, **kw):
    e = {"total_points": points, "goals_scored": goals, "assists": assists,
         "expected_goals": xg, "expected_assists": xa}
    e.update(kw)
    return e


def test_an_overperformer_is_projected_below_what_he_has_scored():
    """Kostoulas: 1.95 expected involvements, 7 actual — 40% of his points."""
    e = player(41, 3, 4, 1.83, 0.12)
    realised = projections._scoring_rate(e, "FWD", 482, luck_weight=0.0)
    adjusted = projections._scoring_rate(e, "FWD", 482, luck_weight=0.7)
    assert adjusted < realised
    assert realised == 41 * 90 / 482


def test_an_underperformer_is_projected_above_what_he_has_scored():
    """Gonzalo: 4.19 expected involvements, 1 actual."""
    e = player(18, 1, 0, 3.46, 0.73)
    assert (projections._scoring_rate(e, "FWD", 595, luck_weight=0.7)
            > projections._scoring_rate(e, "FWD", 595, luck_weight=0.0))


def test_a_player_who_scored_exactly_his_chances_is_barely_moved():
    e = player(30, 2, 1, 2.0, 1.0)
    a = projections._scoring_rate(e, "MID", 540, luck_weight=0.0)
    b = projections._scoring_rate(e, "MID", 540, luck_weight=0.9)
    assert abs(a - b) < 1e-9


def test_zero_weight_is_exactly_the_old_behaviour():
    e = player(41, 3, 4, 1.83, 0.12)
    assert (projections._scoring_rate(e, "FWD", 482, luck_weight=0.0)
            == 41 * 90 / 482)


def test_full_weight_replaces_the_attack_outright():
    e = player(41, 3, 4, 1.83, 0.12)
    got = projections._scoring_rate(e, "FWD", 482, luck_weight=1.0)
    per90 = 90 / 482
    want = (41 - (3 * 4 + 4 * 3) + (1.83 * 4 + 0.12 * 3)) * per90
    assert abs(got - want) < 1e-9


def test_the_correction_is_scaled_by_the_position_goal_value():
    """A midfielder's goal is worth 5 and a forward's 4, so the same missed
    expected goal costs them different amounts."""
    e = player(20, 2, 0, 0.5, 0.0)
    mid = projections._scoring_rate(dict(e), "MID", 450, luck_weight=1.0)
    fwd = projections._scoring_rate(dict(e), "FWD", 450, luck_weight=1.0)
    assert mid < fwd          # the midfielder's overperformance was worth more


def test_bonus_is_left_as_scored_so_this_under_corrects():
    """Bonus correlates with goals but is not rebuilt, so the adjusted rate
    stays above a pure expected-points model. Deliberate: under-correcting is
    the safer error."""
    e = player(41, 3, 4, 1.83, 0.12)
    adjusted = projections._scoring_rate(e, "FWD", 482, luck_weight=1.0)
    pure_attack = (1.83 * 4 + 0.12 * 3) * 90 / 482
    assert adjusted > pure_attack


def test_missing_chance_data_leaves_the_rate_alone():
    """Correcting a number we cannot measure would be inventing one."""
    e = {"total_points": 30}
    assert (projections._scoring_rate(e, "MID", 540, luck_weight=0.7)
            == 30 * 90 / 540)


def test_none_values_from_the_feed_are_treated_as_zero():
    e = player(10, None, None, None, None)
    assert projections._scoring_rate(e, "MID", 450, luck_weight=0.7) == 2.0
