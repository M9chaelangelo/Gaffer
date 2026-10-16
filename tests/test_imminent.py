"""The overdue tables, built from FPL's own Opta expected goals and assists."""
from gaffer import imminent


def _boot():
    return {
        "teams": [{"id": 1, "short_name": "AAA"}, {"id": 2, "short_name": "BBB"}],
        "elements": [
            # creating now, and well behind his season xG -> the real candidate
            dict(id=1, web_name="Due", team=1, element_type=4, now_cost=75,
                 minutes=600, expected_goals="5.0", goals_scored=2,
                 expected_assists="1.0", assists=1, status="a", news="",
                 selected_by_percent="12.3"),
            # scoring freely, no debt -> should rank below on the tie-break
            dict(id=2, web_name="Hot", team=2, element_type=4, now_cost=80,
                 minutes=600, expected_goals="5.0", goals_scored=8,
                 expected_assists="1.0", assists=1, status="a", news="",
                 selected_by_percent="30.0"),
            # behind on the season but not playing well now -> not "imminent"
            dict(id=3, web_name="Cold", team=1, element_type=4, now_cost=60,
                 minutes=600, expected_goals="6.0", goals_scored=1,
                 expected_assists="0.5", assists=0, status="a", news="",
                 selected_by_percent="4.0"),
            # too few minutes for a per-90 to mean anything
            dict(id=4, web_name="Cameo", team=2, element_type=4, now_cost=45,
                 minutes=100, expected_goals="2.0", goals_scored=0,
                 expected_assists="0.0", assists=0, status="a", news="",
                 selected_by_percent="1.0"),
            # injured: never recommend him, however good the numbers
            dict(id=5, web_name="Crocked", team=1, element_type=4, now_cost=90,
                 minutes=600, expected_goals="9.0", goals_scored=1,
                 expected_assists="2.0", assists=0, status="i", news="Knee",
                 selected_by_percent="9.0"),
        ],
    }


def _live(xg_by_id, mins=360):
    """One payload standing in for every gameweek in the window."""
    return {"elements": [{"id": i, "stats": {"expected_goals": str(v),
                                             "expected_assists": "0.1",
                                             "minutes": mins, "goals_scored": 0,
                                             "assists": 0}}
                         for i, v in xg_by_id.items()]}


def test_ranks_on_recent_form_and_flags_the_ones_also_in_debt():
    live = {4: _live({1: 1.2, 2: 1.5, 3: 0.05, 4: 1.0, 5: 2.0})}
    out = imminent.build(_boot(), live)
    names = [r["name"] for r in out["goals_imminent"]]

    assert names[:2] == ["Hot", "Due"], "ranking is on recent form first"
    due = next(r for r in out["goals_imminent"] if r["name"] == "Due")
    hot = next(r for r in out["goals_imminent"] if r["name"] == "Hot")
    assert due["both"] is True, "creating now and 3.0 behind his xG"
    assert hot["both"] is False, "ahead of his xG, so nothing is owed to him"
    assert due["goal_debt"] == 3.0


def test_leaves_out_the_injured_and_the_barely_played():
    live = {4: _live({1: 1.2, 2: 1.5, 3: 0.05, 4: 1.0, 5: 2.0})}
    out = imminent.build(_boot(), live)
    names = [r["name"] for r in out["goals_imminent"]]
    assert "Crocked" not in names, "an injured player is never imminent"
    assert "Cameo" not in names, "100 minutes is not a rate"


def test_cold_form_is_not_imminent_however_large_the_debt():
    """The half of the idea that gets people into trouble: 'he's due'."""
    live = {4: _live({1: 1.2, 2: 1.5, 3: 0.05, 4: 1.0, 5: 2.0})}
    out = imminent.build(_boot(), live)
    rank = [r["name"] for r in out["goals_imminent"]]
    assert rank.index("Cold") > rank.index("Due"), \
        "5.0 of debt must not outrank a player actually creating chances"


def test_window_takes_only_the_most_recent_gameweeks():
    live = {g: _live({1: 0.1}) for g in range(1, 10)}
    live[9] = _live({1: 2.0})
    out = imminent.build(_boot(), live, window=2)
    assert out["window"] == [8, 9]
