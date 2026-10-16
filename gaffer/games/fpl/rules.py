"""Fantasy Premier League: scoring and squad rules."""
from __future__ import annotations

from .. import Game

POSITIONS = ("GKP", "DEF", "MID", "FWD")
GOAL = {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}
CLEAN_SHEET = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
ASSIST = 3
APPEARANCE, APPEARANCE_60 = 1, 1
SAVES_PER_POINT = 3
CONCEDED_PER_POINT = 2
PENALTY_SAVE, PENALTY_MISS = 5, -2
YELLOW, RED, OWN_GOAL = -1, -3, -2
# Defensive contribution is a threshold, not a rate: nothing below it pays,
# and the step is worth 2. This is the single largest scoring difference
# from UCL and it inverts how a holding midfielder should be valued.
DEFCON_POINTS = 2
DEFCON_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}


def score(s, pos):
    """Points for one player's match line. `s` is a stats mapping."""
    g = lambda k: int(s.get(k) or 0)
    mins = g("minutes")
    if mins <= 0:
        return 0
    pts = APPEARANCE + (APPEARANCE_60 if mins >= 60 else 0)
    pts += g("goals_scored") * GOAL[pos] + g("assists") * ASSIST
    if mins >= 60:
        pts += g("clean_sheets") * CLEAN_SHEET[pos]
    if pos in ("GKP", "DEF"):
        pts -= g("goals_conceded") // CONCEDED_PER_POINT
    if pos == "GKP":
        pts += g("saves") // SAVES_PER_POINT + g("penalties_saved") * PENALTY_SAVE
    pts += (g("penalties_missed") * PENALTY_MISS + g("yellow_cards") * YELLOW
            + g("red_cards") * RED + g("own_goals") * OWN_GOAL + g("bonus"))
    thr = DEFCON_THRESHOLD.get(pos)
    if thr is not None and g("defensive_contribution") >= thr:
        pts += DEFCON_POINTS
    return pts


GAME = Game(
    key="fpl",
    name="Fantasy Premier League",
    positions=POSITIONS,
    squad={"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3},
    xi_min={"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1},
    xi_max={"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3},
    budget=100.0,
    max_per_club=3,
    captain_multiplier=2,
    has_vice_captain=True,
    armband_movable_midround=False,
    free_transfers_per_round=1,
    extra_transfer_cost=4,
    chips=("wc", "tc", "bb", "fh"),
    score=score,
)
