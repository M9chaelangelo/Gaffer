"""UCL Fantasy 2026/27 scoring.

Ported from the manager's own lib/fantasy/rules/ucl.ts. It differs from FPL in
four ways that change strategy, not just arithmetic:

  - Ball recoveries score 1 per 3, with NO threshold. FPL's DefCon is a cliff
    at 10 or 12 actions, so a defensive midfielder there is a lottery ticket;
    here he is a floor.
  - Goalkeepers and defenders get 6 for a goal, not 6/6/5/4 by position, which
    makes a set-piece centre-back worth more than his FPL twin.
  - Player of the Match is 3 points, winner takes all.
  - There is NO vice-captain. Instead the armband may be moved during a
    matchday to a player who has not yet kicked off. That is an option, and
    options have value.
"""
from __future__ import annotations

# skill codes in UEFA's feed
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

GOAL = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
GOAL_OUTSIDE_BOX = 1
ASSIST = 3
CLEAN_SHEET = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}

APPEARANCE = 1
APPEARANCE_60 = 1          # on top of the first point, so 2 for 60+

SAVES_PER_POINT = 3
PENALTY_SAVE = 5
PENALTY_MISS = -2
PENALTY_WON = 2
PENALTY_CONCEDED = -1

CONCEDED_PER_POINT = 2     # -1 per 2 conceded, GK and DEF only
RECOVERIES_PER_POINT = 3   # the big one — no threshold
YELLOW, RED, OWN_GOAL = -1, -3, -2
PLAYER_OF_THE_MATCH = 3

SQUAD_SIZE = 15
SQUAD_COMPOSITION = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
STARTING_SIZE = 11
MIN_STARTING = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
MAX_STARTING = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
BUDGET = 100.0
MAX_PER_CLUB = 3           # league phase; rises in the knockout rounds
CAPTAIN_MULTIPLIER = 2
HAS_VICE_CAPTAIN = False

FREE_TRANSFERS_PER_MATCHDAY = 2
EXTRA_TRANSFER_COST = 4
MAX_ROLLED_TRANSFERS = 1   # league phase only


def score(pos, *, played=False, minutes=0, goals=0, goals_outside_box=0,
          assists=0, clean_sheet=False, conceded=0, saves=0, recoveries=0,
          yellow=0, red=0, own_goals=0, pen_saved=0, pen_missed=0,
          pen_won=0, pen_conceded=0, motm=False):
    """Points for one player in one match. The single source of truth."""
    if not played:
        return 0
    p = APPEARANCE + (APPEARANCE_60 if minutes >= 60 else 0)
    p += goals * GOAL[pos] + goals_outside_box * GOAL_OUTSIDE_BOX
    p += assists * ASSIST
    if clean_sheet and minutes >= 60:
        p += CLEAN_SHEET[pos]
    if pos in ("GK", "DEF"):
        p -= conceded // CONCEDED_PER_POINT
    p += saves // SAVES_PER_POINT
    p += recoveries // RECOVERIES_PER_POINT
    p += yellow * YELLOW + red * RED + own_goals * OWN_GOAL
    p += pen_saved * PENALTY_SAVE + pen_missed * PENALTY_MISS
    p += pen_won * PENALTY_WON + pen_conceded * PENALTY_CONCEDED
    if motm:
        p += PLAYER_OF_THE_MATCH
    return p
