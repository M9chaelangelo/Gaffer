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


# --- plugged into the shared engine ------------------------------------------
# Everything above is the scoring. This is what the rest of the engine needs
# to know about the game, and it is the only place UCL differs from FPL as
# far as the solver, the sweep and the simulator are concerned.
from ..games import Game    # noqa: E402


def score(s, pos):
    """Points for one player's match line. `s` is a stats mapping."""
    g = lambda k: int(s.get(k) or 0)
    mins = g("minutes")
    if mins <= 0:
        return 0
    pts = APPEARANCE + (APPEARANCE_60 if mins >= 60 else 0)
    pts += g("goals") * GOAL[pos] + g("assists") * ASSIST
    pts += g("goals_outside_box") * GOAL_OUTSIDE_BOX
    if mins >= 60:
        # Absent data is not a clean sheet. `g()` returns 0 for a missing
        # key, so deriving the clean sheet from `goals_conceded == 0` alone
        # awarded one to every line that simply did not carry the field.
        if "clean_sheet" in s:
            clean = bool(s["clean_sheet"])
        elif "goals_conceded" in s:
            clean = int(s["goals_conceded"] or 0) == 0
        else:
            clean = False
        pts += CLEAN_SHEET[pos] if clean else 0
    if pos in ("GK", "DEF"):
        pts -= g("goals_conceded") // CONCEDED_PER_POINT
    if pos == "GK":
        pts += g("saves") // SAVES_PER_POINT + g("penalties_saved") * PENALTY_SAVE
    # No threshold here, unlike FPL: every third recovery pays, from the first.
    pts += g("recoveries") // RECOVERIES_PER_POINT
    pts += (g("penalties_won") * PENALTY_WON
            + g("penalties_conceded") * PENALTY_CONCEDED
            + g("penalties_missed") * PENALTY_MISS
            + g("yellow_cards") * YELLOW + g("red_cards") * RED
            + g("own_goals") * OWN_GOAL
            + g("motm") * PLAYER_OF_THE_MATCH)
    return pts


GAME = Game(
    key="ucl",
    name="UEFA Champions League Fantasy",
    positions=tuple(SQUAD_COMPOSITION),
    squad=dict(SQUAD_COMPOSITION),
    xi_min=dict(MIN_STARTING),
    xi_max={"GK": 1, "DEF": 5, "MID": 5, "FWD": 3},
    budget=BUDGET,
    max_per_club=MAX_PER_CLUB,
    captain_multiplier=CAPTAIN_MULTIPLIER,
    has_vice_captain=HAS_VICE_CAPTAIN,
    # The armband moves mid-matchday to anyone yet to kick off. This is the
    # single most valuable thing about UCL captaincy and has no FPL analogue.
    armband_movable_midround=True,
    free_transfers_per_round=FREE_TRANSFERS_PER_MATCHDAY,
    extra_transfer_cost=EXTRA_TRANSFER_COST,
    chips=("wc", "lim", "allin"),
    score=score,
)
