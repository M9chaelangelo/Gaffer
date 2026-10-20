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
BUDGET = 100.0             # rises to 105.0 once the league phase is over
BUDGET_KNOCKOUT = 105.0
MAX_PER_CLUB = 3           # league phase only — see MAX_PER_CLUB_BY_MATCHDAY

# The cap is not a constant. It doubles and more as the field narrows, which
# is what makes a chip held back for the knockouts a different instrument from
# one played now: in the final you may own eight players from one club.
MAX_PER_CLUB_BY_MATCHDAY = {
    **{md: 3 for md in range(1, 9)},     # league phase
    9: 4, 10: 4,                         # knockout phase play-offs
    11: 4, 12: 4,                        # round of 16
    13: 5, 14: 5,                        # quarter-finals
    15: 6, 16: 6,                        # semi-finals
    17: 8,                               # final
}

CAPTAIN_MULTIPLIER = 2
HAS_VICE_CAPTAIN = False

FREE_TRANSFERS_PER_MATCHDAY = 2
EXTRA_TRANSFER_COST = 4
MAX_ROLLED_TRANSFERS = 1   # league phase only; nothing carries in the knockouts

# Free transfers by matchday. The three `None` entries are unlimited, and they
# are why the chips are blocked on exactly those matchdays: everybody already
# has the transfers, so a Wildcard there would buy nothing.
FREE_TRANSFERS_BY_MATCHDAY = {
    1: None, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2,
    9: None, 10: 2,
    11: None, 12: 3,
    13: 5, 14: 3,
    15: 5, 16: 3,
    17: 5,
}

# Chips cannot be played on these, for the reason above.
CHIP_BLOCKED_MATCHDAYS = (1, 9, 11)

# Every matchday is spread over two or more days, so the armband can be moved
# and up to four players substituted — EXCEPT these two, where every match
# kicks off at once. A chip played here loses the whole in-play option, which
# is a large part of what the bench is worth.
SIMULTANEOUS_MATCHDAYS = (8, 17)

MAX_SUBS_PER_DAY = 4

# Traps in the live rules that cost points rather than breaking a constraint:
#
#  - Any manual change at all — a substitution, a captain change, even
#    reordering the bench — disables automatic substitutions for the REST of
#    that matchday, and it cannot be undone by switching back.
#  - If your captain then fails to play, he is NOT subbed out and NOBODY
#    scores double. The armband is simply lost for the matchday.
#
# So the in-play flexibility this engine prices is not free: taking it forfeits
# the safety net. Captain a nailed starter on the first day, or do nothing.
MANUAL_CHANGE_DISABLES_AUTOSUBS = True


# --- plugged into the shared engine ------------------------------------------
# Everything above is the scoring constants. Below is what the rest of the
# engine needs to know about the game, and it is the only place UCL differs
# from FPL as far as the solver, the sweep and the simulator are concerned.
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
    chips=("wc", "lim"),
    score=score,
)
