"""One engine, two games.

FPL and UEFA Fantasy ask the same question — pick fifteen, start eleven,
captain one, spend chips well — over different scoring and different squad
rules. Until now that was two codebases: the FPL one ran weekly on live
data, the UCL one was a half-wired silo that had never produced a plan, and
the UCL advice actually given came from a standalone script.

A `Game` is the whole of the difference. Everything that is not on it —
Dixon-Coles scorelines, devigging, correlated simulation, the MILP, the
chip sweep, shrinkage, the ledger — is shared, because none of it cares
which competition it is scoring.

The differences that genuinely change strategy, and so must live here:

  - UCL pays ball recoveries 1 per 3 with NO threshold. FPL's defensive
    contribution is a cliff at 10 (12 for midfielders and forwards), so in
    FPL a holding midfielder is a lottery ticket and in UCL he is a floor.
  - UCL pays a goalkeeper or defender 6 for a goal, against FPL's 6 and 6 —
    but a midfielder 5 and a forward 4, so the positional spread is wider.
  - UCL has a Player of the Match award worth 3. FPL has bonus points.
  - **UCL has no vice-captain.** Instead the armband may be moved mid-
    matchday to a player who has not kicked off, which makes captaining
    someone who plays on day one strictly better than captaining an
    equivalent player on day two — a free second swing that no FPL
    captaincy model has any reason to represent.
  - UCL allows manual substitutions between matchdays, forfeiting the
    removed player's points and disabling auto-subs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class Game:
    key: str
    name: str
    positions: Sequence[str]
    squad: Mapping[str, int]
    xi_min: Mapping[str, int]
    xi_max: Mapping[str, int]
    budget: float
    max_per_club: int
    captain_multiplier: int
    has_vice_captain: bool
    # The armband may move once play has started, to a player yet to kick off.
    armband_movable_midround: bool
    free_transfers_per_round: int
    extra_transfer_cost: int
    chips: Sequence[str]
    score: Callable | None = field(default=None, compare=False)

    @property
    def squad_size(self):
        return sum(self.squad.values())


def fpl():
    from .fpl import rules
    return rules.GAME


def ucl():
    from .ucl import rules
    return rules.GAME


REGISTRY = {"fpl": fpl, "ucl": ucl}


def get(key):
    """Resolve a game by key. Unknown keys name the ones that exist."""
    k = str(key or "fpl").strip().lower()
    if k not in REGISTRY:
        raise KeyError(f"unknown game {key!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[k]()
