"""UEFA Champions League Fantasy — re-exported from the original module.

The scoring lives in `gaffer.ucl.rules`, which was written against UEFA's
published rules and hand-verified. This is the adapter seam, so callers can
say `games.get("ucl")` without knowing where the constants live.
"""
from ...ucl.rules import GAME, score    # noqa: F401
