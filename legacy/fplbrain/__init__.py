"""fplbrain — a decision engine for Fantasy Premier League.

The package is layered so each piece can be used on its own:

    fplbrain.data      fetching and caching raw data (FPL API, odds, xG)
    fplbrain.model     turning raw data into a points distribution per player
    fplbrain.optimize  choosing squads, transfers, lineups and chips
    fplbrain.sim       Monte Carlo simulation of gameweeks and seasons
    fplbrain.report    human-readable output
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
