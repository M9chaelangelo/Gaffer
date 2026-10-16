"""Learning: the models that are trained, and the ledger that scores them.

`train` fits the ten models from historical player-gameweeks. `ledger`
records what the engine predicted and joins it to what happened, which is
the half that was missing — a model nobody scores cannot be improved, and
until now nothing recorded that a GW6 eleven projected at 68.8 returned 30.

The names `train` exported are re-exported here so that `from . import
learn; learn.load("minutes")` keeps working from the modules that have
always called it that way.
"""
from . import ledger, train           # noqa: F401
from .train import expected_bonus, load, MODELS   # noqa: F401


def __getattr__(name):
    # Anything else that used to live on the flat module still resolves.
    return getattr(train, name)
