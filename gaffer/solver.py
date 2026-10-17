"""Which CBC to hand a model to.

PuLP 3.3 deprecated both halves of how this project talked to it, and PuLP 4.0
removes them. One half is a rename: `LpVariable(name, cat=...)` becomes
`prob.add_variable(name, cat=...)`, which 3.3 already understands, so the call
sites just use the new form and this module has nothing to say about it.

The other half is not a rename, and that is why this file exists.
`PULP_CBC_CMD` ran a CBC binary that shipped inside the wheel. `COIN_CMD` runs
whatever `cbc` is on PATH and has nothing to fall back on — under PuLP 4 the
binary is a separate install (`pulp[cbc]`, which is in `requirements.txt` for
exactly this reason). Swapping one name for the other blind would trade a
TypeError for a solver that quietly finds no answer, and an unsolved model
looks precisely like an unsolvable one: the wildcard drafter raises "no legal
squad for 100.0m" either way. A crash is honest; that is not.

So ask which of them can actually reach a binary, newest first.

`LpStatus` went the same way, and it is the same trap in miniature: the
wildcard drafter decides whether it trusts a squad by comparing the status to
the word "Optimal". A status that arrives as `1`, or as an enum, silently
fails that comparison and a perfectly good rebuild is reported as one the
solver could not confirm. So the status becomes a word here, once.
"""
import re
import warnings

import pulp

# Newest first: COIN_CMD is the one that survives PuLP 4, PULP_CBC_CMD is the
# bundled binary that does not.
BACKENDS = ("COIN_CMD", "PULP_CBC_CMD")

# PuLP 3 hands back a bare integer and exposes this mapping as `pulp.LpStatus`;
# 4.0 hands back an enum and drops the mapping. Kept here so a solver that
# gives up says "Infeasible" to the person reading the error, not "-1".
STATUS = {0: "Not Solved", 1: "Optimal", -1: "Infeasible",
          -2: "Unbounded", -3: "Undefined"}

_chosen = None


def _probe(cls):
    """Can this backend find a CBC to run? Construction alone proves nothing."""
    try:
        with warnings.catch_warnings():
            # Constructing PULP_CBC_CMD warns about its own removal. Asking a
            # question is not using it; the answer may well be that we don't.
            warnings.simplefilter("ignore", DeprecationWarning)
            return bool(cls(msg=0).available())
    except Exception:
        # A backend that cannot even be constructed is simply not available.
        return False


def backend():
    """The CBC class to use, decided once."""
    global _chosen
    if _chosen is None:
        for name in BACKENDS:
            cls = getattr(pulp, name, None)
            if cls is not None and _probe(cls):
                _chosen = cls
                break
        else:
            raise RuntimeError(
                "no CBC solver available — PuLP is installed but neither "
                "COIN_CMD nor PULP_CBC_CMD can find a binary. Install one with "
                "`pip install 'pulp[cbc]'`.")
    return _chosen


def status_name(status):
    """The solver's verdict as a word, whether PuLP hands back an int or an
    enum. Both shapes have to produce the same spelling, because callers
    compare it to one — and PuLP 4 spells it `NotSolved` where PuLP 3's table
    said "Not Solved"."""
    name = getattr(status, "name", None)
    if isinstance(name, str):
        spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name.replace("_", " "))
        return spaced.title()               # NotSolved / NOT_SOLVED -> Not Solved
    return STATUS.get(status, str(status))


def solve(model, time_limit, gap=None):
    """Solve `model` quietly under a time limit; return the status name.

    The verdict comes off the return value, not off the model. PuLP 3 returns
    the status integer and also leaves it on `model.status`; PuLP 4 returns a
    stats object and removed the attribute. The return value is the one thing
    both versions give you.

    `gap` is CBC's relative MIP gap. Pass 0 to make it prove optimality.
    Different CBC builds carry different default gaps, so a model that
    returns "Optimal" on one machine returns "Gap Limit" on another with the
    same answer — which is indistinguishable, to a caller comparing the word,
    from a solve that failed. Small models should simply ask for the proof.
    """
    kw = {} if gap is None else {"gapRel": gap}
    result = model.solve(backend()(msg=0, timeLimit=time_limit, **kw))
    status = getattr(result, "status", result)
    return status_name(status)


def proven(status, gap=None):
    """Did the solver finish, or did it merely stop?

    This is the distinction that matters to a caller, and the status word
    alone does not carry it. CBC reports "Gap Limit" when the remaining gap
    meets the tolerance it was given — which, when that tolerance is zero,
    is a proof of optimality and not a shortfall. Different CBC builds reach
    that state on different models, so the same solve came back "Optimal"
    locally and "Gap Limit" on a runner with the same answer.

    "Time Limit" is the one that genuinely means the answer may be poor. It
    cost this project a whole gameweek plan once: a churn penalty pushed
    every week onto the limit, scored 6.8 points below the proven plan, and
    moved the Bench Boost. So the word is kept for the reader and the
    judgement is made here, once.
    """
    if status == "Optimal":
        return True
    return gap == 0 and status == "Gap Limit"
