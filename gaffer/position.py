"""Optimise where you finish, not how many points you score.

Every other objective in this engine maximises expected points. That is the
right target for a manager playing the field, and the wrong one for a
manager who is last in a 42-team league and wants to be first. The two come
apart because **ownership cancels out of expected points and does not
cancel out of variance**:

    owning a player gains  ep x (1 - own)
    missing him costs      ep x own

so for a straight swap, and for captaincy, the edge difference between two
players is exactly their points difference. A multiplier on expected points
that "rewards differentials" is therefore not modelling leverage. It is a
thumb on the scale, and one shipped in this repo for weeks — it eventually
captained a 6.19 over a 9.14 and cost about three points a week.

What differentials actually buy is **variance**, and variance is what turns
a deficit into a win. When the whole league captains Haaland, his hauls move
nobody up the table; they just move everyone's total. Only the players your
rivals do *not* have can change the order.

Pricing that correctly needs the rivals' actual teams, not an ownership
percentage, because finishing first is a question about the maximum of a set
of correlated random variables. So: replay all 41 squads over the same
simulated scorelines, add each manager's current total, and count how often
you come out on top. That number is the objective; expected points is a
diagnostic.

Two honest limits, stated here because they bound every figure this module
produces:

  - **Rivals are frozen.** Their squads are whatever they last submitted.
    They will transfer and change captains, and we cannot see it. The
    estimate is therefore of "if they stand still", which is optimistic
    about them and pessimistic about nobody.
  - **One gameweek is simulated properly; the rest of the season is
    extrapolated** from each manager's own scoring record. A 31-gameweek
    extrapolation is a weak instrument, and `p_first` says which horizon it
    was asked about.
"""
from __future__ import annotations

import numpy as np

# A manager's gameweek-to-gameweek spread, when their own record is too
# short to estimate one. Roughly what an FPL side's scores actually vary by.
DEFAULT_SD = 18.0

# How much of a gameweek's variation is the league breathing together —
# everyone owns Haaland, so everyone's good week correlates. Only the rest
# of it can change the order, which is precisely why this is not 0.
SHARED_FRACTION = 0.55


def squad_scores(squads, samples, columns):
    """(draws, managers) points, replaying each captured squad.

    `multiplier` carries their own lineup and armband — 0 bench, 1 starter,
    2 captain, 3 under a Triple Captain — so no assumption is needed about
    either. A player we did not simulate contributes nothing rather than
    silently scoring zero for everyone equally: he is simply absent from
    the sum, which is the same thing, and the count of missing picks is
    returned so it can be reported instead of hidden.
    """
    n = samples.shape[0]
    out = np.zeros((n, len(squads)))
    missing = 0
    for j, sq in enumerate(squads):
        total = np.zeros(n)
        for pick in sq.get("picks") or []:
            m = pick.get("multiplier") or 0
            if not m:
                continue
            col = columns.get(str(pick.get("element")))
            if col is None:
                missing += 1
                continue
            total += m * samples[:, col]
        out[:, j] = total
    return out, missing


def finishing_order(my_gw, rival_gw, my_total, rival_totals):
    """P(first), expected rank, and the chance of beating each rival.

    Ranks are over this gameweek's totals added to what everyone already
    has. A tie counts as *not* first, which is the conservative reading and
    matches how a league settles a dead heat on its tiebreak rules rather
    than in your favour.
    """
    mine = np.asarray(my_gw, dtype=float) + float(my_total)
    theirs = np.asarray(rival_gw, dtype=float) + np.asarray(
        rival_totals, dtype=float)[None, :]
    beaten = mine[:, None] > theirs
    rank = 1 + (~beaten).sum(axis=1)
    return {
        "p_first": float((rank == 1).mean()),
        "mean_rank": float(rank.mean()),
        "p_top3": float((rank <= 3).mean()),
        "p_beat": beaten.mean(axis=0).tolist(),
        "mean_points": float(np.asarray(my_gw, dtype=float).mean()),
    }


def project_season(my_gw, rival_gw, my_total, rival_totals, remaining,
                   my_sd=None, rival_sds=None, rng=None,
                   shared=SHARED_FRACTION):
    """The same question, carried to the end of the season.

    The simulated gameweek is played out properly; each remaining one is
    drawn from the manager's own mean and spread, with `shared` of the
    variance common to everyone — the part where the whole league owns the
    same captain and rises together. Only the idiosyncratic remainder can
    reorder the table, so treating gameweeks as independent across managers
    would wildly overstate how much a deficit can be clawed back.
    """
    rng = rng or np.random.default_rng(7)
    my_gw = np.asarray(my_gw, dtype=float)
    rival_gw = np.asarray(rival_gw, dtype=float)
    n, k = rival_gw.shape

    mine = my_gw + float(my_total)
    theirs = rival_gw + np.asarray(rival_totals, dtype=float)[None, :]

    if remaining > 0:
        my_mu = float(my_gw.mean())
        rival_mu = rival_gw.mean(axis=0)
        s_me = float(my_sd if my_sd is not None else max(my_gw.std(), DEFAULT_SD))
        s_them = (np.asarray(rival_sds, dtype=float) if rival_sds is not None
                  else np.maximum(rival_gw.std(axis=0), DEFAULT_SD))
        c = np.sqrt(shared)
        i = np.sqrt(max(0.0, 1.0 - shared))
        for _ in range(int(remaining)):
            common = rng.standard_normal(n)
            mine = mine + my_mu + s_me * (c * common + i * rng.standard_normal(n))
            theirs = theirs + rival_mu + s_them * (
                c * common[:, None] + i * rng.standard_normal((n, k)))

    beaten = mine[:, None] > theirs
    rank = 1 + (~beaten).sum(axis=1)
    return {
        "remaining": int(remaining),
        "p_first": float((rank == 1).mean()),
        "p_top3": float((rank <= 3).mean()),
        "mean_rank": float(rank.mean()),
        "mean_total": float(mine.mean()),
        "leader_mean_total": float(theirs.mean(axis=0).max()),
    }


def compare(options, rival_gw, my_total, rival_totals, remaining=0, **kw):
    """Rank candidate decisions by P(first), not by expected points.

    `options` maps a label to that option's simulated gameweek scores for
    you. Every option is scored against the *same* rival draws, so the
    comparison isolates the decision rather than the weather.

    The two columns are reported side by side on purpose: when they
    disagree, that disagreement is the whole reason this module exists, and
    seeing it is more useful than being handed a winner.
    """
    out = []
    for label, my_gw in options.items():
        now = finishing_order(my_gw, rival_gw, my_total, rival_totals)
        row = {"option": label,
               "mean_points": now["mean_points"],
               "p_first_now": now["p_first"],
               "mean_rank_now": now["mean_rank"]}
        if remaining:
            season = project_season(my_gw, rival_gw, my_total, rival_totals,
                                    remaining, **kw)
            row["p_first_season"] = season["p_first"]
            row["p_top3_season"] = season["p_top3"]
        out.append(row)
    key = "p_first_season" if remaining else "p_first_now"
    return sorted(out, key=lambda r: -r[key])


def disagreement(rows):
    """Does the points-maximising option also maximise P(first)?

    When it does not, say so loudly: that is the case the rest of the
    engine cannot see, and the one where following expected points costs
    you the league.
    """
    if len(rows) < 2:
        return None
    key = "p_first_season" if "p_first_season" in rows[0] else "p_first_now"
    best_points = max(rows, key=lambda r: r["mean_points"])
    best_place = max(rows, key=lambda r: r[key])
    if best_points["option"] == best_place["option"]:
        return None
    return {
        "points_pick": best_points["option"],
        "place_pick": best_place["option"],
        "points_cost": round(best_points["mean_points"]
                             - best_place["mean_points"], 2),
        "place_gain": round(best_place[key] - best_points[key], 4),
        "note": (f"{best_place['option']} scores "
                 f"{best_points['mean_points'] - best_place['mean_points']:.2f} "
                 f"fewer points than {best_points['option']} but finishes "
                 f"first {best_place[key] - best_points[key]:+.1%} more often"),
    }


def edge_variance_table(my_total, rival_totals, remaining,
                        edges=(0, 2, 4, 6, 8, 10), sds=(18, 25, 32, 40),
                        league_mean=55.0, league_sd=18.0, draws=30000,
                        rng=None):
    """P(first) across weekly edge and weekly spread, for this table.

    The one question worth asking from 40th of 42: not "what is my chance"
    — it is small — but "what would have to change". It separates the two
    things a manager can actually move, how much better than average they
    score and how much they swing, and shows that they trade off.

    The result is the reason this module exists. Measured on league 88825
    on 17 Oct 2026, 132 points behind with 31 gameweeks left:

        edge/GW      sd 18    sd 25    sd 32    sd 40
              0      0.09%    1.17%    4.47%    9.93%
              4      8.06%   15.86%   21.40%   25.05%
              8     58.97%   55.65%   53.30%   53.57%

    Read the top row first: with no skill edge at all, swinging harder
    takes a hopeless position to a one-in-ten shot. Then read the bottom
    row, where more variance starts to *cost* — once you are good enough
    to win on merit, noise is the enemy. Behind, buy variance; ahead, sell
    it. The crossover here is around +8 a week.
    """
    rng = rng or np.random.default_rng(1)
    theirs = rng.normal(league_mean, league_sd,
                        (draws, len(rival_totals)))
    out = []
    for edge in edges:
        row = {"edge": edge}
        for sd in sds:
            mine = rng.normal(league_mean + edge, sd, draws)
            got = project_season(mine, theirs, my_total, rival_totals,
                                 remaining, my_sd=sd, rng=rng)
            row[f"sd{sd}"] = got["p_first"]
        out.append(row)
    return out
