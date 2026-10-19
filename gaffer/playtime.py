"""Expected minutes from a player's recent match log, not his season average.

`minutes / rounds_played` charges a player for gameweeks he was never in the
side for, and credits him for weeks he no longer plays. Both errors are large
and both land where it hurts, because minutes scale almost the whole
projection:

  - Barcola won his place in GW4 and has since played 71, 87 and 90 minutes.
    Divided by seven rounds he reads as a 49-minute player, and the engine
    ranked him 40th of 119 midfielders.
  - Barkley lost his place in GW5 and has since played 6, 4 and 18. Divided by
    seven rounds he reads as a 47-minute player — which, blended with a start
    probability that has no recency in it either, came out at 57 expected
    minutes. He is a substitute.

So the estimate is built from the actual per-gameweek log, time-decayed, and
split into the three things that decide a man's minutes. Keeping them apart
matters: "starts less often" and "is subbed earlier" are different facts about
a footballer and they do not have the same consequences.

    p_start        how often he has started lately
    mins_if_start  how long he lasts when he does start
    mins_if_sub    what his cameo is worth, when he gets one

    exp_min = p_start * mins_if_start
            + (1 - p_start) * p_sub * mins_if_sub

Availability is NOT applied here. A knock is news about one gameweek; this
module is about a player's role, and the caller multiplies the two.
"""
from __future__ import annotations

# In gameweeks. Fitted, not guessed: predicting each of gameweeks 4-7 from the
# weeks before it, mean absolute error on expected minutes went
#
#     half_life   1 week ahead   2 ahead   3 ahead
#          0.4          18.82      21.76     24.31
#          0.8          19.68      22.55     24.91
#          1.5          20.93      23.46     25.41
#          3.5          22.31      24.30     25.86
#
# so shorter is better at every horizon — minutes are strongly autocorrelated
# week to week and a season average is mostly stale information. The trend runs
# to the edge of the grid, which would make "copy last week" the winner, and
# 0.8 is deliberately a little longer than that: it keeps about three quarters
# of the available gain while averaging over more than a single match, so one
# rotation or one early substitution does not rewrite a player's role and set
# the solver churning. Tune it on more than four gameweeks of evidence before
# moving it.
HALF_LIFE = 0.8

# Pseudo-gameweeks of prior. The prior is the player's OWN season start rate,
# not a flat one-in-two: a man who has started everything and a man who has
# started nothing should not be dragged toward the same number. Using the
# season rate instead of 0.5 was worth about 1.2 minutes of MAE on its own,
# and 0.25 beat both 0.5 and 1.0.
PRIOR_WEIGHT = 0.25

# Used only when there is no log at all to take a start rate from.
PRIOR_START_FALLBACK = 0.5

# What we assume when there is nothing to measure.
FULL_MINUTES = 82.0
CAMEO_MINUTES = 16.0

# A start is worth at least this much, whatever a freak early substitution
# says. Without it one injury inside ten minutes makes a man a cameo risk for
# a month.
MIN_START_MINUTES = 45.0


def estimate(rounds, now, half_life=HALF_LIFE, prior_weight=PRIOR_WEIGHT,
             prior_start=None):
    """Role estimate from a decayed match log.

    `rounds` is [(gw, minutes, starts, slots)] covering only the gameweeks the
    player's CLUB actually played — a blank is not evidence that he was
    dropped. `slots` is how many fixtures the club had that gameweek, so a
    double counts twice and its minutes are read per fixture.
    """
    total_starts = sum(r[2] for r in rounds)
    total_slots = sum(max(1, r[3]) for r in rounds)
    if prior_start is None:
        prior_start = (total_starts / total_slots) if total_slots else PRIOR_START_FALLBACK

    sw = sm = 0.0          # start weight, started weight
    fw = fm = 0.0          # weight and minutes over the weeks he started
    bw = bm = 0.0          # ... and over the weeks he came off the bench
    nw = na = 0.0          # non-start weeks, and those he appeared in at all
    for gw, mins, starts, slots in rounds:
        slots = max(1, slots)
        w = 0.5 ** ((now - gw) / half_life) * slots
        per = mins / slots
        started = starts / slots
        sw += w
        sm += w * started
        if started >= 0.5:
            fw += w
            fm += w * max(per, MIN_START_MINUTES)
        else:
            nw += w
            if per > 0:
                na += w
                bw += w
                bm += w * per
    p_start = ((sm + prior_weight * prior_start)
               / (sw + prior_weight)) if (sw + prior_weight) else prior_start
    mins_if_start = fm / fw if fw else FULL_MINUTES
    mins_if_sub = bm / bw if bw else CAMEO_MINUTES
    # A man who is dropped is usually still on the bench, so default to being
    # used rather than to being ignored — but measure it where we can.
    p_sub = (na / nw) if nw else 0.6
    exp_min = p_start * mins_if_start + (1 - p_start) * p_sub * mins_if_sub
    return {
        "p_start": p_start,
        "mins_if_start": mins_if_start,
        "mins_if_sub": mins_if_sub,
        "p_sub": p_sub,
        "exp_min": min(90.0, exp_min),
        "observed": sw,
    }


def club_slots(fixtures, gws):
    """{team_id: {gameweek: fixtures played}} — a blank is zero, a double two."""
    out = {}
    for x in fixtures:
        g = x.get("event")
        if g is None or g not in gws:
            continue
        for side in ("team_h", "team_a"):
            out.setdefault(x[side], {}).setdefault(g, 0)
            out[x[side]][g] += 1
    return out


def table(boot, fixtures, live_by_gw, now, half_life=HALF_LIFE,
          prior_weight=PRIOR_WEIGHT):
    """Role estimates for every player, from the per-gameweek live payloads.

    `live_by_gw` is {gameweek: live payload}; one request per gameweek rather
    than one per player, which is the only reason this table is affordable.
    """
    gws = sorted(live_by_gw)
    slots = club_slots(fixtures, set(gws))
    logs = {e["id"]: [] for e in boot["elements"]}
    team_of = {e["id"]: e["team"] for e in boot["elements"]}
    for g in gws:
        for row in (live_by_gw[g] or {}).get("elements", []):
            pid = row["id"]
            if pid not in logs:
                continue
            n = slots.get(team_of[pid], {}).get(g, 0)
            if not n:
                continue                      # his club did not play: no evidence
            st = row.get("stats") or {}
            logs[pid].append((g, float(st.get("minutes") or 0),
                              float(st.get("starts") or 0), n))
    return {pid: estimate(rows, now, half_life, prior_weight)
            for pid, rows in logs.items()}
