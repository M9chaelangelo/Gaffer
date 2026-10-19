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

# A week with no minutes at all is weaker evidence than a cameo, and the first
# version of this treated them alike. A cameo says plainly that he was
# available and not picked to start — Barkley's 6, 4 and 18 are a demotion and
# nothing else. A nought says only that he did not play, which covers being
# dropped, being injured, being suspended and being an unused substitute, and
# the feed does not say which. So Thomas, who started six in a row at ninety
# minutes and then missed one game, read as 47% likely to start while carrying
# no injury flag at all.
#
# Discounting the ambiguous evidence leaves Barkley where he was, because none
# of his evidence is ambiguous, and lifts Thomas to a defensible 0.6.
#
# The backtest cannot settle the value, and said so twice. Measured over the
# 272 cases with 2%+ ownership — the players decisions are actually made about,
# and ownership is known before the gameweek, so selecting on it is not
# selecting on the outcome — every setting from 1.0 to 0.3 scored between
# 12.99 and 13.13 minutes of error. That is a one per cent spread on 272
# samples: noise. Over all 1141 cases, including the many fringe players, not
# discounting at all was 0.8 minutes better.
#
# So it is set in the middle, on which error costs more rather than which is
# larger. A six-game starter who missed one week reading 43 expected minutes
# is a wrong transfer. The error the discount adds lands on players who are not
# in contention for a squad place, where twenty minutes against five changes
# nothing. Revisit with a season of evidence instead of four gameweeks.
ZERO_WEIGHT = 0.45

# Whether a man is picked changes faster than how long he lasts once he is on
# the pitch, so the two are measured over different windows. Groß has played
# 90, 90, 90, 90, 90, 90, 70: at the selection half-life his last match alone
# is more than half the evidence and he read as a 78-minute starter, against 84
# over this one. The backtest is indifferent — 19.48 against 19.68 minutes of
# MAE, and the longer window is the better of the two on the 90th percentile —
# so this is chosen on the pattern it describes, not on a score it wins.
DURATION_HALF_LIFE = 2.5

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
             prior_start=None, duration_half_life=DURATION_HALF_LIFE,
             zero_weight=ZERO_WEIGHT):
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
        age = (now - gw)
        w = 0.5 ** (age / half_life) * slots
        dw = 0.5 ** (age / duration_half_life) * slots
        per = mins / slots
        started = starts / slots
        if per <= 0 and started < 0.5:
            w *= zero_weight            # he did not play; the feed will not say why
        sw += w
        sm += w * started
        if started >= 0.5:
            fw += dw
            fm += dw * max(per, MIN_START_MINUTES)
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
