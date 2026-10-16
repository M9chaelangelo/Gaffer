"""Predicting blanks and doubles the way Ben Crellin does — with probabilities.

A blank gameweek is not a mystery. It happens when a league fixture is
postponed because one of the clubs has a cup tie that weekend, and a double
happens when the postponed match is rescheduled into a free midweek. Both
follow mechanically from the cup calendar and from who is still in each cup.
The calendar is published a year ahead. Who is still in is the only unknown,
and it is a knockout bracket, which is exactly the kind of thing a model can
run ten thousand times.

So this does not forecast a blank: it gives every club a probability of
blanking, which is strictly more than a hand-made calendar can say. Crellin's
"GW33 BGW" means four clubs will be in the FA Cup semi-finals. This says
which four, and how sure.

What is known, and what is assumed:

  **Known.** The round dates, which league gameweek each round collides with,
  and that all twenty Premier League clubs enter the FA Cup at the third
  round. Those come from the published calendar.

  **Modelled.** Who survives each round. Ties are drawn at random, so the
  bracket is simulated rather than solved: clubs are shuffled into pairs each
  round and a winner is drawn from their relative strength. Non-league and
  lower-division opponents are in the pool at a declared strength, because a
  Premier League club's third-round tie is usually not against another one.

  **Declared.** Which clubs remain in the EFL Cup, because that competition
  started before this engine did and its draw is not in any feed we can read.
  Unknown means unknown: the EFL numbers are withheld rather than guessed.
"""
from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(30)
N = 20_000

# A lower-division opponent's strength on the same scale as `strengths`,
# where Premier League clubs sit around 1.0.
#
# CALIBRATED, not guessed. The check is the one number this model can be held
# to without waiting a season: historically about 3.4 of the four FA Cup
# semi-finalists are Premier League clubs. A first pass at 0.45 with a 12%
# upset term produced 2.59, which would have quietly understated every
# blank probability by a quarter. Swept against that target:
#
#     strength  upset   E[PL in semi]
#       0.45     0.12       2.59
#       0.35     0.12       2.90
#       0.30     0.08       3.16
#       0.25     0.08       3.33
#       0.20     0.12       3.42   <- here
#
# Re-calibrate if the target moves; the sweep is in the commit message.
LOWER_LEAGUE_STRENGTH = 0.20
PL_SEMI_FINALISTS_TARGET = 3.4    # what the calibration is held to


def win_prob(a, b, upset=0.12):
    """P(a beats b) over a tie, from relative strength.

    `upset` pulls the result toward a coin flip. One-off knockout football is
    far noisier than a league season, and a model that sends the strongest
    side through 95% of the time will predict the wrong four semi-finalists
    with great confidence.
    """
    a, b = max(float(a), 1e-6), max(float(b), 1e-6)
    p = a / (a + b)
    return (1 - upset) * p + upset * 0.5


def simulate_rounds(entrants, strengths, rounds, field_size=None, n=N):
    """P(each club is still in) at the start of every round.

    `entrants` are the Premier League clubs entering at round 0. `field_size`
    pads the draw with lower-division clubs so the early rounds are not an
    all-Premier-League bracket, which would eliminate half of them a round.
    """
    entrants = list(entrants)
    pad = max(0, (field_size or len(entrants)) - len(entrants))
    pool0 = ([(c, float(strengths.get(c, {}).get("overall", 1.0))) for c in entrants]
             + [(None, LOWER_LEAGUE_STRENGTH)] * pad)

    alive = {r: {c: 0 for c in entrants} for r in range(rounds + 1)}
    for _ in range(n):
        pool = list(pool0)
        for r in range(rounds + 1):
            for c, _s in pool:
                if c is not None:
                    alive[r][c] += 1
            if r == rounds or len(pool) < 2:
                break
            RNG.shuffle(pool)
            nxt = []
            for i in range(0, len(pool) - 1, 2):
                a, b = pool[i], pool[i + 1]
                nxt.append(a if RNG.random() < win_prob(a[1], b[1]) else b)
            if len(pool) % 2:
                nxt.append(pool[-1])        # a bye
            pool = nxt
    return {r: {c: v / n for c, v in d.items()} for r, d in alive.items()}


def fa_cup(strengths, teams, n=N):
    """Blank probabilities from the FA Cup, which every league club enters.

    Rounds from the third, where the Premier League comes in: 3rd, 4th, 5th,
    quarter-final, semi-final, final. Reaching the semi-final takes four wins,
    the final five.
    """
    survive = simulate_rounds(teams, strengths, rounds=5, field_size=64, n=n)
    return {
        "semi_final": survive[4],      # alive at the semi -> blanks that weekend
        "final": survive[5],
        "quarter_final": survive[3],
    }


def blanks(cfg, strengths, teams, n=N):
    """P(blank) per club per gameweek, from the declared cup collisions.

    `cfg["cup_blanks"]` maps a gameweek to the round that displaces it:

        cup_blanks:
          33: {cup: fa, round: semi_final}
          37: {cup: fa, round: final}
          30: {cup: efl, round: final}
    """
    fa = fa_cup(strengths, teams, n=n)
    out = {}
    for gw, spec in (cfg.get("cup_blanks") or {}).items():
        cup = str(spec.get("cup", "")).lower()
        rnd = str(spec.get("round", ""))
        if cup == "fa" and rnd in fa:
            probs = fa[rnd]
            out[int(gw)] = {
                "cup": "FA Cup", "round": rnd, "known": True,
                "clubs": sorted(({"team": c, "p_blank": round(p, 3)}
                                 for c, p in probs.items() if p >= 0.01),
                                key=lambda r: -r["p_blank"]),
                "expected_blanking": round(sum(probs.values()), 2),
            }
        else:
            # The EFL Cup began before this engine did and its draw is in no
            # feed here. Saying nothing is better than guessing two finalists.
            out[int(gw)] = {
                "cup": cup.upper() or "unknown", "round": rnd, "known": False,
                "note": ("participants not declared — add `cup_entrants` for "
                         "this competition, or take the call from a published "
                         "calendar"),
            }
    return out


def doubles(cfg, blank_probs):
    """P(double) per club per gameweek.

    A postponed league match is replayed in a free midweek, so the clubs that
    blank are the clubs that later double. Which week it lands in is a
    scheduling decision, not a model output, so the candidate weeks are
    declared and the probability is carried across from the blank.
    """
    out = {}
    for gw, spec in (cfg.get("cup_doubles") or {}).items():
        src = int(spec.get("from_blank_gw", 0))
        b = blank_probs.get(src) or {}
        if not b.get("known"):
            out[int(gw)] = {"known": False, "from_blank_gw": src,
                            "note": b.get("note", "source blank is unknown")}
            continue
        share = float(spec.get("share", 1.0))
        out[int(gw)] = {
            "known": True, "from_blank_gw": src,
            "clubs": [{"team": r["team"], "p_double": round(r["p_blank"] * share, 3)}
                      for r in b["clubs"]],
            "note": (f"carried from the GW{src} blank; `share` {share} is the "
                     "chance the replay lands in this week rather than another"),
        }
    return out


def build(cfg, strengths, teams, n=N):
    b = blanks(cfg, strengths, teams, n=n)
    return {"blanks": b, "doubles": doubles(cfg, b),
            "method": ("knockout brackets simulated with a random draw and a "
                       "12% pull toward a coin flip; league clubs enter the FA "
                       "Cup at the third round")}
