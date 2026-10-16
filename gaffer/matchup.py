"""How two sides match up — not just how good each one is.

The projection already asks "how strong is this attack, how leaky is that
defence". It has never asked the question a human asks looking at a fixture:
*what does this opponent actually concede, and to whom.* A side can have a
respectable goals-against column and still be routinely opened up by
midfield runners, or keep clean sheets all season and then lose three from
corners.

Three things are derivable from feeds already pulled, with no extra calls:

  **Concession by position.** `event/{gw}/live` gives every player's goals
  and assists for a gameweek, and the fixture list says who they were
  against. Summed the other way round, that is what each defence gives up
  to goalkeepers, defenders, midfielders and forwards.

  **Venue splits.** Some sides are genuinely different home and away, and
  the projection applies one global 4.5% nudge to everybody.

  **Set-piece exposure.** Goals conceded where the scorer was a defender is
  the closest proxy available for being got at from corners, since FPL does
  not publish a set-piece flag on goals.

## Why the default weight is zero

Six gameweeks in, a team has conceded perhaps a dozen goals. Split four
ways that is two or three per position — a sample that will happily tell
you a side is "weak to midfielders" on the evidence of one deflection.
Everything here is shrunk hard toward the league average and clamped, and
`matchup_weight` ships at **0.0** so none of it moves a projection until
the ledger can show it helps. That is the same discipline the `ceiling`
model failed: a number that looks like a probability, consumed before
anyone checked it was one.
"""
from __future__ import annotations

POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
PRIOR_GAMES = 8.0          # matches of "exactly average" every side starts with
CLAMP = (0.80, 1.25)       # a matchup is a tilt, never a verdict


def _finished(fixtures):
    return [f for f in (fixtures or [])
            if f.get("finished") and f.get("team_h_score") is not None]


def venue_splits(fixtures, teams):
    """Goals for and against, home and away, per team."""
    out = {t: {"home": {"for": 0, "against": 0, "games": 0},
               "away": {"for": 0, "against": 0, "games": 0}} for t in teams}
    for f in _finished(fixtures):
        h, a = f["team_h"], f["team_a"]
        hs, as_ = int(f["team_h_score"]), int(f["team_a_score"])
        if h in out:
            out[h]["home"]["for"] += hs; out[h]["home"]["against"] += as_
            out[h]["home"]["games"] += 1
        if a in out:
            out[a]["away"]["for"] += as_; out[a]["away"]["against"] += hs
            out[a]["away"]["games"] += 1
    for t, d in out.items():
        for v in ("home", "away"):
            g = max(1, d[v]["games"])
            d[v]["for_per_game"] = round(d[v]["for"] / g, 2)
            d[v]["against_per_game"] = round(d[v]["against"] / g, 2)
    return out


def concession_by_position(boot, fixtures, live_by_gw):
    """What each defence gives up, split by the position that scored it.

    Returns raw counts, games played, and a SHRUNK rate per 90 against the
    league average — the shrunk figure is the only one safe to consume.
    """
    pos_of = {e["id"]: POS.get(e["element_type"], "?") for e in boot["elements"]}
    team_of = {e["id"]: e["team"] for e in boot["elements"]}
    teams = [t["id"] for t in boot["teams"]]
    # which opponent each team faced in each finished gameweek
    opp = {}
    for f in _finished(fixtures):
        g = f.get("event")
        if g is None:
            continue
        opp.setdefault(g, {})[f["team_h"]] = f["team_a"]
        opp[g][f["team_a"]] = f["team_h"]

    conceded = {t: {p: 0.0 for p in POS.values()} for t in teams}
    games = {t: 0 for t in teams}
    for g, payload in sorted((live_by_gw or {}).items()):
        if g not in opp:
            continue
        for t in opp[g]:
            games[t] = games.get(t, 0) + 1
        for el in (payload or {}).get("elements", []):
            s = el.get("stats") or {}
            gls = int(s.get("goals_scored") or 0)
            ast = int(s.get("assists") or 0)
            if not (gls or ast):
                continue
            tm = team_of.get(el["id"])
            against = opp.get(g, {}).get(tm)
            if against is None:
                continue
            # An assist is a chance conceded too, at a third of a goal's weight
            conceded[against][pos_of.get(el["id"], "?")] += gls + ast / 3.0

    league = {p: 0.0 for p in POS.values()}
    total_games = sum(games.values()) or 1
    for t in teams:
        for p in POS.values():
            league[p] += conceded[t][p]
    league = {p: v / total_games for p, v in league.items()}

    out = {}
    for t in teams:
        n = games.get(t, 0)
        row = {"games": n, "raw": {p: round(conceded[t][p], 2) for p in POS.values()},
               "index": {}}
        for p in POS.values():
            avg = league[p]
            if avg <= 0:
                row["index"][p] = 1.0
                continue
            # Shrink toward 1.0 with PRIOR_GAMES of average football.
            rate = conceded[t][p] / max(1, n)
            w = n / (n + PRIOR_GAMES)
            row["index"][p] = round(max(CLAMP[0], min(CLAMP[1],
                                                      w * (rate / avg) + (1 - w))), 3)
        out[t] = row
    return out, league


def multiplier(profile, opponent_id, pos, weight=0.0):
    """How much to tilt a projection for this opponent, at `weight`.

    weight 0 returns exactly 1.0, so the signal can ship and be measured
    before it is ever allowed to change a recommendation.
    """
    if not weight:
        return 1.0
    idx = ((profile or {}).get(opponent_id) or {}).get("index", {}).get(pos, 1.0)
    return max(CLAMP[0], min(CLAMP[1], 1.0 + weight * (idx - 1.0)))


def build(boot, fixtures, live_by_gw, weight=0.0):
    names = {t["id"]: t["short_name"] for t in boot["teams"]}
    profile, league = concession_by_position(boot, fixtures, live_by_gw)
    venues = venue_splits(fixtures, [t["id"] for t in boot["teams"]])
    return {
        "weight": weight,
        "league_per_game": {p: round(v, 3) for p, v in league.items()},
        "by_team": {names[t]: {**row, "venue": venues.get(t)}
                    for t, row in profile.items()},
        "note": ("Concession split by the position that scored, shrunk toward "
                 "the league average with 8 games of prior. weight 0.0 means "
                 "it is measured and reported but does not move a projection."),
    }
