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

# Below this many goals per game, a position's league baseline is too small
# to divide by and the index is not reported at all.
#
# This was a real bug. Goalkeepers score about 0.003 goals per game across
# the league — three thousandths. One freak goalkeeper goal all season gives
# that team a rate of ~0.17, which divided by 0.003 is an index of 55, and
# the clamp dutifully reported it as 1.25: "Forest are weak to goalkeepers",
# from a single event, indistinguishable on the page from a real finding.
# A ratio is only meaningful when its denominator is.
MIN_BASELINE = 0.08


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


def concession_by_position(boot, fixtures, live_by_gw, prior=None):
    """What each defence gives up, split by the position that scored it.

    Returns raw counts, games played, and a SHRUNK rate per 90 against the
    league average — the shrunk figure is the only one safe to consume.
    """
    names = {t["id"]: t["short_name"] for t in boot["teams"]}
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
            if avg < MIN_BASELINE:
                # Not measurable. Say so rather than printing a number that
                # looks like the others and is not.
                row["index"][p] = None
                row.setdefault("unmeasured", []).append(p)
                continue
            # Shrink toward the prior — last season's profile where we have
            # one, the league average where we do not.
            base = ((prior or {}).get(names.get(t, t)) or {}).get(p, 1.0)
            rate = conceded[t][p] / max(1, n)
            w = n / (n + PRIOR_GAMES)
            row["index"][p] = round(max(CLAMP[0], min(CLAMP[1],
                                                      w * (rate / avg) + (1 - w) * base)), 3)
        out[t] = row
    return out, league


def multiplier(profile, opponent_id, pos, weight=0.0):
    """How much to tilt a projection for this opponent, at `weight`.

    weight 0 returns exactly 1.0, so the signal can ship and be measured
    before it is ever allowed to change a recommendation.
    """
    if not weight:
        return 1.0
    idx = ((profile or {}).get(opponent_id) or {}).get("index", {}).get(pos)
    if idx is None:          # unmeasured position, or unknown opponent
        return 1.0
    return max(CLAMP[0], min(CLAMP[1], 1.0 + weight * (idx - 1.0)))


def build(boot, fixtures, live_by_gw, weight=0.0, prior=None):
    names = {t["id"]: t["short_name"] for t in boot["teams"]}
    profile, league = concession_by_position(boot, fixtures, live_by_gw,
                                             prior=prior)
    venues = venue_splits(fixtures, [t["id"] for t in boot["teams"]])
    return {
        "weight": weight,
        "league_per_game": {p: round(v, 3) for p, v in league.items()},
        "by_team": {names[t]: {**row, "venue": venues.get(t)}
                    for t, row in profile.items()},
        "prior_teams": sorted(prior or {}),
        "note": ("Concession split by the position that scored, shrunk with 8 "
                 "games of prior toward last season's profile where there is "
                 "one and the league average otherwise. A position whose "
                 "league baseline is under "
                 f"{MIN_BASELINE} goals a game reports null rather than a "
                 "ratio with no denominator. weight 0.0 means it is measured "
                 "and reported but does not move a projection."),
    }


# --- last season as the prior ------------------------------------------------
# Six gameweeks gives a side two or three goals conceded per position, which
# is not a rate. Previous seasons give thirty-eight games each, and the same
# vaastav archive the models already train on carries what we need: one row
# per player per gameweek with position, goals, assists, fixture and
# was_home. Who a team conceded to last season is a far better starting
# guess than "exactly average".
#
# It is a prior and nothing more. Squads turn over, managers change, and a
# promoted side has no history at all — those fall back to the league mean.

SEASON_DECAY = 0.5        # each season further back counts half as much


def _opponents(df):
    """Map every row to the club it was played against.

    merged_gw carries `opponent_team` as a per-season numeric id with no
    name attached, so the names are recovered from the fixture itself: in
    any fixture the rows with was_home True are one club and the rest the
    other.
    """
    sides = (df.groupby(["season", "fixture", "was_home"])["team"]
               .agg(lambda x: x.mode().iat[0] if len(x) else None)
               .reset_index())
    home = sides[sides["was_home"]].set_index(["season", "fixture"])["team"]
    away = sides[~sides["was_home"]].set_index(["season", "fixture"])["team"]
    key = list(zip(df["season"], df["fixture"]))
    return [ (away.get(k) if h else home.get(k)) for k, h in zip(key, df["was_home"]) ]


def historical(df, seasons=None):
    """Concession index per club per position, from completed seasons."""
    need = {"season", "fixture", "was_home", "team", "position",
            "goals_scored", "assists"}
    if df is None or not need.issubset(set(df.columns)):
        return {}
    d = df.copy()
    if seasons:
        d = d[d["season"].isin(seasons)]
    if d.empty:
        return {}
    d["against"] = _opponents(d)
    d = d[d["against"].notna()]
    d["pos"] = d["position"].map({"GK": "GKP", "GKP": "GKP", "DEF": "DEF",
                                  "MID": "MID", "FWD": "FWD"})
    d = d[d["pos"].notna()]
    d["conceded"] = d["goals_scored"].fillna(0) + d["assists"].fillna(0) / 3.0

    order = sorted(d["season"].unique(), reverse=True)
    wt = {s: SEASON_DECAY ** i for i, s in enumerate(order)}
    d["w"] = d["season"].map(wt)
    d["cw"] = d["conceded"] * d["w"]

    # games each club played, weighted the same way
    games = (d.groupby(["season", "against", "fixture"]).size().reset_index()
               .groupby(["season", "against"]).size().reset_index(name="g"))
    games["gw"] = games["g"] * games["season"].map(wt)
    gsum = games.groupby("against")["gw"].sum()

    tot = d.groupby(["against", "pos"])["cw"].sum()
    rate = (tot / gsum).dropna()
    league = rate.groupby(level="pos").mean()

    out = {}
    for (team, pos), r in rate.items():
        avg = league.get(pos, 0.0)
        if avg < MIN_BASELINE:
            continue
        out.setdefault(team, {})[pos] = round(
            max(CLAMP[0], min(CLAMP[1], r / avg)), 3)
    return out
