"""Clean sheets, goals conceded and saves — modelled, not guessed.

Attacking returns are lumpy and hard to predict. Defensive returns are the
opposite: they are close to a Poisson process driven by two numbers, how many
goals a team concedes and how good the opponent's attack is. That makes them the
most forecastable points in the game, and the reason a cheap defender on a good
run is often better value than a mid-price midfielder.

Everything below is built from expected goals conceded rather than fixture
difficulty ratings, then calibrated against Solio's published match odds for the
teams it covers.
"""
import math

CS_POINTS = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}


def _poisson(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def team_xgc(boot, prior_games=6.0):
    """Minutes-weighted expected goals conceded per 90, per team, shrunk toward
    the league average.

    Each player's expected_goals_conceded_per_90 is a team-level figure measured
    over his own minutes, so weighting by minutes recovers the team's rate
    without double counting.

    The shrinkage matters more than the measurement. Two matches of data will
    happily tell you a side concedes 0.3 expected goals a game, and a model that
    believes it will hand out 68% clean sheet odds that no bookmaker would
    offer. `prior_games` is how many games of league-average evidence to weigh
    against what has actually been observed."""
    num, den = {}, {}
    for e in boot["elements"]:
        if e["minutes"] < 45:
            continue
        t = e["team"]
        num[t] = num.get(t, 0.0) + float(e["expected_goals_conceded_per_90"]) * e["minutes"]
        den[t] = den.get(t, 0) + e["minutes"]
    raw = {t: num[t] / den[t] for t in num if den[t] > 0}
    if not raw:
        return {}
    league = sum(raw.values()) / len(raw)
    games = {t: den[t] / (11 * 90) for t in raw}
    return {t: (games[t] * raw[t] + prior_games * league) / (games[t] + prior_games)
            for t in raw}


def lambdas(boot, fixtures, gws, atk, solio_feed=None, home_factor=0.88,
            market=None):
    """Expected goals conceded for every team in every gameweek.

    lambda = the team's own concession rate, scaled by how good the opponent's
    attack is, adjusted for home advantage.

    `market` overrides that model where a bookmaker has priced the fixture,
    as {(home name, away name): (p_home, p_draw, p_away)} devigged. It is
    not a refinement, it is a replacement, and the reason is measured:
    scored over sixty played matches, the market beat this model on both
    Brier (0.6307 against 0.6523) and log loss (1.0471 against 1.0794) —
    and did so even though our rates were computed with today's team
    strength, which is lookahead bias working in our favour.

    The failure is not scale but *spread*. Across twenty upcoming fixtures
    our p(home) ran 0.30 to 0.56 where the market ran 0.21 to 0.68: a
    correlation of 0.82, so the ranking is sound, against a regression
    slope of 1.55, so the sizing is not. Three shrinkages stack — xGC
    toward the league mean over six games of prior, the opponent-attack
    clamp, then Solio blending — and each is defensible while the three
    together flatten every fixture toward a coin toss. That under-rates
    good teams in good fixtures, which is exactly the decision a manager
    is making.
    """
    tm = {t["id"]: t["short_name"] for t in boot["teams"]}
    base = team_xgc(boot)
    league = sum(base.values()) / max(1, len(base))

    out = {t["id"]: {} for t in boot["teams"]}
    for x in fixtures:
        g = x["event"]
        if g not in gws:
            continue
        for side, opp, home in ((x["team_h"], x["team_a"], True),
                                (x["team_a"], x["team_h"], False)):
            lam = base.get(side, league) * max(0.45, min(2.2, atk.get(opp, 1.0)))
            lam *= home_factor if home else 1 / home_factor
            out[side][g] = max(0.25, lam)

    # Where the market has spoken, use it. Recorded per team-gameweek so the
    # Solio calibration below cannot drag a measured rate back toward a
    # modelled one.
    priced = set()
    if market:
        full = {t["id"]: t["name"] for t in boot["teams"]}
        for x in fixtures:
            g = x["event"]
            if g not in gws:
                continue
            h, a = x["team_h"], x["team_a"]
            pr = market.get((full.get(h), full.get(a)))
            if not pr:
                continue
            lh, la = _market_lambdas(*pr)
            # out[t][g] is what t CONCEDES, so each side takes the other's
            # scoring rate. Getting this backwards would invert every clean
            # sheet in the gameweek.
            out[h][g] = max(0.25, la)
            out[a][g] = max(0.25, lh)
            priced.add((h, g))
            priced.add((a, g))

    # Calibrate against Solio's published expected goals against, for the teams
    # it covers. Using a better model to correct our own where they overlap
    # costs nothing and fixes systematic bias.
    if solio_feed:
        gw0 = min(gws)
        ratios = []
        by_name = {t["name"]: t["id"] for t in boot["teams"]}
        for row in solio_feed.get("bestCleanSheets", []):
            tid = by_name.get(row["team"])
            if tid and out.get(tid, {}).get(gw0):
                ratios.append(row["prGoalsAgainst"] / out[tid][gw0])
        if len(ratios) >= 4:
            k = sorted(ratios)[len(ratios) // 2]      # median, robust to outliers
            for t in out:
                for g in out[t]:
                    if (t, g) not in priced:
                        out[t][g] *= k
    return out, tm


def _market_lambdas(ph, pd_, pa):
    """Goal rates whose scoreline matrix reproduces a devigged 1X2."""
    from .ucl.project import lambdas_from_odds
    return lambdas_from_odds(ph, pd_, pa)


def clean_sheet(lam):
    return math.exp(-lam)


def concede_penalty(lam, max_goals=8):
    """FPL deducts a point for every two goals conceded, so the cost is the
    expected value of floor(goals / 2), not half the expected goals."""
    return -sum(_poisson(k, lam) * (k // 2) for k in range(max_goals + 1))


def defensive_points(pos, lam, saves90=0.0, defcon_prob=0.0):
    """Expected defensive points per 90 for one player in one fixture."""
    pts = CS_POINTS.get(pos, 0) * clean_sheet(lam)
    if pos in ("GKP", "DEF"):
        pts += concede_penalty(lam)
    if pos == "GKP":
        pts += saves90 / 3.0                       # a point per three saves
    pts += 2.0 * defcon_prob                       # defensive contribution points
    return pts


def table(boot, lams, tm, gw, limit=20):
    """Clean sheet odds for the upcoming gameweek, best first."""
    rows = []
    for tid, per_gw in lams.items():
        if gw not in per_gw:
            continue
        lam = per_gw[gw]
        rows.append({"team": tm[tid], "xgc": round(lam, 2),
                     "cs_prob": round(clean_sheet(lam), 3),
                     "concede_cost": round(concede_penalty(lam), 2)})
    rows.sort(key=lambda r: -r["cs_prob"])
    return rows[:limit]


DEFCON_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}


def defcon_probability(element, pos):
    """Chance of hitting the defensive contribution threshold in a match.

    Defenders need 10 clearances, blocks, interceptions and tackles; everyone
    else needs 12 of those plus recoveries. Volume like this is close to
    Poisson, and far more repeatable than goals."""
    thr = DEFCON_THRESHOLD.get(pos)
    if not thr or element["minutes"] < 90:
        return 0.0
    actions = element["clearances_blocks_interceptions"] + element["tackles"]
    if pos in ("MID", "FWD"):
        actions += element["recoveries"]
    rate = actions / (element["minutes"] / 90)
    return 1.0 - sum(_poisson(k, rate) for k in range(thr))
