"""Expected points per player per gameweek.

Two sources, blended:
  1. Solio's public feed - a genuinely strong model, but only publishes leaders.
  2. A local fallback - shrunk points-per-90 scaled by minutes and fixture.

Never trust either blindly. `confidence` tells the report how firm a number is.
"""
DIFF = {1: 1.35, 2: 1.22, 3: 1.02, 4: 0.86, 5: 0.72}

try:
    from . import learn
except Exception:
    learn = None
try:
    from . import setpieces
except Exception:
    setpieces = None
from . import defence, teamform

GOAL_POINTS = {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}


def _minutes_model(boot, played):
    """If a trained minutes model exists, use it for P(60+). It is the single
    biggest source of projection error, and the model beats the heuristic."""
    if learn is None:
        return None
    bundle = learn.load("minutes")
    if not bundle:
        return None
    import pandas as pd
    rows, ids = [], []
    for e in boot["elements"]:
        rows.append({"minutes": e["minutes"],
                     "roll_minutes": e["minutes"] / played,
                     "starts": e["starts"],
                     "value": e["now_cost"],
                     "roll_total_points": e["total_points"] / played})
        ids.append(e["id"])
    X = pd.DataFrame(rows)[bundle["features"]].astype(float)
    return dict(zip(ids, learn.probability(bundle, X)))


def _ceiling_model(boot, played, fx, lams, gws):
    """P(10+ points), per player per gameweek.

    This is the model the README already claimed captaincy used. It did not:
    `projections.py` loaded `minutes` and nothing else, so nine trained models
    sat in models/ as dead weight and every decision came off expected points.

    Expected points and ceiling rank players differently, and that difference
    is the whole argument for the Triple Captain chip — it only cares about the
    tail. A 6.0 projection that hauls one week in five is a better armband than
    a 6.4 that never does, and a mean cannot tell you which is which.

    Predicted in one batch. Seven hundred players across five gameweeks is 3,500
    single-row predicts, and scikit-learn's per-call overhead dwarfs the
    arithmetic; batching turns seconds into milliseconds.

    Returns {player_id: {gw: probability}}, or None when the model is missing —
    the caller then carries on without it rather than failing the solve.
    """
    if learn is None:
        return None
    bundle = learn.load("ceiling")
    if not bundle:
        return None
    import pandas as pd

    def f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    rows, keys = [], []
    for e in boot["elements"]:
        # A player with no fixture has no ceiling to predict.
        for g in gws:
            for (_d, home, _o) in fx[e["team"]][g]:
                rows.append({
                    "roll_minutes": e["minutes"] / played,
                    "roll_total_points": e["total_points"] / played,
                    "roll_expected_goal_involvements":
                        f(e.get("expected_goal_involvements")) / played,
                    "roll_bps": e["bps"] / played,
                    "value": e["now_cost"],
                    "was_home": 1.0 if home else 0.0,
                    # The opponent's attack is what threatens the clean sheet,
                    # but for a ceiling it is this team's own xGC that marks a
                    # game likely to be open — and open games are where hauls
                    # come from.
                    "expected_goals_conceded": lams.get(e["team"], {}).get(g) or 0.0,
                    "roll_threat": f(e.get("threat")) / played,
                    "roll_ict_index": f(e.get("ict_index")) / played,
                })
                keys.append((e["id"], g))

    if not rows:
        return None

    X = pd.DataFrame(rows)[bundle["features"]].astype(float)
    # Through `learn.probability`, so the isotonic map trained with the model
    # is applied. Read raw, this model says 61% where the truth is 20%.
    probs = learn.probability(bundle, X)

    out = {}
    for (pid, g), pr in zip(keys, probs):
        # A double gameweek gives two rows for the same player and week. The
        # chance of hauling in EITHER match is the complement of missing both.
        prev = out.setdefault(pid, {}).get(g)
        out[pid][g] = pr if prev is None else 1 - (1 - prev) * (1 - pr)
    return out


# Same codes the training panel uses. Wrong numbers here would not raise —
# the model would simply be told every defender is a forward.
POS_CODE = {"GKP": 1.0, "GK": 1.0, "DEF": 2.0, "MID": 3.0, "FWD": 4.0}


def _bonus_model(boot, played, fx, lams, gws, postypes):
    """Expected bonus points, per player per gameweek.

    Bonus is the last unmodelled chunk of a defender's score. The projection
    used `bps90 / 34` — crude, but it tracked the thing that actually decides
    bonus, and monotonically.

    The model that replaced it has to clear that bar, so this returns None
    unless it is the classifier: an earlier bonus model regressed the mean,
    learned price, and would have made defenders worse while looking wired in.
    A bundle without `predict_proba` is that model, and the caller falls back
    to the heuristic rather than trusting it.

    Returns {player_id: {gw: expected bonus}}, or None.
    """
    if learn is None:
        return None
    bundle = learn.load("bonus")
    if not bundle or not hasattr(bundle["model"], "predict_proba"):
        return None
    import pandas as pd

    def f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    rows, keys = [], []
    for e in boot["elements"]:
        code = POS_CODE.get(postypes.get(e["element_type"]))
        if code is None:
            continue
        for g in gws:
            for (_d, home, _o) in fx[e["team"]][g]:
                rows.append({
                    "roll_bps": e["bps"] / played,
                    "roll_minutes": e["minutes"] / played,
                    "roll_expected_goal_involvements":
                        f(e.get("expected_goal_involvements")) / played,
                    "roll_influence": f(e.get("influence")) / played,
                    "roll_threat": f(e.get("threat")) / played,
                    "value": e["now_cost"],
                    "was_home": 1.0 if home else 0.0,
                    "pos_code": code,
                    # An open game hands out more bonus at both ends.
                    "expected_goals_conceded":
                        lams.get(e["team"], {}).get(g) or 0.0,
                })
                keys.append((e["id"], g))
    if not rows:
        return None
    try:
        X = pd.DataFrame(rows)[bundle["features"]].astype(float)
        vals = learn.expected_bonus(bundle["model"], X)
    except (KeyError, ValueError):
        # A bundle trained on features this build cannot assemble. Say nothing
        # and fall back; a half-built feature row would predict confidently
        # from zeros.
        return None

    out = {}
    for (pid, g), v in zip(keys, vals):
        # Two matches in a week are two goes at bonus, so these add. That is
        # the opposite of the ceiling model, where the question is whether he
        # hauls at all — and getting the two the same way round matters.
        out.setdefault(pid, {})
        out[pid][g] = out[pid].get(g, 0.0) + float(v)
    return out


def _prior(price, pos):
    if pos == "GKP":
        return 3.2 + (price - 4.0) * 0.50
    if pos == "DEF":
        return 2.6 + (price - 4.0) * 0.90
    if pos == "MID":
        return 2.4 + (price - 4.5) * 0.75
    return 2.6 + (price - 4.5) * 0.55


def _availability(e):
    if e["status"] in ("i", "u", "s", "n"):
        return 0.0
    c = e.get("chance_of_playing_next_round")
    return 1.0 if c is None else c / 100.0


def team_strength(boot, prior_games=6.0, fixtures=None, form_weight=0.0,
                  half_life=2.5, form_prior=2.5):
    """Attack and defence strength from what teams have actually produced, not
    from FPL's fixture difficulty rating.

    FDR is a preseason label on a five-point scale and it does not move when a
    side starts scoring three a game. Expected goals created and conceded do.

    Season totals move slowly too, though. Pass `fixtures` with a `form_weight`
    above zero and recent results are folded in on top, opponent-adjusted and
    time-decayed — so a side in the middle of a run is rated for the run rather
    than for its average since August. See `teamform`."""
    xg, xgc, mins = {}, {}, {}
    for e in boot["elements"]:
        if e["minutes"] < 45:
            continue
        t = e["team"]
        xg[t] = xg.get(t, 0.0) + float(e["expected_goals"]) + \
            float(e["expected_assists"])
        mins[t] = mins.get(t, 0) + e["minutes"]
        xgc[t] = xgc.get(t, 0.0) + float(e["expected_goals_conceded_per_90"])
    if not mins:
        # Nobody has played forty-five minutes yet — preseason, or the first
        # round before a ball is kicked. Every side is average until one has.
        # Crashing here would take the whole solve with it.
        flat = {t["id"]: 1.0 for t in boot["teams"]}
        return flat, dict(flat)
    games = {t: max(1.0, m / (11 * 90)) for t, m in mins.items()}
    raw_atk = {t: xg.get(t, 0.0) / games[t] for t in games}
    # Shrink toward the league mean. Two matches will cheerfully claim a side
    # creates twice the league average, and a projection that believes it
    # compounds the error across every player in that team.
    a_mean = sum(raw_atk.values()) / len(raw_atk)
    atk = {t: (games[t] * raw_atk[t] + prior_games * a_mean) / (games[t] + prior_games)
           for t in games}
    n_def = {t: xgc.get(t, 0.0) / max(1, sum(
        1 for e in boot["elements"] if e["team"] == t and e["minutes"] >= 45))
        for t in games}
    a_avg = sum(atk.values()) / len(atk)
    d_avg = sum(n_def.values()) / len(n_def)
    season_atk = {t: v / a_avg for t, v in atk.items()}
    season_dfn = {t: v / d_avg for t, v in n_def.items()}
    if not fixtures or form_weight <= 0:
        return season_atk, season_dfn

    # Recent results, judged against what these very strengths expected — so
    # the form measure cannot simply reward a soft run of fixtures.
    fa, fd = teamform.recent(boot, fixtures, season_atk, season_dfn,
                             half_life=half_life, prior_games=form_prior)
    return (teamform.blend(season_atk, fa, form_weight),
            teamform.blend(season_dfn, fd, form_weight))


def _scoring_rate(e, pos, mins, luck_weight):
    """Points per 90, with finishing luck taken back out of the attack.

    `total_points / 90` is what a player HAS scored, and most of the gap
    between that and what his chances were worth is noise. Projecting it
    forward forecasts the noise. Measured on 19 Oct 2026, three of the
    manager's own players were the three largest overperformers in the game:

        Groß        xGI 3.32, 9 involvements   +18.7 points, 32% of his total
        Kostoulas   xGI 1.95, 7 involvements   +16.3 points, 40% of his total
        João Pedro  xGI 3.36, 8 involvements   +15.9 points, 34% of his total

    and Saka, on the second-highest expected involvement in the squad, was
    -6.7 — so the engine wanted to sell him and buy them.

    How much to take back out is not a guess. Splitting the season at GW4 and
    correlating each player's over-performance per 90 across the two halves
    gives r = +0.29 over the 69 players with real chance volume in both: some
    finishing skill is real and persistent, but it explains under a tenth of
    the variance. Keeping about that fraction of the observed gap, and
    discarding the rest, is what `luck_weight` does at 0.7.

    Only goals and assists are rebuilt. Bonus is left as scored even though it
    correlates with them, so this under-corrects rather than over-corrects.
    """
    per90 = 90.0 / mins
    realised = e["total_points"] * per90
    if not luck_weight:
        return realised
    if "expected_goals" not in e or "goals_scored" not in e:
        # No chance data, so there is no luck to take out. Correcting a number
        # we cannot measure would be inventing one.
        return realised
    g = GOAL_POINTS[pos]
    attack_scored = (float(e["goals_scored"] or 0) * g
                     + float(e.get("assists") or 0) * 3) * per90
    attack_expected = (float(e["expected_goals"] or 0) * g
                       + float(e.get("expected_assists") or 0) * 3) * per90
    underlying = realised - attack_scored + attack_expected
    return (1 - luck_weight) * realised + luck_weight * underlying


def build(boot, fixtures, gws, solio_feed=None, solio_weight=0.7,
          form_weight=0.30, strength_weight=0.5, minutes_mult=None,
          team_form_weight=0.0, team_form_half_life=2.5, market=None,
          playtime_table=None, luck_weight=0.7):
    teams = {t["id"]: t["short_name"] for t in boot["teams"]}
    # One change, felt everywhere: these two dicts drive the clean-sheet
    # lambdas, the ceiling model's input, and every player's fixture
    # multiplier. Folding form in here beats bolting it onto each in turn.
    atk, dfn = team_strength(boot, fixtures=fixtures,
                             form_weight=team_form_weight,
                             half_life=team_form_half_life)
    # Expected goals conceded per team per gameweek, calibrated against Solio.
    lams, _ = defence.lambdas(boot, fixtures, gws, atk, solio_feed,
                              market=market)
    postypes = {p["id"]: p["singular_name_short"] for p in boot["element_types"]}
    # The 'finished' flag lags, so infer rounds played from actual minutes.
    played = max(1, round(max(e["minutes"] for e in boot["elements"]) / 90))

    p60 = _minutes_model(boot, played)

    fx = {t["id"]: {g: [] for g in gws} for t in boot["teams"]}
    for x in fixtures:
        g = x["event"]
        if g in gws:
            fx[x["team_h"]][g].append((x["team_h_difficulty"], True, x["team_a"]))
            fx[x["team_a"]][g].append((x["team_a_difficulty"], False, x["team_h"]))

    # P(10+ points) per player per gameweek. Needs fx and lams, so it is built
    # here rather than alongside the minutes model.
    ceil = _ceiling_model(boot, played, fx, lams, gws)
    bonus = _bonus_model(boot, played, fx, lams, gws, postypes)

    # Solio publishes name+team, so key on that.
    solio = {}
    if solio_feed:
        for key in ("topProjected", "topCaptains", "topDifferentials", "topGoals",
                    "topAssists", "topBonus", "topDefCon"):
            for r in solio_feed.get(key, []):
                k = (r["name"], r["team"])
                solio[k] = max(solio.get(k, 0.0), r.get("prPoints", 0.0))

    out = {}
    for e in boot["elements"]:
        price = e["now_cost"] / 10
        pos = postypes[e["element_type"]]
        mins = e["minutes"]
        avg_min = mins / played
        role = (playtime_table or {}).get(e["id"])
        if role is not None:
            # Measured from his own match log rather than from the season's
            # round count. `mins / played` charges a man for gameweeks he was
            # never in the side for and credits him for weeks he no longer
            # plays: Barcola, who won his place in GW4 and has since played
            # 71, 87 and 90, read as 60 expected minutes; Barkley, who lost
            # his in GW5 and has since played 6, 4 and 18, read as 57. Both
            # errors are large and minutes scale almost the whole projection.
            # See playtime.py for the fit.
            exp_min = role["exp_min"] * _availability(e)
            # `avg_min` has one other consumer below — the bonus model's
            # "is this a normal week for him" ratio — and what that wants is
            # his minutes WHEN HE PLAYS, which is exactly mins_if_start.
            avg_min = role["mins_if_start"]
        elif p60 is not None and mins > 0:
            exp_min = (0.8 * avg_min + 0.2 * 90) if avg_min >= 70 else 0.85 * avg_min
            exp_min = min(90.0, exp_min) * _availability(e)
            # Blend the heuristic with the learned start probability.
            #
            # The blend used to be computed into a variable the gameweek loop
            # below never read — it took the pre-blend heuristic instead. So
            # the one trained model that was actually loaded moved no
            # projection at all; it survived only as a number on the card.
            #
            # And if he does not start, he does not play twenty minutes. A
            # flat twenty put a twelve-minute floor under every footballer in
            # the game, including one who has played once since August. The
            # cameo is capped by what he has actually averaged, so a man with
            # a minute to his name gets a minute.
            p = p60[e["id"]]
            cameo = min(18.0, avg_min)
            exp_min = 0.4 * exp_min + 0.6 * (p * 82 + (1 - p) * cameo)
            exp_min *= _availability(e)
        else:
            # No match log and no trained model: the season average is all
            # there is.
            exp_min = ((0.8 * avg_min + 0.2 * 90) if avg_min >= 70
                       else 0.85 * avg_min)
            exp_min = min(90.0, exp_min) * _availability(e)
        base_exp_min = exp_min

        # Shrink hard early in the season; trust observed rate more as minutes accrue.
        w = min(0.55, mins / 1200)
        rate = (_scoring_rate(e, pos, mins, luck_weight) if mins >= 45
                else _prior(price, pos))
        base = w * rate + (1 - w) * _prior(price, pos)

        # Form. FPL's `form` field is points per match over the last 30 days.
        # Against the season rate it says whether a player is trending up or
        # down — something a season total flattens away entirely.
        season_ppg = e["total_points"] / max(1, played)
        form = float(e["form"] or 0)
        if season_ppg > 0.5 and form > 0:
            ratio = max(0.6, min(1.6, form / season_ppg))
            base *= (1 - form_weight) + form_weight * ratio
        base = max(0.5, base)

        saves90 = float(e["saves_per_90"] or 0)
        defcon_p = defence.defcon_probability(e, pos)
        bps90 = e["bps"] / max(1e-9, mins / 90) if mins >= 45 else 0.0

        sol = solio.get((e["web_name"], teams[e["team"]]))
        ep, opp, conf = {}, {}, "low" if mins < 90 else "medium"
        # Haul probability per gameweek. Empty when the model is untrained, so
        # every consumer has to handle its absence rather than assume it.
        hauls = (ceil or {}).get(e["id"], {})
        for g in gws:
            tot, labels = 0.0, []
            lam = lams.get(e["team"], {}).get(g)
            # Congestion and European commitments cost minutes, and minutes are
            # what a projection is really made of.
            exp_min = base_exp_min * (minutes_mult.get(e["team"], {}).get(g, 1.0)
                                      if minutes_mult else 1.0)
            for (d, home, o) in fx[e["team"]][g]:
                mult = DIFF[d] * (1.045 if home else 0.955)
                # Blend the coarse fixture rating with measured strength: how
                # well this side is actually creating, and how leaky the
                # opponent actually is. A preseason FDR does not move when a
                # team starts scoring three a game; expected goals do.
                if pos in ("MID", "FWD"):
                    s = atk.get(e["team"], 1.0) * dfn.get(o, 1.0)
                else:
                    s = 1.0 / max(0.5, atk.get(o, 1.0))
                mult *= ((1 - strength_weight)
                         + strength_weight * max(0.6, min(1.7, s)))
                if pos in ("GKP", "DEF") and lam is not None:
                    # Defenders are built from components, not from a scaled
                    # points rate. Clean sheets and goals conceded are the bulk
                    # of their scoring and they are genuinely forecastable, so
                    # modelling them directly beats multiplying last month's
                    # points by a fixture rating.
                    per90 = 2.0
                    per90 += defence.defensive_points(
                        pos, lam, saves90, defcon_p)
                    per90 += (float(e["expected_goals_per_90"]) * GOAL_POINTS[pos]
                              + float(e["expected_assists_per_90"]) * 3)
                    bx = (bonus or {}).get(e["id"], {}).get(g)
                    if bx is None:
                        per90 += bps90 / 34.0      # rough bonus-point yield
                        tot += per90 * (exp_min / 90)
                    else:
                        # The model predicts a match, not ninety minutes, and
                        # it has already seen his usual minutes. What it cannot
                        # see is this particular week — a midweek tie in Europe,
                        # a doubt over a knock — so only that ratio is applied.
                        tot += per90 * (exp_min / 90)
                        tot += bx * min(1.5, exp_min / max(20.0, avg_min))
                else:
                    tot += base * (exp_min / 90) * mult
                    if pos == "MID" and lam is not None:
                        tot += defence.clean_sheet(lam) * (exp_min / 90)
                        tot += 2.0 * defcon_p * (exp_min / 90)
                labels.append(f"{teams[o]}({'H' if home else 'A'})")
            if sol is not None and g == min(gws):
                tot = solio_weight * sol + (1 - solio_weight) * tot
                conf = "high"
            if setpieces:
                tot += setpieces.bonus_per_90(e, pos) * (exp_min / 90)
            ep[g] = round(tot, 3)
            opp[g] = "+".join(labels) if labels else "BLANK"

        out[e["id"]] = {
            "id": e["id"], "name": e["web_name"], "team": teams[e["team"]],
            "team_id": e["team"], "pos": pos, "price": price, "ep": ep, "opp": opp,
            "exp_min": round(base_exp_min), "status": e["status"], "news": e["news"],
            "own": float(e["selected_by_percent"]), "total_points": e["total_points"],
            "confidence": conf, "solio": sol,
            "p60": round(float(p60[e["id"]]), 3) if p60 is not None else None,
            "xgi90": float(e["expected_goal_involvements_per_90"]),
            "setpiece_bonus": (setpieces.bonus_per_90(e, pos)
                               if setpieces else 0.0),
            # Expected bonus for the first gameweek in the horizon. None when
            # the classifier is not trained — the page must show the gap rather
            # than a zero that reads as "never gets bonus".
            "bonus_exp": (round(bonus[e["id"]][gws[0]], 3)
                          if bonus and gws[0] in bonus.get(e["id"], {}) else None),
            "bps90": round(bps90, 2),
            "defcon90": float(e["defensive_contribution_per_90"]),
            "defcon_prob": round(defcon_p, 3),
            "cs_prob": (round(defence.clean_sheet(lams[e["team"]][gws[0]]), 3)
                        if lams.get(e["team"], {}).get(gws[0]) else None),
            "xgc90": float(e["expected_goals_conceded_per_90"]),
            # P(10+ points) per gameweek, and the next week broken out because
            # that is the one the captaincy decision turns on.
            "ceiling": {g: round(float(hauls[g]), 4) for g in gws if g in hauls},
            "ceiling_next": (round(float(hauls[gws[0]]), 4)
                             if gws[0] in hauls else None),
        }
    return out
