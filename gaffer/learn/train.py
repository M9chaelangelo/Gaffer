"""The part that gets smarter over time.

Three models, each answering a question the optimiser cannot answer on its own:

  minutes  — will he actually play 60+? The single largest driver of points,
             and the thing a naive projection gets most wrong.
  price    — will he rise or fall tonight? Drives when to move, and how much
             budget the wildcard will have.
  points   — how many points, given expected minutes, xGI, defensive rate and
             fixture? Trained on outcomes rather than assumed from a prior.

Run `python -m gaffer.learn` to (re)train. Models are cached in models/ and
picked up automatically by projections.py on the next solve.
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold
import joblib

from .. import data

MODELS = "models"

MINUTES_X = ["minutes", "roll_minutes", "starts", "value", "roll_total_points"]
CEILING_X = ["roll_minutes", "roll_total_points", "roll_expected_goal_involvements",
             "roll_bps", "value", "was_home", "expected_goals_conceded",
             "roll_threat", "roll_ict_index"]
# Bonus is awarded off BPS, so BPS has to be the spine of this model. The
# first version regressed the mean and learned price instead: varying BPS per
# match from 0 to 60 moved its answer by 0.06 points, while varying price moved
# it by 0.35. It had found that expensive players collect bonus — which the
# projection already knows, several times over — and missed the mechanism.
#
# Two changes. The target is the class, not the mean: bonus is 0, 1, 2 or 3,
# and almost always 0, so a regressor is rewarded for predicting the base rate
# and an MAE near 0.15 looks good while saying nothing. Expected bonus comes
# back as the probability-weighted sum. And the features now include the
# position and how open the game is, so it no longer has to infer either from
# the price tag.
BONUS_X = ["roll_bps", "roll_minutes", "roll_expected_goal_involvements",
           "roll_influence", "roll_threat", "value", "was_home",
           "pos_code", "expected_goals_conceded"]
CS_X = ["roll_goals_conceded", "expected_goals_conceded", "was_home",
        "roll_clean_sheets", "value"]
DEFCON_X = ["roll_defensive_contribution", "roll_minutes", "roll_tackles",
            "roll_recoveries", "value"]
HERD_X = ["net_share", "roll_total_points", "selected", "value", "roll_bps"]
# How many minutes does a player actually get in his first week back, given he
# has been out? "Fit again" and "playing 90" are very different things.
RETURN_X = ["weeks_out", "mins_before_absence", "value", "roll_total_points",
            "roll_minutes"]
# Does a congested schedule cost him minutes? Rest days are the observable
# fingerprint of a midweek European tie.
ROTATION_X = ["rest_days", "next_rest_days", "congested", "roll_minutes",
              "value", "starts", "roll_total_points"]
PRICE_X = ["net_transfers", "net_share", "selected", "value",
           "roll_total_points", "transfers_balance"]
POINTS_X = ["roll_minutes", "roll_total_points", "roll_expected_goal_involvements",
            "roll_bps", "value", "was_home", "expected_goals_conceded"]


def expected_bonus(model, X):
    """Bonus points per match, as a number, from the class probabilities.

    The model answers "how likely is 0, 1, 2 or 3?". A projection needs one
    number, and the mean of that distribution is it. Defined here rather than
    at each call site so the scorer and the projection cannot drift apart —
    which is how the last bonus model came to be measured by a metric nobody
    could act on.
    """
    probs = model.predict_proba(X)
    return sum(float(k) * probs[:, i] for i, k in enumerate(model.classes_))


def _bonus_mae(frame):
    """The new model's error in the old model's units, so the two are
    comparable and the baseline means something."""
    import joblib
    bundle = joblib.load(f"{MODELS}/bonus.joblib")
    X = frame[bundle["features"]].astype(float)
    return mean_absolute_error(frame["next_bonus"],
                               expected_bonus(bundle["model"], X))


# The bands a reliability table is reported in. Deliberately fine at the top:
# a model used to pick a captain is only ever read above 0.1, and that is
# exactly where a coarse table hides the problem.
BANDS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 1.01)

# How many of the highest predictions the top of the calibration curve has to
# be estimated from. Isotonic regression puts a step wherever the data lets
# it, and the top step of a rare-event model is fitted on a handful of rows:
# held out properly, one fold's curve mapped a prediction of 0.72 to 1.0 on
# the strength of four rows, none of which hauled. Collapsing everything above
# the hundredth-largest prediction onto one point means that step is estimated
# from a hundred rows instead, and the curve cannot promise a rate nobody has
# been observed to achieve.
CAL_TOP_ROWS = 100


def _calibrated(bundle, p):
    """Apply a bundle's calibration curve to raw probabilities."""
    cal = bundle.get("calibrator")
    if cal is None:
        return np.asarray(p, dtype=float)
    hi = bundle.get("calibrator_input_max")
    q = np.asarray(p, dtype=float)
    if hi is not None:
        q = np.minimum(q, float(hi))
    return np.clip(cal.predict(q), 0.0, 1.0)


def reliability(y, p, bands=BANDS):
    """Predicted against realised, in bands. The check AUC cannot make.

    AUC is a ranking metric. A model can order every player correctly and
    still say 61% where the truth is 20%, and a number quoted to a person as
    a rate has to be a rate. The ceiling model scored AUC 0.849 and said
    exactly that.
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    out = []
    for lo, hi in zip(bands, bands[1:]):
        s = (p >= lo) & (p < hi)
        if not s.sum():
            continue
        out.append({"from": lo, "to": hi, "rows": int(s.sum()),
                    "predicted": round(float(p[s].mean()), 4),
                    "actual": round(float(y[s].mean()), 4)})
    return out


def worst_band(rel, floor=0.10, rows=30):
    """The biggest over- or under-statement among bands worth believing.

    Bands below `floor` are left out because nobody acts on a 2% forecast,
    and thin bands are left out because three rows is not a rate.
    """
    live = [b for b in rel if b["from"] >= floor and b["rows"] >= rows
            and b["predicted"] > 0]
    if not live:
        return None
    return max(live, key=lambda b: abs(b["actual"] / b["predicted"] - 1.0))


def _fit(model, df, xs, y, name, scorer, calibrate=False, report=None,
         only=None):
    """Fit, cross-score, and keep the out-of-fold probabilities.

    `only` names the models this run is for; anything else is left exactly as
    it was on disk and reports no score, so `train` can put the previous one
    back. Retraining all ten to change one of them makes the change
    unmeasurable — every projection moves and nothing says which model moved
    it.

    The folds were already predicting on held-out rows to compute the score;
    collecting those predictions costs nothing and is the only honest basis
    for both the reliability table and a calibrator. `calibrate=True` fits an
    isotonic map from predicted to realised and stores it with the model —
    isotonic because it is monotone, so it cannot disturb the ranking the
    AUC measures while it fixes the level.
    """
    if only is not None and name not in only:
        return float("nan"), 0
    d = df.dropna(subset=xs + [y])
    X, Y = d[xs].astype(float), d[y]
    cv = GroupKFold(n_splits=4)
    scores = []
    binary = Y.nunique() == 2 and hasattr(model, "predict_proba")
    oof = np.full(len(d), np.nan)
    for tr, te in cv.split(X, Y, groups=d["season"] + d["GW"].astype(str)):
        model.fit(X.iloc[tr], Y.iloc[tr])
        scores.append(scorer(Y.iloc[te], model, X.iloc[te]))
        if binary:
            oof[te] = model.predict_proba(X.iloc[te])[:, 1]
    model.fit(X, Y)
    bundle = {"model": model, "features": xs}

    if binary:
        ok = ~np.isnan(oof)
        yv, pv = Y.to_numpy()[ok].astype(float), oof[ok]
        rel = reliability(yv, pv)
        if calibrate:
            # out_of_bounds="clip" so a live prediction beyond anything the
            # folds produced lands on the end of the curve rather than
            # extrapolating off it; and the top CAL_TOP_ROWS predictions are
            # collapsed onto one input so that end is a rate and not an
            # accident. See CAL_TOP_ROWS.
            hi = float(np.sort(pv)[-min(CAL_TOP_ROWS, len(pv))])
            cal = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True,
                                     out_of_bounds="clip").fit(
                                         np.minimum(pv, hi), yv)
            bundle["calibrator"] = cal
            bundle["calibrator_input_max"] = hi
        if report is not None:
            report["rows_scored"] = int(ok.sum())
            report["brier"] = round(float(brier_score_loss(yv, pv)), 5)
            report["brier_base_rate"] = round(float(brier_score_loss(
                yv, np.full_like(pv, yv.mean()))), 5)
            report["predicted_max"] = round(float(pv.max()), 3)
            report["reliability"] = rel
            report["worst_band"] = worst_band(rel)
            if calibrate:
                cp = _calibrated(bundle, pv)
                report["calibrated"] = {
                    "brier": round(float(brier_score_loss(yv, cp)), 5),
                    "predicted_max": round(float(cp.max()), 3),
                    "input_max": round(float(bundle["calibrator_input_max"]), 3),
                    "top_rows": min(CAL_TOP_ROWS, len(pv)),
                    "reliability": reliability(yv, cp),
                    "worst_band": worst_band(reliability(yv, cp)),
                }

    os.makedirs(MODELS, exist_ok=True)
    joblib.dump(bundle, f"{MODELS}/{name}.joblib")
    return float(np.mean(scores)), len(d)


def probability(bundle, X):
    """P(class 1) from a bundle, calibrated when the bundle carries a map.

    Every probability the engine reads goes through here, so a model trained
    before calibration existed keeps working and one trained after is read
    correctly without the caller having to know which it has.
    """
    return _calibrated(bundle, bundle["model"].predict_proba(X)[:, 1])


def train(seasons=None, only=None):
    """Fit the models. `only` is a set of names to refit; the rest are left.

    A skipped model keeps the file it already has and the score it already
    had, so `models/scores.json` stays whole and a one-model retrain can be
    measured against the run before it.
    """
    only = set(only) if only else None
    print("loading history…")
    df = data.panel(data.history(seasons))
    print(f"{len(df):,} player-gameweeks")
    # Four is the ceiling FPL awards; anything above it is a data error, not a
    # class worth learning.
    df["bonus_class"] = df["next_bonus"].clip(0, 3)
    out = {}

    rep = {}
    auc, n = _fit(HistGradientBoostingClassifier(max_iter=250), df, MINUTES_X,
                  "started", "minutes",
                  lambda y, m, X: roc_auc_score(y, m.predict_proba(X)[:, 1]),
                  report=rep, only=only)
    out["minutes"] = {"metric": "AUC", "value": round(auc, 3), "rows": n, **rep}

    d = df[df["price_dir"] != 0].copy()
    d["rise"] = (d["price_dir"] > 0).astype(int)
    auc, n = _fit(HistGradientBoostingClassifier(max_iter=250), d, PRICE_X,
                  "rise", "price",
                  lambda y, m, X: roc_auc_score(y, m.predict_proba(X)[:, 1]),
                  only=only)
    out["price"] = {"metric": "AUC", "value": round(auc, 3), "rows": n}

    mae, n = _fit(HistGradientBoostingRegressor(max_iter=300), df, POINTS_X,
                  "next_points", "points",
                  lambda y, m, X: -mean_absolute_error(y, m.predict(X)),
                  only=only)
    out["points"] = {"metric": "MAE", "value": round(-mae, 3), "rows": n}

    # Captaincy is not about the average. It is about the tail: how likely is a
    # double-digit haul? Expected points and ceiling rank players differently,
    # and the Triple Captain chip only cares about the second one.
    #
    # And it is about a PROBABILITY, which AUC does not check. Measured over
    # 82,075 player-gameweeks, this model scored AUC 0.849 and was over-
    # confident by a factor of three in the only range anyone reads it in:
    # the band it predicted at 0.49 came in at 0.175, and the band it
    # predicted at 0.66 came in at 0.20. It put Bruno Fernandes at 61.5% for
    # GW8 — the highest of 668 players, five times the next — and the
    # captaincy note offered that to a manager who had asked for variance.
    # Isotonic calibration is monotone, so the ranking AUC measures survives
    # untouched while the level is corrected.
    rep = {}
    auc, n = _fit(HistGradientBoostingClassifier(max_iter=250), df, CEILING_X,
                  "hauled", "ceiling",
                  lambda y, m, X: roc_auc_score(y, m.predict_proba(X)[:, 1]),
                  calibrate=True, report=rep, only=only)
    out["ceiling"] = {"metric": "AUC", "value": round(auc, 3), "rows": n,
                      "target": "P(10+ points next gameweek)", **rep}

    auc, n = _fit(HistGradientBoostingClassifier(max_iter=250), df, BONUS_X,
                  "bonus_class", "bonus",
                  lambda y, m, X: roc_auc_score(
                      y, m.predict_proba(X), multi_class="ovr",
                      average="macro", labels=list(m.classes_)),
                  only=only)
    out["bonus"] = {"metric": "AUC (macro, 0/1/2/3)", "value": round(auc, 3),
                    "rows": n, "target": "bonus points next gameweek"}
    # The baseline that matters: predicting the player's own recent bonus rate.
    # Recorded because the model it replaces scored a flattering MAE against no
    # baseline at all, and nobody noticed for a season.
    bb = df.dropna(subset=["roll_bonus", "next_bonus"])
    if len(bb):
        out["bonus"]["naive_baseline_mae"] = round(
            float(mean_absolute_error(bb["next_bonus"], bb["roll_bonus"])), 3)
        # Only when this run actually refitted it. `_bonus_mae` reads the
        # model back off disk, and the file left there by a run that skipped
        # bonus is whatever was committed — which, it turns out, is the
        # REGRESSOR the comment above describes being replaced. It has no
        # predict_proba, so this raised, and `projections` had been quietly
        # falling back to the bps90 heuristic for the same reason.
        if out["bonus"].get("rows"):
            out["bonus"]["model_mae"] = round(float(_bonus_mae(bb)), 3)
            out["bonus"]["beats_baseline"] = (
                out["bonus"]["model_mae"] < out["bonus"]["naive_baseline_mae"])

    dd = df[df["position"].isin(["DEF", "GK", "GKP"])]
    rep = {}
    auc, n = _fit(HistGradientBoostingClassifier(max_iter=200), dd, CS_X,
                  "next_clean_sheet", "cleansheet",
                  lambda y, m, X: roc_auc_score(y, m.predict_proba(X)[:, 1]),
                  report=rep, only=only)
    out["cleansheet"] = {"metric": "AUC", "value": round(auc, 3), "rows": n,
                         **rep}

    dc = df.dropna(subset=["roll_defensive_contribution"])
    if len(dc) > 2000 and dc["next_defcon"].nunique() > 1:
        rep = {}
        auc, n = _fit(HistGradientBoostingClassifier(max_iter=200), dc, DEFCON_X,
                      "next_defcon", "defcon",
                      lambda y, m, X: roc_auc_score(y, m.predict_proba(X)[:, 1]),
                      report=rep, only=only)
        out["defcon"] = {"metric": "AUC", "value": round(auc, 3), "rows": n,
                         "target": "P(hits DefCon threshold)", **rep}

    # Herd momentum: who the crowd buys next. Drives price moves and, in a
    # mini-league, tells you which differentials are about to stop being ones.
    mae, n = _fit(HistGradientBoostingRegressor(max_iter=200), df, HERD_X,
                  "next_net_share", "herd",
                  lambda y, m, X: -mean_absolute_error(y, m.predict(X)),
                  only=only)
    out["herd"] = {"metric": "MAE", "value": round(-mae, 4), "rows": n,
                   "target": "next gameweek net transfers as share of owners"}

    # The number that matters: does the model beat "he'll score what he averaged"?
    base = df.dropna(subset=["roll_total_points", "next_points"])
    naive = mean_absolute_error(base["next_points"], base["roll_total_points"])
    out["points"]["naive_baseline_mae"] = round(float(naive), 3)
    out["points"]["beats_baseline"] = bool(-mae < naive)

    # Injury return: rows where the player blanked, predicting next minutes.
    ret = df[(df["blank"] == 1) & df["mins_before_absence"].notna()]
    if len(ret) > 3000:
        mae, n = _fit(HistGradientBoostingRegressor(max_iter=200), ret, RETURN_X,
                      "next_minutes", "injury_return",
                      lambda y, m, X: -mean_absolute_error(y, m.predict(X)),
                      only=only)
        out["injury_return"] = {"metric": "MAE (minutes)", "value": round(-mae, 1),
                                "rows": n, "target": "minutes in the week back"}

    rot = df.dropna(subset=["rest_days", "next_rest_days"])
    if len(rot) > 5000:
        mae, n = _fit(HistGradientBoostingRegressor(max_iter=200), rot, ROTATION_X,
                      "next_minutes", "rotation",
                      lambda y, m, X: -mean_absolute_error(y, m.predict(X)),
                      only=only)
        out["rotation"] = {"metric": "MAE (minutes)", "value": round(-mae, 1),
                           "rows": n, "target": "minutes under fixture congestion"}

    # A model this run skipped reported no rows. Put its previous score back
    # rather than writing a NaN over it: scores.json is the record of what
    # each model is worth, and a partial retrain must not erase the rest.
    kept = {k: v for k, v in out.items() if v.get("rows")}
    if only is not None:
        before = scores()
        out = {**before, **kept}
    else:
        out = kept
    json.dump(out, open(f"{MODELS}/scores.json", "w"), indent=1)
    for k, v in out.items():
        mark = " (refit)" if k in kept else ""
        print(f"  {k:8}{mark} {v}")
    return out


def scores():
    p = f"{MODELS}/scores.json"
    return json.load(open(p)) if os.path.exists(p) else {}


def load(name):
    p = f"{MODELS}/{name}.joblib"
    return joblib.load(p) if os.path.exists(p) else None


# What each model is supposed to be, so that a stale file cannot sit in
# models/ looking wired in. The one that prompted this: `bonus.joblib` is a
# six-feature REGRESSOR, from before the model was rewritten as a classifier
# over the four bonus classes. `projections._bonus_model` checks for
# predict_proba and returns None without it, by design — so the rewrite has
# never been in effect and the engine has been using the bps90 heuristic the
# comment above BONUS_X describes replacing. Nothing failed; it just quietly
# did not happen.
DECLARED = {
    "minutes": MINUTES_X, "price": PRICE_X, "points": POINTS_X,
    "ceiling": CEILING_X, "bonus": BONUS_X, "cleansheet": CS_X,
    "defcon": DEFCON_X, "herd": HERD_X, "injury_return": RETURN_X,
    "rotation": ROTATION_X,
}


def audit():
    """{model: what is wrong with the committed file}, empty when all match.

    Refit anything named here with `python -m gaffer.learn <name>`.
    """
    out = {}
    for name, xs in DECLARED.items():
        bundle = load(name)
        if bundle is None:
            out[name] = "no file in models/"
        elif list(bundle["features"]) != list(xs):
            out[name] = (f"trained on {list(bundle['features'])}, "
                         f"training now declares {list(xs)}")
    return out


if __name__ == "__main__":
    train()
