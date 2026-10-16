"""The learning layer.

The structural model in :mod:`fplbrain.model.points` encodes how football and
FPL scoring work. It is deliberately explicit, which makes it inspectable and
sane in gameweek 1 — but it cannot discover the things nobody thought to encode.
That is what this module is for.

Three separate jobs, which are often conflated and should not be:

**Learning.** A gradient-boosted model trained on the stored history, predicting
a player's points from lagged features. Its most important feature is the
structural model's own projection, so it learns to *correct* that projection
rather than to rebuild it from nothing.

**Blending.** The two projections are combined with a weight chosen by
out-of-sample error, not by taste. Early in a season the learned model has
little to work with and the blend leans structural; by the spring it usually
does not.

**Scoring.** A backtest that grades past projections against what actually
happened, reporting error, rank correlation and calibration. Without it, every
change to the model is a guess.

There is also a small but valuable piece of exact learning here: the Bonus
Points System is a deterministic linear function of match statistics, so
regressing observed BPS on observed stats recovers the true weights, integer for
integer, after a handful of gameweeks.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..config import ELEMENT_TYPE_TO_POSITION

log = logging.getLogger(__name__)

try:  # pragma: no cover - exercised by whichever backend is installed
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover
    HAS_LIGHTGBM = False

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURE_COLUMNS = [
    "structural_xpts",
    "p_start",
    "p_60",
    "n_fixtures",
    "price",
    "is_gkp",
    "is_def",
    "is_mid",
    "is_fwd",
    "was_home",
    "minutes_mean_3",
    "minutes_mean_5",
    "points_mean_3",
    "points_mean_5",
    "xg_mean_5",
    "xa_mean_5",
    "bps_mean_5",
    "defcon_rate_5",
    "cbit_mean_5",
    "saves_mean_5",
    "team_xg",
    "team_xgc",
    "starts_rate_5",
]

TARGET_COLUMN = "actual_points"


@dataclass
class TrainingReport:
    n_rows: int
    n_features: int
    backend: str
    mae: float
    rmse: float
    spearman: float
    baseline_mae: float
    baseline_rmse: float
    improvement: float
    folds: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        verdict = (
            "learned model improves on the structural projection"
            if self.improvement > 0
            else "structural projection is still ahead; keeping the blend conservative"
        )
        return (
            f"{self.n_rows} rows, {self.backend}: MAE {self.mae:.3f} vs "
            f"{self.baseline_mae:.3f} baseline ({self.improvement:+.1%}) — {verdict}"
        )


def build_training_frame(history_rows: Sequence[dict], positions: dict[int, str]) -> pd.DataFrame:
    """Build lagged features and targets from stored gameweek history.

    Every feature is computed from gameweeks strictly *before* the one being
    predicted. That is the whole game: a feature that peeks at the target
    produces a model that looks brilliant in backtest and is useless in
    practice.
    """
    if not history_rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS + [TARGET_COLUMN])

    frame = pd.DataFrame(history_rows)
    if frame.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS + [TARGET_COLUMN])

    frame = frame.sort_values(["element", "gameweek"]).reset_index(drop=True)
    frame["position"] = frame["element"].map(positions).fillna("MID")

    grouped = frame.groupby("element", sort=False)

    def rolling(column: str, window: int, name: str) -> None:
        if column not in frame.columns:
            frame[name] = np.nan
            return
        frame[name] = (
            grouped[column]
            .apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

    rolling("minutes", 3, "minutes_mean_3")
    rolling("minutes", 5, "minutes_mean_5")
    rolling("total_points", 3, "points_mean_3")
    rolling("total_points", 5, "points_mean_5")
    rolling("expected_goals", 5, "xg_mean_5")
    rolling("expected_assists", 5, "xa_mean_5")
    rolling("bps", 5, "bps_mean_5")
    rolling("clearances_blocks_interceptions", 5, "cbit_mean_5")
    rolling("saves", 5, "saves_mean_5")
    rolling("expected_goals_conceded", 5, "team_xgc")

    if "defensive_contribution" in frame.columns:
        frame["_defcon_hit"] = (frame["defensive_contribution"].fillna(0) > 0).astype(float)
        rolling("_defcon_hit", 5, "defcon_rate_5")
    else:
        frame["defcon_rate_5"] = np.nan

    if "starts" in frame.columns:
        frame["_started"] = (frame["starts"].fillna(0) > 0).astype(float)
        rolling("_started", 5, "starts_rate_5")
    else:
        frame["starts_rate_5"] = np.nan

    # Team attacking context: the player's own side's expected goals, taken from
    # team-mates in the same fixture rather than from the player himself.
    if {"fixture", "expected_goals"}.issubset(frame.columns):
        team_xg = (
            frame.groupby(["gameweek", "fixture"])["expected_goals"].transform("sum")
        )
        frame["team_xg"] = (
            frame.groupby("element")["_team_xg_raw"].shift(1)
            if "_team_xg_raw" in frame.columns
            else team_xg.shift(1)
        )
    else:
        frame["team_xg"] = np.nan

    frame["p_start"] = frame["starts_rate_5"]
    frame["p_60"] = (frame["minutes_mean_5"] / 90.0).clip(0, 1)
    frame["n_fixtures"] = 1
    frame["price"] = frame.get("value", pd.Series(50, index=frame.index)) / 10.0
    frame["was_home"] = frame.get("was_home", pd.Series(0, index=frame.index)).fillna(0)

    for position in ("GKP", "DEF", "MID", "FWD"):
        frame[f"is_{position.lower()}"] = (frame["position"] == position).astype(int)

    # Without a stored snapshot, the structural projection is approximated by a
    # simple recency estimate. Where real snapshots exist they are joined in by
    # `attach_snapshots` and overwrite this column.
    frame["structural_xpts"] = frame["points_mean_5"].fillna(2.0)
    frame[TARGET_COLUMN] = frame["total_points"]

    for column in FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan

    return frame[["element", "gameweek", "position"] + FEATURE_COLUMNS + [TARGET_COLUMN]]


def attach_snapshots(frame: pd.DataFrame, snapshots: Sequence[dict]) -> pd.DataFrame:
    """Overwrite the approximate structural feature with real stored snapshots."""
    if frame.empty or not snapshots:
        return frame
    snapshot_frame = pd.DataFrame(snapshots)
    if snapshot_frame.empty:
        return frame
    snapshot_frame = snapshot_frame.rename(
        columns={"for_gameweek": "gameweek", "expected_points": "snapshot_xpts"}
    )
    columns = ["element", "gameweek", "snapshot_xpts"]
    for optional in ("p_start", "p_60", "n_fixtures"):
        if optional in snapshot_frame.columns:
            columns.append(optional)
            snapshot_frame = snapshot_frame.rename(columns={optional: f"snap_{optional}"})
            columns[-1] = f"snap_{optional}"

    merged = frame.merge(snapshot_frame[columns], on=["element", "gameweek"], how="left")
    merged["structural_xpts"] = merged["snapshot_xpts"].fillna(merged["structural_xpts"])
    for optional in ("p_start", "p_60", "n_fixtures"):
        snap_column = f"snap_{optional}"
        if snap_column in merged.columns:
            merged[optional] = merged[snap_column].fillna(merged[optional])
            merged = merged.drop(columns=[snap_column])
    return merged.drop(columns=["snapshot_xpts"])


class PointsLearner:
    """Gradient-boosted correction to the structural projection."""

    def __init__(self, *, backend: str = "auto", random_state: int = 20262027):
        self.backend = backend
        self.random_state = random_state
        self.model: Any = None
        self.feature_names: list[str] = list(FEATURE_COLUMNS)
        self.report: TrainingReport | None = None

    def _new_model(self):
        if self.backend == "ridge":
            return Ridge(alpha=1.0)
        if HAS_LIGHTGBM and self.backend in ("auto", "lightgbm"):
            return lgb.LGBMRegressor(
                n_estimators=400,
                learning_rate=0.04,
                num_leaves=31,
                min_child_samples=40,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                random_state=self.random_state,
                verbose=-1,
            )
        return HistGradientBoostingRegressor(
            max_iter=350,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=self.random_state,
        )

    @property
    def backend_name(self) -> str:
        if self.model is None:
            return "untrained"
        return type(self.model).__name__

    def fit(self, frame: pd.DataFrame, *, min_rows: int = 400) -> TrainingReport:
        """Train on the frame, grading itself with forward-chaining folds.

        Splitting by gameweek rather than at random is essential: a random split
        lets the model see the future, and the reported error becomes fiction.
        """
        frame = frame.dropna(subset=[TARGET_COLUMN])
        if len(frame) < min_rows:
            raise ValueError(
                f"only {len(frame)} training rows; need at least {min_rows}. "
                "Run `fplbrain pull --history` after a few gameweeks have been played."
            )

        gameweeks = sorted(frame["gameweek"].unique())
        folds: list[dict] = []
        predictions: list[np.ndarray] = []
        actuals: list[np.ndarray] = []
        baselines: list[np.ndarray] = []

        # Forward chaining: train on everything before a cut, test on the cut.
        cut_points = gameweeks[max(3, len(gameweeks) // 2) :]
        for cut in cut_points:
            train = frame[frame["gameweek"] < cut]
            test = frame[frame["gameweek"] == cut]
            if len(train) < min_rows // 2 or test.empty:
                continue
            model = self._new_model()
            model.fit(train[self.feature_names].fillna(0.0), train[TARGET_COLUMN])
            predicted = model.predict(test[self.feature_names].fillna(0.0))
            predictions.append(predicted)
            actuals.append(test[TARGET_COLUMN].to_numpy())
            baselines.append(test["structural_xpts"].fillna(0.0).to_numpy())
            folds.append(
                {
                    "gameweek": int(cut),
                    "n": int(len(test)),
                    "mae": float(mean_absolute_error(test[TARGET_COLUMN], predicted)),
                }
            )

        if predictions:
            predicted = np.concatenate(predictions)
            actual = np.concatenate(actuals)
            baseline = np.concatenate(baselines)
            mae = float(mean_absolute_error(actual, predicted))
            rmse = float(math.sqrt(mean_squared_error(actual, predicted)))
            baseline_mae = float(mean_absolute_error(actual, baseline))
            baseline_rmse = float(math.sqrt(mean_squared_error(actual, baseline)))
            spearman = _spearman(predicted, actual)
            improvement = (baseline_mae - mae) / baseline_mae if baseline_mae else 0.0
        else:
            mae = rmse = baseline_mae = baseline_rmse = float("nan")
            spearman = float("nan")
            improvement = 0.0

        # Final model is trained on everything.
        self.model = self._new_model()
        self.model.fit(frame[self.feature_names].fillna(0.0), frame[TARGET_COLUMN])

        self.report = TrainingReport(
            n_rows=len(frame),
            n_features=len(self.feature_names),
            backend=self.backend_name,
            mae=mae,
            rmse=rmse,
            spearman=spearman,
            baseline_mae=baseline_mae,
            baseline_rmse=baseline_rmse,
            improvement=improvement,
            folds=folds,
        )
        return self.report

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model is not trained")
        return np.asarray(self.model.predict(frame[self.feature_names].fillna(0.0)))

    def feature_importance(self) -> list[tuple[str, float]]:
        if self.model is None:
            return []
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            return []
        pairs = list(zip(self.feature_names, [float(x) for x in importances]))
        return sorted(pairs, key=lambda kv: -kv[1])

    def save(self, path: Path) -> None:
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {"model": self.model, "features": self.feature_names, "report": self.report}, fh
            )

    @classmethod
    def load(cls, path: Path) -> "PointsLearner":
        import pickle

        with Path(path).open("rb") as fh:
            blob = pickle.load(fh)
        learner = cls()
        learner.model = blob["model"]
        learner.feature_names = blob["features"]
        learner.report = blob.get("report")
        return learner


def choose_blend_weight(report: TrainingReport | None, *, cap: float = 0.6) -> float:
    """How much to trust the learned model, from its own out-of-sample error.

    The weight is the share of error reduction the learned model actually
    achieved, capped well below 1. The cap is not timidity: the structural model
    handles fixtures, blanks and doubles explicitly, and a learned model trained
    on a single season will not have seen enough of those to be trusted alone.
    """
    if report is None or not np.isfinite(report.mae) or not np.isfinite(report.baseline_mae):
        return 0.0
    if report.baseline_mae <= 0:
        return 0.0
    improvement = (report.baseline_mae - report.mae) / report.baseline_mae
    if improvement <= 0:
        return 0.0
    return float(min(cap, improvement * 2.0))


def blend_projections(
    structural: dict[int, float], learned: dict[int, float], weight: float
) -> dict[int, float]:
    """Combine the two projections element by element."""
    weight = float(np.clip(weight, 0.0, 1.0))
    out = {}
    for element, value in structural.items():
        if element in learned:
            out[element] = (1 - weight) * value + weight * learned[element]
        else:
            out[element] = value
    return out


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    gameweeks: list[int]
    n: int
    mae: float
    rmse: float
    spearman: float
    top10_hit_rate: float
    captain_hit_rate: float
    calibration: list[dict] = field(default_factory=list)
    per_gameweek: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.n} player-gameweeks over GW{min(self.gameweeks)}-{max(self.gameweeks)}: "
            f"MAE {self.mae:.2f}, Spearman {self.spearman:.3f}, "
            f"top-10 hit rate {self.top10_hit_rate:.0%}, "
            f"captain pick in match top scorer {self.captain_hit_rate:.0%}"
        )


def backtest_snapshots(store, gameweeks: Sequence[int] | None = None) -> BacktestResult:
    """Grade stored projections against what actually happened.

    This is the honest scorecard: it only uses projections that were written
    *before* the gameweek was played.
    """
    history = pd.DataFrame(store.history())
    if history.empty:
        raise ValueError("no results stored yet; run `fplbrain pull --history` first")

    available = sorted(history["gameweek"].unique())
    gameweeks = [gw for gw in (gameweeks or available) if gw in available]

    frames = []
    for gameweek in gameweeks:
        snapshots = store.latest_snapshots(gameweek)
        if not snapshots:
            continue
        snapshot_frame = pd.DataFrame(snapshots).rename(columns={"for_gameweek": "gameweek"})
        actual = (
            history[history["gameweek"] == gameweek]
            .groupby("element", as_index=False)["total_points"]
            .sum()
        )
        merged = snapshot_frame.merge(actual, on="element", how="inner")
        if not merged.empty:
            frames.append(merged)

    if not frames:
        raise ValueError(
            "no projection snapshots to score. Snapshots are written by "
            "`fplbrain advise` and `fplbrain project`; they accumulate as the season runs."
        )

    merged = pd.concat(frames, ignore_index=True)
    predicted = merged["expected_points"].to_numpy(dtype=float)
    actual = merged["total_points"].to_numpy(dtype=float)

    per_gameweek = []
    top10_hits = []
    captain_hits = []
    for gameweek, group in merged.groupby("gameweek"):
        group_predicted = group["expected_points"].to_numpy(dtype=float)
        group_actual = group["total_points"].to_numpy(dtype=float)
        per_gameweek.append(
            {
                "gameweek": int(gameweek),
                "n": int(len(group)),
                "mae": float(mean_absolute_error(group_actual, group_predicted)),
                "spearman": _spearman(group_predicted, group_actual),
            }
        )
        # Did the ten highest projections actually finish in the real top fifty?
        top_predicted = set(group.nlargest(10, "expected_points")["element"])
        top_actual = set(group.nlargest(50, "total_points")["element"])
        if top_predicted:
            top10_hits.append(len(top_predicted & top_actual) / len(top_predicted))
        # Would the top projection have been among the gameweek's best returns?
        best_predicted = group.nlargest(1, "expected_points")
        if not best_predicted.empty:
            threshold = group["total_points"].quantile(0.98)
            captain_hits.append(
                float(best_predicted["total_points"].iloc[0] >= max(threshold, 6))
            )

    calibration = _calibration_table(predicted, actual)

    return BacktestResult(
        gameweeks=sorted(int(gw) for gw in merged["gameweek"].unique()),
        n=int(len(merged)),
        mae=float(mean_absolute_error(actual, predicted)),
        rmse=float(math.sqrt(mean_squared_error(actual, predicted))),
        spearman=_spearman(predicted, actual),
        top10_hit_rate=float(np.mean(top10_hits)) if top10_hits else float("nan"),
        captain_hit_rate=float(np.mean(captain_hits)) if captain_hits else float("nan"),
        calibration=calibration,
        per_gameweek=per_gameweek,
    )


def _calibration_table(predicted: np.ndarray, actual: np.ndarray, bins: int = 10) -> list[dict]:
    """Are players projected for 6 points actually scoring 6 on average?"""
    if len(predicted) == 0:
        return []
    order = np.argsort(predicted)
    predicted = predicted[order]
    actual = actual[order]
    chunks = np.array_split(np.arange(len(predicted)), bins)
    rows = []
    for index, chunk in enumerate(chunks):
        if chunk.size == 0:
            continue
        rows.append(
            {
                "bin": index + 1,
                "n": int(chunk.size),
                "predicted_mean": round(float(predicted[chunk].mean()), 3),
                "actual_mean": round(float(actual[chunk].mean()), 3),
                "bias": round(float(actual[chunk].mean() - predicted[chunk].mean()), 3),
            }
        )
    return rows


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    rank_a = pd.Series(a).rank().to_numpy()
    rank_b = pd.Series(b).rank().to_numpy()
    if rank_a.std() == 0 or rank_b.std() == 0:
        return float("nan")
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


# ---------------------------------------------------------------------------
# BPS calibration
# ---------------------------------------------------------------------------

BPS_REGRESSORS = {
    "played_60_plus": lambda row: 1.0 if row.get("minutes", 0) >= 60 else 0.0,
    "played_1_to_60": lambda row: 1.0 if 0 < row.get("minutes", 0) < 60 else 0.0,
    "goals": lambda row: float(row.get("goals", 0) or 0),
    "assists": lambda row: float(row.get("assists", 0) or 0),
    "clean_sheet": lambda row: float(row.get("clean_sheet", 0) or 0),
    "goals_conceded": lambda row: float(row.get("goals_conceded", 0) or 0),
    "saves": lambda row: float(row.get("saves", 0) or 0),
    "cbi": lambda row: float(row.get("clearances_blocks_interceptions", 0) or 0),
    "recoveries": lambda row: float(row.get("recoveries", 0) or 0),
    "tackles": lambda row: float(row.get("tackles", 0) or 0),
    "yellow_cards": lambda row: float(row.get("yellow_cards", 0) or 0),
    "red_cards": lambda row: float(row.get("red_cards", 0) or 0),
    "own_goals": lambda row: float(row.get("own_goals", 0) or 0),
    "penalties_saved": lambda row: float(row.get("penalties_saved", 0) or 0),
    "penalties_missed": lambda row: float(row.get("penalties_missed", 0) or 0),
    "defensive_contribution": lambda row: 1.0
    if float(row.get("defensive_contribution", 0) or 0) > 0
    else 0.0,
}


def calibrate_bps(
    history_rows: Sequence[dict], positions: dict[int, str], *, min_rows: int = 300
) -> dict[str, Any]:
    """Recover the true BPS weights by regressing BPS on match statistics.

    BPS is a deterministic linear function of the underlying stats, so given
    enough observations an ordinary least squares fit reproduces the official
    table almost exactly. Coefficients are fitted per position because several
    weights (goals, clean sheets, goals conceded) differ by position.

    This removes the engine's dependence on published BPS values being correct
    or current — after a few gameweeks it simply learns them.
    """
    rows = [r for r in history_rows if (r.get("minutes") or 0) > 0]
    if len(rows) < min_rows:
        raise ValueError(
            f"only {len(rows)} rows with minutes; need {min_rows} for a stable BPS fit"
        )

    out: dict[str, Any] = {"positions": {}, "n_rows": len(rows)}
    frame = pd.DataFrame(rows)
    frame["position"] = frame["element"].map(positions).fillna("MID")

    for position, group in frame.groupby("position"):
        if len(group) < 60:
            continue
        design = pd.DataFrame(
            {name: group.apply(fn, axis=1) for name, fn in BPS_REGRESSORS.items()}
        )
        # Drop constant columns, which are not identifiable.
        design = design.loc[:, design.std() > 0]
        if design.empty:
            continue
        target = group["bps"].astype(float)

        model = Ridge(alpha=0.01, fit_intercept=True)
        model.fit(design, target)
        predicted = model.predict(design)
        residual = float(np.abs(predicted - target).mean())

        weights = {
            name: round(float(coefficient), 2)
            for name, coefficient in zip(design.columns, model.coef_)
        }
        out["positions"][str(position)] = {
            "weights": weights,
            "intercept": round(float(model.intercept_), 2),
            "mean_abs_error": round(residual, 3),
            "n": int(len(group)),
            "confident": residual < 1.0,
        }
    return out


def write_calibrated_bps(calibration: dict[str, Any], path: Path) -> Path:
    """Persist calibrated BPS weights next to the config for the engine to pick up."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(calibration, fh, indent=2)
    return path


def positions_from_bootstrap(bootstrap) -> dict[int, str]:
    return {
        int(row["id"]): ELEMENT_TYPE_TO_POSITION.get(int(row.get("element_type", 3)), "MID")
        for row in bootstrap.raw.get("elements", [])
    }
