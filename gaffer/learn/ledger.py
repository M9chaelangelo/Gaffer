"""The prediction ledger: what we said would happen, and what did.

This is the thing the engine has never had. Ten trained models, a MILP, a
chip sweep — and no record anywhere of whether any of it was right. In GW6
the projected eleven was 68.8 and the actual return was 30, and nothing in
the system noticed, recorded it, or learned from it. A model that is never
scored cannot improve, and neither can the person reading it.

Two append-only files, both plain JSON Lines so they survive any schema
change and can be read by anything:

  predictions.jsonl  one row per player per gameweek per run, written at
                     solve time, before the gameweek is played.
  outcomes.jsonl     one row per player per gameweek, written afterwards
                     from the game's own result feed.

They are joined on (game, gw, player_id). Nothing is ever rewritten: a
second solve for the same gameweek appends another row with a later
`solved_at`, because changing your mind is itself data — and silently
overwriting the earlier prediction would let the engine mark its own
homework after seeing the answer.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone

PREDICTIONS = "predictions.jsonl"
OUTCOMES = "outcomes.jsonl"


def _append(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":"), default=str) + "\n")
    return len(rows)


def _read(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # One corrupt line must not cost the whole history.
                continue
    return out


def record(dirpath, game, gw, players, meta=None):
    """Write what we expect, before the gameweek is played.

    `players` is an iterable of dicts carrying at least `id` and `ep`.
    """
    now = datetime.now(timezone.utc).isoformat()
    meta = meta or {}
    rows = []
    for p in players:
        if p.get("ep") is None:
            continue
        rows.append({
            "game": game, "gw": int(gw), "player_id": int(p["id"]),
            "name": p.get("name"), "pos": p.get("pos"), "team": p.get("team"),
            "price": p.get("price"), "ep": round(float(p["ep"]), 3),
            "start_prob": p.get("start_prob"),
            "in_squad": bool(p.get("in_squad")),
            "role": p.get("role"), "captain": bool(p.get("captain")),
            "solved_at": now, **meta,
        })
    return _append(os.path.join(dirpath, PREDICTIONS), rows)


def settle(dirpath, game, gw, actuals):
    """Write what happened. `actuals` maps player id to points scored."""
    now = datetime.now(timezone.utc).isoformat()
    rows = [{"game": game, "gw": int(gw), "player_id": int(pid),
             "points": int(pts), "settled_at": now}
            for pid, pts in actuals.items()]
    return _append(os.path.join(dirpath, OUTCOMES), rows)


def latest_predictions(dirpath, game=None):
    """The most recent prediction per (game, gw, player).

    A gameweek may be solved several times — a re-run, a corrected config.
    Only the last one before kickoff is the engine's actual answer.
    """
    best = {}
    for r in _read(os.path.join(dirpath, PREDICTIONS)):
        if game and r.get("game") != game:
            continue
        k = (r["game"], r["gw"], r["player_id"])
        if k not in best or str(r.get("solved_at")) > str(best[k].get("solved_at")):
            best[k] = r
    return best


def joined(dirpath, game=None):
    """Predictions with their outcomes attached, for settled gameweeks only."""
    preds = latest_predictions(dirpath, game)
    out = {}
    for r in _read(os.path.join(dirpath, OUTCOMES)):
        if game and r.get("game") != game:
            continue
        out[(r["game"], r["gw"], r["player_id"])] = r["points"]
    rows = []
    for k, p in preds.items():
        if k in out:
            q = dict(p)
            q["actual"] = out[k]
            q["error"] = round(out[k] - p["ep"], 3)
            rows.append(q)
    return rows


def score(dirpath, game=None, min_ep=0.0):
    """How good the projections have actually been.

    Bias is the one to watch. A model that is merely noisy is tolerable; a
    model that is consistently high has an edge-destroying flaw somewhere,
    and on 16 Oct 2026 there was a candidate in plain sight — `form_weight`
    blends recent POINTS, so a player over-performing his expected goals is
    rewarded for the luck rather than marked down for it.
    """
    rows = [r for r in joined(dirpath, game) if r["ep"] >= min_ep]
    if not rows:
        return {"n": 0, "note": "nothing settled yet"}
    n = len(rows)
    errs = [r["error"] for r in rows]
    mae = sum(abs(e) for e in errs) / n
    bias = sum(errs) / n
    rmse = (sum(e * e for e in errs) / n) ** 0.5
    # Against the only honest baseline: predicting each player's own mean.
    mean_actual = sum(r["actual"] for r in rows) / n
    naive = sum(abs(r["actual"] - mean_actual) for r in rows) / n

    by_gw = defaultdict(list)
    for r in rows:
        by_gw[(r["game"], r["gw"])].append(r)
    weeks = []
    for (g, gw), rs in sorted(by_gw.items()):
        xi = [r for r in rs if r.get("role") == "xi"]
        weeks.append({
            "game": g, "gw": gw, "n": len(rs),
            "mae": round(sum(abs(r["error"]) for r in rs) / len(rs), 3),
            "bias": round(sum(r["error"] for r in rs) / len(rs), 3),
            "xi_projected": round(sum(r["ep"] for r in xi), 1) if xi else None,
            "xi_actual": sum(r["actual"] for r in xi) if xi else None,
        })
    return {
        "n": n, "mae": round(mae, 3), "bias": round(bias, 3),
        "rmse": round(rmse, 3), "naive_mae": round(naive, 3),
        "skill": round(1 - mae / naive, 3) if naive else None,
        "weeks": weeks,
    }
