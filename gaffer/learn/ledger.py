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
from collections import Counter, defaultdict
from datetime import datetime, timezone

PREDICTIONS = "predictions.jsonl"
OUTCOMES = "outcomes.jsonl"
PRICES = "prices.jsonl"


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


def settle(dirpath, game, gw, actuals, final=True):
    """Write what happened. `actuals` maps player id to points scored.

    `final` says whether the gameweek was actually over. It exists because
    it was not always asked: settlement used to fire for any gameweek
    numerically earlier than the one being solved, so a Sunday-morning run
    graded a gameweek whose Sunday fixtures had not kicked off. GW7 2026/27
    was scored at 48 against a true 67, and since the gameweek then counted
    as settled the wrong number was permanent — in the one file whose whole
    job is to say honestly how good the model is.

    Rows are append-only and `joined` takes the last one per player, so a
    premature settlement is corrected by settling again rather than edited.
    """
    now = datetime.now(timezone.utc).isoformat()
    rows = [{"game": game, "gw": int(gw), "player_id": int(pid),
             "points": int(pts), "settled_at": now, "final": bool(final)}
            for pid, pts in actuals.items()]
    return _append(os.path.join(dirpath, OUTCOMES), rows)


def settled_finally(dirpath, game=None):
    """Gameweeks whose outcomes were written after the gameweek ended.

    Anything else is provisional and must be settled again. Rows written
    before `final` existed have no such key and count as provisional,
    which is the safe reading: they were written by the code that did not
    check.
    """
    state = {}
    for r in _read(os.path.join(dirpath, OUTCOMES)):
        if game and r.get("game") != game:
            continue
        gw = r["gw"]
        # Last write per gameweek decides, matching how `joined` reads points.
        state[gw] = bool(r.get("final", False))
    return {gw for gw, ok in state.items() if ok}


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


# --- price forecasts ---------------------------------------------------------
# The engine predicts tonight's price moves from net transfers, calibrated
# against the players who actually rose. Until now it printed that forecast
# and forgot it, so nobody could say whether it was any good — the same blind
# spot the projection ledger was built to close.
#
# No extra fetch is needed to settle these. Each run records every player's
# price alongside the forecast; the NEXT run's price is the answer to the
# previous run's question, so the file scores itself.

def record_prices(dirpath, players, game="fpl"):
    """Snapshot every player's price and the forecast made against it."""
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for p in players:
        if p.get("price") is None:
            continue
        rows.append({"game": game, "at": now, "player_id": int(p["id"]),
                     "name": p.get("name"), "price": round(float(p["price"]), 1),
                     "direction": p.get("price_direction"),
                     "net": p.get("net_transfers")})
    return _append(os.path.join(dirpath, PRICES), rows)


def score_prices(dirpath, game="fpl"):
    """Was the forecast right? Scored against the next snapshot's price.

    A forecast is only counted once a LATER snapshot exists for that player,
    so the most recent run is never marked against itself.
    """
    rows = [r for r in _read(os.path.join(dirpath, PRICES))
            if not game or r.get("game") == game]
    by_player = defaultdict(list)
    for r in rows:
        by_player[r["player_id"]].append(r)

    counts = defaultdict(lambda: Counter())
    moves = []
    for pid, rs in by_player.items():
        rs.sort(key=lambda r: str(r.get("at")))
        for a, b in zip(rs, rs[1:]):
            d = round(b["price"] - a["price"], 1)
            actual = "rise" if d > 0 else ("fall" if d < 0 else "hold")
            said = a.get("direction") or "hold"
            counts[said][actual] += 1
            if actual != "hold":
                moves.append({"name": a.get("name"), "said": said,
                              "actual": actual, "delta": d,
                              "net": a.get("net")})

    total = sum(sum(c.values()) for c in counts.values())
    if not total:
        return {"n": 0, "note": "need two snapshots before a forecast can be scored"}
    hits = sum(counts[k][k] for k in ("rise", "fall", "hold"))
    out = {"n": total, "accuracy": round(hits / total, 3), "matrix": {},
           "moves": moves[-40:]}
    for said in ("rise", "fall", "hold"):
        c = counts.get(said) or Counter()
        n = sum(c.values())
        out["matrix"][said] = {
            "predicted": n,
            "correct": c[said],
            # Precision on the calls that matter: a "hold" that holds is easy
            # and uninformative, a called rise that rises is the whole point.
            "precision": round(c[said] / n, 3) if n else None,
            "breakdown": dict(c),
        }
    return out
