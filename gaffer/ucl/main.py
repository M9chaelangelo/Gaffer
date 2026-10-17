"""Produce a UCL matchday plan: fetch, project, optimise, report.

Runs where there is network — the assistant's own container is denied
gaming.uefa.com, so this is driven by `.github/workflows/ucl.yml` and commits
its output, the same arrangement the FPL solve uses.

    python -m gaffer.ucl.main --matchday 3
    python -m gaffer.ucl.main --matchday 3 --offline      # committed snapshot
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from datetime import datetime, timezone

import yaml

from . import fetch, plan, project

SNAPSHOT = "gaffer/ucl/snapshot.json"
CONFIG = "config/ucl.yaml"
OUT = "docs/ucl.json"
TEXT = "docs/ucl.txt"


def load_config(path=CONFIG):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_snapshot(matchday, offline):
    """Live feed when we can reach it, committed snapshot when we cannot."""
    if not offline:
        try:
            raw = fetch.raw_players(matchday)
            problems = fetch.verify(raw)
            rows = fetch.players(rows=raw)
            fx = fetch.fixtures(rows, matchday)
            if rows and fx:
                return rows, fx, problems, "live"
            problems = problems + ["the feed carried no fixtures for this "
                                   f"matchday ({len(rows)} players)"]
        except Exception as e:                  # noqa: BLE001 - reported, not hidden
            problems = [f"the live feed failed: {e.__class__.__name__}: {e}"]
    else:
        problems = []
    if not os.path.exists(SNAPSHOT):
        # Say WHY. The first version printed only this path, so six minutes
        # of timeouts arrived as one sentence that named no cause.
        raise SystemExit(
            f"no live feed and no committed snapshot at {SNAPSHOT}\n  "
            + "\n  ".join(problems or ["(no reason was recorded)"]))
    with open(SNAPSHOT) as f:
        snap = json.load(f)
    return (snap["players"], snap["fixtures"],
            problems + list(snap.get("problems") or []), "snapshot")


def write_snapshot(rows, fx, matchday, problems):
    keep = [p for p in rows if p["minutes"] > 0 or p["price"] >= 6.0]
    with open(SNAPSHOT, "w") as f:
        json.dump({"matchday": matchday, "fixtures": fx, "players": keep,
                   "problems": problems, "total_players": len(rows),
                   "headers": fetch.LAST_HEADERS,
                   "fetched": datetime.now(timezone.utc).isoformat()},
                  f, ensure_ascii=False, separators=(",", ":"))
    return len(keep)


def eligible(rows, fixtures):
    """Players whose club actually plays this matchday.

    Buying a player with no fixture is never right in the league phase and
    letting the MILP consider them only widens the search.
    """
    teams = set()
    for f in fixtures:
        teams.add(f["home"])
        teams.add(f["away"])
    return [r for r in rows if r["team"] in teams]


def run(matchday, cfg, offline=False, n=project.N, time_limit=180):
    rows, fx, problems, source = load_snapshot(matchday, offline)
    if source == "live":
        write_snapshot(rows, fx, matchday, problems)

    pool = eligible(rows, fx)

    # Known-out players, named by hand. The feed's own availability prose
    # lags a suspension by days and has never once been ahead of it.
    forced_names = list(cfg.get("out") or [])
    forced = set(owned_ids_of(pool, forced_names))
    for r in pool:
        if r["id"] in forced:
            r["forced_out"] = True
    unmatched_out = [n for n in forced_names if _match(pool, n) is None]

    samples, fxrates = project.simulate(pool, fx, odds=cfg.get("odds"), n=n)
    projected = project.summarise(pool, samples)
    columns = {r["id"]: j for j, r in enumerate(pool)}

    squad_names = cfg.get("squad") or []
    owned = resolve(projected, squad_names)
    missing = [s for s in squad_names if s not in owned["named"]]

    result = plan.optimise(
        projected,
        owned_ids=owned["ids"],
        bank=float(cfg.get("bank") or 0.0),
        free_transfers=cfg.get("free_transfers"),
        max_extra_transfers=int(cfg.get("max_extra_transfers") or 0),
        locked=owned_ids_of(projected, cfg.get("locked") or []),
        banned=owned_ids_of(projected, cfg.get("banned") or []),
        time_limit=time_limit,
    )
    band = plan.armband(result["xi"], samples, columns, fx)
    return {
        "matchday": matchday,
        "generated": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "problems": problems,
        "missing_from_squad": missing,
        "unmatched_out": unmatched_out,
        "forced_out": sorted(forced),
        "fixtures": fxrates,
        "plan": result,
        "armband": band,
        "projected": sorted(projected, key=lambda r: -r["ep"]),
    }


def resolve(rows, names):
    """Match configured squad names to feed ids, reporting what did not match."""
    ids, named = [], []
    for name in names:
        hit = _match(rows, name)
        if hit:
            ids.append(hit["id"])
            named.append(name)
    return {"ids": ids, "named": named}


def owned_ids_of(rows, names):
    return [h["id"] for h in (_match(rows, n) for n in names) if h]


def _norm(s):
    """Fold a name to something two spellings of it agree on.

    UEFA writes "B. Fernandes"; a person writing a config writes
    "B.Fernandes" or "Bruno Fernandes". Accents, initials' full stops and
    spacing are all places those two drift apart and none of them carry
    meaning here.
    """
    s = unicodedata.normalize("NFKD", str(s).lower())
    return "".join(c for c in s if c.isalnum())


def _match(rows, name):
    """A feed row for a configured name. Id, then whole name, then substring."""
    key = _norm(name)
    for r in rows:
        if r["id"] == str(name).strip():
            return r
    for r in rows:
        if _norm(r["name"]) == key or _norm(r["full_name"]) == key:
            return r
    hits = [r for r in rows
            if key and (key in _norm(r["name"]) or key in _norm(r["full_name"]))]
    # An ambiguous surname is not a match. Picking the dearest "Silva" is how
    # a plan quietly becomes a plan for somebody else's player.
    return hits[0] if len(hits) == 1 else None


def report(out, write=None):
    """Print the plan. `write` also collects it, for a committed copy.

    The FPL side lost a correct computation to a presentation layer four
    separate times. The text a person actually reads is the product, so it
    is produced once, here, and the file and the log are the same words.
    """
    lines = []

    def w(s=""):
        lines.append(s)
        print(s)

    p = out["plan"]
    w(f"\nUCL matchday {out['matchday']}  —  {out['source']} data, "
      f"{out['generated'][:16]}Z")
    if out["problems"]:
        w("\n!! the feed is not what this was written against:")
        for x in out["problems"]:
            w("   - " + x)
    if out["missing_from_squad"]:
        w("\n!! configured squad members not matched in the feed: "
          + ", ".join(out["missing_from_squad"]))
    if out.get("unmatched_out"):
        w("!! players you marked out were not matched in the feed: "
          + ", ".join(out["unmatched_out"]))

    w("\nfixtures (kick-off order — the armband can move between days)")
    for f in out["fixtures"]:
        w(f"  {(f['kickoff'] or '')[:16]}  {f['home']:>22} {f['lh']:.2f}"
          f" - {f['la']:.2f} {f['away']:<22} [{f['source']}]")

    proof = "proved optimal" if p.get("optimal") else "NOT PROVED OPTIMAL"
    w(f"\nsolver: {p['status']} ({proof})   "
      f"spend {p['spend']:.1f} of {p['budget']:.1f}")
    if p.get("draft"):
        w("\nno squad configured — this is a draft of the best fifteen, "
          "not a transfer plan")
    elif p["out"] or p["in"]:
        w("\ntransfers")
        for a, b in zip(p["out"], p["in"]):
            w(f"  OUT {a['name']:<16} {a['pos']} {a['price']:>4.1f} "
              f"{a['ep']:>5.2f}   ->   IN {b['name']:<16} {b['pos']} "
              f"{b['price']:>4.1f} {b['ep']:>5.2f}")
        if p["hits"]:
            w(f"  ({p['hits']} extra transfer(s), -{4 * p['hits']} points)")
    else:
        w("\ntransfers: none — the squad is already the best use of the budget")

    w("\nstarting eleven")
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    for r in sorted(p["xi"], key=lambda r: (order[r["pos"]], -r["ep"])):
        w(f"  {r['pos']:<4} {r['name']:<18} {r['team']:<18} {r['price']:>4.1f}"
          f"  ep {r['ep']:>5.2f}  blank {r['p_blank']:>5.1%}  haul {r['p_haul']:>5.1%}")
    w("bench (auto-sub order)")
    for i, r in enumerate(sorted(p["bench"], key=lambda r: -r["ep"]), 1):
        w(f"  {i}. {r['pos']:<4} {r['name']:<18} {r['team']:<18} ep {r['ep']:>5.2f}")

    if out["armband"]:
        w("\narmband — captain, and the move you keep in hand")
        for a in out["armband"][:6]:
            tag = (f"  switch to {a['fallback']['name']} if he blanks"
                   if a["fallback"] else "  (last day — no move available)")
            w(f"  {a['player']['name']:<18} {a['day']}  plain {a['captain_ep']:>5.2f}"
              f"  with the move {a['armband_ep']:>5.2f}"
              f"  (+{a['option_gain']:.2f}){tag}")
    w("")
    if write:
        os.makedirs(os.path.dirname(write) or ".", exist_ok=True)
        with open(write, "w") as f:
            f.write("\n".join(lines))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gaffer.ucl")
    ap.add_argument("--matchday", type=int, required=True)
    ap.add_argument("--offline", action="store_true",
                    help="use the committed snapshot, do not fetch")
    ap.add_argument("--sims", type=int, default=project.N)
    ap.add_argument("--time-limit", type=int, default=180)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)

    cfg = load_config()
    out = run(a.matchday, cfg, offline=a.offline, n=a.sims,
              time_limit=a.time_limit)
    report(out, write=TEXT)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {a.out} and {TEXT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
