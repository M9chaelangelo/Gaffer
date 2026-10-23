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
import re
import sys
import unicodedata
from datetime import datetime, timezone

import yaml

from .. import fetch as fplfetch
from .. import odds as oddsmod
from . import availability, bench, fetch, history, plan, project

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
    # Repair a snapshot written before matches-played was derived, so an
    # older file does not quietly reproduce the flat-prior bug.
    rows = fetch._attach_matches(snap["players"])
    return (rows, snap["fixtures"],
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

    # Per-matchday minutes, differenced out of a log of past reads. Nothing to
    # difference yet on the first matchday this runs for; it pays from the next.
    #
    # Only a LIVE read is recorded, the same rule write_snapshot follows. A
    # replay of the committed snapshot carries no news about minutes, and the
    # first version of this wrote on every run — so the test suite, which
    # drives run() with a synthetic pool, filed 114 invented players under
    # matchday 3 in the repository's own log. First-write-wins then kept the
    # invention and would have differenced it at matchday 4.
    mlog = history.load()
    if source == "live":
        mlog, _ = history.record(pool, matchday, log=mlog)
    played = history.per_matchday(mlog, matchday - 1)
    with_log = history.attach(pool, played)
    minutes_note = (f"{with_log} players with per-match minutes from "
                    f"{len(mlog)} recorded reads"
                    if with_log else
                    f"no per-match minutes yet — {len(mlog)} read(s) on file, "
                    "two consecutive ones are needed; using each player's "
                    "share of his club's minutes instead")

    # Availability. UEFA's prose lags by a day or more, so for the English
    # clubs FPL's numeric chance of playing is read instead, and `doubts:` in
    # the config overrides both.
    avail, avail_unmatched, avail_note = [], [], "not fetched"
    doubts = dict(cfg.get("doubts") or {})
    flags = {}
    # `--offline` is about the UEFA feed, which is the thing that gets blocked;
    # it deliberately does NOT suppress this. Re-solving against a committed
    # snapshot with the injury flags silently dropped is how a comparison comes
    # out saying to captain a man who is 75% to play, which is exactly the
    # mistake this was written to stop. If FPL is unreachable too, the note
    # says so and the plan carries on with UEFA's prose.
    if cfg.get("use_fpl_flags", True):
        try:
            flags, avail_unmatched = availability.chances(
                pool, fplfetch.bootstrap())
            avail_note = f"{len(flags)} flagged by FPL"
        except Exception as e:                              # noqa: BLE001
            avail_note = f"FPL flags unreachable: {e.__class__.__name__}"
    avail, doubts_unmatched = availability.apply(pool, flags, doubts)
    if doubts:
        avail_note += f"; {len(doubts)} from config"

    # Measured beats modelled. Configured odds win over the feed, the feed
    # wins over the strength model, and every fixture line says which it got.
    book = dict(cfg.get("odds") or {})
    priced, note = oddsmod.forward(cfg.get("odds_sport", "soccer_uefa_champs_league"))
    unpriced = []
    if priced:
        joined, unpriced = oddsmod.match_fixtures(fx, priced)
        for k, v in joined.items():
            book.setdefault(k, v)
    samples, fxrates, played = project.simulate(pool, fx, odds=book, n=n,
                                                with_minutes=True)
    projected = project.summarise(pool, samples)
    columns = {r["id"]: j for j, r in enumerate(pool)}

    squad_names = cfg.get("squad") or []
    owned = resolve(projected, squad_names)
    missing = [s for s in squad_names if s not in owned["named"]]
    if missing:
        # A budget is the squad's own value plus the bank. Two unmatched
        # names took 9.1m out of it and the solver was asked to buy fifteen
        # players for thirteen players' money: Infeasible, and it still
        # printed an eleven. Planning against a squad we cannot read is
        # worse than not planning.
        raise SystemExit(
            "cannot plan: these squad members in config/ucl.yaml do not "
            "match exactly one player in UEFA's feed.\n"
            + "\n".join(
                f"  {n!r} -> " + (", ".join(owned["near"].get(n) or [])
                                  or "nothing like it in the feed")
                for n in missing)
            + "\n\nUse one of the spellings above, or the player's feed id.")

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
    # The bench and the armband are two prices on one thing: the right to
    # change your mind during the matchday. Automatic substitutions cover an
    # absent starter, and any manual change — the armband move included —
    # switches them off for the rest of the matchday. So price the bench, and
    # price the move net of the cover it gives up, rather than adding two
    # numbers the rules make exclusive.
    benched = bench.order(result["bench"]) if result["xi"] else []
    result["bench"] = benched
    days = sorted({(f.get("kickoff") or "")[:10] for f in fx if f.get("kickoff")})
    if benched:
        insurance = bench.value(result["xi"], benched, samples, played, columns)
        costs = {d: bench.forfeit(result["xi"], benched, samples, played,
                                  columns, fx, d) for d in days}
    else:
        insurance, costs = {"ep": 0.0, "by_slot": [], "expected_subs": 0.0}, {}
    band = plan.armband(result["xi"], samples, columns, fx,
                        switch_cost={d: c["most"] for d, c in costs.items()})
    return {
        "bench_value": insurance,
        "autosub_forfeit": costs,
        "matchday": matchday,
        "generated": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "problems": problems,
        "missing_from_squad": missing,
        "unmatched_out": unmatched_out,
        "availability": [{"name": n, "club": c, "chance": v, "source": src}
                         for n, c, v, src in avail],
        "availability_note": avail_note,
        "minutes_note": minutes_note,
        "availability_unmatched": avail_unmatched,
        "doubts_unmatched": doubts_unmatched,
        "odds_note": note,
        "unpriced_fixtures": unpriced,
        "forced_out": sorted(forced),
        "fixtures": fxrates,
        "plan": result,
        "armband": band,
        "projected": _worth_keeping(projected, owned["ids"]),
    }


# How much of the pool to carry in docs/ucl.json. The whole of it, pretty
# printed, was 1.2MB and 59,000 lines rewritten on every run, which makes a
# diff unreadable and duplicates the snapshot that sits beside it.
KEEP_PROJECTED = 150


def _worth_keeping(projected, owned_ids):
    """The top of the pool by expected points, plus everyone you own.

    Your own players must survive the cut however badly they project —
    a squad member missing from the file reads as a player who does not
    exist, and the whole point of the row is to show why he is benched.
    """
    ranked = sorted(projected, key=lambda r: -r["ep"])
    keep = {r["id"]: r for r in ranked[:KEEP_PROJECTED]}
    own = set(map(str, owned_ids))
    for r in ranked:
        if r["id"] in own:
            keep.setdefault(r["id"], r)
    return sorted(keep.values(), key=lambda r: -r["ep"])


def resolve(rows, names):
    """Match configured squad names to feed ids, reporting what did not match."""
    ids, named, near = [], [], {}
    for name in names:
        hit = _match(rows, name)
        if hit:
            ids.append(hit["id"])
            named.append(name)
        else:
            near[name] = candidates(rows, name)
    return {"ids": ids, "named": named, "near": near}


def candidates(rows, name, limit=4):
    """Feed names a configured name nearly matched, to fix the config with.

    "Kane" matched two players and was correctly refused; "B. Fernandes"
    matched none because the feed writes him "Bruno Fernandes". Saying so
    is the difference between a one-word edit and a hunt.
    """
    key = _norm(_surname(name)) or _norm(name)
    hits = [r for r in rows
            if key and (key in _norm(r["name"]) or key in _norm(r["full_name"]))]
    hits.sort(key=lambda r: -r["price"])
    return [f"{r['name']} ({r['team']}, {r['pos']}, {r['price']:.1f})"
            for r in hits[:limit]]


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


def _parts(name):
    """("b", "fernandes") for "B. Fernandes"; ("bruno", "fernandes") for the
    written-out form. The last word is the surname either way."""
    words = [w for w in re.split(r"[\s.]+", str(name).strip()) if w]
    if not words:
        return "", ""
    return _norm(words[0]) if len(words) > 1 else "", _norm(words[-1])


def _surname(name):
    words = [w for w in re.split(r"[\s.]+", str(name).strip()) if w]
    return words[-1] if words else ""


def _match(rows, name):
    """A feed row for a configured name.

    Id, then the whole name, then an initial and a surname, then a plain
    substring. The initial step exists because UEFA writes "H. Kane" and
    "Bruno Fernandes" in the same column: a config saying "B. Fernandes"
    matched nothing, and a config saying "Kane" matched two men.
    """
    key = _norm(name)
    for r in rows:
        if r["id"] == str(name).strip():
            return r
    for r in rows:
        if _norm(r["name"]) == key or _norm(r["full_name"]) == key:
            return r

    first, last = _parts(name)
    if last:
        hits = []
        for r in rows:
            rf, rl = _parts(r["name"])
            ff, fl = _parts(r["full_name"])
            for cf, cl in ((rf, rl), (ff, fl)):
                if cl != last or not cl:
                    continue
                # "B." matches "Bruno" and "Bruno" matches "B.", but
                # "Bruno" must never match "Martim".
                if not first or not cf or cf.startswith(first) or first.startswith(cf):
                    hits.append(r)
                    break
        hits = list({r["id"]: r for r in hits}.values())
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            # Ambiguous is not a match. Picking the dearest "Silva" is how a
            # plan quietly becomes a plan for somebody else's player.
            return None

    hits = [r for r in rows
            if key and (key in _norm(r["name"]) or key in _norm(r["full_name"]))]
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

    if out.get("doubts_unmatched"):
        w("!! doubts you typed were not matched in the feed: "
          + ", ".join(out["doubts_unmatched"]))
    w(f"\nminutes: {out.get('minutes_note', 'not recorded')}")
    w(f"availability: {out.get('availability_note', 'not fetched')}")
    flagged = [a for a in (out.get("availability") or []) if a["chance"] < 1.0]
    if flagged:
        for a in sorted(flagged, key=lambda a: a["chance"]):
            state = "OUT" if a["chance"] <= 0 else f"{a['chance']:.0%} chance"
            w(f"  {a['name']:<20} {a['club']:<4} {state:<12} [{a['source']}]")
    w(f"\nodds: {out.get('odds_note', 'not fetched')}"
      + (f"  —  no price for: {', '.join(out['unpriced_fixtures'])}"
         if out.get("unpriced_fixtures") else ""))
    w("\nfixtures (kick-off order — the armband can move between days)")
    for f in out["fixtures"]:
        w(f"  {(f['kickoff'] or '')[:16]}  {f['home']:>22} {f['lh']:.2f}"
          f" - {f['la']:.2f} {f['away']:<22} [{f['source']}]")

    proof = "proved optimal" if p.get("optimal") else "NOT PROVED OPTIMAL"
    w(f"\nsolver: {p['status']} ({proof})   "
      f"spend {p['spend']:.1f} of {p['budget']:.1f}")
    if not p.get("optimal"):
        # After a failed solve the variables hold whatever CBC left behind.
        # The first live run printed an eleven from an Infeasible model, with
        # a spend of 90.0 against a budget of 81.0, and it read exactly like
        # advice. Nothing below is a plan, so none of it gets printed.
        w("\nNo plan. The solver did not finish, so the numbers it left "
          "behind are not an eleven\nand must not be read as one. "
          "Fix the cause above and run again.")
        if write:
            os.makedirs(os.path.dirname(write) or ".", exist_ok=True)
            with open(write, "w") as fh:
                fh.write("\n".join(lines))
        return "\n".join(lines)
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
    bv = out.get("bench_value") or {}
    w(f"bench (outfield in this order; the keeper sits in his own slot)   "
      f"worth {bv.get('ep', 0.0):+.2f} points, "
      f"{bv.get('expected_subs', 0.0):.2f} substitutions expected")
    slots = {s["name"]: s for s in bv.get("by_slot") or []}
    for i, r in enumerate(p["bench"], 1):
        sl = slots.get(r["name"]) or {}
        used = (f"  comes on {sl['p_used']:>5.1%} of the time, worth "
                f"{sl['ep']:+.2f}" if sl else "")
        w(f"  {i}. {r['pos']:<4} {r['name']:<18} {r['team']:<18} "
          f"ep {r['ep']:>5.2f}{used}")

    if out["armband"]:
        w("\narmband — captain, and the move you keep in hand")
        w("  (moving it switches the auto-subs off for the rest of the "
          "matchday, so the\n   gain below is already net of the cover "
          "that costs)")
        for d, c in sorted((out.get("autosub_forfeit") or {}).items()):
            if c["most"] <= 0:
                continue
            # UEFA say auto-subs stop for "the rest of the matchday" without
            # saying when they are applied. If they settle each day only the
            # later matches lose cover; if they settle once at the end, the
            # whole bench goes. Both bounds are printed and the larger is
            # charged, because that is the one you cannot be wrong about.
            w(f"   moving after {d} forfeits {c['least']:.2f}-"
              f"{c['most']:.2f} points of cover "
              f"({c['starters_after']} starters left to cover); "
              f"charged at {c['most']:.2f}")
        for a in out["armband"][:6]:
            tag = (f"  switch to {a['fallback']['name']} if he blanks"
                   if a["fallback"] else "  (last day — no move available)")
            if a["fallback"] and a["option_gain"] < 0.01:
                tag = (f"  leave it alone — switching to "
                       f"{a['fallback']['name']} does not cover the "
                       f"{a['switch_cost']:.2f} it forfeits")
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
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {a.out} and {TEXT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
