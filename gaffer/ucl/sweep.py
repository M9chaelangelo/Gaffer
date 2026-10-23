"""Which matchday to play a chip on, measured against what you would do anyway.

The two chips are not the same instrument and the question is not the same
question.

**Limitless** lifts the budget for one matchday. Its value is the gap between
the best eleven money can buy and the best eleven YOUR money can buy, on that
matchday only — a one-matchday quantity, which is why it can be priced
exactly.

**The Wildcard** rebuilds the squad for keeps. Its value runs to the end of
the phase, and the thing it must be measured against is not a squad frozen
from here on. An earlier version of this comparison put the Wildcard at +54
points, which measured the cost of doing nothing for six matchdays and said
nothing about the chip. Both sides here are rolled forward with the two free
transfers a matchday the game actually gives you, so the comparison is
between two strategies rather than between a strategy and paralysis.

What this cannot see is future team news. A player's rates and his chance of
starting are whatever they are today, for every matchday in the window; only
the fixture changes. That is a real limitation and it cuts one way: a horizon
is more confident than it should be about the far end of itself. It is still
better than the alternative, which was to look at one matchday and call it
the phase.
"""
from __future__ import annotations

from . import plan, project
from .rules import (CHIP_BLOCKED_MATCHDAYS, FREE_TRANSFERS_BY_MATCHDAY,
                    SIMULTANEOUS_MATCHDAYS)

UNLIMITED = 1e6


def fixtures_for(sched, matchday):
    """One matchday's fixtures in the shape the simulator wants.

    The committed schedule carries a date and not a kick-off time, and only
    the DAY matters here: it is what decides whether the armband can still
    be moved. A single time on that date gives every fixture the same day,
    which is the fact being used.
    """
    return sorted(
        ({"matchday": str(matchday), "home": f["home"], "away": f["away"],
          "kickoff": f"{f['date']}T20:00:00Z"} for f in sched[matchday]),
        key=lambda f: (f["kickoff"], f["home"]))


def project_window(pool, sched, matchdays, odds=None, n=project.N,
                   seed=project.SEED):
    """{matchday: [projected rows]} for each matchday in the window.

    Each matchday is drawn separately, with its own seed, because a club's
    two fixtures are two different problems and sharing draws between them
    would correlate things that are not correlated.
    """
    out = {}
    for md in matchdays:
        fx = fixtures_for(sched, md)
        samples, _ = project.simulate(pool, fx, odds=odds, n=n, seed=seed + md)
        out[md] = project.summarise(pool, samples)
    return out


def _ep(rows_by_md):
    return {md: {r["id"]: r["ep"] for r in rows} for md, rows in
            rows_by_md.items()}


def _worth(out):
    """An eleven plus the armband, for one matchday."""
    if not out.get("xi"):
        return 0.0
    return sum(r["ep"] for r in out["xi"]) + max(
        (r["ep"] for r in out["xi"]), default=0.0)


def roll(rows_by_md, squad_ids, bank=0.0, ft_by_md=None, locked=(), banned=(),
         time_limit=90, bench_weight=None):
    """Carry a squad forward matchday by matchday, spending its transfers.

    This is the honest baseline and also the honest way to value a Wildcard
    squad: both sides get the transfers the game gives them. `ft_by_md`
    overrides the allowance per matchday — pass 0 for the matchday a chip was
    played on, since the chip has already done the rebuilding.

    Returns the running total and what happened each matchday. A matchday the
    solver cannot finish stops the roll rather than being skipped, because a
    total that silently misses a matchday is worse than a short one.
    """
    current, legs, total = list(squad_ids), [], 0.0
    for md in sorted(rows_by_md):
        ft = (ft_by_md or {}).get(md, FREE_TRANSFERS_BY_MATCHDAY.get(md, 2))
        out = plan.optimise(rows_by_md[md], owned_ids=current, bank=bank,
                            free_transfers=ft, locked=locked, banned=banned,
                            time_limit=time_limit, bench_weight=bench_weight)
        # `optimal`, not `xi`. After a failed solve CBC leaves whatever it
        # last had in the variables, so an Infeasible model can hand back a
        # populated fifteen and a legal-looking eleven — `report()` carries
        # the same lesson from a live run that printed one. Which build of
        # CBC you have decides whether the wreckage looks empty or looks
        # like a plan: this test passed locally and failed on the runner.
        if not out["optimal"] or not out["xi"]:
            legs.append({"matchday": md, "status": out["status"],
                         "worth": None})
            break
        worth = _worth(out)
        total += worth
        legs.append({
            "matchday": md, "status": out["status"], "worth": worth,
            "free_transfers": ft,
            "captain": out["captain"]["name"] if out["captain"] else None,
            "out": [r["name"] for r in out["out"]],
            "in": [r["name"] for r in out["in"]],
            "spend": out["spend"], "bank": out["budget"] - out["spend"],
            # The squad this leg leaves behind, which is the squad you would
            # actually be holding when the next matchday comes around.
            "held_before": list(current), "bank_before": bank,
            "squad_ids": [r["id"] for r in out["squad"]],
        })
        current = [r["id"] for r in out["squad"]]
        bank = out["budget"] - out["spend"]
    return {"total": total, "legs": legs,
            "complete": len(legs) == len(rows_by_md)
            and all(l["worth"] is not None for l in legs)}


def holdings(walk):
    """{matchday: (squad ids, bank)} going INTO each matchday.

    This is the thing a chip has to be measured against, and getting it
    wrong is subtle. Comparing a rebuild at matchday seven against the squad
    you hold TODAY credits the chip with three matchdays of transfers you
    would have made anyway — the same error as freezing the squad, moved
    along the calendar. It showed up as a Wildcard worth more over a
    two-matchday window than over a five-matchday one, which cannot be true.
    """
    return {leg["matchday"]: (leg["held_before"], leg["bank_before"])
            for leg in walk["legs"] if leg.get("worth") is not None}


def tails(walk):
    """{matchday: what the baseline scores from that matchday to the end}."""
    legs = [l for l in walk["legs"] if l.get("worth") is not None]
    out, running = {}, 0.0
    for leg in reversed(legs):
        running += leg["worth"]
        out[leg["matchday"]] = running
    return out


def limitless(rows_by_md, walk, banned=(), locked=(), time_limit=120):
    """What the chip is worth on each matchday: the gap money cannot close.

    `walk` is the baseline roll, and the comparison is against the squad that
    roll leaves you holding when the matchday arrives — not the squad you
    hold today, which would credit the chip with transfers you were going to
    make anyway.

    Note what is NOT the answer here. The best eleven money can buy is the
    same for everybody; the chip is worth what YOUR squad is short by, so a
    manager already near the top of the market gains little from it and one
    holding cheap enablers gains a lot.
    """
    base, out = tails(walk), {}
    mine_at = {leg["matchday"]: leg["worth"] for leg in walk["legs"]
               if leg.get("worth") is not None}
    for md, rows in sorted(rows_by_md.items()):
        if md not in mine_at:
            continue
        best = plan.optimise(rows, budget=UNLIMITED, banned=banned,
                             time_limit=time_limit)
        if not best["optimal"]:
            out[md] = {"matchday": md, "status": best["status"], "gain": None,
                       "blocked": md in CHIP_BLOCKED_MATCHDAYS,
                       "simultaneous": md in SIMULTANEOUS_MATCHDAYS,
                       "optimal": False}
            continue
        out[md] = {
            "matchday": md,
            "blocked": md in CHIP_BLOCKED_MATCHDAYS,
            "simultaneous": md in SIMULTANEOUS_MATCHDAYS,
            "mine": mine_at[md], "best": _worth(best),
            "gain": _worth(best) - mine_at[md],
            "spend": best["spend"],
            "optimal": bool(best["optimal"]),
            "xi": best["xi"], "captain": best["captain"],
            "baseline_tail": base.get(md),
        }
    return out


def wildcard(rows_by_md, walk, locked=(), banned=(),
             window_time_limit=600, roll_time_limit=90, prune=250):
    """What a rebuild on each matchday is worth over the rest of the phase.

    For each candidate matchday the squad is rebuilt for the whole remaining
    window and then rolled forward on its ordinary transfers, because a
    Wildcard squad is not frozen either. The baseline is the same roll
    without the rebuild, from the squad `walk` leaves you holding when that
    matchday arrives.
    """
    mds = sorted(rows_by_md)
    rows = rows_by_md[mds[0]]
    held, base = holdings(walk), tails(walk)
    out = {}
    for md in mds:
        if md not in held:
            continue
        window = {k: v for k, v in rows_by_md.items() if k >= md}
        squad_ids, bank = held[md]
        built = plan.optimise_window(
            rows, _ep(window), owned_ids=squad_ids, bank=bank,
            free_transfers=None, locked=locked, banned=banned,
            time_limit=window_time_limit, prune=prune)
        if not built["optimal"] or not built["squad"]:
            out[md] = {"matchday": md, "status": built["status"],
                       "gain": None}
            continue
        # The chip has already rebuilt the squad, so this matchday gets no
        # transfers; every later one gets its allowance, exactly as the
        # baseline does.
        after = roll(window, [r["id"] for r in built["squad"]],
                     bank=built["budget"] - built["spend"],
                     ft_by_md={md: 0}, locked=locked, banned=banned,
                     time_limit=roll_time_limit)
        out[md] = {
            "matchday": md,
            "blocked": md in CHIP_BLOCKED_MATCHDAYS,
            "window": sorted(window),
            "baseline": base[md], "with_chip": after["total"],
            "gain": after["total"] - base[md],
            "transfers": len(built["in"]),
            "optimal": bool(built["optimal"] and after["complete"]),
            "status": built["status"],
            "squad": built["squad"], "spend": built["spend"],
            "chip_legs": after["legs"],
        }
    return out


def run(pool, sched, matchdays, squad_ids, bank=0.0, odds=None,
        n=project.N, locked=(), banned=(), roll_time_limit=90,
        window_time_limit=600, prune=250):
    """Project the window once, roll the baseline once, price both chips.

    One entry point, because the baseline roll is what both chips are
    measured against and computing it twice is how the two answers drift
    apart.
    """
    by_md = project_window(pool, sched, matchdays, odds=odds, n=n)
    walk = roll(by_md, squad_ids, bank=bank, locked=locked, banned=banned,
                time_limit=roll_time_limit)
    return {
        "projected": by_md,
        "baseline": walk,
        "limitless": limitless(by_md, walk, banned=banned, locked=locked,
                               time_limit=roll_time_limit),
        "wildcard": wildcard(by_md, walk, locked=locked, banned=banned,
                             window_time_limit=window_time_limit,
                             roll_time_limit=roll_time_limit, prune=prune),
    }


def table(lim=None, wc=None):
    """The sweep as a person reads it, blocked matchdays and all."""
    lines = []
    if lim:
        lines += ["", "Limitless — one matchday, no budget",
                  "  md   your best   money's best   gain   notes"]
        for md, r in sorted(lim.items()):
            notes = []
            if r["blocked"]:
                notes.append("CHIP BLOCKED: everyone has unlimited transfers")
            if r["simultaneous"]:
                notes.append("all kick off at once — no armband move, "
                             "no in-play subs")
            if not r["optimal"]:
                notes.append(f"not proved optimal ({r.get('status', '?')})")
            lines.append(f"  {md:<4} {r['mine']:>9.2f}   {r['best']:>12.2f}  "
                         f"{r['gain']:>+6.2f}   {'; '.join(notes)}")
    if wc:
        lines += ["", "Wildcard — a rebuild, measured against rolling your "
                      "own squad forward",
                  "  md   window     hold    rebuild   gain   moves  notes"]
        for md, r in sorted(wc.items()):
            if r.get("gain") is None:
                lines.append(f"  {md:<4} no plan ({r['status']})")
                continue
            notes = []
            if r["blocked"]:
                notes.append("CHIP BLOCKED")
            if not r["optimal"]:
                notes.append("not proved optimal")
            w = f"{r['window'][0]}-{r['window'][-1]}"
            lines.append(f"  {md:<4} {w:<9} {r['baseline']:>7.1f} "
                         f"{r['with_chip']:>9.1f} {r['gain']:>+6.1f} "
                         f"{r['transfers']:>6}  {'; '.join(notes)}")
    return "\n".join(lines)
