"""When a transfer's gain arrives, not just how much of it there is.

A single discounted sum over the horizon answers "how much better is this
player over the next eight weeks", which is not the question a manager asks.
You do not marry a transfer. You hold a player while he is the best use of the
slot and move him on when he stops being that, so what matters is the gain over
the stretch you would ACTUALLY hold him — and two swaps with the same eight-week
total can be completely different assets:

    Saka -> Palmer    +0.1 +1.1 +1.1 +1.1 +1.4 +4.8 +3.7 +1.8   (cumulative)
    Bruno -> Palmer   -0.3 -0.2 -0.9 +1.0 -1.1 +1.2 -2.5 -6.5

The first is worth +4.8 if you take it and move on at GW13, and +1.8 if you sit
on it to GW15. The second looks like a clear no at -6.5, but its whole loss is
in the last two gameweeks; through GW13 it is a coin toss. A manager who intends
to revisit the slot in a month should read the peak. One who wants to buy and
forget should read the end. The horizon sum alone tells neither of them what
they need, and it quietly punishes a player whose fixtures turn late.

Nothing here re-ranks anything on its own. It is a lens on a decision the
solver has already made, because the solver optimises the whole line — it is
free to sell in GW13 — while the report was describing only the endpoint.
"""
from __future__ import annotations


def profile(out_ep, in_ep, gws, decay=1.0):
    """Week-by-week and cumulative gain of swapping `out` for `in`.

    `decay` defaults to 1.0: these are points as they will be scored, not
    discounted for uncertainty. The discount belongs in the decision, not in
    the description of when the points arrive.
    """
    rows, run = [], 0.0
    for i, g in enumerate(gws):
        d = (in_ep.get(g, 0.0) - out_ep.get(g, 0.0)) * (decay ** i)
        run += d
        rows.append({"gw": g, "week": d, "cumulative": run})
    return rows


def summarise(rows):
    """Peak cumulative gain, where it peaks, and what is left by the end.

    `hold_weeks` is how long the swap is worth keeping before the slot wants
    revisiting. `decays` flags a swap whose case rests on getting out in time.
    """
    if not rows:
        return {}
    peak = max(rows, key=lambda r: r["cumulative"])
    end = rows[-1]
    trough = min(rows, key=lambda r: r["cumulative"])
    return {
        "peak": peak["cumulative"],
        "peak_gw": peak["gw"],
        "end": end["cumulative"],
        "end_gw": end["gw"],
        "worst": trough["cumulative"],
        "worst_gw": trough["gw"],
        "hold_weeks": rows.index(peak) + 1,
        # Worth more if you move on at the right time than if you sit on it.
        "decays": peak["cumulative"] - end["cumulative"] > 1.0,
        # Looks bad over the horizon but is fine for a while.
        "front_loaded": end["cumulative"] < 0 <= peak["cumulative"],
    }


def verdict(rows, min_gain=1.5):
    """One line a person can act on."""
    s = summarise(rows)
    if not s:
        return "no projection to compare"
    if s["end"] >= min_gain and not s["decays"]:
        return (f"gains {s['end']:+.1f} by GW{s['end_gw']} and keeps gaining — "
                "hold it")
    if s["peak"] >= min_gain and s["decays"]:
        return (f"worth {s['peak']:+.1f} if you move on around GW{s['peak_gw']}, "
                f"but only {s['end']:+.1f} by GW{s['end_gw']} — a hop-on, hop-off")
    if s["front_loaded"]:
        return (f"roughly level to GW{s['peak_gw']} ({s['peak']:+.1f}), then "
                f"turns: {s['end']:+.1f} by GW{s['end_gw']}")
    if s["end"] < 0 and s["peak"] < min_gain:
        return f"loses {s['end']:+.1f} by GW{s['end_gw']} and never leads — no"
    return (f"{s['end']:+.1f} by GW{s['end_gw']}, peaking at {s['peak']:+.1f} "
            f"in GW{s['peak_gw']}")


def table(rows, width=5):
    """The profile as text, for the report."""
    gws = "".join(f"{r['gw']:>{width}}" for r in rows)
    wk = "".join(f"{r['week']:>+{width}.1f}" for r in rows)
    cum = "".join(f"{r['cumulative']:>+{width}.1f}" for r in rows)
    return f"  gw {gws}\n  wk {wk}\n  cum{cum}"
