"""Invariants that fail loudly, because every bug so far failed quietly.

Not one defect found in this engine was a crash. Each was a confident,
well-formatted, wrong answer:

  - a Triple Captain ranked the best move on the page at +9.27, for a chip
    spent four gameweeks earlier;
  - a Wildcard planned for GW7 after it was played in GW6, which also
    corrupted every other week in the line;
  - a Bench Boost week reported at 60.2 against a true 82.5, printing the
    best week of five as the worst;
  - a team sheet that benched the squad's second- and third-best players to
    start one carrying a shoulder injury;
  - a squad pin that outlived its gameweek by three days, so the solve
    drafted against fifteen players the manager no longer owned.

A test suite cannot catch these, because each was legal, in-range output
from working code fed a wrong assumption. What catches them is asserting
the things that must be true of any answer before it is shown to anyone.

`check` returns violations rather than raising: a deadline-morning run that
dies produces nothing, which is worse than a run that prints its own
warning. `fatal` marks the ones that make the output actively misleading.
"""
from __future__ import annotations

XI_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}
UNAVAILABLE = ("i", "u", "s")        # injured, unavailable, suspended


def _v(code, msg, fatal=False):
    return {"code": code, "message": msg, "fatal": fatal}


def check_chips(weeks, chips_used):
    """No plan may spend a chip that is already gone."""
    spent = {str(c).strip().lower() for c in (chips_used or ())}
    names = {"wildcard": "wc", "triple captain": "tc",
             "bench boost": "bb", "free hit": "fh"}
    out = []
    for w in weeks or []:
        chip = w.get("chip")
        if not chip:
            continue
        key = names.get(str(chip).strip().lower(), str(chip).strip().lower())
        if key in spent:
            out.append(_v("chip-spent",
                          f"GW{w.get('gw')} plans {chip}, already played",
                          fatal=True))
    return out


def check_week_ep(weeks):
    """A week's reported total must equal the players it says will score.

    Under Bench Boost that is all fifteen; otherwise the eleven. Getting
    this wrong understated a chip week by 22.3 points.
    """
    out = []
    for w in weeks or []:
        xi, bench, cap = w.get("xi"), w.get("bench"), w.get("captain")
        if not xi or w.get("ep") is None or not isinstance(cap, dict):
            continue
        gw = w.get("gw")

        def ep_of(p):
            e = p.get("ep")
            return (e.get(gw) if isinstance(e, dict) else
                    (e[0] if isinstance(e, list) and e else e)) or 0.0

        boosted = str(w.get("chip") or "").lower().startswith("bench")
        scoring = list(xi) + list(bench or []) if boosted else list(xi)
        want = sum(ep_of(p) for p in scoring) + ep_of(cap)
        if abs(want - float(w["ep"])) > 0.15:
            out.append(_v("week-ep",
                          f"GW{gw} reports {w['ep']} but its "
                          f"{len(scoring)} scorers sum to {want:.1f}",
                          fatal=True))
    return out


def check_lineup(weeks):
    """The eleven must be a legal formation, and must not field the unfit."""
    out = []
    for w in weeks or []:
        xi = w.get("xi")
        if not xi:
            continue
        gw = w.get("gw")
        if len(xi) != 11:
            out.append(_v("xi-size", f"GW{gw} names {len(xi)} starters",
                          fatal=True))
        shape = {}
        for p in xi:
            shape[p.get("pos")] = shape.get(p.get("pos"), 0) + 1
        for pos, lo in XI_MIN.items():
            n = shape.get(pos, 0)
            if n < lo or n > XI_MAX[pos]:
                out.append(_v("formation",
                              f"GW{gw} has {n} {pos}, legal is "
                              f"{lo}-{XI_MAX[pos]}", fatal=True))
        for p in xi:
            if p.get("status") in UNAVAILABLE:
                out.append(_v("unfit-starter",
                              f"GW{gw} starts {p.get('name')} "
                              f"(status {p.get('status')})", fatal=True))
        # A benched player projected well above a starter is the signature of
        # the Bench Boost indifference bug, and of nothing else benign.
        bench = w.get("bench") or []
        if bench and len(xi) == 11:
            def ep_of(p):
                e = p.get("ep")
                return (e.get(gw) if isinstance(e, dict) else
                        (e[0] if isinstance(e, list) and e else e)) or 0.0
            worst_in = min(ep_of(p) for p in xi if p.get("pos") != "GKP")
            for b in bench:
                if b.get("pos") != "GKP" and ep_of(b) > worst_in + 1.0:
                    out.append(_v("bench-better",
                                  f"GW{gw} benches {b.get('name')} "
                                  f"({ep_of(b):.1f}) behind a starter on "
                                  f"{worst_in:.1f}"))
    return out


def check_squad_pin(cfg, solved_gw):
    """A hand-pinned squad must not outlive the gameweek it was pinned for.

    The pin exists because the API serves the last finished gameweek. That
    makes it right for exactly one deadline and wrong from the next, and it
    has already gone stale once — a GW7 solve drafted against a squad the
    manager had not owned for three days.
    """
    if not cfg.get("current_squad"):
        return []
    pinned = cfg.get("squad_pinned_for")
    if pinned is None:
        return [_v("pin-undated",
                   "current_squad is pinned but `squad_pinned_for` is unset, "
                   "so nothing can tell whether it is stale")]
    if int(pinned) != int(solved_gw):
        return [_v("pin-stale",
                   f"current_squad was pinned for GW{pinned} but this is "
                   f"GW{solved_gw} — re-pin it from the Transfers screen",
                   fatal=True)]
    return []


def check(cfg, weeks, solved_gw, chips_used=()):
    """Every invariant, in one call."""
    out = []
    out += check_chips(weeks, chips_used)
    out += check_week_ep(weeks)
    out += check_lineup(weeks)
    out += check_squad_pin(cfg, solved_gw)
    return out


def render(violations):
    if not violations:
        return "  invariants: all clear"
    lines = []
    for v in violations:
        lines.append(f"  {'!! FATAL' if v['fatal'] else '!! warn '} "
                     f"[{v['code']}] {v['message']}")
    return "\n".join(lines)
