"""A plan is a photograph, and team news moves.

The solve reads FPL's player flags once, projects from them, and writes a
recommendation. That recommendation is then read by a person some hours
later — often most of a day later, because the solve runs in the morning and
the deadline is at tea time. In between, managers hold press conferences.

On 23 October 2026 the gameweek-8 plan was solved at 09:29 with Stach at 5.13
expected points and Saka a fifty-fifty. By 13:53 Stach had a foot injury and
was worth 2.88, and Saka had gone to nought with no return date. The plan on
disk still said to start one and described the other as a doubt. Nothing was
wrong with the arithmetic; the paper had simply aged, and nothing in the
output said so.

So the solve now records what it believed about every player whose
availability the recommendation depends on, and `verify` re-reads those same
flags and says plainly whether the plan still stands. One request, about a
second, and it is the difference between advice you can act on and advice
you have to re-derive.

What it does NOT cover: a player outside the decision set becoming available
or unavailable can change what the best plan would be, and this will not
notice. It answers "is the plan I am holding still valid", not "is it still
optimal".
"""
from __future__ import annotations

import datetime

# The fields that decide whether a plan is still the right plan. `news` is
# included as a presence flag rather than verbatim: the wording is rewritten
# constantly ("Knock" becoming "Minor knock") without the decision changing,
# while news appearing or clearing always matters.
WATCHED = ("status", "chance_of_playing_next_round")


def fingerprint(boot, ids):
    """{player id: what the solve believed about his availability}."""
    want = {int(i) for i in ids}
    out = {}
    for e in boot["elements"]:
        if e["id"] not in want:
            continue
        out[str(e["id"])] = {
            "name": e["web_name"],
            "status": e["status"],
            "chance": e.get("chance_of_playing_next_round"),
            "has_news": bool((e.get("news") or "").strip()),
            "news": (e.get("news") or "").strip(),
        }
    return out


def decision_set(plan, squad_ids):
    """Whose availability this recommendation actually rests on.

    The fifteen you own, plus everyone the plan buys or sells across the
    horizon. A flag on any of them can change the answer; a flag on the rest
    of the game can change what the BEST answer would be, which is a
    different and much larger question.
    """
    ids = set(map(str, squad_ids))
    for w in plan.get("weeks") or []:
        for side in ("in_players", "out_players"):
            for p in w.get(side) or []:
                if isinstance(p, dict) and p.get("id") is not None:
                    ids.add(str(p["id"]))
    return ids


def compare(before, after):
    """What changed, worst first, in words a person can act on."""
    out = []
    for pid, was in (before or {}).items():
        now = (after or {}).get(pid)
        if now is None:
            continue
        if was.get("status") != now.get("status"):
            out.append({
                "id": pid, "name": now.get("name") or was.get("name"),
                "kind": "status", "severity": _severity(was, now),
                "was": was.get("status"), "now": now.get("status"),
                "news": now.get("news") or "",
            })
        elif was.get("chance") != now.get("chance"):
            out.append({
                "id": pid, "name": now.get("name") or was.get("name"),
                "kind": "chance", "severity": _severity(was, now),
                "was": was.get("chance"), "now": now.get("chance"),
                "news": now.get("news") or "",
            })
        elif was.get("has_news") != now.get("has_news"):
            out.append({
                "id": pid, "name": now.get("name") or was.get("name"),
                "kind": "news", "severity": 1 if now.get("has_news") else 0,
                "was": "clear" if not was.get("has_news") else "flagged",
                "now": "flagged" if now.get("has_news") else "clear",
                "news": now.get("news") or "",
            })
    return sorted(out, key=lambda c: -c["severity"])


def _severity(was, now):
    """How much worse he got. Negative means he got better."""
    return _risk(now) - _risk(was)


def _risk(s):
    if s.get("status") in ("i", "s", "u", "n"):
        return 100
    c = s.get("chance")
    if c is None:
        return 0 if not s.get("has_news") else 5
    return max(0, 100 - int(c))


def verdict(changes):
    """Can the plan still be acted on as written?

    `stale` means a player the plan relies on got materially worse and the
    eleven or the transfer may now be wrong. `check` means something moved
    but in your favour, or only in wording.
    """
    if not changes:
        return "fresh", "no change to anyone this plan depends on"
    worse = [c for c in changes if c["severity"] > 0]
    if not worse:
        return "check", (f"{len(changes)} change(s), none of them bad — "
                         "the plan still stands")
    bad = max(worse, key=lambda c: c["severity"])
    return "stale", (f"{len(worse)} player(s) got worse, worst is "
                     f"{bad['name']} ({bad['news'] or bad['now']}) — "
                     "re-solve before acting on this")


def age(solved_at, now=None):
    """How old the plan is, in hours, or None if the timestamp is unusable."""
    if not solved_at:
        return None
    try:
        t = datetime.datetime.fromisoformat(str(solved_at).replace("Z", "+00:00"))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=datetime.timezone.utc)
    now = now or datetime.datetime.now(datetime.timezone.utc)
    return (now - t).total_seconds() / 3600.0


def report(state, boot, now=None):
    """The whole check, ready to print."""
    before = (state or {}).get("availability") or {}
    after = fingerprint(boot, before) if before else {}
    changes = compare(before, after)
    code, why = verdict(changes)
    hours = age((state or {}).get("solved_at"), now)
    if not before:
        code, why = "unknown", ("this plan predates the freshness check — "
                                "re-solve to get one")
    return {"code": code, "why": why, "hours": hours, "changes": changes,
            "watched": len(before), "solved_at": (state or {}).get("solved_at"),
            "deadline": (state or {}).get("deadline")}


def lines(rep):
    """The report as text, loudest first."""
    out = []
    mark = {"fresh": "OK", "check": "..", "stale": "!!", "unknown": "??"}
    age_s = (f"{rep['hours']:.1f}h old" if rep.get("hours") is not None
             else "age unknown")
    out.append(f"{mark.get(rep['code'], '??')} plan is {rep['code']} "
               f"({age_s}, {rep['watched']} players watched): {rep['why']}")
    for c in rep["changes"]:
        arrow = "worse" if c["severity"] > 0 else "better"
        out.append(f"   {c['name']:<16}{c['kind']:<8}{str(c['was']):>8} -> "
                   f"{str(c['now']):<8} {arrow}"
                   + (f"   {c['news']}" if c["news"] else ""))
    return out
