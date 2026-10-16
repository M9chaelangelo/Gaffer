"""What everybody else is doing — the elite, the pack, and the whole game.

Three of the four crowd signals this engine already computes were being
thrown away. `elite.snapshot` reads the top managers in the world off league
314 and returns their ownership, their captains and their chips; only
ownership ever reached the output. `elite.buy_signals` was written and never
called. And FPL publishes, per gameweek, how many of its eleven million
managers played each chip — which nothing read at all.

One honesty constraint runs through all of it, and it is not a limitation of
the code: **nobody's picks for an upcoming deadline are public.** The picks
endpoint serves the last finished gameweek, and `chip_plays` fills in as a
gameweek is played. So the question "is everyone playing their Bench Boost
this week?" cannot be answered before the deadline by anyone, including the
sites that claim to. What can be answered is what the crowd did in every
gameweek so far, which is the better question anyway: it shows whether a
chip is being spent or hoarded.
"""
from __future__ import annotations

CHIP_NAMES = {"wildcard": "Wildcard", "3xc": "Triple Captain",
              "bboost": "Bench Boost", "freehit": "Free Hit",
              "manager": "Assistant Manager"}


def chip_history(boot):
    """How many managers played each chip, per finished gameweek.

    From `events[].chip_plays`, which FPL publishes and nothing here read.
    Counts are of the whole game, so they are the crowd, not the elite.
    """
    total = max(1, int(boot.get("total_players") or 0))
    out = []
    for ev in boot.get("events", []):
        plays = ev.get("chip_plays") or []
        if not plays:
            continue
        row = {"gw": ev["id"], "finished": bool(ev.get("finished")), "chips": {}}
        for p in plays:
            name = CHIP_NAMES.get(p.get("chip_name"), p.get("chip_name"))
            n = int(p.get("num_played") or 0)
            row["chips"][name] = {"played": n, "share": round(n / total, 5)}
        out.append(row)
    return {"total_players": total, "weeks": out}


def chip_appetite(boot, chip="Bench Boost"):
    """Is this chip being spent or hoarded, and how fast.

    `remaining` is a floor, not a count: it assumes one chip per manager per
    half and counts only what has been played in the gameweeks published so
    far.
    """
    hist = chip_history(boot)
    weeks = [w for w in hist["weeks"] if w["finished"]]
    played = [(w["gw"], w["chips"].get(chip, {}).get("share", 0.0)) for w in weeks]
    spent = sum(s for _, s in played)
    peak = max(played, key=lambda r: r[1], default=(None, 0.0))
    return {"chip": chip, "by_gw": played,
            "share_spent": round(spent, 4),
            "share_remaining": round(max(0.0, 1.0 - spent), 4),
            "busiest_gw": peak[0], "busiest_share": round(peak[1], 4)}


def elite_chips(el):
    """What the top managers did with their chips in the last finished week."""
    if not el or not el.get("chips"):
        return {"n": (el or {}).get("n", 0), "played": {}, "note":
                "no chip was played by the sampled elite in that gameweek"}
    return {"n": el.get("n", 0),
            "played": {CHIP_NAMES.get(k, k): round(v, 4)
                       for k, v in sorted(el["chips"].items(),
                                          key=lambda kv: -kv[1])}}


def elite_captains(el, proj, limit=10):
    """Who the best managers actually put the armband on."""
    caps = (el or {}).get("captains") or {}
    rows = []
    for pid, share in sorted(caps.items(), key=lambda kv: -kv[1])[:limit]:
        p = proj.get(pid) or {}
        rows.append({"id": pid, "name": p.get("name", f"#{pid}"),
                     "team": p.get("team"), "share": round(share, 4),
                     "ep_next": p.get("ep_next")})
    return rows


def versus_elite(squad_ids, el, proj, boot=None, limit=12, min_gap=0.15):
    """What the best own that you do not, and what you hold that they have left.

    This is the closest honest answer to "what would a top manager do with my
    team". It is not advice from an expert; it is the revealed preference of
    the best two hundred managers in the world, differenced against the
    fifteen actually owned here.
    """
    own = (el or {}).get("own") or {}
    squad = set(squad_ids or ())
    missing, holding = [], []
    for pid, share in own.items():
        if share < min_gap or pid in squad:
            continue
        p = proj.get(pid) or {}
        missing.append({"id": pid, "name": p.get("name", f"#{pid}"),
                        "team": p.get("team"), "pos": p.get("pos"),
                        "price": p.get("price"), "elite": round(share, 4),
                        "ep_next": p.get("ep_next"),
                        "ep_horizon": p.get("ep_horizon")})
    for pid in squad:
        p = proj.get(pid) or {}
        share = own.get(pid, 0.0)
        holding.append({"id": pid, "name": p.get("name", f"#{pid}"),
                        "team": p.get("team"), "pos": p.get("pos"),
                        "price": p.get("price"), "elite": round(share, 4),
                        "ep_next": p.get("ep_next")})
    missing.sort(key=lambda r: -r["elite"])
    holding.sort(key=lambda r: r["elite"])
    return {"n": (el or {}).get("n", 0),
            "they_own_you_do_not": missing[:limit],
            "you_own_they_mostly_do_not": [h for h in holding
                                           if h["elite"] < 0.10][:limit]}


def build(boot, el, proj, squad_ids):
    """Every crowd signal, in one block."""
    return {
        "elite_sample": (el or {}).get("n", 0),
        "note": ("Nobody's picks for an upcoming deadline are public — the "
                 "picks endpoint serves the last finished gameweek. Elite "
                 "rows below describe what the best managers DID, not what "
                 "they will do."),
        "elite_chips_last_gw": elite_chips(el),
        "elite_captains": elite_captains(el, proj),
        "versus_elite": versus_elite(squad_ids, el, proj, boot),
        "bench_boost_appetite": chip_appetite(boot, "Bench Boost"),
        "triple_captain_appetite": chip_appetite(boot, "Triple Captain"),
        "wildcard_appetite": chip_appetite(boot, "Wildcard"),
        "chip_history": chip_history(boot),
    }
