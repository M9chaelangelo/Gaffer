"""Why, not just what.

A solver that says "sell Maguire, buy Tarkowski" is useless the first week it is
wrong, because you have no way to judge whether it was unlucky or broken. Every
recommendation here carries the numbers it was made from, so you can disagree
with it on the evidence rather than on faith.
"""
from . import horizon


def _fmt_pct(v):
    return f"{v * 100:.0f}%" if v is not None else "—"


def player_card(pid, proj, prof, lg_own, pack_own, elite_own, price_fc, gws):
    p = proj[pid]
    s = prof.get(pid, {})
    return {
        "name": p["name"], "team": p["team"], "pos": p["pos"], "price": p["price"],
        "ep_next": round(p["ep"][gws[0]], 2),
        "ep_horizon": round(sum(p["ep"][g] for g in gws), 1),
        "fixtures": [p["opp"][g] for g in gws],
        "start_prob": p.get("p60"),
        "ceiling": p.get("ceiling_next"),
        "exp_minutes": p["exp_min"],
        "xg90": s.get("xg90"), "xa90": s.get("xa90"),
        "shot_quality": s.get("shot_quality"),
        "attacking_share": s.get("attacking_share"),
        "defcon90": s.get("defcon90"),
        "defcon_prob": p.get("defcon_prob"),
        "cs_prob": p.get("cs_prob"),
        "setpiece": p.get("setpiece_bonus"),
        "bonus_exp": p.get("bonus_exp"),
        "bps90": p.get("bps90"),
        "own_league": lg_own.get(pid, 0.0),
        "own_pack": pack_own.get(pid, 0.0),
        "own_elite": elite_own.get(pid) if elite_own else None,
        "own_overall": p["own"] / 100,
        "price_direction": price_fc.get(pid, {}).get("direction"),
        "net_transfers": price_fc.get(pid, {}).get("net"),
        "status": p["status"], "news": p["news"],
    }


def _reasons(out_c, in_c, wildcard=False):
    """The specific, checkable claims behind one swap."""
    r = []
    d = in_c["ep_horizon"] - out_c["ep_horizon"]
    r.append(f"Projected {d:+.1f} points across the horizon "
             f"({in_c['ep_horizon']} vs {out_c['ep_horizon']}).")
    if d < 0:
        if wildcard:
            r.append(f"A downgrade on its own. It exists to release "
                     f"£{out_c['price'] - in_c['price']:.1f}m, which funds a "
                     f"larger upgrade elsewhere in the rebuild — judge the "
                     f"wildcard as one move, not fifteen.")
        else:
            r.append("Projected to lose points. Treat this as a flag that "
                     "something else in the plan is driving it, not as a "
                     "recommendation on its own.")

    if in_c["start_prob"] and out_c["start_prob"]:
        if in_c["start_prob"] - out_c["start_prob"] > 0.08:
            r.append(f"More certain to play: {_fmt_pct(in_c['start_prob'])} chance of "
                     f"60+ minutes against {_fmt_pct(out_c['start_prob'])}.")
        elif out_c["start_prob"] - in_c["start_prob"] > 0.08:
            r.append(f"Costs you minutes certainty: {_fmt_pct(in_c['start_prob'])} "
                     f"against {_fmt_pct(out_c['start_prob'])}. Taken anyway because "
                     f"the ceiling is higher.")

    if (in_c["shot_quality"] or 0) > (out_c["shot_quality"] or 0) and in_c["shot_quality"]:
        r.append(f"Shoots from better positions — expected goals per unit of threat "
                 f"{in_c['shot_quality']:.4f} vs {out_c['shot_quality'] or 0:.4f}, "
                 f"which means his attempts come from inside the box rather than "
                 f"outside it.")

    if (in_c["attacking_share"] or 0) > 0.20:
        r.append(f"{_fmt_pct(in_c['attacking_share'])} of his team's expected goal "
                 f"involvement runs through him.")

    if in_c["pos"] in ("GKP", "DEF") and in_c["cs_prob"] is not None:
        r.append(f"Clean sheet odds {_fmt_pct(in_c['cs_prob'])} against "
                 f"{_fmt_pct(out_c['cs_prob'])} — worth "
                 f"{4 * (in_c['cs_prob'] - (out_c['cs_prob'] or 0)):+.1f} points "
                 f"a game before anything else.")
    if (in_c["defcon_prob"] or 0) > 0.35:
        r.append(f"{_fmt_pct(in_c['defcon_prob'])} chance of hitting the "
                 f"defensive contribution threshold, worth "
                 f"{2 * in_c['defcon_prob']:.1f} points a game on its own.")
    if (in_c["defcon90"] or 0) >= 10:
        r.append(f"{in_c['defcon90']} defensive actions per 90 — a repeatable "
                 f"floor of defensive contribution points independent of form.")

    if in_c["setpiece"]:
        r.append(f"On set pieces, worth about {in_c['setpiece']:+.2f} points per 90 "
                 f"before anything else happens.")

    gap = out_c["own_pack"] - in_c["own_pack"]
    if gap > 0.2:
        r.append(f"Leverage: {_fmt_pct(out_c['own_pack'])} of the teams above you "
                 f"own the man you are selling, {_fmt_pct(in_c['own_pack'])} own the "
                 f"one you are buying. You gain ground when he returns.")
    elif gap < -0.2:
        r.append(f"Reduces risk rather than adding leverage — "
                 f"{_fmt_pct(in_c['own_pack'])} of your pack already own him.")

    if in_c["own_elite"] is not None and in_c["own_elite"] - in_c["own_overall"] > 0.15:
        r.append(f"The world's best managers are ahead of the crowd on him: "
                 f"{_fmt_pct(in_c['own_elite'])} elite ownership against "
                 f"{_fmt_pct(in_c['own_overall'])} overall.")

    if out_c["status"] != "a":
        r.append(f"The man leaving is flagged: {out_c['news'] or out_c['status']}.")
    if out_c["price_direction"] == "fall":
        r.append(f"He is also falling in price ({out_c['net_transfers']:+,} net "
                 f"transfers), so holding costs team value too.")
    if in_c["price_direction"] == "rise":
        r.append(f"The incoming player is rising ({in_c['net_transfers']:+,} net), "
                 f"so acting now is cheaper than acting later.")
    return r


def _pair(outs, ins):
    """Match departures to arrivals within a position, best against best.

    A swap only means something if the two players could occupy the same squad
    slot. Pairing by list order compares a goalkeeper to a striker."""
    pairs, leftover_in, leftover_out = [], [], []
    for pos in ("GKP", "DEF", "MID", "FWD"):
        o = sorted([p for p in outs if p["pos"] == pos], key=lambda p: -p["price"])
        i = sorted([p for p in ins if p["pos"] == pos], key=lambda p: -p["price"])
        for a, b in zip(o, i):
            pairs.append((a, b))
        leftover_out += o[len(i):]
        leftover_in += i[len(o):]
    return pairs, leftover_out, leftover_in


def transfers(week, proj, prof, lg_own, pack_own, elite_own, price_fc, gws):
    card = lambda p: player_card(p["id"], proj, prof, lg_own, pack_own,
                                 elite_own, price_fc, gws)
    pairs, _, extra_in = _pair(week.get("out_players", []),
                               week.get("in_players", []))
    out = []
    wc = week.get("chip") == "Wildcard"
    for a, b in pairs:
        oc, ic = card(a), card(b)
        # WHEN the gain arrives, not just how much of it there is over the
        # horizon. Two swaps with the same eight-week total can be a keeper
        # and a hop-on-hop-off, and the endpoint alone does not say which.
        rows = horizon.profile(proj[a["id"]]["ep"], proj[b["id"]]["ep"], gws)
        out.append({"out": oc, "in": ic, "reasons": _reasons(oc, ic, wc),
                    "timing": horizon.summarise(rows),
                    "timing_note": horizon.verdict(rows),
                    "weekly_gain": [round(r["week"], 2) for r in rows],
                    "cumulative_gain": [round(r["cumulative"], 2) for r in rows]})
    for b in extra_in:
        ic = card(b)
        out.append({"out": None, "in": ic,
                    "reasons": [f"Added at {ic['ep_horizon']} projected points "
                                f"across the horizon."]})
    # A wildcard is a rebuild, not a list of swaps. Show the biggest changes.
    if week.get("chip") == "Wildcard":
        out.sort(key=lambda t: -(t["in"]["ep_horizon"]
                                 - (t["out"]["ep_horizon"] if t["out"] else 0)))
    return out


def captain(week, proj, prof, pack_own, elite_own, gws):
    c = week["captain"]
    pid = c["id"]
    pack = pack_own.get(pid, 0.0)
    el = (elite_own or {}).get(pid)
    r = [f"Highest projection in your eleven at {c['ep'][week['gw']]:.1f} points, "
         f"against {c['opp'][week['gw']]}."]
    if pack < 0.5:
        r.append(f"Only {_fmt_pct(pack)} of the teams above you own him, so every "
                 f"point he scores is a point of ground gained, doubled.")
    if el is not None and el < 0.4:
        r.append(f"Elite ownership is {_fmt_pct(el)} — captaining him is a "
                 f"differential against the best managers in the world, not just "
                 f"against your league.")
    s = prof.get(pid, {})
    if s.get("xg90"):
        r.append(f"{s['xg90']} expected goals per 90 with "
                 f"{_fmt_pct(s.get('attacking_share'))} of his team's attack "
                 f"running through him.")

    # The armband is decided by the tail, not the mean, so say what the tail is
    # — and name the player with the bigger one when it is not him. An armband
    # recommendation you cannot argue with is one you cannot correct.
    gw = week["gw"]
    haul = c.get("ceiling", {}).get(gw)
    if haul is not None:
        r.append(f"Hauls — 10+ points — in {_fmt_pct(haul)} of weeks like this one.")
        rival = max(
            (proj[q["id"]] for q in week["xi"] if q["id"] != pid
             and proj[q["id"]].get("ceiling", {}).get(gw) is not None),
            key=lambda q: q["ceiling"][gw], default=None)
        if rival is not None and rival["ceiling"][gw] > haul + 0.02:
            r.append(f"{rival['name']} actually has the higher ceiling "
                     f"({_fmt_pct(rival['ceiling'][gw])} against {_fmt_pct(haul)}) "
                     f"but projects {rival['ep'][gw]:.1f} to his "
                     f"{c['ep'][gw]:.1f} — worth the armband only if you need "
                     f"variance rather than points.")
    return {"name": c["name"], "reasons": r}
