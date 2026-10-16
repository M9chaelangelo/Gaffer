"""The template, the ceiling, and the arithmetic of beating a mini-league.

Winning a mini-league is not the same problem as scoring the most points,
and conflating them is why a manager can play a "good" week and lose ground.
What moves you up is not your score; it is your score *minus* what the teams
around you scored. A captain owned by everyone above you is worth nothing in
that race however many points he returns — and a 4-point week from a player
none of them own is worth more than a 9-point week from one they all have.

So this module measures three teams and the distance between them:

  **the template** — the most-owned eleven in the field you are actually
  racing. Not the global template: the one in your league, because those are
  the teams whose scores subtract from yours.

  **the ceiling** — the highest-projecting legal eleven in the game,
  ignoring budget. Not a target, a yardstick: it says how much of the
  available return any squad is capturing.

  **yours** — your best legal eleven from the fifteen you own.

## The arithmetic

Against a field that owns player p with frequency o(p):

  owning him   gains you  ep(p) x (1 - o(p))   of relative ground
  missing him  costs you  ep(p) x o(p)

A player everybody owns (o -> 1) is worth nearly nothing to own and ruinous
to miss. A player nobody owns (o -> 0) is nearly free to miss and worth his
whole projection to own. **Edge** is the sum of the first over your eleven
minus the sum of the second over the template players you lack, and it is
the number to maximise when the goal is a league rather than a rank.
"""
from __future__ import annotations

XI_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}


def _ep(p, gw):
    e = (p or {}).get("ep")
    if isinstance(e, dict):
        return float(e.get(gw, 0.0) or 0.0)
    if isinstance(e, list) and e:
        return float(e[0] or 0.0)
    return float(e or 0.0)


def _ep_sum(p, gws):
    return sum(_ep(p, g) for g in gws)


def best_xi(ids, proj, gw, key=None):
    """The highest-scoring legal eleven from `ids`. Every formation is tried."""
    key = key or (lambda p: _ep(p, gw))
    by_pos = {}
    for i in ids:
        p = proj.get(i)
        if p:
            by_pos.setdefault(p.get("pos"), []).append(p)
    for v in by_pos.values():
        v.sort(key=lambda p: -key(p))
    best = None
    for d in range(XI_MIN["DEF"], XI_MAX["DEF"] + 1):
        for m in range(XI_MIN["MID"], XI_MAX["MID"] + 1):
            for f in range(XI_MIN["FWD"], XI_MAX["FWD"] + 1):
                if 1 + d + m + f != 11:
                    continue
                shape = {"GKP": 1, "DEF": d, "MID": m, "FWD": f}
                if any(len(by_pos.get(k, [])) < n for k, n in shape.items()):
                    continue
                xi = [p for k, n in shape.items() for p in by_pos[k][:n]]
                tot = sum(key(p) for p in xi)
                if best is None or tot > best[0]:
                    best = (tot, xi, f"{d}-{m}-{f}")
    if best is None:
        return None
    tot, xi, shape = best
    return {"total": round(tot, 2), "formation": shape,
            "xi": [{"id": p["id"], "name": p.get("name"), "pos": p.get("pos"),
                    "team": p.get("team"), "price": p.get("price"),
                    "ep": round(_ep(p, gw), 2)} for p in xi]}


def template_xi(own, proj, gw, min_own=0.0):
    """The most-owned legal eleven in a field, by that field's ownership."""
    pool = [i for i, o in own.items() if o >= min_own and i in proj]
    xi = best_xi(pool, proj, gw, key=lambda p: own.get(p["id"], 0.0))
    if xi:
        for r in xi["xi"]:
            r["own"] = round(own.get(r["id"], 0.0), 3)
        xi["total"] = round(sum(r["ep"] for r in xi["xi"]), 2)
    return xi


def ceiling_xi(proj, gw, max_price=None):
    """The best eleven in the game, budget ignored. A yardstick, not a target."""
    ids = [i for i, p in proj.items()
           if (max_price is None or (p.get("price") or 0) <= max_price)]
    return best_xi(ids, proj, gw)


def edge(squad_ids, own, proj, gw, limit=12):
    """Expected ground gained and lost against a field, per the arithmetic above."""
    mine = best_xi(squad_ids, proj, gw)
    # A squad that cannot field a legal eleven — a partial list, a mid-build
    # wildcard draft — is still worth measuring. Falling through to an empty
    # set here would have reported every owned player as a hole.
    my_ids = ({r["id"] for r in mine["xi"]} if mine
              else {i for i in (squad_ids or ()) if i in proj})

    gains = []
    for i in my_ids:
        p = proj.get(i) or {}
        o = own.get(i, 0.0)
        gains.append({"id": i, "name": p.get("name"), "own": round(o, 3),
                      "ep": round(_ep(p, gw), 2),
                      "value": round(_ep(p, gw) * (1 - o), 2)})
    losses = []
    for i, o in own.items():
        if i in my_ids or i not in proj:
            continue
        p = proj[i]
        losses.append({"id": i, "name": p.get("name"), "pos": p.get("pos"),
                       "team": p.get("team"), "price": p.get("price"),
                       "own": round(o, 3), "ep": round(_ep(p, gw), 2),
                       "cost": round(_ep(p, gw) * o, 2)})
    gains.sort(key=lambda r: -r["value"])
    losses.sort(key=lambda r: -r["cost"])
    g, l = sum(r["value"] for r in gains), sum(r["cost"] for r in losses)
    return {"gain": round(g, 2), "cost": round(l, 2), "edge": round(g - l, 2),
            "my_leverage": gains[:limit], "template_holes": losses[:limit]}


def buy_leverage(squad_ids, own, proj, gws, bank=0.0, sell=None, limit=15):
    """Players not owned here, ranked by ground they would gain on the field.

    Over the whole horizon, not one week: a differential is a bet you hold.
    """
    have = set(squad_ids or ())
    sell = sell or {}
    budget = float(bank) + max((sell.get(i, 0.0) for i in have), default=0.0)
    rows = []
    for i, p in proj.items():
        if i in have or (p.get("start_prob") or 0) < 0.6:
            continue
        o = own.get(i, 0.0)
        ep = _ep_sum(p, gws)
        rows.append({"id": i, "name": p.get("name"), "pos": p.get("pos"),
                     "team": p.get("team"), "price": p.get("price"),
                     "own": round(o, 3), "ep_horizon": round(ep, 1),
                     "leverage": round(ep * (1 - o), 1),
                     "affordable": (p.get("price") or 0) <= budget})
    rows.sort(key=lambda r: -r["leverage"])
    return rows[:limit]


def build(squad_ids, lg, proj, gws, bank=0.0, sell=None, elite_own=None):
    gw = gws[0]
    league_own = (lg or {}).get("league_own") or {}
    pack_own = (lg or {}).get("pack_own") or {}
    mine = best_xi(squad_ids, proj, gw)
    out = {
        "gw": gw,
        "mine": mine,
        "template_league": template_xi(league_own, proj, gw),
        "template_pack": template_xi(pack_own, proj, gw),
        "ceiling": ceiling_xi(proj, gw),
        "edge_vs_league": edge(squad_ids, league_own, proj, gw),
        "edge_vs_pack": edge(squad_ids, pack_own, proj, gw),
        "buy_leverage": buy_leverage(squad_ids, pack_own, proj, gws, bank, sell),
    }
    if elite_own:
        out["template_elite"] = template_xi(elite_own, proj, gw)
    return out
