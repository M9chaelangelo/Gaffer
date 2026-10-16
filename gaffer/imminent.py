"""Who is due a return — the overdue tables, from FPL's own Opta feed.

Fantasy Football Scout's "Goals Imminent" and "Assists Imminent" tables are
built on Opta data. FPL's public API now carries the same Opta expected goals
and expected assists — per player on `bootstrap-static`, and per gameweek on
`event/{gw}/live` — so the signal is reproducible without the subscription.
This is not a copy of their table; it is the same question asked of the same
underlying numbers.

Two different things get conflated under "imminent", and conflating them is
how the idea gets a bad name. They are kept apart here:

  - **Debt.** Season expected goals above goals actually scored. A finishing
    deficit. On its own this is the weaker half: finishing variance does
    regress, but slowly, and some players genuinely finish badly — a striker
    two goals under his xG all season may simply be a poor finisher, and
    "he's due" has been the last word of many bad transfers.

  - **Underlying form.** Expected goal involvement per 90 over the last few
    gameweeks, whatever the returns. This is the stronger half: a player
    getting into good positions *now* tends to score soon, and it does not
    depend on believing in regression at all.

A player who shows both — generating chances right now AND behind on the
season — is the real candidate, and `both` marks him. Ranking is by recent
form, with debt as the tie-break, deliberately: form is the half that carries
its own weight.
"""
from __future__ import annotations

POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
WINDOW = 4            # gameweeks of "recent"; roughly a month of football
MIN_MINUTES = 270     # three full games, below which a per-90 is not a rate
MIN_RECENT_MINUTES = 135


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def recent(live_by_gw, window=WINDOW):
    """Per-player xG, xA and minutes over the last `window` gameweeks.

    `live_by_gw` is {gw: payload} from `event/{gw}/live`. Only the most recent
    `window` gameweeks present are used, so a caller may pass more.
    """
    gws = sorted(live_by_gw)[-window:]
    agg = {}
    for g in gws:
        for el in (live_by_gw[g] or {}).get("elements", []):
            s = el.get("stats") or {}
            a = agg.setdefault(el["id"], {"xg": 0.0, "xa": 0.0, "mins": 0,
                                          "goals": 0, "assists": 0})
            a["xg"] += _f(s.get("expected_goals"))
            a["xa"] += _f(s.get("expected_assists"))
            a["mins"] += int(s.get("minutes") or 0)
            a["goals"] += int(s.get("goals_scored") or 0)
            a["assists"] += int(s.get("assists") or 0)
    return agg, gws


def build(boot, live_by_gw, proj=None, window=WINDOW, top=20):
    """The two tables, plus the raw rows they are drawn from."""
    agg, gws = recent(live_by_gw, window)
    teams = {t["id"]: t["short_name"] for t in boot["teams"]}
    rows = []
    for e in boot["elements"]:
        mins = int(e.get("minutes") or 0)
        if mins < MIN_MINUTES or e.get("status") in ("u",):
            continue
        r = agg.get(e["id"], {"xg": 0.0, "xa": 0.0, "mins": 0})
        rm = r["mins"]
        xg, xa = _f(e.get("expected_goals")), _f(e.get("expected_assists"))
        goals, assists = int(e.get("goals_scored") or 0), int(e.get("assists") or 0)
        per90 = lambda v, m: round(v / (m / 90.0), 3) if m >= 90 else None
        row = {
            "id": e["id"],
            "name": e["web_name"],
            "team": teams.get(e["team"], "?"),
            "pos": POS.get(e["element_type"], "?"),
            "price": e["now_cost"] / 10.0,
            "minutes": mins,
            "recent_minutes": rm,
            "xg": round(xg, 2), "goals": goals,
            "xa": round(xa, 2), "assists": assists,
            # Debt: how far the finishing (or the team-mates' finishing, for
            # assists) has fallen short of the chances created.
            "goal_debt": round(xg - goals, 2),
            "assist_debt": round(xa - assists, 2),
            "season_xgi90": per90(xg + xa, mins),
            "recent_xg90": per90(r["xg"], rm),
            "recent_xa90": per90(r["xa"], rm),
            "recent_xgi90": per90(r["xg"] + r["xa"], rm),
            "own": round(_f(e.get("selected_by_percent")), 1),
            "status": e.get("status"),
            "news": e.get("news") or "",
        }
        if proj and e["id"] in proj:
            # Same trap as crowd.py: proj rows key `ep` by gameweek and have
            # no `ep_next`, so this column printed 0.00 for every player.
            ep = (proj[e["id"]] or {}).get("ep")
            row["ep_next"] = (round(ep[min(ep)], 2) if isinstance(ep, dict) and ep
                              else proj[e["id"]].get("ep_next"))
        rows.append(row)

    def rank(kind):
        f90, debt = f"recent_{kind}90", f"{'goal' if kind=='xg' else 'assist'}_debt"
        live = [r for r in rows
                if r["recent_minutes"] >= MIN_RECENT_MINUTES
                and r[f90] is not None and r["status"] == "a"]
        live.sort(key=lambda r: (-r[f90], -r[debt]))
        out = []
        for r in live[:top]:
            q = dict(r)
            q["both"] = r[debt] > 0.5      # creating now AND behind on the season
            out.append(q)
        return out

    return {
        "window": gws,
        "note": ("Opta expected goals and assists from FPL's own feed. Ranked "
                 "on recent underlying form; `both` marks a player who is also "
                 "behind his season expected goals, which is the stronger case."),
        "goals_imminent": rank("xg"),
        "assists_imminent": rank("xa"),
        "rows": rows,
    }
