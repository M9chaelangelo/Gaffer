"""Everything the page needs, as one JSON file.

The engine already computes a twenty-five field card for every player — shot
quality, attacking share, DefCon probability, four kinds of ownership, price
direction. Until now it exported that for the two players in a transfer and
threw the other seven hundred away, which is why the page could only ever show
you the move it had already decided on.

This writes all of it. The page is then a client of the data rather than a
rendering of one conclusion: you can sort, filter, and compare any two players
without the solver having to agree that they matter.

One file, no API, no server — it has to work on GitHub Pages, which serves
static files and nothing else.
"""
import json
import os

from . import analysis, distribution, explain, leaders

# Fields kept per player. Deliberately explicit: an accidental `**p` here would
# ship the whole projection dict, and `ep` alone is a dict per gameweek for
# seven hundred players.
CARD_FIELDS = (
    "name", "team", "pos", "price", "ep_next", "ep_horizon", "fixtures",
    "start_prob", "ceiling", "exp_minutes", "xg90", "xa90", "shot_quality",
    "attacking_share", "defcon90", "defcon_prob", "cs_prob", "setpiece",
    "bonus_exp", "bps90",
    "own_league", "own_pack", "own_elite", "own_overall",
    "price_direction", "net_transfers", "status", "news",
)

# Metrics the compare view plots. `higher_better` False means a lower number is
# the better one, so the bar can be normalised without lying about which player
# is ahead.
METRICS = [
    {"key": "ep_next", "label": "Projected points", "higher_better": True, "fmt": "num"},
    {"key": "ep_horizon", "label": "Points over horizon", "higher_better": True, "fmt": "num"},
    {"key": "start_prob", "label": "Starts", "higher_better": True, "fmt": "pct"},
    # The tail. Ranks players differently from expected points, which is the
    # whole reason it is here rather than folded into one number.
    {"key": "ceiling", "label": "Hauls (10+ pts)", "higher_better": True, "fmt": "pct"},
    {"key": "exp_minutes", "label": "Expected minutes", "higher_better": True, "fmt": "int"},
    {"key": "xg90", "label": "xG per 90", "higher_better": True, "fmt": "num3"},
    {"key": "xa90", "label": "xA per 90", "higher_better": True, "fmt": "num3"},
    {"key": "shot_quality", "label": "Shot quality", "higher_better": True, "fmt": "num4"},
    {"key": "attacking_share", "label": "Share of team attack", "higher_better": True, "fmt": "pct"},
    {"key": "defcon90", "label": "Defensive actions per 90", "higher_better": True, "fmt": "num"},
    {"key": "defcon_prob", "label": "Hits DefCon", "higher_better": True, "fmt": "pct"},
    {"key": "cs_prob", "label": "Clean sheet", "higher_better": True, "fmt": "pct"},
    {"key": "setpiece", "label": "Set-piece value", "higher_better": True, "fmt": "num"},
    {"key": "bonus_exp", "label": "Expected bonus", "higher_better": True, "fmt": "num"},
    {"key": "bps90", "label": "BPS per 90", "higher_better": True, "fmt": "num"},
    {"key": "price", "label": "Price", "higher_better": False, "fmt": "money"},
    {"key": "own_overall", "label": "Owned overall", "higher_better": True, "fmt": "pct"},
    {"key": "own_league", "label": "Owned in your league", "higher_better": True, "fmt": "pct"},
    {"key": "own_elite", "label": "Owned by elite", "higher_better": True, "fmt": "pct"},
]


def build(proj, prof, lg, elite_own, price_fc, gws, squad_ids, weeks,
          deadline, gw, planner=None, hit_verdict=None, clean_sheets=None,
          wildcard=None, team_form=None, strengths=None, ticker=None,
          imminent=None, guards=None, ledger=None, crowd=None,
          template=None, matchup=None):
    """Assemble the page's dataset."""
    cards = {}
    for pid in proj:
        card = explain.player_card(pid, proj, prof, lg["league_own"],
                                   lg["pack_own"], elite_own, price_fc, gws)
        row = {k: card.get(k) for k in CARD_FIELDS}
        row["id"] = pid
        # Per-gameweek points drive the projection chart and the planner. Kept
        # as a plain list aligned to `gameweeks` so the JSON stays small.
        row["ep"] = [round(proj[pid]["ep"][g], 2) for g in gws]
        # The rebuild looks further ahead than the weekly solve, so its weeks
        # run off the end of `ep`. Without this the wildcard pitch shows the
        # same number in every week and the week switcher looks broken.
        if wildcard:
            row["ep_wc"] = [round(proj[pid]["ep"].get(g, 0.0), 2)
                            for g in wildcard["gws"]]
        cards[pid] = row

    # The category boards Solio organises its front page around: who to
    # captain, where the leverage is, who scores, who keeps it out. Computed
    # here rather than in the browser so the page stays a renderer and the
    # ranking rules stay testable.
    now = weeks[0] if weeks else None
    squad = []
    if now:
        xi_ids = [p["id"] for p in now["xi"]]
        bench_ids = [p["id"] for p in now["bench"]]
        for pid in xi_ids + bench_ids:
            squad.append({
                "id": pid,
                "role": "xi" if pid in xi_ids else "bench",
                # Bench order is the order the game substitutes them in.
                "slot": (xi_ids + bench_ids).index(pid),
                "captain": pid == now["captain"]["id"],
            })

    return {
        "gw": gw,
        "deadline": deadline,
        "gameweeks": list(gws),
        "players": list(cards.values()),
        "metrics": METRICS,
        "squad": squad,
        "squad_ids": list(squad_ids),
        "teams": sorted({p["team"] for p in cards.values()}),
        "weeks": [
            {
                "gw": w["gw"],
                "chip": w["chip"],
                "ep": w["ep"],
                "hits": w["hits"],
                "in": [p["id"] for p in w.get("in_players", [])],
                "out": [p["id"] for p in w.get("out_players", [])],
                "xi": [p["id"] for p in w["xi"]],
                "bench": [p["id"] for p in w["bench"]],
                "captain": w["captain"]["id"],
            }
            for w in weeks
        ],
        "hit_verdict": hit_verdict,
        # The rebuild, when one is planned. Player ids only — every card is
        # already in `players`, and shipping them twice would add a third to
        # the file for nothing.
        "wildcard": wildcard,
        # Attack and defence over the last few matches, opponent-adjusted.
        # The one number on the page a human can check against what they
        # actually watched on Saturday.
        "team_form": team_form or [],
        # Attack and defence per club, form-blended and centred on one. The
        # scatter every fixture argument ends up being about.
        "strengths": strengths or {},
        # Every match in the horizon, by gameweek — the ticker. Keys are
        # strings because JSON has no integer keys and the page would have to
        # coerce them back anyway.
        "ticker": {str(g): v for g, v in (ticker or {}).items()},
        # What the eleven might actually score, not just the mean of it.
        # Exact convolution over the per-player component distributions.
        "distribution": (distribution.squad_points(
            list(cards.values()),
            [p["id"] for p in now["xi"]],
            captain_id=now["captain"]["id"],
            bench_ids=[p["id"] for p in now["bench"]],
            chip={"Bench Boost": "bb", "Triple Captain": "tc"}.get(now["chip"]))
            if now else None),
        # Where the projected points come from, split into the buckets that
        # actually produce them.
        "dna": (distribution.points_dna(
            list(cards.values()),
            [p["id"] for p in now["xi"]],
            captain_id=now["captain"]["id"],
            bench_ids=[p["id"] for p in now["bench"]],
            chip={"Bench Boost": "bb", "Triple Captain": "tc"}.get(now["chip"]))
            if now else None),
        # What is wrong with the eleven: who is being out-scored, how exposed
        # the bench is, which clubs the week turns on.
        "imminent": imminent,
        "crowd": crowd,
        "template": template,
        "matchup": matchup,
        "guards": guards,
        "ledger": ledger,
        "diagnosis": (analysis.build(
            list(cards.values()),
            [p["id"] for p in now["xi"]],
            bench_ids=[p["id"] for p in now["bench"]],
            captain_id=now["captain"]["id"],
            squad_ids=list(squad_ids), strengths=strengths) if now else None),
        "boards": leaders.build(
            list(cards.values()), gw,
            xi_ids=[p["id"] for p in now["xi"]] if now else None,
            captain_id=now["captain"]["id"] if now else None,
            clean_sheets=clean_sheets or []),
        "clean_sheets": clean_sheets or [],
        "planner": planner,
        "league": {
            "behind": lg.get("points_behind"),
            "teams": lg.get("n_all"),
            "leader": (lg["rows"][0]["entry_name"] if lg.get("rows") else None),
        },
    }


def write(data, out_dir):
    path = os.path.join(out_dir, "data.json")
    with open(path, "w", encoding="utf-8") as f:
        # Separators trim about 15% off the file; it is downloaded on a phone.
        json.dump(data, f, separators=(",", ":"))
    return path
