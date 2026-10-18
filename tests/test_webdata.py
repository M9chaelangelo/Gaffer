"""Tests for the dataset the page reads.

The app is a client of this file. If a field is dropped or renamed here the
page degrades silently — a column of dashes, a chart of nothing — so the shape
is asserted rather than assumed.
"""
import json

import pytest

from gaffer import webdata


def _proj(gws):
    out = {}
    for pid in range(1, 21):
        out[pid] = {
            "id": pid, "name": f"P{pid}", "team": "ARS" if pid % 2 else "LIV",
            "team_id": pid % 4, "pos": ["GKP", "DEF", "MID", "FWD"][pid % 4],
            "price": 4.5 + pid / 10,
            "ep": {g: 3.0 + (pid % 5) for g in gws},
            "opp": {g: "BOU(H)" for g in gws},
            "exp_min": 80, "status": "a", "news": "", "own": 12.5,
            "total_points": 40, "p60": 0.9,
            "defcon_prob": 0.3, "cs_prob": 0.25, "setpiece_bonus": 0.4,
        }
    return out


def _ctx(gws):
    proj = _proj(gws)
    prof = {pid: {"xg90": 0.3, "xa90": 0.2, "shot_quality": 0.02,
                  "attacking_share": 0.2, "defcon90": 6.0} for pid in proj}
    lg = {"league_own": {1: 0.5}, "pack_own": {1: 0.4},
          "points_behind": 12, "n_all": 42, "rows": [{"entry_name": "Top Team"}]}
    pf = {pid: {"net": 1000, "direction": "rise"} for pid in proj}
    squad_ids = list(range(1, 16))
    xi = [proj[i] for i in range(1, 12)]
    bench = [proj[i] for i in range(12, 16)]
    weeks = [{
        "gw": gws[0], "chip": None, "ep": 62.0, "hits": 0,
        "in": ["P17"], "out": ["P3"],
        "in_players": [proj[17]], "out_players": [proj[3]],
        "xi": xi, "bench": bench, "captain": proj[1],
    }]
    for g in gws[1:]:
        weeks.append({
            "gw": g, "chip": "Wildcard" if g == gws[1] else None, "ep": 70.0,
            "hits": 0, "in": [], "out": [], "in_players": [], "out_players": [],
            "xi": xi, "bench": bench, "captain": proj[1],
        })
    return proj, prof, lg, pf, squad_ids, weeks


def build(gws=(5, 6, 7)):
    gws = list(gws)
    proj, prof, lg, pf, squad_ids, weeks = _ctx(gws)
    return webdata.build(proj, prof, lg, {1: 0.6}, pf, gws, squad_ids, weeks,
                         "Fri 18 Sep 18:30", gws[0],
                         hit_verdict="Roll it.", clean_sheets=[])


def test_every_player_is_exported_not_just_the_transfer_pair():
    d = build()
    # Twenty in the projection; all twenty reach the page.
    assert len(d["players"]) == 20


def test_each_player_carries_the_full_card():
    d = build()
    p = d["players"][0]
    for field in webdata.CARD_FIELDS:
        assert field in p, f"{field} missing from the exported card"
    assert "id" in p and "ep" in p


def test_ep_is_a_list_aligned_to_gameweeks():
    d = build(gws=(5, 6, 7, 8))
    assert d["gameweeks"] == [5, 6, 7, 8]
    for p in d["players"]:
        assert len(p["ep"]) == 4


def test_squad_marks_the_eleven_the_bench_and_the_captain():
    d = build()
    assert len(d["squad"]) == 15
    assert sum(1 for s in d["squad"] if s["role"] == "xi") == 11
    assert sum(1 for s in d["squad"] if s["role"] == "bench") == 4
    assert sum(1 for s in d["squad"] if s["captain"]) == 1


def test_weeks_use_ids_so_two_players_can_share_a_name():
    d = build()
    w = d["weeks"][0]
    assert w["in"] == [17]
    assert w["out"] == [3]
    assert isinstance(w["captain"], int)
    assert len(w["xi"]) == 11


def test_metrics_are_declared_with_direction_and_format():
    d = build()
    assert d["metrics"]
    for m in d["metrics"]:
        assert set(m) == {"key", "label", "higher_better", "fmt"}
        assert isinstance(m["higher_better"], bool)


def test_every_metric_key_exists_on_a_player():
    """The compare view plots these. A typo here is a chart of nothing."""
    d = build()
    p = d["players"][0]
    for m in d["metrics"]:
        assert m["key"] in p, f"metric {m['key']} is not an exported field"


def test_teams_are_listed_for_the_filter():
    d = build()
    assert d["teams"] == ["ARS", "LIV"]


def test_league_context_survives():
    d = build()
    assert d["league"]["behind"] == 12
    assert d["league"]["teams"] == 42


def test_it_writes_valid_json(tmp_path):
    d = build()
    path = webdata.write(d, str(tmp_path))
    reloaded = json.loads(open(path, encoding="utf-8").read())
    assert reloaded["gw"] == 5
    assert len(reloaded["players"]) == 20


def test_no_ep_dict_leaks_into_the_payload(tmp_path):
    """`ep` on the projection is a dict keyed by gameweek. Shipping it for
    seven hundred players would bloat the file the phone downloads."""
    d = build()
    raw = json.dumps(d)
    assert '"opp"' not in raw
    assert '"team_id"' not in raw


def test_wildcard_block_is_exported_when_one_is_drafted():
    """The Wildcard tab is a client of this key. Exporting ids only is
    deliberate — the cards are already in `players`, and a second copy would
    add a third to a file that is downloaded on a phone."""
    gws = [5, 6, 7]
    proj, prof, lg, pf, squad_ids, weeks = _ctx(gws)
    wc = {
        "for_gw": 6, "squad": list(range(1, 16)), "cost": 99.4, "budget": 100.0,
        "in_bank": 0.6, "gws": [6, 7, 8], "score": 410.2, "optimal": True,
        "pool_size": 188, "assumes_no_transfers": True,
        "spend": {"GKP": 9.0, "DEF": 25.0, "MID": 40.0, "FWD": 25.4},
        "weeks": [{"gw": 6, "xi": list(range(1, 12)), "bench": [12, 13, 14, 15],
                   "formation": "3-5-2", "captain": 1, "chip": None, "ep": 71.0}],
        "swaps": [{"out": 3, "in": 17, "gap": 0.4}],
        "change": {"keep": [1], "sell": [2], "buy": [16]},
    }
    d = webdata.build(proj, prof, lg, {1: 0.6}, pf, gws, squad_ids, weeks,
                      "Fri 18 Sep 18:30", gws[0], hit_verdict="Roll it.",
                      clean_sheets=[], wildcard=wc)
    assert d["wildcard"]["for_gw"] == 6
    assert len(d["wildcard"]["squad"]) == 15
    assert d["wildcard"]["weeks"][0]["captain"] == 1
    # Ids, so every one of them has to resolve against the exported cards.
    ids = {p["id"] for p in d["players"]}
    for pid in d["wildcard"]["squad"] + d["wildcard"]["weeks"][0]["bench"]:
        assert pid in ids
    assert json.loads(json.dumps(d["wildcard"])) == wc


def test_wildcard_is_null_when_none_is_planned():
    d = build()
    assert d["wildcard"] is None


def test_boards_are_exported_for_the_page():
    """The Solio-shaped views are a client of this key. Ranking in Python
    rather than the browser keeps the rules testable — and keeps the page a
    renderer rather than a second, untested model."""
    d = build()
    b = d["boards"]
    for key in ("projected", "captains", "differentials", "goals", "assists",
                "defcon", "movers", "clean_sheets", "hauls"):
        assert key in b, key
    ids = {p["id"] for p in d["players"]}
    for board in ("projected", "captains", "differentials", "goals",
                  "assists", "defcon", "movers"):
        for row in b[board]:
            assert row["id"] in ids, f"{board} names a player not exported"
            assert "name" in row and "team" in row
    # The eleven the solve picked is what the haul distribution is over.
    assert b["hauls"] is None or b["hauls"]["n"] <= 11


def test_the_squad_distribution_is_exported():
    """The Lineup view's curve. Its mean has to agree with the week's own
    projected total, or the page states two different numbers for the same
    eleven."""
    d = build()
    dist = d["distribution"]
    assert dist is not None
    assert abs(sum(dist["pmf"]) - 1.0) < 1e-3
    assert set(dist["thresholds"]) == {"40", "60", "80"}
    assert dist["n"] == 11


def test_the_points_breakdown_and_diagnosis_are_exported():
    d = build()
    dna = d["dna"]
    assert abs(sum(r["points"] for r in dna["rows"]) - dna["total"]) < 0.5
    # The breakdown and the curve are two views of the same eleven.
    assert abs(dna["total"] - d["distribution"]["mean"]) < 0.5

    diag = d["diagnosis"]
    for key in ("weak_spots", "autosubs", "exposure", "template"):
        assert key in diag, key
    ids = {p["id"] for p in d["players"]}
    for row in diag["weak_spots"]:
        assert row["out"] in ids and row["in"] in ids


def test_the_fixture_ticker_is_exported_with_string_keys():
    """JSON has no integer keys, so the page would have to coerce them back.
    Exporting them as strings keeps that conversion in one place."""
    gws = [5, 6, 7]
    proj, prof, lg, pf, squad_ids, weeks = _ctx(gws)
    ticker = {5: [{"h": "ARS", "a": "LIV"}], 6: [{"h": "MCI", "a": "CHE"}]}
    d = webdata.build(proj, prof, lg, {1: 0.6}, pf, gws, squad_ids, weeks,
                      "Fri 18 Sep 18:30", gws[0], hit_verdict="Roll it.",
                      clean_sheets=[], ticker=ticker)
    assert set(d["ticker"]) == {"5", "6"}
    assert d["ticker"]["5"][0]["h"] == "ARS"


def test_no_ticker_is_an_empty_map_not_a_missing_key():
    assert build()["ticker"] == {}


def test_the_solver_status_survives_serialisation():
    """Both week writers use hand-maintained allow-lists, so a new field is
    computed, carried on the week, and then silently dropped on the way to
    the file. solver_status did exactly that: every week in plan.json read
    None as though the solver had never reported."""
    import inspect

    from gaffer import main as gmain
    from gaffer import webdata

    assert '"solver_status"' in inspect.getsource(webdata.build)
    src = inspect.getsource(gmain)
    i = src.find('"weeks": [{k: v for k, v in w.items()')
    assert i > 0, "plan.json week allow-list not found"
    assert "solver_status" in src[i:i + 400]


def test_the_league_block_carries_the_rivals_actual_teams():
    """P(finishing first) is a question about the other 41 teams, so the
    serialiser has to carry them. It did not: league.py captured `squads`
    and the allow-list in this block dropped them, so the simulator found
    an empty list and had nobody to beat. Fifth time in this project that a
    hand-maintained serialiser has lost a field the computation produced,
    which is why the block is now a function with a test rather than a dict
    literal buried inside a larger one."""
    from gaffer import webdata

    got = webdata.league_block({
        "points_behind": 132, "n_all": 42,
        "rows": [{"entry_name": "Ballon d'Orgu"}],
        "squads": [{"entry": 7, "name": "Rival", "rank": 1, "total": 452,
                    "picks": [{"element": 11, "multiplier": 2},
                              {"element": 12, "multiplier": 0}]}]})
    assert got["behind"] == 132 and got["teams"] == 42
    assert got["leader"] == "Ballon d'Orgu"
    assert got["squads"][0]["total"] == 452
    assert got["squads"][0]["picks"] == [[11, 2], [12, 0]], (
        "the multiplier is their lineup and their armband; without it a "
        "replay guesses at both")


def test_the_league_block_survives_an_empty_or_partial_snapshot():
    from gaffer import webdata

    assert webdata.league_block({})["squads"] == []
    assert webdata.league_block({})["leader"] is None
    assert webdata.league_block({"squads": [{"entry": 1}]})["squads"][0]["picks"] == []


def test_a_rival_squad_round_trips_from_the_file_into_the_simulator():
    """The join that was broken end to end: serialise, then score."""
    import numpy as np

    from gaffer import position, webdata

    block = webdata.league_block({
        "squads": [{"entry": 1, "picks": [{"element": 5, "multiplier": 2},
                                          {"element": 6, "multiplier": 1},
                                          {"element": 7, "multiplier": 0}]}]})
    squads = [{"picks": [{"element": e, "multiplier": m}
                         for e, m in q["picks"]]} for q in block["squads"]]
    samples = np.array([[3.0, 4.0, 9.0]])
    got, missing = position.squad_scores(squads, samples, {"5": 0, "6": 1, "7": 2})
    assert got[0, 0] == 3.0 * 2 + 4.0      # captain doubled, bench ignored
    assert missing == 0
