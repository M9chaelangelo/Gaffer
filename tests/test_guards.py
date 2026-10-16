"""Invariants. Each test is a bug that actually shipped and was believed."""
from gaffer import guards


def P(pid, name, pos, ep, status="a"):
    return {"id": pid, "name": name, "pos": pos, "ep": {7: ep}, "status": status}


def legal_xi(ep=3.0):
    return ([P(1, "G", "GKP", ep)]
            + [P(i, f"D{i}", "DEF", ep) for i in (2, 3, 4)]
            + [P(i, f"M{i}", "MID", ep) for i in (5, 6, 7, 8)]
            + [P(i, f"F{i}", "FWD", ep) for i in (9, 10, 11)])


def test_a_spent_chip_is_never_planned():
    """GW8 Triple Captain at +9.27, for a chip played in GW3."""
    weeks = [{"gw": 8, "chip": "Triple Captain"}]
    v = guards.check_chips(weeks, ["tc", "wc"])
    assert v and v[0]["code"] == "chip-spent" and v[0]["fatal"]


def test_an_unspent_chip_is_fine():
    assert guards.check_chips([{"gw": 8, "chip": "Bench Boost"}], ["tc", "wc"]) == []


def test_a_bench_boost_week_must_count_the_bench():
    """Reported 60.2 against a true 82.5 — the chip week read as the worst."""
    xi = legal_xi(5.0)
    bench = [P(12, "B1", "DEF", 4.0), P(13, "B2", "MID", 4.0),
             P(14, "B3", "FWD", 4.0), P(15, "B4", "GKP", 4.0)]
    week = {"gw": 7, "chip": "Bench Boost", "xi": xi, "bench": bench,
            "captain": xi[0], "ep": 60.0}          # eleven only: wrong
    v = guards.check_week_ep([week])
    assert v and v[0]["code"] == "week-ep" and v[0]["fatal"]
    week["ep"] = 11 * 5.0 + 4 * 4.0 + 5.0          # all fifteen plus captain
    assert guards.check_week_ep([week]) == []


def test_a_non_boosted_week_counts_only_the_eleven():
    xi = legal_xi(5.0)
    week = {"gw": 7, "chip": None, "xi": xi,
            "bench": [P(12, "B1", "DEF", 9.9)], "captain": xi[0],
            "ep": 11 * 5.0 + 5.0}
    assert [v for v in guards.check_week_ep([week])] == []


def test_an_injured_player_is_never_started():
    xi = legal_xi()
    xi[5] = P(6, "Crocked", "MID", 2.6, status="i")
    v = guards.check_lineup([{"gw": 7, "xi": xi, "captain": xi[0]}])
    assert any(x["code"] == "unfit-starter" and x["fatal"] for x in v)


def test_an_illegal_formation_is_caught():
    xi = legal_xi()[:-1] + [P(11, "D5", "DEF", 3.0)]   # 4 DEF, 4 MID, 2 FWD -> ok
    xi = [P(1, "G", "GKP", 3)] + [P(i, f"M{i}", "MID", 3) for i in range(2, 12)]
    v = guards.check_lineup([{"gw": 7, "xi": xi, "captain": xi[0]}])
    assert any(x["code"] == "formation" for x in v)


def test_a_far_better_player_on_the_bench_is_flagged():
    """Groß 7.42 and B.Fernandes 5.92 benched to start Schade on 2.64."""
    xi = legal_xi(2.6)
    v = guards.check_lineup([{"gw": 7, "xi": xi, "captain": xi[0],
                              "bench": [P(12, "Groß", "MID", 7.42)]}])
    assert any(x["code"] == "bench-better" for x in v)


def test_a_squad_pin_may_not_outlive_its_gameweek():
    """Pinned for GW6, solved for GW7 — three days out of date."""
    cfg = {"current_squad": ["a"] * 15, "squad_pinned_for": 6}
    v = guards.check_squad_pin(cfg, 7)
    assert v and v[0]["code"] == "pin-stale" and v[0]["fatal"]
    assert guards.check_squad_pin({"current_squad": ["a"] * 15,
                                   "squad_pinned_for": 7}, 7) == []


def test_an_undated_pin_is_a_warning_because_nothing_can_check_it():
    v = guards.check_squad_pin({"current_squad": ["a"] * 15}, 7)
    assert v and v[0]["code"] == "pin-undated" and not v[0]["fatal"]


def test_no_pin_no_complaint():
    assert guards.check_squad_pin({}, 7) == []
