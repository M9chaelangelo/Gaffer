"""The template, the ceiling, and the arithmetic of beating a mini-league."""
from gaffer import template


def P(i, name, pos, ep, price=5.0, start=0.9):
    return {"id": i, "name": name, "pos": pos, "price": price,
            "ep": {7: ep, 8: ep}, "start_prob": start, "team": "AAA"}


def pool():
    d = {1: P(1, "GK1", "GKP", 4.0), 2: P(2, "GK2", "GKP", 3.0)}
    for i in range(3, 9):
        d[i] = P(i, f"D{i}", "DEF", 6.0 - (i - 3) * 0.5)
    for i in range(9, 15):
        d[i] = P(i, f"M{i}", "MID", 7.0 - (i - 9) * 0.5)
    for i in range(15, 19):
        d[i] = P(i, f"F{i}", "FWD", 8.0 - (i - 15) * 0.5)
    return d


def test_best_xi_tries_every_formation_not_just_a_greedy_fill():
    out = template.best_xi(list(pool()), pool(), 7)
    shape = out["formation"]
    assert len(out["xi"]) == 11
    d, m, f = (int(x) for x in shape.split("-"))
    assert 3 <= d <= 5 and 2 <= m <= 5 and 1 <= f <= 3 and 1 + d + m + f == 11


def test_the_template_is_the_most_owned_eleven_not_the_best_one():
    """These differ, and the difference is the whole point of the module."""
    pr = pool()
    own = {i: 0.95 for i in (1, 3, 4, 5, 9, 10, 11, 12, 15, 16, 17)}
    own.update({i: 0.01 for i in pr if i not in own})
    t = template.template_xi(own, pr, 7)
    assert {r["id"] for r in t["xi"]} == set(k for k, v in own.items() if v > 0.5)
    best = template.best_xi(list(pr), pr, 7)
    assert t["total"] <= best["total"], "the template never beats the ceiling"


def test_owning_what_the_field_owns_gains_almost_nothing():
    """A player everybody has is worth ~0 in a league race, whatever he scores."""
    pr = {1: P(1, "Everyone", "MID", 10.0)}
    hot = template.edge([1], {1: 1.0}, pr, 7)
    cold = template.edge([1], {1: 0.0}, pr, 7)
    assert hot["gain"] == 0.0
    assert cold["gain"] == 10.0


def test_missing_what_the_field_owns_is_what_costs_you_ground():
    pr = {1: P(1, "Mine", "MID", 5.0), 2: P(2, "Theirs", "MID", 10.0)}
    e = template.edge([1], {1: 0.0, 2: 0.9}, pr, 7)
    assert e["gain"] == 5.0
    assert e["cost"] == 9.0          # 10.0 x 0.9
    assert e["edge"] == -4.0
    assert e["template_holes"][0]["name"] == "Theirs"


def test_buy_leverage_prefers_a_differential_over_a_better_owned_player():
    """6.0 nobody owns beats 9.0 everybody owns, when the goal is a league."""
    pr = {1: P(1, "Owned", "MID", 9.0), 2: P(2, "Differential", "MID", 6.0)}
    rows = template.buy_leverage([], {1: 0.95, 2: 0.02}, pr, [7, 8])
    assert rows[0]["name"] == "Differential"
    assert rows[0]["leverage"] > rows[1]["leverage"]


def test_buy_leverage_skips_players_already_owned_and_the_unlikely_to_start():
    pr = {1: P(1, "Have", "MID", 9.0), 2: P(2, "Benchwarmer", "MID", 9.0, start=0.2)}
    assert template.buy_leverage([1], {}, pr, [7]) == []


def test_affordability_is_reported_not_silently_filtered():
    pr = {1: P(1, "Expensive", "MID", 9.0, price=14.0)}
    rows = template.buy_leverage([], {}, pr, [7], bank=0.5)
    assert rows[0]["affordable"] is False
