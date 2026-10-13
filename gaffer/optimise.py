"""Multi-period transfer and chip planner.

Solves the whole horizon at once: which 15 to hold each week, who starts, who
captains, when to spend a transfer, when a hit pays, and when to burn a chip.
Solving week by week is what makes people take hits they regret.
"""
import pulp

from . import solver, timing

SQUAD = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}


def selling_price(purchase, now):
    """FPL's rule: you keep half of any rise, rounded down to 0.1m."""
    if now <= purchase:
        return now
    return purchase + int((now - purchase) * 10 // 2) / 10


def rivalry_weight(pid, pack_own, rivalry):
    """Reward players the pack above you does not own.

    A player owned by everyone you are chasing cannot gain you places; he can
    only stop you losing them. This tilts the solve toward leverage without
    ignoring expected points."""
    return 1.0 + rivalry * (1.0 - pack_own.get(pid, 0.0))


def plan(proj, squad_ids, sell, bank, gws, free_transfers, cfg,
         pack_own=None, chips_available=None,
         force_no_transfer_gws=(), force_wc_gw=None, calendar=None,
         force_chip=None):
    # A chip already played is not a decision. The sweep learned this; the
    # solver had not, and went on planning a Wildcard in GW8 that was spent in
    # GW6 — which quietly costs you every week the solver reserves for it.
    # Defaults to what config says is left rather than to all four.
    if chips_available is None:
        spent = {str(x).strip().lower() for x in (cfg.get("chips_used") or ())}
        chips_available = tuple(c for c in ("wc", "tc", "bb", "fh")
                                if c not in spent)
    pack_own = pack_own or {}
    rivalry = cfg.get("rivalry", 0.0)
    decay = cfg.get("decay", 0.87)
    bw = cfg.get("bench_weights", [0.16, 0.11, 0.06, 0.02])
    outfield_w = sum(bw[:3]) / 3          # three outfield bench slots
    gk_w = bw[3]                          # the second keeper almost never plays

    def bench_weight(p):
        return gk_w if p["pos"] == "GKP" else outfield_w

    cand = [p for p in proj.values()
            if p["exp_min"] >= 55 and p["status"] == "a" and p["own"] >= 0.6]
    # Cap the pool. Every extra player multiplies the model by eight binaries per
    # gameweek, and CBC slows down long before the extra options are worth it.
    cand.sort(key=lambda p: -sum(p["ep"][g] for g in gws))
    keep = cand[:cfg.get("pool_size", 130)]
    pool = list({p["id"]: p for p in keep + [proj[i] for i in squad_ids]}.values())
    P = {p["id"]: p for p in pool}
    g0 = gws[0]

    m = pulp.LpProblem("gaffer", pulp.LpMaximize)
    x = {(i, g): m.add_variable(f"x_{i}_{g}", cat="Binary") for i in P for g in gws}
    y = {(i, g): m.add_variable(f"y_{i}_{g}", cat="Binary") for i in P for g in gws}
    c = {(i, g): m.add_variable(f"c_{i}_{g}", cat="Binary") for i in P for g in gws}
    tin = {(i, g): m.add_variable(f"i_{i}_{g}", cat="Binary") for i in P for g in gws}
    tout = {(i, g): m.add_variable(f"o_{i}_{g}", cat="Binary") for i in P for g in gws}
    hits = {g: m.add_variable(f"h_{g}", lowBound=0, cat="Integer") for g in gws}
    # Free transfers: one a week, unused ones stack, hard ceiling of five.
    ft = {g: m.add_variable(f"ft_{g}", lowBound=0, upBound=5) for g in gws}
    used = {g: m.add_variable(f"u_{g}", lowBound=0, cat="Integer") for g in gws}
    wc = {g: m.add_variable(f"wc_{g}", cat="Binary") for g in gws}
    tc = {g: m.add_variable(f"tc_{g}", cat="Binary") for g in gws}
    bb = {g: m.add_variable(f"bb_{g}", cat="Binary") for g in gws}

    # Chip effects must attach to the players actually involved. A flat credit
    # makes a chip look free of any squad decision, and the solver then just
    # plays it in the earliest week the decay factor still values.
    boost = {(i, g): m.add_variable(f"bst_{i}_{g}", cat="Binary")
             for i in P for g in gws}    # on the bench AND bench boost active
    trip = {(i, g): m.add_variable(f"trp_{i}_{g}", cat="Binary")
            for i in P for g in gws}     # is captain AND triple captain active

    obj = []
    for gi, g in enumerate(gws):
        d = decay ** gi
        for i, p in P.items():
            w = rivalry_weight(i, pack_own, rivalry)
            ep = p["ep"][g] * w * d
            bwt = bench_weight(p)
            obj.append(ep * y[(i, g)])                       # starter
            obj.append(ep * c[(i, g)])                       # captain doubles
            obj.append(ep * bwt * (x[(i, g)] - y[(i, g)]))   # bench, discounted
            # Bench Boost pays the bench the rest of the way to full value.
            obj.append(ep * (1 - bwt) * boost[(i, g)])
            # Triple Captain adds a third multiple of whoever you actually
            # captain that week, not of the best player in the game.
            obj.append(ep * rivalry_weight(i, pack_own, rivalry) * trip[(i, g)])
            # …and it is chosen on the tail, not the mean. Tripling a player
            # only pays when he hauls: a 6.0 that goes double-digit one week in
            # five beats a 6.4 that never does, and expected points cannot tell
            # those two apart. The learned ceiling (P(10+), AUC 0.849) tilts the
            # armband toward the player who can actually spike.
            #
            # The tilt is deliberately small. It breaks ties between comparable
            # captains rather than overruling the projection — a ceiling model
            # that outvoted expected points would hand the chip to a cheap
            # lottery ticket, which is the opposite of the point.
            haul = p.get("ceiling", {}).get(g)
            if haul is not None:
                obj.append(cfg.get("ceiling_weight", 6.0) * haul * d
                           * trip[(i, g)])
        obj.append(-cfg.get("hit_cost", 4) * d * hits[g])
        # Friction on every move, so the solver stops selling a player and
        # buying him back two weeks later to chase a rounding error.
        obj.append(-cfg.get("min_transfer_gain", 1.5) * d * used[g])
    for g in gws:
        for i in P:
            m += boost[(i, g)] <= x[(i, g)] - y[(i, g)]
            m += boost[(i, g)] <= bb[g]
            m += boost[(i, g)] >= (x[(i, g)] - y[(i, g)]) + bb[g] - 1
            m += trip[(i, g)] <= c[(i, g)]
            m += trip[(i, g)] <= tc[g]
            m += trip[(i, g)] >= c[(i, g)] + tc[g] - 1
    for gi, g in enumerate(gws):
        d = decay ** gi
        # Credit a later wildcard for the information you will have by then.
        # Without this the solver always prefers acting immediately, because it
        # scores every future week with today's projections.
        if calendar is not None and cfg.get("info_value", 0):
            iv = cfg.get("info_value", 12.0)
            bk = cfg.get("break_bonus", 4.0)
            obj.append(timing.wildcard_bonus(g, calendar, iv, bk) * wc[g])
            # A Bench Boost or Triple Captain played later is also better
            # informed — you know who your bench actually is, and who is fit.
            # Scaled down because they commit far less than a full rebuild.
            obj.append(timing.wildcard_bonus(g, calendar, iv * 0.35, bk * 0.5)
                       * (bb[g] + tc[g]))
    m += pulp.lpSum(obj)

    for g in gws:
        m += pulp.lpSum(x[(i, g)] for i in P) == 15
        m += pulp.lpSum(y[(i, g)] for i in P) == 11
        m += pulp.lpSum(c[(i, g)] for i in P) == 1
        for pos, n in SQUAD.items():
            m += pulp.lpSum(x[(i, g)] for i in P if P[i]["pos"] == pos) == n
            s = pulp.lpSum(y[(i, g)] for i in P if P[i]["pos"] == pos)
            m += s >= XI_MIN[pos]
            m += s <= XI_MAX[pos]
        for t in {p["team_id"] for p in P.values()}:
            m += pulp.lpSum(x[(i, g)] for i in P if P[i]["team_id"] == t) <= 3
        for i in P:
            m += y[(i, g)] <= x[(i, g)]
            m += c[(i, g)] <= y[(i, g)]
            # An injured or suspended player cannot be picked. The solver is
            # then free to decide whether he is worth a squad slot at all.
            if P[i]["status"] in ("i", "s", "u", "n") and P[i]["ep"][g] <= 0.05:
                m += y[(i, g)] == 0
        m += pulp.lpSum(x[(i, g)] * P[i]["price"] for i in P) \
             <= bank + sum(sell.get(i, P[i]["price"]) for i in squad_ids) \
             + pulp.lpSum(0 for i in P)

    prev = {i: (1 if i in squad_ids else 0) for i in P}
    for gi, g in enumerate(gws):
        for i in P:
            if gi == 0:
                m += x[(i, g)] == prev[i] + tin[(i, g)] - tout[(i, g)]
            else:
                m += x[(i, g)] == x[(i, gws[gi - 1])] + tin[(i, g)] - tout[(i, g)]
            m += tin[(i, g)] + tout[(i, g)] <= 1
        moves = pulp.lpSum(tin[(i, g)] for i in P)
        if g in force_no_transfer_gws:
            m += moves == 0
        # A wildcard week burns no free transfers.
        m += used[g] >= moves - 15 * wc[g]
        m += moves <= 15 * wc[g] + 20 * (1 - wc[g])
        if gi == 0:
            m += ft[g] == free_transfers
        else:
            pg = gws[gi - 1]
            # Normal week: ft grows by one, minus whatever you spent.
            # Wildcard/Free Hit week: FPL retains your saved transfers as they
            # were, so no +1 is added on top.
            m += ft[g] <= ft[pg] - used[pg] + 1 - wc[pg]
        m += hits[g] >= moves - ft[g] - 15 * wc[g]
        m += wc[g] + tc[g] + bb[g] <= 1
    for i in P:
        m += pulp.lpSum(tin[(i, g)] for g in gws) <= 1
        if i in squad_ids and not cfg.get("allow_rebuy", False):
            m += pulp.lpSum(tin[(i, g)] for g in gws) == 0

    if force_wc_gw is not None:
        m += wc[force_wc_gw] == 1
    if force_chip is not None:
        which, when = force_chip
        m += {"wc": wc, "tc": tc, "bb": bb}[which][when] == 1
    m += pulp.lpSum(wc.values()) <= (1 if "wc" in chips_available else 0)
    m += pulp.lpSum(tc.values()) <= (1 if "tc" in chips_available else 0)
    m += pulp.lpSum(bb.values()) <= (1 if "bb" in chips_available else 0)
    # One chip inside a five-week window. Stacking three is a modelling artefact,
    # not a plan.
    m += pulp.lpSum(wc.values()) + pulp.lpSum(tc.values()) \
         + pulp.lpSum(bb.values()) <= cfg.get("max_chips_in_horizon", 1)

    solver.solve(m, cfg.get("time_limit", 240))

    out = []
    for g in gws:
        xi = sorted([P[i] for i in P if y[(i, g)].value() > 0.5],
                    key=lambda p: (-p["ep"][g]))
        cap = next(P[i] for i in P if c[(i, g)].value() > 0.5)
        # Carry the players themselves, not their names. Two players can share
        # a web name (there are two Palmers), and pairing a wildcard's thirteen
        # moves by list order produces nonsense like a midfielder "replacing" a
        # goalkeeper.
        ins = [P[i] for i in P if tin[(i, g)].value() > 0.5]
        outs = [P[i] for i in P if tout[(i, g)].value() > 0.5]
        out.append({
            "gw": g,
            "in": [p["name"] for p in ins],
            "out": [p["name"] for p in outs],
            "in_players": ins,
            "out_players": outs,
            "hits": int(round(hits[g].value() or 0)),
            "chip": ("Wildcard" if wc[g].value() > 0.5 else
                     "Triple Captain" if tc[g].value() > 0.5 else
                     "Bench Boost" if bb[g].value() > 0.5 else None),
            "xi": xi,
            "bench": [P[i] for i in P if x[(i, g)].value() > 0.5
                      and y[(i, g)].value() < 0.5],
            "captain": cap,
            "ep": round(sum(p["ep"][g] for p in xi) + cap["ep"][g], 1),
        })
    return out


def value_of_hit(proj, squad_ids, sell, bank, gws, free_transfers, cfg,
                 pack_own, base_weeks, calendar=None):
    """Is a hit worth it? Re-solve with hits priced out of reach and compare.

    Takes the plan you already solved as the permissive case, so this costs one
    extra solve rather than two."""
    # Both sides must get the same solver budget. Giving the comparison run less
    # time makes hits look valuable when all you measured was a worse solve.
    tight = {**cfg, "hit_cost": 999}
    strict = plan(proj, squad_ids, sell, bank, gws, free_transfers, tight,
                  pack_own, calendar=calendar)
    gain = (sum(w["ep"] for w in base_weeks) - 4 * sum(w["hits"] for w in base_weeks)) \
        - sum(w["ep"] for w in strict)
    return round(gain, 1), strict


def free_hit_week(proj, gw, budget, pack_own=None, rivalry=0.0):
    """A Free Hit is a different problem: one week, any fifteen you can afford,
    and the squad reverts afterwards. Nothing carries over, so it does not
    belong in the multi-week model at all — it is solved on its own."""
    pack_own = pack_own or {}
    pool = [p for p in proj.values()
            if p["exp_min"] >= 55 and p["status"] == "a" and p["own"] >= 0.6]
    P = {p["id"]: p for p in pool}
    m = pulp.LpProblem("freehit", pulp.LpMaximize)
    x = {i: m.add_variable(f"fx{i}", cat="Binary") for i in P}
    y = {i: m.add_variable(f"fy{i}", cat="Binary") for i in P}
    c = {i: m.add_variable(f"fc{i}", cat="Binary") for i in P}
    m += pulp.lpSum(P[i]["ep"][gw] * rivalry_weight(i, pack_own, rivalry)
                    * (y[i] + c[i]) for i in P)
    m += pulp.lpSum(P[i]["price"] * x[i] for i in P) <= budget
    for pos, n in SQUAD.items():
        m += pulp.lpSum(x[i] for i in P if P[i]["pos"] == pos) == n
        s = pulp.lpSum(y[i] for i in P if P[i]["pos"] == pos)
        m += s >= XI_MIN[pos]
        m += s <= XI_MAX[pos]
    for t in {p["team_id"] for p in P.values()}:
        m += pulp.lpSum(x[i] for i in P if P[i]["team_id"] == t) <= 3
    m += pulp.lpSum(y.values()) == 11
    m += pulp.lpSum(c.values()) == 1
    for i in P:
        m += y[i] <= x[i]
        m += c[i] <= y[i]
    solver.solve(m, 120)
    xi = sorted([P[i] for i in P if y[i].value() > 0.5], key=lambda p: -p["ep"][gw])
    cap = next(P[i] for i in P if c[i].value() > 0.5)
    return {"gw": gw, "xi": xi, "captain": cap,
            "ep": round(sum(p["ep"][gw] for p in xi) + cap["ep"][gw], 1)}


def chip_sweep(proj, squad_ids, sell, bank, gws, free_transfers, cfg, pack_own,
               chip="bb", calendar=None):
    """Force one chip into each candidate week and compare totals like for like.

    The multi-week solve reports where a chip landed. It does not tell you by how
    much, or whether second place was a point behind or twenty. This does."""
    results = {}
    for g in gws:
        c = {**cfg}
        w = plan(proj, squad_ids, sell, bank, gws, free_transfers, c, pack_own,
                 chips_available=(chip,), calendar=calendar, force_chip=(chip, g))
        results[g] = round(sum(x["ep"] for x in w) - 4 * sum(x["hits"] for x in w), 1)
    best = max(results, key=results.get)
    return best, results
