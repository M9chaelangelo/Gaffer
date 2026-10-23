"""Multi-period transfer and chip planner.

Solves the whole horizon at once: which 15 to hold each week, who starts, who
captains, when to spend a transfer, when a hit pays, and when to burn a chip.
Solving week by week is what makes people take hits they regret.
"""
import os

import pulp

from . import planner, solver, timing

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


def bench_order(bench, gw):
    """The order the game substitutes them in, which is a decision.

    The eleven came out of the solver sorted; the bench came out in whatever
    order the pool happened to be in, and `webdata` then numbered those
    positions and labelled them "the order the game substitutes them in".
    For GW8 that put Thomas at 3.84 behind Kostoulas at 2.72, so a blanking
    defender would have been replaced by the worse player.

    The keeper is listed first and not ranked: he has his own slot on the
    site, he is the only man who can replace the starting keeper, and he can
    replace nobody else — so his place among the outfielders is not a choice
    anyone gets to make.
    """
    gk = [p for p in bench if p["pos"] == "GKP"]
    out = [p for p in bench if p["pos"] != "GKP"]
    return gk + sorted(out, key=lambda p: -p["ep"][gw])


def plan(proj, squad_ids, sell, bank, gws, free_transfers, cfg,
         pack_own=None, chips_available=None,
         force_no_transfer_gws=(), force_wc_gw=None, calendar=None,
         force_chip=None):
    # A chip already played is not a decision. The sweep learned this; the
    # solver had not, and went on planning a Wildcard in GW8 that was spent in
    # GW6 — which quietly costs you every week the solver reserves for it.
    # Defaults to what config says is left rather than to all four.
    if chips_available is None:
        chips_available = tuple(
            c for c in ("wc", "tc", "bb", "fh")
            if c not in planner.chips_used(cfg)[0])
    # Chips come in two sets: the first expires at the GW19 deadline, the
    # second arrives for GW20. Constraining `sum(wc) <= 1` across the whole
    # horizon is right only while the horizon sits inside one half — once it
    # straddles GW19 it forbids a second-half chip the manager actually holds.
    spent_second = set(planner.chips_used_second(cfg))
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
    # Bought AND sold inside the horizon. Constrained below, penalised in the
    # objective: see the churn note further down.
    churn = {i: m.add_variable(f"churn_{i}", lowBound=0, upBound=1) for i in P}

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
            # captain that week, not of the best player in the game. `ep`
            # already carries the rivalry weight from four lines up, so
            # multiplying by it again here squared it — at rivalry 0.3 a
            # player nobody in the pack owned was worth 1.69x on the Triple
            # Captain term against 1.3x everywhere else, which is not a
            # leverage tilt, it is a different objective for one chip.
            obj.append(ep * trip[(i, g)])
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
    # Owning a player for a single week is a waste of two transfers.
    obj.append(-cfg.get("churn_cost", 3.0)
               * pulp.lpSum(churn[i] for i in P))
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
    # --- churn ---------------------------------------------------------
    # A transfer's gain is priced, but owning someone for a single week was
    # not. At a five-week horizon that was invisible; at eight the solver
    # started buying a player for one fixture and selling him the next week,
    # twice in the same line — each move "optimal" alone, together a waste
    # of two transfers for one week of ownership.
    #
    # `churn[i]` is forced to 1 when a player is both bought and sold inside
    # the horizon, and costs `churn_cost`. It deliberately does not grade by
    # how long he was held: a sharper version needs a variable per player per
    # pair of weeks, and the flat version already removes the behaviour that
    # matters. If the ledger later shows four-week holds being punished
    # unfairly, grade it then.
    for i in P:
        m += churn[i] >= (pulp.lpSum(tin[(i, g)] for g in gws)
                          + pulp.lpSum(tout[(i, g)] for g in gws) - 1)
    for i in P:
        m += pulp.lpSum(tin[(i, g)] for g in gws) <= 1
        if i in squad_ids and not cfg.get("allow_rebuy", False):
            m += pulp.lpSum(tin[(i, g)] for g in gws) == 0

    if force_wc_gw is not None:
        m += wc[force_wc_gw] == 1
    if force_chip is not None:
        which, when = force_chip
        m += {"wc": wc, "tc": tc, "bb": bb}[which][when] == 1
    # One of each chip per half, not per horizon.
    halves = {"first": [g for g in gws if planner.half_of(g) == "first"],
              "second": [g for g in gws if planner.half_of(g) == "second"]}
    for key, var in (("wc", wc), ("tc", tc), ("bb", bb)):
        for half, weeks in halves.items():
            if not weeks:
                continue
            gone = (key not in chips_available) if half == "first" \
                else (key in spent_second)
            m += pulp.lpSum(var[g] for g in weeks) <= (0 if gone else 1)
    # One chip inside a five-week window. Stacking three is a modelling artefact,
    # not a plan.
    m += pulp.lpSum(wc.values()) + pulp.lpSum(tc.values()) \
         + pulp.lpSum(bb.values()) <= cfg.get("max_chips_in_horizon", 1)

    # The solver's own verdict, which was being thrown away. A MILP that
    # hits its time limit returns a good feasible answer, not a proven best
    # one, and the difference is invisible on the page unless it is said.
    # On 16 Oct the solve used 101 of its 110 seconds, so this is not
    # hypothetical.
    status = solver.solve(m, cfg.get("time_limit", 240))

    out = []
    for g in gws:
        boosted = bb[g].value() > 0.5
        squad_g = [i for i in P if x[(i, g)].value() > 0.5]
        cap = next(P[i] for i in P if c[(i, g)].value() > 0.5)
        if boosted:
            # Under Bench Boost all fifteen score, so the eleven/bench split
            # does not change the total and the solver returns an arbitrary
            # one. On 15 Oct it benched the squad's second- and third-best
            # players. That costs nothing in the model and everything to a
            # reader, who sees a team sheet. Re-derive the eleven the way it
            # would be picked without the chip, so the row is still a lineup
            # if the chip is not played after all.
            pick = planner.best_xi(squad_g, P, g)
            xi = [P[i] for i in pick["xi"]]
            bench = bench_order([P[i] for i in pick["bench"]], g)
        else:
            if os.environ.get("GAFFER_DEBUG_XI") and g == gws[0]:
                print(f"  [debug] GW{g} owned-player view, as the solver sees it")
                for i in sorted(squad_ids, key=lambda i: -P[i]["ep"][g]
                                if i in P else 0):
                    if i not in P:
                        print(f"    {i} NOT IN POOL")
                        continue
                    q = P[i]
                    print(f"    {q['name'][:14]:<15}{q['pos']:<4}"
                          f"ep {q['ep'][g]:>6.3f}  w "
                          f"{rivalry_weight(i, pack_own, rivalry):>5.3f}"
                          f"  bwt {bench_weight(q):>5.3f}"
                          f"  x {x[(i, g)].value():>3.0f}  y {y[(i, g)].value():>3.0f}"
                          f"  status {q['status']}  expmin {q['exp_min']:.0f}")
            xi = sorted([P[i] for i in P if y[(i, g)].value() > 0.5],
                        key=lambda p: (-p["ep"][g]))
            bench = bench_order([P[i] for i in P if x[(i, g)].value() > 0.5
                                 and y[(i, g)].value() < 0.5], g)
        # Carry the players themselves, not their names. Two players can share
        # a web name (there are two Palmers), and pairing a wildcard's thirteen
        # moves by list order produces nonsense like a midfielder "replacing" a
        # goalkeeper.
        ins = [P[i] for i in P if tin[(i, g)].value() > 0.5]
        outs = [P[i] for i in P if tout[(i, g)].value() > 0.5]
        out.append({
            "solver_status": status,
            "optimal": status == "Optimal",
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
            "bench": bench,
            "captain": cap,
            # A Bench Boost week scores all fifteen. Summing the eleven alone
            # understated GW7 by 22.3 points on 15 Oct and printed the chip
            # week as the worst of the five, which is the opposite of true.
            "ep": round(sum(p["ep"][g] for p in (xi + bench if boosted else xi))
                        + cap["ep"][g], 1),
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
