"""Evaluate a plan you wrote yourself.

`optimise.plan` answers "what should I do". This module answers a different
question: "I have decided to do THIS — does it work, and what does it cost me
against the best line?"

The two are not the same job. A solver that only ever hands you its own answer
is impossible to argue with, and the README is right that an argument you
cannot have is one you cannot win. Most managers already hold a plan in their
head — wildcard after the international break, Bench Boost on the double. What
they lack is something that checks it against the rules and prices it.

Three things this does that the solver does not:

  VALIDATES. A plan that sprints past the budget in GW9 is worse than no plan,
  because you find out at 19:29 on a Friday. Every rule the game enforces is
  checked here, at the gameweek it breaks, and ALL breaks are reported — you
  want to know whether it is one mistake or five before you start fixing.

  PRICES THE GAP. Your plan against the solver's, in points, so "Bench Boost in
  GW9 instead of GW7" stops being a matter of taste and becomes a number.

  SURVIVES A BREAK. Evaluation continues past a violation so the rest of the
  plan can still be read. `legal` is False and the numbers are then indicative,
  not real.

Everything here is a pure function of `proj` and the plan. No network, no
solver, no global state — which is why it can be tested, and is.
"""
from dataclasses import dataclass, field

SQUAD = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}
XI_SIZE = 11
MAX_PER_CLUB = 3
MAX_BANKED_FT = 5

# FPL refuses more than twenty transfers in a gameweek unless a chip is played.
HARD_TRANSFER_CAP = 20

# Chips that make every transfer in the week free.
UNLIMITED = {"wc", "fh"}
# Chips where the squad snaps back afterwards: the week is a rental.
REVERTING = {"fh"}

CHIP_NAMES = {
    "wc": "Wildcard", "fh": "Free Hit",
    "tc": "Triple Captain", "bb": "Bench Boost",
}

# The first set of chips dies at the GW19 deadline and cannot be carried over.
FIRST_HALF_LAST_GW = 19


@dataclass
class Violation:
    gw: int
    code: str
    message: str

    def as_dict(self):
        return {"gw": self.gw, "code": self.code, "message": self.message}


@dataclass
class WeekResult:
    gw: int
    squad: list
    xi: list
    bench: list
    captain: int | None
    formation: str
    ep: float
    hit: int
    free_transfers: int
    free_after: int
    bank: float
    chip: str | None
    transfers: int

    def as_dict(self):
        d = self.__dict__.copy()
        d["ep"] = round(self.ep, 2)
        d["bank"] = round(self.bank, 1)
        return d


@dataclass
class PlanResult:
    weeks: list = field(default_factory=list)
    violations: list = field(default_factory=list)
    total_ep: float = 0.0
    total_hits: int = 0

    @property
    def legal(self):
        return not self.violations

    def as_dict(self):
        return {
            "legal": self.legal,
            "total_ep": round(self.total_ep, 2),
            "total_hits": self.total_hits,
            "weeks": [w.as_dict() for w in self.weeks],
            "violations": [v.as_dict() for v in self.violations],
        }


def best_xi(squad, proj, gw, chip=None):
    """The eleven that score, and the bench in substitution order.

    Every legal formation is enumerated rather than picked greedily. There are
    only eight, and greedy selection can paint itself into a corner where the
    last slot must be a defender and only expensive ones are left.
    """
    by_pos = {}
    for pid in squad:
        p = proj.get(pid)
        if not p:
            continue
        by_pos.setdefault(p["pos"], []).append(p)
    for lst in by_pos.values():
        lst.sort(key=lambda p: -p["ep"].get(gw, 0.0))

    best = None
    for d in range(XI_MIN["DEF"], XI_MAX["DEF"] + 1):
        for m in range(XI_MIN["MID"], XI_MAX["MID"] + 1):
            for f in range(XI_MIN["FWD"], XI_MAX["FWD"] + 1):
                g = XI_SIZE - d - m - f
                if g != 1:
                    continue
                shape = {"GKP": g, "DEF": d, "MID": m, "FWD": f}
                if any(len(by_pos.get(pos, [])) < n for pos, n in shape.items()):
                    continue
                picked, total = [], 0.0
                for pos, n in shape.items():
                    for p in by_pos[pos][:n]:
                        picked.append(p)
                        total += p["ep"].get(gw, 0.0)
                if best is None or total > best[1]:
                    best = (picked, total, f"{d}-{m}-{f}")

    if best is None:
        return None

    picked, total, formation = best
    chosen = {p["id"] for p in picked}
    bench = [pid for pid in squad if pid not in chosen]
    # Outfield first, keeper last: the reserve keeper can only replace a keeper.
    bench.sort(key=lambda pid: (
        proj[pid]["pos"] == "GKP", -proj[pid]["ep"].get(gw, 0.0)
    ))

    captain = max(picked, key=lambda p: p["ep"].get(gw, 0.0))["id"] if picked else None
    return {
        "xi": [p["id"] for p in picked],
        "bench": bench,
        "formation": formation,
        "base": total,
        "captain": captain,
    }


def evaluate(plan_steps, proj, squad_ids, bank, first_ft, gws,
             sell=None, chips_used_before=()):
    """Walk the plan forward, gameweek by gameweek.

    `plan_steps` is {gw: {"in": [...], "out": [...], "chip": "bb"|None}} with
    ids, not names. `sell` maps player id to what you would actually get for
    him — FPL lets you keep only half of a rise, so a player is usually worth
    less than his listed price.
    """
    sell = sell or {}
    result = PlanResult()
    squad = list(squad_ids)
    free = first_ft
    spent_chips = dict.fromkeys(chips_used_before, True)

    for gw in gws:
        step = plan_steps.get(gw) or {}
        chip = step.get("chip")
        moves_in = list(step.get("in") or [])
        moves_out = list(step.get("out") or [])

        squad_before, bank_before = list(squad), bank

        # --- the chip ----------------------------------------------------
        if chip:
            if chip not in CHIP_NAMES:
                result.violations.append(Violation(
                    gw, "chip-unknown",
                    f'"{chip}" is not an FPL chip. Use one of: '
                    + ", ".join(sorted(CHIP_NAMES)) + "."))
                chip = None
            elif spent_chips.get(chip):
                result.violations.append(Violation(
                    gw, "chip-spent",
                    f"{CHIP_NAMES[chip]} is already used earlier in this plan. "
                    "Each chip plays once per half."))
                chip = None
            else:
                spent_chips[chip] = True

        # --- the transfers ------------------------------------------------
        if len(moves_in) != len(moves_out):
            result.violations.append(Violation(
                gw, "unbalanced",
                f"GW{gw} lists {len(moves_out)} out and {len(moves_in)} in. "
                "A transfer is always one for one."))

        pairs = list(zip(moves_out, moves_in))
        done = 0
        for out_id, in_id in pairs:
            po, pi = proj.get(out_id), proj.get(in_id)
            if not po or not pi:
                result.violations.append(Violation(
                    gw, "unknown-player",
                    f"GW{gw}: player id {out_id if not po else in_id} is not in "
                    "this season's data."))
                continue
            if out_id not in squad:
                result.violations.append(Violation(
                    gw, "not-owned",
                    f"GW{gw}: you cannot sell {po['name']} — he is not in the "
                    "squad at that point in the plan."))
                continue
            if in_id in squad:
                result.violations.append(Violation(
                    gw, "already-owned",
                    f"GW{gw}: {pi['name']} is already in the squad."))
                continue
            if po["pos"] != pi["pos"]:
                result.violations.append(Violation(
                    gw, "position",
                    f"GW{gw}: {po['name']} is a {po['pos']} and {pi['name']} is a "
                    f"{pi['pos']}. The squad has to stay "
                    + "/".join(f"{n} {p}" for p, n in SQUAD.items()) + "."))
                continue

            squad = [i for i in squad if i != out_id] + [in_id]
            bank += sell.get(out_id, po["price"]) - pi["price"]
            done += 1

        if done > HARD_TRANSFER_CAP and not chip:
            result.violations.append(Violation(
                gw, "transfer-cap",
                f"GW{gw} makes {done} transfers. FPL caps a gameweek at "
                f"{HARD_TRANSFER_CAP} unless a chip is played."))

        # --- budget and the club limit -------------------------------------
        if bank < -1e-9:
            result.violations.append(Violation(
                gw, "over-budget",
                f"GW{gw} is {abs(bank):.1f}m over budget. Sell someone dearer, "
                "or move a transfer to a later week."))

        clubs = {}
        for pid in squad:
            p = proj.get(pid)
            if p:
                clubs[p["team"]] = clubs.get(p["team"], 0) + 1
        for club, n in sorted(clubs.items()):
            if n > MAX_PER_CLUB:
                result.violations.append(Violation(
                    gw, "club-limit",
                    f"GW{gw} holds {n} players from {club}. The limit is "
                    f"{MAX_PER_CLUB}."))

        shape = {}
        for pid in squad:
            p = proj.get(pid)
            if p:
                shape[p["pos"]] = shape.get(p["pos"], 0) + 1
        if shape != SQUAD and len(squad) == 15:
            result.violations.append(Violation(
                gw, "composition",
                f"GW{gw} squad is "
                + "/".join(f"{shape.get(p, 0)} {p}" for p in SQUAD)
                + ", not " + "/".join(f"{n} {p}" for p, n in SQUAD.items()) + "."))

        # --- the cost --------------------------------------------------------
        paid = 0 if chip in UNLIMITED else max(0, done - free)
        hit = -4 * paid if paid else 0

        # --- points ----------------------------------------------------------
        lineup = best_xi(squad, proj, gw, chip)
        if lineup is None:
            result.violations.append(Violation(
                gw, "no-xi",
                f"GW{gw}: no legal eleven can be picked from this squad."))
            ep, xi, bench, formation, captain = 0.0, [], [], "-", None
        else:
            xi = lineup["xi"]
            bench = lineup["bench"]
            formation = lineup["formation"]
            captain = lineup["captain"]
            ep = lineup["base"]
            if chip == "bb":
                ep += sum(proj[i]["ep"].get(gw, 0.0) for i in bench if i in proj)
            if captain is not None:
                # Triple Captain is one extra helping of the armband, not two.
                ep += proj[captain]["ep"].get(gw, 0.0) * (2 if chip == "tc" else 1)

        # Free transfers: one a week, stacking to five. A Wildcard or Free Hit
        # week spends none of your saved ones — and earns no extra one either.
        if chip in UNLIMITED:
            free_after = min(MAX_BANKED_FT, free)
        else:
            free_after = min(MAX_BANKED_FT, free - min(done, free) + 1)

        result.weeks.append(WeekResult(
            gw=gw, squad=list(squad), xi=xi, bench=bench, captain=captain,
            formation=formation, ep=ep, hit=hit, free_transfers=free,
            free_after=free_after, bank=round(bank, 1), chip=chip,
            transfers=done,
        ))
        result.total_ep += ep + hit
        result.total_hits += hit
        free = free_after

        # Free Hit rents the squad for a week; everything snaps back.
        if chip in REVERTING:
            squad, bank = squad_before, bank_before

    return result


def chips_used(cfg):
    """The chips already played, from `chips_used` in config.

    FPL's public API does not say which chips an entry has spent, so this is
    declared rather than discovered — the same reason `free_transfers` is.
    Returns (spent, unknown): the names that are chips, and the ones that are
    not, so a typo is reported instead of silently meaning "nothing is spent".
    """
    raw = (cfg or {}).get("chips_used") or ()
    if isinstance(raw, str):
        raw = [raw]
    spent, unknown = [], []
    for name in raw:
        key = str(name).strip().lower()
        (spent if key in CHIP_NAMES else unknown).append(key)
    return tuple(dict.fromkeys(spent)), unknown


def chips_played(history):
    """The chip names an entry's public history says it has already played.

    FPL's own names differ from ours ("3xc", "bboost", "freehit", "wildcard"),
    and unknown ones are dropped rather than guessed at: a name we cannot map
    must not silently become "nothing was played".
    """
    alias = {
        "3xc": "tc", "tc": "tc",
        "bboost": "bb", "bb": "bb",
        "freehit": "fh", "fh": "fh",
        "wildcard": "wc", "wc": "wc",
    }
    out = []
    for c in ((history or {}).get("chips") or []):
        name = c.get("name") if isinstance(c, dict) else c
        key = alias.get(str(name).strip().lower())
        if key:
            out.append(key)
    return tuple(dict.fromkeys(out))


def reconcile_chips(declared, played):
    """Where `chips_used` disagrees with the entry's actual history.

    Returns (missing, surplus): chips played but not declared, and chips
    declared but not played. The first is the dangerous one — it is the case
    that had the sweep recommending a Triple Captain already spent.
    """
    declared, played = set(declared or ()), set(played or ())
    return tuple(sorted(played - declared)), tuple(sorted(declared - played))


def chip_windows(gws, spent=()):
    """Which gameweeks each chip may legally be played in.

    The first set of chips expires at the GW19 deadline, so a plan that saves
    Bench Boost for GW21 while still holding a first-half chip is not late —
    it is impossible. A chip already spent has no window at all.
    """
    return {
        chip: [g for g in gws if g <= FIRST_HALF_LAST_GW]
        for chip in CHIP_NAMES
        if chip not in spent
    }


def sweep_chip(chip, plan_steps, proj, squad_ids, bank, first_ft, gws, sell=None):
    """Try one chip in every free week and report what each is worth.

    Exact within the horizon: every placement is evaluated, not sampled. The
    limit is the horizon itself — if the right week for a Bench Boost is GW30
    and you are planning five weeks, nothing here can see it. That is why the
    horizon is reported alongside the answer.
    """
    taken = {g for g, s in plan_steps.items()
             if (s or {}).get("chip") and (s or {}).get("chip") != chip}

    stripped = {g: {**(s or {}), "chip": None if (s or {}).get("chip") == chip
                    else (s or {}).get("chip")}
                for g, s in plan_steps.items()}
    base = evaluate(stripped, proj, squad_ids, bank, first_ft, gws, sell).total_ep

    out = []
    for g in gws:
        if g in taken or g > FIRST_HALF_LAST_GW:
            continue
        trial = {gg: dict(s or {}) for gg, s in stripped.items()}
        trial.setdefault(g, {})
        trial[g] = {**trial.get(g, {}), "chip": chip}
        res = evaluate(trial, proj, squad_ids, bank, first_ft, gws, sell)
        if not res.legal:
            continue
        out.append({"gw": g, "gain": round(res.total_ep - base, 2)})

    out.sort(key=lambda r: -r["gain"])
    return out


def resolve(names, proj):
    """Turn the names you typed in config into player ids.

    Names are what a human writes; ids are what the model uses. The mapping is
    FPL's `web_name`, which is unique often enough to be usable and ambiguous
    often enough to need saying so — two Bruno Fernandes and you want to be
    told, not guessed at.
    """
    if not names:
        return [], []
    index = {}
    for p in proj.values():
        index.setdefault(p["name"].lower(), []).append(p["id"])

    ids, missing = [], []
    for name in names:
        hits = index.get(str(name).strip().lower(), [])
        if len(hits) == 1:
            ids.append(hits[0])
        elif not hits:
            missing.append(str(name))
        else:
            # Ambiguous: take the first but say so, rather than silently
            # planning around the wrong player.
            ids.append(hits[0])
            missing.append(f"{name} (ambiguous — {len(hits)} players share it)")
    return ids, missing


def from_config(cfg, proj, gws):
    """Build a plan from `my_plan` and `chip_plan` in config.

    `chip_plan` is folded in so the two cannot contradict each other: a chip
    locked there and not mentioned in `my_plan` still shows up in the plan
    being graded.
    """
    raw = cfg.get("my_plan") or {}
    chip_plan = cfg.get("chip_plan") or {}

    steps, missing = {}, []
    for gw in gws:
        entry = raw.get(gw) or {}
        out_ids, m1 = resolve(entry.get("out"), proj)
        in_ids, m2 = resolve(entry.get("in"), proj)
        missing += [f"GW{gw}: {x}" for x in m1 + m2]

        chip = entry.get("chip")
        if not chip:
            for which, when in chip_plan.items():
                if when == gw:
                    chip = which
                    break

        steps[gw] = {"out": out_ids, "in": in_ids, "chip": chip}

    return steps, missing


def grade(cfg, proj, squad_ids, bank, first_ft, gws, sell=None,
          solver_total=None):
    """The whole planner in one call: check it, price it, and suggest.

    `solver_total` is what the optimiser's own line is worth over the same
    horizon. The gap between that and your plan is the only number that
    settles an argument about a chip week.
    """
    steps, missing = from_config(cfg, proj, gws)
    spent, unknown = chips_used(cfg)
    result = evaluate(steps, proj, squad_ids, bank, first_ft, gws, sell,
                      chips_used_before=spent)

    placed = {s.get("chip") for s in steps.values() if s.get("chip")}
    sweeps = {}
    for chip in CHIP_NAMES:
        # A chip already played is not a decision. Sweeping it produces a
        # recommendation the manager cannot act on, which is worse than
        # silence: the GW8 Triple Captain it suggested was worth 9.3 points
        # and did not exist.
        if chip in spent:
            continue
        ranked = sweep_chip(chip, steps, proj, squad_ids, bank, first_ft, gws, sell)
        if not ranked:
            continue
        current = next((g for g, s in steps.items() if s.get("chip") == chip), None)
        best = ranked[0]
        here = next((r["gain"] for r in ranked if r["gw"] == current), None)
        entry = {
            "name": CHIP_NAMES[chip],
            "ranked": ranked[:6],
            "current_gw": current,
            "best_gw": best["gw"],
            "best_gain": best["gain"],
            "gain_here": here,
            "move_gain": None if here is None else round(best["gain"] - here, 2),
            "placed": chip in placed,
            "valued": True,
        }

        # Wildcard and Free Hit are worth only the players they let you change,
        # and the sweep moves the chip's label between weeks without moving any
        # transfers. Their gain is therefore identically zero in every week —
        # not a tie to be broken, an answer that was never computed. Reporting
        # it as `best_gw` with a gain of 0.0 reads as a recommendation, and the
        # first week in the list silently becomes "the best week".
        #
        # Valuing them properly means re-drafting the squad for each candidate
        # week, which is what wildcard.py does once over a 168-player pool in
        # three minutes. Doing it per week per chip inside the sweep is a
        # different order of cost, so it is declared unvalued instead of
        # guessed at.
        if chip in UNLIMITED:
            entry.update(
                valued=False,
                ranked=[{"gw": r["gw"], "gain": None} for r in ranked[:6]],
                best_gw=None, best_gain=None, gain_here=None, move_gain=None,
                note=(f"{CHIP_NAMES[chip]} is not priced by the sweep: its "
                      "value is the squad it buys, and the sweep holds the "
                      "squad fixed. Compare drafts instead "
                      "(--wildcard N per candidate week)."),
            )

        sweeps[chip] = entry

    return {
        "plan": result.as_dict(),
        "steps": {g: dict(s) for g, s in steps.items()},
        "unresolved": missing + [
            f'"{name}" in chips_used is not an FPL chip. Use one of: '
            + ", ".join(sorted(CHIP_NAMES)) + "."
            for name in unknown
        ],
        "chips_used": list(spent),
        "chips": sweeps,
        "horizon": {"from": gws[0], "to": gws[-1]},
        "solver_total": solver_total,
        "gap": (None if solver_total is None
                else round(result.total_ep - solver_total, 2)),
    }
