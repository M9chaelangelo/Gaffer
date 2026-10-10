"""Run the gaffer. `python -m gaffer.main` — everything else is automatic."""
import argparse, json, os, sys
from datetime import datetime, timezone

import yaml

from . import (fetch, league, projections, prices, optimise, report, timing,
               styles, elite, explain, defence, calibrate, rotation, planner,
               webdata, wildcard, teamform)


def resolve_names(proj, names):
    """Turn a list of web names from config into player ids.

    Two players can share a web name, so ties go to the one with more points
    this season — which is the one you meant. Returns the ids it found and the
    names it did not, because silently dropping a name you typed is how a
    keep list stops keeping anybody."""
    by_name = {}
    for i, p in proj.items():
        by_name.setdefault(p["name"].lower(), []).append(i)
    ids, missing = [], []
    for nm in names or []:
        hit = by_name.get(str(nm).lower())
        if hit:
            ids.append(max(hit, key=lambda i: proj[i]["total_points"]))
        else:
            missing.append(nm)
    return ids, missing


def purchase_prices(boot, entry_id, squad_ids):
    """Rebuild what you paid, so selling prices are right."""
    els = {e["id"]: e for e in boot["elements"]}
    paid = {i: (els[i]["now_cost"] - els[i]["cost_change_start"]) / 10
            for i in squad_ids if i in els}
    try:
        for t in reversed(fetch.transfers(entry_id)):
            if t["element_in"] in paid:
                paid[t["element_in"]] = t["element_in_cost"] / 10
    except Exception:
        pass
    return paid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--free-transfers", default=None)
    ap.add_argument("--only-on-deadline-day", action="store_true",
                    help="exit quietly unless a deadline falls today")
    ap.add_argument("--wildcard", default=None,
                    help="draft a wildcard squad for this gameweek, whatever "
                         "chip_plan says")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    print("fetching…")
    boot = fetch.bootstrap()
    fx = fetch.fixtures()
    gw = fetch.current_gw(boot)
    gws = [g for g in range(gw, gw + cfg["horizon"])
           if any(e["id"] == g for e in boot["events"])]
    # A wildcard is judged over a longer run than a single transfer, so the
    # projections are built further out than the weekly plan needs. Everything
    # except the rebuild still works on `gws`: a five-week solve that suddenly
    # became a ten-week one would change every other recommendation on the page.
    span = max(cfg["horizon"], cfg.get("wildcard_horizon", 8))
    gws_long = [g for g in range(gw, gw + span)
                if any(e["id"] == g for e in boot["events"])]
    ev = next(e for e in boot["events"] if e["id"] == gw)
    deadline = datetime.fromisoformat(
        ev["deadline_time"].replace("Z", "+00:00")).strftime("%a %d %b %H:%M UTC")

    if args.only_on_deadline_day:
        d = datetime.fromisoformat(ev["deadline_time"].replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        hours = (d - now).total_seconds() / 3600
        state_path = os.path.join(cfg["out_dir"], "state.json")
        try:
            done = json.load(open(state_path)).get("solved_gw")
        except Exception:
            done = None
        # FPL can move a deadline, but never within 24 hours of it. So: always
        # solve on the day, and also solve early if this gameweek has no plan yet
        # and the deadline is close. That way a deadline shifting onto a Sunday
        # or a Tuesday cannot slip past us.
        if d.date() != now.date() and not (done != gw and 0 < hours < 72):
            print(f"nothing due — GW{gw} closes {d:%a %d %b %H:%M} UTC "
                  f"({hours:.0f}h away, plan on file for GW{done})")
            return
    print(f"gameweek {gw}, planning {gws}")
    feed = fetch.solio(cfg["solio_url"])
    mins_mult = rotation.congestion(
        boot, cfg.get("european_teams", ()), cfg.get("european_weeks", ()),
        cfg.get("european_haircut", 0.12), cfg.get("midweek_haircut", 0.06))
    proj = projections.build(boot, fx, gws_long, feed, cfg["solio_weight"],
                             cfg.get("form_weight", 0.30),
                             cfg.get("strength_weight", 0.5), mins_mult,
                             cfg.get("team_form_weight", 0.0),
                             cfg.get("team_form_half_life", 2.5))
    if cfg.get("calibrate_to_solio", True):
        scale, n = calibrate.fit(proj, feed, gw)
        if n:
            print(f"  calibration vs Solio: x{scale:.3f} on {n} shared players")
        proj = calibrate.apply(proj, scale, gws_long)

    last = gw - 1
    my = fetch.picks(cfg["entry_id"], last)
    squad_ids = [p["element"] for p in my["picks"]]
    bank = my["entry_history"]["bank"] / 10

    # Which chips are gone is public, and `chips_used` in config is typed by
    # hand, so it decays. A chip declared as held but already played is the
    # expensive direction: the sweep ranks it, and the best number on the page
    # becomes a move that cannot be made. Reported rather than overridden —
    # config stays the single source of truth, it just stops being able to
    # drift silently.
    try:
        played = planner.chips_played(fetch.history(cfg["entry_id"]))
        missing, surplus = planner.reconcile_chips(
            planner.chips_used(cfg)[0], played)
        if missing:
            print("  !! chips played but not in chips_used: "
                  + ", ".join(missing)
                  + " — the sweep will offer chips you do not have")
        if surplus:
            print("  !! chips_used lists chips your history does not show: "
                  + ", ".join(surplus))
    except Exception as e:                                  # noqa: BLE001
        # A diagnostic must never be the reason a deadline-morning solve dies.
        print(f"  (could not check chips against your history: {e})")

    # FPL's public API shows last gameweek's squad and does not expose transfers
    # you have already made for the upcoming one. If you have moved this week,
    # list the current fifteen in config as `current_squad` — otherwise the
    # solver plans from a team you no longer own.
    override = cfg.get("current_squad")
    if override:
        resolved, missing = resolve_names(proj, override)
        if missing:
            print(f"  !! could not resolve in current_squad: {missing}")
        if len(resolved) == 15:
            gone = [proj[i]["name"] for i in squad_ids if i not in resolved]
            added = [proj[i]["name"] for i in resolved if i not in squad_ids]
            if gone or added:
                print(f"  using current_squad override (out: {gone}, in: {added})")
            bank = float(cfg.get("bank", bank))
            squad_ids = resolved
        else:
            print(f"  !! current_squad has {len(resolved)} valid names, need 15 — "
                  f"falling back to the API squad")
    ft = int(args.free_transfers) if (args.free_transfers or "").strip() \
        else cfg.get("free_transfers", 1)

    paid = purchase_prices(boot, cfg["entry_id"], squad_ids)
    sell = {i: optimise.selling_price(paid.get(i, proj[i]["price"]), proj[i]["price"])
            for i in squad_ids}

    print("reading your mini-league…")
    lg = league.snapshot(cfg["league_id"], cfg["entry_id"], last,
                         cfg.get("rival_depth", 10))

    cal = timing.calendar(boot)

    # Anyone injured, suspended or gone. The solver already refuses to start
    # them; this decides whether they deserve a squad slot at all.
    flagged = []
    for i in squad_ids:
        p = proj[i]
        if p["status"] == "a":
            continue
        horizon_ep = sum(p["ep"][g] for g in gws)
        flagged.append({
            "name": p["name"], "status": p["status"], "news": p["news"],
            "ep": p["ep"][gw],
            "verdict": ("sell — no return inside the horizon" if horizon_ep < 4
                        else "hold — he is back before this plan ends"),
        })

    # A locked chip plan. Once you have decided, the weekly run should execute
    # that decision rather than re-litigating it every deadline on three more
    # days of noise.
    plan_chips = cfg.get("chip_plan") or {}
    forced = None
    forced_wc = plan_chips.get("wc") if plan_chips.get("wc") in gws else None
    for which in ("tc", "bb", "fh"):
        if plan_chips.get(which) in gws:
            forced = (which, plan_chips[which])
            break
    if plan_chips:
        pending = {k: v for k, v in plan_chips.items() if v in gws}
        print(f"  chip plan in force: {plan_chips}"
              + (f" — active this horizon: {pending}" if pending else ""))

    print("solving…")
    weeks = optimise.plan(proj, squad_ids, sell, bank, gws, ft, cfg,
                          lg["pack_own"], calendar=cal,
                          force_wc_gw=forced_wc, force_chip=forced)
    gain, _ = optimise.value_of_hit(proj, squad_ids, sell, bank, gws, ft, cfg,
                                    lg["pack_own"], weeks, cal)
    thr = cfg.get("hit_threshold", 6.0)
    verdict = (f"Taking a hit gains about {gain} points over {len(gws)} gameweeks. "
               + ("Worth it." if gain >= thr else
                  f"Not worth it — the bar is {thr}. Roll the transfer instead."))

    # --- the rebuild ---------------------------------------------------------
    # A wildcard is not a transfer, so it does not come out of the transfer
    # model. It gets its own solve: a bigger pool that reaches down to the
    # four-pound enablers, and a longer horizon, because you are buying a
    # fixture run rather than a weekend.
    wc_draft = None
    wc_gw = plan_chips.get("wc")
    if (args.wildcard or "").strip():
        try:
            wc_gw = int(args.wildcard)
        except ValueError:
            print(f"  !! --wildcard {args.wildcard!r} is not a gameweek; "
                  f"falling back to chip_plan")
    if wc_gw and wc_gw in gws_long:
        wc_weeks = [g for g in gws_long if g >= wc_gw][:cfg.get("wildcard_horizon", 8)]
        budget = round(bank + sum(sell.values()), 1)
        keep_ids, keep_miss = resolve_names(proj, cfg.get("wildcard_keep"))
        ban_ids, ban_miss = resolve_names(proj, cfg.get("wildcard_ban"))
        if keep_miss or ban_miss:
            print(f"  !! wildcard keep/ban names not found: "
                  f"{keep_miss + ban_miss}")
        bb_gw = plan_chips.get("bb") if plan_chips.get("bb") in wc_weeks else None
        tc_gw = plan_chips.get("tc") if plan_chips.get("tc") in wc_weeks else None
        print(f"drafting the GW{wc_gw} wildcard over {wc_weeks} "
              f"on {budget}m…")
        try:
            cand = wildcard.pool(proj, wc_weeks, cfg, keep_ids, ban_ids)
            wc_draft = wildcard.draft(proj, wc_weeks, budget, cfg,
                                      lg["pack_own"], keep=keep_ids,
                                      ban=ban_ids, bb_gw=bb_gw, tc_gw=tc_gw,
                                      candidates=cand)
            wc_draft["swaps"] = wildcard.swaps(
                wc_draft["squad"], proj, wc_weeks, budget, cfg,
                lg["pack_own"], bb_gw=bb_gw, tc_gw=tc_gw, candidates=cand)
            wc_draft["change"] = wildcard.summarise(wc_draft, proj, squad_ids)
            wc_draft["for_gw"] = wc_gw
            # The money is only right if nothing is bought between now and the
            # wildcard. Usually true — you are saving transfers for it — but it
            # is an assumption, not a fact, so the page says so.
            wc_draft["assumes_no_transfers"] = wc_gw > gws[0]
            wc_draft["pool_size"] = len(cand)
            kept = len(wc_draft["change"]["keep"])
            print(f"  {wc_draft['cost']}m spent, {wc_draft['in_bank']}m left, "
                  f"{kept} of your fifteen survive")
        except ValueError as e:
            print(f"  !! wildcard draft failed: {e}")

    # The weekly model also has an opinion about the wildcard week, and it is
    # the weaker one — five weeks, and a pool that starts above the enablers.
    # Showing both left two squads on the page, each labelled the wildcard.
    # From the chip week onwards the plan is the drafter's.
    if wc_draft:
        adopted = wildcard.adopt(weeks, wc_draft, proj, squad_ids)
        if adopted is not weeks:
            moved = sum(1 for a, b in zip(adopted, weeks)
                        if [p["id"] for p in a["xi"]] != [p["id"] for p in b["xi"]])
            print(f"  the plan now buys the drafted fifteen "
                  f"({moved} of {len(weeks)} weeks rewritten)")
            weeks = adopted

    atk, dfn = projections.team_strength(
        boot, fixtures=fx, form_weight=cfg.get("team_form_weight", 0.0),
        half_life=cfg.get("team_form_half_life", 2.5))
    lams, tmn = defence.lambdas(boot, fx, gws, atk, fetch.solio(cfg["solio_url"]))
    cs_table = defence.table(boot, lams, tmn, gw, limit=12)

    # Who is actually in form, so the page can be checked against what you
    # watched on Saturday rather than only acted on.
    season_atk, season_dfn = projections.team_strength(boot)
    fa, fd = teamform.recent(boot, fx, season_atk, season_dfn,
                             half_life=cfg.get("team_form_half_life", 2.5))
    form_table = teamform.table(boot, fa, fd)
    # Attack and defence per club, keyed by the three-letter code the cards
    # use, so the page can put every team on one scatter.
    codes = {t["id"]: t["short_name"] for t in boot["teams"]}
    strengths = {codes[tid]: {"attack": round(atk.get(tid, 1.0), 3),
                              "defence": round(dfn.get(tid, 1.0), 3),
                              # One number for tiering: creating more than the
                              # league and conceding less than it both count.
                              "overall": round(atk.get(tid, 1.0)
                                               / max(0.3, dfn.get(tid, 1.0)), 3)}
                 for tid in codes}
    # The fixture ticker: every match in the horizon, by gameweek. The page
    # cannot derive this from the player cards — those carry one opponent per
    # player, not the round.
    ticker = {}
    for x in fx:
        g = x.get("event")
        if g in gws_long and x.get("team_h") in codes and x.get("team_a") in codes:
            ticker.setdefault(g, []).append(
                {"h": codes[x["team_h"]], "a": codes[x["team_a"]]})
    for g in ticker:
        ticker[g].sort(key=lambda m: m["h"])
    if form_table:
        hot = ", ".join(f"{r['team']} {r['form']:.2f}" for r in form_table[:4])
        print(f"  team form (weight {cfg.get('team_form_weight', 0.0)}): {hot}")

    prof = styles.profile(boot)
    try:
        el = elite.snapshot(last, top_n=cfg.get("elite_sample", 100))
        elite_own = el["own"]
    except Exception as e:
        print(f"  elite sample unavailable ({e})")
        elite_own = {}

    pf = prices.forecast(boot)
    watch = sorted(
        [{"name": proj[i]["name"],
          "where": "your squad" if i in squad_ids else "target",
          "net": pf[i]["net"], "direction": pf[i]["direction"]}
         for i in proj if pf[i]["direction"] != "hold"
         and (i in squad_ids or proj[i]["own"] > 3)],
        key=lambda r: -abs(r["net"]))

    gems = sorted(
        [{"name": p["name"], "team": p["team"], "price": p["price"],
          "proj": p["ep"][gw], "league": lg["league_own"].get(i, 0.0),
          "pack": lg["pack_own"].get(i, 0.0)}
         for i, p in proj.items()
         if lg["league_own"].get(i, 0.0) <= 0.10 and i not in squad_ids
         and p["exp_min"] >= 60 and p["status"] == "a"],
        key=lambda g: -g["proj"])

    behind = lg["points_behind"]
    line = (f"You are {behind} points off the top of {lg['rows'][0]['entry_name']} "
            f"in a {lg['n_all']}-team league." if behind is not None else "")

    rationale = []
    for w in weeks:
        rationale.append({
            "gw": w["gw"], "chip": w["chip"],
            "captain": explain.captain(w, proj, prof, lg["pack_own"], elite_own, gws),
            "transfers": explain.transfers(w, proj, prof, lg["league_own"],
                                           lg["pack_own"], elite_own, pf, gws),
        })

    # Grade the plan Michael actually wrote, against the line the solver just
    # found. The solver answers "what should I do"; this answers "does what I
    # intend to do work, and what does it cost me".
    solver_total = round(sum(w["ep"] for w in weeks), 2)
    graded = planner.grade(cfg, proj, squad_ids, bank, ft, gws,
                           sell=sell, solver_total=solver_total)
    if graded["unresolved"]:
        print("  !! names in my_plan that could not be resolved: "
              + ", ".join(graded["unresolved"]))
    if not graded["plan"]["legal"]:
        print(f"  !! your plan has {len(graded['plan']['violations'])} rule "
              "break(s) — see the report")

    os.makedirs(cfg["out_dir"], exist_ok=True)

    # The interactive app lives in docs/index.html and is a static file in the
    # repo; it reads data.json. The server-rendered page is still written, as
    # report.html — it needs no JavaScript and is the thing that still works if
    # the app breaks or the browser is ancient.
    data = webdata.build(proj, prof, lg, elite_own, pf, gws, squad_ids, weeks,
                         deadline, gw, planner=graded, hit_verdict=verdict,
                         clean_sheets=cs_table, wildcard=wc_draft,
                         team_form=form_table, strengths=strengths,
                         ticker=ticker)
    webdata.write(data, cfg["out_dir"])

    out = os.path.join(cfg["out_dir"], "report.html")
    report.render({"weeks": weeks, "deadline": deadline, "gems": gems,
                   "prices": watch, "hit_verdict": verdict, "flagged": flagged,
                   "pack_own": lg["pack_own"], "league_line": line,
                   "rationale": rationale, "clean_sheets": cs_table,
                   "planner": graded, "wildcard": wc_draft, "proj": proj,
                   "team_form": form_table}, out)
    json.dump({"gw": gw, "generated": datetime.now(timezone.utc).isoformat(),
               "weeks": [{k: v for k, v in w.items()
                          if k in ("gw", "in", "out", "hits", "chip", "ep")}
                         for w in weeks],
               "hit_verdict": verdict, "gems": gems[:20],
               "rationale": rationale, "planner": graded,
               # Names, not ids: this file is meant to be readable on its own.
               "wildcard": (None if not wc_draft else {
                   "for_gw": wc_draft["for_gw"],
                   "over": wc_draft["gws"],
                   "cost": wc_draft["cost"],
                   "budget": wc_draft["budget"],
                   "in_bank": wc_draft["in_bank"],
                   "squad": [proj[i]["name"] for i in wc_draft["squad"]],
                   "keep": [proj[i]["name"] for i in wc_draft["change"]["keep"]],
                   "sell": [proj[i]["name"] for i in wc_draft["change"]["sell"]],
                   "buy": [proj[i]["name"] for i in wc_draft["change"]["buy"]],
                   "closest_calls": [
                       {"player": proj[r["out"]]["name"],
                        "alternative": (proj[r["in"]]["name"] if r["in"] else None),
                        "gap": r["gap"]}
                       for r in wc_draft["swaps"][:6]],
               })},
              open(os.path.join(cfg["out_dir"], "plan.json"), "w"), indent=1)
    json.dump({"solved_gw": gw, "deadline": ev["deadline_time"],
               "solved_at": datetime.now(timezone.utc).isoformat()},
              open(os.path.join(cfg["out_dir"], "state.json"), "w"), indent=1)
    print(f"wrote {out}; next deadline GW{gw} {deadline}")


if __name__ == "__main__":
    main()
