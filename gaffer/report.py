"""Renders the week's decision as a single page you can read on a phone."""
import html
from datetime import datetime, timezone

CSS = """
:root{--ink:#0d1117;--panel:#151b23;--line:#243040;--chalk:#eef2ee;--mute:#8d9bab;
--rise:#4ade80;--fall:#f0883e;--flag:#e5534b}
*{box-sizing:border-box}
body{margin:0;background:var(--ink);color:var(--chalk);
font:16px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
-webkit-text-size-adjust:100%}
.wrap{max-width:56rem;margin:0 auto;padding:1.25rem 1rem 4rem}
.deadline{color:var(--mute);font-size:.82rem;letter-spacing:.04em}
h1{font-size:clamp(2rem,7vw,3.1rem);line-height:1.05;margin:.3rem 0 .1rem;
font-weight:800;letter-spacing:-.02em}
.call{background:var(--panel);border-left:3px solid var(--rise);padding:1rem 1.1rem;
margin:1.25rem 0 2rem;border-radius:0 6px 6px 0}
.call b{display:block;font-size:1.25rem;margin-bottom:.25rem}
h2{font-size:1.05rem;font-weight:700;margin:2.2rem 0 .6rem;
padding-bottom:.4rem;border-bottom:1px solid var(--line)}
table{width:100%;border-collapse:collapse;font-size:.9rem}
th{text-align:left;color:var(--mute);font-weight:600;padding:.4rem .5rem .4rem 0;
font-size:.78rem}
td{padding:.42rem .5rem .42rem 0;border-top:1px solid var(--line);vertical-align:top}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.rise{color:var(--rise)}.fall{color:var(--fall)}.flag{color:var(--flag)}
.tag{font-size:.72rem;color:var(--mute)}
.week{background:var(--panel);border-radius:6px;padding:.9rem 1rem;margin-bottom:.7rem}
.week h3{margin:0 0 .35rem;font-size:.95rem}
.week ul{margin:.3rem 0 .9rem;padding-left:1.1rem;font-size:.87rem;color:#c6d0da}
.poslabel{color:var(--mute);font-size:.7rem;letter-spacing:.12em;margin:1.1rem 0 .35rem}
.row{display:flex;align-items:center;gap:.6rem;margin:.3rem 0}
.lbl{flex:0 0 41%;font-size:.87rem;line-height:1.2}
.lbl em{display:block;font-style:normal;color:var(--mute);font-size:.72rem}
.track{flex:1;height:7px;background:#1b232d;border-radius:4px;overflow:hidden}
.track i{display:block;height:100%;border-radius:4px}
.val{flex:0 0 2.2rem;text-align:right;font-size:.85rem;
font-variant-numeric:tabular-nums}
svg.cmp{width:100%;height:auto;margin:.5rem 0 .2rem;overflow:visible}
svg.cmp text{font:11px ui-sans-serif,sans-serif}
svg.cmp .lb{fill:var(--mute)}
svg.cmp .k{fill:var(--fall);font-weight:700}
svg.cmp .k2{fill:var(--rise);font-weight:700}
svg.cmp .a{fill:var(--fall);opacity:.85}
svg.cmp .b{fill:var(--rise);opacity:.85}
.week li{margin:.22rem 0}
.chip{color:var(--rise);font-weight:700}
.note{color:var(--mute);font-size:.85rem;margin-top:.3rem}
footer{color:var(--mute);font-size:.78rem;margin-top:3rem;
border-top:1px solid var(--line);padding-top:1rem}
@media(prefers-reduced-motion:no-preference){.call{animation:in .4s ease-out}}
@keyframes in{from{opacity:0;transform:translateY(4px)}to{opacity:1}}
"""


CHIP_LABEL = {"wc": "Wildcard", "fh": "Free Hit",
              "tc": "Triple Captain", "bb": "Bench Boost", None: "—"}


def _name(ctx, pid):
    """Player id to name, using whatever the report already has to hand."""
    for w in ctx.get("weeks", []):
        for pl in list(w.get("xi", [])) + list(w.get("bench", [])):
            if pl.get("id") == pid:
                return pl["name"]
    return str(pid)


def _esc(s):
    return html.escape(str(s))


def _bar_row(label, sub, value, vmax, width=100, colour="var(--rise)"):
    w = 0 if not vmax else max(1.5, min(100.0, value / vmax * width))
    return (f"<div class=row><span class=lbl>{_esc(label)}"
            f"<em>{_esc(sub)}</em></span>"
            f"<span class=track><i style='width:{w:.1f}%;background:{colour}'></i>"
            f"</span><span class=val>{value:.1f}</span></div>")


def _compare_svg(out_c, in_c):
    """One chart per swap: the same metrics for both players, normalised so the
    longer bar is always the better number. If the incoming player's bars are
    not visibly longer, the transfer is not obviously right — which is
    information too."""
    metrics = [
        ("Projected points", out_c["ep_horizon"], in_c["ep_horizon"]),
        ("Starts 60+ mins", (out_c["start_prob"] or 0) * 100,
         (in_c["start_prob"] or 0) * 100),
        ("xG per 90", (out_c["xg90"] or 0) * 100, (in_c["xg90"] or 0) * 100),
        ("xA per 90", (out_c["xa90"] or 0) * 100, (in_c["xa90"] or 0) * 100),
        ("Shot quality", (out_c["shot_quality"] or 0) * 1000,
         (in_c["shot_quality"] or 0) * 1000),
        ("Share of team attack", (out_c["attacking_share"] or 0) * 100,
         (in_c["attacking_share"] or 0) * 100),
        ("Defensive actions/90", out_c["defcon90"] or 0, in_c["defcon90"] or 0),
        ("Pack ownership", out_c["own_pack"] * 100, in_c["own_pack"] * 100),
    ]
    rowh, pad, w = 26, 118, 560
    h = len(metrics) * rowh + 34
    s = [f"<svg viewBox='0 0 {w} {h}' class=cmp role=img "
         f"aria-label='Comparison of {_esc(out_c['name'])} and {_esc(in_c['name'])}'>"]
    s.append(f"<text x='{pad}' y='12' class=k>{_esc(out_c['name'])}</text>")
    s.append(f"<text x='{pad + 210}' y='12' class=k2>{_esc(in_c['name'])}</text>")
    for n, (label, a, b) in enumerate(metrics):
        y = 30 + n * rowh
        top = max(a, b, 0.001)
        aw, bw = (a / top) * (w - pad - 46), (b / top) * (w - pad - 46)
        s.append(f"<text x='0' y='{y + 9}' class=lb>{_esc(label)}</text>")
        s.append(f"<rect x='{pad}' y='{y}' width='{max(aw, 1):.1f}' height='7' "
                 f"rx='2' class=a/>")
        s.append(f"<rect x='{pad}' y='{y + 10}' width='{max(bw, 1):.1f}' "
                 f"height='7' rx='2' class=b/>")
    s.append("</svg>")
    return "".join(s)


def render(ctx, path):
    w = ctx["weeks"]
    now = w[0]
    if now["chip"]:
        headline = f"Play the {now['chip']}"
    elif now["in"]:
        headline = " · ".join(f"{o} → {i}" for o, i in zip(now["out"], now["in"]))
    else:
        headline = "No transfer. Set the team and wait."

    p = []
    p.append(f"<!doctype html><html lang=en><head><meta charset=utf-8>"
             f"<meta name=viewport content='width=device-width,initial-scale=1'>"
             f"<title>Gaffer · GW{now['gw']}</title><style>{CSS}</style></head><body>"
             f"<div class=wrap>")
    p.append(f"<p class=deadline>Gameweek {now['gw']} closes {_esc(ctx['deadline'])}</p>")
    p.append(f"<h1>{_esc(headline)}</h1>")
    p.append(f"<div class=call><b>Captain {_esc(now['captain']['name'])}</b>"
             f"{now['ep']} projected points this week. "
             f"{ctx['league_line']}</div>")

    p.append("<h2>Your team this week</h2>")
    vmax = max([pl["ep"][now["gw"]] for pl in now["xi"]] or [1])
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        group = [pl for pl in now["xi"] if pl["pos"] == pos]
        if not group:
            continue
        p.append(f"<p class=poslabel>{pos}</p>")
        for pl in sorted(group, key=lambda q: -q["ep"][now["gw"]]):
            po = ctx["pack_own"].get(pl["id"], 0) * 100
            mark = " (C)" if pl["id"] == now["captain"]["id"] else ""
            flag = " ⚠" if pl["status"] != "a" else ""
            sub = (f"{pl['opp'][now['gw']]} · {po:.0f}% of your pack own him")
            col = "var(--rise)" if pl["id"] == now["captain"]["id"] else "#5b8db8"
            p.append(_bar_row(pl["name"] + mark + flag, sub,
                              pl["ep"][now["gw"]], vmax, colour=col))
    p.append("<p class=poslabel>BENCH</p>")
    for b in now["bench"]:
        p.append(_bar_row(b["name"], b["opp"][now["gw"]], b["ep"][now["gw"]],
                          vmax, colour="#3d4c5c"))

    if ctx.get("rationale"):
        p.append("<h2>Why</h2>")
        for rat in ctx["rationale"]:
            if not rat["transfers"] and rat["gw"] != w[0]["gw"]:
                continue
            p.append(f"<div class=week><h3>GW{rat['gw']}</h3>")
            p.append(f"<b>Captain {_esc(rat['captain']['name'])}</b><ul>")
            for reason in rat["captain"]["reasons"]:
                p.append(f"<li>{_esc(reason)}</li>")
            p.append("</ul>")
            for t in rat["transfers"]:
                if not t["out"]:
                    p.append(f"<b>In: {_esc(t['in']['name'])}</b>")
                else:
                    p.append(f"<b>{_esc(t['out']['name'])} &rarr; "
                             f"{_esc(t['in']['name'])}</b>")
                    p.append(_compare_svg(t["out"], t["in"]))
                p.append("<ul>")
                for reason in t["reasons"]:
                    p.append(f"<li>{_esc(reason)}</li>")
                p.append("</ul>")
            p.append("</div>")

    p.append("<h2>Next five gameweeks</h2>")
    for wk in w:
        moves = (" · ".join(f"{o} → {i}" for o, i in zip(wk["out"], wk["in"]))
                 or "roll the transfer")
        chip = f" <span class=chip>{wk['chip']}</span>" if wk["chip"] else ""
        hit = f" <span class=fall>−{wk['hits']*4}</span>" if wk["hits"] else ""
        p.append(f"<div class=week><h3>GW{wk['gw']}{chip}{hit}</h3>"
                 f"{_esc(moves)}<div class=note>Captain {_esc(wk['captain']['name'])} · "
                 f"{wk['ep']} projected</div></div>")

    # --- the rebuild ------------------------------------------------------
    wc = ctx.get("wildcard")
    if wc:
        pr = ctx.get("proj") or {}

        def nm(pid):
            return pr[pid]["name"] if pid in pr else str(pid)

        p.append(f"<h2>The GW{wc['for_gw']} wildcard</h2>")
        p.append(f"<div class=call><b>{wc['cost']}m spent, "
                 f"{wc['in_bank']}m in the bank</b>"
                 f"{len(wc['change']['keep'])} of your fifteen survive, "
                 f"{len(wc['change']['buy'])} arrive. Drafted over "
                 f"GW{wc['gws'][0]}–{wc['gws'][-1]}.</div>")
        if wc.get("assumes_no_transfers"):
            p.append("<p class=note>Priced on today's squad value, which "
                     "assumes you make no transfers before the wildcard. "
                     "If you do, the budget moves.</p>")
        if not wc.get("optimal", True):
            p.append("<p class=note>The solver hit its time limit — this is "
                     "the best squad it found, not a proven optimum.</p>")

        squad = sorted(wc["squad"],
                       key=lambda i: (["GKP", "DEF", "MID", "FWD"].index(pr[i]["pos"])
                                      if i in pr else 9, -pr[i]["price"]))
        first = wc["weeks"][0]
        p.append("<table><tr><th>Player</th><th class=n>£</th>"
                 "<th class=n>Proj</th><th>Role</th></tr>")
        for pid in squad:
            q = pr.get(pid, {})
            role = ("XI" if pid in first["xi"] else "bench")
            if pid == first["captain"]:
                role = "captain"
            tag = "rise" if pid in wc["change"]["buy"] else "tag"
            p.append(f"<tr><td class={tag}>{_esc(nm(pid))} "
                     f"<span class=tag>{_esc(q.get('team', ''))} "
                     f"{_esc(q.get('pos', ''))}</span></td>"
                     f"<td class=n>{q.get('price', 0)}</td>"
                     f"<td class=n>{q.get('ep', {}).get(first['gw'], 0):.1f}</td>"
                     f"<td class=tag>{role}</td></tr>")
        p.append("</table>")
        p.append(f"<p class=note>Sold: "
                 f"{_esc(', '.join(nm(i) for i in wc['change']['sell']) or '—')}</p>")

        if wc.get("swaps"):
            p.append("<h2>The closest calls</h2><table><tr><th>Drafted</th>"
                     "<th>Nearest alternative</th><th class=n>Costs you</th></tr>")
            for r in wc["swaps"][:8]:
                gap = "—" if r["gap"] is None else f"{r['gap']:.2f}"
                p.append(f"<tr><td>{_esc(nm(r['out']))}</td>"
                         f"<td>{_esc(nm(r['in']) if r['in'] else '—')}</td>"
                         f"<td class=n>{gap}</td></tr>")
            p.append("</table><p class=note>Points over the whole draft "
                     "horizon, holding the other fourteen fixed. A number near "
                     "zero is a coin toss — take the player you want to watch.</p>")

    if ctx.get("team_form"):
        p.append("<h2>Team form</h2><table><tr><th>Team</th>"
                 "<th class=n>Attack</th><th class=n>Defence</th>"
                 "<th class=n>Form</th></tr>")
        for r in ctx["team_form"][:10]:
            cls = "rise" if r["form"] > 1.03 else ("fall" if r["form"] < 0.97 else "")
            p.append(f"<tr><td>{_esc(r['team'])}</td>"
                     f"<td class=n>{r['attack']:.2f}</td>"
                     f"<td class=n>{r['defence']:.2f}</td>"
                     f"<td class='n {cls}'>{r['form']:.2f}</td></tr>")
        p.append("</table><p class=note>Goals scored and conceded over the "
                 "last few matches against what the fixtures warranted, "
                 "time-decayed and opponent-adjusted. 1.00 is exactly to "
                 "expectation. Defence above 1.00 means leakier than it "
                 "should have been.</p>")

    p.append("<h2>Nobody in your league owns these</h2><table>"
             "<tr><th>Player</th><th class=n>£</th><th class=n>Proj</th>"
             "<th class=n>League</th><th class=n>Pack</th></tr>")
    for g in ctx["gems"][:12]:
        p.append(f"<tr><td>{_esc(g['name'])} <span class=tag>{_esc(g['team'])}</span></td>"
                 f"<td class=n>{g['price']}</td><td class=n>{g['proj']:.1f}</td>"
                 f"<td class=n>{g['league']*100:.0f}%</td>"
                 f"<td class=n>{g['pack']*100:.0f}%</td></tr>")
    p.append("</table>")

    if ctx.get("clean_sheets"):
        p.append("<h2>Clean sheet odds</h2><table><tr><th>Team</th>"
                 "<th class=n>xGC</th><th class=n>Clean sheet</th>"
                 "<th class=n>Concede cost</th></tr>")
        for cs in ctx["clean_sheets"]:
            p.append(f"<tr><td>{_esc(cs['team'])}</td>"
                     f"<td class=n>{cs['xgc']:.2f}</td>"
                     f"<td class=n>{cs['cs_prob'] * 100:.0f}%</td>"
                     f"<td class=n>{cs['concede_cost']:.2f}</td></tr>")
        p.append("</table><p class=note>Expected goals conceded, shrunk toward "
                 "the league average and calibrated against Solio's published "
                 "match odds. Concede cost is the expected points lost to goals "
                 "against, which FPL charges at one per two conceded.</p>")

    p.append("<h2>Price moves tonight</h2><table>"
             "<tr><th>Player</th><th>Direction</th><th class=n>Net transfers</th></tr>")
    for r in ctx["prices"][:12]:
        cls = "rise" if r["direction"] == "rise" else "fall"
        arrow = "rising" if r["direction"] == "rise" else "falling"
        p.append(f"<tr><td>{_esc(r['name'])} <span class=tag>{_esc(r['where'])}</span></td>"
                 f"<td class={cls}>{arrow}</td>"
                 f"<td class=n>{r['net']:+,}</td></tr>")
    p.append("</table>")

    if ctx.get("flagged"):
        p.append("<h2>Flagged in your squad</h2><table>"
                 "<tr><th>Player</th><th>Status</th><th class=n>Proj</th>"
                 "<th>Verdict</th></tr>")
        for fl in ctx["flagged"]:
            p.append(f"<tr><td class=flag>{_esc(fl['name'])}</td>"
                     f"<td class=tag>{_esc(fl['news'] or fl['status'])}</td>"
                     f"<td class=n>{fl['ep']:.1f}</td>"
                     f"<td>{_esc(fl['verdict'])}</td></tr>")
        p.append("</table>")

    p.append(f"<h2>Is a hit worth it?</h2><p>{_esc(ctx['hit_verdict'])}</p>")
    if ctx.get("wc_timing"):
        p.append("<h2>When to wildcard</h2><table><tr><th>Week</th>"
                 "<th class=n>Raw</th><th class=n>Info credit</th>"
                 "<th class=n>Adjusted</th></tr>")
        for w in ctx["wc_timing"]:
            p.append(f"<tr><td>GW{w['gw']}{' (after break)' if w['after_break'] else ''}</td>"
                     f"<td class=n>{w['raw']:.1f}</td><td class=n>+{w['info']:.1f}</td>"
                     f"<td class=n>{w['adjusted']:.1f}</td></tr>")
        p.append("</table><p class=note>Info credit is what waiting is worth: "
                 "more matches played, and fitness resolved after a break.</p>")

    # --- your own plan ---------------------------------------------------
    # Rendered last because it is the part you argue with rather than read.
    pl = ctx.get("planner")
    if pl and pl["plan"]["weeks"]:
        p.append("<h2>Your plan</h2>")

        gap = pl.get("gap")
        if gap is None:
            verdict = "No solver line to compare against."
        elif gap >= -0.5:
            verdict = (f"Your plan is worth {pl['plan']['total_ep']:.1f} points over "
                       f"GW{pl['horizon']['from']}–{pl['horizon']['to']} — level with "
                       "the solver's own line. Back yourself.")
        else:
            verdict = (f"Your plan is worth {pl['plan']['total_ep']:.1f} points over "
                       f"GW{pl['horizon']['from']}–{pl['horizon']['to']}, "
                       f"{abs(gap):.1f} behind the solver's line "
                       f"({pl['solver_total']:.1f}).")
        p.append(f"<div class=call>{_esc(verdict)}</div>")

        if pl["unresolved"]:
            p.append("<p class=note><b>Names not found:</b> "
                     + _esc(", ".join(pl["unresolved"]))
                     + ". Those moves were skipped — check the spelling against "
                     "FPL's own short names.</p>")

        if not pl["plan"]["legal"]:
            p.append("<p class=note><b>This plan cannot be played as written.</b> "
                     "The numbers below assume the illegal moves simply did not "
                     "happen, so treat them as indicative.</p>")
            for v in pl["plan"]["violations"]:
                p.append(f"<p class=flagline>GW{v['gw']}: {_esc(v['message'])}</p>")

        p.append("<table><tr><th>GW</th><th>Chip</th><th>Moves</th>"
                 "<th class=n>FT</th><th class=n>Bank</th><th class=n>Hit</th>"
                 "<th class=n>Points</th></tr>")
        for w in pl["plan"]["weeks"]:
            step = pl["steps"].get(w["gw"]) or pl["steps"].get(str(w["gw"])) or {}
            moves = " · ".join(
                f"{_name(ctx, o)} → {_name(ctx, i)}"
                for o, i in zip(step.get("out") or [], step.get("in") or [])
            ) or "—"
            chip = CHIP_LABEL.get(w["chip"], "—")
            hit = f"{w['hit']}" if w["hit"] else "—"
            p.append(f"<tr><td>GW{w['gw']}</td><td>{_esc(chip)}</td>"
                     f"<td>{_esc(moves)}</td><td class=n>{w['free_transfers']}</td>"
                     f"<td class=n>{w['bank']:.1f}</td><td class=n>{hit}</td>"
                     f"<td class=n>{w['ep']:.1f}</td></tr>")
        p.append("</table>")

        chips = [c for c in pl["chips"].values() if c["ranked"]]
        if chips:
            p.append("<h2>Chip weeks</h2>")
            p.append("<table><tr><th>Chip</th><th>Yours</th><th>Best in horizon</th>"
                     "<th class=n>Worth moving</th></tr>")
            for c in chips:
                where = f"GW{c['current_gw']}" if c["current_gw"] else "unplaced"
                if not c.get("valued", True):
                    # Not "no gain" — not priced. Saying "GW6 (+0.0)" here
                    # would read as a recommendation.
                    best, move = "not priced by the sweep", "—"
                else:
                    best = f"GW{c['best_gw']} ({c['best_gain']:+.1f})"
                    move = ("—" if c["move_gain"] in (None, 0)
                            else f"{c['move_gain']:+.1f}")
                p.append(f"<tr><td>{_esc(c['name'])}</td><td>{_esc(where)}</td>"
                         f"<td>{_esc(best)}</td>"
                         f"<td class=n>{_esc(move)}</td></tr>")
            p.append("</table>")
            p.append("<p class=note>Every free week in the horizon was tried, so "
                     "this is exact — as far as the horizon reaches. If the right "
                     "week for a chip is beyond "
                     f"GW{pl['horizon']['to']}, nothing here can see it. "
                     "Wildcard and Free Hit are not priced here at all: their "
                     "value is the squad they buy, and this sweep holds the "
                     "squad fixed. Compare drafts for each candidate week "
                     "instead.</p>")

    p.append("<footer>Projections blend Solio Analytics' public feed with a local "
             "fallback model. Mini-league ownership from the FPL API. "
             f"Built {_esc(datetime.now(timezone.utc).strftime('%d %b %H:%M UTC'))}. "
             "Recommendations, not instructions — you press the button."
             "</footer></div></body></html>")

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(p))
    return path
