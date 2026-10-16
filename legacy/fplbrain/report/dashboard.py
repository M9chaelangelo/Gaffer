"""A single-file dashboard: pitch view, routes, and analysis panels.

The terminal output is dense and good for a quick check. This is for the ten
minutes before a deadline, when the question is not "what does the model say"
but "show me my team and let me poke at it". So it draws the squad on a pitch
the way every FPL tool does, and puts the reasoning in panels beside it.

Everything is one self-contained HTML file with no external assets, no build
step and no server: the tabs and gameweek switching are a few lines of inline
script. It opens from disk, works offline, and can be mailed to someone.

Four panels, each answering a question a manager actually asks:

    Swing       how far from the template am I, and where?
    Health      is this squad good, and are the fixtures kind?
    Robustness  what breaks this plan — budget, free transfers, autosubs?
    Points DNA  where are my projected points actually coming from?
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..advisor import Advice

# Position colours, used for the shirt blocks on the pitch.
POSITION_COLOUR = {
    "GKP": "#2bb3a3",
    "DEF": "#3d7de8",
    "MID": "#7c5cf0",
    "FWD": "#e0603c",
}


def _e(value: Any) -> str:
    return html.escape(str(value))



def _edge_class(edge: float) -> str:
    return "good" if edge > 0 else "bad"


def _tick(owned: bool) -> str:
    return "&#10003;" if owned else ""


def _marker(recommended: bool) -> str:
    return "&rarr; " if recommended else ""


def _hit_label(hits: int) -> str:
    return str(hits * 4) if hits else "&mdash;"


def _captain_label(projections, captain) -> str:
    return _e(projections.name(captain)) if captain else "&mdash;"


@dataclass
class Panel:
    """One analysis card."""

    key: str
    title: str
    verdict: str
    tone: str  # "good", "warn", "bad"
    body: str

    def render(self) -> str:
        return (
            f'<section class="panel" id="panel-{_e(self.key)}">'
            f'<header><h3>{_e(self.title)}</h3>'
            f'<span class="verdict {_e(self.tone)}">{_e(self.verdict)}</span></header>'
            f"<div class=\"panel-body\">{self.body}</div></section>"
        )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _template_share(advice: Advice) -> tuple[float, list[str], list[str]]:
    """How much of the squad is template, and which players are not.

    Uses the sampled crowd when there is one, because measured ownership among
    good managers is a far better definition of "template" than the overall
    figure, which is dominated by teams nobody has touched since August.
    """
    projections = advice.projections
    squad = advice.squad.element_ids
    if advice.crowd is not None and advice.crowd.available:
        ownership = advice.crowd.ownership()
    else:
        ownership = {
            element: player.selected_by_percent
            for element, player in projections.players.items()
        }

    template, differential = [], []
    for element in squad:
        share = ownership.get(element, 0.0)
        (template if share >= 40 else differential).append(projections.name(element))
    share = 100.0 * len(template) / max(len(squad), 1)
    return share, template, differential


def _swing_panel(advice: Advice) -> Panel:
    share, template, differential = _template_share(advice)
    tone = "good" if 25 <= share <= 70 else "warn"
    verdict = f"{share:.0f}% template"

    rows = "".join(
        f"<li>{_e(name)}</li>" for name in differential[:12]
    ) or "<li>none — this is the template squad</li>"

    source = (
        f"measured across {len(advice.crowd)} {advice.crowd.source} squads"
        if advice.crowd is not None and advice.crowd.available
        else "based on overall ownership, which averages the whole game"
    )

    body = (
        f'<p class="muted">How far your squad sits from the field — {_e(source)}.</p>'
        f'<div class="meter"><span style="width:{min(share, 100):.0f}%"></span></div>'
        f"<p><strong>{len(differential)}</strong> differentials, "
        f"<strong>{len(template)}</strong> template players.</p>"
        f'<ul class="chips">{rows}</ul>'
        "<p class=\"muted\">A squad with no differentials cannot gain rank; one with "
        "no template players loses rank every time the template hauls.</p>"
    )
    return Panel("swing", "Swing", verdict, tone, body)


def _health_panel(advice: Advice) -> Panel:
    summary = advice.outcome_summary
    edge = 0.0
    if advice.rank is not None:
        edge = advice.rank.summary().get("edge", 0.0)
    tone = "good" if edge >= 0 else "warn"
    verdict = f"{edge:+.1f} pts/GW vs field" if advice.rank else "no field comparison"

    projected = summary.get("mean", 0.0)
    low = summary.get("p10", 0.0)
    high = summary.get("p90", 0.0)

    weak = [
        row for row in advice.bench if row["xpts"] > 3.0
    ]  # bench players outscoring a typical starter
    weak_html = (
        "".join(
            f"<li>{_e(row['name'])} on the bench at {row['xpts']:.1f} xPts</li>"
            for row in weak
        )
        or "<li>nothing obviously misplaced</li>"
    )

    body = (
        f'<p class="muted">What this gameweek is likely to bring.</p>'
        f'<div class="stat-row">'
        f'<div class="stat"><div class="k">Projected</div><div class="v">{projected:.0f}</div></div>'
        f'<div class="stat"><div class="k">p10–p90</div><div class="v">{low:.0f}–{high:.0f}</div></div>'
        f'<div class="stat"><div class="k">Autosub value</div>'
        f'<div class="v">{advice.lineup.autosub_value:+.2f}</div></div>'
        f"</div>"
        f"<h4>Bench check</h4><ul>{weak_html}</ul>"
    )
    return Panel("health", "Health", verdict, tone, body)


def _robustness_panel(advice: Advice) -> Panel:
    squad = advice.squad
    risky = [row for row in advice.bench if row["p_appear"] < 0.5]
    blanks = [r for r in advice.risks if "blanks" in r]

    tone = "good"
    problems = []
    if squad.bank < 0.3 and squad.free_transfers <= 1:
        tone = "warn"
        problems.append(
            f"£{squad.bank:.1f}m in the bank with {squad.free_transfers} free transfer — "
            "a price rise could block the move you want."
        )
    if len(risky) >= 3:
        tone = "warn"
        problems.append(
            f"{len(risky)} bench players under 50% to appear, so autosubs may not cover you."
        )
    if blanks:
        tone = "warn"
        problems.extend(blanks[:3])
    if not problems:
        problems.append("Nothing tight — budget, transfers and bench all have room.")

    verdict = "safe" if tone == "good" else f"{len(problems)} to watch"
    body = (
        '<p class="muted">What could break this plan.</p>'
        f'<div class="stat-row">'
        f'<div class="stat"><div class="k">Bank</div><div class="v">£{squad.bank:.1f}m</div></div>'
        f'<div class="stat"><div class="k">Free transfers</div>'
        f'<div class="v">{squad.free_transfers}</div></div>'
        f'<div class="stat"><div class="k">Squad value</div>'
        f'<div class="v">£{squad.squad_value:.1f}m</div></div>'
        f"</div>"
        + "".join(f"<p>{_e(p)}</p>" for p in problems)
    )
    return Panel("robustness", "Robustness", verdict, tone, body)


def _dna_panel(advice: Advice) -> Panel:
    """Where the projected points actually come from."""
    projections = advice.projections
    totals: dict[str, float] = {}
    for element in advice.lineup.starters:
        breakdown = projections.get(element, advice.gameweek).breakdown
        for key, value in breakdown.as_dict().items():
            totals[key] = totals.get(key, 0.0) + value

    labels = {
        "appearance": "Appearance",
        "goals": "Goals",
        "assists": "Assists",
        "clean_sheet": "Clean sheets",
        "bonus": "Bonus",
        "saves": "Saves",
        "defensive_contribution": "Def. contribution",
        "goals_conceded": "Goals conceded",
        "cards": "Cards",
        "penalties": "Penalties",
    }
    colours = {
        "goals": "#3d7de8",
        "assists": "#e0603c",
        "clean_sheet": "#2bb3a3",
        "bonus": "#7c5cf0",
        "defensive_contribution": "#d8a13a",
        "appearance": "#5c6370",
        "saves": "#2b8fb3",
    }

    positive = {k: v for k, v in totals.items() if v > 0}
    total = sum(positive.values()) or 1.0

    bar = "".join(
        f'<span style="width:{100 * value / total:.1f}%;background:{colours.get(key, "#48505e")}"'
        f' title="{_e(labels.get(key, key))}"></span>'
        for key, value in sorted(positive.items(), key=lambda kv: -kv[1])
    )
    rows = "".join(
        f"<tr><td><i style=\"background:{colours.get(key, '#48505e')}\"></i>"
        f"{_e(labels.get(key, key))}</td>"
        f'<td class="num">{value:.1f}</td>'
        f'<td class="num">{100 * value / total:.0f}%</td></tr>'
        for key, value in sorted(positive.items(), key=lambda kv: -kv[1])
    )
    negative = sum(v for v in totals.values() if v < 0)

    body = (
        f'<p class="muted">Your starting XI projects {total:.1f} points from these '
        f"sources, less {abs(negative):.1f} from cards and goals conceded.</p>"
        f'<div class="dna">{bar}</div>'
        f'<table class="mini"><tbody>{rows}</tbody></table>'
    )
    return Panel("dna", "Points DNA", f"{total - abs(negative):.0f} pts", "good", body)


# ---------------------------------------------------------------------------
# Pitch
# ---------------------------------------------------------------------------


def _player_card(advice: Advice, element: int, *, benched: bool = False, order: int | None = None) -> str:
    projections = advice.projections
    position = projections.position(element)
    points = projections.expected_points(element, advice.gameweek)
    projection = projections.get(element, advice.gameweek)

    tag = ""
    if element == advice.lineup.captain:
        tag = '<b class="armband">C</b>'
    elif element == advice.lineup.vice_captain:
        tag = '<b class="armband vice">V</b>'

    player = projections.players.get(element)
    opponent = ""
    for fixture in projection.fixtures:
        side = "H" if fixture.is_home else "A"
        opponent = f"{projections.team_name(fixture.opponent_id)} ({side})"
        break
    if projection.n_fixtures == 0:
        opponent = "blank"
    elif projection.n_fixtures > 1:
        opponent += f" +{projection.n_fixtures - 1}"

    flag = ""
    if player is not None and player.availability < 1.0:
        severity = "out" if player.availability == 0 else "doubt"
        flag = f'<i class="flag {severity}" title="{_e(player.news)}"></i>'

    classes = "card benched" if benched else "card"
    colour = POSITION_COLOUR.get(position, "#48505e")
    order_badge = f'<span class="order">{order}</span>' if order else ""

    return (
        f'<figure class="{classes}" style="--pos:{colour}">'
        f"{order_badge}"
        f'<div class="shirt">{flag}<span class="pts">{points:.1f}</span></div>'
        f"<figcaption><strong>{_e(projections.name(element))}</strong>{tag}"
        f"<small>{_e(opponent)}</small></figcaption></figure>"
    )


def _pitch(advice: Advice) -> str:
    projections = advice.projections
    rows: dict[str, list[int]] = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
    for element in advice.lineup.starters:
        rows[projections.position(element)].append(element)
    for position in rows:
        rows[position].sort(key=lambda e: -projections.expected_points(e, advice.gameweek))

    lines = "".join(
        f'<div class="line">{"".join(_player_card(advice, e) for e in rows[position])}</div>'
        for position in ("GKP", "DEF", "MID", "FWD")
        if rows[position]
    )
    bench = "".join(
        _player_card(advice, element, benched=True, order=index)
        for index, element in enumerate(advice.lineup.bench, start=1)
    )
    return (
        f'<div class="pitch">{lines}</div>'
        f'<div class="bench"><span class="bench-label">Bench</span>{bench}</div>'
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _routes(advice: Advice) -> str:
    if not advice.routes:
        return ""
    cards = []
    for route in advice.routes:
        stats = route.summary()
        decision = next(
            (d for d in route.outcome.plan.decisions if d.gameweek == advice.gameweek), None
        )
        if decision and decision.transfers_in:
            move = " · ".join(
                f"{advice.projections.name(o)} → {advice.projections.name(i)}"
                for o, i in zip(decision.transfers_out, decision.transfers_in)
            )
        else:
            move = "no transfer this week"

        cards.append(
            f'<article class="route{" pick" if route.is_recommendation else ""}">'
            f'<header><span class="role">{_e(route.role)}</span>'
            f"<h4>{_e(route.headline)}</h4></header>"
            f'<p class="move">{_e(move)}</p>'
            f'<div class="stat-row">'
            f'<div class="stat"><div class="k">Mean</div><div class="v">{stats["mean"]:.0f}</div></div>'
            f'<div class="stat"><div class="k">p10</div><div class="v">{stats["p10"]:.0f}</div></div>'
            f'<div class="stat"><div class="k">p90</div><div class="v">{stats["p90"]:.0f}</div></div>'
            f'<div class="stat"><div class="k">Hits</div><div class="v">{stats["hits"]}</div></div>'
            f"</div>"
            f'<p class="muted">{_e(route.rationale)}</p>'
            f"</article>"
        )
    return (
        '<section class="routes"><h2>Three routes from here</h2>'
        '<p class="muted">Each is planned by the optimiser over the full horizon and '
        "then simulated, so the comparison includes the transfers each plan would go "
        "on to make. Pick the one that matches what you are trying to do.</p>"
        f'<div class="route-grid">{"".join(cards)}</div></section>'
    )


def _crowd_section(advice: Advice) -> str:
    crowd = advice.crowd
    if crowd is None or not crowd.available:
        note = crowd.note if crowd is not None else (
            "Run with --crowd to measure what the best managers own."
        )
        return f'<section><h2>The field</h2><p class="muted">{_e(note)}</p></section>'

    from ..data.crowd import compare_to_crowd

    rows = compare_to_crowd(crowd, advice.projections, advice.squad.element_ids, top_n=15)
    body = "".join(
        f"<tr><td>{_e(row.name)}</td><td>{_e(row.position)}</td>"
        f'<td class="num">{row.crowd_ownership:.0f}%</td>'
        f'<td class="num">{row.overall_ownership:.0f}%</td>'
        f'<td class="num {_edge_class(row.edge)}">{row.edge:+.0f}</td>'
        f'<td class="num">{row.crowd_captaincy:.0f}%</td>'
        f"<td>{_tick(row.owned_by_you)}</td></tr>"
        for row in rows
    )
    return (
        f"<section><h2>The field — {_e(crowd.source)}</h2>"
        f'<p class="muted">{len(crowd)} squads from GW{crowd.gameweek}, '
        f"±{crowd.margin_of_error():.0f} percentage points. Squads are public only "
        "after a deadline, so this is last gameweek's teams.</p>"
        '<table><thead><tr><th>Player</th><th>Pos</th><th class="num">Cohort</th>'
        '<th class="num">Overall</th><th class="num">Edge</th>'
        '<th class="num">Capt</th><th>You</th></tr></thead>'
        f"<tbody>{body}</tbody></table></section>"
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

_STYLE = """
:root{--bg:#0d1117;--panel:#151b24;--line:#232c38;--fg:#e6edf3;--muted:#8b949e;
--accent:#f43f5e;--good:#2bb3a3;--warn:#d8a13a;--bad:#e0603c;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1360px;margin:0 auto;padding:1.5rem 1.25rem 4rem}
h1{font-size:1.4rem;margin:0}
h2{font-size:.95rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
margin:2rem 0 .75rem;font-weight:600}
h3{font-size:.95rem;margin:0}
h4{font-size:.85rem;margin:1rem 0 .35rem;color:var(--muted)}
.muted{color:var(--muted)}
.topbar{display:flex;align-items:center;gap:1rem;flex-wrap:wrap;
border-bottom:1px solid var(--line);padding-bottom:1rem;margin-bottom:1.25rem}
.pill{background:var(--panel);border:1px solid var(--line);border-radius:999px;
padding:.3rem .8rem;font-size:.8rem}
.pill b{color:var(--fg)}
.headline{background:linear-gradient(90deg,rgba(244,63,94,.14),transparent);
border-left:3px solid var(--accent);border-radius:6px;padding:.9rem 1.1rem;margin-bottom:1.25rem;
font-size:1.02rem}
.layout{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:1.25rem}
@media(max-width:980px){.layout{grid-template-columns:1fr}}
.pitch{background:
linear-gradient(180deg,#12321f,#0e2718);border:1px solid var(--line);
border-radius:10px 10px 0 0;padding:1.25rem .5rem;display:flex;flex-direction:column;gap:1.1rem}
.line{display:flex;justify-content:center;gap:.5rem;flex-wrap:wrap}
.bench{background:var(--panel);border:1px solid var(--line);border-top:none;
border-radius:0 0 10px 10px;padding:.9rem .5rem;display:flex;justify-content:center;
gap:.5rem;flex-wrap:wrap;align-items:center}
.bench-label{color:var(--muted);font-size:.72rem;text-transform:uppercase;
letter-spacing:.06em;margin-right:.4rem}
.card{margin:0;width:88px;text-align:center;position:relative}
.card .shirt{height:46px;border-radius:6px;background:var(--pos);display:flex;
align-items:center;justify-content:center;position:relative;
box-shadow:inset 0 -14px 18px rgba(0,0,0,.25)}
.card.benched .shirt{opacity:.72;height:38px}
.card .pts{font-weight:700;font-size:1rem;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.5)}
.card figcaption{background:#0b0f15cc;border:1px solid var(--line);border-top:none;
border-radius:0 0 5px 5px;padding:.22rem .25rem;display:block}
.card strong{display:block;font-size:.74rem;white-space:nowrap;overflow:hidden;
text-overflow:ellipsis}
.card small{display:block;color:var(--muted);font-size:.66rem}
.armband{background:var(--accent);color:#fff;border-radius:3px;padding:0 .25rem;
font-size:.6rem;margin-left:.2rem;vertical-align:top}
.armband.vice{background:#48505e}
.order{position:absolute;left:-4px;top:-6px;background:var(--line);color:var(--muted);
border-radius:50%;width:16px;height:16px;font-size:.62rem;line-height:16px;z-index:2}
.flag{position:absolute;top:4px;right:4px;width:8px;height:8px;border-radius:50%}
.flag.doubt{background:var(--warn)}.flag.out{background:var(--bad)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;
padding:.9rem 1rem;margin-bottom:.75rem}
.panel header{display:flex;justify-content:space-between;align-items:center;gap:.5rem}
.verdict{font-size:.75rem;padding:.15rem .5rem;border-radius:999px;border:1px solid var(--line)}
.verdict.good{color:var(--good);border-color:var(--good)}
.verdict.warn{color:var(--warn);border-color:var(--warn)}
.verdict.bad{color:var(--bad);border-color:var(--bad)}
.panel-body{margin-top:.6rem;font-size:.85rem}
.stat-row{display:flex;gap:.5rem;flex-wrap:wrap;margin:.6rem 0}
.stat{flex:1;min-width:76px;background:#0d131b;border:1px solid var(--line);
border-radius:6px;padding:.45rem .55rem}
.stat .k{color:var(--muted);font-size:.66rem;text-transform:uppercase;letter-spacing:.04em}
.stat .v{font-size:1.1rem;font-weight:600}
.meter{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin:.5rem 0}
.meter span{display:block;height:100%;background:var(--accent)}
.dna{display:flex;height:14px;border-radius:4px;overflow:hidden;margin:.6rem 0}
.dna span{display:block}
ul.chips{list-style:none;padding:0;margin:.5rem 0;display:flex;flex-wrap:wrap;gap:.3rem}
ul.chips li{background:#0d131b;border:1px solid var(--line);border-radius:4px;
padding:.12rem .45rem;font-size:.75rem}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:.35rem .5rem;border-bottom:1px solid var(--line);font-size:.82rem}
th{color:var(--muted);font-size:.7rem;text-transform:uppercase;letter-spacing:.04em}
td.num,th.num{text-align:right}
table.mini td{padding:.2rem .3rem;border:none;font-size:.78rem}
table.mini i{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:.4rem}
.good{color:var(--good)}.bad{color:var(--bad)}
.route-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:.75rem}
.route{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:.9rem 1rem}
.route.pick{border-color:var(--accent);box-shadow:0 0 0 1px rgba(244,63,94,.25)}
.route .role{font-size:.66rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.route h4{margin:.15rem 0 .5rem;color:var(--fg);font-size:.95rem}
.route .move{font-size:.84rem;margin:.3rem 0}
.scroll{overflow-x:auto}
footer{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--line);
color:var(--muted);font-size:.78rem}
"""


def render(advice: Advice) -> str:
    """Render the dashboard as one self-contained HTML page."""
    projections = advice.projections
    deadline = (
        advice.deadline.strftime("%a %d %b, %H:%M UTC") if advice.deadline else "unknown"
    )

    panels = "".join(
        panel.render()
        for panel in (
            _swing_panel(advice),
            _health_panel(advice),
            _robustness_panel(advice),
            _dna_panel(advice),
        )
    )

    plan_rows = "".join(
        f"<tr><td>GW{d.gameweek}</td>"
        f"<td>{_e(', '.join(projections.name(e) for e in d.transfers_out) or '—')}</td>"
        f"<td>{_e(', '.join(projections.name(e) for e in d.transfers_in) or '—')}</td>"
        f'<td class="num">{_hit_label(d.hits)}</td>'
        f'<td class="num">{d.free_transfers_before:.0f}</td>'
        f'<td class="num">£{d.bank_after:.1f}m</td>'
        f"<td>{_captain_label(projections, d.captain)}</td></tr>"
        for d in advice.plan
    )

    captain_rows = "".join(
        f"<tr><td>{_marker(c.recommended)}{_e(c.name)}</td>"
        f'<td class="num">{c.expected_points:.1f}</td>'
        f'<td class="num">{c.ceiling:.0%}</td>'
        f'<td class="num">{c.floor:.0%}</td>'
        f'<td class="num">{c.effective_ownership:.0f}%</td></tr>'
        for c in advice.captains[:8]
    )

    chip_rows = "".join(
        f"<tr><td>{_e(o.label)}</td><td>GW{o.gameweek}</td>"
        f'<td class="num">{o.gain:+.1f}</td><td>{_e(o.detail[:90])}</td></tr>'
        for o in advice.chip_opportunities[:10]
    )

    risks = "".join(f"<li>{_e(r)}</li>" for r in advice.risks[:12]) or "<li>nothing flagged</li>"
    notes = "".join(f"<li>{_e(n)}</li>" for n in advice.notes)

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>fplbrain — GW{advice.gameweek}</title><style>{_STYLE}</style></head><body>
<div class="wrap">
  <div class="topbar">
    <h1>Gameweek {advice.gameweek}</h1>
    <span class="pill">Deadline <b>{_e(deadline)}</b></span>
    <span class="pill">Bank <b>£{advice.squad.bank:.1f}m</b></span>
    <span class="pill">FT <b>{advice.squad.free_transfers}</b></span>
    <span class="pill">Squad <b>£{advice.squad.squad_value:.1f}m</b></span>
    <span class="pill">Source <b>{_e(advice.squad.source)}</b></span>
  </div>

  <div class="headline">{_e(advice.headline())}</div>

  <div class="layout">
    <div>{_pitch(advice)}</div>
    <div>{panels}</div>
  </div>

  {_routes(advice)}

  <h2>Planned path</h2>
  <div class="scroll"><table><thead><tr><th>GW</th><th>Out</th><th>In</th>
  <th class="num">Hit</th><th class="num">FT</th><th class="num">Bank</th><th>Captain</th>
  </tr></thead><tbody>{plan_rows}</tbody></table></div>

  <h2>Captaincy</h2>
  <div class="scroll"><table><thead><tr><th>Player</th><th class="num">xPts</th>
  <th class="num">10+</th><th class="num">Blank</th><th class="num">Eff. own</th>
  </tr></thead><tbody>{captain_rows}</tbody></table></div>

  <h2>Chips</h2>
  <div class="scroll"><table><thead><tr><th>Chip</th><th>GW</th>
  <th class="num">Gain</th><th>Detail</th></tr></thead><tbody>{chip_rows}</tbody></table></div>

  {_crowd_section(advice)}

  <h2>Watch list</h2><ul>{risks}</ul>

  <h2>Where the numbers come from</h2><ul class="muted">{notes}</ul>

  <footer>Generated by fplbrain {advice.generated_at:%Y-%m-%d %H:%M} UTC.
  Projections are estimates and carry the uncertainty of the models and data behind
  them; a good decision can still produce a bad gameweek.</footer>
</div></body></html>"""


def write(advice: Advice, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(advice), encoding="utf-8")
    return path
