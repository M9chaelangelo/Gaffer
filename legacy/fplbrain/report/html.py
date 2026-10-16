"""A self-contained HTML briefing.

One file, no external assets, readable in any browser and printable. The layout
follows the order the decisions actually get made in: what to do, who to captain,
how to set the bench, whether to play a chip, and what the week is likely to look
like.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Sequence

from ..advisor import Advice
from ..sim.scenarios import compare

_STYLE = """
:root {
  color-scheme: light dark;
  --bg: #ffffff; --fg: #16181d; --muted: #5c6370; --line: #e3e6ec;
  --panel: #f7f8fa; --accent: #128a5b; --warn: #b0451a; --bad: #b3261e;
}
@media (prefers-color-scheme: dark) {
  :root { --bg:#14161a; --fg:#e8eaed; --muted:#9aa1ad; --line:#2a2e36;
          --panel:#1b1e24; --accent:#3ecf8e; --warn:#e2914f; --bad:#f2726a; }
}
* { box-sizing: border-box; }
body { margin:0; padding:2rem 1.25rem 4rem; background:var(--bg); color:var(--fg);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.wrap { max-width: 1040px; margin: 0 auto; }
h1 { font-size: 1.6rem; margin:0 0 .25rem; letter-spacing:-.01em; }
h2 { font-size: 1.05rem; margin:2.4rem 0 .7rem; padding-bottom:.35rem;
     border-bottom:1px solid var(--line); letter-spacing:.02em; text-transform:uppercase;
     color:var(--muted); font-weight:600; }
.sub { color:var(--muted); margin:0 0 1.5rem; }
.headline { background:var(--panel); border:1px solid var(--line); border-left:3px solid var(--accent);
  border-radius:6px; padding:1rem 1.15rem; margin:1.25rem 0 0; font-size:1.05rem; }
table { border-collapse: collapse; width:100%; margin:.5rem 0 1rem; font-variant-numeric: tabular-nums; }
th, td { text-align:left; padding:.45rem .6rem; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-weight:600; font-size:.78rem; text-transform:uppercase; letter-spacing:.04em; }
td.num, th.num { text-align:right; }
tr.pick td { font-weight:600; }
.tag { display:inline-block; padding:.05rem .4rem; border-radius:4px; font-size:.72rem;
       background:var(--panel); border:1px solid var(--line); color:var(--muted); margin-left:.35rem; }
.tag.c { border-color:var(--accent); color:var(--accent); }
.good { color:var(--accent); } .warn { color:var(--warn); } .bad { color:var(--bad); }
ul.notes { padding-left:1.1rem; color:var(--muted); }
ul.notes li { margin:.2rem 0; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.75rem; margin:.75rem 0 1.25rem; }
.stat { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:.7rem .85rem; }
.stat .k { color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.04em; }
.stat .v { font-size:1.35rem; font-weight:600; margin-top:.15rem; }
.bar { height:6px; background:var(--line); border-radius:3px; overflow:hidden; min-width:60px; }
.bar > span { display:block; height:100%; background:var(--accent); }
footer { margin-top:3rem; color:var(--muted); font-size:.8rem; border-top:1px solid var(--line); padding-top:1rem; }
code { background:var(--panel); padding:.05rem .3rem; border-radius:3px; font-size:.85em; }
.scroll { overflow-x:auto; }
"""


def _e(value: Any) -> str:
    return html.escape(str(value))


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]], numeric: Sequence[int] = ()) -> str:
    numeric = set(numeric)
    head = "".join(
        f'<th class="num">{_e(h)}</th>' if i in numeric else f"<th>{_e(h)}</th>"
        for i, h in enumerate(headers)
    )
    body = []
    for row in rows:
        cells = "".join(
            f'<td class="num">{cell}</td>' if i in numeric else f"<td>{cell}</td>"
            for i, cell in enumerate(row)
        )
        body.append(f"<tr>{cells}</tr>")
    return (
        f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _bar(fraction: float) -> str:
    width = max(0.0, min(1.0, fraction)) * 100
    return f'<div class="bar"><span style="width:{width:.0f}%"></span></div>'


def render(advice: Advice) -> str:
    """Render an :class:`Advice` into a standalone HTML page."""
    projections = advice.projections
    parts: list[str] = []

    deadline = (
        advice.deadline.strftime("%A %d %B, %H:%M UTC") if advice.deadline else "unknown"
    )
    parts.append(f"<h1>Gameweek {advice.gameweek}</h1>")
    parts.append(f'<p class="sub">Deadline {_e(deadline)} · squad read from '
                 f'{_e(advice.squad.source)} data · '
                 f'£{advice.squad.bank:.1f}m in the bank · '
                 f'{advice.squad.free_transfers} free transfer(s)</p>')
    parts.append(f'<div class="headline">{_e(advice.headline())}</div>')

    # -- summary tiles -------------------------------------------------------
    if advice.outcome_summary:
        summary = advice.outcome_summary
        tiles = [
            ("Projected points", f"{summary.get('mean', 0):.0f}"),
            ("Range (p10–p90)", f"{summary.get('p10', 0):.0f}–{summary.get('p90', 0):.0f}"),
        ]
        if advice.rank:
            rank = advice.rank.summary()
            tiles.append(("Field average", f"{rank['field_mean']:.0f}"))
            tiles.append(("Green arrow", f"{rank['p_green_arrow']:.0%}"))
        parts.append(
            '<div class="grid">'
            + "".join(
                f'<div class="stat"><div class="k">{_e(k)}</div><div class="v">{_e(v)}</div></div>'
                for k, v in tiles
            )
            + "</div>"
        )

    # -- transfers -----------------------------------------------------------
    parts.append("<h2>Transfers</h2>")
    rows = []
    for index, option in enumerate(advice.transfer_options[:5]):
        if option.is_hold:
            move = "<em>hold — roll the free transfer</em>"
        else:
            outs = ", ".join(_e(projections.name(e)) for e in option.out_elements)
            ins = ", ".join(_e(projections.name(e)) for e in option.in_elements)
            move = f"{outs} &rarr; <strong>{ins}</strong>"
        gain_class = "good" if option.net_gain > 0.5 else ("bad" if option.net_gain < -0.5 else "")
        rows.append(
            [
                ("<strong>&rarr;</strong> " if index == 0 else "") + _e(option.description),
                move,
                f"-{option.hits * 4}" if option.hits else "—",
                f'<span class="{gain_class}">{option.net_gain:+.1f}</span>',
            ]
        )
    parts.append(
        _table(
            ["Option", "Move", "Hit", f"Gain over {len(advice.transfer_options and advice.transfer_options[0].horizon or [])} GWs"],
            rows,
            numeric=(2, 3),
        )
    )

    if advice.plan:
        plan_rows = []
        for decision in advice.plan:
            ins = ", ".join(_e(projections.name(e)) for e in decision.transfers_in) or "—"
            outs = ", ".join(_e(projections.name(e)) for e in decision.transfers_out) or "—"
            plan_rows.append(
                [
                    f"GW{decision.gameweek}",
                    outs,
                    ins,
                    f"{decision.hits * 4}" if decision.hits else "—",
                    f"{decision.free_transfers_before:.0f}",
                    f"{decision.bank_after:.1f}",
                    _e(projections.name(decision.captain)) if decision.captain else "—",
                ]
            )
        parts.append("<h2>Planned path</h2>")
        parts.append(
            _table(
                ["GW", "Out", "In", "Hit", "FT", "Bank", "Captain"],
                plan_rows,
                numeric=(3, 4, 5),
            )
        )

    # -- lineup --------------------------------------------------------------
    parts.append(f"<h2>Starting XI — {_e(advice.lineup.formation_label())}</h2>")
    xi_rows = []
    for element in advice.lineup.starters:
        projection = projections.get(element, advice.gameweek)
        tag = ""
        if element == advice.lineup.captain:
            tag = '<span class="tag c">C</span>'
        elif element == advice.lineup.vice_captain:
            tag = '<span class="tag">V</span>'
        xi_rows.append(
            [
                _e(projections.position(element)),
                _e(projections.name(element)) + tag,
                _e(projections.team_name(projections.team_of(element))),
                f"{projections.expected_points(element, advice.gameweek):.1f}",
                f"{projection.p_haul:.0%}",
                f"{projection.p_blank:.0%}",
            ]
        )
    parts.append(
        _table(
            ["Pos", "Player", "Team", "xPts", "10+", "Blank"], xi_rows, numeric=(3, 4, 5)
        )
    )

    parts.append("<h2>Bench order</h2>")
    bench_rows = [
        [
            str(row["order"]),
            _e(row["name"]),
            _e(row["position"]),
            f"{row['xpts']:.1f}",
            f"{row['p_appear']:.0%}",
            _bar(row["p_appear"]),
        ]
        for row in advice.bench
    ]
    parts.append(
        _table(["#", "Player", "Pos", "xPts", "Plays", ""], bench_rows, numeric=(3, 4))
    )
    parts.append(
        f'<p class="sub">Getting the order right is worth '
        f"{advice.lineup.autosub_value:+.2f} points this week through autosubs.</p>"
    )

    # -- captaincy -----------------------------------------------------------
    parts.append("<h2>Captaincy</h2>")
    captain_rows = []
    for candidate in advice.captains[:8]:
        marker = "<strong>&rarr;</strong> " if candidate.recommended else ""
        captain_rows.append(
            [
                marker + _e(candidate.name),
                f"{candidate.expected_points:.1f}",
                f"{candidate.ceiling:.0%}",
                f"{candidate.floor:.0%}",
                f"{candidate.effective_ownership:.0f}%",
                _bar(candidate.ceiling * 2),
            ]
        )
    parts.append(
        _table(
            ["Player", "xPts", "10+", "Blank", "Eff. own", ""],
            captain_rows,
            numeric=(1, 2, 3, 4),
        )
    )
    if advice.captains and advice.captains[0].reason:
        parts.append(f'<p class="sub">{_e(advice.captains[0].reason)}</p>')

    # -- chips ---------------------------------------------------------------
    if advice.chip_opportunities:
        parts.append("<h2>Chips</h2>")
        chip_rows = []
        for opportunity in advice.chip_opportunities[:12]:
            chip_rows.append(
                [
                    _e(opportunity.label),
                    f"GW{opportunity.gameweek}",
                    f"{opportunity.gain:+.1f}",
                    _e(opportunity.detail[:110]),
                ]
            )
        parts.append(_table(["Chip", "Gameweek", "Gain", "Detail"], chip_rows, numeric=(2,)))
        if advice.chip_advice:
            parts.append(
                '<ul class="notes">'
                + "".join(f"<li>{_e(line)}</li>" for line in advice.chip_advice)
                + "</ul>"
            )

    # -- paths ---------------------------------------------------------------
    if advice.paths:
        parts.append("<h2>Paths compared</h2>")
        path_rows = []
        for row in compare(advice.paths):
            if not row["feasible"]:
                continue
            path_rows.append(
                [
                    _e(row["strategy"]),
                    f"{row['mean']:.1f}",
                    f"{row['p10']:.0f}",
                    f"{row['p90']:.0f}",
                    str(row["hits"]),
                    f"{row['p_beats_hold']:.0%}",
                ]
            )
        parts.append(
            _table(
                ["Strategy", "Mean", "p10", "p90", "Hits", "Beats hold"],
                path_rows,
                numeric=(1, 2, 3, 4, 5),
            )
        )
        parts.append(
            '<p class="sub">Each path is planned by the optimiser and then simulated '
            "over the whole horizon, so the comparison includes the transfers each "
            "plan would go on to make.</p>"
        )

    # -- risks and provenance ------------------------------------------------
    if advice.risks:
        parts.append("<h2>Watch list</h2>")
        parts.append(
            '<ul class="notes">'
            + "".join(f"<li>{_e(risk)}</li>" for risk in advice.risks[:14])
            + "</ul>"
        )

    parts.append("<h2>Where the numbers come from</h2>")
    provenance = list(advice.notes)
    if advice.projections.odds and advice.projections.odds.available:
        provenance.append(
            f"Bookmaker odds for {len(advice.projections.odds.matches)} fixtures, "
            "margin removed with Shin's method."
        )
    provenance.append(
        f"Team ratings fitted on {advice.projections.ratings.matches_used} matches."
    )
    provenance.append(
        f"Projections generated {advice.projections.generated_at:%Y-%m-%d %H:%M} UTC."
    )
    parts.append(
        '<ul class="notes">' + "".join(f"<li>{_e(n)}</li>" for n in provenance) + "</ul>"
    )

    parts.append(
        "<footer>Generated by fplbrain. Projections are estimates: they carry the "
        "uncertainty of the models and data behind them, and a good decision can "
        "still produce a bad gameweek.</footer>"
    )

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>FPL GW{advice.gameweek} briefing</title>"
        f"<style>{_STYLE}</style></head><body><div class='wrap'>"
        + "".join(parts)
        + "</div></body></html>"
    )


def write(advice: Advice, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(advice), encoding="utf-8")
    return path


def write_json(advice: Advice, path: Path) -> Path:
    """A machine-readable dump of the same advice."""
    projections = advice.projections
    payload = {
        "generated_at": advice.generated_at.isoformat(),
        "gameweek": advice.gameweek,
        "deadline": advice.deadline.isoformat() if advice.deadline else None,
        "squad": {
            "source": advice.squad.source,
            "bank": advice.squad.bank,
            "free_transfers": advice.squad.free_transfers,
            "players": [
                {
                    "element": e,
                    "name": projections.name(e),
                    "position": projections.position(e),
                    "price": projections.price(e),
                    "xpts": round(projections.expected_points(e, advice.gameweek), 2),
                }
                for e in advice.squad.element_ids
            ],
        },
        "transfers": [
            {
                "description": option.description,
                "out": [projections.name(e) for e in option.out_elements],
                "in": [projections.name(e) for e in option.in_elements],
                "hits": option.hits,
                "gain": round(option.net_gain, 2),
            }
            for option in advice.transfer_options
        ],
        "lineup": {
            "formation": advice.lineup.formation_label(),
            "starters": [projections.name(e) for e in advice.lineup.starters],
            "bench": [projections.name(e) for e in advice.lineup.bench],
            "captain": projections.name(advice.lineup.captain),
            "vice_captain": projections.name(advice.lineup.vice_captain),
            "expected_points": round(advice.lineup.expected_points, 2),
        },
        "captains": [
            {
                "name": c.name,
                "xpts": round(c.expected_points, 2),
                "ceiling": round(c.ceiling, 4),
                "effective_ownership": round(c.effective_ownership, 2),
            }
            for c in advice.captains[:8]
        ],
        "chips": [
            {
                "chip": o.chip,
                "gameweek": o.gameweek,
                "gain": round(o.gain, 2),
                "detail": o.detail,
            }
            for o in advice.chip_opportunities
        ],
        "paths": compare(advice.paths),
        "gameweek_outcome": advice.outcome_summary,
        "rank": advice.rank.summary() if advice.rank else None,
        "risks": advice.risks,
        "notes": advice.notes,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
