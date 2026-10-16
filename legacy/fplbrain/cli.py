"""Command line interface.

    fplbrain advise            the full weekly briefing
    fplbrain team              what the engine thinks of your current squad
    fplbrain project           projected points for every player
    fplbrain captain           captaincy options ranked
    fplbrain bench             starting XI and bench order
    fplbrain chips             chip valuations across the horizon
    fplbrain paths             compare strategies by simulation
    fplbrain crowd             what the best managers and your watchlist own
    fplbrain build             build a squad from scratch (pre-season or wildcard)
    fplbrain pull              refresh cached data and store results
    fplbrain train             train the learned model on stored history
    fplbrain backtest          grade past projections against real results
    fplbrain calibrate-bps     recover the true BPS weights from observed data
    fplbrain demo              run the whole thing on a synthetic season
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .advisor import Advisor, format_advice_text
from .config import get_rules, get_settings, reset_config_cache
from .data.fpl import FplClient
from .data.odds import OddsClient
from .data.store import Store
from .model.engine import ProjectionEngine, summarise_projection_set
from .report import dashboard as dashboard_report
from .report import html as html_report

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="A decision engine for Fantasy Premier League.",
)
console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _client(offline: bool = False, demo: bool = False):
    if demo:
        from .data.synth import SyntheticClient, SyntheticSeason

        return SyntheticClient(SyntheticSeason(gameweeks_played=6))
    return FplClient(get_settings(), offline=offline)


def _odds(offline: bool = False, demo: bool = False):
    if demo:
        from .data.odds import OddsSnapshot

        return OddsSnapshot(available=False, reason="demo mode uses model ratings only")
    return OddsClient(get_settings(), offline=offline).fetch()


def _progress(status) -> "callable":
    """Feed engine progress into a rich status line.

    The first projection of a season fetches history for every player behind a
    polite rate limit. That is minutes of work, and a spinner with no numbers on
    it is indistinguishable from a hang.
    """

    def report(done: int, total: int, label: str) -> None:
        pct = (done / total * 100) if total else 100.0
        status.update(f"Fetching {label} — {done}/{total} ({pct:.0f}%)")

    return report


def _entry_id(entry: Optional[int]) -> int:
    resolved = entry or get_settings().entry_id
    if resolved is None:
        console.print(
            "[bold red]No FPL entry id.[/] Set [cyan]manager.entry_id[/] in "
            "config/settings.yaml (or config/settings.local.yaml), or pass "
            "[cyan]--entry 1234567[/].\n"
            "Find it by logging in and reading the number in the URL of your "
            "Points page: fantasy.premierleague.com/entry/[bold]1234567[/]/event/1"
        )
        raise typer.Exit(code=2)
    return resolved


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Log what the engine is doing."),
) -> None:
    _setup_logging(verbose)
    reset_config_cache()


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"fplbrain {__version__}")


@app.command()
def doctor(
    offline: bool = typer.Option(False, "--offline"),
) -> None:
    """Check that everything the engine depends on actually works."""
    from .setup_wizard import run_checks

    with console.status("Checking…"):
        results = run_checks(offline=offline)

    table = Table(title="fplbrain health check")
    table.add_column("")
    table.add_column("Component")
    table.add_column("Status")
    for result in results:
        table.add_row(result.symbol, result.name, result.detail)
    console.print(table)

    missing = [r for r in results if not r.ok and r.optional]
    broken = [r for r in results if not r.ok and not r.optional]
    if broken:
        console.print("\n[red]Something required is not working.[/] Fix those first.")
        raise typer.Exit(code=1)
    if missing:
        console.print(
            "\nThe engine will run without the items marked [yellow]--[/], with "
            "reduced accuracy. Run [cyan]fplbrain setup[/] to add them."
        )
    else:
        console.print("\n[green]Everything is configured.[/]")


@app.command()
def setup(
    skip_cookie: bool = typer.Option(False, "--skip-cookie"),
    skip_odds: bool = typer.Option(False, "--skip-odds"),
) -> None:
    """Add your FPL session cookie and odds API key, and verify both."""
    from .setup_wizard import (
        COOKIE_INSTRUCTIONS,
        ODDS_INSTRUCTIONS,
        check_cookie,
        check_odds,
        clean_cookie,
        secrets_path,
        write_secret,
    )

    console.print("[bold]fplbrain setup[/] — both of these are free and optional.\n")

    if not skip_cookie:
        console.print(COOKIE_INSTRUCTIONS)
        cookie = typer.prompt(
            "Paste your Cookie header (or press Enter to skip)",
            default="",
            show_default=False,
            hide_input=True,
        )
        if cookie.strip():
            path = write_secret("FPL_COOKIE", clean_cookie(cookie))
            console.print(f"Stored in {path} (permissions 600).")
            reset_config_cache()
            result = check_cookie(get_settings(), FplClient(get_settings()))
            console.print(f"  {result.symbol} {result.detail}\n")
        else:
            console.print("[yellow]Skipped.[/] Your squad will be read post-deadline only.\n")

    if not skip_odds:
        console.print(ODDS_INSTRUCTIONS)
        key = typer.prompt(
            "Paste your Odds API key (or press Enter to skip)",
            default="",
            show_default=False,
        )
        if key.strip():
            path = write_secret("ODDS_API_KEY", key.strip())
            console.print(f"Stored in {path}.")
            reset_config_cache()
            result = check_odds(get_settings())
            console.print(f"  {result.symbol} {result.detail}\n")
        else:
            console.print("[yellow]Skipped.[/] Projections will use the engine's ratings.\n")

    console.print(
        f"Secrets live in [cyan]{secrets_path()}[/], which is git-ignored.\n"
        "Run [cyan]fplbrain doctor[/] any time to re-check, and "
        "[cyan]fplbrain advise[/] when you are ready."
    )


@app.command()
def advise(
    entry: Optional[int] = typer.Option(None, "--entry", "-e", help="FPL entry id."),
    gameweek: Optional[int] = typer.Option(None, "--gameweek", "-g"),
    horizon: Optional[int] = typer.Option(None, "--horizon", help="Gameweeks to plan over."),
    max_hits: int = typer.Option(1, "--max-hits", help="Most -4 hits the plan may take."),
    html_out: Optional[Path] = typer.Option(None, "--html", help="Write an HTML briefing here."),
    plain_html: bool = typer.Option(
        False, "--plain-html", help="Use the simple report instead of the dashboard."
    ),
    json_out: Optional[Path] = typer.Option(None, "--json", help="Write machine-readable JSON."),
    no_paths: bool = typer.Option(False, "--no-paths", help="Skip strategy simulation (faster)."),
    no_chips: bool = typer.Option(False, "--no-chips", help="Skip chip valuation (faster)."),
    crowd_sample: int = typer.Option(
        0,
        "--crowd",
        help="Sample this many top managers to measure real captaincy shares.",
    ),
    offline: bool = typer.Option(False, "--offline", help="Use cached data only."),
    demo: bool = typer.Option(False, "--demo", help="Run on a synthetic season."),
    store: bool = typer.Option(True, "--store/--no-store", help="Record projections for backtesting."),
) -> None:
    """The full weekly briefing: transfers, captain, bench, chips and paths."""
    settings = get_settings()
    client = _client(offline, demo)
    entry_id = 1 if demo else _entry_id(entry)

    with console.status("Fetching data and projecting…") as status:
        odds = _odds(offline, demo)
        engine = ProjectionEngine(client, settings=settings, odds=odds)
        boot = client.bootstrap()
        next_gw = boot.next_gameweek()
        start = gameweek or (next_gw.id if next_gw else 1)
        length = horizon or settings.planning_horizon
        projections = engine.run(
            list(range(start, min(start + length, 39))), progress=_progress(status)
        )

    with console.status("Optimising…"):
        advisor = Advisor(client, settings=settings)
        advice = advisor.advise(
            entry_id,
            gameweek=start,
            horizon=length,
            projections=projections,
            max_hits=max_hits,
            include_paths=not no_paths,
            include_chips=not no_chips,
            crowd_sample=crowd_sample,
        )

    console.print(format_advice_text(advice))

    if store and not demo:
        try:
            warehouse = Store(settings.db_path)
            warehouse.record_projections(projections)
            warehouse.log_decision(
                advice.gameweek,
                "advice",
                {"headline": advice.headline(), "squad": advice.squad.element_ids},
            )
        except Exception as exc:  # noqa: BLE001 - storing must never break advice
            console.print(f"[yellow]Could not record snapshot: {exc}[/]")

    if html_out:
        if plain_html:
            html_report.write(advice, html_out)
        else:
            dashboard_report.write(advice, html_out)
        console.print(f"\n[green]Dashboard written to[/] {html_out}")
    if json_out:
        html_report.write_json(advice, json_out)
        console.print(f"[green]JSON written to[/] {json_out}")


@app.command()
def team(
    entry: Optional[int] = typer.Option(None, "--entry", "-e"),
    gameweek: Optional[int] = typer.Option(None, "--gameweek", "-g"),
    offline: bool = typer.Option(False, "--offline"),
    demo: bool = typer.Option(False, "--demo"),
) -> None:
    """Show your current squad with the engine's view of each player."""
    client = _client(offline, demo)
    entry_id = 1 if demo else _entry_id(entry)
    squad = client.squad(entry_id, gameweek)
    boot = client.bootstrap()
    next_gw = boot.next_gameweek()
    start = gameweek or (next_gw.id if next_gw else 1)

    with console.status("Projecting…") as status:
        projections = ProjectionEngine(client, odds=_odds(offline, demo)).run(
            list(range(start, min(start + 5, 39))), progress=_progress(status)
        )

    table = Table(title=f"Entry {entry_id} — squad as of GW{squad.gameweek} ({squad.source})")
    table.add_column("Pos")
    table.add_column("Player")
    table.add_column("Team")
    table.add_column("£m", justify="right")
    for gw in projections.gameweeks:
        table.add_column(f"GW{gw}", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Status")

    order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    for element in sorted(squad.element_ids, key=lambda e: order.get(projections.position(e), 9)):
        player = projections.players.get(element)
        if player is None:
            continue
        cells = [f"{projections.expected_points(element, gw):.1f}" for gw in projections.gameweeks]
        from .data.fpl import summarise_availability

        table.add_row(
            projections.position(element),
            projections.name(element),
            projections.team_name(projections.team_of(element)),
            f"{projections.price(element):.1f}",
            *cells,
            f"{projections.horizon_points(element):.1f}",
            summarise_availability(player)[:38] or "—",
        )
    console.print(table)
    console.print(
        f"Bank £{squad.bank:.1f}m · squad value £{squad.squad_value:.1f}m · "
        f"{squad.free_transfers} free transfer(s) · chips used: "
        f"{', '.join(squad.chips_used) or 'none'}"
    )


@app.command()
def project(
    top: int = typer.Option(30, "--top", "-n"),
    position: Optional[str] = typer.Option(None, "--position", "-p", help="GKP, DEF, MID or FWD."),
    max_price: Optional[float] = typer.Option(None, "--max-price"),
    horizon: Optional[int] = typer.Option(None, "--horizon"),
    offline: bool = typer.Option(False, "--offline"),
    demo: bool = typer.Option(False, "--demo"),
) -> None:
    """Projected points for every player over the horizon."""
    settings = get_settings()
    client = _client(offline, demo)
    boot = client.bootstrap()
    next_gw = boot.next_gameweek()
    start = next_gw.id if next_gw else 1
    length = horizon or settings.planning_horizon

    with console.status("Projecting…") as status:
        projections = ProjectionEngine(client, odds=_odds(offline, demo)).run(
            list(range(start, min(start + length, 39))), progress=_progress(status)
        )

    rows = summarise_projection_set(projections, top_n=10_000)
    if position:
        rows = [r for r in rows if r["position"] == position.upper()]
    if max_price is not None:
        rows = [r for r in rows if r["price"] <= max_price]

    table = Table(title=f"Projections GW{start}–{min(start + length - 1, 38)}")
    for column in ("Player", "Team", "Pos"):
        table.add_column(column)
    for column in ("£m", "xPts", "per £m", "Start%"):
        table.add_column(column, justify="right")
    for row in rows[:top]:
        table.add_row(
            row["name"],
            row["team"],
            row["position"],
            f"{row['price']:.1f}",
            f"{row['xpts']:.1f}",
            f"{row['xpts_per_million']:.2f}",
            f"{row['minutes_p_start']:.0%}",
        )
    console.print(table)
    for note in projections.notes:
        console.print(f"[dim]{note}[/]")


@app.command()
def captain(
    entry: Optional[int] = typer.Option(None, "--entry", "-e"),
    gameweek: Optional[int] = typer.Option(None, "--gameweek", "-g"),
    objective: str = typer.Option("rank", "--objective", help="ev, ceiling or rank."),
    offline: bool = typer.Option(False, "--offline"),
    demo: bool = typer.Option(False, "--demo"),
) -> None:
    """Rank captaincy options for your squad."""
    client = _client(offline, demo)
    entry_id = 1 if demo else _entry_id(entry)
    squad = client.squad(entry_id)
    boot = client.bootstrap()
    start = gameweek or (boot.next_gameweek().id if boot.next_gameweek() else 1)

    with console.status("Projecting…") as status:
        projections = ProjectionEngine(client, odds=_odds(offline, demo)).run(
            [start], progress=_progress(status)
        )

    from .optimize.lineup import captaincy_analysis, choose_starting_eleven
    from .sim.rivals import build_field_model

    starters, _, _ = choose_starting_eleven(squad.element_ids, projections, start)
    field = build_field_model(projections, start, boot.total_players)
    candidates = captaincy_analysis(
        starters,
        projections,
        start,
        objective=objective,
        effective_ownership=field.effective_ownership(),
    )

    table = Table(title=f"Captaincy — GW{start} ({objective} objective)")
    table.add_column("Player")
    for column in ("xPts", "10+", "Blank", "Eff. own", "Utility"):
        table.add_column(column, justify="right")
    for candidate in candidates[:10]:
        table.add_row(
            ("→ " if candidate.recommended else "  ") + candidate.name,
            f"{candidate.expected_points:.1f}",
            f"{candidate.ceiling:.0%}",
            f"{candidate.floor:.0%}",
            f"{candidate.effective_ownership:.0f}%",
            f"{candidate.rank_utility:.2f}",
        )
    console.print(table)
    if candidates and candidates[0].reason:
        console.print(f"[dim]{candidates[0].reason}[/]")


@app.command()
def bench(
    entry: Optional[int] = typer.Option(None, "--entry", "-e"),
    gameweek: Optional[int] = typer.Option(None, "--gameweek", "-g"),
    offline: bool = typer.Option(False, "--offline"),
    demo: bool = typer.Option(False, "--demo"),
) -> None:
    """Pick the starting XI and order the bench."""
    client = _client(offline, demo)
    entry_id = 1 if demo else _entry_id(entry)
    squad = client.squad(entry_id)
    boot = client.bootstrap()
    start = gameweek or (boot.next_gameweek().id if boot.next_gameweek() else 1)

    with console.status("Projecting…") as status:
        projections = ProjectionEngine(client, odds=_odds(offline, demo)).run(
            [start], progress=_progress(status)
        )

    from .optimize.lineup import bench_order_report, optimise_lineup

    lineup = optimise_lineup(squad.element_ids, projections, start)

    table = Table(title=f"Starting XI — {lineup.formation_label()}")
    table.add_column("Pos")
    table.add_column("Player")
    table.add_column("xPts", justify="right")
    for element in lineup.starters:
        tag = " (C)" if element == lineup.captain else (" (V)" if element == lineup.vice_captain else "")
        table.add_row(
            projections.position(element),
            projections.name(element) + tag,
            f"{projections.expected_points(element, start):.1f}",
        )
    console.print(table)

    bench_table = Table(title="Bench order")
    bench_table.add_column("#")
    bench_table.add_column("Player")
    bench_table.add_column("Pos")
    bench_table.add_column("xPts", justify="right")
    bench_table.add_column("Plays", justify="right")
    for row in bench_order_report(lineup.bench, projections, start):
        bench_table.add_row(
            str(row["order"]), row["name"], row["position"],
            f"{row['xpts']:.1f}", f"{row['p_appear']:.0%}",
        )
    console.print(bench_table)
    console.print(f"Autosub value of this order: [bold]{lineup.autosub_value:+.2f}[/] pts")


@app.command()
def chips(
    entry: Optional[int] = typer.Option(None, "--entry", "-e"),
    horizon: int = typer.Option(8, "--horizon"),
    offline: bool = typer.Option(False, "--offline"),
    demo: bool = typer.Option(False, "--demo"),
) -> None:
    """Value every available chip across the horizon."""
    client = _client(offline, demo)
    entry_id = 1 if demo else _entry_id(entry)
    squad = client.squad(entry_id)
    boot = client.bootstrap()
    start = boot.next_gameweek().id if boot.next_gameweek() else 1
    gameweeks = list(range(start, min(start + horizon, 39)))

    with console.status("Projecting…") as status:
        projections = ProjectionEngine(client, odds=_odds(offline, demo)).run(
            gameweeks, progress=_progress(status)
        )

    from .optimize.chips import ChipPlanner

    selling = squad.selling_prices({e: projections.price(e) for e in squad.element_ids})
    planner = ChipPlanner(get_rules())
    with console.status("Valuing chips…"):
        opportunities = planner.opportunities(
            squad.element_ids,
            projections,
            gameweeks=gameweeks,
            bank=squad.bank,
            selling_prices=selling,
            free_transfers=squad.free_transfers,
            chips_used=squad.chips_used,
        )

    table = Table(title=f"Chip valuations GW{gameweeks[0]}–{gameweeks[-1]}")
    table.add_column("Chip")
    table.add_column("GW", justify="right")
    table.add_column("Gain", justify="right")
    table.add_column("Detail")
    for opportunity in opportunities[:20]:
        table.add_row(
            opportunity.label,
            str(opportunity.gameweek),
            f"{opportunity.gain:+.1f}",
            opportunity.detail[:70],
        )
    console.print(table)

    chip_set = get_rules().chip_set_for_gameweek(start)
    deadline = get_rules().chip_window(chip_set)[1] if chip_set else None
    for line in planner.recommend(opportunities, deadline_gameweek=deadline):
        console.print(f"• {line}")


@app.command()
def paths(
    entry: Optional[int] = typer.Option(None, "--entry", "-e"),
    horizon: int = typer.Option(6, "--horizon"),
    draws: int = typer.Option(3000, "--draws"),
    offline: bool = typer.Option(False, "--offline"),
    demo: bool = typer.Option(False, "--demo"),
) -> None:
    """Compare strategies by planning each one and simulating it."""
    client = _client(offline, demo)
    entry_id = 1 if demo else _entry_id(entry)

    with console.status("Running scenarios…"):
        advisor = Advisor(client)
        advice = advisor.advise(
            entry_id,
            horizon=horizon,
            include_rank=False,
            include_paths=True,
            include_chips=True,
            draws=draws,
        )

    from .sim.scenarios import compare

    table = Table(title=f"Strategy comparison over {horizon} gameweeks")
    table.add_column("Strategy")
    for column in ("Mean", "p10", "p90", "Hits", "Beats hold"):
        table.add_column(column, justify="right")
    for row in compare(advice.paths):
        if not row["feasible"]:
            continue
        table.add_row(
            row["strategy"],
            f"{row['mean']:.1f}",
            f"{row['p10']:.0f}",
            f"{row['p90']:.0f}",
            str(row["hits"]),
            f"{row['p_beats_hold']:.0%}",
        )
    console.print(table)


@app.command()
def build(
    budget: float = typer.Option(100.0, "--budget"),
    horizon: int = typer.Option(6, "--horizon"),
    offline: bool = typer.Option(False, "--offline"),
    demo: bool = typer.Option(False, "--demo"),
) -> None:
    """Build a squad from scratch — for pre-season or a wildcard."""
    client = _client(offline, demo)
    boot = client.bootstrap()
    start = boot.next_gameweek().id if boot.next_gameweek() else 1
    gameweeks = list(range(start, min(start + horizon, 39)))

    with console.status("Projecting…") as status:
        projections = ProjectionEngine(client, odds=_odds(offline, demo)).run(
            gameweeks, progress=_progress(status)
        )

    from .optimize.milp import SquadOptimiser, shortlist

    with console.status("Optimising squad…"):
        result = SquadOptimiser().solve(
            projections,
            gameweeks=gameweeks,
            current_squad=[],
            bank=budget,
            free_transfers=15,
            candidate_pool=shortlist(projections),
            time_limit=180,
        )

    if not result.ok or not result.decisions:
        console.print(f"[red]Optimiser returned {result.status}[/]")
        raise typer.Exit(code=1)

    decision = result.decisions[0]
    table = Table(title=f"Optimal squad for GW{start}–{gameweeks[-1]} (£{budget:.1f}m)")
    table.add_column("Pos")
    table.add_column("Player")
    table.add_column("Team")
    table.add_column("£m", justify="right")
    table.add_column("xPts", justify="right")
    table.add_column("XI")

    order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    for element in sorted(decision.squad, key=lambda e: order[projections.position(e)]):
        table.add_row(
            projections.position(element),
            projections.name(element),
            projections.team_name(projections.team_of(element)),
            f"{projections.price(element):.1f}",
            f"{projections.horizon_points(element, gameweeks):.1f}",
            "start" if element in decision.starters else "bench",
        )
    console.print(table)
    spend = sum(projections.price(e) for e in decision.squad)
    console.print(
        f"Spend £{spend:.1f}m · bank £{budget - spend:.1f}m · "
        f"captain {projections.name(decision.captain)}"
    )


@app.command()
def crowd(
    entry: Optional[int] = typer.Option(None, "--entry", "-e"),
    gameweek: Optional[int] = typer.Option(None, "--gameweek", "-g"),
    sample: int = typer.Option(100, "--sample", help="How many top managers to sample."),
    source: str = typer.Option(
        "Top 100",
        "--source",
        help="A cohort name, or: bands (Top 100/1k/10k), cohorts, all.",
    ),
    offline: bool = typer.Option(False, "--offline"),
    demo: bool = typer.Option(False, "--demo"),
) -> None:
    """What the best managers — and anyone you track — actually own.

    Squads are public only after a deadline has passed, so this always shows the
    most recent completed gameweek, never the one you are currently setting.
    """
    from .data.crowd import CrowdClient, compare_to_crowd, differentials

    client = _client(offline, demo)
    entry_id = 1 if demo else _entry_id(entry)
    boot = client.bootstrap()

    finished = boot.finished_gameweeks()
    if gameweek is None:
        if not finished:
            console.print(
                "[yellow]No gameweek has finished yet, so nobody's squad is public.[/]\n"
                "This becomes useful once GW1 has been played."
            )
            raise typer.Exit(code=0)
        gameweek = max(finished)

    from .data.crowd import DEFAULT_RANK_BANDS, load_cohorts

    crowd_client = CrowdClient(client, get_settings())
    configured = load_cohorts()

    if source == "bands":
        specs = DEFAULT_RANK_BANDS
    elif source == "cohorts":
        specs = configured
    elif source == "all":
        specs = list(DEFAULT_RANK_BANDS) + configured
    else:
        specs = [s for s in (list(DEFAULT_RANK_BANDS) + configured) if s.name.lower() == source.lower()]
        if not specs:
            specs = [DEFAULT_RANK_BANDS[0]]

    if not specs:
        console.print(
            "[yellow]No cohorts configured.[/] Add some to config/watchlist.yaml, "
            "or use [cyan]--source bands[/] for the standard rank bands."
        )
        raise typer.Exit(code=0)

    snapshots = []
    with console.status("Reading squads…") as status:
        for spec in specs:
            snapshots.append(
                crowd_client.cohort(spec, gameweek, sample, progress=_progress(status))
            )

        # The crowd is read from the last completed gameweek, but the points
        # column has to look forward — projecting a gameweek that has already
        # been played returns zero for everyone, since it has no fixtures left.
        next_gw = boot.next_gameweek()
        projection_gameweek = next_gw.id if next_gw else gameweek
        projections = ProjectionEngine(client, odds=_odds(offline, demo)).run(
            [projection_gameweek], progress=_progress(status)
        )

    try:
        squad = client.squad(entry_id).element_ids
    except Exception:  # noqa: BLE001 - the crowd view is useful without your squad
        squad = []

    for snapshot in snapshots:
        if not snapshot.available:
            console.print(f"[yellow]{snapshot.source}:[/] {snapshot.note}")
            continue

        margin = snapshot.margin_of_error()
        table = Table(
            title=(
                f"{snapshot.source.title()} ownership — GW{snapshot.gameweek}, "
                f"{len(snapshot)} squads (±{margin:.0f}pp)"
            )
        )
        table.add_column("Player")
        table.add_column("Pos")
        for column in ("£m", "Crowd", "Overall", "Edge", "Capt", f"xPts GW{projection_gameweek}"):
            table.add_column(column, justify="right")
        table.add_column("You")

        for row in compare_to_crowd(snapshot, projections, squad, top_n=20):
            edge = f"{row.edge:+.0f}" if row.overall_ownership else "—"
            table.add_row(
                row.name,
                row.position,
                f"{row.price:.1f}",
                f"{row.crowd_ownership:.0f}%",
                f"{row.overall_ownership:.0f}%",
                edge,
                f"{row.crowd_captaincy:.0f}%" if row.crowd_captaincy else "—",
                f"{row.expected_points:.1f}",
                "yes" if row.owned_by_you else "",
            )
        console.print(table)

        chips = snapshot.chip_usage()
        if chips:
            console.print(
                "Chips played: "
                + ", ".join(f"{name} {share:.0f}%" for name, share in chips.items())
            )

        if squad:
            yours, theirs = differentials(snapshot, squad)
            if yours:
                console.print(
                    "[bold]Your differentials:[/] "
                    + ", ".join(projections.name(e) for e in yours[:10])
                )
            if theirs:
                console.print(
                    "[bold]Template you are missing:[/] "
                    + ", ".join(projections.name(e) for e in theirs[:10])
                )
        console.print()


@app.command()
def pull(
    history: bool = typer.Option(True, "--history/--no-history", help="Store per-player results."),
    force: bool = typer.Option(False, "--force", help="Bypass the cache."),
    demo: bool = typer.Option(False, "--demo"),
) -> None:
    """Refresh cached data and record results into the local store."""
    settings = get_settings()
    client = _client(False, demo)
    warehouse = Store(settings.db_path)

    boot = client.bootstrap(force_refresh=force)
    console.print(f"Players: {len(boot.players)} · teams: {len(boot.teams)}")
    fixtures = client.fixtures(force_refresh=force)
    console.print(f"Fixtures: {len(fixtures)} ({sum(1 for f in fixtures if f.finished)} played)")

    if history:
        from .data.fpl import iter_player_gameweeks

        elements = [p.id for p in boot.players if p.minutes > 0 or p.price >= 4.5]
        with console.status(f"Fetching history for {len(elements)} players…"):
            rows = list(iter_player_gameweeks(client, elements))
        stored = warehouse.record_history(rows)
        console.print(f"Stored {stored} player-gameweek rows.")

    odds = OddsClient(settings).fetch(force_refresh=force)
    if odds.available:
        console.print(f"Odds: {len(odds.matches)} fixtures priced.")
    else:
        console.print(f"[yellow]Odds unavailable:[/] {odds.reason}")

    console.print(warehouse.counts())


@app.command()
def train(
    model_path: Path = typer.Option(Path("data/model.pkl"), "--out"),
    demo: bool = typer.Option(False, "--demo"),
) -> None:
    """Train the learned model on stored history."""
    settings = get_settings()
    client = _client(False, demo)
    warehouse = Store(settings.db_path)

    from .model.ml import (
        PointsLearner,
        attach_snapshots,
        build_training_frame,
        choose_blend_weight,
        positions_from_bootstrap,
    )

    history = warehouse.history()
    if not history:
        console.print(
            "[red]No stored history.[/] Run [cyan]fplbrain pull --history[/] first "
            "(and note that a model needs several gameweeks of results to be useful)."
        )
        raise typer.Exit(code=1)

    positions = positions_from_bootstrap(client.bootstrap())
    frame = build_training_frame(history, positions)
    snapshots: list[dict] = []
    for gameweek in warehouse.gameweeks_recorded():
        snapshots.extend(warehouse.latest_snapshots(gameweek))
    frame = attach_snapshots(frame, snapshots)

    learner = PointsLearner()
    try:
        report = learner.fit(frame)
    except ValueError as exc:
        console.print(f"[yellow]{exc}[/]")
        raise typer.Exit(code=1) from exc

    console.print(report.summary())
    weight = choose_blend_weight(report)
    console.print(f"Suggested blend weight on the learned model: [bold]{weight:.2f}[/]")

    table = Table(title="Most useful features")
    table.add_column("Feature")
    table.add_column("Importance", justify="right")
    for name, importance in learner.feature_importance()[:12]:
        table.add_row(name, f"{importance:.0f}")
    console.print(table)

    learner.save(model_path)
    console.print(f"[green]Model saved to[/] {model_path}")


@app.command()
def backtest(
    demo: bool = typer.Option(False, "--demo"),
) -> None:
    """Grade past projections against what actually happened."""
    settings = get_settings()
    warehouse = Store(settings.db_path)
    from .model.ml import backtest_snapshots

    try:
        result = backtest_snapshots(warehouse)
    except ValueError as exc:
        console.print(f"[yellow]{exc}[/]")
        raise typer.Exit(code=1) from exc

    console.print(result.summary())

    table = Table(title="Per gameweek")
    for column in ("GW", "n", "MAE", "Spearman"):
        table.add_column(column, justify="right")
    for row in result.per_gameweek:
        table.add_row(
            str(row["gameweek"]), str(row["n"]), f"{row['mae']:.2f}", f"{row['spearman']:.3f}"
        )
    console.print(table)

    calibration = Table(title="Calibration — is a 6-point projection worth 6 points?")
    for column in ("Bin", "n", "Predicted", "Actual", "Bias"):
        calibration.add_column(column, justify="right")
    for row in result.calibration:
        calibration.add_row(
            str(row["bin"]),
            str(row["n"]),
            f"{row['predicted_mean']:.2f}",
            f"{row['actual_mean']:.2f}",
            f"{row['bias']:+.2f}",
        )
    console.print(calibration)


@app.command("calibrate-bps")
def calibrate_bps_command(
    out: Path = typer.Option(Path("data/bps_calibrated.json"), "--out"),
    demo: bool = typer.Option(False, "--demo"),
) -> None:
    """Recover the true BPS weights by regressing observed BPS on match stats."""
    settings = get_settings()
    client = _client(False, demo)
    warehouse = Store(settings.db_path)

    from .model.ml import calibrate_bps, positions_from_bootstrap, write_calibrated_bps

    history = warehouse.history()
    positions = positions_from_bootstrap(client.bootstrap())
    try:
        calibration = calibrate_bps(history, positions)
    except ValueError as exc:
        console.print(f"[yellow]{exc}[/]")
        raise typer.Exit(code=1) from exc

    for position, block in calibration["positions"].items():
        table = Table(
            title=f"{position} — fitted BPS weights "
            f"(n={block['n']}, mean abs error {block['mean_abs_error']})"
        )
        table.add_column("Statistic")
        table.add_column("BPS", justify="right")
        for name, weight in sorted(block["weights"].items(), key=lambda kv: -abs(kv[1])):
            table.add_row(name, f"{weight:+.2f}")
        console.print(table)
        if not block["confident"]:
            console.print(
                "[yellow]Fit is not tight yet — more gameweeks will sharpen it.[/]"
            )

    write_calibrated_bps(calibration, out)
    console.print(f"[green]Written to[/] {out}")


@app.command()
def demo(
    html_out: Optional[Path] = typer.Option(None, "--html"),
    gameweeks_played: int = typer.Option(6, "--played", help="How far into the fake season."),
) -> None:
    """Run the whole engine on a synthetic season — no network, no account."""
    from .data.synth import SyntheticClient, SyntheticSeason

    client = SyntheticClient(SyntheticSeason(gameweeks_played=gameweeks_played))
    with console.status("Running the full pipeline on a synthetic season…"):
        advisor = Advisor(client)
        advice = advisor.advise(1, horizon=4, include_paths=True, include_chips=True, draws=2000)
    console.print(format_advice_text(advice))
    if html_out:
        dashboard_report.write(advice, html_out)
        console.print(f"\n[green]Dashboard written to[/] {html_out}")


if __name__ == "__main__":  # pragma: no cover
    app()
