"""Guided setup and a health check.

Two credentials make the engine meaningfully better, and both are fiddly to
obtain by hand. Neither can be obtained *for* you — a session cookie only exists
inside a browser you are logged into, and an API key needs an account in your
name — so the job here is to make supplying them as short as possible, to store
them somewhere that cannot be committed, and above all to verify they actually
work rather than leaving you to find out from a silent fallback three weeks
later.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

from .config import CONFIG_DIR, SECRETS_FILENAME, Settings, get_settings, reset_config_cache
from .data.fpl import AuthRequired, FplClient
from .data.odds import OddsClient
from .util.http import HttpClient, HttpError


@dataclass
class CheckResult:
    """One line of the health report."""

    name: str
    ok: bool
    detail: str
    optional: bool = False

    @property
    def symbol(self) -> str:
        if self.ok:
            return "[green]OK[/]"
        return "[yellow]--[/]" if self.optional else "[red]XX[/]"


def secrets_path(config_dir: Path | None = None) -> Path:
    return (config_dir or CONFIG_DIR) / SECRETS_FILENAME


def write_secret(key: str, value: str, config_dir: Path | None = None) -> Path:
    """Upsert one ``KEY=value`` line in the git-ignored secrets file.

    The file is created readable only by its owner: a session cookie is
    equivalent to a password, and a default-permissions file in a shared
    checkout is a genuine risk rather than a theoretical one.
    """
    path = secrets_path(config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    replaced = False
    if path.exists():
        for raw in path.read_text().splitlines():
            stripped = raw.strip()
            if stripped and not stripped.startswith("#") and stripped.split("=")[0].strip() == key:
                lines.append(f"{key}={value}")
                replaced = True
            else:
                lines.append(raw)
    else:
        lines.append("# Written by `fplbrain setup`. Git-ignored — never commit this file.")
    if not replaced:
        lines.append(f"{key}={value}")

    path.write_text("\n".join(lines).rstrip() + "\n")
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600
    except OSError:  # pragma: no cover - platform dependent
        pass

    os.environ[key] = value
    reset_config_cache()
    return path


def clean_cookie(raw: str) -> str:
    """Normalise a cookie pasted from developer tools.

    People paste all sorts of things here: the whole ``Cookie:`` header, a
    single ``pl_profile=...`` pair, or the bare token. All three are accepted,
    and a leading header name is stripped.
    """
    value = raw.strip()
    if value.lower().startswith("cookie:"):
        value = value.split(":", 1)[1].strip()
    return value


# -- verification -----------------------------------------------------------


def check_entry(settings: Settings, client: FplClient) -> CheckResult:
    entry_id = settings.entry_id
    if entry_id is None:
        return CheckResult(
            "Entry id",
            False,
            "not set — put manager.entry_id in config/settings.yaml, or pass --entry",
        )
    try:
        payload = client.entry(entry_id)
    except HttpError as exc:
        return CheckResult("Entry id", False, f"{entry_id} could not be read: {exc}")

    name = f"{payload.get('player_first_name', '')} {payload.get('player_last_name', '')}".strip()
    team = payload.get("name", "")
    points = payload.get("summary_overall_points")
    rank = payload.get("summary_overall_rank")
    detail = f"{entry_id} — {team or 'unnamed'}"
    if name:
        detail += f" ({name})"
    if points is not None:
        detail += f", {points} pts"
    if rank:
        detail += f", rank {rank:,}"
    return CheckResult("Entry id", True, detail)


def check_api(client: FplClient) -> CheckResult:
    try:
        boot = client.bootstrap()
    except HttpError as exc:
        return CheckResult("FPL API", False, f"unreachable: {exc}")
    next_gw = boot.next_gameweek()
    detail = f"{len(boot.players)} players, {len(boot.teams)} teams"
    if next_gw is not None:
        deadline = next_gw.deadline
        detail += f"; next is GW{next_gw.id}"
        if deadline:
            detail += f" ({deadline:%a %d %b %H:%M} UTC)"
    return CheckResult("FPL API", True, detail)


def check_cookie(settings: Settings, client: FplClient) -> CheckResult:
    if not settings.cookie:
        return CheckResult(
            "FPL session cookie",
            False,
            "not set — the engine will read your squad only after each deadline, "
            "and infer your bank and free transfers. Run `fplbrain setup`.",
            optional=True,
        )
    entry_id = settings.entry_id
    if entry_id is None:
        return CheckResult("FPL session cookie", False, "cannot verify without an entry id")
    try:
        payload = client.my_team(entry_id)
    except AuthRequired as exc:
        return CheckResult("FPL session cookie", False, str(exc), optional=True)
    except HttpError as exc:
        return CheckResult(
            "FPL session cookie",
            False,
            f"rejected by FPL ({exc}) — the session has probably expired; "
            "log in again and re-run `fplbrain setup`",
            optional=True,
        )

    transfers = payload.get("transfers", {}) or {}
    bank = float(transfers.get("bank", 0) or 0) / 10.0
    value = float(transfers.get("value", 0) or 0) / 10.0
    picks = len(payload.get("picks", []) or [])
    return CheckResult(
        "FPL session cookie",
        True,
        f"live squad readable — {picks} players, £{bank:.1f}m bank, £{value:.1f}m value",
    )


def check_odds(settings: Settings) -> CheckResult:
    if not settings.odds_api_key:
        return CheckResult(
            "Bookmaker odds",
            False,
            "no ODDS_API_KEY — projections will use the engine's own team ratings. "
            "Free key at the-odds-api.com, then `fplbrain setup`.",
            optional=True,
        )
    snapshot = OddsClient(settings).fetch()
    if not snapshot.available:
        return CheckResult("Bookmaker odds", False, snapshot.reason, optional=True)
    priced = len(snapshot.matches)
    with_totals = sum(1 for m in snapshot.matches if m.p_over is not None)
    return CheckResult(
        "Bookmaker odds",
        True,
        f"{priced} fixtures priced ({with_totals} with over/under), "
        f"margin removed with {settings.odds.get('devig', 'shin')}",
    )


def check_store(settings: Settings) -> CheckResult:
    from .data.store import Store

    try:
        store = Store(settings.db_path)
        counts = store.counts()
    except Exception as exc:  # noqa: BLE001 - a broken store must not hide the rest
        return CheckResult("Local store", False, f"could not open {settings.db_path}: {exc}")

    results = counts.get("player_gameweek", 0)
    snapshots = counts.get("projection_snapshot", 0)
    if results == 0:
        return CheckResult(
            "Local store",
            False,
            "empty — run `fplbrain pull --history` after a gameweek finishes so the "
            "model has something to learn from and backtest against",
            optional=True,
        )
    return CheckResult(
        "Local store",
        True,
        f"{results} player-gameweeks stored, {snapshots} projection snapshots",
    )


def run_checks(offline: bool = False) -> list[CheckResult]:
    """Everything the engine depends on, verified rather than assumed.

    A health check has to fail fast. The normal client retries four times with
    exponential backoff, which is right for a data pull you want to succeed but
    wrong here: an unreachable API would leave you staring at a spinner for
    minutes before being told what you wanted to know in seconds.
    """
    reset_config_cache()
    settings = get_settings()
    client = FplClient(
        settings,
        offline=offline,
        http=HttpClient(
            settings.cache_dir / "fpl",
            offline=offline,
            max_retries=2,
            timeout=8.0,
        ),
    )

    results = [check_api(client), check_entry(settings, client)]
    results.append(check_cookie(settings, client))
    results.append(check_odds(settings))
    results.append(check_store(settings))
    return results


COOKIE_INSTRUCTIONS = """\
[bold]Getting your FPL session cookie[/] (free — it is just your own login)

  1. Log in at https://fantasy.premierleague.com in a normal browser.
  2. Open developer tools: [cyan]F12[/], or [cyan]Option+Cmd+I[/] on a Mac.
  3. Go to the [bold]Network[/] tab and reload the [bold]My Team[/] page.
  4. Click any request to fantasy.premierleague.com in the list.
  5. Under [bold]Request Headers[/], copy the whole value of [cyan]Cookie:[/].

It is stored in config/secrets.env, readable only by you, and git-ignored.
A session cookie is equivalent to your password: never paste it into a chat,
an issue, or a screenshot.
"""

ODDS_INSTRUCTIONS = """\
[bold]Getting an odds API key[/] (free tier, no card required)

  1. Sign up at https://the-odds-api.com — the free plan allows 500 requests
     a month, and a weekly pull uses about four.
  2. Copy the API key from your dashboard.

Odds are the single biggest accuracy gain available to the engine: bookmakers
price team news and motivation that no public statistic captures.
"""
