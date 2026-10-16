# Runbook

How to get an optimised team, every week, without re-deriving any of this.

## The one-liner

> "Optimise my team" — plus a screenshot of your **Transfers** page.

That screenshot carries the four things the public API cannot tell us, and
getting any of them wrong is how a plan turns into fiction:

| What | Where on the screenshot | Why it cannot be fetched |
|---|---|---|
| Your fifteen | the pitch | the API serves the last *finished* gameweek, so moves made for the upcoming deadline are invisible |
| Bank | "Budget" | same |
| Free transfers | "Free Transfers" | never exposed publicly |
| Chips left | the Play buttons | `entry/{id}/history` has them, but nothing reads it yet — see Known gaps |

With that screenshot the run is deterministic. Without it, the engine drafts
against whatever the API last saw, which on 10 Oct 2026 was a squad sharing
three players out of fifteen with reality.

## The weekly sequence

1. **Screenshot the Transfers page** and send it.
2. Config is updated from it: `current_squad`, `bank`, `free_transfers`,
   `chips_used`.
3. The run is dispatched on GitHub Actions — **Actions → gaffer → Run
   workflow**, with `force: true` and the real free-transfer count, plus
   `wildcard: N` when a rebuild is in play.
4. Output lands as a commit: `docs/plan.json`, `docs/data.json`,
   `docs/index.html`.
5. You get the squad, the captain, the chip call and the caveats.

It runs on Actions rather than anywhere else for one reason: **Actions has
network access to the FPL API and Solio.** A sandboxed assistant container
usually does not, so "run it locally" is not a substitute.

Locally, if you want it:

```
pip install -r requirements.txt
python -m gaffer.main --free-transfers 0 --wildcard 7
```

## What it actually pulls

Live, every run, from the official API (`fantasy.premierleague.com/api`):

- `bootstrap-static` — every player, price, ownership, form, status and news,
  plus FPL's own xG, xA, xGI and the raw defensive actions (clearances, blocks,
  interceptions, tackles, recoveries) that DefCon probability is computed from
- `fixtures` — the calendar and difficulty
- `entry/{id}` and `entry/{id}/event/{gw}/picks` — your team as of the last
  finished gameweek
- `entry/{id}/transfers` — your transfer history
- `leagues-classic/{id}/standings` — your mini-league's squads, for effective
  ownership, and **league 314**, the global Overall league, whose top pages are
  the best managers in the world. That is where elite ownership comes from.

Plus `fpl.solioanalytics.com` for the upcoming gameweek's projections and match
odds, and `vaastav/Fantasy-Premier-League` on GitHub for the 84,577
player-gameweeks the models train on.

So yes: player stats are pulled live, and "what everyone else is doing" is
pulled live in three flavours — your league, the pack above you, and the global
elite. Every player card carries `own_league`, `own_pack`, `own_elite` and
`own_overall`. Elite minus overall is the signal; a player at 67% elite and 35%
overall is a crowd that has not caught up.

What is **not** pulled: the FPL content community. Expert wildcard templates
(FPL Mate's 100 Experts, Scout Selection, and the rest) are not scraped, so if
you want the draft compared against them, send a screenshot and it gets
compared by hand.

## The three FPL codebases, and which one to use

| Where | What | Runs? |
|---|---|---|
| **`fpl-gaffer`** (this repo) | Python. Ten trained models, Solio calibration, elite tracking, MILP transfer solve, dedicated wildcard drafter, static web app on Pages. | **Yes — on Actions, against live data. Use this.** |
| `assistant-playground` `main`, `fplbrain/` | Python. Dixon-Coles scorelines, Shin devigging, points as a pmf, correlated simulation, its own MILP. Merged 30 Sep. | Only on your own machine; never run against live data. |
| `assistant-playground` branch `claude/fpl-ml-assistant-dm5k6k`, `lib/fantasy/` | TypeScript. FPL **and UCL Fantasy**, Poisson model, simulate, captain/chips/lineup/transfers optimisers, a Next.js page at `/fantasy`. Unmerged; its cron was removed because the Hobby plan's two cron slots were taken. | No. |

Three engines is two too many. `fpl-gaffer` is the one that executes, so it is
the entry point; the other two are a parts bin. The one thing in `fplbrain`
that gaffer lacks is points as a full distribution rather than a mean — worth
porting if captaincy tail risk ever matters more than it does today.

## Known gaps

**The API read disagreed with reality (10 Oct 2026).** `entry/3907632/event/5/picks`
returned a fifteen sharing only Rogers, João Pedro and Haaland with the
manager's actual GW6 squad. Twelve changes between two gameweeks is impossible
at zero points of hit, so this is not the documented
"transfers-for-the-upcoming-deadline are hidden" blind spot. Unresolved. Until
it is, **the screenshot is authoritative and `current_squad` stays pinned.**
To diagnose: compare `entry/{id}/event/{gw}/picks` across the last few
gameweeks against `entry/{id}/transfers` and see where they diverge.

**`chips_used` is hand-declared and silently decays.** On 10 Oct the config
still planned a Triple Captain that had been spent, and the sweep ranked it the
best move on the page at +9.3 points. Fixed by honouring `chips_used` — but the
declaration itself is still manual. `entry/{id}/history` exposes chips played
publicly; reading it and warning on a mismatch would close this for good.

**`ceiling` is uncalibrated and currently switched off** (`ceiling_weight: 0`).
AUC 0.849 but a median of 0.019 against one player at 0.705. Needs a Platt or
isotonic wrapper and a calibration figure in `scores.json` before it steers the
armband again.

**Wildcard and Free Hit are not priced by the chip sweep.** The sweep moves a
chip's label without moving transfers, and those two chips are worth only the
squad they buy. They now report `valued: false`. Pricing them means re-drafting
per candidate week.

**The publish step pushes without rebasing.** A concurrent push to the same
branch fails the run *after* a successful five-minute solve and the output is
thrown away. Happened once on 10 Oct.

## Dials worth knowing

- `rivalry` (0.35) — 0 is pure expected points, 1 is pure differential-chasing.
  Raise it when you are behind in the mini-league.
- `team_form_weight` (0.35), `team_form_half_life` (2.5 matches) — how hard
  recent team form overrides season-long numbers. This is what makes the draft
  concentrate on whoever is hot; turn it down if that reads as recency bias.
- `info_value` (22.0) — what perfect information is worth on a rebuild, which
  is what pushes a wildcard later. A judgement, not a measurement;
  `timing.breakeven()` says how big it would have to be to flip your decision.
- `min_transfer_gain` (1.5) — the bar a transfer must clear to be suggested.
- `solio_weight` (0.7) — how much to trust Solio over the local model.


## UCL matchday

    # on a runner, which is the only place UEFA is reachable
    python -m gaffer.ucl.main --matchday 3

Or dispatch the `ucl` workflow. It writes `docs/ucl.json` and
`docs/ucl.txt` and commits both, plus a refreshed `gaffer/ucl/snapshot.json`
so the next run works offline.

Before every deadline, in `config/ucl.yaml`:

- **`squad`** — confirm it against the live team. A squad one player wrong
  produces a transfer plan wrong in two places. Names fold accents, full
  stops and spacing, and anything unmatched is printed at the top of the
  run rather than guessed at.
- **`out`** — anyone suspended or injured, by name. UEFA's availability prose
  has never once been ahead of a suspension, and a player left at a
  one-in-ten chance of playing also carries a tenth of a haul, which is
  enough to keep him in an eleven.
- **`odds`** — devigged 1X2 per fixture when a book is reachable. These beat
  the strength model, and the report says `[odds]` or `[model]` per line.
- `free_transfers` is 2 a matchday and at most one rolls.

Known gaps: the MILP picks the eleven against the plain captaincy and the
armband is re-picked exactly afterwards, so the eleven itself is chosen one
step short of optimal; and `MAX_ROLLED_TRANSFERS` is not enforced by the
solver — set `free_transfers` by hand.
