# Architecture

One engine, two games, and a ledger that scores it.

```
gaffer/
  games/            what differs between competitions, and nothing else
    __init__.py       the Game descriptor + registry
    fpl/rules.py      FPL scoring, squad rules, chips
    ucl/rules.py      re-export of gaffer/ucl/rules.py
  ucl/              UEFA feed client, scoring, snapshot
  learn/
    train.py          the ten models (was gaffer/learn.py)
    ledger.py         predictions in, outcomes back, a score
  guards.py         invariants that fail loudly
  imminent.py       who is due a goal or an assist, from Opta xG/xA
  …                 projections, optimise, planner, wildcard, report, …
legacy/             fplbrain and the TypeScript engine, absorbed, not running
```

## The three layers

**Measure.** `fetch` (FPL), `ucl/fetch` (UEFA), Solio, bookmaker odds,
`vaastav/Fantasy-Premier-League` for 84,577 historical player-gameweeks.
Everything downstream is only as good as this, so what is measured and what
is estimated is marked at every boundary.

**Decide.** `projections` turns measurements into expected points;
`optimise` is the MILP that picks transfers, lineup and chips over a
five-week horizon; `planner` grades a plan you write yourself and sweeps
each chip across every free week; `wildcard` drafts fifteen from nothing.

**Check.** `guards` asserts invariants before a human reads anything.
`learn/ledger` records every projection and joins it to the actual result.

## Why a Game object

FPL and UCL ask the same question over different scoring. Until now that was
two codebases, and the UCL one had never produced a plan. A `Game` carries
the whole of the difference — squad shape, budget, chips, the scoring
function, and the strategic flags:

| | FPL | UCL |
|---|---|---|
| Vice-captain | yes | **no** |
| Armband movable mid-round | no | **yes**, to anyone yet to kick off |
| Free transfers per round | 1 | 2 |
| Defensive returns | threshold: 10/12 actions, then 2 pts | **rate: 1 per 3 recoveries, no threshold** |

Those last two rows change strategy, not arithmetic. A holding midfielder is
a lottery ticket in FPL and a floor in UCL. A captain who plays on day one of
a UCL matchday buys a free second swing that has no FPL analogue.

## The ledger is the point

Ten trained models and a MILP, and until 16 Oct 2026 nothing recorded
whether any of it was right. GW6 projected an eleven at 68.8 and returned 30;
nothing noticed.

`learn/ledger.py` writes two append-only JSON Lines files under
`docs/ledger/`: what was predicted, before kickoff, and what happened, after.
They join on `(game, gw, player_id)`. Re-solving a gameweek appends rather
than overwrites — changing your mind is data, and silently replacing a
prediction after seeing the result is how a model marks its own homework.

It reports MAE, **bias**, RMSE and skill against the honest baseline of
predicting each player's own mean. Bias is the one to watch: noise is
tolerable, a model that is consistently high has a flaw.

This is what makes every other question answerable — whether `form_weight`
helps or hurts, whether Solio beats the local model, whether the chip sweep
is calibrated. Before the ledger those were opinions.

## Why it runs on GitHub Actions

The assistant's container is denied `fantasy.premierleague.com` and
`gaming.uefa.com` by the environment's network policy. Actions runners are
not. That single constraint decides the whole deployment: the solve runs in
CI, commits its output, and the assistant reads the committed result.

## The UCL side

`gaffer/ucl/` is now a plan, not a silo. It had sat half-wired for weeks
while the advice actually given came from a standalone script; the pieces
each worked and nothing joined them, which is precisely the failure a test
of the pieces cannot catch. `tests/test_ucl.py` ends with an end-to-end case
for that reason.

    fetch.py      UEFA's one working endpoint, read defensively
    strength.py   team goal rates from the feed, when no book is reachable
    project.py    correlated Monte Carlo over one matchday
    plan.py       the MILP, and the armband
    main.py       fetch -> project -> optimise -> report -> commit

Three things differ from FPL in ways that change the answer, not just the
arithmetic:

**The armband moves.** There is no vice-captain; instead the captaincy may be
moved during a matchday to a player who has not kicked off. So the captaincy
is worth `E[max(s, E[f])]` where `s` is the Tuesday captain's realised score
and `f` the best Wednesday fallback — never less than `max(E[s], E[f])`, and
the gap is a free second swing with no FPL analogue. `plan.armband` computes
it from the simulated samples and names both players.

**Recoveries pay 1 per 3 with no threshold**, where FPL's defensive
contribution is a cliff at 10. A holding midfielder is a floor here and a
lottery ticket there.

**Goal rates are modelled, not devigged.** The FPL side gets its rates from
bookmaker odds. For UCL the runner usually cannot reach a book, so
`strength.py` fits them from what the feed carries: goals so far, plus a
squad-value prior whose exponent is *fitted* on this season's own scoring
rather than asserted. Measured odds in `config/ucl.yaml` always win, and
every fixture line in the report says which it used.
