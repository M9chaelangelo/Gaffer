# Learnings

Written 16 Oct 2026, after six gameweeks of FPL and two UCL matchdays, from
evidence in this repository rather than from general principle. The manager
is bottom of the mini-league this engine was built to win, so the list below
is an account of why, not a victory lap.

---

## 1. Every single defect was a confident wrong answer, never a crash

Not one bug found in six weeks raised an exception. Each was legal,
well-formatted, in-range output from working code fed a wrong assumption:

| What it said | What was true | Cost |
|---|---|---|
| Triple Captain in GW8, ranked best on the page at **+9.27** | spent in GW3 | would have been unplayable |
| Wildcard in GW7, eight free transfers | spent in GW6 | corrupted all five weeks of the line |
| Bench Boost week worth **60.2**, the worst of five | worth **82.5**, the best | nearly moved the chip to the wrong week |
| Team sheet benching Groß (7.42) and B.Fernandes (5.92) to start Schade (2.64, injured) | — | ~8 points if followed without the chip |
| Squad pinned from the Transfers screen | three days out of date | GW7 drafted against fifteen players no longer owned |

**A test suite cannot catch this class.** The code was correct; the
assumption was not. What catches it is asserting what must be true of *any*
answer — now `gaffer/guards.py`, with one test per bug above.

## 2. A fix on an unmerged branch is not a fix

The `chips_available` fix was written on 13 October and sat on
`claude/gw7-plan` while `main` kept the bug and went on producing Wildcard
recommendations for a spent chip. The UCL rules sat unmerged on
`claude/ucl-support` for the entire UCL campaign; the advice actually given
for MD2 came from a standalone script, not the engine.

**Merge it or it does not exist.**

## 3. Fixing the symptom where you find it leaves the root cause

`chips_used` was honoured in the chip sweep and not in the solver one layer
below — the same bug, in two places, found a day apart. The second cost a
whole corrupted five-week plan.

**When a wrong assumption is found, grep for every consumer of it.**

## 4. The model rewards luck — this is the live edge-destroyer

`form_weight: 0.30` blends in each player's recent **points**. Points include
over-performance. So a player finishing above his expected goals is rewarded
for the luck, and the model projects it forward.

Measured on 16 Oct, with the new overdue tables:

| | Underlying support | Model said | Gap |
|---|---|---|---|
| Saka | 5.89/wk | 5.58 | −0.31 |
| **Semenyo** | **3.02/wk** | **6.47** | **+3.45** |

Semenyo has banked ~3.2 goal involvements above expectation; his xG90 is
0.057 against Saka's 0.646. The engine recommended selling Saka to buy him.
**That recommendation was wrong and was withdrawn.**

A model that chases returns instead of the process behind them is
systematically buying high and selling low — the precise opposite of an edge.

## 5. An uncalibrated probability must never reach a decision

The `ceiling` model scores AUC 0.849 — a respectable *ranking*. But AUC says
nothing about whether the numbers are probabilities, and the captaincy code
multiplied the probability directly. Median 0.019, one player at 0.705
(~25σ), Haaland — the likeliest hauler in the game — at 0.11, and several of
the highest "ceilings" belonging to players projected for roughly zero.

At `ceiling_weight: 6.0` that handed B.Fernandes about +4.2 points of
phantom armband preference. Now 0.0, and it stays there until a Platt or
isotonic wrapper and a calibration error live in `scores.json` next to the
AUC.

**Report calibration beside every AUC, or do not consume the number.**

## 6. Declared state decays silently

`current_squad`, `bank`, `free_transfers` and `chips_used` are hand-kept,
because the public API exposes none of them for the upcoming deadline. Three
of the four have already gone stale once. The pin now carries
`squad_pinned_for` and a guard refuses a plan built on a pin from an earlier
gameweek.

**Anything declared by hand needs an expiry and a check.**

## 7. Nothing was being scored, so nothing could improve

This is the big one, and it is the reason for the rest. GW6: projected 68.8,
returned 30. No record, no post-mortem, no adjustment. Six gameweeks of
predictions evaporated.

`learn/ledger.py` now records every projection before kickoff and joins the
result afterwards, reporting MAE, **bias**, RMSE and skill against the naive
baseline. Everything contested above — does `form_weight` help, is Solio
better than the local model, is the chip sweep calibrated — becomes a
measurement instead of an argument.

## 8. Three engines meant two parts bins

`fpl-gaffer`, `fplbrain` and a TypeScript engine, all answering the same
question, only one of them running. Consolidated; see `legacy/README.md` for
what is worth porting out of the other two. **Points as a distribution
rather than a mean** is the best thing still sitting unported, and it is
exactly what captaincy needs.

## 9. The network policy is load-bearing, not an annoyance

The container cannot reach `fantasy.premierleague.com`, `gaming.uefa.com` or
Fantasy Football Scout. GitHub Actions runners can. Every design decision
follows from that: the solve runs in CI, commits its output, and the
assistant reads the commit.

The same constraint has an upside. Fantasy Football Scout's "Goals Imminent"
tables are Opta-derived and paywalled — but FPL's own free API now carries
the same Opta xG and xA, per player and per gameweek. `gaffer/imminent.py`
reproduces the signal from the allowed source. **Check whether the expensive
source's underlying data is already free somewhere.**

## 10. Separate "he's due" from "he's playing well"

The overdue tables keep these apart deliberately. Season xG above goals
scored is a *finishing deficit* — the weak half, because variance regresses
slowly and some players simply finish badly. Recent xGI per 90 is the strong
half, and needs no belief in regression at all. Ranking on debt alone is how
"he's due" became the last word of many bad transfers.

---

## What these imply for actually winning

1. **Score before tuning.** Let the ledger run until there is a bias number,
   then change one dial at a time.
2. **Turn `form_weight` down.** Point 4 says it imports noise. Measure it.
3. **Port points-as-a-distribution.** Captaincy is a tail decision.
4. **Trust process over returns.** Buy the xG, sell the over-performance.
   Being bottom of the league is an argument for *more* discipline here,
   not less.
5. **Differentials must be chosen, not defaulted into.** `rivalry: 0.35`
   silently moved the armband off Haaland onto a 0% owned player projecting
   1.3 points lower, while the report called it "highest projection" — which
   was false. Leverage is a decision to make deliberately when behind.
