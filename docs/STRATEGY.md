# Strategy — GW6 to GW19

Entry 3907632. Written 10 Oct 2026 from the GW6 solve (`solved_at` 08:41 UTC),
run on GitHub Actions against the live FPL API and Solio.

**Built on the squad from the manager's own Transfers screen, not the API read.**
The two disagree — see Known gaps in `RUNBOOK.md`.

## Where things stand

| | |
|---|---|
| Next deadline | GW6, Sat 10 Oct 10:00 UTC |
| Budget (selling value + bank) | **£99.1m** |
| Free transfers | **0** — every GW6 move costs 4 |
| Chips spent | Triple Captain |
| Chips held | Wildcard, Bench Boost, Free Hit — all expire at the GW19 deadline, Sat 2 Jan 13:30 GMT |

## This week: GW6 — no transfer

**Do nothing. Fix the lineup instead.**

The solver's own line proposes Ballard → Thomas, but that carries `hits: 1`,
and with zero free transfers it costs 4 points. Thomas projects 1.87 this week
against Ballard's 0.00, so the move buys about +1.9 for −4. The engine's own
verdict agrees and is blunter:

> Taking a hit gains about −12.4 points over 5 gameweeks. Not worth it — the
> bar is 6.0.

Ballard is sold for free on the wildcard in seven days. Paying 4 points to be
rid of him a week early is paying for nothing.

What does need doing, and is free:

| Player | Status | Start prob | GW6 projection |
|---|---|---|---|
| **Ballard** (SUN) | `i` — hamstring, unknown return | — | **0.00** |
| Robinson (FUL) | `d` — ankle, 75% | 0.87 | 1.94 |
| Konsa (ARS) | `d` — unspecified, 75% | 0.83 | 2.18 |

**Ballard must not be in your eleven** — he is a guaranteed zero, and an
auto-sub only covers him if someone on your bench actually plays. Check the
bench order so the first substitute is a nailed starter.

## GW7: the wildcard

Drafted over GW7–13 on £99.1m, fully spent, £0.0m left. Three of the current
fifteen survive: **Haaland, Groß, Kelleher.**

Sold: Palmer, Saka, Rogers, João Pedro, Gonzalo, Hill, Ballard, Konsa, Leno,
Robinson, Crooks, Greaves.

| | Player | Club | £ | Owned | Elite | XI |
|---|---|---|---|---|---|---|
| GKP | Trafford | LEE | 5.0 | 5.5% | 7% | ● |
| GKP | Kelleher | BRE | 5.0 | 6.1% | 2% | |
| DEF | Mukiele | SUN | 5.4 | 1.9% | 1% | ● |
| DEF | Muharemović | LEE | 5.0 | 4.5% | 10% | ● |
| DEF | Silva | BOU | 5.0 | 0.5% | 0% | ● |
| DEF | Khalaili | CRY | 5.0 | 0.3% | 0% | |
| DEF | Davis | IPS | 4.0 | 7.7% | 9% | |
| MID | B.Fernandes | MUN | 11.9 | 37.6% | 23% | ● |
| MID | Cunha | MUN | 7.9 | 6.0% | 2% | ● |
| MID | Schade | BRE | 6.2 | 15.8% | 26% | ● |
| MID | Stach | LEE | 6.0 | 2.4% | 2% | ● |
| MID | Groß | BHA | 5.9 | 35.6% | 66% | ● |
| FWD | Haaland | MCI | 15.6 | 74.2% | 88% | ● |
| FWD | Kostoulas | BHA | 5.6 | 8.1% | 1% | ● |
| FWD | Emersonn | IPS | 5.6 | 4.1% | 0% | |

**GW7 XI (3-5-2), captain Haaland.** Bench: Khalaili, Davis, Emersonn, Kelleher.

Projected: GW7 81.0, GW8 71.2, **GW9 86.4 (Bench Boost)**, GW10 74.1,
GW11 80.1, GW12 74.0, GW13 64.5.

`optimal: false` — the draft hit its 180-second limit without proving
optimality. A good feasible answer, not a proven best one.

### How far this sits from the crowd

Against FPL Mate's **100 Experts wildcard template** (sample of 14), the draft
shares five of fifteen: Silva, Muharemović, Groß, Schade, Haaland.

It declines nine the experts are heavily on — Calafiori (100% of experts),
Rogers (86%), Saka (86%), Gonzalo (86%), João Pedro (71%), Hall (57%),
Raya (57%), De Cuyper (50%), Barnes (36%) — and adds B.Fernandes at £11.9m,
whom no expert in that sample holds.

That is a large differential bet. It is either edge or model error, and the
engine's own honesty about the points model (7% better than "he'll score what
he's been averaging") argues for humility. The two picks to weigh:

- **Groß — take it.** 35.6% overall against **66% elite**. The best managers
  are nearly twice as likely to own him, and the experts are at 100%.
- **B.Fernandes — the open question.** 37.6% overall, **23% elite**: the one
  place the crowd is ahead of the elite, and it is the second-largest spend in
  the squad. Elite-minus-overall is the signal this engine is built to read,
  and here it points against the pick.

Independent corroboration from the community charts: FPL Labs' DefCon-versus-xGI
plot puts **Muharemović in the "doing both" quadrant**, which supports that
pick; it puts **Davis in "attacking only"**, which is a weaker basis for a
defender who is there to be cheap and start.

## Chips

**Wildcard — GW7.** Not GW6. The fixture swing lands at GW7 (Bournemouth and
Spurs improve, Fulham begin Hull and Coventry, City host Ipswich), and GW6's
team news is unresolved where GW7's will not be. This is a judgement, not a
model output — the sweep structurally cannot price a wildcard week.

**Bench Boost — GW9, worth about 7.6 points.** The one chip the sweep genuinely
prices:

| Week | Gain |
|---|---|
| GW9 | **+7.57** |
| GW8 | +7.27 |
| GW6 | +6.87 |
| GW10 | +6.42 |

A 1.15-point spread across four weeks — the week is close to noise, the
magnitude is the finding. Better than the 4.5 the previous (wrong) squad
produced, but still modest.

**Free Hit — unplaced.** No blanks before the GW19 expiry, so there is no week
it belongs in. Spend it as one week of fixture arbitrage between GW12 and GW18.
An unplayed chip scores zero.

## What this engine is not telling you

Four findings from 10 Oct, all cases where the output was confident and wrong.

1. **The squad it drafted against was not yours.** The API read for this entry
   shared three players out of fifteen with the real team. Now pinned from the
   screenshot. Root cause unresolved — `RUNBOOK.md`.
2. **The sweep recommended spent chips.** `chips_used_before` existed with a
   violation ready; nothing passed it. It ranked a Triple Captain at +9.27, the
   best number on the page, for a chip long gone. Now honoured, and the run
   cross-checks `chips_used` against the entry's public history.
3. **The armband was steered by an uncalibrated model.** `ceiling` AUC 0.849,
   median 0.019, one player at 0.705, Haaland at 0.11, and several of the top
   ceilings belonging to players projected for zero. `ceiling_weight: 0`.
4. **Wildcard and Free Hit were never priced** and read as though they were.
   Now `valued: false`.

Standing limitations: free transfers and the current squad are declared, not
discovered; the points model's edge is 7%; Solio only publishes the upcoming
gameweek, so GW8+ is the local model alone; today's Solio calibration was
×1.006 on 50 shared players.
