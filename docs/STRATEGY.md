# Strategy — GW6 to GW19

Entry 3907632. Written 10 Oct 2026 from the GW6 solve (`docs/plan.json`,
`solved_at` 08:01 UTC). Every number here came out of the engine running on
GitHub Actions against the live FPL API and Solio, not from a hand estimate.

Re-read this against a fresh solve before acting on it. It is a record of what
the model thought on one morning, which is the only thing that makes it
possible to find out later whether it was any good.

## Where things stand

| | |
|---|---|
| Next deadline | GW6, Sat 10 Oct 10:00 UTC |
| Budget (selling value + bank) | **£99.4m** |
| Chips spent | Triple Captain |
| Chips held | **Wildcard, Bench Boost, Free Hit** — all expire at the GW19 deadline, Sat 2 Jan 13:30 GMT |
| Free transfers assumed | 1 (declared, not discovered — see below) |

The budget is the number to notice. £99.4m is roughly £1m below a naive
"squad value plus bank" estimate, because selling prices surrender half of
every rise. A plan drafted against £100.5m is drafted against money that does
not exist.

## This week: GW6

**One transfer, free, no hit: Isak → Kostoulas.**

Isak is out. The solver takes the free transfer to replace him and does nothing
else. On hits it is explicit:

> Taking a hit gains about −6.4 points over 5 gameweeks. Not worth it — the bar
> is 6.0. Roll the transfer instead.

Do not wildcard today. The wildcard is GW7.

## GW7: the wildcard

Drafted over GW7–13 on £99.4m. **£99.1m spent, £0.3m left.** The solve proved
optimality within its pool of 168 players (`optimal: true`), which the earlier
run of the morning did not.

One of the current fifteen survives: Haaland.

| | Player | Club | £ | Owned | Elite |
|---|---|---|---|---|---|
| GKP | Trafford | LEE | 5.0 | 5.5% | 6% |
| GKP | Kelleher | BRE | 5.0 | 6.1% | 2% |
| DEF | Mukiele | SUN | 5.4 | 1.9% | 1% |
| DEF | Muharemović | LEE | 5.0 | 4.5% | 10% |
| DEF | Silva | BOU | 5.0 | 0.5% | 0% |
| DEF | Khalaili | CRY | 5.0 | 0.3% | 0% |
| DEF | Davis | IPS | 4.0 | 7.6% | 8% |
| MID | B.Fernandes | MUN | 11.9 | 37.6% | 22% |
| MID | Cunha | MUN | 7.9 | 6.0% | 2% |
| MID | Schade | BRE | 6.2 | 15.6% | 26% |
| MID | Stach | LEE | 6.0 | 2.4% | 2% |
| MID | Groß | BHA | 5.9 | 35.2% | 67% |
| FWD | Haaland | MCI | 15.6 | 74.2% | 87% |
| FWD | Kostoulas | BHA | 5.6 | 7.8% | 1% |
| FWD | Emersonn | IPS | 5.6 | 4.1% | 0% |

Spend by position: GK £10.0m, DEF £24.4m, MID £37.9m, FWD £26.8m.

**GW7 XI (3-5-2), captain Haaland:** Trafford; Silva, Muharemović, Mukiele;
Groß, B.Fernandes, Schade, Cunha, Stach; Haaland, Kostoulas.
Bench in order: Khalaili, Davis, Emersonn, Kelleher.

Projected run: GW7 80.8, GW8 71.0, **GW9 86.2 (Bench Boost)**, GW10 73.9,
GW11 79.9, GW12 73.8, GW13 64.3. Total 465.7 over seven weeks.

Captain by week: Haaland (7), B.Fernandes (8), Haaland (9), B.Fernandes (10),
Haaland (11), Groß (12), B.Fernandes (13).

### Why it looks like this

`team form (weight 0.35): BHA 1.36, LEE 1.17, MCI 1.12, BRE 1.12`

The draft is a bet on four in-form clubs. Three Leeds players (at the
three-per-club cap), two Brighton, two Brentford, two Man United, two Ipswich.
That is the mechanism, and it is also the risk: `team_form_half_life` is 2.5
matches, so this is six games of evidence with the most recent weighted
heaviest. If Brighton's run was a good fortnight rather than a new level, this
squad is wrong in a correlated way rather than in fifteen independent ways.

### The two picks to argue with

**B.Fernandes at £11.9m is the squad's open question.** 37.6% of managers own
him and only 22% of the elite do — the one place in the squad where the best
managers are *underweight* relative to the crowd. He is also the second-largest
spend. Elite ownership minus overall ownership is the signal this engine is
built to read, and here it points the other way.

**Groß is the opposite, and the strongest single endorsement in the draft:**
35.2% overall against **67% elite**. The best managers are twice as likely to
own him as the field.

### Closest calls

For each pick, the nearest alternative and what taking it would cost over the
horizon:

| Pick | Alternative | Cost of switching |
|---|---|---|
| Muharemović | Bogle | **−0.10** |
| Kelleher | Tzolakis | 0.37 |
| Emersonn | McBurnie | 0.50 |
| Stach | Anderson | 0.63 |
| Cunha | Anderson | 0.73 |
| Davis | Thomas | 0.94 |

Muharemović's is negative: Bogle scores 0.1 higher in isolation and is excluded
by the three-per-club limit on Leeds, not by merit. Anything under about half a
point is a coin toss — take the player you would rather watch.

## Chips

**Wildcard — GW7.** Not GW6, for two reasons that are about fixtures and
information rather than projections. The swing lands at GW7 (Bournemouth and
Spurs both improve, Fulham begin Hull and Coventry, City host Ipswich), and
GW6's team news is unresolved in a way GW7's will not be — João Pedro is
reported back on 17 Oct, Isak and Gakpo are out.

This is a judgement, not a model output, and the honest reason is below: the
sweep cannot price a wildcard week at all.

**Bench Boost — GW9, and it barely matters.** This is the one chip the sweep
genuinely prices, and the answer is nearly flat:

| Week | Gain |
|---|---|
| GW9 | **+4.67** |
| GW6 | +4.49 |
| GW10 | +4.16 |
| GW8 | +4.02 |

A 0.65-point spread across four weeks. GW9 wins, but anyone claiming the week
matters here is reading noise. The more useful reading is the magnitude: **a
Bench Boost on this squad is worth about four and a half points.** That is a
thin return for a chip, and it is what a £0.3m bench of Khalaili, Davis,
Emersonn and Kelleher buys. If you want more from it, the squad has to be built
for it, which costs points in the other eleven.

**Free Hit — unplaced, and that is deliberate.** There are no blank gameweeks
before the GW19 expiry, so there is no week it obviously belongs in. Treat it as
one week of fixture arbitrage and spend it on the worst fixture set between
GW12 and GW18. The only genuinely wrong move is letting it expire: an unplayed
chip scores zero.

## What this engine is not telling you

Three findings from reading the solve on 10 Oct. All three were cases where the
report was confident and wrong, which is worse than silent.

**1. The sweep used to recommend chips that were already spent.**
`evaluate()` has always accepted `chips_used_before` and carried a `chip-spent`
violation for it, but nothing passed the argument and `grade()` swept all four
chips unconditionally. On this entry it ranked a Triple Captain in GW8 at
**+9.27 points** — the largest number on the page, for a chip gone weeks
earlier. Fixed: `chips_used` in `config.yaml` is now read and honoured.

**2. The armband was steered by an uncalibrated model.** `ceiling` scores AUC
0.849, which is why it was trusted. But AUC is rank-based and says nothing
about whether outputs are probabilities, and `ceiling_weight` multiplies the
probability directly. Across 667 players the median ceiling was 0.019 and the
mean 0.028; B.Fernandes read **0.705**, about 25σ out, while Haaland — the
likeliest hauler in the game — read 0.11, below Dibling and Rice. Four of the
top ten belonged to players projected for roughly zero points, so it was not
conditioning on minutes either. `ceiling_weight` is now **0.0**. Restore it
once `ceiling` is wrapped in a calibrator and `scores.json` records calibration
error beside the AUC.

**3. Wildcard and Free Hit are not priced by the sweep, and used to look as
though they were.** `sweep_chip` moves a chip's *label* between weeks and never
moves transfers. A wildcard is worth exactly the squad it buys, so its gain was
identically 0.0 in every week — and the report rendered that as
`best_gw: 6, best_gain: 0.0`, which reads as "GW6 is the best wildcard week".
It was the first of four ties. Both chips now report `valued: false` and name
no best week. Pricing them properly means re-drafting per candidate week, which
is a different order of cost from the rest of the sweep.

Standing limitations, unchanged and documented upstream:

- **Free transfers are declared, not discovered.** The public API does not
  expose them. The run above assumed 1. Getting it wrong is the commonest cause
  of a bad plan.
- **The wildcard assumes no transfers between now and GW7** — true here, since
  the GW6 move is a like-for-like injury replacement.
- The points model beats its naive baseline by 7%. That is the honest edge.
- The Bench Boost term in the weekly solve is an approximation, not a 15-man
  solve.
- Solio only publishes the upcoming gameweek, so GW8+ projections are the local
  model alone. Today's calibration against Solio was ×1.006 on 50 shared
  players — the local model is currently in line with theirs, which was not
  true earlier in the season.

## Weekly rhythm

1. Make the transfer in the FPL app.
2. Update `current_squad` and `bank` in `config.yaml` if you moved for a
   deadline that has not passed — the API cannot see it.
3. Keep `chips_used` current the moment a chip is played.
4. Trigger the run with your real free-transfer count.
5. Read `docs/index.html`, and treat it as an argument.
