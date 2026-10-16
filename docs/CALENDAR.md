# Season calendar — blanks, doubles and European weeks

Source: **Ben Crellin's season calendar**, read from a screenshot on
16 Oct 2026. Recorded here because it is the one input the engine cannot
derive for itself.

## Why this cannot be computed

The FPL fixtures API lists matches that are *scheduled*. A blank gameweek
happens when a club's league match is postponed for a cup tie, and a double
when that match is rescheduled into another week — neither appears in the
API until it happens, often weeks later. The forecast comes from reading cup
progression and the rules for rescheduling, which is a human calendar.
Everything else the engine uses is measured; this is declared, so it carries
a date and a source.

## The forecast

| Gameweek | Date | Call | Confidence |
|---|---|---|---|
| **GW30** | 20 Mar | **Blank** | calendar marks it red |
| **GW32** | 17 Apr | **Double** | "Likely DGW" |
| **GW33** | 24 Apr | **Blank** | red |
| **GW35** | 8 May | **Double** | "Likely DGW" |
| **GW36** | 15 May | **Double** | "Likely DGW" |
| GW37 | 22 May | Possible blank | — |

**Nothing falls before GW19.** That independently confirms what the chip
sweep already said: the first-half Free Hit has no natural home, because
there is no blank to spend it on before it expires at the GW19 deadline.
Two sources, one from fixture structure and one from the model, agreeing
that an unplayed chip is the likely outcome.

## Second-half chip plan

The first-half chips (Wildcard spent GW6, Triple Captain spent GW3, Bench
Boost and Free Hit expiring GW19) cannot reach any of this. The second-half
set can, and this is where they belong:

- **Free Hit → GW30 or GW33.** A blank is the one week a Free Hit is worth
  its name: field eleven players who actually have a fixture, then revert.
  GW33 also collides with the FA Cup semi-finals, which usually makes it the
  emptier of the two.
- **Bench Boost → GW35 or GW36.** Fifteen players with two fixtures each is
  the chip's maximum. Prefer whichever has more of the squad doubling.
- **Triple Captain → GW32.** The earliest likely double, and the one with
  the least risk of a late reschedule unwinding it.
- **Wildcard → GW29 or GW31**, to build a squad loaded on the clubs that
  double in 32/35/36 before the chips are played.

## European weeks

The gameweek that *follows* a European midweek, which is where rotation
shows up:

| Matchday | Date | Hits |
|---|---|---|
| MD2 | 13 Oct | GW7 |
| MD3 | 20 Oct | GW8 |
| MD4 | 3 Nov | GW10 |
| MD5 | 24 Nov | GW12 |
| MD6 | 8 Dec | GW15 |
| MD6 (UECL) | 15 Dec | GW16 |
| MD7 | 19 Jan | GW22 |
| MD8 | 26 Jan | GW23 |

**This corrected an error.** `european_weeks` was `[7, 8, 10, 13, 16, 18]`:
GW13 and GW18 are not European weeks, and GW12, GW15, GW22 and GW23 were
missing. Eight clubs' minutes were being discounted in the wrong weeks.
GW7, GW8 and GW10 were right, which is why nothing in the current five-week
horizon moved.

## Maintenance

A semi-final result moves these. Re-read the calendar when cups resolve —
particularly after the FA Cup 5th round (6 Mar) and the EFL semi-finals
(12 Jan, 2 Feb), which decide most of the second-half reschedules.
