# legacy — the two codebases that were folded in

Everything fantasy-related that used to live elsewhere now lives here, so
there is one repository and one history. Nothing in this directory runs. It
is a parts bin, kept because it contains work that is genuinely better than
what the live engine does in a few specific places, and deleting it would
throw that away.

| | was | lines | status |
|---|---|---|---|
| `fplbrain/` | `assistant-playground` `main`, merged 30 Sep | 12,420 Python | superseded, parts worth porting |
| `ts-fantasy/` | `assistant-playground` branch `claude/fpl-ml-assistant-dm5k6k`, never merged | 4,303 TypeScript | superseded, UCL rules already ported |

## Why there were three engines

Each was written to answer the same question and none knew about the others.
Only `gaffer/` ever ran against live data, because only it runs on GitHub
Actions — the assistant's own container is denied `fantasy.premierleague.com`
and `gaming.uefa.com` by the environment's network policy, so "run it
locally" was never a substitute for either of the other two. That is the
whole reason this consolidation is into gaffer and not into one of them.

## What has already been taken

- **UCL scoring.** `ts-fantasy/lib/fantasy/rules/ucl.ts` was ported to
  `gaffer/ucl/rules.py` and hand-verified against UEFA's published rules.
  It is now the `ucl` entry in `gaffer/games/`.
- **Dixon-Coles scorelines and odds devigging.** The idea came from
  `fplbrain/model/match.py`; the live implementation is in the UCL simulator.
- **Correlated Monte Carlo** — one scoreline drawn per fixture and shared by
  every player in it, so team-mates correlate. From `fplbrain/sim/`.

## What is still worth porting

1. **Points as a full distribution, not a mean** (`fplbrain/model/points.py`).
   The live engine projects an expected value and then picks a captain on it.
   A captain decision is about the tail, not the mean — this is the single
   most valuable thing left in here, and the reason the `ceiling` model was
   bolted on and then had to be switched off for being uncalibrated.
2. **Rival simulation** (`fplbrain/sim/rivals.py`). Beating a mini-league is
   not the same problem as maximising points, and the live engine only
   approximates it with the `rivalry` dial.
3. **The Next.js surface** (`ts-fantasy/app/fantasy/`). The live engine
   publishes a static page; this was interactive. Its cron was removed
   because the Vercel Hobby plan's two cron slots were taken.

Port them properly or delete them. Do not start a fourth engine.
