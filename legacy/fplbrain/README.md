# fplbrain

A decision engine for Fantasy Premier League. It reads your team, projects every
player as a full probability distribution rather than a single number, and then
solves the actual decisions you face each week: transfers, captaincy, the
starting XI, bench order, and when to play each chip.

Built for the **2026/27** rules — two sets of four chips, defensive contribution
points, the reworked Bonus Points System, and up to five banked free transfers.

```
$ fplbrain advise --entry 1234567

Gameweek 12 — deadline Sat 21 Nov 11:00 UTC
============================================================

TRANSFERS
 -> One free transfer: out Gvardiol / in Virgil (+3.4 pts over 6 GWs)
    Hold and roll the transfer: hold, roll the free transfer (+0.0 pts over 6 GWs)
    Two moves, taking a hit if it pays: out Gvardiol, Sarr / in Virgil, Rogers,
      -4 hit (+1.9 pts over 6 GWs)

STARTING XI (3-5-2)
  ...
CAPTAIN
 -> Haaland            8.2 xPts  ceiling 38.4%  EO  71.2%
    Saka               6.1 xPts  ceiling 24.1%  EO  22.8%
    8.2 xPts; 38% chance of 10+; 71% owned; +2.1 xPts clear of Saka

CHIPS
  Bench Boost: best in GW14 (+18.2 pts) — play it. ...
  Wildcard: best in GW12 (+9.4 pts) — hold. ...
```

---

## What it actually does

### Projections are distributions, not point estimates

Most tools give you an expected-points number. That is the wrong object for most
FPL decisions. Captaincy is a bet on the upper tail, bench order is a bet on the
probability of a blank, and chip timing depends on the shape of the whole
distribution. So every player-fixture projection here is a probability mass
function over points, built by convolving the individual scoring components:

```
no appearance   →  0 points, nothing else can happen
under 60 mins   →  1 appearance point, no clean sheet, rates scaled down
60+ mins        →  2 appearance points, clean sheet live
```

Within each branch, goals, assists, clean sheets, goals conceded, saves, cards,
defensive contribution and bonus are convolved together, and the branches are
mixed by their probabilities. That gives you `p_haul` (10+ points), `p_blank`
(2 or fewer), quantiles, and a mean — from one coherent model.

### The football model

| Layer | What it does |
|---|---|
| **Team strength** | Attack and defence ratings fitted by weighted Poisson MLE on results, with exponential time decay and a ridge pull toward FPL's own published strength grades — so it works in gameweek 3, not just gameweek 30. |
| **Market odds** | 1X2 and over/under markets, margin removed with **Shin's method** (better than proportional scaling for favourites, which is exactly where captaincy is decided), then solved back into each side's expected goals under a Dixon-Coles model. Odds override the ratings model where they exist. |
| **Scorelines** | Dixon-Coles adjusted double Poisson, which corrects the independent model's known underestimate of 0-0, 1-0, 0-1 and 1-1. Clean sheets and goals-conceded deductions both come from this joint distribution. |
| **Minutes** | Kept separate and explicit, because minutes dominate everything. Start probability is a recency-weighted rate shrunk toward a price-based prior, discounted by injury flags and fixture congestion. |
| **Per-90 rates** | xG and xA rather than goals and assists, shrunk toward positional priors by how many 90s a player has actually played, blended with last season where the API exposes it. |
| **Defensive contribution** | +2 for 10 CBIT (defenders) or 12 CBIRT (midfielders and forwards). Modelled as the tail of an overdispersed count — a negative binomial, because defensive actions come in bunches and Poisson understates the threshold. Blended with a player's observed hit rate once he has enough games. |
| **Bonus** | BPS is simulated for all 22 players in a fixture and the top three are awarded 3-2-1, because bonus is a within-match ranking and cannot be computed one player at a time. |

### Optimisation is a real MILP, not a heuristic

The weekly question is not "who is the best player to bring in". It is "which
sequence of transfers over the next N weeks maximises points, given a budget,
three-per-club limits, free transfers that bank up to five, and the option to
take -4 hits". That is a mixed-integer program and it is solved as one, with
squad, starting XI, captain, buy and sell variables per gameweek, and proper
flow constraints on money and free transfers.

Chips are given to the model as fixed decisions, so the chip planner can
enumerate scenarios and compare them like for like.

### Simulation is correlated

Adding up independent player distributions badly understates how a gameweek
really behaves. Team-mates share a clean sheet — three defenders from one club
is one bet, not three — and they share goals. So the simulator works top-down:
it draws a scoreline for each fixture, then attributes those goals and assists
to individuals, then resolves minutes, defensive actions, cards, saves and
bonus.

The analytic and simulated models are built completely differently and are
checked against each other in the test suite; they agree to within a few percent
across the league.

### Rank, not just points

Points are a means to an end. `fplbrain` builds a synthetic field of thousands of
rival managers sampled from real ownership, scores them against the same
simulated gameweek your squad faces, and reads your rank distribution off the
result. Rivals pick their XI in advance, like everyone else — letting them
choose the best eleven from the simulated results would hand the whole field
about ten points of hindsight and make your own rank look far worse than it is.

**Effective ownership** (ownership plus captaincy, since captains count twice) is
what actually moves rank, and it feeds the captaincy objective directly.

### What everyone else is doing

Overall ownership is published, and the engine already uses it. But it averages
ten million managers, most of whom set a squad in August and stop. Two sharper
signals are available:

```bash
fplbrain crowd                       # the top 100
fplbrain crowd --source bands        # top 100, top 1k and top 10k side by side
fplbrain crowd --source all          # rank bands plus your configured cohorts
fplbrain crowd --source "Experts"    # one named cohort
fplbrain advise --crowd 200          # measure real captaincy shares, not modelled
```

**Rank bands** need no setup: Top 100, Top 1k and Top 10k are always available,
because the overall league contains everyone. A band is sampled *across* its
range rather than from the front — estimating what the top 10k own by reading
the top 100 would measure the top 100.

**Cohorts** are anything else, defined in `config/watchlist.yaml`: a classic
league (which is how a group of experts or creators is usually published — one
id covers the whole group), or a list of named individuals.

**Elite ownership** samples the top of the overall league. A player on 8% of all
teams but 40% of the top thousand is a different proposition to one at 40%
everywhere, and the table reports both alongside the gap between them.

**A watchlist** tracks named managers by entry id — content creators, a rival in
your mini-league, anyone whose thinking you want to see. Add them to
`config/watchlist.yaml`; the id is the number in the URL of their team page.
That file ships empty on purpose, because a guessed id silently tracks a
stranger.

With `--crowd`, measured captaincy replaces the engine's modelled estimate,
which feeds straight into the `rank` captaincy objective and the rival field.

Two honest limits. Squads are public **only after a deadline has passed**, so you
always see last gameweek's team, never the one being set now — and before the
first deadline of a season there is nothing to read at all. And a sample is not
a census: every table prints its 95% margin of error, which at 100 managers is
about ±10 points.

There is no feed of what creators *say*, only of what their teams show. This
reads squads, not opinions.

### A second opinion on expected goals

FPL publishes xG, xA and xGC and the engine uses them. A different xG model,
built on different event data, is a useful check on that — so the rate estimator
can blend one in:

```yaml
advanced:
  provider: understat   # fpl (default) · understat · opta
  weight: 0.35
```

**Understat** is free, needs no key, and is a genuinely independent shot-level
model. **Opta** is implemented in full against its documented SDAPI shape, but
it is a Stats Perform product sold under a commercial contract — there is no
free tier and no self-service signup, so it needs `OPTA_OUTLET_KEY` and
`OPTA_TOURNAMENT_CALENDAR` and will say so plainly without them. That client has
never been run against the live service, because doing so requires a licence.

The weight is well below one on purpose. Two xG models disagree because they
define things differently; neither is the truth. The blend also reports how many
players it actually matched across sources, because silent name-matching failure
would be worse than not trying.

### Three routes, not one answer

A single recommendation hides the choice you are actually making. Each week the
engine offers three genuinely different plans and lets you pick:

| Route | Chosen by | For when |
|---|---|---|
| **Recommended** | highest expected points | no strong view either way |
| **Best floor** | highest tenth percentile | protecting a rank |
| **Highest ceiling** | highest ninetieth percentile | chasing one |

Each is planned by the optimiser over the whole horizon and then simulated, so
the comparison includes the transfers each plan would go on to make. A plan that
fills one slot cannot fill another — three near-identical plans are not a choice.

### The dashboard

```bash
fplbrain advise --html briefing.html
```

One self-contained HTML file: your squad on a pitch, the three routes, the
planned path, captaincy, chips, and four analysis panels —

* **Swing** — how far from the template you are, and which players make you different
* **Health** — what the gameweek is likely to bring, and whether anything is misplaced on the bench
* **Robustness** — what breaks the plan: budget, free transfers, autosub cover
* **Points DNA** — where your projected points actually come from, by scoring component

No build step, no server, no external assets. `--plain-html` gives the simpler
document version instead.

### It learns

Three separate jobs, deliberately not conflated:

- **Learning** — a gradient-boosted model trained on stored history, whose most
  important feature is the structural model's own projection, so it learns to
  *correct* rather than to rebuild. Forward-chaining validation by gameweek; a
  random split would let it see the future.
- **Blending** — the weight on the learned model is set by its measured
  out-of-sample error reduction, not by taste, and capped well below 1.
- **Scoring** — `fplbrain backtest` grades projections that were written *before*
  each gameweek was played, reporting error, rank correlation and a calibration
  table ("is a 6-point projection actually worth 6 points?").

There is also a piece of exact learning: **BPS is a deterministic linear function
of match statistics**, so `fplbrain calibrate-bps` regresses observed BPS on
observed stats and recovers the official weights, integer for integer, after a
handful of gameweeks. That removes any dependence on the published BPS table
being correct or current.

---

## Install

```bash
git clone <this repo> && cd assistant-playground
pip install -e .            # add ".[ml]" for LightGBM, ".[dev]" for pytest
```

Python 3.10+. The MILP uses CBC, which ships with PuLP — no separate solver
install. Tested against pandas 3.0, numpy 2.x and scikit-learn 1.9.

### On Windows

Clone somewhere you can write — a shell that opens in `C:\WINDOWS\system32`
cannot create the directory, and every later command then fails for the wrong
reason:

```powershell
cd $HOME
git clone -b <branch> https://github.com/M9chaelangelo/assistant-playground.git
cd assistant-playground
py -m pip install -e ".[ml]"
```

`pip install -e` puts `fplbrain.exe` in a Scripts directory that is usually not
on PATH, so `fplbrain` on its own gives "not recognized". Either add the
directory pip names in its warning, or skip the problem entirely and use the
module form, which is equivalent:

```powershell
py -m fplbrain.cli doctor
py -m fplbrain.cli setup
py -m fplbrain.cli build
```

## Try it with no account and no network

```bash
fplbrain demo --html briefing.html
```

This runs the entire pipeline against a synthetic Premier League with the same
schema as the real API: projections, optimiser, chips, simulation, report.

## Point it at your team

```bash
fplbrain setup     # walks through both credentials and verifies each one
fplbrain doctor    # re-check any time
fplbrain advise --html briefing.html
```

`setup` asks for two things. Both are free, both are optional, and the engine
tells you what it is missing rather than degrading silently.

**An FPL session cookie.** Without one the engine reads your squad only *after*
each deadline, and has to infer your bank and free transfers from your transfer
history. With one it reads your live pre-deadline squad, exact bank, exact free
transfers and true selling prices. It is just your own browser login — no
signup, no cost. FPL puts bot protection in front of its login form, so the
supported path is copying the cookie rather than posting a password; see
[docs/AUTH.md](../docs/AUTH.md).

**An odds API key.** A free key from [the-odds-api.com](https://the-odds-api.com)
allows 500 requests a month; a weekly pull uses about four. Odds are the single
biggest accuracy gain available, because bookmakers price team news and
motivation that no public statistic captures.

Both are stored in `config/secrets.env`, written with `0600` permissions and
git-ignored. `FPL_COOKIE` and `ODDS_API_KEY` in the real environment take
precedence, which is what you want in CI.

> A session cookie is equivalent to your password. Never paste it into a chat,
> an issue, or a screenshot — including to an AI assistant. `setup` reads it
> with hidden input for that reason.

### Setting the entry id

`config/settings.yaml` ships with `entry_id: 3907632`. To change it, either edit
that file, or override without touching the repo:

```bash
export FPL_ENTRY_ID=1234567     # or: fplbrain advise --entry 1234567
```

Your entry id is the number in the URL of your Points page:
`fantasy.premierleague.com/entry/`**`1234567`**`/event/12`. It is public, so it
is safe to commit.

## Commands

| Command | What it gives you |
|---|---|
| `fplbrain setup` | Guided, verified setup of the session cookie and odds key |
| `fplbrain doctor` | Checks every dependency actually works, not just that it is set |
| `fplbrain advise` | The full weekly briefing — transfers, XI, bench order, captain, chips, paths, rank |
| `fplbrain team` | Your squad with per-gameweek projections and injury flags |
| `fplbrain project` | Projections for every player (`--position MID --max-price 8.0`) |
| `fplbrain captain` | Captaincy ranked by expected points, ceiling or rank utility |
| `fplbrain bench` | Starting XI and the optimal bench order, with its autosub value |
| `fplbrain chips` | Every available chip valued in every gameweek of the horizon |
| `fplbrain paths` | Strategies planned and simulated against each other |
| `fplbrain crowd` | Elite ownership, tracked managers, and your differentials |
| `fplbrain build` | A squad from scratch — pre-season, or a wildcard draft |
| `fplbrain pull` | Refresh caches and record results into the local store |
| `fplbrain train` | Train the learned model on stored history |
| `fplbrain backtest` | Grade past projections against what actually happened |
| `fplbrain calibrate-bps` | Recover the true BPS weights from observed data |

Useful flags: `--offline` (cached data only), `--no-paths` / `--no-chips`
(much faster), `--horizon N`, `--max-hits N`, `--json out.json`.

### A weekly routine

```bash
fplbrain pull --history                 # after the gameweek finishes
fplbrain advise --html briefing.html    # before the next deadline
```

Then, once you have a decent stretch of the season stored:

```bash
fplbrain backtest        # is the model actually any good?
fplbrain calibrate-bps   # learn the real BPS weights
fplbrain train           # fit the learned correction layer
```

## Configuration

- `config/rules.yaml` — the rules of the game: scoring, chips, squad
  constraints, BPS weights. A mid-season rule tweak is a config edit, not a code
  change.
- `config/settings.yaml` — you: entry id, horizons, odds provider, risk
  appetite, simulation sizes.

Both can be shadowed by a `*.local.yaml` sibling that is git-ignored, so personal
settings and secrets never end up in a commit.

The setting most worth thinking about is the captaincy objective:

```yaml
risk:
  captain_objective: rank   # "ev", "ceiling" or "rank"
```

`ev` maximises expected points. `ceiling` weights the upper tail. `rank` trades
expected points against how much of the field shares the pick — captaining a
60%-owned player is close to a no-op for rank, while a 5%-owned captain with a
similar projection moves it in both directions.

## Tests

```bash
pytest
```

77 tests, all against the synthetic season, so the suite needs no network and no
credentials. They cover the scoreline maths, the odds devigging, every scoring
rule, the MILP's compliance with squad and transfer rules, the agreement between
the analytic and simulated models, chip availability across the half-season
split, and — importantly — that the training features never peek at the target.

## What it does not do

- **It does not make transfers for you.** Everything is read-only; you enter the
  moves yourself.
- **It does not predict price changes.** Price movements are driven by transfer
  volumes the API only partly exposes.
- **It has no injury feed beyond FPL's own.** The `news` and
  `chance_of_playing` fields are used, but a press conference an hour before the
  deadline is not.
- **It cannot see a manager's intentions.** Rotation is modelled statistically
  from minutes; it does not know that a coach has said someone will be rested.

## Honest limitations

The projections are estimates and carry the uncertainty of the models and the
data behind them. Two are worth naming:

- **Early season.** Before about gameweek 6, per-90 rates are dominated by their
  priors and team ratings lean on FPL's published strength grades. The engine
  says so in the report's provenance section.
- **The BPS prior.** The 2026/27 announcements gave the *changes* to the Bonus
  Points System, not a full published table. `config/rules.yaml` encodes the
  known changes on top of last season's values and marks the rest as estimates —
  which is exactly why `calibrate-bps` exists. Run it once real gameweeks have
  been played.

A good decision can still produce a bad gameweek. The tool is built to get the
decisions right on average, not to be right every week.
