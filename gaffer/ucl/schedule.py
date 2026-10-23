"""The whole league phase, not just the matchday in front of you.

UEFA's fantasy feed carries the current matchday's fixtures and nothing else,
so every decision this engine made was a one-matchday decision. That is fine
for a captain and wrong for everything else: a Wildcard buys a squad for the
rest of the phase, and a club with a kind matchday four and a brutal five
through eight is a club to hop on and hop off, not to build around.

The schedule is therefore committed, in `schedule.json`, rather than fetched.
The runner cannot reach gaming.uefa.com, the draw happened in August and the
fixtures do not move; a file read beats a request that fails. Its provenance
is in the file, and `validate()` checks it against the format's own rules —
every club eight distinct opponents, four home and four away, once per
matchday — so a bad paste is a failing test rather than a quiet wrong plan.

Club names are stored as the source writes them. Joining them to the feed is
an explicit table, not a fuzzy match: "Inter Milan" and "Inter" are the same
club and "Slavia Prague" and "Slavia Praha" are the same club, but a matcher
loose enough to see that is loose enough to confuse Manchester's two.
"""
from __future__ import annotations

import json
import os
import unicodedata
from collections import Counter

from . import strength

PATH = os.path.join(os.path.dirname(__file__), "schedule.json")

MATCHDAYS = tuple(range(1, 9))

# Written out in full rather than derived. A club that stops matching is then
# a club this table does not mention, which is a one-line fix; a clever
# matcher that stops matching is a morning.
ALIASES = {
    "Atlético Madrid": "Atleti",
    "Bayern Munich": "Bayern München",
    "Borussia Dortmund": "B. Dortmund",
    "Inter Milan": "Inter",
    "Manchester City": "Man City",
    "Manchester United": "Man Utd",
    "PSV Eindhoven": "PSV",
    "Paris Saint-Germain": "Paris",
    "RB Leipzig": "Leipzig",
    "Shakhtar Donetsk": "Shakhtar",
    "Slavia Prague": "Slavia Praha",
    "Slovan Bratislava": "S. Bratislava",
    "VfB Stuttgart": "Stuttgart",
}


def _fold(s):
    s = unicodedata.normalize("NFKD", str(s or ""))
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()


def load(path=PATH):
    """{matchday: [{home, away, date}]}, names as the source writes them."""
    with open(path) as f:
        raw = json.load(f)
    return {int(md): list(v) for md, v in raw["matchdays"].items()}


def provenance(path=PATH):
    with open(path) as f:
        raw = json.load(f)
    return {k.lstrip("_"): v for k, v in raw.items() if k.startswith("_")}


def validate(sched):
    """Everything the competition's own format guarantees. Returns problems.

    This is not a smoke test. The schedule was typed from a page, and the
    failure it guards against is a plausible one: a fixture read off the
    wrong row gives one club nine games and another seven, which no plan
    would ever reveal — it would just quietly misprice two clubs for the
    rest of the phase.
    """
    bad = []
    if sorted(sched) != list(MATCHDAYS):
        bad.append(f"matchdays are {sorted(sched)}, expected {list(MATCHDAYS)}")
    for md, fx in sorted(sched.items()):
        if len(fx) != 18:
            bad.append(f"matchday {md} has {len(fx)} fixtures, expected 18")
        seen = Counter()
        for f in fx:
            seen[f["home"]] += 1
            seen[f["away"]] += 1
        twice = sorted(c for c, k in seen.items() if k > 1)
        if twice:
            bad.append(f"matchday {md}: {', '.join(twice)} play twice")

    clubs = sorted({f[end] for fx in sched.values() for f in fx
                    for end in ("home", "away")})
    if len(clubs) != 36:
        bad.append(f"{len(clubs)} clubs, expected 36")
    for club in clubs:
        home = sum(1 for fx in sched.values() for f in fx if f["home"] == club)
        away = sum(1 for fx in sched.values() for f in fx if f["away"] == club)
        opps = [f["away"] if f["home"] == club else f["home"]
                for fx in sched.values() for f in fx
                if club in (f["home"], f["away"])]
        if home != 4 or away != 4:
            bad.append(f"{club}: {home} home and {away} away, expected 4 and 4")
        if len(set(opps)) != len(opps):
            dupes = sorted(c for c, k in Counter(opps).items() if k > 1)
            bad.append(f"{club} plays {', '.join(dupes)} twice")
    return bad


def resolve(sched, feed_teams):
    """Rewrite club names to the feed's spellings. Returns (schedule, missing).

    `missing` is every club the table could not place. It is returned rather
    than raised because a plan for one matchday does not need the other
    seven, and losing the whole horizon over one renamed club would be worse
    than losing that club.
    """
    by_fold = {_fold(t): t for t in feed_teams}
    name = {}
    for club in {f[end] for fx in sched.values() for f in fx
                 for end in ("home", "away")}:
        hit = by_fold.get(_fold(ALIASES.get(club, club)))
        if hit is None:
            hit = by_fold.get(_fold(club))
        if hit is not None:
            name[club] = hit
    out = {md: [{**f, "home": name.get(f["home"], f["home"]),
                 "away": name.get(f["away"], f["away"])} for f in fx]
           for md, fx in sched.items()}
    missing = sorted({c for fx in sched.values() for f in fx
                      for c in (f["home"], f["away"])} - set(name))
    return out, missing


def runs(sched, matchdays=None):
    """{club: {matchday: (opponent, at_home)}} over the matchdays given."""
    want = set(matchdays or sched)
    out = {}
    for md, fx in sched.items():
        if md not in want:
            continue
        for f in fx:
            out.setdefault(f["home"], {})[md] = (f["away"], True)
            out.setdefault(f["away"], {})[md] = (f["home"], False)
    return out


def difficulty(sched, players, matchdays=None):
    """What each club's run is worth, matchday by matchday.

    `scored` is the club's own expected goals in that fixture and `conceded`
    the opponent's, both from `strength`, so the two sides of a player's
    score — his attacking returns and his clean sheet — are kept apart. A
    single "difficulty" number adds a striker's problem to a defender's and
    is useful to neither.
    """
    rate, mu = strength.ratings(players)
    out = {}
    for club, fixtures in runs(sched, matchdays).items():
        for md, (opp, at_home) in fixtures.items():
            home, away = (club, opp) if at_home else (opp, club)
            lh, la = strength.lambdas(rate, home, away, mu)
            mine, theirs = (lh, la) if at_home else (la, lh)
            out.setdefault(club, {})[md] = {
                "opponent": opp, "home": at_home,
                "scored": mine, "conceded": theirs,
                # P(clean sheet) under the Poisson the simulator draws from.
                "cs": float(pow(2.718281828459045, -theirs)),
            }
    return out


def summarise(diff, matchdays=None):
    """One row per club: the run totalled, and where in it the points are.

    `front` and `back` split the window in half so that two clubs with the
    same total can be told apart. A club whose good fixtures are all at the
    front is a club to own now and sell; one whose are at the back is a club
    to buy later, and the total alone says they are the same club.

    The halves are averages, not sums. An odd window gives them different
    lengths — five matchdays split three and two — and the first version
    subtracted a two-matchday sum from a three-matchday one, which made
    thirty-four of thirty-six clubs look back-loaded.
    """
    rows = []
    for club, by_md in diff.items():
        mds = sorted(md for md in by_md if matchdays is None or md in matchdays)
        if not mds:
            continue
        half = (len(mds) + 1) // 2
        total = lambda ks, key: sum(by_md[k][key] for k in ks)
        mean = lambda ks, key: total(ks, key) / len(ks) if ks else 0.0
        rows.append({
            "club": club, "matchdays": mds,
            "scored": total(mds, "scored"), "conceded": total(mds, "conceded"),
            "cs": total(mds, "cs"),
            "front_scored": mean(mds[:half], "scored"),
            "back_scored": mean(mds[half:], "scored"),
            "front_cs": mean(mds[:half], "cs"),
            "back_cs": mean(mds[half:], "cs"),
            "opponents": [by_md[k]["opponent"] for k in mds],
            "home": [by_md[k]["home"] for k in mds],
        })
    return sorted(rows, key=lambda r: -r["scored"])


def table(diff, matchdays=None, write=None):
    """The hop-on, hop-off matrix: every club's run, and when it turns."""
    rows = summarise(diff, matchdays)
    if not rows:
        return ""
    mds = rows[0]["matchdays"]
    lines = ["", "league-phase run — expected goals for a club's own attack, "
                 "and its clean-sheet odds",
             "(upper case is home; `turn` is the back half of the window less "
             "the front, per matchday,", " so a positive number is a club to "
             "buy later and a negative one a club to sell)"]
    head = "  " + f"{'club':<18}" + "".join(f"  {'MD' + str(md):<14}"
                                            for md in mds)
    lines.append(head + f"  {'goals':>6} {'turn':>6} {'cs':>6}")
    for r in rows:
        cells = ""
        for md in mds:
            d = diff[r["club"]][md]
            opp = d["opponent"][:9]
            cells += (f"  {opp.upper() if d['home'] else opp:<9}"
                      f"{d['scored']:>4.1f} ")
        turn = r["back_scored"] - r["front_scored"]
        lines.append(f"  {r['club']:<18}{cells} {r['scored']:>6.2f} "
                     f"{turn:>+6.2f} {r['cs']:>6.2f}")
    text = "\n".join(lines)
    if write:
        os.makedirs(os.path.dirname(write) or ".", exist_ok=True)
        with open(write, "w") as f:
            f.write(text + "\n")
    return text
