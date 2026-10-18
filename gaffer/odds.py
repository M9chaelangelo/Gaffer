"""Bookmaker odds: the market's own forecast, and how ours compares to it.

The market is the best public forecast of a football match that exists. It
is not perfect, but it is the thing a model has to beat before anyone
should believe it, and until now this engine has had no way to find out.

Two sources, for two different jobs, and they are not interchangeable:

  - **football-data.co.uk** publishes *closing* 1X2 odds for matches that
    have already been played, as free CSV files. Historical, so it cannot
    price next Saturday — but it is exactly what is needed to score our
    goal rates against the market's, which is the question "is the model
    any good" in its measurable form.

  - **the-odds-api.com** publishes *forward* 1X2 for fixtures not yet
    played, for both the Premier League and the Champions League, under a
    key. That is the one that can feed a live solve.

Deliberately not used: scraping a bookmaker's own site. Their terms forbid
it, the HTML changes without notice, and an odds feed that silently breaks
is worse than one that is absent.

Devigging uses the multiplicative (proportional) method: divide each
implied probability by the overround. It is the standard choice and the
one the UCL side already assumes.
"""
from __future__ import annotations

import csv
import io
import os
import urllib.request

FOOTBALL_DATA = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
ODDS_API = "https://api.the-odds-api.com/v4/sports/{sport}/odds"

# football-data writes club names its own way. Only the ones that differ
# from the FPL short name need a line here.
ALIASES = {
    "Man City": "Man City", "Man United": "Man Utd", "Nott'm Forest": "Nott'm Forest",
    "Newcastle": "Newcastle", "Tottenham": "Spurs", "Wolves": "Wolves",
    "Sheffield United": "Sheffield Utd", "West Ham": "West Ham",
    "Leeds": "Leeds", "Hull": "Hull City", "Coventry": "Coventry",
}

# Preference order for the 1X2 columns. Avg* is the market consensus and
# the right default; B365 is the fallback when the average is missing.
COLUMN_SETS = (("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A"),
               ("MaxH", "MaxD", "MaxA"))


def _get(url, timeout=45):
    # Honest headers. UEFA's edge blocks a request that claims to be Chrome
    # and is not; there is no reason to find out whether this one does too.
    req = urllib.request.Request(url, headers={"Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def devig(home, draw, away):
    """Decimal odds to probabilities that sum to one.

    Returns None when a price is missing or nonsensical rather than
    propagating a NaN into a goal rate that then looks like a forecast.
    """
    try:
        inv = [1.0 / float(home), 1.0 / float(draw), 1.0 / float(away)]
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    total = sum(inv)
    if not (1.0 < total < 1.5):     # a book with no margin, or a parse error
        return None
    return tuple(x / total for x in inv)


def overround(home, draw, away):
    """The bookmaker's margin, as a fraction above 1. None if unparseable."""
    try:
        return (1.0 / float(home) + 1.0 / float(draw) + 1.0 / float(away)) - 1.0
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _pick_columns(row):
    for cols in COLUMN_SETS:
        if all(row.get(c) for c in cols):
            return cols
    return None


def history(season="2627", div="E0", text=None):
    """Closing 1X2 and the actual score, one row per played match.

    `season` is football-data's own form: 2627 is 2026/27. Pass `text` to
    parse a file already fetched.
    """
    raw = text if text is not None else _get(
        FOOTBALL_DATA.format(season=season, div=div)).decode("utf-8-sig",
                                                             "replace")
    out = []
    for row in csv.DictReader(io.StringIO(raw)):
        if not row.get("HomeTeam"):
            continue
        cols = _pick_columns(row)
        if not cols:
            continue
        p = devig(*(row[c] for c in cols))
        if p is None:
            continue
        try:
            hg, ag = int(row["FTHG"]), int(row["FTAG"])
        except (TypeError, ValueError, KeyError):
            continue
        out.append({
            "date": row.get("Date", ""),
            "home": ALIASES.get(row["HomeTeam"], row["HomeTeam"]),
            "away": ALIASES.get(row["AwayTeam"], row["AwayTeam"]),
            "home_goals": hg, "away_goals": ag,
            "p_home": p[0], "p_draw": p[1], "p_away": p[2],
            "book": cols[0].rstrip("H"),
            "overround": overround(*(row[c] for c in cols)),
        })
    return out


def forward(sport="soccer_epl", api_key=None, regions="uk", timeout=45):
    """Devigged 1X2 for fixtures not yet played.

    Needs a key from the-odds-api.com. It is read from the environment, not
    from a file in the repository and never from a chat message. Returns
    `[]` with a reason rather than raising, so a solve without odds falls
    back to the model instead of dying.
    """
    key = api_key or os.environ.get("ODDS_API_KEY")
    if not key:
        return [], "no ODDS_API_KEY in the environment"
    url = (f"{ODDS_API.format(sport=sport)}?regions={regions}"
           f"&markets=h2h&oddsFormat=decimal&apiKey={key}")
    try:
        import json
        games = json.loads(_get(url, timeout=timeout))
    except Exception as e:                       # noqa: BLE001 — reported
        return [], f"{e.__class__.__name__}: {e}"
    if not isinstance(games, list):
        return [], f"unexpected response: {str(games)[:120]}"

    out = []
    for g in games:
        prices = _consensus(g)
        if not prices:
            continue
        p = devig(*prices)
        if p is None:
            continue
        out.append({
            "home": g.get("home_team", ""), "away": g.get("away_team", ""),
            "kickoff": g.get("commence_time", ""),
            "p_home": p[0], "p_draw": p[1], "p_away": p[2],
            "books": len(g.get("bookmakers") or []),
            "overround": overround(*prices),
        })
    return out, f"{len(out)} fixtures priced"


def _consensus(game):
    """Mean decimal price across books, home/draw/away.

    One book is one opinion; the average of several is the market. A game
    quoted by a single book still returns, but `books` says so.
    """
    home, draw, away = [], [], []
    hteam, ateam = game.get("home_team"), game.get("away_team")
    for b in (game.get("bookmakers") or []):
        for m in (b.get("markets") or []):
            if m.get("key") != "h2h":
                continue
            for o in (m.get("outcomes") or []):
                name, price = o.get("name"), o.get("price")
                if not price:
                    continue
                if name == hteam:
                    home.append(price)
                elif name == ateam:
                    away.append(price)
                elif str(name).lower() == "draw":
                    draw.append(price)
    if not (home and draw and away):
        return None
    return (sum(home) / len(home), sum(draw) / len(draw),
            sum(away) / len(away))


def score_against_market(rows, predict):
    """How our goal rates compare with the market's, on matches now played.

    `predict(home, away)` returns our (lambda_home, lambda_away), or None
    where we have no view. Reports Brier score and log loss for both, over
    the same matches, because a model that is only scored against itself
    can never be shown to be wrong.
    """
    import math

    from scipy.stats import poisson

    def outcome_probs(lh, la, maxg=9):
        h = poisson.pmf(range(maxg + 1), lh)
        a = poisson.pmf(range(maxg + 1), la)
        ph = pd_ = pa = 0.0
        for i in range(maxg + 1):
            for j in range(maxg + 1):
                q = h[i] * a[j]
                if i > j:
                    ph += q
                elif i == j:
                    pd_ += q
                else:
                    pa += q
        t = ph + pd_ + pa
        return (ph / t, pd_ / t, pa / t)

    n = 0
    brier = {"model": 0.0, "market": 0.0}
    logloss = {"model": 0.0, "market": 0.0}
    for r in rows:
        got = predict(r["home"], r["away"])
        if not got:
            continue
        ours = outcome_probs(*got)
        theirs = (r["p_home"], r["p_draw"], r["p_away"])
        actual = (1.0 if r["home_goals"] > r["away_goals"] else 0.0,
                  1.0 if r["home_goals"] == r["away_goals"] else 0.0,
                  1.0 if r["home_goals"] < r["away_goals"] else 0.0)
        n += 1
        for name, p in (("model", ours), ("market", theirs)):
            brier[name] += sum((pi - ai) ** 2 for pi, ai in zip(p, actual))
            hit = max(1e-12, sum(pi for pi, ai in zip(p, actual) if ai))
            logloss[name] -= math.log(hit)
    if not n:
        return {"n": 0, "note": "no overlap between our fixtures and the file"}
    return {
        "n": n,
        "brier": {k: round(v / n, 4) for k, v in brier.items()},
        "log_loss": {k: round(v / n, 4) for k, v in logloss.items()},
        "verdict": ("the model beats the market"
                    if brier["model"] < brier["market"] else
                    "the market beats the model"),
    }


# --- joining a price list to a fixture list ----------------------------------
# The two feeds spell clubs differently and neither is wrong: UEFA writes
# "Man City", "Bayern München", "S. Bratislava"; the odds API writes
# "Manchester City", "Bayern Munich", "ŠK Slovan Bratislava". Matching has to
# survive that without ever pairing the wrong two teams, because a fixture
# joined to the wrong price is worse than a fixture with no price at all.

# Noise words only. An earlier version also dropped "united", "borussia",
# "real" and "city" as though they carried no information, which left
# "Man Utd" as "man" and "Borussia Dortmund" as "dortmund" and silently
# lost five of eighteen fixtures to the model. The distinctive word is
# usually the one a stripper is tempted by.
_NOISE = {"fc", "afc", "sk", "ss", "as", "sc", "cf", "ac", "bk", "kv", "vfb",
          "cp", "bv", "rb", "ssc", "if", "il", "fk", "ak", "b", "the", "and",
          "hove", "albion", "hotspur", "eindhoven", "athens", "praha",
          "prague"}

# Words shared by clubs that are not the same club. "Sporting CP" and
# "Sporting Gijon" are both Sporting and both long enough to pass a naive
# length test, which would have priced a Portuguese fixture off a Spanish
# one. A generic word never identifies a club on its own.
_GENERIC = {"sporting", "real", "atletico", "athletic", "dynamo", "dinamo",
            "olympique", "racing", "union", "standard", "young", "boys",
            "red", "star", "sport", "club", "city", "united", "rovers",
            "wanderers", "county", "town", "borussia", "bayer", "bayern"}

# Nordic and German letters that NFKD leaves alone, because they are
# letters in their own right rather than accented vowels. The odds feed
# writes "Bodo Glimt" where UEFA writes "Bodø/Glimt".
_TRANSLIT = str.maketrans({"ø": "o", "Ø": "o", "å": "a", "Å": "a",
                           "æ": "ae", "Æ": "ae", "ß": "ss", "đ": "d",
                           "ı": "i", "ł": "l", "ð": "d", "þ": "th"})

# The pairs no amount of folding joins, because the two feeds share no word
# longer than three letters. Explicit beats clever: a wrong join prices a
# fixture with another match's odds, which is worse than no price at all.
_ALIASES = {
    "manutd": "manchesterunited", "manunited": "manchesterunited",
    "mancity": "manchestercity",
    "spurs": "tottenham", "wolves": "wolverhampton",
    "inter": "internazionale", "psg": "parissaintgermain",
    "paris": "parissaintgermain", "atleti": "atleticomadrid",
    "bayern": "bayernmunich", "bayernmunchen": "bayernmunich",
    "psv": "psveindhoven", "bodo": "bodoglimt",
    # UEFA writes "Sporting CP", the book writes "Sporting Lisbon", and
    # "sporting" alone is deliberately refused — Sporting Gijon exists.
    "sporting": "sportinglisbon",
}

MIN_TOKEN = 5          # the shortest word trusted to identify a club alone


def _tokens(name):
    import unicodedata
    s = str(name).lower().translate(_TRANSLIT)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c if (c.isalnum() or c.isspace()) else " " for c in s)
    return [w for w in s.split() if w and w not in _NOISE]


def fold(name):
    """A club name reduced to what both feeds agree on."""
    t = _tokens(name)
    joined = "".join(t)
    return _ALIASES.get(joined, joined)


def _same_club(a, b):
    """Same club, or not. Never "probably"."""
    fa, fb = fold(a), fold(b)
    if not fa or not fb:
        return False
    if fa == fb:
        return True
    # A prefix match is only safe when the shorter name says something.
    # "Sporting" is a prefix of "Sporting Gijon" and of Sporting CP, and
    # they are different clubs on different sides of Iberia.
    shorter = fa if len(fa) <= len(fb) else fb
    if shorter not in _GENERIC and (fa.startswith(fb) or fb.startswith(fa)):
        return True
    ta, tb = set(_tokens(a)), set(_tokens(b))
    # One name is the other plus a league prefix: "Lens" and "RC Lens",
    # "Leipzig" and "RB Leipzig". Enumerating every country's prefixes is a
    # losing game; a subset is the general form of it. The smaller side must
    # still carry a real word, so a stray initial cannot match everything.
    small, large = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if (small and small <= large
            and any(len(w) >= 4 and w not in _GENERIC for w in small)):
        return True
    # Otherwise a long, *distinctive* word carried by both identifies the
    # club: Dortmund, Bratislava, Feyenoord. Short ones do not — "city" is
    # four clubs — and neither do generic ones, or Sporting CP would be
    # priced off Sporting Gijon.
    def distinctive(ts):
        return {w for w in ts if len(w) >= MIN_TOKEN and w not in _GENERIC}
    return bool(distinctive(ta) & distinctive(tb))

def match_fixtures(fixtures, priced):
    """{(home, away): devigged probabilities} for the fixtures we can join.

    Returns the map and the fixtures left unpriced, so a run says which
    lines are market and which are model instead of quietly mixing them.
    Both sides of a fixture must match, and a price already used is not
    reused — two fixtures claiming the same line means the fold collided
    and the right answer is to price neither.
    """
    out, used, missed = {}, set(), []
    for f in fixtures:
        hit = None
        for i, r in enumerate(priced):
            if i in used:
                continue
            if _same_club(f["home"], r["home"]) and _same_club(f["away"], r["away"]):
                hit = (i, r)
                break
        if hit is None:
            missed.append(f"{f['home']} v {f['away']}")
            continue
        used.add(hit[0])
        r = hit[1]
        out[(f["home"], f["away"])] = (r["p_home"], r["p_draw"], r["p_away"])
    return out, missed
