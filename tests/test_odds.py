"""Bookmaker odds: devigging, ingestion, and scoring the model against them."""
from __future__ import annotations

import pytest

from gaffer import odds


# --- devigging ---------------------------------------------------------------

def test_devigged_probabilities_sum_to_one():
    p = odds.devig(2.0, 3.5, 4.0)
    assert p is not None
    assert sum(p) == pytest.approx(1.0)
    assert p[0] > p[1] > p[2]           # shortest price is likeliest


def test_the_margin_is_reported_and_is_what_the_book_actually_took():
    # 1/2.0 + 1/3.5 + 1/4.0 = 1.0357...
    assert odds.overround(2.0, 3.5, 4.0) == pytest.approx(0.03571, abs=1e-4)


def test_a_book_with_no_margin_is_a_parse_error_not_a_bargain():
    """Three prices summing to under 1.0 implied is arbitrage, which does
    not exist in a published closing line. It means the columns were read
    wrong, and a wrong column must not become a goal rate."""
    assert odds.devig(5.0, 5.0, 5.0) is None       # implied 0.6, impossible
    assert odds.devig(1.01, 1.01, 1.01) is None    # implied 2.97, nonsense


def test_a_missing_or_unparseable_price_returns_none_rather_than_a_nan():
    assert odds.devig(None, 3.5, 4.0) is None
    assert odds.devig("", 3.5, 4.0) is None
    assert odds.devig(0, 3.5, 4.0) is None
    assert odds.devig("evens", 3.5, 4.0) is None
    assert odds.overround(None, 3.5, 4.0) is None


# --- the historical file -----------------------------------------------------

CSV = (
    "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,AvgH,AvgD,AvgA,B365H,B365D,B365A\n"
    "E0,11/10/2026,16:30,Liverpool,Man City,0,3,A,2.70,3.60,2.45,2.75,3.50,2.50\n"
    "E0,12/10/2026,20:00,Tottenham,Man United,2,1,H,2.10,3.40,3.50,2.05,3.45,3.60\n"
    "E0,12/10/2026,20:00,Leeds,Everton,,,,2.10,3.40,3.50,2.05,3.45,3.60\n"
    "E0,13/10/2026,20:00,Hull,Brentford,1,1,D,,,,,,\n"
)


def test_the_file_parses_into_matches_with_probabilities_and_a_score():
    rows = odds.history(text=CSV)
    assert len(rows) == 2, "a match with no score or no price is not a row"
    r = rows[0]
    assert r["home"] == "Liverpool" and r["away"] == "Man City"
    assert (r["home_goals"], r["away_goals"]) == (0, 3)
    assert sum((r["p_home"], r["p_draw"], r["p_away"])) == pytest.approx(1.0)
    assert r["book"] == "Avg"


def test_club_names_are_translated_to_the_ones_the_rest_of_the_engine_uses():
    rows = odds.history(text=CSV)
    assert rows[1]["home"] == "Spurs"        # football-data writes Tottenham
    assert rows[1]["away"] == "Man Utd"      # and Man United


def test_the_consensus_column_is_preferred_over_a_single_book():
    rows = odds.history(text=CSV)
    assert rows[0]["book"] == "Avg"
    # and B365 is the fallback when the average is absent
    only_b365 = CSV.replace("2.70,3.60,2.45", ",,")
    rows = odds.history(text=only_b365)
    assert rows[0]["book"] == "B365"


# --- the forward feed --------------------------------------------------------

def test_no_key_means_no_odds_and_a_reason_not_an_exception(monkeypatch):
    """A solve without odds must fall back to the model, not die."""
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    rows, note = odds.forward("soccer_epl")
    assert rows == [] and "ODDS_API_KEY" in note


def test_a_fixture_is_priced_from_the_mean_across_books():
    game = {
        "home_team": "Arsenal", "away_team": "Chelsea",
        "commence_time": "2026-10-25T14:00:00Z",
        "bookmakers": [
            {"markets": [{"key": "h2h", "outcomes": [
                {"name": "Arsenal", "price": 2.0},
                {"name": "Chelsea", "price": 4.0},
                {"name": "Draw", "price": 3.4}]}]},
            {"markets": [{"key": "h2h", "outcomes": [
                {"name": "Arsenal", "price": 2.2},
                {"name": "Chelsea", "price": 3.8},
                {"name": "Draw", "price": 3.6}]}]},
        ],
    }
    assert odds._consensus(game) == pytest.approx((2.1, 3.5, 3.9))


def test_a_fixture_missing_a_side_is_skipped_rather_than_half_priced():
    game = {"home_team": "Arsenal", "away_team": "Chelsea",
            "bookmakers": [{"markets": [{"key": "h2h", "outcomes": [
                {"name": "Arsenal", "price": 2.0},
                {"name": "Draw", "price": 3.4}]}]}]}
    assert odds._consensus(game) is None


# --- scoring -----------------------------------------------------------------

def test_a_model_that_is_the_market_scores_like_the_market():
    rows = odds.history(text=CSV)
    # Goal rates chosen to roughly reproduce each book line; the point is
    # that both columns are computed over the same matches, so a tie is a
    # tie and the comparison means something.
    got = odds.score_against_market(rows, lambda h, a: (1.4, 1.3))
    assert got["n"] == 2
    assert set(got["brier"]) == {"model", "market"}
    assert set(got["log_loss"]) == {"model", "market"}
    assert got["verdict"] in ("the model beats the market",
                              "the market beats the model")


def test_a_model_with_no_view_on_a_fixture_is_excluded_from_both_columns():
    rows = odds.history(text=CSV)
    got = odds.score_against_market(
        rows, lambda h, a: (1.4, 1.3) if h == "Liverpool" else None)
    assert got["n"] == 1, "scoring only what we predicted keeps it honest"


def test_no_overlap_says_so_instead_of_dividing_by_zero():
    rows = odds.history(text=CSV)
    got = odds.score_against_market(rows, lambda h, a: None)
    assert got["n"] == 0 and "note" in got


def test_being_right_scores_better_than_being_wrong():
    """Scored on one match, so the result is not washed out by symmetry.

    The first version of this test used both fixtures, which went opposite
    ways — Liverpool lost at home, Spurs won at home — so a home-heavy and
    an away-heavy model scored *identically*. That was the scorer being
    right and the test being wrong.
    """
    rows = [r for r in odds.history(text=CSV) if r["home"] == "Liverpool"]
    assert rows[0]["home_goals"] < rows[0]["away_goals"]      # 0-3, away win

    right = odds.score_against_market(rows, lambda h, a: (0.6, 2.6))
    wrong = odds.score_against_market(rows, lambda h, a: (2.6, 0.6))
    assert right["brier"]["model"] < wrong["brier"]["model"]
    assert right["log_loss"]["model"] < wrong["log_loss"]["model"]


def test_the_market_column_is_the_same_whatever_our_model_says():
    """Otherwise the comparison is against a moving target."""
    rows = odds.history(text=CSV)
    a = odds.score_against_market(rows, lambda h, a: (1.4, 1.3))
    b = odds.score_against_market(rows, lambda h, a: (2.9, 0.4))
    assert a["brier"]["market"] == b["brier"]["market"]
    assert a["log_loss"]["market"] == b["log_loss"]["market"]
