"""Fetch the UCL feed and commit a snapshot, so development needs no network.

The assistant's container is denied gaming.uefa.com by the environment's
network policy. Rather than build against guesses, this runs where there is
network and leaves a trimmed, committed copy behind.
"""
from __future__ import annotations

import json
import sys

from . import fetch


def main(matchday=2):
    rows = fetch.players(matchday)
    problems = fetch.verify(fetch.raw_players(matchday))
    fx = fetch.fixtures(rows, matchday)

    if problems:
        print("!! the feed no longer matches what this was written against:")
        for p in problems:
            print("   -", p)

    print(f"{len(rows)} players, {len(fx)} fixtures for matchday {matchday}")
    print("\nfixtures (kick-off order — the armband can move between days):")
    for f in fx:
        print(f"  {f['kickoff'][:16]}  {f['home']:22} v {f['away']}")

    by_pos = {}
    for p in rows:
        by_pos.setdefault(p["pos"], []).append(p)
    print("\nmost expensive by position:")
    for pos in ("GK", "DEF", "MID", "FWD"):
        top = sorted(by_pos.get(pos, []), key=lambda x: -x["price"])[:4]
        print(f"  {pos}: " + ", ".join(
            f"{p['name']} {p['price']:.1f} ({p['team']}, {p['own']:.0f}%)" for p in top))

    print("\navailability wording actually used by UEFA:")
    kinds = {}
    for p in rows:
        kinds[p["trained"]] = kinds.get(p["trained"], 0) + 1
    for k, n in sorted(kinds.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {n:5}  {k!r}")

    # a trimmed snapshot: everyone who has played, plus every named squad member
    keep = [p for p in rows if p["minutes"] > 0 or p["price"] >= 6.0]
    out = {"matchday": matchday, "fixtures": fx, "players": keep,
           "problems": problems, "total_players": len(rows)}
    with open("gaffer/ucl/snapshot.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"\nwrote gaffer/ucl/snapshot.json with {len(keep)} players")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 2))
