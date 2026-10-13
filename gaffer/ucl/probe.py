"""Find UEFA's UCL Fantasy endpoints by trying them from a machine with network.

UEFA publishes no developer API: no portal, no keys, no documented paths. The
game's own site is served by JSON feeds, but their shapes move between seasons
and nobody outside UEFA promises they will not. So rather than hard-code a
guess, this probes a list of candidates and reports what actually answers.

Run it where there is network — the gaffer workflow on GitHub Actions — and
read the summary it writes to probe.json.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (fpl-gaffer ucl probe)",
      "Accept": "application/json, text/plain, */*"}

SEASON = 2027      # UEFA labels 2026/27 as season 2027
COMP = 1           # Champions League

CANDIDATES = [
    # the classic UCL Fantasy feeds
    "https://gaming.uefa.com/en/uclfantasy/services/api/Feed/players?gamedayId=1&language=en",
    "https://gaming.uefa.com/en/uclfantasy/services/api/Feed/matches?gamedayId=1&language=en",
    "https://gaming.uefa.com/en/uclfantasy/services/api/feed/players?optType=2&gamedayId=1",
    "https://gaming.uefa.com/en/uclfantasy/services/api/Feed/gamedays?language=en",
    "https://gaming.uefa.com/en/uclfantasy/services/api/feed/competition?language=en",
    # the newer naming the site has used
    "https://gaming.uefa.com/en/uclfantasyfootball/services/api/Feed/players?gamedayId=1&language=en",
    "https://gamingapi.uefa.com/v2/games/1/players?gamedayId=1",
    "https://gamingapi.uefa.com/v2/games/1/gamedays",
    # the public-ish stats and match services the site itself calls
    f"https://match.uefa.com/v5/matches?competitionId={COMP}&seasonYear={SEASON}&limit=5",
    f"https://matchstats.uefa.com/v1/team-statistics?competitionId={COMP}&seasonYear={SEASON}",
    f"https://comp-stats.uefa.com/v1/player-ranking?competitionId={COMP}&seasonYear={SEASON}&limit=5",
    f"https://comp-stats.uefa.com/v1/players?competitionId={COMP}&seasonYear={SEASON}&limit=5",
    f"https://compstats.uefa.com/v1/player-ranking?competitionId={COMP}&seasonYear={SEASON}&limit=5",
    "https://api.uefa.com/v1/competitions",
]


def probe(url, timeout=12):
    req = urllib.request.Request(url, headers=UA)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read(200_000)
            ms = int((time.time() - t0) * 1000)
            ctype = r.headers.get("Content-Type", "")
            out = {"url": url, "status": r.status, "ms": ms, "content_type": ctype,
                   "bytes": len(raw)}
            try:
                data = json.loads(raw)
            except Exception:
                out["parsed"] = False
                out["head"] = raw[:300].decode("utf-8", "replace")
                return out
            out["parsed"] = True
            # describe the shape without dumping thousands of players
            if isinstance(data, dict):
                out["keys"] = list(data)[:15]
                for k, v in data.items():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        out["list_key"] = k
                        out["list_len"] = len(v)
                        out["item_keys"] = list(v[0])[:30]
                        out["sample"] = {kk: v[0][kk] for kk in list(v[0])[:12]}
                        break
            elif isinstance(data, list) and data:
                out["list_len"] = len(data)
                if isinstance(data[0], dict):
                    out["item_keys"] = list(data[0])[:30]
                    out["sample"] = {k: data[0][k] for k in list(data[0])[:12]}
            return out
    except Exception as e:
        return {"url": url, "status": None, "error": f"{type(e).__name__}: {e}"[:200]}


def main():
    results = [probe(u) for u in CANDIDATES]
    ok = [r for r in results if r.get("status") == 200 and r.get("parsed")]
    with open("probe.json", "w") as f:
        json.dump({"ok": ok, "all": results}, f, indent=1)

    print(f"\n{len(ok)} of {len(results)} candidates returned parseable JSON\n")
    for r in results:
        mark = "OK " if r in ok else "   "
        if r.get("status") == 200:
            print(f"{mark}{r['status']} {r['url']}")
            if r.get("item_keys"):
                print(f"     list '{r.get('list_key','(root)')}' len={r.get('list_len')}")
                print(f"     item keys: {r['item_keys']}")
                print(f"     sample:    {json.dumps(r.get('sample', {}), ensure_ascii=False)[:400]}")
            elif r.get("keys"):
                print(f"     keys: {r['keys']}")
        else:
            print(f"{mark}{r.get('status') or 'ERR'} {r['url']}  {r.get('error','')}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
