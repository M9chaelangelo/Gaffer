"""Is the plan on disk still the plan? One request, about a second.

    python -m gaffer.verify

Run it before you act. A solve is a photograph of team news at the moment it
ran, and the deadline is hours later; this re-reads the flags for everyone
the recommendation depends on and says whether anything moved.

Exit status is the answer, so it can gate a workflow: 0 the plan stands,
1 somebody got worse and it should be re-solved, 2 it could not be checked.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import fetch, freshness


def load_state(out_dir):
    p = os.path.join(out_dir, "state.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="docs")
    ap.add_argument("--config", default=None,
                    help="read out_dir from this config instead")
    a = ap.parse_args(argv)

    out_dir = a.out_dir
    if a.config:
        import yaml
        with open(a.config) as f:
            out_dir = (yaml.safe_load(f) or {}).get("out_dir", out_dir)

    state = load_state(out_dir)
    if not state:
        print(f"?? no plan at {out_dir}/state.json — nothing to verify")
        return 2
    try:
        boot = fetch.bootstrap()
    except Exception as e:                                  # noqa: BLE001
        print(f"?? could not reach FPL to check: {e.__class__.__name__}: {e}")
        print("   the plan may be fine; this says only that it was not checked")
        return 2

    rep = freshness.report(state, boot)
    for line in freshness.lines(rep):
        print(line)
    if rep["code"] == "stale":
        print("\n   re-solve:  python -m gaffer.main --config config.yaml")
        return 1
    return 0 if rep["code"] in ("fresh", "check") else 2


if __name__ == "__main__":
    sys.exit(main())
