"""Who is about to rise or fall.

FPL never published its price algorithm. This is a net-transfer heuristic
calibrated against players who already moved this gameweek, so it
recalibrates itself every week instead of trusting a hardcoded threshold.

## Flow is relative to the people already holding him

The first version compared raw net transfers against one global threshold,
which is ownership-blind, and a third-party price site disagreed with it on
exactly the players that error predicts. On 16 Oct 2026:

    player        net      owned   net/owner   ours    theirs
    Gross      +665,126    42.2%     0.142     rise    very likely rise
    Joao Pedro +557,121    71.4%     0.070     rise    UNLIKELY to change

Half a million transfers into a player owned by seven million people is a
smaller event than four hundred thousand into one owned by four million,
and a single threshold cannot tell them apart. The flow is now measured as
a share of current owners, which is the quantity that actually moves a
price, and the calibration is done on that same scale.

## What this still cannot see

FPL's counter is not `transfers_in_event`. That field resets at the
gameweek boundary while prices move nightly on a rolling window, and a
player who changed price yesterday has effectively been reset. That is why
the same site had Davis "likely to rise" while this sees a small negative
flow for the gameweek. The rolling counter is not published anywhere, so
the gap is named rather than modelled — and the price ledger now scores
every forecast, so the size of it will be a number soon rather than a
guess.
"""

# Below this many owners, a share is noise: a player owned by two hundred
# people moves to 100% of owners on a quiet afternoon.
MIN_OWNERS = 20_000


def owners(e, total):
    try:
        pct = float(e.get("selected_by_percent") or 0.0)
    except (TypeError, ValueError):
        pct = 0.0
    return max(MIN_OWNERS, pct / 100.0 * max(total, 1))


def net_share(e, total):
    """Net transfers as a fraction of the people who already hold him."""
    net = (e.get("transfers_in_event") or 0) - (e.get("transfers_out_event") or 0)
    return net / owners(e, total), net


def calibrate(boot):
    """The share of owners that has actually been enough to move a price."""
    total = boot.get("total_players") or 10_000_000
    shares = sorted(net_share(e, total)[0]
                    for e in boot["elements"] if e.get("cost_change_event", 0) > 0)
    if len(shares) >= 5:
        return shares[len(shares) // 10]      # 10th percentile of confirmed risers
    return 0.06                               # a sane default before any has moved


def forecast(boot):
    total = boot.get("total_players") or 10_000_000
    up = calibrate(boot)
    down = -up * 1.6                          # falls need a bigger flow than rises
    out = {}
    for e in boot["elements"]:
        share, net = net_share(e, total)
        if share >= up:
            d, prox = "rise", min(1.0, share / (up * 2))
        elif share <= down:
            d, prox = "fall", min(1.0, share / (down * 2))
        else:
            d, prox = "hold", 0.0
        out[e["id"]] = {"net": net, "share": round(share, 4),
                        "direction": d, "closeness": round(prox, 2)}
    return out


def squad_drift(forecasts, squad_ids):
    """Net expected team-value change tonight, in tenths of a million."""
    r = sum(1 for i in squad_ids if forecasts.get(i, {}).get("direction") == "rise")
    f = sum(1 for i in squad_ids if forecasts.get(i, {}).get("direction") == "fall")
    return r - f
