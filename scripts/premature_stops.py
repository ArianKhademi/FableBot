#!/usr/bin/env python3
"""Premature stop analysis — how often did the hard 1% stop cut a trade that
would have come back?

For every trade closed by the hard stop, fetch the price bars AFTER the stop and
check whether price returned to break-even within a horizon, and how far in your
favor it then went. A high recovery rate means the stop (or the entry timing) is
too tight — you're booking losses on trades that revert, which is exactly the
risk to watch for with mean-reversion.

Read-only. Limited to trades whose post-stop bars are still in the live data
window (recent trades for intraday; the WRDS/CSV history feed extends this later).
Uses each instrument's CURRENT config timeframe, so it's most accurate for books
whose timeframe hasn't changed — run it on the experiment book for a clean 15m
mean-reversion read:

  python3 scripts/premature_stops.py --horizon 24
  python3 scripts/premature_stops.py --config config_experiment.yaml \
      --portfolio state_experiment/portfolio.json --horizon 24
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fablebot import analytics
from fablebot.config import load_config
from fablebot.data import source_from_config

ROOT = Path(__file__).resolve().parent.parent


def _post_bars(df: pd.DataFrame, closed_at: str, horizon: int):
    """The (high, low) bars strictly after the stop time, aligned to the frame's
    timezone so the comparison is valid."""
    ts = pd.Timestamp(closed_at)
    idx = df.index
    if getattr(idx, "tz", None) is not None:
        ts = ts.tz_localize("UTC") if ts.tz is None else ts
        ts = ts.tz_convert(idx.tz)
    elif ts.tz is not None:
        ts = ts.tz_localize(None)
    post = df[df.index > ts].head(horizon)
    return list(zip(post["high"].tolist(), post["low"].tolist()))


def main() -> int:
    ap = argparse.ArgumentParser(description="Premature hard-stop recovery analysis")
    ap.add_argument("--config", default=ROOT / "config.yaml")
    ap.add_argument("--portfolio", default=ROOT / "state/portfolio.json")
    ap.add_argument("--horizon", type=int, default=24, help="bars to look ahead after a stop")
    args = ap.parse_args()

    config = load_config(args.config)
    pf = Path(args.portfolio)
    if not pf.exists():
        print(f"No portfolio at {pf}.", file=sys.stderr)
        return 1
    trades = json.loads(pf.read_text()).get("trades", [])
    stops = [t for t in trades if analytics._is_stop(t) and t.get("closed_at")]
    if not stops:
        print("No hard-stop trades recorded yet — nothing to analyse.")
        return 0

    source = source_from_config(config, cache_dir=None)
    bars: dict[str, pd.DataFrame] = {}
    for key in {t["instrument"] for t in stops}:
        inst = config.instruments.get(key)
        if inst is None:
            continue
        try:
            bars[key] = source.fetch(inst.symbol, inst.timeframe)
        except Exception as exc:
            print(f"[premature] {key}: fetch failed ({exc})", file=sys.stderr)

    blank = lambda: {"n": 0, "analyzable": 0, "recovered": 0, "loss": 0.0, "left": 0.0}
    overall, by_strategy = blank(), defaultdict(blank)
    recovered_rows = []
    for t in stops:
        df = bars.get(t["instrument"])
        groups = (overall, by_strategy[t.get("strategy", "?")])
        for d in groups:
            d["n"] += 1
        if df is None:
            continue
        post = _post_bars(df, t["closed_at"], args.horizon)
        if not post:
            continue  # post-stop bars are outside the available data window
        r = analytics.stop_recovery(t["direction"], t["entry_price"], post, args.horizon)
        left = r["best_favorable_per_unit"] * t.get("units", 0.0)
        for d in groups:
            d["analyzable"] += 1
            if r["recovered"]:
                d["recovered"] += 1
                d["loss"] += min(t.get("pnl", 0.0), 0.0)
                d["left"] += left
        if r["recovered"]:
            recovered_rows.append((t["closed_at"][:16], t["instrument"], t.get("strategy", "?"),
                                   t.get("pnl", 0.0), left, r["bars_to_recover"]))

    pct = lambda d: (d["recovered"] / d["analyzable"] * 100.0) if d["analyzable"] else 0.0
    print("=" * 72)
    print("PREMATURE STOP ANALYSIS  (hard stops that returned to break-even)")
    print("=" * 72)
    print(f"\nHorizon: {args.horizon} bars past each stop. Analyzable (post-stop bars still in "
          f"the data window): {overall['analyzable']}/{overall['n']} hard-stop trades.\n")
    print(f"  OVERALL: {overall['recovered']}/{overall['analyzable']} ({pct(overall):.0f}%) recovered to break-even")
    print(f"           ${overall['loss']:,.0f} booked on them, ~${overall['left']:,.0f} of favorable move left on the table")
    print("\n  by strategy:")
    for s, d in sorted(by_strategy.items(), key=lambda kv: -pct(kv[1])):
        if d["analyzable"]:
            print(f"    {s:18} {d['recovered']:>2}/{d['analyzable']:<2} ({pct(d):>3.0f}%) recovered | ${d['loss']:,.0f} avoidable loss")
    if recovered_rows:
        print("\n  biggest premature stops (loss booked -> would-have favorable):")
        for ts, k, s, pnl, left, n in sorted(recovered_rows, key=lambda r: -r[4])[:12]:
            print(f"    {ts}  {k:5} {s:16} {pnl:+,.0f} -> +{left:,.0f} within {n} bars")
    print("\nHigh recovery % = the stop or entry is too tight (booking losses on trades that revert);"
          "\nlow % = the stop is doing its job (cutting trades that kept going against you).")
    print("Note: recovery = price *touched* break-even intrabar within the horizon, so on very"
          "\nvolatile instruments a wick can trigger it — read high %s as 'stops are getting wicked"
          "\nout' rather than 'guaranteed profit'. Widening the stop (atr_stop_mult) is the lever.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
