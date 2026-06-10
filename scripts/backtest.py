#!/usr/bin/env python3
"""Backtest one configured instrument over recent history.

  python scripts/backtest.py BTC --days 120
  python scripts/backtest.py SPX
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fablebot.backtest import run_backtest
from fablebot.config import load_config
from fablebot.data import fetch_ohlcv
from fablebot.strategies import build_strategy

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="FableBot single-instrument backtest")
    parser.add_argument("instrument", help="instrument key from config.yaml (e.g. BTC)")
    parser.add_argument("--config", default=ROOT / "config.yaml")
    parser.add_argument("--equity", type=float, default=100_000.0)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.instrument not in config.instruments:
        sys.exit(f"Unknown instrument {args.instrument!r}; choose from {sorted(config.instruments)}")
    inst = config.instruments[args.instrument]

    print(f"Fetching {inst.symbol} ({inst.timeframe})...")
    df = fetch_ohlcv(inst.symbol, inst.timeframe)
    print(f"{len(df)} bars from {df.index[0]} to {df.index[-1]}")

    strategy = build_strategy(inst.strategy, inst.params)
    result = run_backtest(
        df,
        strategy,
        starting_equity=args.equity,
        risk_per_trade=config.account.risk_per_trade,
        instrument=args.instrument,
    )
    print(f"\n{inst.name} — {strategy.name} on {inst.timeframe}")
    print(result.summary())


if __name__ == "__main__":
    main()
