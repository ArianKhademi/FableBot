#!/usr/bin/env python3
"""Run the FableBot paper-trading loop, or fire a report on demand.

  python scripts/run_bot.py                   # live loop
  python scripts/run_bot.py --once            # single evaluation cycle
  python scripts/run_bot.py --report morning  # 7am-style briefing now
  python scripts/run_bot.py --report evening  # 9pm-style recap now
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fablebot.config import load_config
from fablebot.engine import TradingEngine

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="FableBot paper-trading bot")
    parser.add_argument("--config", default=ROOT / "config.yaml")
    parser.add_argument("--once", action="store_true", help="run a single cycle and exit")
    parser.add_argument("--report", choices=["morning", "evening"])
    args = parser.parse_args()

    config = load_config(args.config)

    # Two routed sources (per-instrument backend, e.g. crypto -> Binance):
    #  - live feed: UNcached so each cycle trades on fresh prices.
    #  - history feed: cached + deep, for training live ML/RL models (the live
    #    window is too shallow to train on); daily retrains hit the cache.
    from fablebot.data import source_from_config
    from fablebot.optimizer import OPTIMIZER_FETCH_DAYS
    cache_dir = config.data.cache_dir
    if cache_dir and not Path(cache_dir).is_absolute():
        cache_dir = ROOT / cache_dir
    live_source = source_from_config(config, cache_dir=None)
    history_source = source_from_config(config, cache_dir=cache_dir)

    def history_fetch(symbol: str, timeframe: str):
        return history_source.fetch(symbol, timeframe, days=OPTIMIZER_FETCH_DAYS.get(timeframe, 60))

    engine = TradingEngine(
        config,
        state_path=ROOT / "state/portfolio.json",
        fetch=live_source.fetch,
        journal_path=ROOT / "state/journal.jsonl",
        shadow_state_path=ROOT / "state/shadow.json",
        history_fetch=history_fetch,
    )

    if args.report == "morning":
        engine.send_morning_report()
    elif args.report == "evening":
        engine.run_cycle()  # refresh prices first
        engine.send_evening_report()
    elif args.once:
        for line in engine.run_cycle():
            print(line)
    else:
        try:
            engine.run_forever()
        except KeyboardInterrupt:
            print("\nFableBot stopped.")


if __name__ == "__main__":
    main()
