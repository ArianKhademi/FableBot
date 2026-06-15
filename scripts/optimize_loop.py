#!/usr/bin/env python3
"""Continuous walk-forward parameter optimizer — run this alongside run_bot.py.

Runs forever, round-robins through all instruments that have param_bounds,
samples random parameter sets, validates them out-of-sample, and logs
every trial to state/backtest_history.jsonl. Promising configurations
(OOS Sharpe > baseline + 0.15, >= 8 OOS trades) enter the proposal pipeline
as candidates; each loop pass also validates one pending candidate
(regime folds + perturbation test) and marks it qualified or rejected.
Only qualified proposals reach the nightly reviewer.

  python3 scripts/optimize_loop.py                       # all cores, throttled
  python3 scripts/optimize_loop.py --workers 4           # cap at 4 cores
  python3 scripts/optimize_loop.py --no-throttle         # dedicated run, full tilt
  python3 scripts/optimize_loop.py --instrument BTC      # one instrument only

Each backtest is independent and CPU-bound, so trials run across worker
processes (default: all cores) for ~Ncores x more trials/day. The 1.5s
inter-pass throttle keeps a core free when sharing the box with the live bot;
--no-throttle removes it for a dedicated optimisation run.

Nothing here changes config.yaml or any live code. It only produces
evidence that the reviewer reads and may act on.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fablebot.config import load_config
from fablebot.optimizer import BacktestHistory, ContinuousOptimizer, TRIAL_SLEEP
from fablebot.pipeline import ProposalStore

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="FableBot continuous optimizer")
    parser.add_argument("--config", default=ROOT / "config.yaml")
    parser.add_argument("--history", default=ROOT / "state/backtest_history.jsonl")
    parser.add_argument("--proposals", default=ROOT / "state/proposals.jsonl")
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count() or 1,
        help="parallel backtest workers (default: all cores)",
    )
    parser.add_argument(
        "--sleep", type=float, default=None,
        help="seconds between passes (default: throttle setting)",
    )
    parser.add_argument(
        "--no-throttle", action="store_true",
        help="remove the inter-pass pause (use on a dedicated run)",
    )
    parser.add_argument("--instrument", help="only optimize this instrument key")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.instrument:
        if args.instrument not in config.instruments:
            sys.exit(f"Unknown instrument {args.instrument!r}; choose from {sorted(config.instruments)}")
        # Restrict to the single instrument by hiding the others
        config.instruments = {args.instrument: config.instruments[args.instrument]}

    # Fetch through the configured (optionally cached) data source.
    from fablebot.data import get_source
    cache_dir = config.data.cache_dir
    if cache_dir and not Path(cache_dir).is_absolute():
        cache_dir = ROOT / cache_dir
    source = get_source(config.data.source, cache_dir=cache_dir, ttl_minutes=config.data.cache_ttl_minutes)

    history = BacktestHistory(args.history)
    proposals = ProposalStore(args.proposals)
    optimizer = ContinuousOptimizer(
        config, history, proposals,
        fetch_fn=lambda symbol, timeframe, days: source.fetch(symbol, timeframe, days),
        friction_bps=config.costs.friction_bps,
    )

    try:
        optimizer.run_forever(sleep=args.sleep, workers=args.workers, throttle=not args.no_throttle)
    except KeyboardInterrupt:
        print("\n[optimizer] stopped.")


if __name__ == "__main__":
    main()
