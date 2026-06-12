#!/usr/bin/env python3
"""Continuous walk-forward parameter optimizer — run this alongside run_bot.py.

Runs forever, round-robins through all instruments that have param_bounds,
samples random parameter sets, validates them out-of-sample, and logs
every trial to state/backtest_history.jsonl. Promising configurations
(OOS Sharpe > baseline + 0.15, >= 8 OOS trades) go to state/proposals.jsonl
for the nightly reviewer.

  python3 scripts/optimize_loop.py
  python3 scripts/optimize_loop.py --sleep 3   # slower, less CPU
  python3 scripts/optimize_loop.py --instrument BTC  # one instrument only

Nothing here changes config.yaml or any live code. It only produces
evidence that the reviewer reads and may act on.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fablebot.config import load_config
from fablebot.optimizer import (
    BacktestHistory,
    ContinuousOptimizer,
    ProposalQueue,
    TRIAL_SLEEP,
)

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="FableBot continuous optimizer")
    parser.add_argument("--config", default=ROOT / "config.yaml")
    parser.add_argument("--history", default=ROOT / "state/backtest_history.jsonl")
    parser.add_argument("--proposals", default=ROOT / "state/proposals.jsonl")
    parser.add_argument("--sleep", type=float, default=TRIAL_SLEEP, help="seconds between trials")
    parser.add_argument("--instrument", help="only optimize this instrument key")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.instrument:
        if args.instrument not in config.instruments:
            sys.exit(f"Unknown instrument {args.instrument!r}; choose from {sorted(config.instruments)}")
        # Restrict to the single instrument by hiding the others
        config.instruments = {args.instrument: config.instruments[args.instrument]}

    history = BacktestHistory(args.history)
    proposals = ProposalQueue(args.proposals)
    optimizer = ContinuousOptimizer(config, history, proposals)

    try:
        optimizer.run_forever(sleep=args.sleep)
    except KeyboardInterrupt:
        print("\n[optimizer] stopped.")


if __name__ == "__main__":
    main()
