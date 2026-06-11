#!/usr/bin/env python3
"""Validation gate for proposed changes.

Any change the learning loop proposes must pass this before it goes in a PR:

  1. the full test suite (including the risk-invariant tests)
  2. a backtest of every configured instrument over recent history

Run it on main and on the proposal branch; the PR must include both outputs
so the diff in behavior is visible, not asserted.

  python3 scripts/validate.py
  python3 scripts/validate.py --skip-backtests   # offline / tests only
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent


def run_tests() -> bool:
    print("== test suite ==")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True
    )
    print(result.stdout.strip().splitlines()[-1] if result.stdout.strip() else result.stderr)
    return result.returncode == 0


def run_backtests() -> bool:
    from fablebot.backtest import run_backtest
    from fablebot.config import load_config
    from fablebot.data import fetch_ohlcv
    from fablebot.strategies import build_strategy

    print("\n== backtests ==")
    config = load_config(ROOT / "config.yaml")
    ok = True
    for key, inst in config.instruments.items():
        try:
            df = fetch_ohlcv(inst.symbol, inst.timeframe)
            strategy = build_strategy(inst.strategy, inst.params)
            result = run_backtest(
                df,
                strategy,
                risk_per_trade=config.account.risk_per_trade,
                instrument=key,
            )
            print(f"{key} ({inst.strategy}, {inst.timeframe}): {result.summary()}")
        except Exception as exc:
            print(f"{key}: backtest failed — {exc}")
            ok = False
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="FableBot change validation gate")
    parser.add_argument(
        "--skip-backtests", action="store_true", help="run only the test suite (no network)"
    )
    args = parser.parse_args()

    tests_ok = run_tests()
    backtests_ok = True if args.skip_backtests else run_backtests()

    if tests_ok and backtests_ok:
        print("\nVALIDATION PASSED")
    else:
        print("\nVALIDATION FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
