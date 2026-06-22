#!/usr/bin/env python3
"""Is an instrument FLAT (no open paper position) right now?

Reads the live broker state in ``state/portfolio.json``. An instrument is OPEN
iff it appears in ``positions``; absent => FLAT. Used to gate config changes that
must not land mid-trade — e.g. switching NDX's timeframe would orphan a position
opened on the old timeframe.

  python3 scripts/position_flat.py NDX

Exit codes (chosen so a shell can branch on them):
  0  FLAT     — safe (instrument has no open position)
  1  OPEN     — not safe (instrument is currently in a trade)
  2  UNKNOWN  — portfolio.json missing/unreadable; treat as NOT safe, take no action
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FLAT, OPEN, UNKNOWN = 0, 1, 2


def open_instruments(portfolio: dict) -> set:
    """Instrument keys with an open position in a parsed portfolio dict."""
    positions = portfolio.get("positions", {})
    return set(positions) if isinstance(positions, dict) else set()


def is_flat(portfolio: dict, instrument: str) -> bool:
    return instrument not in open_instruments(portfolio)


def classify(portfolio_path: Path, instrument: str) -> tuple[int, str]:
    """Return (exit_code, human_message). A missing or unparseable file is
    UNKNOWN, never FLAT — we must not signal 'safe' on uncertain state."""
    if not portfolio_path.exists():
        return UNKNOWN, f"UNKNOWN {instrument}: {portfolio_path} does not exist — cannot confirm flat"
    try:
        portfolio = json.loads(portfolio_path.read_text())
    except Exception as exc:  # noqa: BLE001 — any parse failure is 'unknown'
        return UNKNOWN, f"UNKNOWN {instrument}: cannot parse {portfolio_path} ({exc})"
    openset = sorted(open_instruments(portfolio))
    if instrument in openset:
        return OPEN, f"OPEN {instrument} (open: {openset})"
    return FLAT, f"FLAT {instrument} (open: {openset})"


def main() -> int:
    ap = argparse.ArgumentParser(description="Report whether an instrument is flat")
    ap.add_argument("instrument", help="instrument key, e.g. NDX")
    ap.add_argument("--portfolio", default=ROOT / "state/portfolio.json",
                    help="path to portfolio.json (default: state/portfolio.json)")
    args = ap.parse_args()
    code, msg = classify(Path(args.portfolio), args.instrument)
    print(msg, file=sys.stderr if code == UNKNOWN else sys.stdout)
    return code


if __name__ == "__main__":
    sys.exit(main())
