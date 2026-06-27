#!/usr/bin/env python3
"""Recover a paper book corrupted by a runaway loop.

Drops every trade after the book's realized-equity PEAK (the loss trades the loop
piled on) and flattens open positions, restoring the book to its best pre-incident
state — which the append-only trade log preserves exactly. Backs up portfolio.json
to portfolio.json.bak first.

Run AFTER stopping the bot, so it can't overwrite the restore:
  pkill -f run_bot.py
  python3 scripts/restore_book.py            # the 4 experiment books (default)
  python3 scripts/restore_book.py state_experiment   # a specific book
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ["state_experiment", "state_experiment_trail",
           "state_experiment_regime", "state_experiment_adaptive"]


def restore(state_dir: str) -> None:
    pf = ROOT / state_dir / "portfolio.json"
    if not pf.exists():
        print(f"{state_dir}: no portfolio.json — skipping")
        return
    p = json.loads(pf.read_text())
    trades = p.get("trades", [])

    cum = peak = 0.0
    peak_i = -1
    for i, t in enumerate(trades):
        cum += t.get("pnl", 0.0)
        if cum > peak:
            peak, peak_i = cum, i

    before_n, before_real = len(trades), p.get("realized_pnl", 0.0)
    kept = trades[: peak_i + 1]
    shutil.copy2(pf, pf.with_suffix(".json.bak"))
    p["trades"] = kept
    p["realized_pnl"] = round(sum(t.get("pnl", 0.0) for t in kept), 2)
    p["positions"] = {}  # the loop's open positions are garbage — restart flat
    pf.write_text(json.dumps(p, indent=2))

    se = p.get("starting_equity", 0.0)
    print(f"{state_dir:<28} {before_n:>3} trades / {before_real:+,.0f}  ->  "
          f"{len(kept):>2} trades / {p['realized_pnl']:+,.0f}  "
          f"(equity ${se + p['realized_pnl']:,.0f})   [backup written]")


if __name__ == "__main__":
    for d in (sys.argv[1:] or DEFAULT):
        restore(d)
