#!/usr/bin/env python3
"""Combined dashboard — every paper book on one page.

Discovers each book (state/, state_experiment*/, …), reads its portfolio, and
renders one comparison page: equity, return, P&L and win rate per book, plus
their equity trajectories overlaid. Open it to watch all the bots at once without
touching the per-book dashboards (which keep updating on their own).

  python3 scripts/dashboard_all.py --open
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fablebot import dashboard
from fablebot.config import load_config
from fablebot.data import source_from_config

ROOT = Path(__file__).resolve().parent.parent


def _config_for(state_dir: Path) -> Path:
    """The config that goes with a state dir, by convention: state/ -> config.yaml,
    state_experiment -> config_experiment.yaml, state_experiment_trail -> ..."""
    name = state_dir.name
    return ROOT / ("config.yaml" if name == "state" else f"config_{name[len('state_'):]}.yaml")


def _marks(config, positions: dict) -> dict:
    if not positions:
        return {}
    try:
        source = source_from_config(config, cache_dir=None)
    except Exception:
        return {}
    out = {}
    for key in positions:
        inst = config.instruments.get(key)
        if inst is None:
            continue
        try:
            out[key] = float(source.fetch(inst.symbol, inst.timeframe)["close"].iloc[-1])
        except Exception:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Combined dashboard for all paper books")
    ap.add_argument("--out", default=ROOT / "state/dashboard_all.html")
    ap.add_argument("--no-marks", action="store_true", help="skip live price fetch (faster)")
    ap.add_argument("--open", action="store_true", help="open the page when done")
    ap.add_argument("--exclude", default="",
                    help="comma-separated state dir names to hide (e.g. 'state' to drop the main book)")
    args = ap.parse_args()

    # Every state dir with a matching config — even a fresh book that hasn't
    # traded yet (no portfolio.json until its first fill); show it at its start.
    out = Path(args.out)
    excluded = {x.strip() for x in args.exclude.split(",") if x.strip()}
    state_dirs = sorted(p for p in ROOT.glob("state*")
                        if p.is_dir() and _config_for(p).exists() and p.name not in excluded)
    books = []
    for sd in state_dirs:
        cfg = load_config(_config_for(sd))
        pf = sd / "portfolio.json"
        if pf.exists():
            try:
                portfolio = json.loads(pf.read_text())
            except Exception:
                continue
        else:
            portfolio = {"starting_equity": cfg.account.starting_equity,
                         "realized_pnl": 0.0, "positions": {}, "trades": []}
        marks = {} if args.no_marks else _marks(cfg, portfolio.get("positions", {}))
        snap = dashboard.build_snapshot(portfolio, marks)
        # Link the book's row to its own dashboard.html (the live bot / dashboard.py
        # writes one per state dir), as a path relative to this combined page.
        dash = sd / "dashboard.html"
        href = os.path.relpath(dash, out.parent) if dash.exists() else None
        books.append({"name": cfg.name, "href": href, "account": snap["account"],
                      "stats": snap["stats"], "equity_curve": snap["equity_curve"]})

    if not books:
        print("No books found (no state*/portfolio.json yet). Start a bot first.", file=sys.stderr)
        return 1
    generated = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dashboard.render_overview(books, generated))
    print(f"[dashboard] wrote {out}  ({len(books)} books: {', '.join(b['name'] for b in books)})")
    if args.open:
        subprocess.run(["open", str(out)], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
