#!/usr/bin/env python3
"""Daily performance digest across every paper book -> one Markdown file.

Discovers each book (state/, state_experiment*/, …), reads its portfolio, and
computes account state, trade stats, per-instrument/strategy breakdowns and
trailing-7-day P&L, then writes a single Markdown report — built for a cron so
you wake up to how the bots and the A/B experiments are stacking up. Read-only:
it only reads portfolio.json, never touches trading state.

  python3 scripts/bot_report.py                    # -> ~/Desktop/FableBot_Report.md
  python3 scripts/bot_report.py --out report.md --print
  python3 scripts/bot_report.py --no-marks         # skip live price fetch (faster)

Suggested cron (Apple python so macOS lets cron read/write files; 7:30am daily):
  30 7 * * * cd /Users/<you>/Documents/Business/FableBot && /usr/bin/python3 scripts/bot_report.py >> logs/bot_report.log 2>&1
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fablebot import dashboard, reporting
from fablebot.config import load_config
from fablebot.data import source_from_config

ROOT = Path(__file__).resolve().parent.parent


def _config_for(state_dir: Path) -> Path:
    """state/ -> config.yaml; state_experiment_trail -> config_experiment_trail.yaml."""
    name = state_dir.name
    return ROOT / ("config.yaml" if name == "state" else f"config_{name[len('state_'):]}.yaml")


def _marks(config, positions: dict) -> dict:
    """Last price per open instrument (live feed, cache_dir=None -> never CSV)."""
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
    ap = argparse.ArgumentParser(description="Daily performance digest across all paper books")
    ap.add_argument("--out", default=Path.home() / "Desktop" / "FableBot_Report.md")
    ap.add_argument("--no-marks", action="store_true", help="skip live price fetch (unrealized shown as 0)")
    ap.add_argument("--print", dest="echo", action="store_true", help="also print the report to stdout")
    args = ap.parse_args()

    now = datetime.now()  # naive local — matches the (tz-stripped) trade timestamps
    state_dirs = sorted(p for p in ROOT.glob("state*") if p.is_dir() and _config_for(p).exists())
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
        books.append({
            "name": cfg.name,
            "account": snap["account"], "stats": snap["stats"],
            "by_instrument": snap["by_instrument"], "by_strategy": snap["by_strategy"],
            "weekly": reporting.weekly_pnl(portfolio.get("trades", []), now),
        })

    generated = now.astimezone().strftime("%Y-%m-%d %H:%M %Z")
    md = reporting.render_report(books, generated)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"[bot_report] wrote {out}  ({len(books)} books)")
    if args.echo:
        print("\n" + md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
