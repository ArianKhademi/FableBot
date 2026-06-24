#!/usr/bin/env python3
"""Generate the read-only performance dashboard (state/dashboard.html).

Reads portfolio.json + engine state, fetches the current price for each open
position to mark it to market, and writes a self-contained HTML page you open in
a browser (it auto-refreshes itself). Read-only — it never touches trading state.

  python3 scripts/dashboard.py            # write state/dashboard.html
  python3 scripts/dashboard.py --open     # ...and open it in your browser

Keep it fresh with cron (every 5 min), pinned to Apple's python like the others:
  */5 * * * * cd /…/FableBot && /usr/bin/python3 scripts/dashboard.py >> logs/dashboard.log 2>&1
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fablebot import dashboard
from fablebot.config import load_config
from fablebot.data import source_from_config

ROOT = Path(__file__).resolve().parent.parent


def current_marks(config, positions: dict) -> dict:
    """Latest price per open instrument, from the live (uncached) feed. A fetch
    failure leaves that instrument unmarked rather than guessing."""
    if not positions:
        return {}
    source = source_from_config(config, cache_dir=None)
    marks: dict[str, float] = {}
    for key in positions:
        inst = config.instruments.get(key)
        if inst is None:
            continue
        try:
            df = source.fetch(inst.symbol, inst.timeframe)
            marks[key] = float(df["close"].iloc[-1])
        except Exception as exc:
            print(f"[dashboard] {key}: could not mark ({exc})", file=sys.stderr)
    return marks


def build_snapshot(portfolio: dict, marks: dict) -> dict:
    positions = portfolio.get("positions", {})
    trades = portfolio.get("trades", [])
    pos_list = []
    for key, p in positions.items():
        mark = marks.get(key)
        unreal = dashboard.unrealized_pnl({key: p}, marks) if mark is not None else 0.0
        pos_list.append({
            "instrument": p.get("instrument", key), "direction": p.get("direction", "?"),
            "entry_price": float(p.get("entry_price", 0)), "stop_price": float(p.get("stop_price", 0)),
            "units": float(p.get("units", 0)), "mark": mark, "unrealized": unreal,
        })
    recent = sorted((t for t in trades if t.get("closed_at")), key=lambda t: t["closed_at"], reverse=True)[:12]
    return {
        "account": dashboard.account_summary(
            portfolio.get("starting_equity", 0.0), portfolio.get("realized_pnl", 0.0), positions, marks),
        "stats": dashboard.trade_stats(trades),
        "equity_curve": dashboard.equity_curve(portfolio.get("starting_equity", 0.0), trades),
        "by_strategy": dashboard.strategy_breakdown(trades),
        "by_instrument": dashboard.instrument_breakdown(trades),
        "positions": pos_list,
        "recent_trades": recent,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate the FableBot performance dashboard")
    ap.add_argument("--config", default=ROOT / "config.yaml")
    ap.add_argument("--portfolio", default=ROOT / "state/portfolio.json")
    ap.add_argument("--out", default=ROOT / "state/dashboard.html")
    ap.add_argument("--state-dir", default=ROOT / "state")
    ap.add_argument("--no-marks", action="store_true", help="skip live price fetch (faster; no unrealized)")
    ap.add_argument("--open", action="store_true", help="open the page in your browser when done")
    args = ap.parse_args()

    pf_path = Path(args.portfolio)
    if not pf_path.exists():
        print(f"[dashboard] no portfolio at {pf_path} — has the bot run yet?", file=sys.stderr)
        return 1
    portfolio = json.loads(pf_path.read_text())

    marks = {}
    if not args.no_marks:
        try:
            marks = current_marks(load_config(args.config), portfolio.get("positions", {}))
        except Exception as exc:
            print(f"[dashboard] live marks unavailable ({exc}) — showing realized only", file=sys.stderr)

    snapshot = build_snapshot(portfolio, marks)
    engine = dashboard.read_engine_state(Path(args.state_dir))
    generated = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
    html = dashboard.render_html(snapshot, engine, generated)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"[dashboard] wrote {out}  (equity {dashboard._money(snapshot['account']['total_equity'])}, "
          f"{snapshot['stats']['count']} trades, {len(snapshot['positions'])} open)")
    if args.open:
        subprocess.run(["open", str(out)], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
