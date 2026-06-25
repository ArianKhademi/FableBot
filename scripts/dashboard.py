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


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate the FableBot performance dashboard")
    ap.add_argument("--config", default=ROOT / "config.yaml")
    ap.add_argument("--state-dir", default=ROOT / "state",
                    help="a book's state dir; --portfolio and --out default to files inside it, "
                         "so `--state-dir state_experiment --open` just works for any bot")
    ap.add_argument("--portfolio", default=None, help="defaults to <state-dir>/portfolio.json")
    ap.add_argument("--out", default=None, help="defaults to <state-dir>/dashboard.html")
    ap.add_argument("--no-marks", action="store_true", help="skip live price fetch (faster; no unrealized)")
    ap.add_argument("--open", action="store_true", help="open the page in your browser when done")
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    pf_path = Path(args.portfolio) if args.portfolio else state_dir / "portfolio.json"
    if not pf_path.exists():
        print(f"[dashboard] no portfolio at {pf_path} — has the bot run yet?", file=sys.stderr)
        return 1
    portfolio = json.loads(pf_path.read_text())

    config = load_config(args.config)
    marks = {}
    if not args.no_marks:
        try:
            marks = current_marks(config, portfolio.get("positions", {}))
        except Exception as exc:
            print(f"[dashboard] live marks unavailable ({exc}) — showing realized only", file=sys.stderr)

    snapshot = dashboard.build_snapshot(portfolio, marks)
    engine = dashboard.read_engine_state(state_dir)
    generated = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
    html = dashboard.render_html(snapshot, engine, generated, title=config.name)

    out = Path(args.out) if args.out else state_dir / "dashboard.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"[dashboard] wrote {out}  (equity {dashboard._money(snapshot['account']['total_equity'])}, "
          f"{snapshot['stats']['count']} trades, {len(snapshot['positions'])} open)")
    if args.open:
        subprocess.run(["open", str(out)], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
