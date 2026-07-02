#!/usr/bin/env python3
"""Flatten a paper book — close every open position at the current mark and go to
cash — so you can safely turn a bot off without leaving positions hanging.

Each close is recorded as a realized trade (reason "manual flatten", friction
applied), and portfolio.json is backed up first. Run AFTER stopping the bot, so it
can't reopen underneath you:

  pkill -f 'config.yaml --state-dir state'     # stop the main bot
  python3 scripts/flatten_book.py state         # flatten it (default: the main book)
  python3 scripts/flatten_book.py state_experiment_trail   # or any other book
"""

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fablebot.config import load_config
from fablebot.data import source_from_config

ROOT = Path(__file__).resolve().parent.parent
_SIGN = {"long": 1.0, "short": -1.0}


def _config_for(state_dir: str) -> Path:
    name = Path(state_dir).name
    return ROOT / ("config.yaml" if name == "state" else f"config_{name[len('state_'):]}.yaml")


def flatten(state_dir: str) -> None:
    pf = ROOT / state_dir / "portfolio.json"
    if not pf.exists():
        print(f"{state_dir}: no portfolio.json — skipping")
        return
    p = json.loads(pf.read_text())
    pos = p.get("positions", {})
    if not pos:
        print(f"{state_dir}: already flat")
        return

    cfg = load_config(_config_for(state_dir))
    src = source_from_config(cfg, cache_dir=None)          # live marks, never CSV
    friction = cfg.costs.friction_bps / 10_000.0
    now = datetime.now(timezone.utc).isoformat()
    shutil.copy2(pf, pf.with_suffix(".json.bak"))

    total = 0.0
    for key, v in list(pos.items()):
        inst = cfg.instruments.get(key)
        try:
            mark = float(src.fetch(inst.symbol, inst.timeframe)["close"].iloc[-1])
        except Exception as exc:
            print(f"   {key}: couldn't fetch a mark ({exc}) — LEAVING it open")
            continue
        units, entry, d = float(v.get("units", 0)), float(v.get("entry_price", 0)), v.get("direction")
        pnl = (mark - entry) * units * _SIGN.get(d, 0.0) - abs(mark * units) * friction
        p.setdefault("trades", []).append({
            "instrument": key, "direction": d, "units": units, "entry_price": entry,
            "exit_price": mark, "pnl": round(pnl, 2), "closed_at": now,
            "opened_at": v.get("opened_at", now), "close_reason": "manual flatten",
            "strategy": v.get("strategy", ""),
        })
        del pos[key]
        total += pnl
        print(f"   {key} {d} closed @ {mark:,.2f}   pnl {pnl:+,.2f}")

    p["realized_pnl"] = round(p.get("realized_pnl", 0.0) + total, 2)
    p["positions"] = pos
    pf.write_text(json.dumps(p, indent=2))
    se = p.get("starting_equity", 0.0)
    status = "FLAT" if not pos else f"{len(pos)} still open (mark fetch failed)"
    print(f"{state_dir}: {status}, realized this flatten {total:+,.2f} -> equity ${se + p['realized_pnl']:,.0f}  [backup written]")


if __name__ == "__main__":
    for d in (sys.argv[1:] or ["state"]):
        flatten(d)
