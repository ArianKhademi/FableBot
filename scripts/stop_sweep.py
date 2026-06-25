#!/usr/bin/env python3
"""Stop sweep — backtest a grid of (hard-stop width × trailing-stop width) for a
live strategy and report which combo is best.

`premature_stops.py` only *diagnoses* that stops are too tight; this *recommends*
a width. For each instrument it re-runs the same walk-forward backtest the gate
uses over a grid of `atr_stop_mult` (the hard 1% stop's ATR distance) and
`trail_atr_mult` (the trailing stop; 0 = no trail), and prints the OOS Sharpe of
each — so you can see, e.g., "BTC momentum: best at hard 4.0 / trail 1.5".

Read-only — evidence, like every backtest. Nothing here edits config.

  python3 scripts/stop_sweep.py --instrument BTC
  python3 scripts/stop_sweep.py                         # every traded instrument
  python3 scripts/stop_sweep.py --instrument GOLD --stops 2,3,4,5,6 --trails 0,1,1.5,2,3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fablebot.config import load_config
from fablebot.optimizer import OPTIMIZER_FETCH_DAYS, run_walk_forward

ROOT = Path(__file__).resolve().parent.parent


def sweep(df, inst, stop_mults, trail_mults, near_mults=(None,), risk=0.01, friction=0.0) -> list[dict]:
    """Backtest each (atr_stop_mult, trail_atr_mult, trail_near_mult) combo. trail
    0 = no trail; near None = constant trail (the adaptive narrowing trail is
    mean-reversion only). Pure: takes the data + grid, one row per combo."""
    rows = []
    for s in stop_mults:
        for t in trail_mults:
            for n in near_mults:
                params = {**inst.params, "atr_stop_mult": s}
                if t and t > 0:
                    params["trail_atr_mult"] = t
                else:
                    params.pop("trail_atr_mult", None)
                # the adaptive narrowing trail only makes sense with a trail set
                if n is not None and t and t > 0:
                    params["trail_near_mult"] = n
                else:
                    params.pop("trail_near_mult", None)
                trial = run_walk_forward(df, inst, params, risk_per_trade=risk, friction_bps=friction)
                rows.append({
                    "stop": s, "trail": t, "near": n,
                    "oos_sharpe": trial.oos_sharpe if trial else None,
                    "oos_trades": trial.oos_trades if trial else 0,
                })
    return rows


def _grid(arg, default):
    return [float(x) for x in arg.split(",")] if arg else default


def main() -> int:
    ap = argparse.ArgumentParser(description="Grid-sweep the hard + trailing stop widths")
    ap.add_argument("--config", default=ROOT / "config.yaml")
    ap.add_argument("--instrument", help="one instrument key; default = all traded")
    ap.add_argument("--stops", help="comma list of atr_stop_mult (default 1.5..6)")
    ap.add_argument("--trails", help="comma list of trail_atr_mult, 0=none (default 0..4)")
    ap.add_argument("--near", help="comma list of trail_near_mult to tune the adaptive narrowing "
                                   "trail (mean_reversion only); enables a 3-D sweep")
    args = ap.parse_args()

    config = load_config(args.config)
    friction = config.costs.friction_bps
    from fablebot.data import source_from_config
    cache_dir = config.data.cache_dir
    if cache_dir and not Path(cache_dir).is_absolute():
        cache_dir = ROOT / cache_dir
    source = source_from_config(config, cache_dir=cache_dir)

    stop_mults = _grid(args.stops, [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
    trail_mults = _grid(args.trails, [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    near_arg = _grid(args.near, None) if args.near else None
    insts = ([config.instruments[args.instrument]] if args.instrument
             else list(config.instruments.values()))

    for inst in insts:
        try:
            df = source.fetch(inst.symbol, inst.timeframe, days=OPTIMIZER_FETCH_DAYS.get(inst.timeframe, 60))
        except Exception as exc:
            print(f"[sweep] {inst.key}: data fetch failed ({exc}) — skipping", file=sys.stderr)
            continue
        # the narrowing trail is mean-reversion only; ignore --near for other strategies
        near_mults = tuple(near_arg) if (near_arg and inst.strategy == "mean_reversion") else (None,)
        rows = sweep(df, inst, stop_mults, trail_mults, near_mults=near_mults,
                     risk=config.account.risk_per_trade, friction=friction)
        cur = inst.params.get("atr_stop_mult"), inst.params.get("trail_atr_mult", 0.0)
        scored = [r for r in rows if r["oos_sharpe"] is not None]
        best = max(scored, key=lambda r: r["oos_sharpe"]) if scored else None

        print("=" * 72)
        print(f"STOP SWEEP — {inst.key} ({inst.strategy}, {inst.timeframe})   "
              f"current: hard {cur[0]} / trail {cur[1] or 'none'}")
        print("=" * 72)
        if near_mults != (None,):
            if near_arg and inst.strategy != "mean_reversion":
                print("  (--near ignored — adaptive narrowing trail is mean_reversion only)")
            print("\nAdaptive trail — top combos (hard / trail / near → OOS Sharpe):\n")
            for r in sorted(scored, key=lambda r: -r["oos_sharpe"])[:12]:
                print(f"  hard {r['stop']:>4.1f} / trail {r['trail']:>4.1f} / near {r['near']:>4.1f}"
                      f"   ->  {r['oos_sharpe']:>6.2f}  ({r['oos_trades']} OOS trades)")
            if best:
                print(f"\n  best: hard {best['stop']:.1f} / trail {best['trail']:.1f} / "
                      f"near {best['near']:.1f}  ->  OOS Sharpe {best['oos_sharpe']:.2f}")
        else:
            print("\nOOS Sharpe by hard stop (rows) × trailing stop (cols, 0=none):\n")
            print("  hard\\trail " + "".join(f"{t:>7.1f}" for t in trail_mults))
            for s in stop_mults:
                cells = []
                for t in trail_mults:
                    r = next(x for x in rows if x["stop"] == s and x["trail"] == t)
                    cells.append(f"{r['oos_sharpe']:>7.2f}" if r["oos_sharpe"] is not None else "      —")
                mark = " <- current" if abs(s - (cur[0] or -9)) < 1e-9 else ""
                print(f"  {s:>5.1f}     " + "".join(cells) + mark)
            if best:
                print(f"\n  best: hard {best['stop']:.1f} / trail {best['trail'] or 'none'}  "
                      f"-> OOS Sharpe {best['oos_sharpe']:.2f} ({best['oos_trades']} OOS trades)")
        print("\nWalk-forward OOS Sharpe (single 75/25 split) — directional, not gospel; a higher"
              "\nvalue means that stop/trail kept more of the edge. Apply by editing the param in config.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
