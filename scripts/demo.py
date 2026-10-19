#!/usr/bin/env python3
"""Deterministic offline showcase. Reads no config, market data, or account state."""

from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fablebot.backtest import run_backtest
from fablebot.dsl import seed_mean_reversion
from fablebot.strategies import build_strategy

STARTING_EQUITY = 100_000.0
FRICTION_BPS = 3.0


def synthetic_bars(seed: int = 7, bars: int = 500) -> pd.DataFrame:
    """Invented daily prices with oscillation, drift and seeded noise."""
    if bars < 100:
        raise ValueError("the demo needs at least 100 bars")
    rng = np.random.default_rng(seed)
    t = np.arange(bars)
    closes = 100 * np.exp(0.0002 * t + 0.06 * np.sin(t / 12) + rng.normal(0, 0.008, bars))
    opens = np.concatenate(([closes[0]], closes[:-1]))
    spread = rng.uniform(0.15, 0.65, bars)
    return pd.DataFrame({
        "open": opens,
        "high": np.maximum(opens, closes) + spread,
        "low": np.minimum(opens, closes) - spread,
        "close": closes,
        "volume": rng.integers(1000, 5000, bars),
    }, index=pd.date_range("2024-01-01", periods=bars, freq="D", name="date"))


def evaluate(seed: int = 7) -> tuple[dict, str]:
    df = synthetic_bars(seed)
    csv = df.to_csv(float_format="%.8f", date_format="%Y-%m-%d")
    strategies = [(name, build_strategy(name, {})) for name in
                  ("mean_reversion", "momentum_breakout", "trend_following")]
    strategies.append(("DSL mean reversion", build_strategy("genome", {}, genome=seed_mean_reversion())))
    results = []
    for label, strategy in strategies:
        result = run_backtest(df, strategy, starting_equity=STARTING_EQUITY,
                              risk_per_trade=0.01, instrument="SYNTH",
                              friction_bps=FRICTION_BPS)
        results.append({
            "strategy": label, "trades": result.trades, "wins": result.wins,
            "total_pnl": round(result.total_pnl, 2),
            "return_pct": round(result.return_pct, 4),
            "max_drawdown_pct": round(result.max_drawdown_pct, 4),
            "equity_curve": [round(value, 2) for value in result.equity_curve],
        })
    return {
        "data_kind": "synthetic", "seed": seed, "bars": len(df),
        "data_sha256": hashlib.sha256(csv.encode()).hexdigest(),
        "starting_equity": STARTING_EQUITY, "risk_per_trade": 0.01,
        "friction_bps": FRICTION_BPS,
        "notice": "Invented prices: a software demonstration, not evidence of market performance.",
        "results": results,
    }, csv


def render_html(summary: dict) -> str:
    """Embedded SVG charts keep the saved dashboard usable without a CDN."""
    cards = []
    for result in summary["results"]:
        curve = result["equity_curve"]
        lo, hi = min(curve), max(curve)
        span = hi - lo or 1.0
        points = " ".join(f"{20 + i * 560 / max(1, len(curve) - 1):.2f},"
                          f"{170 - (value - lo) * 140 / span:.2f}" for i, value in enumerate(curve))
        cards.append(f'''<article><h2>{escape(result['strategy'])}</h2>
<p class="metrics">{result['trades']} trades · {result['return_pct']:+.2f}% return ·
{result['max_drawdown_pct']:.2f}% maximum drawdown</p>
<svg viewBox="0 0 600 200" role="img" aria-label="Synthetic mark-to-market equity curve">
<text x="20" y="18">${hi:,.0f}</text><text x="20" y="194">${lo:,.0f}</text>
<polyline points="{points}" fill="none" stroke="#70d7bf" stroke-width="2"/></svg>
<p>Final simulated P&amp;L: <strong>${result['total_pnl']:+,.2f}</strong></p></article>''')
    return f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FableBot · Synthetic demo</title>
<style>
body{{margin:0;background:#111b29;color:#e7edf4;font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1100px;margin:auto;padding:40px 24px}}h1{{font-size:40px;margin:0}}
h2{{font-size:20px}}.label{{color:#70d7bf;letter-spacing:.12em;font-size:13px}}
.notice{{background:#26344a;padding:16px;border-left:4px solid #70d7bf}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}}
article{{border:1px solid #39475b;border-radius:12px;padding:20px}}svg{{width:100%}}
svg text{{fill:#aab7c9;font-size:12px}}.metrics,footer{{color:#aab7c9}}code{{overflow-wrap:anywhere}}
</style></head><body><main><p class="label">OFFLINE · SYNTHETIC DATA · PAPER ONLY</p>
<h1>FableBot</h1><p>Three named strategies and a composable strategy DSL, replayed through the same backtester.</p>
<p class="notice">{escape(summary['notice'])}</p>
<p>{summary['bars']} invented daily bars · seed {summary['seed']} · $100,000 per independent simulation ·
1% sizing budget · {summary['friction_bps']:g} bps adverse fill friction.</p>
<div class="grid">{''.join(cards)}</div>
<footer><p>These are separate single-instrument runs, not a combined portfolio. Gap fills can exceed the sizing budget.
Each chart has its own vertical scale. Curves show the existing backtest's per-bar marks; final P&amp;L also includes final liquidation friction.</p>
<p>No market data, credentials, network assets, or account files are used. This demo does not train models or search for profitable strategies.</p>
<p>Generated input SHA-256: <code>{summary['data_sha256']}</code></p></footer>
</main></body></html>
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("demo-output"),
                        help="new directory to create; existing paths are never overwritten")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.output.exists():
        parser.error("--output already exists; choose a new directory")
    summary, csv = evaluate(args.seed)
    # mkdir is also the race-safe overwrite guard. No config/state path is read.
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "synthetic.csv").write_text(csv, encoding="utf-8")
    (args.output / "dashboard.html").write_text(render_html(summary), encoding="utf-8")
    print(f"Synthetic demo written to {args.output / 'dashboard.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
