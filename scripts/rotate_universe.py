#!/usr/bin/env python3
"""Scan the universe and propose which instruments the live book should trade.

A daily scan: screen the universe by liquidity, score each candidate (plus the
current book) by the best gated edge found on it, read which instruments have a
live open position (sticky), then propose a rotation under the guardrails in
fablebot.rotation. Writes a recommendation — it never edits config.yaml or the
live book itself; the reviewer (or you) applies the add/drop by editing config.

  python3 scripts/rotate_universe.py
  python3 scripts/rotate_universe.py --max-active 6 --screen-top 12 --workers 8

Cron (e.g. daily, before the nightly review):
  0 21 * * * cd /path/to/FableBot && python3 scripts/rotate_universe.py \
      --write analysis/rotation.md >> logs/rotation.log 2>&1
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fablebot import rotation
from fablebot.config import load_config, universe_instruments
from fablebot.optimizer import OPTIMIZER_FETCH_DAYS

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Propose a live-book rotation from the universe")
    parser.add_argument("--config", default=ROOT / "config.yaml")
    parser.add_argument("--portfolio", default=ROOT / "state/portfolio.json")
    parser.add_argument("--max-active", type=int, default=6, help="cap on concurrently traded instruments")
    parser.add_argument("--min-score", type=float, default=0.3, help="min OOS Sharpe to add a newcomer")
    parser.add_argument("--churn-margin", type=float, default=0.3,
                        help="a challenger must beat a current holding by this to replace it")
    parser.add_argument("--screen-top", type=int, default=None, help="screen the universe to N most liquid first")
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--write", help="also write the recommendation to this path")
    parser.add_argument("--proposals", default=ROOT / "state/rotation_proposals.jsonl")
    args = parser.parse_args()

    config = load_config(args.config)
    friction = config.costs.friction_bps

    from fablebot.data import source_from_config
    from fablebot.universe import screen_universe
    cache_dir = config.data.cache_dir
    if cache_dir and not Path(cache_dir).is_absolute():
        cache_dir = ROOT / cache_dir
    source = source_from_config(config, cache_dir=cache_dir)

    current = list(config.instruments.keys())
    candidates = list(config.instruments.values()) + universe_instruments(config)
    # de-dup by key (current book may also appear in the universe)
    seen, uniq = set(), []
    for inst in candidates:
        if inst.key not in seen:
            seen.add(inst.key)
            uniq.append(inst)
    candidates = uniq
    if args.screen_top:
        screened = {i.key for i in screen_universe(candidates, source, top_n=args.screen_top)}
        candidates = [i for i in candidates if i.key in screened or i.key in current]
        print(f"[rotate] screened to {len(candidates)} candidates by liquidity", flush=True)

    # fetch (network, main) then score (CPU, workers)
    tasks, keys = [], []
    for inst in candidates:
        try:
            df = source.fetch(inst.symbol, inst.timeframe, days=OPTIMIZER_FETCH_DAYS.get(inst.timeframe, 60))
        except Exception as exc:
            print(f"[rotate] {inst.key}: data fetch failed ({exc}) — skipping", flush=True)
            continue
        tasks.append((inst.symbol, inst.timeframe, df, friction, args.min_trades))
        keys.append(inst.key)

    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        results = list(pool.map(rotation._score_worker, tasks)) if pool else [rotation._score_worker(t) for t in tasks]
    finally:
        if pool is not None:
            pool.shutdown()
    scores = {k: s for k, s in zip(keys, results)}

    open_keys = rotation.read_open_positions(args.portfolio)
    perf = rotation.performance_tilt(rotation.read_trades(args.portfolio))
    plan = rotation.propose_rotation(
        current, scores, open_keys, args.max_active, args.min_score, args.churn_margin,
        performance=perf,
    )

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    lines = ["# Universe rotation recommendation", "",
             f"max_active={args.max_active}, min_score={args.min_score}, churn_margin={args.churn_margin}, "
             f"friction={friction}bps. Open positions are sticky (never dropped to rotate). "
             "Scores are backtest tradeability; the live-P&L tilt (±R, in score units) adjusts "
             "the ranking by what the bot has actually earned.", "",
             "## Scores (best gated seed-strategy OOS Sharpe per symbol)"]
    for k, s in ranked:
        tag = " [open]" if k in open_keys else (" [current]" if k in current else "")
        tilt = f" · live tilt {perf[k]:+.2f}" if k in perf else ""
        lines.append(f"- {k}: {s:.3f}{tilt}{tag}" if s > float("-inf")
                     else f"- {k}: — (no qualifying edge){tilt}{tag}")
    lines += [
        "", "## Proposed book change (human-gated — edit config.yaml to apply)",
        f"- **add**: {plan['add'] or 'none'}",
        f"- **drop**: {plan['drop'] or 'none'}",
        f"- **keep**: {plan['keep'] or 'none'}",
        f"- kept sticky (open position): {plan['kept_sticky'] or 'none'}",
    ]
    report = "\n".join(lines)
    print("\n" + report)

    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scores": scores, "open_positions": sorted(open_keys), **plan,
    }
    proposals_path = Path(args.proposals)
    proposals_path.parent.mkdir(parents=True, exist_ok=True)
    with proposals_path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")

    from fablebot.runlog import record_run
    record_run("rotate_universe", f"add={plan['add'] or 'none'} drop={plan['drop'] or 'none'}")
    if args.write:
        out = Path(args.write)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n")
        print(f"\n[written to {out}]", file=sys.stderr)


if __name__ == "__main__":
    main()
