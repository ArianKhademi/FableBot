#!/usr/bin/env python3
"""Evolutionary genome search — discover *new strategies*, not just new numbers.

Runs a genetic search over the strategy DSL for each instrument: it breeds a
population of genomes (seeded from the hand-written strategies and any
persisted survivors), scores each by walk-forward OOS Sharpe on the search
slice, and keeps the population on disk so a restart resumes learning. The best
evolved genome for each symbol is then put through the *same hardened gate* the
numeric optimizer uses (non-overlapping folds + perturbation + embargoed
held-out + deflated Sharpe); survivors are appended to a genome-proposals log.

  python3 scripts/evolve.py --generations 10
  python3 scripts/evolve.py --instrument BTC --generations 20 --workers 8

Like the optimizer, this only produces evidence — it never edits config.yaml.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fablebot import search
from fablebot.config import load_config
from fablebot.optimizer import BARS_PER_YEAR, split_search_holdout

ROOT = Path(__file__).resolve().parent.parent


def evaluate_generation(ga, df, symbol, timeframe, risk, friction, pool, oos_sharpes):
    pending = []
    g = ga.ask()
    while g is not None:
        pending.append(g)
        g = ga.ask()
    args = [(df, symbol, timeframe, g, risk, friction) for g in pending]
    results = list(pool.map(search._eval_worker, args)) if pool else [search._eval_worker(a) for a in args]
    for genome, (sharpe, trades) in zip(pending, results):
        ga.tell(genome, search.genome_fitness(sharpe, trades))
        if np.isfinite(sharpe):
            oos_sharpes.append(sharpe)


def evolve_instrument(inst, df, args, pool, proposals_path):
    risk = 0.01
    friction = getattr(args, "friction_bps", 0.0)
    search_df, holdout_df = split_search_holdout(df)
    rng = np.random.default_rng(args.seed)
    pop_path = ROOT / "state" / f"population_{inst.key}.json"
    current = search.current_genome_for(inst)
    ga = search.resume_or_new(
        pop_path, rng, population_size=args.population,
        seeds=[current] + search.dsl.seed_genomes(),
    )

    oos_sharpes: list[float] = []
    for gen in range(args.generations):
        evaluate_generation(ga, search_df, inst.symbol, inst.timeframe, risk, friction, pool, oos_sharpes)
        ga.evolve()
        ga.save(pop_path)
    evaluate_generation(ga, search_df, inst.symbol, inst.timeframe, risk, friction, pool, oos_sharpes)
    ga.save(pop_path)

    leaders = ga.best(args.top)
    print(f"\n=== {inst.key} ({inst.symbol}) — best genomes after gen {ga.generation} ===")
    for genome, fit in leaders:
        print(f"  fitness {fit:+.3f}  {genome['name']:18s} id={search.dsl.genome_id(genome)}")

    # validate the single best genome through the hardened gate
    if not leaders:
        return
    best_genome, _ = leaders[0]
    n_trials = len(oos_sharpes)
    bpy = BARS_PER_YEAR.get(inst.timeframe, 252)
    sr_std_perbar = float(np.std(oos_sharpes)) / (bpy ** 0.5) if oos_sharpes else 0.0
    verdict = search.validate_genome(
        search_df, holdout_df, inst.symbol, inst.timeframe,
        candidate=best_genome, current=current, rng=rng,
        n_trials=n_trials, sr_std_perbar=sr_std_perbar, risk_per_trade=risk,
        friction_bps=friction,
    )
    status = "QUALIFIED" if verdict["qualified"] else "rejected"
    print(f"  [{status}] best genome vs current — {verdict['reason']}")
    if verdict["qualified"]:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "instrument": inst.key, "symbol": inst.symbol, "timeframe": inst.timeframe,
            "genome": best_genome, "genome_id": search.dsl.genome_id(best_genome),
            "verdict": {k: v for k, v in verdict.items() if k != "reason"},
            "reason": verdict["reason"],
        }
        proposals_path.parent.mkdir(parents=True, exist_ok=True)
        with proposals_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(f"  -> appended qualified genome to {proposals_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FableBot evolutionary genome search")
    parser.add_argument("--config", default=ROOT / "config.yaml")
    parser.add_argument("--instrument", help="only evolve this instrument key")
    parser.add_argument("--universe", action="store_true",
                        help="evolve over the config 'universe' scan list, not the traded instruments")
    parser.add_argument("--screen-top", type=int, default=None,
                        help="with --universe, screen to the N most liquid symbols first")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--top", type=int, default=5, help="leaderboard size")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--genome-proposals", default=ROOT / "state/genome_proposals.jsonl")
    args = parser.parse_args()

    config = load_config(args.config)
    args.friction_bps = config.costs.friction_bps
    proposals_path = Path(args.genome_proposals)

    from fablebot.config import universe_instruments
    from fablebot.data import source_from_config
    from fablebot.optimizer import OPTIMIZER_FETCH_DAYS
    from fablebot.universe import screen_universe

    cache_dir = config.data.cache_dir
    if cache_dir and not Path(cache_dir).is_absolute():
        cache_dir = ROOT / cache_dir
    source = source_from_config(config, cache_dir=cache_dir)

    if args.universe:
        targets = universe_instruments(config)
        if args.screen_top:
            targets = screen_universe(targets, source, top_n=args.screen_top)
            print(f"[evolve] screened universe to {len(targets)} symbols by liquidity")
    elif args.instrument:
        targets = [config.instruments[args.instrument]]
    else:
        targets = list(config.instruments.values())

    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        for inst in targets:
            try:
                days = OPTIMIZER_FETCH_DAYS.get(inst.timeframe, 60)
                df = source.fetch(inst.symbol, inst.timeframe, days=days)
            except Exception as exc:
                print(f"[evolve] {inst.key}: data fetch failed ({exc}) — skipping", flush=True)
                continue
            evolve_instrument(inst, df, args, pool, proposals_path)
    finally:
        if pool is not None:
            pool.shutdown()


if __name__ == "__main__":
    main()
