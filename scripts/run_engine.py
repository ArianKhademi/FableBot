#!/usr/bin/env python3
"""Unified self-improving engine: one orchestrator that runs both searches.

Each cycle, for every instrument, it:
  1. runs a batch of random parameter trials through the hardened numeric
     optimizer (candidates -> validation -> qualified), and
  2. advances the evolutionary genome search one generation (persisted, so it
     keeps learning across runs).

It interleaves the two so the parameter search and the structure search both
make progress, then prints the pipeline view. Designed for cron or a long
foreground run; --cycles bounds it (0 = forever).

  python3 scripts/run_engine.py --cycles 1 --gens-per-cycle 1
  python3 scripts/run_engine.py --cycles 0 --workers 8          # run forever

Evidence only — never edits config.yaml.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fablebot import search
from fablebot.config import load_config
from fablebot.data import source_from_config
from fablebot.optimizer import BacktestHistory, ContinuousOptimizer, split_search_holdout
from fablebot.pipeline import CANDIDATE, ProposalStore, QUALIFIED

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="FableBot unified engine")
    parser.add_argument("--config", default=ROOT / "config.yaml")
    parser.add_argument("--cycles", type=int, default=1, help="0 = run forever")
    parser.add_argument("--trials-per-cycle", type=int, default=8)
    parser.add_argument("--gens-per-cycle", type=int, default=1)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    cache_dir = config.data.cache_dir
    if cache_dir and not Path(cache_dir).is_absolute():
        cache_dir = ROOT / cache_dir
    source = source_from_config(config, cache_dir=cache_dir)
    friction = config.costs.friction_bps

    history = BacktestHistory(ROOT / "state/backtest_history.jsonl")
    proposals = ProposalStore(ROOT / "state/proposals.jsonl")
    optimizer = ContinuousOptimizer(
        config, history, proposals,
        fetch_fn=lambda s, tf, d: source.fetch(s, tf, d),
        seed=args.seed, friction_bps=friction,
    )

    instruments = [i for i in config.instruments.values() if i.param_bounds]
    rng = np.random.default_rng(args.seed)
    populations = {}  # instrument key -> GeneticSearch

    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    cycle = 0
    try:
        while args.cycles == 0 or cycle < args.cycles:
            cycle += 1
            print(f"\n=== engine cycle {cycle} ===", flush=True)
            for inst in instruments:
                # 1) numeric optimizer: a batch of parameter trials + drain one validation
                optimizer.validate_one()
                optimizer.trial_batch(inst, n=args.trials_per_cycle, pool=pool)

                # 2) genome search: one generation, persisted
                try:
                    df = source.fetch(inst.symbol, inst.timeframe)
                except Exception as exc:
                    print(f"[engine] {inst.key}: data fetch failed ({exc})", flush=True)
                    continue
                search_df, _ = split_search_holdout(df)
                pop_path = ROOT / "state" / f"population_{inst.key}.json"
                ga = populations.get(inst.key) or search.resume_or_new(
                    pop_path, rng, population_size=args.population,
                    seeds=[search.current_genome_for(inst)] + search.dsl.seed_genomes(),
                )
                populations[inst.key] = ga
                for _ in range(args.gens_per_cycle):
                    g = ga.ask()
                    while g is not None:
                        t = search.evaluate_genome(search_df, inst.symbol, inst.timeframe, g, friction_bps=friction)
                        ga.tell(g, search.genome_fitness(t.oos_sharpe, t.oos_trades) if t else -1e9)
                        g = ga.ask()
                    ga.evolve()
                ga.save(pop_path)

            n_cand = len(proposals.by_status(CANDIDATE))
            n_qual = len(proposals.by_status(QUALIFIED))
            print(f"[engine] cycle {cycle} done — pipeline: {n_cand} candidates, {n_qual} qualified", flush=True)
            if args.sleep:
                time.sleep(args.sleep)
    except KeyboardInterrupt:
        print("\n[engine] stopped.")
    finally:
        if pool is not None:
            pool.shutdown()


if __name__ == "__main__":
    main()
