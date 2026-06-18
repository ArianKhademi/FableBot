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
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fablebot import search
from fablebot.config import load_config
from fablebot.data import source_from_config
from fablebot.optimizer import (
    BARS_PER_YEAR, BacktestHistory, ContinuousOptimizer, deflation_inputs, split_search_holdout,
)
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
    parser.add_argument("--validate-every", type=int, default=25,
                        help="every N cycles, run the best genome per instrument through the full "
                             "gate and write qualified ones to genome_proposals.jsonl")
    parser.add_argument("--genome-proposals", default=ROOT / "state/genome_proposals.jsonl")
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
    prev_best: dict[str, float] = {}  # instrument -> best genome fitness last cycle
    genome_proposals = Path(args.genome_proposals)

    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    cycle = 0
    try:
        while args.cycles == 0 or cycle < args.cycles:
            cycle += 1
            trials_run = 0
            best_line = []
            gates_run = cycle % args.validate_every == 0
            genomes_qualified = 0
            for inst in instruments:
                # 1) numeric optimizer: parameter trials + drain one validation
                optimizer.validate_one()
                trials_run += len(optimizer.trial_batch(inst, n=args.trials_per_cycle, pool=pool))

                # 2) genome search: evolve, persisted across restarts
                try:
                    df = source.fetch(inst.symbol, inst.timeframe)
                except Exception as exc:
                    print(f"[engine] {inst.key}: data fetch failed ({exc})", flush=True)
                    continue
                search_df, holdout_df = split_search_holdout(df)
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

                best = ga.best(1)
                if best:
                    fit = best[0][1]
                    delta = fit - prev_best.get(inst.key, fit)
                    best_line.append(f"{inst.key} {fit:.2f}{'↑' if delta > 1e-9 else ''}")
                    prev_best[inst.key] = fit

                # 3) periodically: put the best genome through the FULL gate and
                #    surface qualified ones so the engine actually proposes them.
                if gates_run and best:
                    genomes_qualified += _gate_best_genome(
                        ga, inst, search_df, holdout_df, friction, rng, genome_proposals
                    )

            n_cand = len(proposals.by_status(CANDIDATE))
            n_qual = len(proposals.by_status(QUALIFIED))
            gate_note = f", {genomes_qualified} genome(s) qualified→{genome_proposals.name}" if gates_run else ""
            print(
                f"cycle {cycle} — {trials_run} trials | genome best: {', '.join(best_line)} | "
                f"pipeline: {n_cand} candidates, {n_qual} qualified{gate_note}",
                flush=True,
            )
            if args.sleep:
                time.sleep(args.sleep)
    except KeyboardInterrupt:
        print("\n[engine] stopped.")
    finally:
        if pool is not None:
            pool.shutdown()


def _gate_best_genome(ga, inst, search_df, holdout_df, friction, rng, proposals_path) -> int:
    """Validate the instrument's best evolved genome through the full hardened
    gate; append it to the genome-proposals log if it qualifies. Returns 1/0."""
    best_genome, _ = ga.best(1)[0]
    current = search.current_genome_for(inst)
    fitnesses = [f for _, f in ga.history]
    n_eff, sr_std_perbar = deflation_inputs(fitnesses, BARS_PER_YEAR.get(inst.timeframe, 252))
    verdict = search.validate_genome(
        search_df, holdout_df, inst.symbol, inst.timeframe,
        candidate=best_genome, current=current, rng=rng,
        n_trials=n_eff, sr_std_perbar=sr_std_perbar, friction_bps=friction,
    )
    if not verdict["qualified"]:
        return 0
    proposals_path.parent.mkdir(parents=True, exist_ok=True)
    with proposals_path.open("a") as fh:
        fh.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "instrument": inst.key, "symbol": inst.symbol, "timeframe": inst.timeframe,
            "genome": best_genome, "genome_id": search.dsl.genome_id(best_genome),
            "verdict": {k: v for k, v in verdict.items() if k != "reason"},
            "reason": verdict["reason"],
        }) + "\n")
    return 1


if __name__ == "__main__":
    main()
