#!/usr/bin/env python3
"""Train ML signal models and put them through the hardened gate.

For each instrument: fetch history via the configured data source, train a
gradient-boosted-tree model on the train split, and validate it the same way as
everything else — non-overlapping folds (retrained per fold), reseed stability,
embargoed held-out confirmation, and the deflated-Sharpe correction — against
the instrument's current strategy. Qualified models (their *spec*, so the
engine retrains on current data, never a frozen pickle) are appended to a
proposals log for the human-gated reviewer.

  python3 scripts/train_ml.py
  python3 scripts/train_ml.py --instrument BTC --horizon 5

Evidence only — never edits config.yaml.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fablebot import ml
from fablebot.config import load_config
from fablebot.optimizer import BARS_PER_YEAR, OPTIMIZER_FETCH_DAYS, deflation_inputs, split_search_holdout
from fablebot.search import current_genome_for

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="FableBot ML signal-model trainer")
    parser.add_argument("--config", default=ROOT / "config.yaml")
    parser.add_argument("--instrument", help="only train this instrument key")
    parser.add_argument("--horizon", type=int, default=5, help="forward-return label horizon (bars)")
    parser.add_argument("--n-estimators", type=int, default=60)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--ml-proposals", default=ROOT / "state/ml_proposals.jsonl")
    args = parser.parse_args()

    config = load_config(args.config)
    targets = (
        [config.instruments[args.instrument]] if args.instrument
        else list(config.instruments.values())
    )
    model_kwargs = {"n_estimators": args.n_estimators, "max_depth": args.max_depth}
    proposals_path = Path(args.ml_proposals)

    from fablebot.data import source_from_config
    cache_dir = config.data.cache_dir
    if cache_dir and not Path(cache_dir).is_absolute():
        cache_dir = ROOT / cache_dir
    source = source_from_config(config, cache_dir=cache_dir)

    from fablebot.runlog import record_run
    print(f"[ml] START {datetime.now(timezone.utc).isoformat(timespec='seconds')} — {len(targets)} instrument(s)", flush=True)
    n_qualified = 0

    for inst in targets:
        try:
            days = OPTIMIZER_FETCH_DAYS.get(inst.timeframe, 60)
            df = source.fetch(inst.symbol, inst.timeframe, days=days)
        except Exception as exc:
            print(f"[ml] {inst.key}: data fetch failed ({exc}) — skipping", flush=True)
            continue

        search_df, holdout_df = split_search_holdout(df)
        friction = config.costs.friction_bps
        # quick scan to estimate the multiple-comparison benchmark
        oos_sharpes = []
        for seed in range(6):
            t = ml.ml_walk_forward(search_df, inst, model_kwargs=model_kwargs, seed=seed, friction_bps=friction)
            if t is not None and np.isfinite(t.oos_sharpe):
                oos_sharpes.append(t.oos_sharpe)
        bpy = BARS_PER_YEAR.get(inst.timeframe, 252)
        n_eff, sr_std_perbar = deflation_inputs(oos_sharpes, bpy)

        verdict = ml.validate_ml(
            search_df, holdout_df, inst, current_genome=current_genome_for(inst),
            rng=np.random.default_rng(0), n_trials=n_eff,
            sr_std_perbar=sr_std_perbar, model_kwargs={**model_kwargs, "horizon": args.horizon},
            friction_bps=friction,
        )
        status = "QUALIFIED" if verdict["qualified"] else "rejected"
        print(f"[ml] {inst.key}: [{status}] {verdict['reason']}", flush=True)
        if verdict["qualified"]:
            record = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "instrument": inst.key, "symbol": inst.symbol, "timeframe": inst.timeframe,
                "model": "ml_gbt",
                "model_spec": {"horizon": args.horizon, **model_kwargs},
                "verdict": {k: v for k, v in verdict.items() if k != "reason"},
                "reason": verdict["reason"],
            }
            n_qualified += 1
            proposals_path.parent.mkdir(parents=True, exist_ok=True)
            with proposals_path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
            print(f"  -> appended qualified ML model to {proposals_path}")

    summary = f"{n_qualified}/{len(targets)} instruments qualified an ML model"
    record_run("train_ml", summary)
    print(f"[ml] DONE {datetime.now(timezone.utc).isoformat(timespec='seconds')} — {summary}", flush=True)


if __name__ == "__main__":
    main()
