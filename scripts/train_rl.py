#!/usr/bin/env python3
"""Train the RL positioning agent and put it through the hardened gate.

For each instrument: fetch history via the configured data source, learn a
Q-learning policy on the train split, and validate it the same way as every
other layer — non-overlapping folds (policy relearned per fold), reseed
stability, embargoed held-out confirmation, and the deflated-Sharpe correction —
against the instrument's current strategy. Qualified agents (their *spec*, so
the engine relearns on current data, never a frozen table) are appended to a
proposals log for the human-gated reviewer.

  python3 scripts/train_rl.py
  python3 scripts/train_rl.py --instrument BTC --epochs 30

The RL agent only ever proposes direction/timing — the risk engine still owns
sizing and stops. Evidence only — never edits config.yaml.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fablebot import rl
from fablebot.config import load_config
from fablebot.optimizer import BARS_PER_YEAR, OPTIMIZER_FETCH_DAYS, split_search_holdout
from fablebot.search import current_genome_for

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="FableBot RL agent trainer")
    parser.add_argument("--config", default=ROOT / "config.yaml")
    parser.add_argument("--instrument", help="only train this instrument key")
    parser.add_argument("--epochs", type=int, default=20, help="Q-learning epochs")
    parser.add_argument("--rl-proposals", default=ROOT / "state/rl_proposals.jsonl")
    args = parser.parse_args()

    config = load_config(args.config)
    targets = (
        [config.instruments[args.instrument]] if args.instrument
        else list(config.instruments.values())
    )
    agent_kwargs = {"epochs": args.epochs}
    proposals_path = Path(args.rl_proposals)

    from fablebot.data import source_from_config
    cache_dir = config.data.cache_dir
    if cache_dir and not Path(cache_dir).is_absolute():
        cache_dir = ROOT / cache_dir
    source = source_from_config(config, cache_dir=cache_dir)

    for inst in targets:
        try:
            days = OPTIMIZER_FETCH_DAYS.get(inst.timeframe, 60)
            df = source.fetch(inst.symbol, inst.timeframe, days=days)
        except Exception as exc:
            print(f"[rl] {inst.key}: data fetch failed ({exc}) — skipping", flush=True)
            continue

        search_df, holdout_df = split_search_holdout(df)
        friction = config.costs.friction_bps
        oos_sharpes = []
        for seed in range(6):
            t = rl.rl_walk_forward(search_df, inst, agent_kwargs=agent_kwargs, seed=seed, friction_bps=friction)
            if t is not None and np.isfinite(t.oos_sharpe):
                oos_sharpes.append(t.oos_sharpe)
        bpy = BARS_PER_YEAR.get(inst.timeframe, 252)
        sr_std_perbar = (float(np.std(oos_sharpes)) / (bpy ** 0.5)) if oos_sharpes else 0.0

        verdict = rl.validate_rl(
            search_df, holdout_df, inst, current_genome=current_genome_for(inst),
            rng=np.random.default_rng(0), n_trials=len(oos_sharpes) * 50,
            sr_std_perbar=sr_std_perbar, agent_kwargs=agent_kwargs, friction_bps=friction,
        )
        status = "QUALIFIED" if verdict["qualified"] else "rejected"
        print(f"[rl] {inst.key}: [{status}] {verdict['reason']}", flush=True)
        if verdict["qualified"]:
            record = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "instrument": inst.key, "symbol": inst.symbol, "timeframe": inst.timeframe,
                "model": "rl_qlearn",
                "agent_spec": agent_kwargs,
                "verdict": {k: v for k, v in verdict.items() if k != "reason"},
                "reason": verdict["reason"],
            }
            proposals_path.parent.mkdir(parents=True, exist_ok=True)
            with proposals_path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
            print(f"  -> appended qualified RL agent to {proposals_path}")


if __name__ == "__main__":
    main()
