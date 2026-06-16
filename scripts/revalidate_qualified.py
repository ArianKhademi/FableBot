#!/usr/bin/env python3
"""Re-test the proposals that qualified under the OLD gate against the NEW one.

For every currently-`qualified` proposal it re-runs the full hardened pipeline on
fresh data — the candidate entry gate (raised trade floor, beat-baseline,
deflated Sharpe) and then validate_candidate (non-overlapping purged folds, the
tighter perturbation bar, the embargoed held-out slice) — all with the
configured transaction-cost friction. It reports, per instrument, how many of
the old survivors still qualify and which stage culls the rest.

Caveat: this validates on *current* data. For 4h/1h instruments (deep history)
the window overlaps the original heavily; for 15m it has largely rolled over, so
those are "do these params still hold" rather than "would they have qualified
then". The breakdown is printed per instrument so you can weight accordingly.

  python3 scripts/revalidate_qualified.py                  # report only (read-only)
  python3 scripts/revalidate_qualified.py --sample 50 --workers 8
  python3 scripts/revalidate_qualified.py --apply          # also demote the failures

With --apply, every proposal that fails re-validation is transitioned to
'rejected' in the proposal store (append-only, with a dated reason) — so stale
proposals that qualified under the old gate stop showing as 'qualified' and the
nightly reviewer / shipping slate no longer surface them. Without --apply the
proposal store is never modified.
"""

import argparse
import collections
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fablebot.config import load_config
from fablebot.optimizer import (
    BARS_PER_YEAR, DSR_THRESHOLD, MIN_IMPROVEMENT, MIN_TRADES, MIN_TRIALS_FOR_DSR,
    OPTIMIZER_FETCH_DAYS, BacktestHistory, deflated_sharpe_ratio, equity_returns,
    run_walk_forward, split_search_holdout, validate_candidate,
)

ROOT = Path(__file__).resolve().parent.parent


def _revalidate_one(args):
    """Apply the new candidate gate + validation to one old-qualified proposal.
    Returns the stage it fails at, or 'qualified' if it survives."""
    search_df, holdout_df, inst, proposed, current, baseline_sharpe, n_trials, sr_std_perbar, friction, seed = args
    trial = run_walk_forward(search_df, inst, proposed, 0.01, friction)
    if trial is None or trial.oos_trades < MIN_TRADES or trial.is_trades < MIN_TRADES:
        return "trade_floor"
    if not (trial.oos_sharpe > baseline_sharpe + MIN_IMPROVEMENT):
        return "improvement"
    if n_trials >= MIN_TRIALS_FOR_DSR:
        dsr = deflated_sharpe_ratio(equity_returns(trial.oos_equity_curve), n_trials, sr_std_perbar)
        if dsr < DSR_THRESHOLD:
            return "deflated_sharpe"
    verdict = validate_candidate(
        search_df, inst, proposed, current, np.random.default_rng(seed),
        risk_per_trade=0.01, holdout_df=holdout_df, friction_bps=friction,
    )
    if verdict["qualified"]:
        return "qualified"
    reason = verdict["reason"]
    if reason.startswith("regime"):
        return "folds"
    if reason.startswith("perturbation"):
        return "perturbation"
    if reason.startswith("held-out"):
        return "held_out"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-validate old-qualified proposals under the new gate")
    parser.add_argument("--config", default=ROOT / "config.yaml")
    parser.add_argument("--proposals", default=ROOT / "state/proposals.jsonl")
    parser.add_argument("--history", default=ROOT / "state/backtest_history.jsonl")
    parser.add_argument("--sample", type=int, default=None, help="re-test only N per instrument (sanity/speed)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--write", help="also write the report to this path")
    parser.add_argument(
        "--apply", action="store_true",
        help="demote proposals that fail re-validation to 'rejected' (append-only, "
             "with reason) so the stale old-gate qualifieds stop resurfacing",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    friction = config.costs.friction_bps
    history = BacktestHistory(args.history)
    from fablebot.pipeline import REJECTED, ProposalStore
    store = ProposalStore(args.proposals) if args.apply else None
    today = datetime.now(timezone.utc).date().isoformat()
    demoted = 0

    # latest status + the candidate row (params/instrument) per proposal id
    events = [json.loads(l) for l in Path(args.proposals).read_text().splitlines() if l.strip()]
    latest, cand_rows = {}, {}
    for e in events:
        pid = e["proposal_id"]
        latest.setdefault(pid, {}).update(e)
        if e.get("status") == "candidate":
            cand_rows[pid] = e
    qualified = [cand_rows[pid] for pid, p in latest.items()
                 if p.get("status") == "qualified" and pid in cand_rows]

    by_inst = collections.defaultdict(list)
    for p in qualified:
        by_inst[p["instrument"]].append(p)

    from fablebot.data import get_source
    cache_dir = config.data.cache_dir
    if cache_dir and not Path(cache_dir).is_absolute():
        cache_dir = ROOT / cache_dir
    source = get_source(config.data.source, cache_dir=cache_dir, ttl_minutes=config.data.cache_ttl_minutes)

    overall = collections.Counter()
    per_inst_rows = []
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        for key in sorted(by_inst, key=lambda k: -len(by_inst[k])):
            props = by_inst[key]
            if args.sample:
                props = props[: args.sample]
            inst = config.instruments.get(key)
            if inst is None:
                continue
            try:
                df = source.fetch(inst.symbol, inst.timeframe, days=OPTIMIZER_FETCH_DAYS.get(inst.timeframe, 60))
            except Exception as exc:
                print(f"[revalidate] {key}: data fetch failed ({exc}) — skipping", flush=True)
                continue
            search_df, holdout_df = split_search_holdout(df)
            # baseline (current config) on the current search slice, with friction
            base = run_walk_forward(search_df, inst, dict(inst.params), 0.01, friction)
            baseline_sharpe = base.oos_sharpe if base else float("-inf")
            prior = [t for t in history.read(key, limit=None) if not t.get("is_baseline")]
            n_trials = len(prior)
            sr = np.array([t["oos_sharpe"] for t in prior if np.isfinite(t.get("oos_sharpe", float("nan")))])
            sr_std_perbar = (float(sr.std()) / (BARS_PER_YEAR.get(inst.timeframe, 252) ** 0.5)) if sr.size else 0.0

            tasks = [
                (search_df, holdout_df, inst, p["proposed_params"], p["current_params"],
                 baseline_sharpe, n_trials, sr_std_perbar, friction, abs(hash(p["proposal_id"])) % (2**31))
                for p in props
            ]
            results = list(pool.map(_revalidate_one, tasks)) if pool else [_revalidate_one(t) for t in tasks]
            stages = collections.Counter(results)
            overall.update(stages)
            survived = stages.get("qualified", 0)
            per_inst_rows.append((key, len(props), survived, stages))
            print(f"[revalidate] {key}: {survived}/{len(props)} still qualify "
                  f"({survived/len(props)*100:.0f}%) — culls: "
                  f"{dict((k,v) for k,v in stages.items() if k!='qualified')}", flush=True)
            if store is not None:
                for p, stage in zip(props, results):
                    if stage != "qualified":
                        store.set_status(
                            p["proposal_id"], REJECTED,
                            reason=f"re-validation under hardened gate ({today}): failed at {stage}",
                        )
                        demoted += 1
    finally:
        if pool is not None:
            pool.shutdown()

    total = sum(n for _, n, _, _ in per_inst_rows)
    surv = sum(s for _, _, s, _ in per_inst_rows)
    lines = ["# Re-validation of old-qualified proposals under the new gate", "",
             f"Friction: {friction} bps. Tested {total} proposals that qualified under the old gate.", ""]
    lines.append(f"**{surv}/{total} ({surv/total*100:.1f}%) still qualify under the new gate.**" if total else "no proposals")
    lines.append("")
    lines.append("| Instrument | old-qualified | survive new | survival % | top cull |")
    lines.append("|---|---|---|---|---|")
    for key, n, s, stages in per_inst_rows:
        culls = {k: v for k, v in stages.items() if k != "qualified"}
        top = max(culls.items(), key=lambda kv: kv[1])[0] if culls else "—"
        lines.append(f"| {key} | {n} | {s} | {s/n*100:.0f}% | {top} |")
    lines += ["", "Cull stage totals (where the old survivors now fail):"]
    for stage, c in overall.most_common():
        if stage != "qualified":
            lines.append(f"- {stage}: {c}")
    if store is not None:
        lines += ["", f"Applied: demoted {demoted} proposals to 'rejected' in {args.proposals}."]
    report = "\n".join(lines)
    print("\n" + report)
    if args.write:
        out = Path(args.write)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n")
        print(f"\n[written to {out}]", file=sys.stderr)


if __name__ == "__main__":
    main()
