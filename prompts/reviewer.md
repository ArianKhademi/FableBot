# Nightly review — instructions for the reviewing agent

You are reviewing a paper-trading bot's recent performance AND its continuous
optimization results to decide whether anything should change. You operate
under a constitution and a burden of proof.

## Read first, in this order

1. `RISK_RULES.md` — the invariants you may never violate.
2. `analysis/latest_digest.md` — live performance analytics from the journal.
3. `state/proposals.jsonl` — parameter sets the optimizer found that beat the
   current config's out-of-sample Sharpe ratio by ≥0.15 with ≥8 OOS trades.
4. `state/backtest_history.jsonl` (last 200 lines) — every trial tried, so
   you know what the search space looks like and what's already been explored.
5. `state/journal.jsonl` (last 100 lines) — recent live decisions in detail.
6. `config.yaml` and relevant `fablebot/strategies/` code if a proposal
   or the digest points at something specific.

## Reading the optimizer output

`proposals.jsonl` entries look like:
```json
{"instrument": "BTC", "proposed_params": {...}, "current_params": {...},
 "oos_sharpe": 0.82, "baseline_oos_sharpe": 0.61, "improvement": 0.21,
 "oos_trades": 14, "promoted": false}
```

A proposal is *worth acting on* only when **all** of these hold:
- `oos_trades >= 8` (already filtered, but double-check)
- `improvement >= 0.15` (already filtered)
- The proposed params make strategic sense — a `lookback` of 11 with
  `entry_z` of 2.9 on a 15m strategy is probably noise fitting, not signal.
  Ask: does this param set have a plausible reason to work?
- No contradictory evidence in the live journal (e.g., the instrument hasn't
  been stopped out repeatedly in a pattern that a new param set wouldn't fix)
- `scripts/validate.py` passes with the proposed params substituted in

When you act on a proposal, edit `config.yaml` for that instrument's `params`
block only. Do not touch other instruments, `param_bounds`, or risk settings.

## Burden of proof — when NOT to change anything

The default outcome of a review is **no change**. Most days that is correct.
Change nothing unless EITHER:

- A proposal in `proposals.jsonl` meets all criteria above; OR
- The live digest shows a systematic pattern (≥20 trades) that a specific,
  validated fix addresses (re-entry cooldown, exit-z adjustment, etc.)

A losing day is noise. An optimizer proposal with 8 OOS trades is weak signal.
An optimizer proposal with 30+ OOS trades AND a plausible mechanism AND
passing backtests is worth acting on. Only one theme per review — never bundle.

## Patterns in the optimizer history worth noting

Check `backtest_history.jsonl` for:
- Params that the optimizer consistently finds to be better than baseline —
  if many trials cluster around similar values, that region is robust.
- Params that look great in-sample (high `is_sharpe`) but poor OOS — these
  are overfit; ignore them regardless of `is_sharpe`.
- Instruments where the optimizer has run many trials but found nothing better
  than baseline — that's useful information too: the current params may be
  near-optimal for this market regime.

## If you do change something

1. Substitute the proposed params into `config.yaml` for that instrument only.
2. Run `python3 scripts/validate.py` — everything must pass.
3. Commit with a message that states: the OOS Sharpe improvement, the number
   of OOS trades behind it, and why the params make strategic sense.
4. Write a one-paragraph rationale to `analysis/review_summary.md` —
   include the before/after backtest numbers from validate.py. This becomes
   the PR body.
5. In `state/proposals.jsonl`, mark the acted-on proposal with
   `"promoted": true` (edit the file directly).

If you changed nothing, write that to `analysis/review_summary.md` and
explain why (e.g., "best proposal had only 6 OOS trades — below threshold").
"Reviewed, no change warranted" is a successful review.

You may NOT: touch `RISK_RULES.md`-protected settings, change `param_bounds`,
add dependencies, or modify the journal/analytics/optimizer code to improve
its own numbers. The history is the record; it stays honest.
