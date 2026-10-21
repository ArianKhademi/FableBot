# Validation behavior and limitations

This document describes the implementation, not a claim that its methods prove
an edge. `qualified` is an internal proposal status awaiting human review.

## Existing checks

The numeric optimizer ordinarily reserves a 15% tail, with a 2% gap before it,
then uses a 75/25 split inside the search portion. Candidates need at least 20
trades in both parts and an evaluation Sharpe improvement of 0.15. Later checks
compare time blocks, perturb parameters, and compare against the baseline on the
reserved tail. Evolved genomes and learned models have related validators; they
do not have identical entry criteria.

The numeric DSR gate is skipped before ten logged trials. Its effective count is
estimated from dimensionality (60 trials per parameter, a 30 minimum ceiling,
and a 400 maximum ceiling), bounded by the observed count. Sharpe dispersion is
winsorized. This is a practical heuristic, not an estimate derived from the
correlations between full trial return streams.

The ML/RL command-line trainers estimate dispersion from at most six seeds.
They therefore do not reach the validators' ten-effective-trial DSR threshold.
This preparation change preserves that behavior and makes it visible; it does
not invent additional independent evidence by increasing a loop count.

## Reading a verdict

For ML, RL, and genome validators:

- `deflated_sharpe_checked` says whether DSR was actually computed.
- `deflated_sharpe` retains the legacy numeric value for compatibility. When the
  check was skipped, its historical `1.0` sentinel is **not a probability or a
  successful statistical test**. Always read the flag first.
- `holdout_checked` says whether a holdout was supplied. Insufficient/failed
  evaluations can still cause rejection even when it is true.
- `limitations` and `reason` explicitly report skipped checks.

The numeric stage-two validator already returns `holdout_checked`; its reason
now also identifies missing confirmation. Existing qualification decisions,
thresholds, strategy parameters, and learned-model training remain unchanged.
Old stored verdicts lack the new flags. Do not infer that an absent flag means
true, or that historical `1.0` values establish a completed check.

## Remaining research limitations

- For short histories, `split_search_holdout` returns no holdout. Validators
  retain the legacy ability to qualify from their other checks in that case.
- Multiple candidates can be checked against the same tail. Reusing a holdout
  for repeated decisions reduces its independence; a fixed embargo does not
  remove all forms of selection bias.
- Time blocks are contiguous windows, not proof of independent economic regimes.
  The fixed fractional gaps are not an event/label-overlap-aware purge.
- Numeric and genome perturbation changes parameters. ML/RL stability checks
  reseed training. These are different tests, not interchangeable guarantees.
- Annualization uses a shared timeframe table, including equity-market bar
  counts for instruments that may trade around the clock. Cross-asset Sharpes
  should not be treated as directly calibrated comparisons.
- The simulator fills on the signal bar's close, with simple adverse basis-point
  friction. It does not model realistic liquidity, spreads, borrow costs,
  portfolio margin, or detailed futures rolls. Final liquidation costs affect
  total P&L after the last mark in the returned equity curve.
- Paper-loop fills and backtest execution assumptions differ. A hand-edited
  model configuration can train directly; the runtime does not enforce proposal
  approval. The review gate is a workflow, not a security boundary.

A future methodology change needs its own design, baselines and comparison
results. Passing software tests does not validate market performance.
