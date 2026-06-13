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

## The proposal pipeline

Proposals move through stages (run `python3 scripts/proposals.py` for the
pipeline view):

```
candidate -> [regime folds + perturbation tests] -> qualified -> promoted
                                                 \-> rejected
```

- **candidate**: one walk-forward trial beat the baseline. Weak evidence.
  NOT actionable — the optimizer will validate it on its own; leave it alone.
- **qualified**: survived stage-two validation — beat the current params in
  >= 75% of rolling regime folds AND its ±10%-jittered neighbours still beat
  the baseline on median. **This is the only status you may act on.**
- **rejected**: failed validation. Never act on these, but note the reasons —
  repeated rejections in one parameter region tell you that region is noise.
- **promoted**: already shipped in an earlier review.

A **qualified** proposal is worth acting on only when ALL of these also hold:
- The proposed params make strategic sense — ask: is there a plausible
  market reason this works, or does it merely survive statistics?
- No contradictory evidence in the live journal digest.
- `scripts/validate.py` passes with the proposed params substituted in.

When you act on one, edit `config.yaml` for that instrument's `params` block
only. Do not touch other instruments, `param_bounds`, or risk settings. Then
record the transition:

```
python3 scripts/proposals.py --mark <proposal_id> promoted --reason "shipped in PR"
```

If you decide a qualified proposal should NOT ship (implausible params,
contradicting live evidence), mark it rejected with the reason — that keeps
the pipeline honest and stops it resurfacing tomorrow:

```
python3 scripts/proposals.py --mark <proposal_id> rejected --reason "..."
```

## When the qualified queue is large (≥5 proposals)

If there are 5 or more qualified proposals, do this analysis **before** picking
one to ship — include it in the PR body whether or not you ship anything:

1. **Group by instrument.** List how many qualified proposals exist per
   instrument (e.g. "BTC: 8, GOLD: 3, SPX: 1").

2. **Cluster the parameters.** For each instrument with ≥3 qualified proposals,
   identify which parameters are consistently different from the current config.
   Example: "All 8 BTC proposals have entry_channel in 12–16 vs. current 20.
   The optimizer strongly and repeatedly prefers shorter channels for BTC."
   A tight cluster is stronger evidence than a single high-OOS outlier.

3. **Flag anomalies.** If one instrument has ≫ more qualified proposals than
   the others, it likely means the current config for that instrument is
   significantly mis-tuned for the current regime — note this explicitly.

4. **Still ship only one.** After the analysis, pick the single proposal with
   the best combination of: highest OOS Sharpe improvement, most OOS trades,
   tightest parameter cluster (if applicable), and clearest strategic
   rationale. Ship that one. The rest stay in the queue — they will be
   reviewed in future nights, accumulating more evidence as the optimizer
   continues running.

A large queue is useful signal, not a mandate to batch changes. Batching
makes attribution impossible: if results improve or worsen, you won't know why.

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
