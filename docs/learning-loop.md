# Learning and proposal loop

The research loop generates evidence for a human review. It does not establish
that a strategy will profit, and it does not automatically deploy a proposal.

1. The paper engine journals signals, holds, stops and blocked entries. A shadow
   book records counterfactual positions for correlation-filter analysis.
2. Analytics summarize trades, excursions, stop-outs and re-entries. Small
   samples are marked as noise rather than treated as reliable performance.
3. Numeric and genome searches evaluate alternatives on historical data.
   Model trainers produce learned-signal specifications.
4. Validators record candidate outcomes. Read [validation limitations](validation.md)
   before interpreting a `qualified` status; some checks can be skipped.
5. The optional Claude/GitHub reviewer reads the evidence and may propose a code
   change in a draft PR. A human decides what to merge and how to restart a book.

## Evidence files

| Producer | Default record |
| --- | --- |
| Numeric optimizer | `state/backtest_history.jsonl`, `state/proposals.jsonl` |
| Genome search | `state/population_*.json`, `state/genome_proposals.jsonl` |
| ML / RL trainers | `state/ml_proposals.jsonl`, `state/rl_proposals.jsonl` |
| Universe rotation | `state/rotation_proposals.jsonl` |
| Paper engine | `state/portfolio.json`, `state/journal.jsonl` |

These records are local experiment outputs. They are intentionally excluded from
Git. Manually changing `config.yaml` can bypass the proposal workflow; approval is
not an enforced runtime security boundary.

## Running it

Use the [operations guide](operations.md) for commands and explicit state paths.
The synthetic demo starts none of these processes. Do not run the reviewer as a
setup check: it can edit code, change Git branches, push, and open PRs. Its
legacy shell script shares a checkout with other processes; use a dedicated
checkout for such automation and review its permissions before enabling it.
