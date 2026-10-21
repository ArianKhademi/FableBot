> Historical implementation brief. This records plans and observations from the original build;
> it is not the current behavior contract. See [architecture](PROJECT_OVERVIEW.md) and
> [validation limitations](validation.md) for the current implementation.

# Phase 2 — From parameter tuning to a self-improving strategy engine

> **Status: implemented.** Workstreams A–I below are all shipped behind tests
> (`tests/test_{optimizer,pipeline,dsl,search,ml,rl,data_universe,backtest}.py`),
> with `RISK_RULES.md` / `tests/test_invariants.py` green throughout, plus
> transaction-cost realism, an IS/OOS overfitting diagnostic and a unified
> orchestrator (`scripts/run_engine.py`). Remaining back-burner items: a paid
> data feed and options/futures (Workstream H-paid / G-options). This document
> is kept as the original brief and rationale.

This is a build brief for a fresh Claude Code session running **locally**, where
the `state/` files (the real `backtest_history.jsonl`, `proposals.jsonl`,
`journal.jsonl`) are present. Read it top to bottom, then read the actual state
before writing code — the diagnosis below should be confirmed against the real
numbers, not taken on faith.

---

## Part 1 — What FableBot is today

A multi-instrument **paper-trading** bot with a learning loop. Nothing trades
real money; the whole point is to learn safely.

- **5 instruments**, each pinned to one hand-written strategy:
  - SPX, NDX → `mean_reversion` (z-score, 15m bars)
  - BTC → `momentum_breakout` (Donchian channels, 1h)
  - GOLD, OIL → `trend_following` (EMA crossover, 4h)
- **Risk constitution** (`RISK_RULES.md`, enforced by `tests/test_invariants.py`):
  hard 1% per-trade stop, ATR-based sizing, gap protection, one position per
  instrument, correlation filter, trailing stops one-way only. These are tested
  code, not guidelines, and **must survive every change below**.
- **Decision journal + shadow book** (`fablebot/journal.py`, `engine.py`):
  every cycle logs the action, reason, and full indicator context; blocked
  trades run in a parallel paper book so the cost of the correlation filter is
  measurable.
- **Analytics** (`fablebot/analytics.py`): win rate, expectancy, profit factor,
  near-miss stop-outs, re-entry/whipsaw detection, shadow-book counterfactual.
- **Continuous walk-forward optimizer** (`fablebot/optimizer.py`): random param
  search within `param_bounds`, 75/25 train/OOS split, annualised Sharpe, logs
  every trial to `state/backtest_history.jsonl`.
- **Proposal pipeline** (`fablebot/pipeline.py`): candidate → validation
  (regime folds + perturbation) → qualified → promoted/rejected, append-only
  audit trail in `state/proposals.jsonl`.
- **Nightly Claude reviewer** (`prompts/reviewer.md`, `scripts/nightly_review.sh`):
  reads the digest + pipeline, ships at most one validated change per night as a
  draft PR, human-gated merge.

### Already done in the previous session (do NOT redo)
- `prompts/reviewer.md` gained a "large qualified queue" section: cluster
  proposals by instrument and rank before shipping one.
- `scripts/proposals.py` gained a **Pipeline totals** block at the bottom
  (validated-passed, rejected, and the **validation pass rate**).

---

## Part 2 — Diagnosis: why the current engine is too loose AND too narrow

**First, confirm against real state.** Run `python3 scripts/proposals.py` and
read the new pass-rate line. "350 rejected out of 2,500" conflates two funnels:
2,500 is *trials*; only trials that clear `oos_trades ≥ 8`, `is_trades ≥ 8`, and
`baseline + 0.15` OOS Sharpe become *candidates*. The 350 rejected are failed
*candidates*. The number that tells you if the gate is too soft is the **pass
rate among validated candidates** — if it's above ~30%, it's too soft.

### Validation is statistically too weak (fix all of these)
1. **`MIN_TRADES = 8` is the worst offender.** A Sharpe on 8 trades has a huge
   standard error — most such "edges" are noise. Raise the floor substantially
   (target dozens of OOS trades) and/or weight by trade count.
2. **Multiple-comparisons / selection bias.** Keeping the best of thousands of
   random draws guarantees spurious winners. There is no correction. Add one:
   e.g. a deflated-Sharpe / minimum-track-record style adjustment, or a
   significance test on the OOS return stream, scaled by how many configs were
   tried.
3. **Validation reuses the data selection saw.** Candidates are picked on the
   full-history walk-forward, then "validated" on folds + perturbations of the
   *same* history. Add a **genuinely held-out confirmation set** (e.g. an
   embargoed final slice, or nested walk-forward) that the selection never
   touched.
4. **The 4 regime folds overlap** (~55% each, stepped) so they're correlated —
   "won 3 of 4" is weaker than it sounds. Use non-overlapping or purged/embargoed
   folds.
5. **The perturbation bar is too low** — "median jittered Sharpe > baseline",
   which the candidate already clears by construction. Require stability *near
   the candidate's own* Sharpe (e.g. median jitter within X% of it), not merely
   above the thing it already beat.

### Correctness bug to fix
- **Baseline never refreshes after a promotion.** `_ensure_baseline()` returns
  the first baseline ever logged. After config.yaml changes, candidates are
  still graded against the *old* config, inflating the candidate count. The
  baseline must be re-measured whenever the live params change.

### The architectural ceiling
The optimizer only **tunes numbers inside 3 fixed strategies**, each locked to
one instrument. It can never discover a new signal, a better indicator, or that
strategy X suits instrument Y. That is the real limit, and Part 3 is about
removing it.

### The "6 years to ship them all" worry is a false constraint
It assumes every qualified proposal must ship, one per night. With stricter
validation far fewer qualify, and "one change per *instrument* per night" is
still cleanly attributable (the books are independent) — so the human-review
cadence stops being the bottleneck. The optimizer's search never was.

---

## Part 3 — The build (confirmed requirements)

Decisions confirmed by the user: go the full distance — composable strategy DSL,
then ML signal models, then an RL agent; guided/evolutionary search that learns
from history; all-CPU-core parallelism with a throttle knob; an expandable
universe (many tickers, eventually options/futures); real data sources as a
later necessity. Build it in the staged order below — each layer is useful on
its own and de-risks the next.

### Workstream A — Harden validation (do this FIRST)
Tighten the five weaknesses and fix the baseline bug above. This is the
foundation: every later layer (DSL, ML, RL) generates far more candidates, so a
rigorous gate must exist *before* the search space explodes — otherwise the
pipeline floods with noise. Add regression tests to `tests/test_optimizer.py` /
`tests/test_pipeline.py` proving the stricter gate rejects known-noisy inputs.

### Workstream B — Composable strategy DSL (the core unlock)
Replace "3 hardcoded strategies" with a **search space of strategies**:
- An indicator palette: RSI, MACD, ADX/DI, Bollinger/Keltner bands, z-score,
  Donchian, EMA/SMA crossovers, ATR regime, volume features, etc. (extend
  `fablebot/indicators.py`).
- Entry/exit **rule templates** that compose indicators into a `Strategy`
  (threshold crosses, band touches, regime gates, multi-condition AND/OR).
- A **strategy genome**: a serialisable spec (indicators + rules + params) that
  `build_strategy` can instantiate. The optimizer searches over genomes, not
  just numbers.
- **Decouple strategy from instrument**: any genome can be tried on any symbol.
  The current 5 fixed assignments become just the seed population.
- Hard constraint: every genome still emits `Signal`s and goes through the
  unchanged risk engine (sizing, hard stop, trailing). The DSL changes *what
  signals fire*, never *how risk is managed*.

### Workstream C — Guided / evolutionary search (the "keeps learning" part)
Replace pure random sampling with a search that uses the accumulated history:
- **Bayesian optimization** over continuous params within a genome.
- **Genetic search** over genomes: mutation (swap an indicator, nudge a rule)
  and crossover of the best validated survivors, seeded from
  `backtest_history.jsonl`.
- Persist the population/surrogate so restarts resume learning rather than
  starting blind. This is what makes it behave "like an ML model that learns
  from previous iterations."

### Workstream D — ML signal models (after B + C are stable)
Add a new strategy *type* that trains a model (start with gradient-boosted trees
on engineered features; the journal/history schema was designed for this) to
produce entry signals per instrument. Must use the same walk-forward + held-out
validation as everything else — ML is the easiest thing to overfit, so it earns
its place only by surviving the Workstream-A gate on data it never saw.

### Workstream E — RL agent (LAST, highest risk)
An RL policy that learns positioning from features. Build this only once the
DSL, guided search, ML models, deeper data, and the hardened gate all exist —
RL benefits from every one of them and is the most overfit-prone and the hardest
to keep inside the hard-stop/ATR-sizing constitution. The action space must be
constrained so the risk engine still owns sizing and stops; the agent proposes
direction/timing, not leverage.

### Workstream F — Throughput (parallelism)
- Parallelise trials with `multiprocessing` (each backtest is independent and
  CPU-bound). Expect ~Ncores× more trials/day.
- Add a CLI knob, e.g. `scripts/optimize_loop.py --workers N` (default: all
  cores; let the user dial it down to reserve cores for other work).
- The 1.5s `TRIAL_SLEEP` throttle should be off by default in a dedicated run
  and configurable.

### Workstream G — Universe expansion
The 5 tickers were just a starting set; make the universe **config-driven and
large**:
- Move the symbol list to config / a screener so it can scan many equities and
  ETFs without code changes.
- Combined with Workstream B (any strategy on any symbol), the optimizer can
  discover which symbols are even worth trading.
- **Options/futures are a separate, later phase**: continuous futures fit the
  existing price-based engine, but options need contract/expiry/Greeks modelling
  and a rethink of ATR sizing and the hard stop — do equities/ETFs/crypto/
  continuous-futures first, options as its own project.

### Workstream H — Data
Longer/deeper history so rankings span more regimes:
- **Now, free, no API:** daily history is free and decades-deep (yfinance daily,
  Stooq) — good for daily/4h genomes. Crypto has free deep *intraday* history
  via exchange APIs (e.g. Binance 1m back years). Use these first.
- **yfinance limits to know:** 1m ≈ last 7–8 days, 15m ≈ last 60 days, no 30s.
  So deep *intraday equity* history is the one thing that needs a paid source.
- **Later:** a market-data API (Polygon / Tiingo / Databento) or a bulk download
  stored locally to avoid repeated API calls. Keep this on the back burner until
  the engine above justifies the cost; design the data layer behind an interface
  so swapping sources is a config change.

### Workstream I — Pipeline shipping cadence
Add the capability to promote up to **one change per instrument per night**
(~5/night) rather than one total, still cleanly attributable. With Workstream A
in place this may rarely be needed, but build the option.

---

## Part 4 — Suggested order of operations
1. **A** — harden validation + fix the baseline bug (gate must be solid first).
2. **F** — parallelism (cheap, makes everything after it faster to iterate).
3. **B** — the strategy DSL (the core architectural unlock).
4. **C** — guided/evolutionary search over the DSL.
5. **G** + **H (free tier)** — expand the universe and the data history.
6. **D** — ML signal models.
7. **E** — the RL agent.
8. **I** — per-instrument nightly shipping (small, do whenever convenient).
9. **H (paid)** + **options/futures** — only once the above pays off.

Each step ships behind tests and leaves `RISK_RULES.md` and
`tests/test_invariants.py` green. Prefer many small PRs over one giant one.

---

## Part 5 — Guardrails that hold throughout
- **The risk constitution is immutable.** Sizing, the hard 1% stop, gap
  protection, one-position-per-instrument, paper-trading-only — no layer here
  may weaken them. `tests/test_invariants.py` must stay green.
- **Selection and validation stay on separate data.** Anything chosen on one
  slice is confirmed on a slice it never saw. No exceptions, especially for ML/RL.
- **The history stays honest.** Never modify analytics/optimizer code to flatter
  its own numbers; `backtest_history.jsonl` is the permanent record.
- **The optimizer only produces evidence.** It never edits `config.yaml` or live
  code. The nightly reviewer (human-gated) is still the only thing that ships.
