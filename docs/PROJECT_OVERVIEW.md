# FableBot — Project Overview

*An in-depth tour of what has been built, how the pieces fit together, the
validation methodology, and a detailed research roadmap with phased testing.*

FableBot is a **multi-instrument, paper-only algorithmic trading system** that
trades a portfolio of equities, indices, commodities, and crypto on simulated
capital, **and improves itself** through a governed search-and-review loop. No
real money is ever at risk: every order fills against a paper broker. The design
goal is not a single clever strategy but a **durable machine for discovering,
validating, and safely promoting** strategies — with a hard wall between the
parts that *propose* changes (optimizers, genome search, ML) and the parts that
*authorize* them (a human, assisted by an automated adversarial reviewer).

---

## 1. Design principles (the constitution)

These are encoded in [`RISK_RULES.md`](../RISK_RULES.md) and enforced by
[`tests/test_invariants.py`](../tests/test_invariants.py). They are immutable —
the optimizer cannot edit them, and the reviewer rejects any PR that weakens
them.

1. **Paper-only.** The broker is simulated. There is no live-order code path.
2. **Risk is fixed law, not a tunable.** ATR-based position sizing, a hard 1%
   stop on every trade, and gap protection are invariants. Search may tune
   *entries and exits*; it may never touch the risk envelope.
3. **Stops only ever tighten.** A trailing stop can ratchet toward price but can
   never loosen — true for both the constant trail and the adaptive trail.
4. **Evidence, not authority.** Optimizers, genome search, and ML models produce
   *proposals with backtested evidence*. They never write `config.yaml`. Only a
   human (advised by the reviewer) promotes a change.
5. **Selection and validation live on separate data.** Anything used to *pick* a
   strategy is never the same data used to *bless* it. Walk-forward OOS splits,
   purged folds, and an embargoed held-out set keep selection honest.
6. **No flattering the numbers.** Analytics and the optimizer are never modified
   to make results look better. A worse-but-true number always wins.

---

## 2. Architecture at a glance

```
            data feeds                         the learning loop
  yfinance / Stooq / Binance / CSV        run_engine  (numeric + genome search)
            │                                   │  walk-forward, deflated Sharpe
            ▼                                   ▼
   OHLCVSource → Cache → Routing          proposals pipeline  (funnel + slate)
            │                                   │
            ▼                                   ▼
   Strategy layer                         nightly adversarial review (Claude)
   ├ named: mean_reversion, momentum_           │  → draft PR
   │        breakout, trend_following,           ▼
   │        regime (ADX switch)            human merges in Cursor  →  config.yaml
   ├ genome (composable DSL)                     │
   └ ml / rl models                              ▼
            │                              live books pick up the new config
            ▼
   Risk engine  (sizing, hard stop, gap, trail, correlation, shadow book)
            │
            ▼
   Paper broker → portfolio.json + decision journal → dashboards
```

The system runs as a small number of **long-lived processes** (the live books
and the engine) plus a set of **on-demand, read-only analysis tools**, all
coordinated by cron for the recurring jobs.

---

## 3. The layers in detail

### 3.1 Data ([`fablebot/data.py`](../fablebot/data.py))

A single `OHLCVSource` interface abstracts every feed, so swapping data is a
config change, never a code change:

- **`YFinanceSource`** — broad coverage, the default live feed.
- **`StooqSource`** — deep daily history for indices/commodities.
- **`BinanceSource`** — deep intraday history for crypto.
- **`CSVSource`** — clean local daily history from a file tree (one CSV per
  symbol). This is how a research-grade **WRDS/CRSP** export feeds the *search*
  without changing the live feed. It is opt-in via `data.csv_dir` and is wired
  for **history only** — the live loop always pulls a real-time feed.
- **`CachedSource`** — an on-disk parquet/CSV cache wrapping any source, so
  repeated backtests don't re-hit the network.
- **`RoutingSource`** — per-instrument routing (e.g. crypto → Binance, indices →
  Stooq), assembled by `source_from_config`.

**WRDS/CRSP ingestion** ([`scripts/ingest_wrds.py`](../scripts/ingest_wrds.py))
streams a multi-GB CRSP daily export (the CIZ schema: `PERMNO`, `Ticker`,
`DlyCalDt`, `DlyPrc`, `DlyOpen/High/Low`, `DlyVol`, `DlyFacPrc`) row by row,
filters to a watchlist of tickers, applies the cumulative price-adjustment factor
(`adjusted = price / DlyFacPrc`), takes `abs()` of negative prices (CRSP encodes
bid/ask midpoints as negatives), and writes compact per-symbol adjusted CSVs into
`data/`. It never loads the whole file into memory. See
[`docs/wrds-data.md`](wrds-data.md).

### 3.2 Strategy layer

Four **named strategies**, each a thin `Strategy` subclass that emits a `Signal`
(open long/short, close, hold) and knows *nothing* about sizing or stops:

- **`mean_reversion`** — fades z-score overextensions on indices; the workhorse
  for the experiment books.
- **`momentum_breakout`** — Donchian-style breakouts.
- **`trend_following`** — EMA-crossover trend capture.
- **`regime`** ([`strategies/regime.py`](../fablebot/strategies/regime.py)) — a
  **regime switch**: it reads ADX (trend strength) and **mean-reverts when the
  market is choppy (ADX below a threshold) and rides momentum when the market is
  trending (ADX above it)**. This is the "market/regime check" — a single
  instrument can behave correctly in both regimes instead of being locked to one.

Beyond the named strategies, the search explores two richer hypothesis spaces:

- **The genome DSL** ([`fablebot/dsl.py`](../fablebot/dsl.py)) — a *composable*
  strategy representation: a set of features/indicators, a tree of entry/exit
  conditions, and a risk block. A genome can express strategies no hand-written
  class anticipated, and can replace any named strategy per instrument.
- **ML & RL** ([`ml.py`](../fablebot/ml.py), [`rl.py`](../fablebot/rl.py)) —
  gradient-boosted-tree signal models (pure numpy) and a constrained tabular
  Q-learning agent, both gated by the same validation bar as everything else.

**The trailing-stop hook.** `Strategy.current_trail_mult(df, direction)` returns
the ATR multiple for the trailing stop *for the current bar*. The default returns
a **constant** (`trail_atr_mult`), so momentum/trend behave exactly as before.
`MeanReversion` **overrides** it to implement an **adaptive narrowing trail**:
far from the mean (just entered, `|z|` large) the trail is *loose* so noise
doesn't shake the trade out; as price reverts toward the mean (`|z|` shrinks
toward `exit_z`) the trail *tightens* to lock in the reversion. It interpolates
linearly on `|z|` from `trail_atr_mult` (loose) to `trail_near_mult` (tight). The
invariant holds: a per-bar multiple can only ever *tighten* the realized stop,
never loosen it. *(This adaptive trail ships in the open PR #37; the constant
trail is already live.)*

### 3.3 Risk engine

Sits between the strategy's signal and the broker. Every trade goes through:

- **ATR position sizing** — risk a fixed fraction (1%) of equity per trade,
  sized so the hard stop equals that fraction.
- **Hard 1% stop** — non-negotiable, on every position.
- **Gap protection** — models overnight/weekend gaps through the stop.
- **Trailing stop** — constant or adaptive (above); only ever tightens.
- **Correlation filter** — refuses to stack highly-correlated positions so the
  book isn't secretly one big bet.
- **Shadow book** — when a filter *blocks* a trade, the system still records the
  counterfactual, so we can measure whether the filter helped or cost us.

### 3.4 Execution, state & observability

- **Paper broker + portfolio** — fills at modeled prices with friction (bps),
  persists to `portfolio.json`.
- **Decision journal** — every evaluation (even HOLDs) is logged with the
  indicator context the strategy saw, so the learning loop can reconstruct *why*.
- **Dashboards** ([`fablebot/dashboard.py`](../fablebot/dashboard.py)) — a
  read-only HTML view: KPIs (equity, return %, realized + **live** unrealized
  P&L, win rate), equity curve, open positions with live marks/stops, full trade
  history, per-strategy/instrument breakdowns, and an engine panel. The live bot
  **regenerates its own dashboard every cycle** from marks it already computed.
  `scripts/dashboard_all.py` renders a **combined** comparison page across every
  book. Generating or viewing a dashboard can never touch trading state.
- **`analyze.py` / `engine_status.py` / `proposals.py`** — journal → performance
  digest; is the search learning and what's cleared the gate; the numeric
  pipeline and shipping slate.

### 3.5 Multiple independent books

`run_bot.py --state-dir <dir>` runs a **second, fully isolated** paper account —
its own config, `portfolio.json`, journal, and dashboard — so experiments run
**simultaneously** without touching the main book. This turns the system into an
**A/B testing rig**: hold instruments fixed, change one variable, compare on
separate dashboards. See §6 for the current matrix.

---

## 4. The learning loop

`run_engine.py` is the unified orchestrator. It runs two searches against
**cached history** (never the live book):

1. **Numeric optimization** — TPE Bayesian search over a fixed strategy's
   parameters.
2. **Genome search** — a genetic algorithm over the DSL (crossover, mutation,
   elitism, random immigrants) that can invent new strategy structures.

Every candidate is scored by the **same walk-forward backtest the gate uses**,
and the results flow into the **proposals pipeline**: a funnel from raw trials →
candidates that clear the statistical bar → a human-readable shipping slate.
Nothing here writes `config.yaml`.

### 4.1 Validation methodology (why a number is trustworthy)

A backtest number is easy to fool yourself with. The gate stacks several
defenses:

- **Walk-forward, 75/25 train/OOS split** — parameters are chosen on the first
  75% and scored only on the last 25% (out-of-sample). The reported number is the
  **OOS Sharpe**, not the in-sample fit.
- **Deflated Sharpe ratio** — corrects the OOS Sharpe for the **number of
  configurations tried**. Search a thousand combos and some will look great by
  luck; the deflation penalty discounts exactly that multiple-comparisons
  inflation.
- **Purged, non-overlapping regime folds** — evaluate across distinct market
  regimes with purging so train/test windows don't leak across the boundary.
- **Perturbation test** — jitter the data/params slightly; an edge that
  evaporates under small perturbation was overfit.
- **Embargoed held-out set** — a final slice of data the search never sees, used
  only to bless a finalist.
- **Friction modeled throughout** — costs in basis points are applied in every
  backtest, so an "edge" that's really just ignoring transaction costs dies.

Because **the live backtest models trailing stops** (constant *and* adaptive via
the `current_trail_mult` hook), tuning the stop geometry in a backtest is
meaningful rather than a fiction the live book wouldn't reproduce.

### 4.2 Governance — the review loop

The recurring jobs run under cron (a P&L-aware universe-rotation recommendation
in the evening, a nightly review, weekly model training). The centerpiece is the
**nightly adversarial review**: a headless Claude Code run reads the day's
proposals and the diff, argues *against* the change, and opens a **draft PR** if
(and only if) the evidence holds. A human then merges in Cursor. This is the
wall between *proposing* and *authorizing*: the machine can assemble the
strongest possible case, but a person pulls the trigger.

---

## 5. Analysis & tuning tools (read-only evidence)

These don't run continuously; they're run on demand to answer a specific
question, and each one only ever produces evidence:

- **`premature_stops.py`** — for every hard-stop exit in the live journal, asks
  *did the trade recover to break-even afterward?* (and how many bars / how far
  favorable). Diagnoses stops that are getting wicked out.
- **`stop_sweep.py`** — *recommends* a width where `premature_stops` only
  diagnoses. Re-runs the walk-forward backtest over a grid of hard-stop ×
  trailing-stop multiples and prints OOS Sharpe per combo. `--near` adds a third
  dimension to tune the **adaptive** mean-reversion trail (`trail_near_mult`).
- **`genome_insights.py`** — mines the best genomes *across* instruments for
  recurring structure: which indicators and which risk shapes the winners share.
- **`rotate_universe.py`** ([`fablebot/rotation.py`](../fablebot/rotation.py)) —
  a P&L-aware rotation recommendation that **tilts toward instruments the live
  book is actually making money on** (bounded expectancy-in-R tilt) so a
  profitable incumbent isn't dropped for a marginally higher backtest score.

### What we've learned so far (empirical)

- **Premature stops:** on live data, ~44/67 closes were hard stops and **~82%
  of stopped trades recovered** to break-even afterward (trend ~92%, momentum
  ~84%, mean-reversion ~57%) — stops were getting wicked. This motivated the
  whole stop-geometry investigation.
- **Stop sweep (BTC):** the **trailing** stop (~2.5 ATR) dominates; the hard-stop
  *width* barely moves OOS Sharpe, and the current setting is near-optimal in
  backtest. Conclusion: BTC's losses are a **regime/strategy** problem, not a
  stop-width problem — which is exactly what the `regime` strategy targets.
- **Genome insights:** **~89% of winning genomes use a trailing stop**, and EMA
  is the most common indicator (≈8/22 instruments) — independent confirmation
  that trailing is the right structural bet, and the reason the adaptive trail is
  worth tuning rather than abandoning.

---

## 6. Current experiments (the five books)

| # | Book | Config | State dir | Hypothesis under test |
|---|---|---|---|---|
| 1 | **Main (live)** | `config.yaml` | `state/` | the real portfolio |
| 2 | 15m mean-reversion | `config_experiment.yaml` | `state_experiment` | does MR generalize across liquid ETFs? |
| 3 | 15m MR + constant trail | `config_experiment_trail.yaml` | `state_experiment_trail` | does a constant trailing stop help MR? |
| 4 | Regime (ADX switch) | `config_experiment_regime.yaml` | `state_experiment_regime` | does switching beat either strategy alone? |
| 5 | 15m MR + adaptive trail | `config_experiment_adaptive.yaml` | `state_experiment_adaptive` | does a *narrowing* trail beat a constant one? |

*(Books 1–4 are on `main`; book 5 ships with the adaptive-trail PR #37.)* Each is
a real, isolated paper account with its own dashboard; `dashboard_all.py` shows
all five side by side.

---

## 7. Detailed next steps & testing phases (research roadmap)

The phases are ordered so each one's *evidence* feeds the next. Every phase ends
with a concrete, falsifiable question and the read-only tool that answers it.

### Phase A — Data foundation (in progress)
- **Goal:** put the search on clean, survivorship-bias-free daily history.
- **Steps:** finish CRSP ingestion (`ingest_wrds.py` → per-symbol CSVs), set
  `data.csv_dir: data` in `config.yaml`, restart `run_engine` so it builds the
  CSV-backed source at startup. Broaden `data/wrds_tickers.txt` as needed.
- **Exit test:** the search's universe screen and backtests run end-to-end on CSV
  history; spot-check a handful of symbols against a known source for adjustment
  correctness.

### Phase B — Stop geometry (active)
- **Goal:** stop leaving money on the table to wicks without widening risk.
- **Steps:** merge the adaptive MR trail (#37); run `stop_sweep.py --near` per MR
  instrument to choose `trail_near_mult`; re-run `premature_stops.py` after a few
  weeks of live trading to confirm the recovery rate drops.
- **Exit test:** OOS Sharpe of the adaptive trail ≥ the constant trail on the
  held-out set, *and* the live recovery-after-stop rate falls materially.

### Phase C — Regime awareness
- **Goal:** make each instrument behave correctly in both trending and choppy
  markets, and add a **market-wide** regime filter on top of the per-instrument
  `regime` strategy.
- **Steps:** evaluate book #4 (regime ADX) vs books #2/#3 over a multi-month
  window; test an index-level "risk-on/off" gate that scales exposure by a broad
  trend/volatility read.
- **Exit test:** the regime book's OOS Sharpe and drawdown beat the better of the
  single-strategy books across at least two distinct regimes.

### Phase D — Portfolio construction
- **Goal:** treat the book as a portfolio, not a pile of independent bets.
- **Steps:** tune the correlation filter and sizing jointly; let
  `rotate_universe.py`'s P&L tilt inform (never auto-apply) the live universe;
  measure the shadow book to value the filters.
- **Exit test:** portfolio-level Sharpe and max drawdown beat the equal-weight,
  no-filter baseline out-of-sample.

### Phase E — ML / RL maturation
- **Goal:** let learned models compete fairly with hand-written strategies.
- **Steps:** expand features, retrain weekly under the same gate, compare gated
  ML/RL candidates against the genome winners on the embargoed set.
- **Exit test:** a learned model clears the *same* deflated-Sharpe bar as the
  named strategies on data it never saw — or we keep the simpler strategy.

### Phase F — Live-readiness gating (research only; stays paper)
- **Goal:** define, in advance, exactly what evidence would justify trusting a
  strategy with real capital — without ever flipping that switch here.
- **Steps:** codify a promotion checklist (minimum OOS trade count, deflated
  Sharpe floor, drawdown ceiling, stability across folds, friction sensitivity);
  require the adversarial reviewer to sign off against it.
- **Exit test:** a finalist passes the full checklist on the embargoed set, with
  the reviewer unable to refute it. (Promotion to real money is explicitly out of
  scope for this system.)

---

## 8. Operating model — what runs, and when

**Continuously (long-lived):**
- The **live books** — book #1 (main) and the experiment books #2–#5, each
  `run_bot.py --config … --state-dir …`. Light; polls every few minutes; each
  refreshes its own dashboard.
- **`run_engine.py`** — the numeric + genome search, against cached history.

**On a schedule (cron):**
- P&L-aware universe-rotation *recommendation* in the evening.
- The nightly adversarial review → draft PR.
- Weekly model training.

**On demand (one-shot, read-only):**
- `ingest_wrds.py` (run once when you get a new CRSP export — it stops on its own
  and emits small per-symbol CSVs, not another giant file).
- `stop_sweep.py`, `premature_stops.py`, `genome_insights.py`,
  `rotate_universe.py`, and the dashboards.

**The human:** reviews draft PRs and merges in Cursor. Only the human edits
`config.yaml`.

---

## 9. Open questions & known risks

- **Backtest ≠ live.** Even with walk-forward + deflation, paper fills are
  optimistic about liquidity and slippage. The friction model is a guess; widen
  it before trusting a thin-margin edge.
- **Regime shifts outrun the search.** The engine learns on history; a genuinely
  new regime can invalidate every prior fold. The market-wide regime filter
  (Phase C) is the main mitigation.
- **Multiple-comparisons creep.** Each new tool and book adds searches; the
  deflated Sharpe must keep pace with how many things we try, or "winners" are
  just survivors of a wide net.
- **CSV history bias.** A WRDS/CRSP export is clean but still a *selection* —
  watch for survivorship and corporate-action edge cases in the adjustment.
- **It must stay paper.** Every safeguard above assumes simulated capital. The
  value of this project is the *machine and the methodology*, not a green light to
  trade real money.
