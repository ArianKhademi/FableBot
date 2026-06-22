# FableBot

A multi-instrument **paper-trading** bot that watches five markets so you don't
have to. It runs three different strategy styles — each matched to how its
market actually moves — sizes every position off volatility, enforces a hard 1%
stop on every trade, and refuses to stack correlated risk-on exposure.

But it doesn't just trade. FableBot **logs every decision, measures itself, and
improves itself**: a continuous optimizer backtests new ideas around the clock,
a staged proposal pipeline filters luck from signal with a statistically
rigorous gate, and a nightly Claude Code review turns the surviving evidence
into draft pull requests you approve or reject. It can change its strategies; it
can never change its risk rules.

It no longer only *tunes* three fixed strategies. A composable strategy DSL
turns "three strategies" into a **search space of strategies**; an evolutionary
search breeds and refines them; gradient-boosted-tree and reinforcement-learning
models add learned signals — and **every one of them has to clear the same
out-of-sample gate** (deflated Sharpe, purged regime folds, an embargoed
held-out set) before it can be proposed. See [the strategy engine](#the-strategy-engine-phase-2).

It also sends two messages a day:

- **7am** — what's happening in the market (incl. relevant news headlines)
- **9pm** — how the bot did

Five instruments. Two messages. Zero screen time — and a system that gets
better while you sleep.

## Strategy map

| Instrument | Symbol    | Strategy           | Timeframe | Why |
|------------|-----------|--------------------|-----------|-----|
| S&P 500    | `^GSPC`   | Mean reversion     | 15-min    | Indices make small overextensions every few hours — fade them back to the mean. |
| Nasdaq     | `^IXIC`   | Mean reversion     | 15-min    | Same regime as the S&P: stretch, snap back. |
| Bitcoin    | `BTC-USD` | Momentum breakout  | 1-hour    | Crypto trends harder than indices — ride the move instead of fading it. |
| Gold       | `GC=F`    | Trend following    | 4-hour    | Commodities move in cleaner waves; the slow timeframe filters intraday whipsaws. |
| Oil        | `CL=F`    | Trend following    | 4-hour    | Same wave structure as gold. |

The instrument list is the *starting* set, not a hard limit — it's
config-driven, and expanding the universe is on the roadmap (see below).

### Mean reversion (S&P 500, Nasdaq — 15m)

Price is measured as a z-score against its 20-bar mean. When an index
stretches more than 2 standard deviations from the mean, the bot fades the
move; it exits when price reverts back toward the mean.

### Momentum breakout (Bitcoin — 1h)

Donchian channel breakout: a close above the prior 20-bar high opens a long
(below the 20-bar low opens a short). The exit is the opposite side of a
shorter 10-bar channel, so winners are given room to trend.

### Trend following (Gold, Oil — 4h)

A 20/50 EMA crossover system. Cross up → long, cross down → short, with the
position flipped on the opposite signal. Slow by design.

## Risk engine

- **ATR-based position sizing, per instrument.** Each trade risks a fixed
  fraction of equity (default 1%). Size = risk budget ÷ (ATR × stop
  multiplier). A quiet day on gold gets a bigger size than a volatile day on
  bitcoin — risk stays constant even when volatility doesn't.
- **Hard 1% stop. No exceptions.** Every position carries a stop priced so
  that hitting it loses exactly the per-trade risk budget. On top of that,
  the broker force-closes any position whose unrealized loss reaches the risk
  budget (covers gaps past the stop). There is no "let me give it room."
- **Trailing stops on trend trades.** Bitcoin, gold, and oil ratchet their
  stop an ATR multiple behind price as the move works (`trail_atr_mult` in
  config), converting open profit into locked-in profit. Stops only ever
  tighten — widening is constitutionally impossible.
- **Correlation filter.** S&P, Nasdaq, and Bitcoin are tagged `risk_on`. If
  two of them are already long, the bot will not open a third long in the
  group — that would double the real exposure, not diversify it.

These rules live in [`RISK_RULES.md`](RISK_RULES.md) and are enforced by
`tests/test_invariants.py`. They are **not tunable** — not by you in a hurry,
not by the optimizer, not by the nightly reviewer.

## Daily reports

- **07:00 — morning briefing:** overnight move per instrument, current
  volatility regime (ATR as % of price), open positions, what each strategy is
  watching, and **recent news headlines per instrument** (Fed / CPI / jobs and
  other market-moving items are flagged `*** HIGH IMPACT ***`). News is *soft
  context only* — it never blocks or forces a trade; strategies trade on price.
- **21:00 — evening recap:** trades closed today, win/loss count, realized and
  unrealized P&L, and current equity.

Reports print to the console, are archived under `reports/`, and are POSTed to a
webhook if `FABLEBOT_WEBHOOK_URL` is set (works with Slack/Discord-style JSON
webhooks).

## The learning loop

The bot keeps the evidence needed to get better and proposes its own
improvements for human approval. Full design in
[`docs/learning-loop.md`](docs/learning-loop.md).

### 1. Decision journal + shadow book (always on)

- **`state/journal.jsonl`** records every decision — opens, closes, stops,
  holds, *and the signals the correlation filter blocked* — with the full
  indicator context the strategy saw, plus **MAE/MFE** (the worst and best a
  trade looked while open) on every closed trade.
- **Shadow book**: blocked trades run in a parallel paper book with identical
  sizing and stops, so the correlation filter's real cost/benefit is *measured,
  not assumed*.

### 2. Analytics (deterministic)

`python3 scripts/analyze.py` computes expectancy / win rate / profit factor
per strategy and instrument, **near-miss stop-outs** (trades up ≥50% of their
risk before dying — an exit problem, not an entry problem), **re-entries within
24h of a stop-out** (the whipsaw / revenge-trade pattern), and the shadow
book's counterfactual P&L. Buckets with fewer than 20 trades are explicitly
flagged as noise, and the reviewer is bound to respect that.

### 3. Continuous optimizer (always running)

`python3 scripts/optimize_loop.py` runs **walk-forward backtests forever** in
the background, because waiting for live trades (~2–5/day) to accumulate is far
too slow. Each trial:

1. samples a random parameter set within each instrument's `param_bounds`,
2. splits the instrument's history **75% train / 25% out-of-sample**,
3. backtests both halves and computes an annualised **Sharpe ratio**,
4. logs *every* trial to `state/backtest_history.jsonl` (including the current
   config's baseline) so the record is always honest.

Walk-forward is the point: a config must beat the baseline on data it was never
fit to. High in-sample / low out-of-sample Sharpe is the signature of
overfitting, and it's left in the log as a warning, not acted on.

### 4. The proposal pipeline

One good out-of-sample number can be luck, so proposals are **staged**, and
every transition is an append-only event in `state/proposals.jsonl`:

```
trial ──(beats baseline OOS Sharpe by ≥0.15, ≥20 trades, passes deflated Sharpe)──► candidate
candidate ──(purged regime folds + perturbation + embargoed held-out)──► qualified ──► promoted
                                                                    └──► rejected
```

- **candidate** — one walk-forward win that also clears the **deflated Sharpe
  ratio** (its OOS edge must beat the *expected maximum* Sharpe of N null trials,
  so the bar rises automatically with how many configs were tried — the
  multiple-comparisons correction). Still weak evidence on its own.
- **validation** (automatic, interleaved with the search) — three harder tests:
  *regime robustness* across **non-overlapping, embargo-purged** folds (must win
  ≥75% head-to-head vs the current config); *perturbation* (jitter ±10% — the
  median must stay within tolerance of the candidate's **own** Sharpe, not merely
  beat the baseline it already cleared); and an **embargoed held-out slice the
  search never touched** (the candidate must beat the current config there too —
  the test selection bias cannot game).
- **qualified** — survived all of it. The only status the nightly reviewer may
  act on, and even then only with strategic plausibility and a passing
  `scripts/validate.py`.
- **promoted / rejected** — the reviewer's recorded verdict, kept forever.

The baseline each candidate is graded against is keyed to the params it
measured and re-measured whenever the live config changes — so a promotion never
leaves new candidates graded against a stale config.

`python3 scripts/proposals.py` renders the live pipeline view and ends with a
**funnel** that reconciles the whole search — trials run → how many became
candidates → passed / rejected / awaiting — plus the candidate→qualified pass
rate and the overall trial→qualified rate. (If that overall rate is high, the
gate is too loose — exactly the thing the roadmap's first workstream tightens.)

### 5. Nightly Claude review (proposes via PR)

`scripts/nightly_review.sh` (cron it after the evening recap) runs Claude Code
headless against [`prompts/reviewer.md`](prompts/reviewer.md). The reviewer
reads the digest, the proposal pipeline, the backtest history, and the journal
under a constitution — burden of proof, minimum sample sizes, **at most one
validated change per instrument per review** (the books are independent, so up
to ~5/night stays cleanly attributable; never two on the same book, never
bundled), and "no change" as a successful default. The pipeline view hands it a
ready-made *shipping slate* — each instrument's single strongest qualified
proposal. Any edit must pass `scripts/validate.py` (the full test suite *plus* a
backtest of every instrument); only then does it open a **draft PR** whose body
is the evidence. The running bot is untouched until you merge.

## The strategy engine (Phase 2)

The loop above no longer only nudges numbers inside three fixed strategies — it
searches a space of strategies, and can learn entirely new signals. Every layer
emits the same `Signal`s through the **unchanged risk engine** (sizing, the hard
stop, gap protection, trailing) and must clear the same hardened gate before it
can be proposed. Full brief: [`docs/phase-2-learning-engine.md`](docs/phase-2-learning-engine.md).

- **Composable strategy DSL** (`fablebot/dsl.py`) — a serialisable *genome*
  (indicator palette + rule trees + risk block) that `build_strategy`
  instantiates. The palette spans RSI, MACD, ADX/DI, Bollinger/Keltner, z-score,
  Donchian, EMA/SMA, ATR and volume features; rules compose them with
  threshold/cross/AND/OR logic. The three hand-written strategies are just seed
  genomes (and reproduce bit-for-bit). Any genome can trade any symbol.
- **Guided / evolutionary search** (`fablebot/search.py`, `scripts/evolve.py`) —
  a genetic search over genomes (selection + crossover + mutation, seeded from
  the seeds and any survivors on disk, **persisted so restarts resume learning**)
  plus a dependency-free TPE Bayesian optimiser for refining an elite genome's
  numbers.
- **ML signal models** (`fablebot/ml.py`, `scripts/train_ml.py`) — a numpy
  gradient-boosted-tree ensemble trained on engineered, leakage-free features to
  predict forward returns; thresholded into entries. Fit on train only, judged
  out-of-sample.
- **RL agent** (`fablebot/rl.py`) — a constrained tabular Q-learning policy that
  proposes **direction/timing only**; the risk engine still owns sizing and
  stops. The most overfit-prone layer, gated the hardest.
- **Parallelism** (`scripts/optimize_loop.py --workers N`) — trials run across
  CPU cores; results are deterministic vs the serial path. `--no-throttle` for a
  dedicated run.
- **Config-driven universe** (`config.yaml: universe`, `scripts/evolve.py
  --universe`) — scan many equities/ETFs/crypto; a liquidity screener ranks them.
  Combined with the DSL, the search discovers *which symbols are worth trading*.
- **Pluggable data** (`fablebot/data.py`) — an `OHLCVSource` interface
  (yfinance / Stooq deep-daily / Binance deep-intraday) behind an on-disk cache;
  swapping feeds is a config change.
- **Transaction costs** (`config.yaml: costs.friction_bps`) — commission +
  slippage applied to every fill during the search, so the gate can't certify a
  frictionless-only edge.
- **Unified engine** (`scripts/run_engine.py`) — one orchestrator that advances
  the numeric optimizer and the evolutionary genome search together, prints a
  per-cycle heartbeat, and periodically puts the best evolved genomes through
  the full gate into `genome_proposals.jsonl`. `scripts/engine_status.py` shows,
  read-only, whether the search is learning and what has cleared the gate.
- **Dynamic universe rotation** (`scripts/rotate_universe.py`) — a daily scan
  scores the whole universe by the best gated edge per symbol and proposes which
  instruments the live book should trade, instead of a fixed five. Guardrailed:
  open positions are sticky (never force-closed to rotate), concurrency is
  capped, and hysteresis resists churn. It recommends; the reviewer applies.

## Setup (from a fresh clone)

```bash
python3 -m pip install -r requirements.txt   # Python 3.9+
python3 -m pytest                            # sanity check — all green, no network needed
```

**API keys / `.env`:** none required to run or search. All data sources are
keyless — yfinance and Stooq (equities/ETFs, daily + intraday) and Binance's
public endpoint (crypto). The only optional setting is an env var for report
delivery:

```bash
export FABLEBOT_WEBHOOK_URL="https://hooks.slack.com/..."   # optional; omit to print/file only
```

**Only for the nightly reviewer** (`scripts/nightly_review.sh`) you also need
two CLIs on `PATH`, both authenticated: `claude` (Claude Code) and `gh` (GitHub,
`gh auth login`). Nothing else uses them.

**Restarting is safe — you cannot lose search progress.** All state lives under
`state/` and resumes automatically: `portfolio.json` (live positions),
`backtest_history.jsonl` (every trial), and `population_*.json` (each genome
population, reloaded at its saved generation). Kill and restart `run_bot` /
`run_engine` freely. The `cycle N` counter `run_engine` prints is per-process
and restarts at 1 — that is **cosmetic**; the startup line reports the real
resumed state (`N trials on disk; resuming genome populations: BTC@gen1954, …`).

## Running the full system

There are three evidence pipelines, and they land in **different files** — which
is why `scripts/proposals.py` (numeric only) can read zero while the genome
search is busy:

| Pipeline | Produced by | Gated by | Lands in |
|---|---|---|---|
| Numeric params | `run_engine` / `optimize_loop` | automatic | `state/proposals.jsonl` |
| Evolved genomes | `run_engine` (every `--validate-every` cycles) or `evolve.py` | automatic | `state/genome_proposals.jsonl` |
| ML / RL models | `train_ml.py` / `train_rl.py` | automatic | `state/ml_proposals.jsonl` / `rl_proposals.jsonl` |
| Universe rotation | `rotate_universe.py` | guardrails | `state/rotation_proposals.jsonl` |

**Leave running (daemons, e.g. in tmux or as launchd agents):**
```bash
python3 scripts/run_bot.py                       # live paper trader; retrains live models daily
python3 scripts/run_engine.py --cycles 0 --workers 8   # numeric + genome search, gates & surfaces both
```
`run_engine` supersedes `optimize_loop` (it does the numeric search **and** the
genome search, and now validates the best genomes through the gate). Run **one**
of them, not both — they share state files. It prints a per-cycle heartbeat so
you can see it working.

Once the traded set has *converged* (engine_status shows a high population
saturation / `⚠ spinning`, and the best genome stops rising), point the engine at
the broad scan list instead — it keeps discovering on fresh, lower-noise symbols
rather than idling on the exhausted five:
```bash
python3 scripts/run_engine.py --cycles 0 --universe    # genome search over config.yaml: universe
```
Universe symbols carry no `param_bounds`, so this runs the **genome (structure)
search only** — the numeric param tuner needs a bounded space and is skipped for
them. Qualified discoveries still land in `state/genome_proposals.jsonl` for the
reviewer to ship; the live book changes only when a human (or the nightly
reviewer) edits `config.yaml`.

**Cron (periodic):**
```cron
45 20 * * * cd /path/to/FableBot && python3 scripts/rotate_universe.py --screen-top 12 >> logs/rotation.log 2>&1
30 21 * * * /path/to/FableBot/scripts/nightly_review.sh >> /path/to/FableBot/logs/review.log 2>&1
0 3 * * 0  cd /path/to/FableBot && { python3 scripts/train_ml.py; python3 scripts/train_rl.py; } >> logs/train.log 2>&1
```

**Check on it anytime (read-only):**
```bash
python3 scripts/engine_status.py     # is it learning? what's cleared the gate?
python3 scripts/proposals.py         # the numeric pipeline + shipping slate
```

Nothing above edits `config.yaml` or the live book — every pipeline only
produces gated *evidence*. The **nightly reviewer** turns that evidence into
human-gated draft PRs (≤1 change per instrument); you review and merge, and the
bot adopts it on restart.

The bot is **paper-trading only**: it fetches real market data and simulates
fills, keeping state in `state/portfolio.json`. Wiring it to a real broker means
implementing the `Broker` interface in `fablebot/portfolio.py` — and accepting
the consequences.

## Quick start

```bash
pip3 install -r requirements.txt

# Run the test suite (synthetic data, no network needed)
python3 -m pytest

# Backtest a single instrument over recent history
python3 scripts/backtest.py BTC --days 120

# Run the live paper-trading loop, or fire a report on demand
python3 scripts/run_bot.py
python3 scripts/run_bot.py --once               # one cycle and exit
python3 scripts/run_bot.py --report morning     # 7am-style briefing now
python3 scripts/run_bot.py --report evening     # 9pm-style recap now

# Performance digest from the journal
python3 scripts/analyze.py --days 30

# Continuous optimizer + proposal pipeline
python3 scripts/optimize_loop.py                # all cores, throttled
python3 scripts/optimize_loop.py --workers 4 --no-throttle
python3 scripts/optimize_loop.py --instrument BTC
python3 scripts/proposals.py                    # pipeline + funnel + shipping slate

# Strategy engine (Phase 2)
python3 scripts/run_engine.py --cycles 1        # optimizer + genome search, one cycle
python3 scripts/evolve.py --generations 10      # evolutionary genome search
python3 scripts/evolve.py --universe --screen-top 10  # scan the wider universe
python3 scripts/train_ml.py --instrument BTC    # train + gate an ML signal model

# Validation gate (what every proposed change must pass)
python3 scripts/validate.py
```

## Configuration

Everything lives in [`config.yaml`](config.yaml): instruments, strategy
parameters, the risk budget, correlation groups, report times, and timezone.
Each instrument also defines `param_bounds` — the search space the optimizer is
allowed to explore. The optimizer only ever moves numbers *within* those
bounds; it cannot touch the risk settings.

Phase 2 adds three optional sections: `data` (which `OHLCVSource` and cache to
fetch through), `universe` (a large, config-driven scan list for the genome
search — add tickers without code changes), and `costs` (`friction_bps`, the
commission + slippage applied during the search). An instrument may also carry a
`genome` to run a composable-DSL strategy, or a `model` block
(`{type: ml_gbt, ...}` / `{type: rl_qlearn, ...}`) that the live engine trains
on deep history at startup and **re-trains daily** — so genomes, ML and RL all
trade live, not just numeric params. The risk engine still owns every fill.

## Project layout

```
fablebot/
├── config.py         # config loading / dataclasses
├── data.py           # OHLCV fetching (yfinance) + 4h resampling
├── indicators.py     # ATR, EMA, z-score, Donchian channels
├── risk.py           # ATR position sizing, hard stop, correlation filter
├── portfolio.py      # positions, paper broker, trade log, persistence
├── engine.py         # evaluation cycle + scheduler + shadow book
├── journal.py        # append-only decision journal (JSONL)
├── analytics.py      # digest: expectancy, stop quality, counterfactuals
├── news.py           # Yahoo Finance headlines + high-impact flagging
├── reports.py        # 7am briefing / 9pm recap builders
├── notify.py         # console / file / webhook delivery
├── backtest.py       # bar-by-bar simulator (+ optional transaction costs)
├── optimizer.py      # parallel walk-forward search + hardened validation gate
├── pipeline.py       # staged proposal store (append-only, audited)
├── dsl.py            # composable strategy DSL: genomes + operators
├── search.py         # genetic search + TPE + genome-aware gate
├── ml.py             # gradient-boosted-tree signal models
├── rl.py             # constrained Q-learning agent
├── universe.py       # universe screening (liquidity)
└── strategies/
    ├── base.py
    ├── mean_reversion.py
    ├── momentum_breakout.py
    └── trend_following.py

scripts/
├── run_bot.py          # live loop / on-demand reports
├── backtest.py         # single-instrument backtest
├── analyze.py          # journal → performance digest
├── optimize_loop.py    # continuous numeric optimizer (--workers, --no-throttle)
├── evolve.py           # evolutionary genome search (--universe)
├── train_ml.py         # train + gate ML signal models
├── run_engine.py       # unified orchestrator (optimizer + genome search)
├── proposals.py        # pipeline + funnel + shipping slate, status transitions
├── validate.py         # tests + per-instrument backtests (the PR gate)
└── nightly_review.sh   # headless Claude Code review → draft PR

prompts/reviewer.md     # the reviewer's constitution
RISK_RULES.md           # the immutable risk constitution
docs/learning-loop.md   # full design of the self-improvement loop
docs/phase-2-learning-engine.md  # the strategy-engine brief
```

## Roadmap

**Phase 2 — the strategy engine — is implemented.** The validation gate is
hardened (deflated Sharpe, purged folds, embargoed held-out, raised trade
floor, baseline-refresh fix); parameter tuning has become a composable strategy
DSL with a guided/evolutionary search that learns from history; ML and RL signal
models are in, all behind the same gate; trials run multi-core; the universe is
config-driven with a screener; data sits behind a swappable cached source; and
transaction costs are modelled in the search. The staged brief, statistical
rationale and guardrails are in
[`docs/phase-2-learning-engine.md`](docs/phase-2-learning-engine.md).

Still on the back burner (by design — they need the engine above to justify the
cost): a **paid market-data feed** (Polygon/Tiingo/Databento) for deep intraday
*equity* history, and **options/futures**, which need contract/expiry/Greeks
modelling and a rethink of ATR sizing — their own project, after equities /
ETFs / crypto / continuous futures.

## Disclaimer

This is an educational paper-trading project. Nothing here is financial advice,
and past performance of any strategy — simulated or otherwise — does not predict
future results.
