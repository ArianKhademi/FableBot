# FableBot

A multi-instrument **paper-trading** bot that watches five markets so you don't
have to. It runs three different strategy styles — each matched to how its
market actually moves — sizes every position off volatility, enforces a hard 1%
stop on every trade, and refuses to stack correlated risk-on exposure.

But it doesn't just trade. FableBot **logs every decision, measures itself, and
improves itself**: a continuous optimizer backtests new parameter sets around
the clock, a staged proposal pipeline filters luck from signal, and a nightly
Claude Code review turns the surviving evidence into draft pull requests you
approve or reject. It can change its strategies; it can never change its risk
rules.

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
trial ──(beats baseline OOS Sharpe by ≥0.15 with ≥8 trades)──► candidate
candidate ──(regime folds + perturbation test)──► qualified ──► promoted
                                              └──► rejected
```

- **candidate** — one walk-forward win. Weak evidence; the reviewer may not act
  on these.
- **validation** (automatic, interleaved with the search) — two harder tests:
  *regime robustness* (re-run head-to-head against the current params across
  rolling time folds; must win ≥75%) and *perturbation* (jitter the params ±10%
  and re-test; a knife-edge curve fit collapses, a real edge survives).
- **qualified** — survived both. The only status the nightly reviewer may act
  on, and even then only with strategic plausibility and a passing
  `scripts/validate.py`.
- **promoted / rejected** — the reviewer's recorded verdict, kept forever.

`python3 scripts/proposals.py` renders the live pipeline view and ends with a
**funnel** that reconciles the whole search — trials run → how many became
candidates → passed / rejected / awaiting — plus the candidate→qualified pass
rate and the overall trial→qualified rate. (If that overall rate is high, the
gate is too loose — exactly the thing the roadmap's first workstream tightens.)

### 5. Nightly Claude review (proposes via PR)

`scripts/nightly_review.sh` (cron it after the evening recap) runs Claude Code
headless against [`prompts/reviewer.md`](prompts/reviewer.md). The reviewer
reads the digest, the proposal pipeline, the backtest history, and the journal
under a constitution — burden of proof, minimum sample sizes, **one validated
change per review, never bundled**, and "no change" as a successful default.
When a large queue of qualified proposals piles up it clusters and ranks them by
instrument before shipping the single best one. Any edit must pass
`scripts/validate.py` (the full test suite *plus* a backtest of every
instrument); only then does it open a **draft PR** whose body is the evidence.
The running bot is untouched until you merge.

## Running the full system

Three processes, typically two tmux panes plus a cron entry:

```bash
# Pane 1 — the live paper-trading loop
python3 scripts/run_bot.py

# Pane 2 — the continuous optimizer (runs forever alongside the bot)
python3 scripts/optimize_loop.py

# Cron — the nightly review, after the 21:00 recap
30 21 * * * /path/to/FableBot/scripts/nightly_review.sh >> /path/to/FableBot/logs/review.log 2>&1
```

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
python3 scripts/optimize_loop.py                # run forever
python3 scripts/optimize_loop.py --instrument BTC
python3 scripts/proposals.py                    # pipeline + funnel view

# Validation gate (what every proposed change must pass)
python3 scripts/validate.py
```

## Configuration

Everything lives in [`config.yaml`](config.yaml): instruments, strategy
parameters, the risk budget, correlation groups, report times, and timezone.
Each instrument also defines `param_bounds` — the search space the optimizer is
allowed to explore. The optimizer only ever moves numbers *within* those
bounds; it cannot touch the risk settings.

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
├── backtest.py       # bar-by-bar simulator
├── optimizer.py      # continuous walk-forward search + validation
├── pipeline.py       # staged proposal store (append-only, audited)
└── strategies/
    ├── base.py
    ├── mean_reversion.py
    ├── momentum_breakout.py
    └── trend_following.py

scripts/
├── run_bot.py          # live loop / on-demand reports
├── backtest.py         # single-instrument backtest
├── analyze.py          # journal → performance digest
├── optimize_loop.py    # continuous optimizer (run forever)
├── proposals.py        # pipeline + funnel view, status transitions
├── validate.py         # tests + per-instrument backtests (the PR gate)
└── nightly_review.sh   # headless Claude Code review → draft PR

prompts/reviewer.md     # the reviewer's constitution
RISK_RULES.md           # the immutable risk constitution
docs/learning-loop.md   # full design of the self-improvement loop
docs/phase-2-learning-engine.md  # the roadmap (see below)
```

## Roadmap

The system today tunes parameters inside three fixed strategies. The next phase
turns it into a self-improving strategy *engine* — harden the validation gate,
move from parameter tuning to a composable strategy DSL, add guided /
evolutionary search that learns from history, then ML and RL signal models, with
multi-core parallelism, an expandable trading universe, and deeper data. The
full staged plan, with the statistical rationale and the guardrails that must
hold throughout, is in
[`docs/phase-2-learning-engine.md`](docs/phase-2-learning-engine.md).

## Disclaimer

This is an educational paper-trading project. Nothing here is financial advice,
and past performance of any strategy — simulated or otherwise — does not predict
future results.
