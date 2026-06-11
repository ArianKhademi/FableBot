# FableBot

A multi-instrument trading bot that watches five markets so you don't have to.
It runs three different strategy styles — each matched to how its market
actually moves — sizes every position off volatility, enforces a hard 1% stop
on every trade, and refuses to stack correlated risk-on exposure.

It sends two messages a day:

- **7am** — what's happening in the market
- **9pm** — how the bot did

That's the whole job. Two messages. Five instruments. Zero screen time.

## Strategy map

| Instrument | Symbol    | Strategy           | Timeframe | Why |
|------------|-----------|--------------------|-----------|-----|
| S&P 500    | `^GSPC`   | Mean reversion     | 15-min    | Indices make small overextensions every few hours — fade them back to the mean. |
| Nasdaq     | `^IXIC`   | Mean reversion     | 15-min    | Same regime as the S&P: stretch, snap back. |
| Bitcoin    | `BTC-USD` | Momentum breakout  | 1-hour    | Crypto trends harder than indices — ride the move instead of fading it. |
| Gold       | `GC=F`    | Trend following    | 4-hour    | Commodities move in cleaner waves; the slow timeframe filters intraday whipsaws. |
| Oil        | `CL=F`    | Trend following    | 4-hour    | Same wave structure as gold. |

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
- **Correlation filter.** S&P, Nasdaq, and Bitcoin are tagged `risk_on`. If
  two of them are already long, the bot will not open a third long in the
  group — that would double the real exposure, not diversify it.

## Daily reports

- **07:00 — morning briefing:** overnight move per instrument, current
  volatility regime (ATR as % of price), open positions, and what each
  strategy is watching.
- **21:00 — evening recap:** trades closed today, win/loss count, realized
  and unrealized P&L, and current equity.

Reports print to the console, are archived under `reports/`, and are POSTed
to a webhook if `FABLEBOT_WEBHOOK_URL` is set (works with Slack/Discord-style
JSON webhooks).

## The learning loop

The bot doesn't just trade — it keeps the evidence needed to get better, and
proposes its own improvements for human approval. Full design in
[`docs/learning-loop.md`](docs/learning-loop.md).

- **Decision journal** (`state/journal.jsonl`): every decision — opens,
  closes, stops, holds, *and signals the correlation filter blocked* — with
  the indicator context the strategy saw, plus MAE/MFE on every closed trade.
- **Shadow book**: blocked trades run in a parallel paper book with identical
  sizing and stops, so the correlation filter's real cost/benefit is measured,
  not assumed.
- **Analytics** (`python3 scripts/analyze.py`): expectancy per strategy and
  instrument, near-miss stop-outs, re-entries after stops, and the filter
  counterfactual — with small samples explicitly flagged as noise.
- **Nightly Claude review** (`scripts/nightly_review.sh`, cron it at 21:30):
  Claude Code reads the digest, journal, and code under the constraints in
  [`prompts/reviewer.md`](prompts/reviewer.md), and — only when the evidence
  clears the bar — opens a draft PR with validated changes. You merge or
  reject; the running bot is untouched until you do.
- **The constitution** ([`RISK_RULES.md`](RISK_RULES.md), enforced by
  `tests/test_invariants.py`): the hard stop, ATR sizing, gap protection,
  and the 1% risk ceiling are not tunable, by the reviewer or anyone else.

## Quick start

```bash
pip3 install -r requirements.txt

# Run the test suite (synthetic data, no network needed)
python3 -m pytest

# Backtest a single instrument over recent history
python3 scripts/backtest.py BTC --days 120

# Run the live paper-trading loop (fetches data from Yahoo Finance)
python3 scripts/run_bot.py

# Generate either report on demand
python3 scripts/run_bot.py --report morning
python3 scripts/run_bot.py --report evening
```

The bot is **paper-trading only**: it fetches real market data and simulates
fills, keeping its state in `state/portfolio.json`. Wiring it to a real
broker means implementing the `Broker` interface in `fablebot/portfolio.py`
— and accepting the consequences.

## Configuration

Everything lives in [`config.yaml`](config.yaml): instruments, strategy
parameters, risk budget, correlation groups, report times, and timezone.

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
├── reports.py        # 7am briefing / 9pm recap builders
├── notify.py         # console / file / webhook delivery
├── backtest.py       # bar-by-bar simulator
└── strategies/
    ├── base.py
    ├── mean_reversion.py
    ├── momentum_breakout.py
    └── trend_following.py
```

## Disclaimer

This is an educational paper-trading project. Nothing here is financial
advice, and past performance of any strategy — simulated or otherwise — does
not predict future results.
