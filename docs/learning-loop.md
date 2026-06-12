# The learning loop

The bot logs everything, measures itself nightly, and proposes its own
improvements as pull requests — which a human approves or rejects. It can
change its strategies; it cannot change its risk rules.

```
 every cycle              nightly (21:30)                你 (whenever)
┌─────────────┐    ┌──────────────────────────┐    ┌──────────────────┐
│ trading loop │    │ scripts/nightly_review.sh │    │  review the PR   │
│              │    │                          │    │                  │
│ decisions ───┼───►│ 1. analyze.py → digest   │    │ merge → the bot  │
│ trades       │    │ 2. claude reads digest,  │───►│ runs the change  │
│ shadow book  │    │    journal, code          │    │ reject → it      │
│      ↓       │    │ 3. edits + validate.py   │    │ doesn't          │
│ journal.jsonl│    │ 4. draft PR with evidence │    │                  │
└─────────────┘    └──────────────────────────┘    └──────────────────┘
```

## Layer 1 — the decision journal (always on)

`state/journal.jsonl`, written by the engine every cycle, one entry per
instrument per cycle:

- **decision** entries: the action (`open_long`, `close`, `hold`, `blocked`,
  `stopped`), the reason, and the full indicator context the strategy saw —
  z-score, channel levels, EMAs, ATR regime, equity at the time.
- **trade_closed** entries: the complete trade including risk budget and
  **MAE/MFE** (the worst and best the trade looked while open — the data
  that tells you whether a stop-out was a bad entry or a bad exit).

### The shadow book

When the correlation filter blocks an entry, the bot opens the trade anyway
— in a parallel paper book (`state/shadow.json`) that uses identical sizing,
stops, and exits but never touches equity. When yesterday's blocked NDX long
resolves, the journal knows what the filter actually cost or saved. Over a
real sample this answers "is the filter tuned right?" with data instead of
opinion.

## Layer 2 — analytics (deterministic, nightly or on demand)

```bash
python3 scripts/analyze.py --days 30
```

Computes from the journal: win rate / expectancy / profit factor overall,
per strategy, and per instrument; stop-out counts and **near-miss stop-outs**
(trades up ≥50% of their risk before dying — exit problems, not entry
problems); **re-entries within 24h of a stop-out** and their outcomes (the
whipsaw / revenge-trade pattern); and the shadow book's counterfactual P&L.
Buckets with fewer than 20 trades are flagged `SMALL SAMPLE — treat as
noise`, and the reviewer is bound to respect that.

## Layer 3 — the Claude reviewer (nightly, proposes via PR)

`scripts/nightly_review.sh` (cron it after the evening recap):

1. regenerates the digest,
2. checks out a fresh `bot-review/<date>` branch off `main`,
3. runs Claude Code headless against `prompts/reviewer.md` — the reviewer
   constitution: burden of proof, minimum sample sizes, one theme per
   review, "no change" is the default and a *successful* outcome,
4. any edit must pass `scripts/validate.py` (full test suite, including the
   risk-invariant tests, plus backtests of every instrument),
5. if and only if a change was committed, pushes and opens a **draft PR**
   whose body is the evidence: what the journal showed, the sample size,
   the before/after backtest numbers.

The running bot is untouched until you merge. Rejecting a PR is also
feedback — the branch and its rationale stay in the history.

## What it can and cannot learn

**Can:** strategy parameters, entry/exit logic, new defensive filters
(cooldowns, regime filters, time-of-day filters), correlation-filter tuning,
even pausing or replacing a strategy — with evidence.

**Cannot (see `RISK_RULES.md`, enforced by `tests/test_invariants.py`):**
remove or widen the hard stop, abandon ATR sizing, disable gap protection,
pyramid, raise the risk budget above 1%, delete the correlation concept, or
go live with real money. The constitution is tested code, not a comment.

## Setup on your machine

```bash
# the bot itself (keep it running under tmux/screen/systemd)
python3 scripts/run_bot.py

# the nightly review
crontab -e
# 30 21 * * * /path/to/FableBot/scripts/nightly_review.sh >> /path/to/FableBot/logs/review.log 2>&1
```

Requirements: `claude` (Claude Code CLI) authenticated, `gh` authenticated
against the repo, and the bot writing to `state/journal.jsonl` (the default).

## Layer 4 — the continuous optimizer (always running)

```bash
python3 scripts/optimize_loop.py          # start alongside run_bot.py
python3 scripts/optimize_loop.py --sleep 3  # slower, less CPU
python3 scripts/optimize_loop.py --instrument BTC  # one instrument
```

Because accumulating live trade history is slow (~2–5 trades/day), the
optimizer runs walk-forward backtests continuously in the background to
explore the parameter space immediately:

1. Round-robins through instruments that have `param_bounds` in config.
2. Fetches up to 700 days of history per instrument (longer than the live
   trading window) and splits 75% train / 25% out-of-sample.
3. Randomly samples a parameter set within the bounds, runs the backtest
   on both halves, computes annualised Sharpe ratio.
4. Logs **every trial** to `state/backtest_history.jsonl` — including the
   current config's baseline — so the history is always honest.
5. When OOS Sharpe beats the baseline by ≥0.15 **and** there are ≥8 OOS
   trades behind it, writes a proposal to `state/proposals.jsonl`.

The nightly reviewer reads both files. `scripts/proposals.py` renders a
human-readable summary of what the optimizer found.

### Walk-forward, not simple backtesting

Optimizing on the full history and looking at in-sample Sharpe is the
fastest way to build a strategy that looks great in backtests and fails
live. The optimizer holds out the last 25% of history as a blind test:
a configuration must beat the baseline on data it was never trained on.
High in-sample / low OOS Sharpe in the history = overfit, ignore it.

### What it cannot do

The optimizer changes numbers in `param_bounds` (entry thresholds, EMAs,
channel lengths, stop multipliers). It cannot propose removing the hard
stop, adding leverage, or anything else `RISK_RULES.md` protects. The
reviewer is still the decision-maker — the optimizer only produces
evidence.

## Full system diagram

```
 run_bot.py                optimize_loop.py         nightly_review.sh
┌────────────┐   prices   ┌──────────────────┐   ┌──────────────────────┐
│ 5-min cycle│            │ random param      │   │ 1. analyze.py        │
│            │            │ sampling          │   │ 2. proposals.py      │
│ decisions ─┼──────────► │ walk-forward      │   │ 3. claude reviews    │
│ shadow book│            │ 75/25 split       │   │    digest + proposals │
│ excursions │            │ Sharpe filter     │   │    + history          │
│     ↓      │            │       ↓           │   │ 4. validate.py       │
│ journal    │            │ backtest_history  │   │ 5. draft PR if       │
│ .jsonl     │            │ proposals.jsonl   │──►│    warranted          │
└────────────┘            └──────────────────┘   └──────────────────────┘
                                                           ↓ you merge
```

## Setup (two terminal tabs, or two tmux panes)

```bash
# Tab 1 — the live trading loop
python3 scripts/run_bot.py

# Tab 2 — the continuous optimizer (runs forever alongside the bot)
python3 scripts/optimize_loop.py

# Cron — the nightly review at 21:30
30 21 * * * /Users/ariankhademi/Documents/Business/FableBot/scripts/nightly_review.sh >> /Users/ariankhademi/Documents/Business/FableBot/logs/review.log 2>&1
```

## Why not train an ML/RL model directly?

The journal is deliberately structured so the data will be there if trade
volume ever justifies it. Until then, the walk-forward optimizer + Claude
reviewer gives the same "learn from backtests" benefit without the
overfitting trap: every proposal must survive on held-out data it never
saw.
