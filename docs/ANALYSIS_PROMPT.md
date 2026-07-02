# FableBot — performance-analysis prompt

Paste the block below into a fresh Claude Code chat (in this repo) to get a full,
evidence-based performance review. It's written so a chat with **no prior context**
can run it correctly. Swap "today" for a date/range if you want a specific window.

---

Run a full performance analysis of the FableBot paper-trading books. This is a
**paper-only** system; never edit `config.yaml`, never ship anything on faith —
produce evidence and, at most, a PR the human reviews. Steps:

1. **Inventory the books.** Discover every live book (`state*/` dirs with a
   matching `config_*.yaml`). For each, read `portfolio.json` / `journal.jsonl`
   directly — do NOT trust `state/dashboard_all.html`, which is a periodic
   snapshot and may be stale. Report per book: equity, return %, realized +
   unrealized P&L, win rate, trailing-7-day P&L, and open positions.

2. **Pull today's closed trades** per book: time, instrument, direction, pnl,
   close_reason. Separate winners from losers, and note the *sequence* (e.g. did
   early trades win and later ones lose?).

3. **Get the market context.** Fetch the traded instruments' daily move, intraday
   range, and whether the session **trended one way or chopped** (and which
   sectors — a tech-only selloff differs from a broad one). Deep daily history is
   in `data/*.csv` (CRSP); intraday via yfinance (`get_source('yfinance')`).

4. **Correlate the losses to the regime.** Were the losing trades **fighting the
   trend** — shorts in an up day, longs (dip-buying) in a down day? Mean-reversion
   bleeds in sustained trends; that's usually the story.

5. **Compare against what's already built.** For the diagnosed failure mode, say
   which existing tool addresses it and which book would have handled today:
   - `trend_filter_ma` + `trend_filter_side` (`both` = symmetric/all-weather;
     `shorts` = keep-upside but no bear brake) on MeanReversion
   - the correlation filter (concentration cap), the adaptive trail, the regime
     switch (ADX), the stale-feed + no-same-bar-re-entry guards.
   Check the shadow book (`state*/shadow.json`) to value what the filters blocked.

6. **Propose ONE improvement to test** for the identified problem — and
   **backtest it on the deep CRSP daily history before recommending it**
   (`fablebot.backtest.run_backtest` over `data/*.csv`, several regimes/years).
   Report return, **max drawdown**, and win rate vs the unfiltered baseline. Be
   explicit about regime-dependence and evidence limits (15m history is only
   ~60 days; a single full-history path is not the walk-forward OOS gate).

7. **What's already been tried** (so you don't re-suggest a dead end): ADX-gating
   the trend filter *backfired* (it lags → re-admits losers); a short trend MA
   over-filtered; the auto-search engine has converged and passes nothing through
   the gate. Read `docs/PROJECT_OVERVIEW.md` and recent PRs for the full history.

Output a concise report: per-book P&L, the market context, the diagnosed failure
mode, which built tool addresses it, and one concrete backtested improvement to
test next (with a PR if it's warranted).
