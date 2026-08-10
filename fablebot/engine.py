"""The trading engine: one evaluation cycle per instrument, plus the daily
scheduler that fires the 7am briefing and 9pm recap.

The cycle for each instrument:
  1. fetch bars at the instrument's timeframe
  2. enforce the hard stop against the latest bar
  3. ask the strategy for a signal
  4. entries pass through the correlation filter, then get ATR-sized
  5. the paper broker records the fill

Everything the engine decides — including HOLDs and signals the correlation
filter blocked — is written to the decision journal. Blocked entries also
open *shadow positions*: paper-paper trades tracked with the same stops and
exits but no effect on equity, so the learning loop can measure what the
filter actually saved or cost.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from .config import Config, InstrumentConfig
from .data import fetch_ohlcv
from .journal import Journal
from .notify import Notifier
from .portfolio import PaperBroker, Position
from .reports import build_snapshot, evening_recap, morning_briefing
from .risk import CorrelationFilter, size_position
from .strategies import build_strategy
from .strategies.base import Action

FetchFn = Callable[[str, str], pd.DataFrame]


# A live feed returns the SAME last bar on every poll when the market is closed
# (overnight, weekends, holidays). Acting on a frozen bar is meaningless — and for
# mean-reversion it's destructive: the one stale "oversold" bar keeps firing an
# entry whose stop the same bar's low has already breached. The no-same-bar-
# re-entry guard below stops the *loop* outright; this staleness check is the
# second layer that idles the instrument entirely once the feed is clearly frozen.
# Each threshold is just over one bar — long enough that the in-progress bar (or a
# single missed poll) never reads as stale, short enough to go quiet within minutes
# of the close. Generous for daily so weekends/holidays aren't mistaken for a dead
# feed.
_STALE_AFTER = {
    "15m": timedelta(minutes=40),
    "1h": timedelta(minutes=90),
    "4h": timedelta(hours=6),
    "1d": timedelta(days=4),
}
_STALE_AFTER_DEFAULT = timedelta(minutes=90)


def feed_is_stale(last_ts, timeframe: str, now=None) -> bool:
    """True when the most recent bar is too old to trade on. Only meaningful for a
    real-time, timezone-aware feed; callers skip the check for tz-naive frames
    (synthetic test data, CSV/backtest history), which aren't a live clock."""
    last_ts = pd.Timestamp(last_ts)
    if now is None:
        now = pd.Timestamp.now(tz=last_ts.tz) if last_ts.tzinfo is not None else pd.Timestamp.now()
    return (now - last_ts) > _STALE_AFTER.get(timeframe, _STALE_AFTER_DEFAULT)


class TradingEngine:
    def __init__(
        self,
        config: Config,
        state_path: Optional[Path] = Path("state/portfolio.json"),
        fetch: FetchFn = fetch_ohlcv,
        notifier: Optional[Notifier] = None,
        journal_path: Optional[Path] = None,
        shadow_state_path: Optional[Path] = None,
        history_fetch: Optional[FetchFn] = None,
    ):
        self.config = config
        self.fetch = fetch
        # Deep-history fetch used to train model-backed instruments; falls back
        # to the live feed if not supplied (which may be too shallow to train —
        # then the instrument falls back to its named strategy).
        self.history_fetch = history_fetch or fetch
        self.notifier = notifier or Notifier()
        self.broker = PaperBroker(config.account.starting_equity, state_path)
        # The shadow book holds trades the correlation filter blocked: same
        # sizing, stops, and exits, but no effect on equity. Counterfactuals.
        self.shadow = PaperBroker(0.0, shadow_state_path)
        self.journal = Journal(journal_path)
        self.correlation = CorrelationFilter(
            config.correlation.groups,
            config.correlation.max_same_direction_per_group,
        )
        # Prop-firm rule simulation (futures books): trailing-drawdown floor,
        # profit target, contract cap. Persisted next to the portfolio so a
        # blown account stays blown across restarts.
        self.prop = None
        if config.prop is not None:
            from .prop_rules import PropAccount
            prop_path = Path(state_path).parent / "prop.json" if state_path else None
            self.prop = PropAccount(
                starting_equity=config.account.starting_equity,
                max_drawdown=config.prop.max_drawdown,
                drawdown_mode=config.prop.drawdown_mode,
                profit_target=config.prop.profit_target,
                consistency_pct=config.prop.consistency_pct,
                buffer=config.prop.buffer,
                max_contracts=config.prop.max_contracts,
                state_path=prop_path,
            )
        self._prop_day: Optional[str] = None
        self._prop_last_equity: float = config.account.starting_equity
        self.strategies = self._build_strategies()
        self.last_prices: dict[str, float] = {}
        # (bar timestamp, direction) of the last exit per instrument — used to
        # refuse re-opening the same position on the very bar it just exited on
        # (the stop-out/re-enter loop). Cleared naturally as the bar advances.
        self._last_exit: dict[str, tuple] = {}
        # Bar timestamp each open position was ENTERED on. While the live loop
        # is still polling that same (forming) bar, stops/excursions must only
        # see post-entry prices (the close), never the bar's pre-entry range —
        # a pre-entry spike low must not fire the stop of a position that
        # didn't exist yet.
        self._entry_bar: dict[str, object] = {}
        # Deferred trailing: (candidate_stop, bar_ts) computed from a bar's
        # extreme. Applied only once a LATER bar arrives, so a stop derived
        # from bar B's high is never tested against bar B's own low (which may
        # have printed before the stop existed). Mirrors backtest semantics.
        self._pending_trail: dict[str, tuple] = {}
        # Shadow-book twins of the guards above — without them the shadow
        # ledger churns stop/re-open every poll on a blocked signal and the
        # counterfactual "what the filter saved" is graded on garbage.
        self._shadow_last_exit: dict[str, tuple] = {}
        self._shadow_entry_bar: dict[str, object] = {}
        self._shadow_pending_trail: dict[str, tuple] = {}
        # Daily circuit breaker: (session date, direction) -> count of LOSING
        # hard stops across the whole book. Keyed by the schedule-timezone
        # session date (a bar-date key would live in per-feed timezones and
        # reset mid-session for 24/7 symbols). Rebuilt from the journal at
        # startup so a restart can't silently re-arm a tripped brake.
        self._daily_stops: dict[tuple, int] = {}
        self._rebuild_daily_stops()

    def _session_date(self) -> str:
        return datetime.now(ZoneInfo(self.config.schedule.timezone)).date().isoformat()

    def _rebuild_daily_stops(self) -> None:
        """Re-count today's losing stop exits from the journal, so a mid-day
        restart resumes the circuit breaker where it left off instead of
        granting a fresh allowance during exactly the cascade it tripped on."""
        if self.config.account.max_daily_stops is None:
            return
        try:
            tz = ZoneInfo(self.config.schedule.timezone)
            today = datetime.now(tz).date()
            for entry in self.journal.read():
                if entry.get("event") != "trade_closed" or entry.get("shadow"):
                    continue
                trade = entry.get("trade", {})
                if trade.get("pnl", 0) >= 0 or "stop hit" not in trade.get("close_reason", ""):
                    continue
                try:
                    closed = datetime.fromisoformat(trade.get("closed_at", ""))
                except ValueError:
                    continue
                if closed.tzinfo is not None and closed.astimezone(tz).date() == today:
                    key = (today.isoformat(), trade.get("direction"))
                    self._daily_stops[key] = self._daily_stops.get(key, 0) + 1
        except Exception as exc:  # a corrupt journal must not stop the bot
            print(f"[breaker] journal rebuild skipped ({type(exc).__name__}: {exc})", flush=True)

    def _build_one_strategy(self, inst: InstrumentConfig):
        """Build the strategy for one instrument. A `model` spec trains a live
        ML/RL model on deep history; on insufficient data it falls back to the
        named strategy/genome so the bot never goes dark."""
        model_cfg = getattr(inst, "model", None)
        if model_cfg:
            from .live_models import build_model_strategy
            try:
                df = self.history_fetch(inst.symbol, inst.timeframe)
                strat = build_model_strategy(model_cfg, df)
            except Exception as exc:
                print(f"[engine] {inst.key}: model fetch/train failed ({exc})", flush=True)
                strat = None
            if strat is not None:
                print(f"[engine] {inst.key}: trained live model '{model_cfg.get('type')}'", flush=True)
                return strat
            print(f"[engine] {inst.key}: model not ready — falling back to '{inst.strategy}'", flush=True)
        return build_strategy(inst.strategy, inst.params, genome=getattr(inst, "genome", None))

    def _build_strategies(self) -> dict:
        return {key: self._build_one_strategy(inst) for key, inst in self.config.instruments.items()}

    def retrain_models(self) -> None:
        """Re-train every model-backed instrument on fresh history. No-op when
        no instrument uses a live model."""
        for key, inst in self.config.instruments.items():
            if getattr(inst, "model", None):
                self.strategies[key] = self._build_one_strategy(inst)

    # -- evaluation ----------------------------------------------------------

    def evaluate_instrument(self, inst: InstrumentConfig, df: pd.DataFrame) -> list[str]:
        """Run one cycle for one instrument against ``df``. Returns log lines."""
        log: list[str] = []
        key = inst.key
        strategy = self.strategies[key]
        if len(df) < strategy.min_bars():
            return [f"{key}: only {len(df)} bars, need {strategy.min_bars()} — skipping"]

        # Don't act on a frozen feed (market closed): a stale bar drives a
        # degenerate stop-out/re-enter loop. Only real-time tz-aware feeds are
        # checked; tz-naive frames (tests, CSV history) are never "stale".
        idx = df.index
        if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None and feed_is_stale(idx[-1], inst.timeframe):
            return [f"{key}: feed stale (last bar {idx[-1]}) — market closed, holding"]

        bar = df.iloc[-1]
        high, low, close = float(bar["high"]), float(bar["low"]), float(bar["close"])
        self.last_prices[key] = close
        bar_ts = idx[-1] if isinstance(idx, pd.DatetimeIndex) else None
        context = strategy.context(df)

        def journal(action: str, reason: str, shadow: bool = False) -> None:
            position = self.broker.positions.get(key)
            self.journal.record_decision(
                instrument=key,
                action=action,
                reason=reason,
                price=close,
                atr=context.get("atr", float("nan")),
                context=context,
                position_direction=position.direction if position else None,
                equity=self.broker.equity,
                shadow=shadow,
            )

        bar_open = float(bar["open"])
        # While still polling the position's own ENTRY bar, the bar's high/low
        # include pre-entry range the trade never lived through — stop-check
        # and track excursions against the current price (close) only.
        on_entry_bar = bar_ts is not None and self._entry_bar.get(key) == bar_ts
        eff_high, eff_low = (close, close) if on_entry_bar else (high, low)

        # Apply a deferred trail candidate once its source bar has passed —
        # from now on this bar's range legitimately tests the tightened stop.
        pending = self._pending_trail.get(key)
        if pending is not None and bar_ts is not None and pending[1] != bar_ts:
            del self._pending_trail[key]
            if self.broker.tighten_stop(key, pending[0]):
                log.append(f"{key}: stop trailed to {pending[0]:,.2f}" + self._acct())
                journal("trail", f"stop trailed to {pending[0]:,.2f}")

        self.broker.update_excursions(key, eff_high, eff_low)
        stopped = self.broker.enforce_stops(key, eff_high, eff_low, close, open_=None if on_entry_bar else bar_open)
        if stopped is not None:
            # Guard bookkeeping FIRST — if journaling raises (disk full), the
            # same-bar guard and breaker must already be armed.
            self._entry_bar.pop(key, None)
            self._pending_trail.pop(key, None)
            if bar_ts is not None:
                self._last_exit[key] = (bar_ts, stopped.direction)
                if stopped.pnl < 0:
                    # Only LOSING stops count toward the breaker — a trailed
                    # stop that locks profit is a win, not cascade evidence.
                    day_key = (self._session_date(), stopped.direction)
                    self._daily_stops[day_key] = self._daily_stops.get(day_key, 0) + 1
            log.append(f"{key}: STOPPED OUT {stopped.pnl:+,.2f} ({stopped.close_reason})" + self._acct())
            journal("stopped", stopped.close_reason)
            self.journal.record_trade(stopped)

        # Advance the shadow book here — before entries — so a shadow opened
        # this cycle isn't stop-checked against the bar it entered on (the
        # real book gets the same courtesy by construction).
        self._run_shadow(key, strategy, df, high, low, close, log, bar_ts=bar_ts, bar_open=bar_open)

        position = self.broker.positions.get(key)
        signal = strategy.evaluate(df, position.direction if position else None)
        opened_this_cycle = False
        # Refuse to re-open the position we just exited on this very bar (the
        # stale/forming-bar stop-out -> re-enter loop). A new bar clears it.
        same_bar_reentry = (signal.opens and position is None and signal.atr > 0
                            and self._last_exit.get(key) == (bar_ts, signal.direction))

        if signal.action is Action.CLOSE and position is not None:
            trade = self.broker.close_position(key, close, signal.reason)
            self._entry_bar.pop(key, None)
            self._pending_trail.pop(key, None)
            if bar_ts is not None:
                self._last_exit[key] = (bar_ts, trade.direction)
            log.append(f"{key}: closed {trade.direction} {trade.pnl:+,.2f} ({signal.reason})" + self._acct())
            journal("close", signal.reason)
            self.journal.record_trade(trade)
        elif same_bar_reentry:
            log.append(f"{key}: {signal.direction} signal — just exited this bar, holding (no same-bar re-entry)")
            journal("blocked", f"{signal.reason} | no same-bar re-entry")
        elif signal.opens and position is None and signal.atr > 0:
            direction = signal.direction
            allowed, why = self.correlation.allows(
                key, direction, self.broker.open_directions()
            )
            # Futures books size in whole contracts, within the prop account's
            # remaining contract capacity.
            cap = None
            if inst.contract_multiplier and self.prop is not None and self.prop.max_contracts:
                cap = max(self.prop.max_contracts - self._open_contracts(), 0)
            sized = size_position(
                equity=self.broker.equity,
                risk_per_trade=self.config.account.risk_per_trade,
                entry_price=close,
                atr=signal.atr,
                atr_stop_mult=strategy.atr_stop_mult,
                direction=direction,
                contract_multiplier=inst.contract_multiplier,
                max_contracts=cap,
            )
            # Daily circuit breaker: after max_daily_stops same-direction hard
            # stops today, the book stops fighting that tape until tomorrow.
            # Blocked entries still go to the shadow book so the brake's cost
            # (blocked recoveries) stays measurable, like the correlation filter.
            limit = self.config.account.max_daily_stops
            tripped = (
                limit is not None
                and self._daily_stops.get((self._session_date(), direction), 0) >= limit
            )
            if self.prop is not None and self.prop.blown:
                log.append(f"{key}: {direction} signal blocked — prop account blown, no new entries")
                journal("blocked", f"{signal.reason} | prop account blown")
            elif inst.contract_multiplier and sized.contracts == 0:
                log.append(
                    f"{key}: {direction} signal skipped — one contract "
                    f"(${signal.atr * strategy.atr_stop_mult * inst.contract_multiplier:,.0f} risk) "
                    f"exceeds the risk budget or contract cap"
                )
                journal("blocked", f"{signal.reason} | below one-contract size")
            elif tripped:
                why_cb = f"circuit breaker: {limit} {direction} hard stops today — no more {direction} entries"
                log.append(f"{key}: {direction} signal blocked — {why_cb}")
                journal("blocked", f"{signal.reason} | {why_cb}")
                self._open_shadow(key, direction, close, sized, strategy.name, signal.reason, bar_ts)
            elif not allowed:
                log.append(f"{key}: {direction} signal blocked — {why}")
                journal("blocked", f"{signal.reason} | {why}")
                self._open_shadow(key, direction, close, sized, strategy.name, signal.reason, bar_ts)
            else:
                self.broker.open_position(
                    Position(
                        instrument=key,
                        direction=direction,
                        units=sized.units,
                        entry_price=close,
                        stop_price=sized.stop_price,
                        risk_amount=sized.risk_amount,
                        strategy=strategy.name,
                        reason=signal.reason,
                    )
                )
                if bar_ts is not None:
                    self._entry_bar[key] = bar_ts
                size_str = (f"{sized.contracts} contract(s)" if inst.contract_multiplier
                            else f"{sized.units:.4f}")
                log.append(
                    f"{key}: opened {direction} {size_str} @ {close:,.2f} "
                    f"(stop {sized.stop_price:,.2f}, risking {sized.risk_amount:,.2f}) — "
                    f"{signal.reason}" + self._acct()
                )
                journal(signal.action.value, signal.reason)
                opened_this_cycle = True
        else:
            journal("hold", signal.reason)

        # Trailing stop, DEFERRED: compute this bar's chandelier candidate from
        # its (post-entry) extreme, but only ARM it once the next bar arrives —
        # a stop derived from this bar's high must never be tested against this
        # same bar's low, which may have printed before the stop existed. The
        # entry bar itself never trails (its extremes predate the position).
        position = self.broker.positions.get(key)
        if position is not None and not opened_this_cycle and bar_ts is not None and not on_entry_bar:
            trail_mult = strategy.current_trail_mult(df, position.direction)
            if trail_mult and signal.atr > 0:
                extreme = high if position.direction == "long" else low
                offset = signal.atr * trail_mult
                candidate = extreme - offset if position.direction == "long" else extreme + offset
                tightens = (candidate > position.stop_price) if position.direction == "long" \
                    else (candidate < position.stop_price)
                if tightens:
                    prev = self._pending_trail.get(key)
                    if prev is None or prev[1] != bar_ts:
                        self._pending_trail[key] = (candidate, bar_ts)
                    else:
                        better = max(prev[0], candidate) if position.direction == "long" \
                            else min(prev[0], candidate)
                        self._pending_trail[key] = (better, bar_ts)
        return log

    # -- shadow book ----------------------------------------------------------

    def _open_shadow(
        self, key: str, direction: str, close: float, sized, strategy_name: str, reason: str,
        bar_ts=None,
    ) -> None:
        if key in self.shadow.positions:
            return
        # Same-bar guard, shadow edition: without it a blocked-but-still-firing
        # signal churns one shadow stop/re-open per poll and the counterfactual
        # ledger fills with phantom round-trips the real book could never take.
        if bar_ts is not None and self._shadow_last_exit.get(key) == (bar_ts, direction):
            return
        self.shadow.open_position(
            Position(
                instrument=key,
                direction=direction,
                units=sized.units,
                entry_price=close,
                stop_price=sized.stop_price,
                risk_amount=sized.risk_amount,
                strategy=strategy_name,
                reason=f"[shadow] {reason}",
            )
        )
        if bar_ts is not None:
            self._shadow_entry_bar[key] = bar_ts

    def _run_shadow(
        self, key: str, strategy, df, high: float, low: float, close: float, log: list[str],
        bar_ts=None, bar_open: float = None,
    ) -> None:
        """Advance the shadow position for this instrument, if any, with the
        same stop/exit/entry-bar/deferred-trail semantics as the real book —
        anything less and the counterfactual is graded on different physics."""
        if key not in self.shadow.positions:
            return
        on_entry_bar = bar_ts is not None and self._shadow_entry_bar.get(key) == bar_ts
        eff_high, eff_low = (close, close) if on_entry_bar else (high, low)

        pending = self._shadow_pending_trail.get(key)
        if pending is not None and bar_ts is not None and pending[1] != bar_ts:
            del self._shadow_pending_trail[key]
            self.shadow.tighten_stop(key, pending[0])

        self.shadow.update_excursions(key, eff_high, eff_low)
        stopped = self.shadow.enforce_stops(key, eff_high, eff_low, close,
                                            open_=None if on_entry_bar else bar_open)
        if stopped is not None:
            self._shadow_entry_bar.pop(key, None)
            self._shadow_pending_trail.pop(key, None)
            if bar_ts is not None:
                self._shadow_last_exit[key] = (bar_ts, stopped.direction)
            log.append(f"{key}: shadow trade stopped {stopped.pnl:+,.2f} (would-have-been)")
            self.journal.record_trade(stopped, shadow=True)
            return
        direction = self.shadow.positions[key].direction
        signal = strategy.evaluate(df, direction)
        if signal.action is Action.CLOSE:
            trade = self.shadow.close_position(key, close, signal.reason)
            self._shadow_entry_bar.pop(key, None)
            self._shadow_pending_trail.pop(key, None)
            if bar_ts is not None:
                self._shadow_last_exit[key] = (bar_ts, trade.direction)
            log.append(f"{key}: shadow trade closed {trade.pnl:+,.2f} (would-have-been)")
            self.journal.record_trade(trade, shadow=True)
        elif bar_ts is not None and not on_entry_bar:
            # Deferred trail, same rule as the real book.
            trail_mult = strategy.current_trail_mult(df, direction)
            if trail_mult and signal.atr > 0:
                pos = self.shadow.positions[key]
                extreme = high if direction == "long" else low
                offset = signal.atr * trail_mult
                candidate = extreme - offset if direction == "long" else extreme + offset
                tightens = (candidate > pos.stop_price) if direction == "long" else (candidate < pos.stop_price)
                if tightens:
                    prev = self._shadow_pending_trail.get(key)
                    if prev is None or prev[1] != bar_ts:
                        self._shadow_pending_trail[key] = (candidate, bar_ts)
                    else:
                        better = max(prev[0], candidate) if direction == "long" else min(prev[0], candidate)
                        self._shadow_pending_trail[key] = (better, bar_ts)

    def _acct(self) -> str:
        """Compact account state, appended to trade-event log lines so the
        terminal shows equity + live unrealized at each open/close/trail."""
        total = self.broker.total_equity(self.last_prices)
        unreal = total - self.broker.equity
        return f" | acct ${total:,.0f} (real {self.broker.realized_pnl:+,.0f}, unreal {unreal:+,.0f})"

    def _portfolio_state(self) -> dict:
        """The broker's current state as a portfolio dict — built in-memory (not
        read from disk), so it's always current and works for a brand-new book
        whose portfolio.json doesn't exist yet."""
        from dataclasses import asdict
        return {
            "starting_equity": self.broker.starting_equity,
            "realized_pnl": self.broker.realized_pnl,
            "positions": {k: asdict(p) for k, p in self.broker.positions.items()},
            "trades": [asdict(t) for t in self.broker.trades],
        }

    def _write_dashboard(self) -> None:
        """Regenerate the read-only dashboard from current state, reusing the
        marks this cycle already computed (no extra fetch). Best-effort and fully
        guarded — a dashboard failure must never disturb the trading loop."""
        path = self.broker.state_path
        if path is None:
            return
        try:
            from . import dashboard
            snapshot = dashboard.build_snapshot(self._portfolio_state(), self.last_prices)
            engine_state = dashboard.read_engine_state(Path(path).parent)
            ts = datetime.now(ZoneInfo(self.config.schedule.timezone)).strftime("%Y-%m-%d %H:%M")
            (Path(path).parent / "dashboard.html").write_text(
                dashboard.render_html(snapshot, engine_state, ts, title=self.config.name)
            )
        except Exception as exc:
            print(f"[dashboard] regen skipped ({type(exc).__name__}: {exc})", flush=True)

    def _open_contracts(self) -> int:
        """Whole (micro) contracts currently open across the book — units carry
        contracts x multiplier, so divide back out per instrument."""
        total = 0
        for key, pos in self.broker.positions.items():
            inst = self.config.instruments.get(key)
            mult = getattr(inst, "contract_multiplier", None) if inst else None
            if mult:
                total += int(round(pos.units / mult))
        return total

    def run_cycle(self) -> list[str]:
        """One full pass over all instruments."""
        log: list[str] = []
        for key, inst in self.config.instruments.items():
            try:
                df = self.fetch(inst.symbol, inst.timeframe)
            except Exception as exc:
                log.append(f"{key}: data fetch failed ({exc})")
                continue
            try:
                log.extend(self.evaluate_instrument(inst, df))
            except Exception as exc:
                # One instrument's failure must never take down the live loop.
                log.append(f"{key}: evaluation error ({type(exc).__name__}: {exc}) — skipped")
        # Prop-rule bookkeeping: feed the marked equity each cycle; EOD rolls
        # are handled when the session date changes (see run_forever).
        if self.prop is not None:
            today = self._session_date()
            if getattr(self, "_prop_day", None) is None:
                self._prop_day = today
            elif self._prop_day != today:
                note = self.prop.end_of_day(self._prop_last_equity, self._prop_day)
                if note:
                    log.append(note)
                self._prop_day = today
            equity = self.broker.total_equity(self.last_prices)
            self._prop_last_equity = equity
            note = self.prop.update(equity, today)
            if note:
                log.append(note)
        return log

    # -- reports -------------------------------------------------------------

    def send_morning_report(self) -> None:
        snapshots = []
        for key, inst in self.config.instruments.items():
            try:
                df = self.fetch(inst.symbol, inst.timeframe)
            except Exception:
                continue
            snapshots.append(build_snapshot(key, inst.name, inst.symbol, df, self.strategies[key]))
            self.last_prices[key] = float(df["close"].iloc[-1])
        self.notifier.send("Morning briefing", morning_briefing(snapshots, self.broker))

    def send_evening_report(self) -> None:
        self.notifier.send(
            "Evening recap", evening_recap(self.broker, self.last_prices)
        )

    # -- main loop -----------------------------------------------------------

    def run_forever(self) -> None:
        tz = ZoneInfo(self.config.schedule.timezone)
        interval = self.config.schedule.poll_interval_minutes * 60
        sent_on: dict[str, Optional[str]] = {"morning": None, "evening": None}
        has_models = any(getattr(i, "model", None) for i in self.config.instruments.values())
        retrained_on: Optional[str] = datetime.now(tz).date().isoformat()  # trained at init
        print(
            f"FableBot running — {len(self.config.instruments)} instruments, "
            f"polling every {self.config.schedule.poll_interval_minutes}m, "
            f"reports at {self.config.schedule.morning_report} / "
            f"{self.config.schedule.evening_report} {self.config.schedule.timezone}",
            flush=True,
        )
        while True:
            now = datetime.now(tz)
            # Re-train live models once a day so they track the market.
            today = now.date().isoformat()
            if has_models and retrained_on != today:
                print(f"[{now:%H:%M}] retraining live models for {today}", flush=True)
                try:
                    self.retrain_models()
                except Exception as exc:
                    print(f"[retrain] failed, keeping current models ({type(exc).__name__}: {exc})", flush=True)
                retrained_on = today
            for line in self.run_cycle():
                print(f"[{now:%H:%M}] {line}", flush=True)
            self._write_dashboard()  # refresh the read-only dashboard from this cycle's marks
            # Reports must never kill the trading loop (disk-full archive write,
            # webhook hiccup, a bad snapshot fetch) — same rule as the dashboard.
            for which, at in (("morning", self.config.schedule.morning_report),
                              ("evening", self.config.schedule.evening_report)):
                try:
                    self._maybe_report(now, which, at, sent_on)
                except Exception as exc:
                    print(f"[report] {which} failed ({type(exc).__name__}: {exc})", flush=True)
            time.sleep(interval)

    def _maybe_report(
        self, now: datetime, which: str, at: str, sent_on: dict[str, Optional[str]]
    ) -> None:
        today = now.date().isoformat()
        if sent_on[which] == today or now.strftime("%H:%M") < at:
            return
        sent_on[which] = today
        if which == "morning":
            self.send_morning_report()
        else:
            self.send_evening_report()
