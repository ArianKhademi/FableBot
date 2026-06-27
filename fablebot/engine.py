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
# entry whose stop the same bar's low has already breached, so the loop opens and
# stops out every cycle, bleeding the risk budget each time. Treat a bar older than
# this (per timeframe) as stale and idle until a fresh one arrives. Generous for
# daily so weekends/holidays aren't mistaken for a dead feed.
_STALE_AFTER = {
    "15m": timedelta(hours=1),
    "1h": timedelta(hours=3),
    "4h": timedelta(hours=10),
    "1d": timedelta(days=4),
}
_STALE_AFTER_DEFAULT = timedelta(hours=3)


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
        self.strategies = self._build_strategies()
        self.last_prices: dict[str, float] = {}

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

        self.broker.update_excursions(key, high, low)
        stopped = self.broker.enforce_stops(key, high, low, close)
        if stopped is not None:
            log.append(f"{key}: STOPPED OUT {stopped.pnl:+,.2f} ({stopped.close_reason})" + self._acct())
            journal("stopped", stopped.close_reason)
            self.journal.record_trade(stopped)

        # Advance the shadow book here — before entries — so a shadow opened
        # this cycle isn't stop-checked against the bar it entered on (the
        # real book gets the same courtesy by construction).
        self._run_shadow(key, strategy, df, high, low, close, log)

        position = self.broker.positions.get(key)
        signal = strategy.evaluate(df, position.direction if position else None)
        opened_this_cycle = False

        if signal.action is Action.CLOSE and position is not None:
            trade = self.broker.close_position(key, close, signal.reason)
            log.append(f"{key}: closed {trade.direction} {trade.pnl:+,.2f} ({signal.reason})" + self._acct())
            journal("close", signal.reason)
            self.journal.record_trade(trade)
        elif signal.opens and position is None and signal.atr > 0:
            direction = signal.direction
            allowed, why = self.correlation.allows(
                key, direction, self.broker.open_directions()
            )
            sized = size_position(
                equity=self.broker.equity,
                risk_per_trade=self.config.account.risk_per_trade,
                entry_price=close,
                atr=signal.atr,
                atr_stop_mult=strategy.atr_stop_mult,
                direction=direction,
            )
            if not allowed:
                log.append(f"{key}: {direction} signal blocked — {why}")
                journal("blocked", f"{signal.reason} | {why}")
                self._open_shadow(key, direction, close, sized, strategy.name, signal.reason)
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
                log.append(
                    f"{key}: opened {direction} {sized.units:.4f} @ {close:,.2f} "
                    f"(stop {sized.stop_price:,.2f}, risking {sized.risk_amount:,.2f}) — "
                    f"{signal.reason}" + self._acct()
                )
                journal(signal.action.value, signal.reason)
                opened_this_cycle = True
        else:
            journal("hold", signal.reason)

        # Trailing stop: ratchet toward price on every bar after the entry
        # bar (trailing against the entry bar itself would re-tighten the
        # freshly placed stop). Only ever tightens — see Position.trail.
        position = self.broker.positions.get(key)
        if position is not None and not opened_this_cycle:
            trail_mult = strategy.current_trail_mult(df, position.direction)
            if trail_mult and self.broker.trail_stop(key, high, low, signal.atr, trail_mult):
                new_stop = self.broker.positions[key].stop_price
                log.append(f"{key}: stop trailed to {new_stop:,.2f}" + self._acct())
                journal("trail", f"stop trailed to {new_stop:,.2f}")
        return log

    # -- shadow book ----------------------------------------------------------

    def _open_shadow(
        self, key: str, direction: str, close: float, sized, strategy_name: str, reason: str
    ) -> None:
        if key in self.shadow.positions:
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

    def _run_shadow(
        self, key: str, strategy, df, high: float, low: float, close: float, log: list[str]
    ) -> None:
        """Advance the shadow position for this instrument, if any, with the
        same stop and exit logic the real book uses."""
        if key not in self.shadow.positions:
            return
        self.shadow.update_excursions(key, high, low)
        stopped = self.shadow.enforce_stops(key, high, low, close)
        if stopped is not None:
            log.append(f"{key}: shadow trade stopped {stopped.pnl:+,.2f} (would-have-been)")
            self.journal.record_trade(stopped, shadow=True)
            return
        direction = self.shadow.positions[key].direction
        signal = strategy.evaluate(df, direction)
        if signal.action is Action.CLOSE:
            trade = self.shadow.close_position(key, close, signal.reason)
            log.append(f"{key}: shadow trade closed {trade.pnl:+,.2f} (would-have-been)")
            self.journal.record_trade(trade, shadow=True)
        else:
            # Shadow trades trail too (with the same dynamic multiple), or the
            # counterfactual would flatter whichever book didn't.
            trail_mult = strategy.current_trail_mult(df, direction)
            if trail_mult:
                self.shadow.trail_stop(key, high, low, signal.atr, trail_mult)

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
                self.retrain_models()
                retrained_on = today
            for line in self.run_cycle():
                print(f"[{now:%H:%M}] {line}", flush=True)
            self._write_dashboard()  # refresh the read-only dashboard from this cycle's marks
            self._maybe_report(now, "morning", self.config.schedule.morning_report, sent_on)
            self._maybe_report(now, "evening", self.config.schedule.evening_report, sent_on)
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
