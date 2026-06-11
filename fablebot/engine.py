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
from datetime import datetime
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


class TradingEngine:
    def __init__(
        self,
        config: Config,
        state_path: Optional[Path] = Path("state/portfolio.json"),
        fetch: FetchFn = fetch_ohlcv,
        notifier: Optional[Notifier] = None,
        journal_path: Optional[Path] = None,
        shadow_state_path: Optional[Path] = None,
    ):
        self.config = config
        self.fetch = fetch
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
        self.strategies = {
            key: build_strategy(inst.strategy, inst.params)
            for key, inst in config.instruments.items()
        }
        self.last_prices: dict[str, float] = {}

    # -- evaluation ----------------------------------------------------------

    def evaluate_instrument(self, inst: InstrumentConfig, df: pd.DataFrame) -> list[str]:
        """Run one cycle for one instrument against ``df``. Returns log lines."""
        log: list[str] = []
        key = inst.key
        strategy = self.strategies[key]
        if len(df) < strategy.min_bars():
            return [f"{key}: only {len(df)} bars, need {strategy.min_bars()} — skipping"]

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
            log.append(f"{key}: STOPPED OUT {stopped.pnl:+,.2f} ({stopped.close_reason})")
            journal("stopped", stopped.close_reason)
            self.journal.record_trade(stopped)

        # Advance the shadow book here — before entries — so a shadow opened
        # this cycle isn't stop-checked against the bar it entered on (the
        # real book gets the same courtesy by construction).
        self._run_shadow(key, strategy, df, high, low, close, log)

        position = self.broker.positions.get(key)
        signal = strategy.evaluate(df, position.direction if position else None)

        if signal.action is Action.CLOSE and position is not None:
            trade = self.broker.close_position(key, close, signal.reason)
            log.append(f"{key}: closed {trade.direction} {trade.pnl:+,.2f} ({signal.reason})")
            journal("close", signal.reason)
            self.journal.record_trade(trade)
        elif signal.opens and position is None:
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
                    f"{signal.reason}"
                )
                journal(signal.action.value, signal.reason)
        else:
            journal("hold", signal.reason)
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

    def run_cycle(self) -> list[str]:
        """One full pass over all instruments."""
        log: list[str] = []
        for key, inst in self.config.instruments.items():
            try:
                df = self.fetch(inst.symbol, inst.timeframe)
            except Exception as exc:
                log.append(f"{key}: data fetch failed ({exc})")
                continue
            log.extend(self.evaluate_instrument(inst, df))
        return log

    # -- reports -------------------------------------------------------------

    def send_morning_report(self) -> None:
        snapshots = []
        for key, inst in self.config.instruments.items():
            try:
                df = self.fetch(inst.symbol, inst.timeframe)
            except Exception:
                continue
            snapshots.append(build_snapshot(key, inst.name, df, self.strategies[key]))
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
        print(
            f"FableBot running — {len(self.config.instruments)} instruments, "
            f"polling every {self.config.schedule.poll_interval_minutes}m, "
            f"reports at {self.config.schedule.morning_report} / "
            f"{self.config.schedule.evening_report} {self.config.schedule.timezone}",
            flush=True,
        )
        while True:
            now = datetime.now(tz)
            for line in self.run_cycle():
                print(f"[{now:%H:%M}] {line}", flush=True)
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
