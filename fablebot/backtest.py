"""Bar-by-bar backtester.

Replays history through a strategy with the same risk engine the live loop
uses: ATR sizing, the hard stop checked against each bar's high/low, and
fills at the close of the signal bar. Single-instrument by design — it's a
tool for checking that a strategy's logic behaves, not a portfolio
simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .portfolio import PaperBroker, Position
from .risk import size_position
from .strategies.base import Action, Strategy


@dataclass
class BacktestResult:
    trades: int
    wins: int
    losses: int
    total_pnl: float
    return_pct: float
    max_drawdown_pct: float
    equity_curve: list[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    def summary(self) -> str:
        return (
            f"trades={self.trades} win_rate={self.win_rate:.1%} "
            f"pnl={self.total_pnl:+,.2f} return={self.return_pct:+.2f}% "
            f"max_dd={self.max_drawdown_pct:.2f}%"
        )


def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    starting_equity: float = 100_000.0,
    risk_per_trade: float = 0.01,
    instrument: str = "TEST",
    friction_bps: float = 0.0,
) -> BacktestResult:
    """Replay ``df`` through ``strategy`` with the live risk engine.

    ``friction_bps`` models combined commission + slippage: every fill executes
    that many basis points *against* the trade (you buy a touch higher, sell a
    touch lower), so a strategy that only works frictionless cannot pass the
    gate. Position sizing uses the actual (slipped) fill price, so the hard stop
    is still priced off what was really paid. Defaults to 0 — the friction-free
    backtest used by the unit tests; the optimizer/search pass the configured
    cost so the *trial process* is honest.
    """
    broker = PaperBroker(starting_equity, state_path=None)
    equity_curve: list[float] = []
    warmup = strategy.min_bars()
    friction = friction_bps / 1e4

    def _fill(price: float, direction: str, opening: bool) -> float:
        # adverse slippage: opening a long / closing a short pays up; the reverse
        # receives less.
        sign = 1.0 if direction == "long" else -1.0
        return price * (1.0 + (sign if opening else -sign) * friction)

    for i in range(warmup, len(df) + 1):
        window = df.iloc[:i]
        bar = window.iloc[-1]
        close = float(bar["close"])

        broker.enforce_stops(instrument, float(bar["high"]), float(bar["low"]), close)

        position = broker.positions.get(instrument)
        signal = strategy.evaluate(window, position.direction if position else None)
        opened_this_bar = False

        if signal.action is Action.CLOSE and position is not None:
            broker.close_position(instrument, _fill(close, position.direction, opening=False), signal.reason)
        elif signal.opens and position is None and signal.atr > 0:
            entry_fill = _fill(close, signal.direction, opening=True)
            sized = size_position(
                equity=broker.equity,
                risk_per_trade=risk_per_trade,
                entry_price=entry_fill,
                atr=signal.atr,
                atr_stop_mult=strategy.atr_stop_mult,
                direction=signal.direction,
            )
            broker.open_position(
                Position(
                    instrument=instrument,
                    direction=signal.direction,
                    units=sized.units,
                    entry_price=entry_fill,
                    stop_price=sized.stop_price,
                    risk_amount=sized.risk_amount,
                    strategy=strategy.name,
                    reason=signal.reason,
                )
            )
            opened_this_bar = True

        pos = broker.positions.get(instrument)
        if pos is not None and not opened_this_bar:
            trail_mult = strategy.current_trail_mult(window, pos.direction)
            if trail_mult:
                broker.trail_stop(
                    instrument, float(bar["high"]), float(bar["low"]),
                    signal.atr, trail_mult,
                )
        equity_curve.append(broker.total_equity({instrument: close}))

    # liquidate at the final close so the result reflects everything
    if instrument in broker.positions:
        pos = broker.positions[instrument]
        broker.close_position(instrument, _fill(float(df["close"].iloc[-1]), pos.direction, opening=False), "end of backtest")

    wins = sum(1 for t in broker.trades if t.pnl > 0)
    final_equity = broker.equity
    peak, max_dd = -float("inf"), 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak * 100.0)

    return BacktestResult(
        trades=len(broker.trades),
        wins=wins,
        losses=len(broker.trades) - wins,
        total_pnl=broker.realized_pnl,
        return_pct=(final_equity / starting_equity - 1.0) * 100.0,
        max_drawdown_pct=max_dd,
        equity_curve=equity_curve,
    )
