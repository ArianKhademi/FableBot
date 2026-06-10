"""The two daily messages.

7am: what's happening in the market. 9pm: how the bot did.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import pandas as pd

from .portfolio import PaperBroker, Trade
from .strategies.base import Strategy


@dataclass
class MarketSnapshot:
    instrument: str
    name: str
    last_price: float
    change_pct: float  # % change over roughly the last session
    atr_pct: float  # ATR as % of price — the volatility regime
    watching: str  # what the strategy is waiting for


def build_snapshot(
    key: str, name: str, df: pd.DataFrame, strategy: Strategy
) -> MarketSnapshot:
    last = float(df["close"].iloc[-1])
    # ~one session back, capped by available history
    back = min(len(df) - 1, _bars_per_day(df))
    prev = float(df["close"].iloc[-1 - back]) if back > 0 else last
    cur_atr = strategy.current_atr(df)
    return MarketSnapshot(
        instrument=key,
        name=name,
        last_price=last,
        change_pct=(last / prev - 1.0) * 100.0 if prev else 0.0,
        atr_pct=(cur_atr / last) * 100.0 if last else 0.0,
        watching=strategy.watching(df),
    )


def _bars_per_day(df: pd.DataFrame) -> int:
    if len(df) < 2:
        return 1
    step = df.index[-1] - df.index[-2]
    seconds = max(step.total_seconds(), 1.0)
    return max(int(24 * 3600 / seconds), 1)


def _vol_label(atr_pct: float) -> str:
    if atr_pct < 0.3:
        return "quiet"
    if atr_pct < 1.0:
        return "normal"
    return "volatile"


def morning_briefing(
    snapshots: list[MarketSnapshot], broker: PaperBroker, now: Optional[datetime] = None
) -> str:
    now = now or datetime.now()
    lines = [f"Good morning — market check for {now:%A, %B %d}.", ""]
    for snap in snapshots:
        lines.append(
            f"{snap.name} ({snap.instrument}): {snap.last_price:,.2f} "
            f"({snap.change_pct:+.2f}% on the day), "
            f"vol {_vol_label(snap.atr_pct)} (ATR {snap.atr_pct:.2f}% of price)"
        )
        if snap.watching:
            lines.append(f"  watching: {snap.watching}")
    lines.append("")
    if broker.positions:
        lines.append("Open positions:")
        prices = {s.instrument: s.last_price for s in snapshots}
        for key, pos in sorted(broker.positions.items()):
            pnl = pos.unrealized_pnl(prices[key]) if key in prices else float("nan")
            lines.append(
                f"  {key} {pos.direction} {pos.units:.4f} @ {pos.entry_price:,.2f} "
                f"(stop {pos.stop_price:,.2f}, unrealized {pnl:+,.2f})"
            )
    else:
        lines.append("No open positions — all strategies flat, waiting for setups.")
    return "\n".join(lines)


def evening_recap(
    broker: PaperBroker,
    prices: dict[str, float],
    today: Optional[date] = None,
    now: Optional[datetime] = None,
) -> str:
    now = now or datetime.now()
    today = today or now.date()
    todays = [t for t in broker.trades if _trade_date(t) == today]
    wins = [t for t in todays if t.pnl > 0]
    losses = [t for t in todays if t.pnl <= 0]
    realized_today = sum(t.pnl for t in todays)
    total = broker.total_equity(prices)

    lines = [f"Evening recap — {now:%A, %B %d}.", ""]
    if todays:
        lines.append(
            f"Closed {len(todays)} trade(s): {len(wins)} win / {len(losses)} loss, "
            f"realized {realized_today:+,.2f} today."
        )
        for t in todays:
            lines.append(
                f"  {t.instrument} {t.direction}: {t.entry_price:,.2f} -> "
                f"{t.exit_price:,.2f} = {t.pnl:+,.2f} ({t.close_reason})"
            )
    else:
        lines.append("No trades closed today.")
    lines.append("")
    if broker.positions:
        lines.append("Still open:")
        for key, pos in sorted(broker.positions.items()):
            pnl = pos.unrealized_pnl(prices[key]) if key in prices else float("nan")
            lines.append(f"  {key} {pos.direction}: unrealized {pnl:+,.2f}")
    else:
        lines.append("Flat overnight.")
    lines.append("")
    ret_pct = (total / broker.starting_equity - 1.0) * 100.0
    lines.append(
        f"Equity: {total:,.2f} ({ret_pct:+.2f}% since inception, "
        f"realized P&L {broker.realized_pnl:+,.2f})."
    )
    return "\n".join(lines)


def _trade_date(trade: Trade) -> date:
    return datetime.fromisoformat(trade.closed_at).date()
