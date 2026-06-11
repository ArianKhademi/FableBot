"""Positions, trade log, and the paper broker.

The paper broker simulates fills at the prices the engine hands it and
enforces the hard stop two ways:

1. the ATR-placed stop price is checked against every bar's high/low;
2. independently, any position whose unrealized loss reaches its risk budget
   is force-closed at market — so a gap straight through the stop still gets
   cut, no exceptions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Position:
    instrument: str
    direction: str  # 'long' | 'short'
    units: float
    entry_price: float
    stop_price: float
    risk_amount: float
    strategy: str
    reason: str
    opened_at: str = field(default_factory=_utcnow)
    # Excursion tracking (dollars): the best and worst the trade looked while
    # open. Feeds the learning loop's stop-quality analysis.
    max_favorable_pnl: float = 0.0
    max_adverse_pnl: float = 0.0

    def unrealized_pnl(self, price: float) -> float:
        sign = 1.0 if self.direction == "long" else -1.0
        return (price - self.entry_price) * self.units * sign

    def update_excursions(self, high: float, low: float) -> None:
        best = high if self.direction == "long" else low
        worst = low if self.direction == "long" else high
        self.max_favorable_pnl = max(self.max_favorable_pnl, self.unrealized_pnl(best))
        self.max_adverse_pnl = min(self.max_adverse_pnl, self.unrealized_pnl(worst))

    def stop_hit(self, high: float, low: float) -> bool:
        if self.direction == "long":
            return low <= self.stop_price
        return high >= self.stop_price


@dataclass
class Trade:
    instrument: str
    direction: str
    units: float
    entry_price: float
    exit_price: float
    pnl: float
    strategy: str
    open_reason: str
    close_reason: str
    opened_at: str
    closed_at: str = field(default_factory=_utcnow)
    risk_amount: float = 0.0
    max_favorable_pnl: float = 0.0
    max_adverse_pnl: float = 0.0


class Broker:
    """Execution interface. Implement this to wire up a real broker."""

    def open_position(self, position: Position) -> None:
        raise NotImplementedError

    def close_position(self, instrument: str, price: float, reason: str) -> Trade:
        raise NotImplementedError


class PaperBroker(Broker):
    def __init__(self, starting_equity: float, state_path: Optional[Path] = None):
        self.starting_equity = starting_equity
        self.realized_pnl = 0.0
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.state_path = Path(state_path) if state_path else None
        if self.state_path and self.state_path.exists():
            self._load()

    # -- accounting ---------------------------------------------------------

    @property
    def equity(self) -> float:
        return self.starting_equity + self.realized_pnl

    def total_equity(self, prices: dict[str, float]) -> float:
        unrealized = sum(
            pos.unrealized_pnl(prices[key])
            for key, pos in self.positions.items()
            if key in prices
        )
        return self.equity + unrealized

    def open_directions(self) -> dict[str, str]:
        return {key: pos.direction for key, pos in self.positions.items()}

    # -- execution ----------------------------------------------------------

    def open_position(self, position: Position) -> None:
        if position.instrument in self.positions:
            raise ValueError(f"{position.instrument} already has an open position")
        self.positions[position.instrument] = position
        self._save()

    def close_position(self, instrument: str, price: float, reason: str) -> Trade:
        pos = self.positions.pop(instrument)
        pnl = pos.unrealized_pnl(price)
        trade = Trade(
            instrument=pos.instrument,
            direction=pos.direction,
            units=pos.units,
            entry_price=pos.entry_price,
            exit_price=price,
            pnl=pnl,
            strategy=pos.strategy,
            open_reason=pos.reason,
            close_reason=reason,
            opened_at=pos.opened_at,
            risk_amount=pos.risk_amount,
            max_favorable_pnl=max(pos.max_favorable_pnl, pnl),
            max_adverse_pnl=min(pos.max_adverse_pnl, pnl),
        )
        self.realized_pnl += pnl
        self.trades.append(trade)
        self._save()
        return trade

    def update_excursions(self, instrument: str, high: float, low: float) -> None:
        pos = self.positions.get(instrument)
        if pos is not None:
            pos.update_excursions(high, low)
            self._save()

    def enforce_stops(self, instrument: str, high: float, low: float, close: float) -> Optional[Trade]:
        """The hard stop, checked every cycle. Returns the closing trade if
        the position was cut."""
        pos = self.positions.get(instrument)
        if pos is None:
            return None
        if pos.stop_hit(high, low):
            return self.close_position(
                instrument, pos.stop_price, f"hard stop hit at {pos.stop_price:,.2f}"
            )
        # Gap protection: loss already past the risk budget at the close.
        if pos.unrealized_pnl(close) <= -pos.risk_amount:
            return self.close_position(
                instrument, close, "hard stop (gap): loss reached risk budget"
            )
        return None

    # -- persistence --------------------------------------------------------

    def _save(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "starting_equity": self.starting_equity,
            "realized_pnl": self.realized_pnl,
            "positions": {k: asdict(p) for k, p in self.positions.items()},
            "trades": [asdict(t) for t in self.trades],
        }
        self.state_path.write_text(json.dumps(state, indent=2))

    def _load(self) -> None:
        state = json.loads(self.state_path.read_text())
        self.starting_equity = state["starting_equity"]
        self.realized_pnl = state["realized_pnl"]
        self.positions = {k: Position(**p) for k, p in state["positions"].items()}
        self.trades = [Trade(**t) for t in state["trades"]]
