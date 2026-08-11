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
from dataclasses import asdict, dataclass, field, fields
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
    # True once the stop has been ratcheted from its entry placement, so a
    # stop exit can be labeled honestly ("trail stop hit" vs "hard stop hit")
    # and loss-only logic (the daily circuit breaker) can tell them apart.
    stop_trailed: bool = False

    def unrealized_pnl(self, price: float) -> float:
        sign = 1.0 if self.direction == "long" else -1.0
        return (price - self.entry_price) * self.units * sign

    def update_excursions(self, high: float, low: float) -> bool:
        """Returns True when either extreme moved (lets callers skip a save)."""
        best = high if self.direction == "long" else low
        worst = low if self.direction == "long" else high
        fav = max(self.max_favorable_pnl, self.unrealized_pnl(best))
        adv = min(self.max_adverse_pnl, self.unrealized_pnl(worst))
        moved = fav != self.max_favorable_pnl or adv != self.max_adverse_pnl
        self.max_favorable_pnl, self.max_adverse_pnl = fav, adv
        return moved

    def tighten_to(self, candidate: float) -> bool:
        """Move the stop to ``candidate`` only if that tightens it (long: up,
        short: down). The only-tighten invariant lives here."""
        if candidate != candidate:  # NaN guard
            return False
        if self.direction == "long":
            if candidate > self.stop_price:
                self.stop_price = candidate
                self.stop_trailed = True
                return True
        else:
            if candidate < self.stop_price:
                self.stop_price = candidate
                self.stop_trailed = True
                return True
        return False

    def trail(self, high: float, low: float, atr: float, trail_mult: float) -> bool:
        """Ratchet the stop toward price, chandelier-style: an ATR multiple
        behind the bar's favorable extreme. Stops only ever tighten — a
        candidate that would widen the stop is ignored. Returns True if the
        stop moved."""
        if atr <= 0 or trail_mult <= 0:
            return False
        extreme = high if self.direction == "long" else low
        offset = atr * trail_mult
        return self.tighten_to(extreme - offset if self.direction == "long" else extreme + offset)

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
    def __init__(
        self,
        starting_equity: float,
        state_path: Optional[Path] = None,
        friction_bps: float = 0.0,
    ):
        self.starting_equity = starting_equity
        self.realized_pnl = 0.0
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        # Slippage applied to STOP fills (adverse: a long's stop fills lower, a
        # short's higher). Backtests pass the configured friction so stop-heavy
        # configs can't collect frictionless exits; the live paper book keeps 0.
        self.friction_bps = float(friction_bps)
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
        if pos is not None and pos.update_excursions(high, low):
            self._save()  # only rewrite state when an extreme actually moved

    def trail_stop(
        self, instrument: str, high: float, low: float, atr: float, trail_mult: float
    ) -> bool:
        pos = self.positions.get(instrument)
        if pos is None:
            return False
        moved = pos.trail(high, low, atr, trail_mult)
        if moved:
            self._save()
        return moved

    def tighten_stop(self, instrument: str, candidate: float) -> bool:
        """Apply a pre-computed stop candidate (only-tighten). Used by the
        engine's deferred trailing: a stop derived from bar B's extreme becomes
        active on bar B+1, never tested against the bar that produced it."""
        pos = self.positions.get(instrument)
        if pos is None:
            return False
        moved = pos.tighten_to(candidate)
        if moved:
            self._save()
        return moved

    def _stop_fill(self, pos: Position, raw: float) -> float:
        """Adverse slippage on a stop fill (long exits lower, short higher)."""
        if not self.friction_bps:
            return raw
        slip = self.friction_bps / 1e4
        return raw * (1.0 - slip) if pos.direction == "long" else raw * (1.0 + slip)

    def enforce_stops(
        self,
        instrument: str,
        high: float,
        low: float,
        close: float,
        open_: Optional[float] = None,
    ) -> Optional[Trade]:
        """The hard stop, checked every cycle. Returns the closing trade if the
        position was cut. When the bar's ``open_`` is supplied and price GAPPED
        past the stop, the fill is the open — the first price that actually
        traded — not the stop level nothing ever traded at. Trailed stops are
        labeled distinctly so profit-locks and loss stops stay separable."""
        pos = self.positions.get(instrument)
        if pos is None:
            return None
        if pos.stop_hit(high, low):
            fill = pos.stop_price
            reason_price = pos.stop_price
            if open_ is not None:
                gapped = (open_ <= pos.stop_price) if pos.direction == "long" else (open_ >= pos.stop_price)
                if gapped:
                    fill = open_
            label = "trail stop hit at" if pos.stop_trailed else "hard stop hit at"
            if fill != reason_price:
                return self.close_position(
                    instrument, self._stop_fill(pos, fill),
                    f"{label} {reason_price:,.2f} (gapped, filled {fill:,.2f})",
                )
            return self.close_position(
                instrument, self._stop_fill(pos, fill), f"{label} {reason_price:,.2f}"
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
        # Tolerant load: drop keys the current dataclasses don't know, so a
        # rollback to older code (or a removed field) can't hold the book's
        # history hostage with a TypeError at startup.
        pos_fields = {f.name for f in fields(Position)}
        trade_fields = {f.name for f in fields(Trade)}
        self.positions = {
            k: Position(**{n: v for n, v in p.items() if n in pos_fields})
            for k, p in state["positions"].items()
        }
        self.trades = [
            Trade(**{n: v for n, v in t.items() if n in trade_fields}) for t in state["trades"]
        ]
