"""Risk engine: ATR-based position sizing, the hard 1% stop, and the
correlation filter.

Sizing keeps dollar risk constant across instruments: a quiet day on gold
gets a bigger size than a volatile day on bitcoin, because the size is the
risk budget divided by the volatility-scaled stop distance.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SizedTrade:
    units: float
    stop_distance: float  # price distance from entry to stop
    stop_price: float
    risk_amount: float  # dollars at risk if the stop is hit


def size_position(
    equity: float,
    risk_per_trade: float,
    entry_price: float,
    atr: float,
    atr_stop_mult: float,
    direction: str,
) -> SizedTrade:
    """Risk a fixed fraction of equity, with the stop placed at an ATR
    multiple. units * stop_distance == risk budget by construction."""
    if atr != atr or atr <= 0:  # NaN <= 0 is False, so check NaN explicitly
        raise ValueError(f"ATR must be a positive number, got {atr}")
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")
    if equity <= 0:
        # A depleted book must never size a position: negative risk flips the
        # units sign and turns every rule (stops, gap checks) inside out.
        raise ValueError(f"equity depleted ({equity:,.2f}) — refusing to size a position")
    risk_amount = equity * risk_per_trade
    stop_distance = atr * atr_stop_mult
    units = risk_amount / stop_distance
    stop_price = entry_price - stop_distance if direction == "long" else entry_price + stop_distance
    return SizedTrade(
        units=units,
        stop_distance=stop_distance,
        stop_price=stop_price,
        risk_amount=risk_amount,
    )


class CorrelationFilter:
    """Blocks piling into correlated exposure.

    If ``max_same_direction`` positions in a group are already open in one
    direction (e.g. S&P and Nasdaq both long), a new same-direction entry in
    that group (e.g. Bitcoin long) is rejected — it would double the real
    risk-on exposure, not diversify it.
    """

    def __init__(self, groups: dict[str, list[str]], max_same_direction: int = 2):
        self.groups = {name: set(members) for name, members in groups.items()}
        self.max_same_direction = max_same_direction

    def allows(
        self,
        instrument: str,
        direction: str,
        open_positions: dict[str, str],
    ) -> tuple[bool, str]:
        """``open_positions`` maps instrument key -> 'long'/'short'."""
        for group_name, members in self.groups.items():
            if instrument not in members:
                continue
            aligned = [
                key
                for key, pos_dir in open_positions.items()
                if key != instrument and key in members and pos_dir == direction
            ]
            if len(aligned) >= self.max_same_direction:
                return (
                    False,
                    f"correlation filter: {', '.join(sorted(aligned))} already "
                    f"{direction} in group '{group_name}'",
                )
        return True, "ok"
