from __future__ import annotations

from .base import Action, Signal, Strategy
from .mean_reversion import MeanReversion
from .momentum_breakout import MomentumBreakout
from .trend_following import TrendFollowing

STRATEGIES: dict[str, type[Strategy]] = {
    "mean_reversion": MeanReversion,
    "momentum_breakout": MomentumBreakout,
    "trend_following": TrendFollowing,
}


def build_strategy(name: str, params: dict, genome: dict | None = None) -> Strategy:
    """Instantiate a strategy. With ``genome`` set, build a composable
    :class:`~fablebot.dsl.GenomeStrategy` (any genome on any symbol); otherwise
    look ``name`` up in the hand-written registry."""
    if genome is not None:
        from ..dsl import GenomeStrategy  # lazy: avoids an import cycle via base
        return GenomeStrategy(genome)
    try:
        cls = STRATEGIES[name]
    except KeyError:
        raise ValueError(f"Unknown strategy {name!r}; choose from {sorted(STRATEGIES)}")
    return cls(params)


__all__ = [
    "Action",
    "Signal",
    "Strategy",
    "MeanReversion",
    "MomentumBreakout",
    "TrendFollowing",
    "STRATEGIES",
    "build_strategy",
]
