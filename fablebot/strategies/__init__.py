from .base import Action, Signal, Strategy
from .mean_reversion import MeanReversion
from .momentum_breakout import MomentumBreakout
from .trend_following import TrendFollowing

STRATEGIES: dict[str, type[Strategy]] = {
    "mean_reversion": MeanReversion,
    "momentum_breakout": MomentumBreakout,
    "trend_following": TrendFollowing,
}


def build_strategy(name: str, params: dict) -> Strategy:
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
