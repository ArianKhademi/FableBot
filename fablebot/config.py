"""Configuration loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class AccountConfig:
    starting_equity: float = 100_000.0
    risk_per_trade: float = 0.01


@dataclass
class ScheduleConfig:
    timezone: str = "America/New_York"
    morning_report: str = "07:00"
    evening_report: str = "21:00"
    poll_interval_minutes: int = 5


@dataclass
class CorrelationConfig:
    groups: dict[str, list[str]] = field(default_factory=dict)
    max_same_direction_per_group: int = 2


@dataclass
class InstrumentConfig:
    key: str
    symbol: str
    name: str
    strategy: str
    timeframe: str
    params: dict = field(default_factory=dict)
    # param_bounds: {param_name: [min, max]} — defines the optimizer's search
    # space. Only params listed here are varied; the rest are frozen at their
    # current value. Integer params (lookback, channels) are rounded to int.
    param_bounds: dict = field(default_factory=dict)
    # Optional composable-DSL genome. When present the instrument runs a
    # GenomeStrategy built from this spec instead of the named ``strategy``;
    # ``strategy`` then just labels it. Lets any genome trade any symbol.
    genome: dict | None = None
    # Optional live ML/RL model spec, e.g. {type: ml_gbt, horizon: 5, ...} or
    # {type: rl_qlearn, epochs: 20}. When present the live engine trains it on
    # current deep history (and re-trains on a schedule) and trades it instead
    # of the named strategy. Takes precedence over ``genome``.
    model: dict | None = None
    # Optional per-instrument data source override (e.g. "binance" for crypto).
    # Falls back to config.data.source when unset. The symbol is translated as
    # needed (BTC-USD -> BTCUSDT for Binance), so news/display stay on Yahoo.
    source: str | None = None


@dataclass
class DataConfig:
    # Which OHLCVSource to use and how to cache it. Swapping sources (incl. a
    # future paid feed) is a config change, not a code change.
    source: str = "yfinance"
    cache_dir: str | None = None
    cache_ttl_minutes: float = 1440.0


@dataclass
class UniverseConfig:
    """A large, config-driven scan list. Combined with the DSL (any genome on
    any symbol), the search can discover which symbols are even worth trading —
    distinct from the handful actually traded in ``instruments``."""
    default_timeframe: str = "1d"
    default_strategy: str = "trend_following"
    symbols: list[dict] = field(default_factory=list)


@dataclass
class CostsConfig:
    # Combined commission + slippage applied to every backtest fill during the
    # *trial process* so the gate can't certify a frictionless-only edge.
    friction_bps: float = 3.0


@dataclass
class Config:
    account: AccountConfig
    schedule: ScheduleConfig
    correlation: CorrelationConfig
    instruments: dict[str, InstrumentConfig]
    data: DataConfig = field(default_factory=DataConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    costs: CostsConfig = field(default_factory=CostsConfig)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    instruments = {
        key: InstrumentConfig(key=key, **spec)
        for key, spec in raw.get("instruments", {}).items()
    }
    return Config(
        account=AccountConfig(**raw.get("account", {})),
        schedule=ScheduleConfig(**raw.get("schedule", {})),
        correlation=CorrelationConfig(**raw.get("correlation", {})),
        instruments=instruments,
        data=DataConfig(**raw.get("data", {})),
        universe=UniverseConfig(**raw.get("universe", {})),
        costs=CostsConfig(**raw.get("costs", {})),
    )


def universe_instruments(config: Config) -> list[InstrumentConfig]:
    """Expand the ``universe`` config into InstrumentConfigs for scanning. Each
    entry may override timeframe/strategy/name; otherwise the universe defaults
    apply. These carry no param_bounds — the genome search supplies structure."""
    uni = config.universe
    out: list[InstrumentConfig] = []
    for spec in uni.symbols:
        symbol = spec["symbol"]
        out.append(
            InstrumentConfig(
                key=spec.get("key", symbol),
                symbol=symbol,
                name=spec.get("name", symbol),
                strategy=spec.get("strategy", uni.default_strategy),
                timeframe=spec.get("timeframe", uni.default_timeframe),
                params=spec.get("params", {}),
                param_bounds=spec.get("param_bounds", {}),
                source=spec.get("source"),
            )
        )
    return out
