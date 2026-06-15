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


@dataclass
class Config:
    account: AccountConfig
    schedule: ScheduleConfig
    correlation: CorrelationConfig
    instruments: dict[str, InstrumentConfig]


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
    )
