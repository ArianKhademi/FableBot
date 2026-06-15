"""Universe screening.

The config can list far more symbols than are worth searching every night.
Screening ranks them by recent activity (average dollar volume) and drops names
without enough history, so the genome search spends its budget on tradeable
instruments. Combined with the DSL (any genome on any symbol), this is how the
engine decides *which symbols are even worth trading*.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def average_dollar_volume(df: pd.DataFrame, lookback: int = 60) -> float:
    """Mean close x volume over the last ``lookback`` bars — a liquidity proxy."""
    if df is None or len(df) == 0 or "volume" not in df.columns:
        return 0.0
    tail = df.tail(lookback)
    val = (tail["close"].astype(float) * tail["volume"].astype(float)).mean()
    return float(val) if pd.notna(val) else 0.0


def screen_universe(
    instruments: list,
    source,
    top_n: Optional[int] = None,
    min_bars: int = 200,
    lookback: int = 60,
):
    """Fetch each instrument, drop those without ``min_bars`` of history, and
    return them ranked by liquidity (most active first), capped at ``top_n``.
    Symbols whose data can't be fetched are skipped, not fatal."""
    ranked: list[tuple] = []
    for inst in instruments:
        try:
            df = source.fetch(inst.symbol, inst.timeframe)
        except Exception:
            continue
        if df is None or len(df) < min_bars:
            continue
        ranked.append((inst, average_dollar_volume(df, lookback)))
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    chosen = [inst for inst, _ in ranked]
    return chosen[:top_n] if top_n else chosen
