import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_ohlcv(closes, start="2026-01-01", freq="1h", spread=0.5) -> pd.DataFrame:
    """Build a synthetic OHLCV frame around a sequence of closes."""
    closes = np.asarray(closes, dtype=float)
    index = pd.date_range(start, periods=len(closes), freq=freq)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) + spread
    lows = np.minimum(opens, closes) - spread
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.full(len(closes), 1000.0),
        },
        index=index,
    )


@pytest.fixture
def ohlcv_factory():
    return make_ohlcv
