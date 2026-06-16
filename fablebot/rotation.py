"""Dynamic universe rotation — pick which instruments the live book trades.

Instead of trading a fixed five forever, a daily scan scores the whole universe
(plus the current book) by the best gated edge it can find on each symbol, and
proposes a rotation: add the strongest newcomers, drop stale holdings. It is
*evidence*, not an automatic live swap — the change is human-gated through the
reviewer, exactly like every other proposal.

Guardrails (in :func:`propose_rotation`):
  - **Never force-close an open position to rotate.** Instruments with a live
    position are sticky — kept regardless of score until they close naturally.
  - **Cap concurrency.** At most ``max_active`` instruments trade at once
    (open positions may carry the book slightly past the cap, by design).
  - **Hysteresis.** A challenger must beat a current holding by ``churn_margin``
    to replace it, so the book doesn't thrash on noise.
  - **Quality floor.** A newcomer must clear ``min_score`` to be added at all.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional


def score_symbol(symbol: str, timeframe: str, df, friction: float = 0.0, min_trades: int = 20) -> float:
    """A symbol's tradeability score: the best out-of-sample Sharpe (with
    friction) achievable by a seed strategy on it, requiring enough trades in
    both halves. -inf if nothing clears the trade floor. This ranks symbols;
    the actual add still goes through the full hardened gate via the reviewer."""
    from . import dsl, search

    best = float("-inf")
    for genome in dsl.seed_genomes():
        trial = search.evaluate_genome(df, symbol, timeframe, genome, friction_bps=friction)
        if (
            trial is not None
            and trial.oos_trades >= min_trades
            and trial.is_trades >= min_trades
            and math.isfinite(trial.oos_sharpe)
        ):
            best = max(best, trial.oos_sharpe)
    return best


def _score_worker(args):
    """Picklable: score one symbol across worker processes."""
    symbol, timeframe, df, friction, min_trades = args
    return score_symbol(symbol, timeframe, df, friction, min_trades)


def read_open_positions(portfolio_path: Path | str) -> set[str]:
    """Instrument keys with a live open position — these are sticky in a
    rotation and never dropped."""
    path = Path(portfolio_path)
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text()).get("positions", {}).keys())
    except Exception:
        return set()


def propose_rotation(
    current: list[str],
    scores: dict[str, float],
    open_keys,
    max_active: int,
    min_score: float = 0.0,
    churn_margin: float = 0.0,
) -> dict:
    """Decide the next live book. Pure. ``current`` is today's traded keys,
    ``scores`` maps every candidate key (current + universe) to its score,
    ``open_keys`` are instruments with a live position (sticky)."""
    current, open_keys = set(current), set(open_keys)

    # A current holding gets a churn_margin bonus, so a challenger must beat it
    # by that margin to outrank it — hysteresis against thrashing.
    def effective(k: str) -> float:
        return scores.get(k, float("-inf")) + (churn_margin if k in current else 0.0)

    eligible = [k for k in scores if scores[k] >= min_score or k in current or k in open_keys]
    ranked = sorted(eligible, key=effective, reverse=True)

    target: list[str] = []
    for k in ranked:
        if len(target) >= max_active and k not in open_keys:
            continue
        if k not in target:
            target.append(k)
    target_set = set(target) | open_keys  # open positions kept even past the cap

    # open holdings kept purely because they're sticky (would otherwise rank out)
    by_score_top = set(target[:max_active])
    return {
        "target": sorted(target_set),
        "add": sorted(target_set - current),
        "drop": sorted(current - target_set),
        "keep": sorted(current & target_set),
        "kept_sticky": sorted(open_keys - by_score_top),
    }
