"""Build a *trained* model-backed strategy for live trading.

An instrument configured with a ``model`` block (instead of, or alongside, a
named ``strategy``/``genome``) trades a freshly-trained ML or RL model. The
model is never a frozen pickle: the live engine trains it on current deep
history at startup and re-trains it on a schedule, so it tracks the market —
the same discipline the offline trainers use. Sizing, the hard stop, gap
protection and trailing remain entirely with the risk engine; the model only
chooses entries/direction.

Config shape (per instrument in config.yaml):

    model:
      type: ml_gbt        # or rl_qlearn
      horizon: 5          # ml_gbt: label horizon; rest -> GBT kwargs
      n_estimators: 60
      max_depth: 3
    # type: rl_qlearn -> epochs / lr / gamma / epsilon / cost
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .strategies.base import Strategy


def build_model_strategy(model_cfg: dict, df: pd.DataFrame) -> Optional[Strategy]:
    """Train the model described by ``model_cfg`` on ``df`` and return a ready
    Strategy, or None if there isn't enough data to train (caller falls back)."""
    cfg = dict(model_cfg)
    kind = cfg.pop("type", None)

    if kind == "ml_gbt":
        from .ml import MLModel, MLStrategy

        model = MLModel(**cfg)
        return MLStrategy(model) if model.train(df) else None

    if kind == "rl_qlearn":
        from .rl import QLearningAgent, RLStrategy

        agent = QLearningAgent(**cfg)
        return RLStrategy(agent) if agent.train(df) else None

    raise ValueError(f"unknown live model type {kind!r}; choose from ml_gbt, rl_qlearn")
