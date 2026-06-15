"""Guided / evolutionary search over the strategy DSL.

Pure random sampling rediscovers nothing: each run starts blind. This module
makes the search *accumulate*:

  - :class:`GeneticSearch` evolves a population of genomes (selection +
    crossover + mutation, seeded from the hand-written strategies and any
    survivors loaded from disk). It persists, so a restart resumes the
    population instead of starting over.
  - :func:`tpe_suggest` is a lightweight Tree-structured Parzen Estimator: a
    dependency-free Bayesian optimiser that proposes the next value of a
    continuous parameter from where good observations have clustered versus
    bad ones. The genetic layer searches *structure*; TPE refines an elite
    genome's *numbers*.

Neither touches config.yaml or live code — they only produce genomes the
hardened gate then has to certify, exactly like the random optimizer.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import dsl


# ── Bayesian (TPE) suggestion for a single continuous parameter ──────────────


def tpe_suggest(
    observations: list[tuple[float, float]],
    lo: float,
    hi: float,
    rng: np.random.Generator,
    n_candidates: int = 32,
    gamma: float = 0.25,
    is_int: bool = False,
) -> float:
    """Suggest the next value in ``[lo, hi]`` given ``(value, score)`` history
    (higher score better). Models the top ``gamma`` of observations ("good")
    and the rest ("bad") as Gaussian kernel densities and returns the candidate
    maximising the good/bad density ratio — Hyperopt's TPE, without scipy.
    Falls back to uniform exploration until enough history exists."""
    def clamp(x):
        x = min(max(x, lo), hi)
        return int(round(x)) if is_int else round(float(x), 4)

    if len(observations) < 4:
        return clamp(rng.uniform(lo, hi))

    ordered = sorted(observations, key=lambda o: o[1], reverse=True)
    n_good = max(1, int(math.ceil(gamma * len(ordered))))
    good = np.array([o[0] for o in ordered[:n_good]], dtype=float)
    bad = np.array([o[0] for o in ordered[n_good:]], dtype=float)
    if bad.size == 0:
        bad = good

    span = max(hi - lo, 1e-9)

    def bandwidth(a):
        return max(float(a.std()), span * 0.1, 1e-6)

    bw_good, bw_bad = bandwidth(good), bandwidth(bad)
    centers = good[rng.integers(0, good.size, n_candidates)]
    candidates = np.clip(centers + rng.normal(0.0, bw_good, n_candidates), lo, hi)

    def density(samples, points, bw):
        z = (points[:, None] - samples[None, :]) / bw
        return np.mean(np.exp(-0.5 * z * z) / bw, axis=1) + 1e-12

    ratio = density(good, candidates, bw_good) / density(bad, candidates, bw_bad)
    return clamp(candidates[int(np.argmax(ratio))])


def tpe_refine_genome(
    genome: dict,
    history: list[tuple[dict, float]],
    rng: np.random.Generator,
) -> dict:
    """Propose a tuned copy of ``genome`` by running :func:`tpe_suggest` on each
    of its scalar params, using ``(genome, fitness)`` history for the values
    that param has taken before."""
    overrides = {}
    for path, current in dsl.genome_param_paths(genome).items():
        name = path.split(".")[-1]
        bounds = dsl._PARAM_BOUNDS.get(name)
        if bounds is None:
            continue
        lo, hi, is_int = bounds
        obs = [
            (dsl.genome_param_paths(g).get(path), f)
            for g, f in history
            if path in dsl.genome_param_paths(g)
        ]
        obs = [(v, f) for v, f in obs if v is not None and math.isfinite(f)]
        overrides[path] = tpe_suggest(obs, lo, hi, rng, is_int=is_int) if obs else current
    refined = dsl.apply_genome_params(genome, overrides)
    dsl._repair(refined)
    dsl.validate_genome(refined)
    return refined


# ── fitness ──────────────────────────────────────────────────────────────────


def genome_fitness(oos_sharpe: float, oos_trades: int, min_trades: int = 20) -> float:
    """Walk-forward fitness: OOS Sharpe, penalised when the trade count is too
    thin to trust (keeps the GA from chasing 3-trade flukes)."""
    if not math.isfinite(oos_sharpe):
        return -1e9
    penalty = 0.25 * max(0, min_trades - max(oos_trades, 0))
    return oos_sharpe - penalty


# ── genetic search over genomes ──────────────────────────────────────────────


class GeneticSearch:
    """Evolves a population of genomes. Caller drives it: :meth:`ask` for a
    genome to evaluate, :meth:`tell` its fitness; once the generation is scored,
    :meth:`evolve` breeds the next one. Elitism keeps the best survivors intact;
    the rest come from tournament selection + crossover + mutation."""

    def __init__(
        self,
        rng: np.random.Generator,
        population_size: int = 24,
        elite_frac: float = 0.25,
        crossover_prob: float = 0.6,
        mutation_rate: float = 0.6,
        seeds: Optional[list[dict]] = None,
    ):
        self.rng = rng
        self.population_size = int(population_size)
        self.elite_frac = elite_frac
        self.crossover_prob = crossover_prob
        self.mutation_rate = mutation_rate
        self.generation = 0
        self.scores: dict[str, float] = {}
        self.history: list[tuple[dict, float]] = []  # every (genome, fitness) ever scored

        seeds = list(seeds or dsl.seed_genomes())
        self.population: list[dict] = [copy.deepcopy(g) for g in seeds]
        while len(self.population) < self.population_size:
            self.population.append(dsl.sample_genome(self.rng))
        self.population = self.population[: self.population_size]
        self._queue: list[dict] = list(self.population)

    # -- evaluation loop ------------------------------------------------------

    def ask(self) -> Optional[dict]:
        """Next un-scored genome in this generation, or None when all scored."""
        while self._queue:
            g = self._queue.pop(0)
            if dsl.genome_id(g) not in self.scores:
                return g
        return None

    def tell(self, genome: dict, fitness: float) -> None:
        self.scores[dsl.genome_id(genome)] = float(fitness)
        self.history.append((copy.deepcopy(genome), float(fitness)))

    def generation_complete(self) -> bool:
        return self.ask() is None

    def _score_of(self, genome: dict) -> float:
        return self.scores.get(dsl.genome_id(genome), -1e9)

    def _tournament(self, k: int = 3) -> dict:
        contenders = [self.population[i] for i in self.rng.integers(0, len(self.population), k)]
        return max(contenders, key=self._score_of)

    def evolve(self) -> None:
        """Breed the next generation from the scored current one."""
        ranked = sorted(self.population, key=self._score_of, reverse=True)
        n_elite = max(1, int(round(self.elite_frac * self.population_size)))
        next_pop = [copy.deepcopy(g) for g in ranked[:n_elite]]  # elitism

        while len(next_pop) < self.population_size:
            if self.rng.random() < self.crossover_prob:
                child = dsl.crossover_genome(self._tournament(), self._tournament(), self.rng)
            else:
                child = copy.deepcopy(self._tournament())
            if self.rng.random() < self.mutation_rate:
                child = dsl.mutate_genome(child, self.rng)
            next_pop.append(child)

        self.population = next_pop
        self._queue = [g for g in next_pop if dsl.genome_id(g) not in self.scores]
        self.generation += 1

    # -- inspection -----------------------------------------------------------

    def best(self, k: int = 5) -> list[tuple[dict, float]]:
        scored = [(g, f) for g, f in self.history]
        # de-dup by genome id, keeping the best fitness seen
        best_by_id: dict[str, tuple[dict, float]] = {}
        for g, f in scored:
            gid = dsl.genome_id(g)
            if gid not in best_by_id or f > best_by_id[gid][1]:
                best_by_id[gid] = (g, f)
        return sorted(best_by_id.values(), key=lambda gf: gf[1], reverse=True)[:k]

    # -- persistence ----------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "population_size": self.population_size,
            "elite_frac": self.elite_frac,
            "crossover_prob": self.crossover_prob,
            "mutation_rate": self.mutation_rate,
            "population": self.population,
            "scores": self.scores,
            "history": [{"genome": g, "fitness": f} for g, f in self.history],
        }

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: Path | str, rng: np.random.Generator) -> "GeneticSearch":
        data = json.loads(Path(path).read_text())
        search = cls(
            rng,
            population_size=data["population_size"],
            elite_frac=data["elite_frac"],
            crossover_prob=data.get("crossover_prob", 0.6),
            mutation_rate=data["mutation_rate"],
            seeds=data["population"],
        )
        search.generation = data["generation"]
        search.scores = {k: float(v) for k, v in data["scores"].items()}
        search.history = [(h["genome"], float(h["fitness"])) for h in data.get("history", [])]
        search._queue = [g for g in search.population if dsl.genome_id(g) not in search.scores]
        return search


def resume_or_new(path: Path | str, rng: np.random.Generator, **kwargs) -> GeneticSearch:
    """Load a persisted population if present, else start a fresh one — this is
    what makes the engine 'keep learning across restarts'."""
    path = Path(path)
    if path.exists():
        return GeneticSearch.load(path, rng)
    return GeneticSearch(rng, **kwargs)


# ── evaluating and validating a genome on real data ──────────────────────────


def genome_instrument(symbol: str, timeframe: str, genome: dict, key: Optional[str] = None):
    """Wrap a genome as an InstrumentConfig so it can run on any symbol through
    the existing backtest path."""
    from .config import InstrumentConfig

    return InstrumentConfig(
        key=key or symbol, symbol=symbol, name=symbol,
        strategy=genome.get("name", "genome"), timeframe=timeframe,
        params={}, param_bounds={}, genome=genome,
    )


def evaluate_genome(df, symbol: str, timeframe: str, genome: dict, risk_per_trade: float = 0.01):
    """Walk-forward a genome on ``df``; returns the Trial (or None)."""
    from .optimizer import run_walk_forward

    return run_walk_forward(df, genome_instrument(symbol, timeframe, genome), {}, risk_per_trade)


def _eval_worker(args):
    """Picklable: evaluate one genome and return (oos_sharpe, oos_trades) so a
    generation can be scored across worker processes."""
    df, symbol, timeframe, genome, risk = args
    trial = evaluate_genome(df, symbol, timeframe, genome, risk)
    if trial is None:
        return (float("-inf"), 0)
    return (trial.oos_sharpe, trial.oos_trades)


def current_genome_for(inst) -> dict:
    """The genome an instrument currently runs — its explicit genome, or the
    seed genome matching its hand-written strategy + params."""
    if getattr(inst, "genome", None):
        return inst.genome
    p = inst.params
    if inst.strategy == "mean_reversion":
        return dsl.seed_mean_reversion(
            p.get("lookback", 20), p.get("entry_z", 2.0), p.get("exit_z", 0.5),
            p.get("atr_period", 14), p.get("atr_stop_mult", 2.0))
    if inst.strategy == "momentum_breakout":
        return dsl.seed_momentum_breakout(
            p.get("entry_channel", 20), p.get("exit_channel", 10),
            p.get("atr_period", 14), p.get("atr_stop_mult", 2.5), p.get("trail_atr_mult", 2.5))
    return dsl.seed_trend_following(
        p.get("fast_ema", 20), p.get("slow_ema", 50),
        p.get("atr_period", 14), p.get("atr_stop_mult", 3.0), p.get("trail_atr_mult", 3.0))


def _perturb_genome(genome: dict, rng: np.random.Generator, frac: float = 0.1) -> dict:
    """Jitter a genome's numeric params by ±frac for the stability test."""
    overrides = {}
    for path, val in dsl.genome_param_paths(genome).items():
        name = path.split(".")[-1]
        jittered = float(val) * (1.0 + float(rng.uniform(-frac, frac)))
        overrides[path] = dsl._clamp(name, jittered)
    g = dsl.apply_genome_params(genome, overrides)
    dsl._repair(g)
    return g


def validate_genome(
    search_df,
    holdout_df,
    symbol: str,
    timeframe: str,
    candidate: dict,
    current: dict,
    rng: np.random.Generator,
    n_trials: int = 0,
    sr_std_perbar: float = 0.0,
    risk_per_trade: float = 0.01,
) -> dict:
    """Apply the hardened gate to an *evolved genome* vs the instrument's
    current genome: non-overlapping regime folds, a parameter-perturbation
    stability test, an embargoed held-out confirmation, and the deflated-Sharpe
    multiple-comparisons correction. Same rigor the numeric optimizer uses —
    just comparing two genomes instead of two parameter sets."""
    from .optimizer import (
        BARS_PER_YEAR, DSR_THRESHOLD, MIN_HOLDOUT_TRADES, PERTURBATIONS,
        deflated_sharpe_ratio, equity_returns, perturbation_passes,
        run_walk_forward, sharpe_ratio, validation_folds,
    )
    import math as _math

    from .backtest import run_backtest

    cand_inst = genome_instrument(symbol, timeframe, candidate)
    curr_inst = genome_instrument(symbol, timeframe, current)

    folds_won = folds_valid = 0
    for fold in validation_folds(search_df):
        c = run_walk_forward(fold, cand_inst, {}, risk_per_trade)
        b = run_walk_forward(fold, curr_inst, {}, risk_per_trade)
        if c is None or b is None:
            continue
        folds_valid += 1
        folds_won += int(c.oos_sharpe > b.oos_sharpe)

    cand_full = run_walk_forward(search_df, cand_inst, {}, risk_per_trade)
    curr_full = run_walk_forward(search_df, curr_inst, {}, risk_per_trade)
    candidate_oos = cand_full.oos_sharpe if cand_full else float("-inf")
    baseline_oos = curr_full.oos_sharpe if curr_full else float("-inf")

    perturbed = []
    for _ in range(PERTURBATIONS):
        t = run_walk_forward(search_df, genome_instrument(symbol, timeframe, _perturb_genome(candidate, rng)), {}, risk_per_trade)
        if t is not None:
            perturbed.append(t.oos_sharpe)
    median_perturbed = float(np.median(perturbed)) if perturbed else float("-inf")

    def inst_strategy(inst):
        from .strategies import build_strategy
        return build_strategy(inst.strategy, {}, genome=inst.genome)

    def full_slice(inst):
        try:
            strat = inst_strategy(inst)
            if holdout_df is None or len(holdout_df) < strat.min_bars() * 2:
                return None
            res = run_backtest(holdout_df, strat, risk_per_trade=risk_per_trade, instrument=inst.key)
        except Exception:
            return None
        return sharpe_ratio(res.equity_curve, BARS_PER_YEAR.get(timeframe, 252)), res.trades

    cand_h = full_slice(cand_inst)
    curr_h = full_slice(curr_inst)
    holdout_checked = holdout_df is not None
    h_cand = cand_h[0] if cand_h else float("-inf")
    h_curr = curr_h[0] if curr_h else float("-inf")

    # deflated-Sharpe correction on the candidate's OOS return stream
    dsr = 1.0
    if cand_full is not None and n_trials >= 10:
        dsr = deflated_sharpe_ratio(equity_returns(cand_full.oos_equity_curve), n_trials, sr_std_perbar)

    folds_ok = folds_valid >= 3 and folds_won >= _math.ceil(0.75 * folds_valid)
    perturb_ok = perturbation_passes(median_perturbed, candidate_oos, baseline_oos)
    holdout_ok = (not holdout_checked) or (
        cand_h is not None and curr_h is not None and cand_h[1] >= MIN_HOLDOUT_TRADES and h_cand > h_curr
    )
    dsr_ok = dsr >= DSR_THRESHOLD

    if not folds_ok:
        reason = f"regime test failed: beat current in {folds_won}/{folds_valid} folds"
    elif not perturb_ok:
        reason = f"perturbation failed: jitter median {median_perturbed:.3f} unstable vs own {candidate_oos:.3f}"
    elif not dsr_ok:
        reason = f"deflated-Sharpe {dsr:.3f} < {DSR_THRESHOLD} after {n_trials} trials"
    elif not holdout_ok:
        reason = f"held-out failed: candidate {h_cand:.3f} vs current {h_curr:.3f}"
    else:
        reason = (
            f"won {folds_won}/{folds_valid} folds; jitter stable; "
            f"held-out {h_cand:.3f} > {h_curr:.3f}; DSR {dsr:.3f}"
        )

    return {
        "qualified": folds_ok and perturb_ok and holdout_ok and dsr_ok,
        "folds_won": folds_won, "folds_valid": folds_valid,
        "candidate_oos_sharpe": candidate_oos, "baseline_oos_sharpe": baseline_oos,
        "median_perturbed_sharpe": median_perturbed, "deflated_sharpe": dsr,
        "holdout_checked": holdout_checked,
        "holdout_candidate_sharpe": h_cand, "holdout_current_sharpe": h_curr,
        "reason": reason,
    }
