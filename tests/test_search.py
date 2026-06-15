"""Guided / evolutionary search: TPE param suggestion, the genetic genome
search, persistence, and the genome-aware hardened gate."""

import numpy as np
import pytest

from fablebot import dsl, search
from tests.conftest import make_ohlcv


# ── TPE (Bayesian) suggestion ────────────────────────────────────────────────

def test_tpe_explores_when_history_is_sparse():
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = search.tpe_suggest([(1.0, 0.5)], 0.0, 10.0, rng)
        assert 0.0 <= x <= 10.0


def test_tpe_concentrates_near_the_optimum():
    """A peaked objective at x*=7: after enough guided suggestions the search
    should propose values clustered near the optimum, not uniformly."""
    rng = np.random.default_rng(1)
    obj = lambda x: -((x - 7.0) ** 2)
    obs = [(float(v), obj(v)) for v in rng.uniform(0, 10, 6)]
    suggestions = []
    for _ in range(40):
        x = search.tpe_suggest(obs, 0.0, 10.0, rng)
        obs.append((x, obj(x)))
        suggestions.append(x)
    late = np.array(suggestions[-15:])
    assert abs(np.median(late) - 7.0) < 1.5  # converged toward the peak


def test_genome_fitness_penalises_thin_trades():
    assert search.genome_fitness(1.0, 50) == pytest.approx(1.0)
    assert search.genome_fitness(1.0, 4) < 0  # few trades heavily penalised
    assert search.genome_fitness(float("nan"), 50) < -1e8


# ── genetic search ───────────────────────────────────────────────────────────

def _drain_and_score(s, fitness):
    g = s.ask()
    while g is not None:
        s.tell(g, fitness(g))
        g = s.ask()


def test_population_includes_seeds():
    rng = np.random.default_rng(0)
    seeds = dsl.seed_genomes()
    s = search.GeneticSearch(rng, population_size=12, seeds=seeds)
    ids = {dsl.genome_id(g) for g in s.population}
    assert {dsl.genome_id(g) for g in seeds} <= ids
    assert len(s.population) == 12


def test_genetic_search_improves_a_synthetic_objective():
    """Drive the GA on a fitness it can actually climb (atr_stop_mult -> 4.0)
    and confirm the best genome improves on the best seed."""
    rng = np.random.default_rng(3)
    target = 4.0
    fitness = lambda g: -((g["risk"]["atr_stop_mult"] - target) ** 2)
    s = search.GeneticSearch(rng, population_size=20, mutation_rate=0.8, seeds=dsl.seed_genomes())

    _drain_and_score(s, fitness)
    seed_best = max(f for _, f in s.history)
    for _ in range(15):
        s.evolve()
        _drain_and_score(s, fitness)

    best_genome, best_fit = s.best(1)[0]
    assert best_fit > seed_best  # learned something better than any seed
    assert abs(best_genome["risk"]["atr_stop_mult"] - target) < 1.0
    assert s.generation == 15


def test_persistence_resumes_population_and_scores(tmp_path):
    rng = np.random.default_rng(5)
    fitness = lambda g: -((g["risk"]["atr_stop_mult"] - 3.0) ** 2)
    s = search.GeneticSearch(rng, population_size=10, seeds=dsl.seed_genomes())
    _drain_and_score(s, fitness)
    s.evolve()
    path = tmp_path / "pop.json"
    s.save(path)

    resumed = search.GeneticSearch.load(path, np.random.default_rng(9))
    assert resumed.generation == s.generation
    assert resumed.scores == s.scores
    assert [dsl.genome_id(g) for g in resumed.population] == [dsl.genome_id(g) for g in s.population]
    assert len(resumed.history) == len(s.history)


def test_resume_or_new(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "pop.json"
    fresh = search.resume_or_new(path, rng, population_size=8)
    assert fresh.generation == 0
    fresh.save(path)
    again = search.resume_or_new(path, np.random.default_rng(1))
    assert again.generation == 0
    assert again.population_size == 8


# ── evaluating + validating genomes on real data ─────────────────────────────

def _trending(n=900, seed=2):
    closes = 100 + np.cumsum(np.random.default_rng(seed).normal(0.15, 0.7, n))
    return make_ohlcv(np.abs(closes) + 30, freq="4h")


def test_evaluate_genome_returns_trial():
    df = _trending()
    trial = search.evaluate_genome(df, "GOLD", "4h", dsl.seed_trend_following())
    assert trial is not None
    assert not np.isnan(trial.oos_sharpe)


def test_validate_genome_returns_full_verdict():
    df = _trending()
    from fablebot.optimizer import split_search_holdout
    s_df, h_df = split_search_holdout(df)
    verdict = search.validate_genome(
        s_df, h_df, "GOLD", "4h",
        candidate=dsl.seed_trend_following(fast_ema=15, slow_ema=40),
        current=dsl.seed_trend_following(fast_ema=20, slow_ema=50),
        rng=np.random.default_rng(0),
    )
    assert set(verdict) >= {
        "qualified", "folds_won", "folds_valid", "deflated_sharpe",
        "holdout_candidate_sharpe", "holdout_current_sharpe", "reason",
    }
    assert isinstance(verdict["qualified"], bool)


def test_validate_genome_rejects_identical_candidate():
    """A genome identical to the current one can't beat itself in any fold."""
    df = _trending()
    from fablebot.optimizer import split_search_holdout
    s_df, h_df = split_search_holdout(df)
    g = dsl.seed_trend_following()
    verdict = search.validate_genome(s_df, h_df, "GOLD", "4h", candidate=g, current=g, rng=np.random.default_rng(0))
    assert verdict["qualified"] is False
    assert verdict["folds_won"] == 0
