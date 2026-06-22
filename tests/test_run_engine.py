"""run_engine target selection: the --universe flag must let a converged engine
search the broad scan list (genome-only) instead of spinning on the traded set."""

import importlib.util
from pathlib import Path

from fablebot import dsl, search
from fablebot.config import load_config

ROOT = Path(__file__).resolve().parent.parent


def _run_engine():
    spec = importlib.util.spec_from_file_location("run_engine", ROOT / "scripts" / "run_engine.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_targets_are_the_traded_instruments():
    mod = _run_engine()
    config = load_config(ROOT / "config.yaml")
    targets = mod.select_targets(config, universe=False)
    keys = {t.key for t in targets}
    assert keys == set(config.instruments)        # exactly the traded book
    assert "SPX" in keys
    assert all(t.param_bounds for t in targets)   # traded set has a numeric search space


def test_universe_targets_are_the_scan_list_without_param_bounds():
    mod = _run_engine()
    config = load_config(ROOT / "config.yaml")
    targets = mod.select_targets(config, universe=True)
    symbols = {t.symbol for t in targets}
    assert {"SPY", "QQQ", "BTC-USD"} <= symbols   # broad daily scan list
    assert len(targets) >= 10
    # No param_bounds -> the loop's `if inst.param_bounds` guard skips the numeric
    # tuner for these, so they get the genome (structure) search only.
    assert all(not t.param_bounds for t in targets)


def test_universe_instruments_yield_valid_seed_genomes():
    """The genome search seeds each instrument with current_genome_for(); universe
    instruments have empty params, so this must still build a valid genome."""
    mod = _run_engine()
    config = load_config(ROOT / "config.yaml")
    for inst in mod.select_targets(config, universe=True):
        g = search.current_genome_for(inst)
        assert dsl.genome_id(g)                    # builds + hashes -> structurally valid
