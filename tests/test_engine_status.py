"""engine_status's population classifier: a parked population (not updated by the
current run, e.g. while running --universe) must not be reported as 'spinning' or
told to restart for the immigrant fix."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location("engine_status", ROOT / "scripts" / "engine_status.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_stale_population_is_parked_not_spinning():
    m = _mod()
    # Saturated (95%) but untouched for 10h: parked, NOT spinning, no restart nag.
    s = m.classify_population(0.95, 600)
    assert "parked" in s
    assert "spinning" not in s
    assert "restart" not in s
    # Even a low-saturation stale pop reads as parked (it isn't being worked).
    assert "parked" in m.classify_population(0.2, 600)


def test_fresh_population_uses_saturation_bands():
    m = _mod()
    assert "spinning" in m.classify_population(0.95, 1.0)        # active + saturated
    assert m.classify_population(0.6, 1.0) == "nearing convergence"
    assert m.classify_population(0.2, 1.0) == "exploring"


def test_parked_threshold_boundary():
    m = _mod()
    assert "parked" not in m.classify_population(0.95, m.PARKED_MIN - 1)  # just-active
    assert "parked" in m.classify_population(0.95, m.PARKED_MIN + 1)      # just-parked
