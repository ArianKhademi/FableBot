"""The position-flat detector that gates config changes which must not land
mid-trade (e.g. NDX -> daily). A wrong 'flat' would let a switch orphan an open
position, so the unknown/missing case must never read as flat."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location("position_flat", ROOT / "scripts" / "position_flat.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_open_and_flat_classification():
    m = _mod()
    pf = {"positions": {"GOLD": {}, "NDX": {}, "OIL": {}}}
    assert m.open_instruments(pf) == {"GOLD", "NDX", "OIL"}
    assert m.is_flat(pf, "NDX") is False     # open
    assert m.is_flat(pf, "SPX") is True      # absent => flat


def test_empty_portfolio_is_flat():
    m = _mod()
    assert m.is_flat({"positions": {}}, "NDX") is True
    assert m.is_flat({}, "NDX") is True       # no positions key at all


def test_classify_exit_codes(tmp_path):
    m = _mod()
    p = tmp_path / "portfolio.json"

    p.write_text(json.dumps({"positions": {"NDX": {"direction": "short"}}}))
    assert m.classify(p, "NDX")[0] == m.OPEN     # 1

    p.write_text(json.dumps({"positions": {"GOLD": {}}}))
    assert m.classify(p, "NDX")[0] == m.FLAT     # 0

    assert m.classify(tmp_path / "missing.json", "NDX")[0] == m.UNKNOWN  # 2


def test_unreadable_file_is_unknown_not_flat(tmp_path):
    """Critical: corrupt/partial state must NOT be reported as flat, or the
    watcher could promote a switch mid-trade."""
    m = _mod()
    p = tmp_path / "portfolio.json"
    p.write_text("{not valid json")
    code, _ = m.classify(p, "NDX")
    assert code == m.UNKNOWN
    assert code != m.FLAT


def test_nondict_positions_is_unknown_not_flat(tmp_path):
    """Parseable but structurally wrong `positions` must be UNKNOWN, never FLAT.
    Most dangerous case: `{"positions": ["NDX"]}` lists NDX open yet has no dict
    keys — coercing it to an empty set would read as FLAT and orphan the trade."""
    m = _mod()
    p = tmp_path / "portfolio.json"
    for payload in ('{"positions": null}', '{"positions": ["NDX","GOLD"]}',
                    '{"positions": "NDX"}', '{"positions": 5}'):
        p.write_text(payload)
        code, _ = m.classify(p, "NDX")
        assert code == m.UNKNOWN, payload
        assert code != m.FLAT, payload


def test_nondict_toplevel_is_unknown_not_flat(tmp_path):
    """A bare list/string/number/null/bool at top level is corrupt state, not
    flat — and must not crash with a traceback either."""
    m = _mod()
    p = tmp_path / "portfolio.json"
    for payload in ('["NDX"]', '"NDX"', 'null', '42', 'true'):
        p.write_text(payload)
        code, _ = m.classify(p, "NDX")
        assert code == m.UNKNOWN, payload
