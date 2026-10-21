"""Exercise the offline CLI in-process so the test network guard covers it."""

import json
from pathlib import Path
import runpy
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/demo.py"


def run_demo(monkeypatch, output):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output", str(output), "--seed", "7"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    return exit_info.value.code


def test_demo_is_deterministic_offline_and_isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real_state = tmp_path / "state"
    real_state.mkdir()
    sentinel = real_state / "portfolio.json"
    sentinel.write_text("do not read or overwrite")
    first, second = tmp_path / "one", tmp_path / "two"
    assert run_demo(monkeypatch, first) == 0
    assert run_demo(monkeypatch, second) == 0
    assert sentinel.read_text() == "do not read or overwrite"
    assert {p.name for p in first.iterdir()} == {"summary.json", "synthetic.csv", "dashboard.html"}
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    summary = json.loads((first / "summary.json").read_text())
    assert summary["data_kind"] == "synthetic"
    assert len(summary["results"]) == 4
    assert all(r["trades"] > 0 for r in summary["results"])
    # Named MR and its DSL equivalent should make the same decisions.
    assert summary["results"][0]["total_pnl"] == summary["results"][3]["total_pnl"]
    html = (first / "dashboard.html").read_text()
    assert "<svg" in html and "<script" not in html
    assert "https://" not in html and "http://" not in html


def test_demo_refuses_existing_output(tmp_path, monkeypatch):
    sentinel = tmp_path / "portfolio.json"
    sentinel.write_text("untouched")
    assert run_demo(monkeypatch, tmp_path) == 2
    assert sentinel.read_text() == "untouched"
    assert not (tmp_path / "dashboard.html").exists()
