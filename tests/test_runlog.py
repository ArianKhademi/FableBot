"""The scheduled-job run log (proof a cron fired)."""

from fablebot.runlog import record_run, recent_runs


def test_record_and_recent_roundtrip(tmp_path):
    p = tmp_path / "run_log.jsonl"
    assert recent_runs(path=p) == []  # nothing yet
    record_run("train_ml", "0/5 qualified", path=p)
    record_run("rotate_universe", "add=['SPY'] drop=none", path=p)
    rows = recent_runs(limit=10, path=p)
    assert [r["job"] for r in rows] == ["train_ml", "rotate_universe"]
    assert rows[-1]["summary"] == "add=['SPY'] drop=none"
    assert "ts" in rows[0]


def test_recent_runs_respects_limit(tmp_path):
    p = tmp_path / "run_log.jsonl"
    for i in range(20):
        record_run("train_ml", f"run {i}", path=p)
    rows = recent_runs(limit=5, path=p)
    assert len(rows) == 5
    assert rows[-1]["summary"] == "run 19"  # most recent kept
