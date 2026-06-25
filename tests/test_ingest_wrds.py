"""WRDS/CRSP ingester: split a multi-ticker export into per-symbol adjusted CSVs.
Runs the script end to end on a tiny synthetic CRSP file."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "ingest_wrds.py"


def _read_csv(path):
    rows = [line.split(",") for line in path.read_text().splitlines()]
    return rows[0], rows[1:]


def test_ingest_splits_adjusts_and_filters(tmp_path):
    # CRSP legacy schema: PRC negative = bid/ask midpoint; CFACPR adjusts for splits.
    raw = tmp_path / "crsp.csv"
    raw.write_text(
        "PERMNO,date,TICKER,OPENPRC,ASKHI,BIDLO,PRC,VOL,CFACPR\n"
        "1,2026-01-02,AAPL,100,102,99,101,1000,1\n"
        "1,2026-01-03,AAPL,50,51,49,-50.5,800,1\n"      # negative PRC -> magnitude
        "1,2026-01-06,AAPL,198,202,196,200,500,2\n"     # 2:1 split: /2 -> 100, vol *2 -> 1000
        "9,2026-01-02,ZZZ,5,5,5,5,10,1\n"               # not in the tickers list -> skipped
    )
    tickers = tmp_path / "tickers.txt"
    tickers.write_text("AAPL\n")

    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(raw),
         "--tickers", str(tickers), "--out-dir", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr

    aapl = tmp_path / "AAPL.csv"
    assert aapl.exists()
    assert not (tmp_path / "ZZZ.csv").exists()          # filtered out
    header, rows = _read_csv(aapl)
    assert header == ["date", "open", "high", "low", "close", "volume"]
    assert len(rows) == 3
    assert rows[0][4] == "101.0"                          # unadjusted close
    assert rows[1][4] == "50.5"                           # |−50.5|
    assert rows[2][4] == "100.0" and rows[2][5] == "1000.0"  # split-adjusted close + volume
