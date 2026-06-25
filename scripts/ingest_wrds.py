#!/usr/bin/env python3
"""Split a big WRDS/CRSP daily export into per-symbol, split-adjusted CSVs that
the search reads via ``config.data.csv_dir``.

WRDS hands you ONE CSV for the whole query (every ticker, every day). This
*streams* it — so a 50 GB+ file never loads into memory — keeps only the tickers
you care about, applies CRSP's cumulative price-adjustment factor (so a split
doesn't look like a crash), and writes ``data/<TICKER>.csv`` in the OHLCV format
``CSVSource`` expects. You don't need to move the download; point ``--input`` at
it wherever it is.

  python3 scripts/ingest_wrds.py --input ~/Downloads/crsp_daily.csv
  python3 scripts/ingest_wrds.py --input data/crsp_raw.csv --tickers data/wrds_tickers.txt --out-dir data
  python3 scripts/ingest_wrds.py --input ~/Downloads/crsp_daily.csv --limit 200000   # quick dry run
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Case-insensitive column aliases — covers CRSP's legacy and CIZ (v2) schemas.
_ALIASES = {
    "ticker": ("ticker", "htick", "hticker", "ticker_x"),
    "date":   ("date", "caldt", "dlycaldt", "mthcaldt", "datadate", "dlydate"),
    "open":   ("openprc", "dlyopen", "open"),
    "high":   ("askhi", "dlyhigh", "high"),
    "low":    ("bidlo", "dlylow", "low"),
    "close":  ("prc", "dlyprc", "dlyclose", "close"),
    "volume": ("vol", "dlyvol", "volume"),
    "adj":    ("cfacpr", "dlyfacpr", "dlycfacpr"),  # cumulative price adjustment factor
}


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest a WRDS/CRSP daily export into per-symbol adjusted CSVs")
    ap.add_argument("--input", required=True, help="the big CRSP daily CSV from WRDS")
    ap.add_argument("--tickers", default=ROOT / "data/wrds_tickers.txt")
    ap.add_argument("--out-dir", default=ROOT / "data")
    ap.add_argument("--limit", type=int, default=0, help="stop after N input rows (quick dry run)")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"input not found: {inp}", file=sys.stderr)
        return 1
    want = {t.strip().upper() for t in Path(args.tickers).read_text().split() if t.strip()}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with inp.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        index = {h.strip().lower(): i for i, h in enumerate(header)}
        col = {field: next((index[a] for a in al if a in index), None) for field, al in _ALIASES.items()}
        print("[ingest] detected columns:",
              {f: (header[col[f]] if col[f] is not None else "—") for f in _ALIASES})
        for need in ("ticker", "date", "close"):
            if col[need] is None:
                print(f"[ingest] ERROR: no '{need}' column found in header:\n  {header}", file=sys.stderr)
                return 2
        if col["adj"] is None:
            print("[ingest] note: no CFACPR adjustment column found — writing raw prices.", file=sys.stderr)

        writers, handles, counts = {}, {}, {}

        def writer_for(tk: str):
            if tk not in writers:
                h = (out_dir / f"{tk}.csv").open("w", newline="")
                w = csv.writer(h)
                w.writerow(["date", "open", "high", "low", "close", "volume"])
                handles[tk], writers[tk], counts[tk] = h, w, 0
            return writers[tk]

        rows = 0
        for r in reader:
            rows += 1
            if args.limit and rows > args.limit:
                break
            if rows % 5_000_000 == 0:
                print(f"[ingest] {rows:,} rows scanned…", flush=True)
            try:
                tk = (r[col["ticker"]] or "").strip().upper()
            except IndexError:
                continue
            if tk not in want:
                continue
            close = _num(r[col["close"]])
            if close is None:
                continue
            cf = _num(r[col["adj"]]) if col["adj"] is not None else None
            cf = cf if (cf and cf != 0.0) else 1.0

            def adj(field):
                i = col[field]
                v = _num(r[i]) if i is not None else None
                return abs(v) / cf if v is not None else None

            c = abs(close) / cf
            o, hi, lo = adj("open"), adj("high"), adj("low")
            vol = _num(r[col["volume"]]) if col["volume"] is not None else None
            writer_for(tk).writerow([
                r[col["date"]], o if o is not None else c, hi if hi is not None else c,
                lo if lo is not None else c, c, (vol * cf) if vol is not None else 0.0,
            ])
            counts[tk] += 1
        for h in handles.values():
            h.close()

    print(f"\n[ingest] scanned {rows:,} rows; wrote {len(writers)} symbols to {out_dir}/")
    for tk in sorted(counts):
        print(f"    {tk:6} {counts[tk]:>8,} bars")
    missing = sorted(want - set(writers))
    if missing:
        print(f"  (no rows found for: {missing} — check the ticker spelling in the export)")
    print("\nNext: set `csv_dir: data` under `data:` in config.yaml to use these in the search.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
