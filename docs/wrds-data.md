# Using WRDS / CRSP daily data for the search

The research adapters can read locally supplied daily CSVs, including exports
from CRSP through WRDS. The live feed is unchanged. The public synthetic demo
does not use this interface or require any vendor data.

> **Data rights:** use only data you are entitled to access under your
> institution's and vendor's agreements. Those agreements are not part of this
> repository. Do not commit exports, caches, credentials, or real-data screenshots.
> A code license would not grant redistribution rights to third-party data.

The repository includes a CSV reader and streaming ingester, not a WRDS download
client. The fixed ticker list is not a point-in-time universe: importing CRSP does
not by itself eliminate survivorship bias or adjustment errors.

## 1. Export the data

From WRDS (web query: *Get Data → CRSP → Stock / Security Files → Daily Stock
File*, or the `wrds` Python package), pull **daily** OHLCV for each symbol you
want and save one CSV per symbol. Two layouts are accepted automatically:

- **Standard** — `Date,Open,High,Low,Close,Volume`
- **CRSP** — `caldt,OPENPRC,ASKHI,BIDLO,PRC,VOL` (the reader takes the magnitude
  of `PRC`, since CRSP stores it negative when it's a bid/ask midpoint)

A close-only file (`Date,Close`) also works — the missing OHLC is filled from
close so it still backtests.

## 2. Drop them in a folder

```
data/
  SPY.csv
  QQQ.csv
  GLD.csv
  ...
```

Files are matched by symbol. Index symbols with a caret match the bare name too
(`^GSPC` → `GSPC.csv`). Anything without a CSV falls back to the normal feed, so
you can add symbols incrementally.

## 3. Point the config at it

```yaml
data:
  source: yfinance
  cache_dir: state/data_cache
  csv_dir: data        # <- enables the local CSV history feed
```

That's it. The next `run_engine` / `rotate_universe` / `evolve` run reads daily
bars for any symbol with a CSV from `data/`, and the live `run_bot` is unchanged.
`csv_dir` is **opt-in** — leave it unset and behaviour is exactly as before.

## Scope

CSV history covers daily bars only; intraday history still uses the configured
network source. Close-only input is supported for compatibility, but filling
missing OHLC from the close cannot reproduce intrabar stop behavior. Check input
provenance, corporate actions and universe construction before interpreting a
backtest. Better data alone does not validate a strategy.
