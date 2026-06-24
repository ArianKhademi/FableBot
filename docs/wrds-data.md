# Using WRDS / CRSP daily data for the search

yfinance is fine for the *live* feed, but its daily history is shallow and the
adjustments are rough — which is part of why nothing has cleared the gate. If you
have WRDS access, you can feed the **optimizer and genome search** clean, deep,
survivorship-bias-free CRSP daily bars instead. The live feed stays on
yfinance/Binance; only the history/search path reads the CSVs.

> **Licensing.** WRDS data is licensed through your institution, typically for
> academic research. Confirm your use is permitted before wiring it in. This repo
> only *reads* CSVs you place locally — it never redistributes WRDS data.

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

This improves the *quality of the search data*, not its outcome — clean data
removes the "bad data" excuse and makes the gate's verdicts trustworthy; it does
not by itself manufacture a profitable strategy. Intraday history (1h/4h) still
comes from the live sources; CSV covers daily only.
