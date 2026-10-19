# Architecture

FableBot is a paper-trading and strategy-research project. Strategies propose
signals; the risk layer sizes them; `PaperBroker` simulates execution. Research
searches propose alternatives for human review rather than automatically
rewriting the running configuration.

## Components

| Component | Responsibility |
| --- | --- |
| `fablebot/indicators.py`, `strategies/` | Backward-looking indicators and named signal strategies |
| `fablebot/dsl.py` | Composable features, conditions and risk parameters; seeded genomes |
| `fablebot/risk.py` | Validated sizing inputs and configured correlation groups |
| `fablebot/portfolio.py` | Paper positions, fills, stops, trailing, excursions and persistence |
| `fablebot/engine.py` | Polling, stale-feed handling, signals, paper accounts and journals |
| `fablebot/backtest.py` | Single-instrument historical replay through sizing and paper stops |
| `fablebot/search.py`, `optimizer.py` | Genetic and TPE search, train/evaluation splits and validation |
| `fablebot/ml.py`, `rl.py` | NumPy GBT and tabular Q-learning signal models |
| `fablebot/pipeline.py` | Append-only proposal status records |
| `fablebot/data.py` | Yahoo, Stooq, Binance, local CSV and cache adapters |
| `fablebot/journal.py`, `analytics.py` | Decision records, excursions and aggregate analysis |
| `fablebot/dashboard.py`, `reporting.py` | Account views and reports from paper state |
| `fablebot/prop_rules.py` | Experimental account drawdown and contract-cap simulations |

## Data and state boundaries

The paper loop uses external price feeds. Research can use locally supplied daily
CSVs, with fallback to a network source when a matching file is absent. Source
selection and caching are configurable, including per-instrument overrides.
Neither path is the offline demo.

Each paper book can use its own `state*` directory. Journals, portfolios, cached
prices, search populations and proposal logs are generated local state. They are
not fixtures and are not needed by the synthetic test suite. Existing CLIs use
some repository-relative defaults; run advanced operations from the repository
root and pass explicit paths when maintaining separate books.

The introductory `scripts/demo.py` imports the backtester and named/DSL
strategies only. It creates synthetic prices, uses no external source or account
configuration, and emits a static dashboard with embedded SVG charts. The
operational dashboards remain separate and load Chart.js from a CDN.

## Governance and limitations

[RISK_RULES.md](../RISK_RULES.md) describes the risk budget and stop policy.
Stops are modeled exit instructions, not guaranteed prices or loss caps.
[Validation behavior](validation.md) documents skipped checks, effective-trial
heuristics, holdout reuse and simulation assumptions.

The optional nightly reviewer can modify code and open a draft PR. It is not a
sandboxed guarantee that risk rules cannot change. It is excluded from normal
setup, tests and the demo. See [operations](operations.md) and the
[learning loop](learning-loop.md).

The original [phase-two brief](phase-2-learning-engine.md) is retained as
historical design context. Dated analysis files are historical paper records,
not a verified performance benchmark.
