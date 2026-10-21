# Optional operations

Start with the offline demo in the [README](../README.md). The commands below
are advanced workflows: some contact market-data providers, write account/search
state, or operate on GitHub. They are not part of installation or offline CI.
Run them from the repository root using the isolated environment's Python.

## Paper accounts and market data

```sh
.venv/bin/python scripts/run_bot.py --config config.yaml --state-dir state
.venv/bin/python scripts/run_bot.py --config config_experiment_trail.yaml --state-dir state_experiment_trail
```

These are long-running **paper** loops, with external price feeds. `--once` runs
one evaluation cycle; it still fetches prices and can modify the selected paper
account. `--report morning` fetches news; reports may send to a webhook if
`FABLEBOT_WEBHOOK_URL` is set. Each book needs its own state directory.

The main configuration starts with daily index mean reversion, hourly Bitcoin
momentum, and four-hour commodity trend following. Experiment configs retain
historical alternatives: fixed/adaptive trailing, regime switching, trend
filters, observed-price stops and simplified futures-account rules. Their names
and settings are experiments, not recommendations or certified prop-firm rules.

Data adapters include Yahoo Finance, Stooq, Binance and local daily CSVs. APIs
can change, rate-limit or become unavailable. See [data rights](wrds-data.md).
`data.csv_dir` enables local history for matching symbols, but missing files fall
back to a network feed. It does not make an arbitrary command offline.

## Research commands

```sh
.venv/bin/python scripts/backtest.py BTC --days 120
.venv/bin/python scripts/run_engine.py --cycles 1
.venv/bin/python scripts/evolve.py --generations 10
.venv/bin/python scripts/train_ml.py --instrument BTC
.venv/bin/python scripts/train_rl.py --instrument BTC
```

These fetch history and can write cache, populations, proposal or run logs. A
continuous `run_engine.py --cycles 0` runs indefinitely. Check each script's
`--help` and use dedicated research state before running an experiment. The ML/RL
trainers currently skip DSR because their six seed runs do not meet the ten-trial
threshold; [verdict flags and limitations](validation.md) explain the result.

## Reports and dashboards

```sh
.venv/bin/python scripts/analyze.py --days 30
.venv/bin/python scripts/proposals.py
.venv/bin/python scripts/engine_status.py
.venv/bin/python scripts/dashboard.py --state-dir state --no-marks
.venv/bin/python scripts/bot_report.py --no-marks --out reports/paper-summary.md
```

Without `--no-marks`, the dashboard/report scripts can fetch live prices. The
legacy HTML dashboards reference Chart.js on a CDN. Only the introductory
synthetic dashboard embeds all its visual assets. The `--open` options use a
macOS helper; elsewhere, open the resulting HTML file manually.

## Validation

```sh
.venv/bin/python scripts/validate.py --skip-backtests
```

This runs software tests only. Omitting `--skip-backtests` also downloads recent
market history. The network-backed validation path is retained for compatibility;
it is not a reproducible performance benchmark and is not used by offline CI.

## Optional reviewer and scheduling

`scripts/nightly_review.sh` requires authenticated `claude`, `gh` and Git. It can
create branches, edit source, commit, push and open draft PRs. It also calls
`gh auth setup-git`. Do not run it to verify installation. Use a dedicated
checkout, review `prompts/reviewer.md`, and understand the permissions first.
Nothing in the normal setup installs a cron job or enables this reviewer.

Existing overrides are preserved:

- `FABLEBOT_PYTHON`: explicitly select the intended environment, for example
  `/path/to/FableBot/.venv/bin/python`. The legacy default remains
  `/usr/bin/python3` for existing macOS cron users; set this override elsewhere.
- `FABLEBOT_SECRETS`: optional local shell file loaded by the automation. Its
  default is `$HOME/.config/fablebot/secrets.env`. Never commit it or include it
  in a public archive. A shell file is executable code and must be trusted.
- `FABLEBOT_WEBHOOK_URL`: optional report destination; omit for local output.

`promote_when_flat.sh` can change a PR's draft status and send notifications.
`bots.sh`, `flatten_book.py` and `restore_book.py` can affect running processes
or saved paper books. They are maintenance tools, not demonstration commands.
Use explicit paths in any scheduler. Keep historical account/data directories
and their backups outside release artifacts.
