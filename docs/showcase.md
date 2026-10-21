# Historical activity showcase

The [interactive preview](https://ariankhademi.github.io/FableBot/) and
[recording](media/fablebot-showcase.mp4) show aggregate activity from existing
paper-trading outputs. They do not run a strategy, fetch prices, or place trades.
Open `index.html` from a local clone to use the same preview offline.

## Source and scope

The source was ten local `state*/portfolio.json` files. Recorded opens and closes
span June 10–September 15, 2026 UTC. This is the coverage of saved activity, not
a claim about continuous underlying price history. The snapshot was prepared on
October 21, 2026 UTC.

Current configurations specify Yahoo Finance through `yfinance`. Individual
trade records do not preserve provider identity or the exact input bars, so
per-fill provenance cannot be established from these outputs alone.

The published HTML contains book names, counts by instrument and strategy,
recorded date ranges, open-position counts, and archive-comparison labels.
The recording captures only that review page. It includes no raw trade rows,
prices, balances, credentials, private paths, or unrelated desktop content.
Raw inputs and the local provenance manifest remain on the owner's machine.

## History limitations

Four current books are shorter prefixes of their pre-rollback archives:

| Book | Retained closed records | Later archived records omitted |
| --- | ---: | ---: |
| Adaptive trailing + breaker | 95 | 12 |
| Observed stops | 109 | 18 |
| Trend filter | 47 | 4 |
| Shorts-only filter | 82 | 17 |

Matching an archive does not independently audit a book's complete history.
Some books have no matching pre-rollback archive. Periods and configurations
differ, and books may share history, so counts must not be treated as independent
samples or compared as a performance ranking. The preview deliberately makes
no return, profitability, or live-brokerage claim.

## Reproducible software example

The separate `scripts/demo.py` and synthetic test fixtures remain available for
reproducible, offline software checks. They are not the source of this showcase.
