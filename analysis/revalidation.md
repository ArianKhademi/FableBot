> Historical paper-trading record, not a current performance claim or a reproducible benchmark.
> Raw account/data files are intentionally not included. See [validation limitations](../docs/validation.md).

# Re-validation of old-qualified proposals under the new gate

Friction: 3.0 bps. Tested 984 proposals that qualified under the old gate.

**0/984 (0.0%) still qualify under the new gate.**

| Instrument | old-qualified | survive new | survival % | top cull |
|---|---|---|---|---|
| OIL | 506 | 0 | 0% | improvement |
| GOLD | 237 | 0 | 0% | improvement |
| BTC | 167 | 0 | 0% | deflated_sharpe |
| NDX | 62 | 0 | 0% | deflated_sharpe |
| SPX | 12 | 0 | 0% | improvement |

Cull stage totals (where the old survivors now fail):
- improvement: 555
- deflated_sharpe: 354
- trade_floor: 75

Applied: demoted 984 proposals to 'rejected' in `state/proposals.jsonl`.
