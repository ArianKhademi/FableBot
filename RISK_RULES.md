# Risk rules

These are project policy and tested software constraints for paper trading.
Changes to them require explicit human review. A passing test suite does not
make a stop fill certain or establish a maximum realized loss.

1. **Stop at entry.** Strategies routed through the engine/backtester receive
   an ATR-based stop. The intended loss at that stop is the sizing budget before
   execution friction. Gap fills, slippage and polling delays can exceed it.
   Strategy hooks must not bypass enforcement or widen a placed stop; trailing
   helpers only tighten it.
2. **ATR-based sizing.** Units equal risk budget divided by stop distance.
   Futures simulations round down to whole contracts and can apply a contract
   cap. No martingale, averaging down or fixed-size substitution.
3. **Stop enforcement.** The paper broker checks modeled bar ranges (or the
   engine's observed-price mode) and closes on a stop or risk-budget breach.
   This is simulated execution, not protection against a real market gap.
4. **One position per instrument.** The paper broker rejects pyramiding.
5. **Configured correlation groups.** The engine blocks additional
   same-direction positions at the configured group limit. Groups are manually
   assigned categories, not dynamically estimated correlations.
6. **Paper only.** No real-money broker adapter. Adding one is a separate human
   decision outside the research/reviewer loop.
7. **Risk budget ceiling.** Account construction and sizing reject non-finite,
   zero, negative, or greater-than-1% risk fractions. Finite positive equity and
   stop-distance inputs are required. This ceiling controls intended sizing,
   not realized losses after execution.

The relevant regression tests include `tests/test_invariants.py`,
`tests/test_risk.py`, `tests/test_portfolio.py`, `tests/test_trailing.py`,
`tests/test_observed_stops.py` and `tests/test_prop_rules.py`. Direct construction
of a `Position` remains a low-level API; the proposal/reviewer workflow is not a
security sandbox around arbitrary code changes.
