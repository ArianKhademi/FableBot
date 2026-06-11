# Risk rules — the constitution

These invariants are **not tunable**. The learning loop may rewrite strategy
logic, add filters, change parameters, even replace a strategy wholesale —
but every proposal must leave these intact, and the test suite enforces them
(`tests/test_invariants.py`). A PR that touches any of these must say so in
its title and will not be auto-anything: it waits for an explicit human yes.

1. **Hard per-trade stop.** Every position has a stop at entry, priced so
   that hitting it loses no more than `account.risk_per_trade` of equity.
   No strategy may open a position without one, widen one after entry, or
   override the broker's stop enforcement.

2. **ATR-based sizing.** Position size is always risk budget ÷ stop
   distance. No fixed sizes, no martingale, no "double up after a loss."

3. **Gap protection stays on.** The broker force-closes any position whose
   unrealized loss reaches its risk budget, even if the stop price never
   printed. This check cannot be bypassed or made conditional.

4. **One position per instrument.** No pyramiding, no averaging down.

5. **The correlation filter may be tuned, not deleted.** Its group
   membership and threshold are fair game with evidence; removing the
   concept of correlated-exposure limits is not.

6. **Paper trading only.** No change may wire the bot to a live broker or
   real funds. That decision belongs to a human, made outside this loop.

7. **Risk budget ceiling.** `risk_per_trade` may never exceed 0.01 (1%).
   The learning loop may propose lowering it, never raising it.
