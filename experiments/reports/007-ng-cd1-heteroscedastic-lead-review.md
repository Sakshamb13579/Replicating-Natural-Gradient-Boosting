# Lead Review: 007-NG-CD1 Heteroscedastic Dataset

**Reviewed:** 2026-08-23  
**Verdict:** PASS / FROZEN  
**Status:** `AUDIT_FROZEN`

## Verified

- Generator SHA-256 matches `2ac6fe06d86ce368eb9e27287fa601814fcd494ffc8a3398fa9efd5b2302c126`.
- Data hashes match the report and manifest.
- Independent parsing confirms 12,000 finite regime rows, 4,000 rows per regime, 4,000 unique base IDs, exact 7,200/2,400/2,400 combined split counts, and identical base inputs/noise/split across regimes.
- Generator uses independent explicit PCG64 RNG instances for data seed 0 and split seed 1.
- Regime B is a valid constant-noise control; natural gradient is not required to win it.
- Hypotheses and metrics directly test conditional uncertainty rather than arbitrary synthetic accuracy.

## Decision

Accept and freeze the generator, data, and split manifest. Future model handoffs must not regenerate or alter them. Each regime-model pair remains a separate execution task. No fix is required.

Checkpoint `007-NG-CD1` is closed.
