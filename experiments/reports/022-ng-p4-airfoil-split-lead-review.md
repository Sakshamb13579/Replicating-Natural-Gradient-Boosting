# Lead Review: 022-NG-P4 Airfoil Split Freeze

**Reviewed:** 2026-08-23  
**Verdict:** PASS / PILOT_READY

## Verified

- Processed CSV and split-file hashes match.
- Prep code preserves all 1,503 rows and numeric values with row IDs.
- Each of five folds has 1,353/150 outer and 1,082/271 inner counts, disjointness, full coverage, and no test leakage.
- Outer construction, explicit inner seeds, paired reuse, metrics, cap, and stage selector are predeclared.
- Grouped-condition alternative is disclosed but not mixed into the pilot.

## Decision

Freeze `airfoil-self-noise-v1` for three separate model-variant handoffs. No fix required. Checkpoint `022-NG-P4` is closed.
