# Lead Review: 023-NG-P4 Airfoil Natural

**Reviewed:** 2026-08-23  
**Verdict:** FUNCTIONAL PASS / comparison pending

## Verified

- All five frozen split hashes match 022; source status is unchanged.
- Five fold results, 750 predictions, positive scales, finite interval bounds, aggregates, and checksums are internally consistent.
- Mean NLL `2.2180`, RMSE `2.3884`, and nominal coverage 0.365/0.679/0.901 are supported.
- Four of five folds hit cap 2,000, and under-coverage is explicit.

## Decision

Accept natural-model feasibility and preserve its cap/calibration limitations. No superiority or overall pilot verdict is allowed until ordinary and homoscedastic variants complete on the same folds. No fix required.
