# Lead Review: 024-NG-P4 Airfoil Ordinary

**Reviewed:** 2026-08-23  
**Verdict:** FUNCTIONAL PASS / provisional natural advantage

## Verified

- Same five split hashes, unchanged runner/source, finite outputs, and complete predictions/artifacts.
- Ordinary mean NLL `2.8319`, RMSE `5.0046`, and 5/5 cap hits are supported.
- Paired calculation confirms natural has lower NLL on 5/5 folds with mean ordinary-minus-natural delta `+0.6138`.
- Wider ordinary intervals explain near-nominal 95% coverage despite much worse RMSE; coverage alone is not superiority.

## Decision

Accept ordinary-model execution and the provisional two-way natural advantage. Final Airfoil verdict remains blocked on homoscedastic completion. No fix required.
