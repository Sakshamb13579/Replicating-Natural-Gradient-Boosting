# Lead Review: 028-NG-P5 Heteroscedastic Natural

**Reviewed:** 2026-08-23  
**Verdict:** STRONG FUNCTIONAL PASS / comparison pending

## Verified

- Frozen CSV/split identities pass; model inputs contain only `x1,x2`.
- Separate natural models execute for regimes A/B/C with validation-NLL stage selection, no cap hits, finite positive scales, and complete predictions.
- Regime A/C scale correlations, MAE, oracle/model NLL gaps, coverage, and fixed-bin summaries are internally consistent and monotonic.
- Constant regime B correctly reports null true-scale correlation and records mild spurious predicted variation.
- Source/environment remain unchanged.

## Decision

Accept natural-model feasibility and strong ground-truth scale recovery. Do not claim natural-gradient superiority until ordinary and homoscedastic variants finish. No fix required.
