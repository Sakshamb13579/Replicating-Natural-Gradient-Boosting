# Lead Review: 017-NG-D1 Yacht Estimator Cap

**Reviewed:** 2026-08-23  
**Fix:** `017F1-NG-D1`  
**Verdict:** CONTROLLED PASS  
**Decision:** Retain cap 2,000

## Verified

- a01 is preserved and correctly reclassified as confounded because cap-dependent global RNG consumption changes later inner validation splits in the official script.
- a02 uses frozen outer/inner indices and shared 4,000-stage natural-model trajectories for both cap views.
- Independent recomputation confirms 20 folds, mean NLL `2.0351` at cap view 2,000 and `2.2029` at cap view 4,000, mean delta `+0.16779`, and paired outcomes 2 wins / 5 ties / 13 losses for extra headroom.
- Split index lengths/uniqueness and source status checks pass.
- Fold-1 post-hoc verification confirms `max_iter` views match the staged shared trajectory. The in-run verification flags are historical and known false because of a checking bug, not a model-run failure.

## Decision

Retain `n_estimators=2000` for protocol fidelity and better controlled probabilistic performance. Disclose cap saturation and the negative controlled sensitivity result. Do not claim cap saturation explains the absolute Yacht paper mismatch.

The paper/script RMSE-versus-NLL selector discrepancy remains a separate diagnostic question. Checkpoint `017-NG-D1` is closed after one fix cycle.
