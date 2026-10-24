# Lead Review: 025-NG-P4 Airfoil Three-Way Pilot

**Reviewed:** 2026-08-23  
**Verdict:** DIRECTIONAL PASS WITH MIXED ADVANTAGE

## Verified

- Homoscedastic run uses exact frozen folds, official fixed-variance scale adjustment, positive constant scales, finite predictions, and unchanged runner/source.
- Mean NLL ranks natural `2.2180`, homoscedastic `2.2464`, ordinary `2.8319`; natural wins NLL on 4/5 folds.
- Mean RMSE ranks homoscedastic `2.0727`, natural `2.3884`, ordinary `5.0046`; homoscedastic wins RMSE on 5/5 folds.
- Mean homoscedastic-minus-natural NLL delta is only `+0.0284` with SE `0.0537`; this is weak directional separation, not decisive superiority.
- Ordinary coverage is obtained with much wider intervals (95% width 15.02 vs 6.90 natural and 6.16 homoscedastic).
- Summary files correctly report ordinary width 15.02; the Cursor completion message's 19.7 value was a transient typo and is not present in evidence files.

## Decision

Airfoil is a successful and scientifically informative new-data pilot. It strongly supports natural over ordinary multiparameter boosting on these folds, but only weakly favours natural over fixed variance by NLL, while fixed variance is better for RMSE. Preserve cap hits and under-coverage as limitations. Do not generalize beyond Airfoil or claim statistical significance from five folds.

Checkpoint `025-NG-P4` and the Airfoil existing-data pilot are closed.
