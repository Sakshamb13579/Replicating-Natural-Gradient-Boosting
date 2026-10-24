# Lead Review: 005-NG-FD4 Parkinsons Telemonitoring

**Reviewed:** 2026-08-23  
**Fix:** `005F1-NG-FD4`  
**Verdict:** MODERATE  
**Status:** `AUDIT_DRAFT`

## Verified

- Stored hashes and direct data checks match: 5,875 rows, 22 columns, 42 subjects, 101-168 recordings per subject, no non-finite values or duplicate rows, and motor/total UPDRS correlation `0.9472`.
- `motor_UPDRS` is the primary target; `subject#` and `total_UPDRS` are forbidden predictors.
- The fix correctly treats patients, not recordings, as independent evaluation units.
- Draft five-fold subject-group CV yields about 8-9 held-out subjects per fold and preserves paired model comparisons.
- Recording-level and subject-macro metrics are distinguished; coverage remains exploratory and cannot establish clinical calibration.

## Decision

Accept `MODERATE`. The dataset is proposal-worthy and scientifically meaningful, but its 42-patient cohort, clinical framing, and grouped protocol make it a higher-governance extension than Airfoil. No split is frozen and no training is authorized yet.

Checkpoint `005-NG-FD4` is closed after one verified fix cycle.
