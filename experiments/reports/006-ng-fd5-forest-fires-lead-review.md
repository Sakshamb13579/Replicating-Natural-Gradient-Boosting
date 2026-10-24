# Lead Review: 006-NG-FD5 Forest Fires

**Reviewed:** 2026-08-23  
**Fix:** `006F1-NG-FD5`  
**Verdict:** MODERATE  
**Status:** `AUDIT_DRAFT` - stress-test candidate

## Verified

- Stored hashes and direct checks match: 517 rows, 13 columns, 247 zero targets, raw maximum `1090.84`, raw skew `12.8096`, log1p skew `1.2143`, four duplicate rows, 36 spatial cells, and maximum 52 rows per cell.
- Donor documentation supports `log(area+1)` modeling.
- The fix correctly distinguishes log-scale NLL/coverage, raw median `expm1(mu_z)`, lognormal-style mean `exp(mu_z+sigma_z^2/2)-1`, and asymmetric raw intervals.
- Raw-area Normal likelihood is described as computable but scientifically misspecified, not impossible.

## Decision

Accept `MODERATE` as a stress-test dataset, not primary calibration evidence. Any pilot must freeze the log1p target, categorical encoding, duplicate policy, split, and raw-scale summary before execution. No split or pilot is authorized now.

Checkpoint `006-NG-FD5` is closed after one verified fix cycle. All five NGBoost existing-data desk audits are now complete at draft status.
