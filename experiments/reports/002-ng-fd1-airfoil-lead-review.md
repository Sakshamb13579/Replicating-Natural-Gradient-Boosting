# Lead Review: 002-NG-FD1 Airfoil Data Feasibility

**Reviewed:** 2026-08-23  
**Fix reviewed:** `002F1-NG-FD1`  
**Verdict:** STRONG  
**Status:** `AUDIT_DRAFT` - source and feasibility verified; split not frozen

## Independently Verified

- UCI dataset ID 291, DOI `10.24432/C5VW2C`, 1,503 instances, five features, regression target, no reported missing values, and CC BY 4.0 licence.
- Downloaded zip SHA-256: `5c7767ba53ad827d3f48ba1eb9434117f4892df8f10bc4c99e118a9e8a7ae07c`.
- Extracted data SHA-256: `74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3`.
- Metadata evidence SHA-256: `913da48b6ea2680afed32893844ef36dc71c08772ac665b3de695e717c0e71a1`.
- Direct raw-data recomputation: 1,503 rows, six columns including target, no ragged rows, no non-finite values, and no duplicate rows.
- Target range `103.38` to `140.987`, mean `124.83594`, sample SD `6.89866`, and population-moment skewness `-0.41853`.
- Exactly 106 repeated experimental-condition groups when grouping by attack angle, chord length, velocity, and displacement thickness. Group sizes range from 8 to 19 and sum to all 1,503 rows.
- Airfoil is absent from the ten datasets named in NGBoost Tables 1-3 and absent from the pinned official regression loader.

The stored UCI metadata file is a normalized evidence bundle assembled from the listed authoritative UCI API endpoints and dataset page, not a byte-for-byte raw HTTP capture. Its substantive fields were checked against the live UCI responses during lead review.

## Scientific Assessment

Airfoil is a strong new-existing-data candidate because it is small, clean, licensed, physically interpretable, and suitable for uncertainty-aware regression. Its repeated frequency sweeps create a useful protocol distinction:

- Row-level 90/10 holdouts preserve alignment with the paper's evaluation style but may interpolate within wind-tunnel conditions represented in both train and test.
- Condition-group holdout tests generalization to unseen aerodynamic settings and reduces that dependence, but changes the estimand and has only 106 effective groups.

Neither split is frozen at this checkpoint. A pilot handoff must predeclare which scientific question is primary and may treat the other split as an optional robustness analysis.

## Claim Boundary

Supported: authoritative availability, licence, data integrity, paper absence, low resource burden, and scientific suitability.

Not supported: that NGBoost has never been used on this dataset, that natural-gradient NGBoost will win, or that either draft split is already selected.

## Decision

Accept the `STRONG` feasibility verdict and close `002-NG-FD1` after one verified fix cycle. Retain `AUDIT_DRAFT` until the split question is fixed as part of pilot selection. No further Airfoil work is authorized now.
