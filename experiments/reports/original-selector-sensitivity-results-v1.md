# Original-Data Selector Sensitivity Results v1

## Status

The controlled Yacht and Kin8nm selector sensitivity is complete and reportable.
All six dataset/variant runs used the frozen v1 manifests, completed at 2,000
stages, passed their artifact checksum manifests, and were combined into a checked
six-row comparison. A separate read-only integrity check revalidated the Yacht
and Kin8nm processed CSV and split-file hashes, all stored fold-index hashes,
and outer/inner partitioning against their frozen manifests.

This is a controlled sensitivity analysis, not the paper's unpublished execution
and not a bit-for-bit rerun of the historical released-script results.

## Mean Results

| Dataset | Variant | Paper NLL | Historical NLL | Controlled RMSE-selector NLL | Controlled NLL-selector NLL | NLL-selected minus RMSE-selected | Controlled RMSE-selector RMSE | Controlled NLL-selector RMSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Yacht | Natural | 0.20 | 2.1060 | 1.9556 | 0.4115 | -1.5440 | 0.6131 | 0.6241 |
| Yacht | Ordinary | 3.29 | 11.5473 | 11.5617 | 3.5133 | -8.0484 | 13.8165 | 14.0568 |
| Yacht | Homoscedastic | 2.02 | 13.2821 | 12.7399 | 12.7399 | 0.0000 | 0.5603 | 0.5603 |
| Kin8nm | Natural | -0.49 | -0.4191 | -0.4186 | -0.4196 | -0.0010 | 0.1692 | 0.1726 |
| Kin8nm | Ordinary | -0.52 | -0.5078 | -0.5083 | -0.5075 | +0.0007 | 0.1566 | 0.1568 |
| Kin8nm | Homoscedastic | -0.37 | -0.3623 | -0.3685 | -0.3685 | 0.0000 | 0.1607 | 0.1607 |

Paper values are Duan et al. Table 2 test NLL. Historical values are the preserved
released-script-style local results. Controlled columns apply both selectors to the
same fitted inner and outer trajectories. For Yacht Natural, paper `0.20`,
historical RMSE-selected `2.1060`, and controlled paired
`1.9556` (RMSE-selected) versus `0.4115` (NLL-selected) are distinct evidence
lines. **Only `1.9556 → 0.4115` is a within-experiment selector comparison.**

## Findings

- Yacht natural-gradient selection changed on all 20 folds. NLL selection reduced
  mean test NLL from `1.9556` to `0.4115` while increasing mean test RMSE from
  `0.6131` to `0.6241`. NLL selection won test NLL on 17 of 20 folds; RMSE
  selection won test RMSE on 15 of 20 folds.
- Yacht ordinary-gradient selection also changed on all 20 folds. NLL selection
  reduced mean test NLL from `11.5617` to `3.5133`, close to the paper's `3.29`
  point estimate, while increasing mean test RMSE from `13.8165` to `14.0568`.
- Yacht homoscedastic selectors were identical on every fold. Under this controlled
  comparator, the validation residual RMS is re-estimated at each stage, making its
  validation NLL ordering equivalent to its validation squared-error ordering. The
  selector therefore cannot explain its large NLL mismatch.
- Kin8nm selector effects were small. Natural-gradient mean test NLL changed by
  `-0.0010`, ordinary by `+0.0007`, and homoscedastic not at all. These controlled
  results use five folds, not the paper's twenty repetitions.
- The experiment supports stage selection as a major contributor to the Yacht
  natural and ordinary NLL discrepancies. It does not establish stage selection as
  the sole cause: Yacht natural remains above the paper point estimate (`0.4115`
  versus `0.20`), Yacht homoscedastic remains far from the paper, and the controlled
  protocol differs from the paper in other documented ways.
- The opposing NLL and RMSE movements are expected rather than contradictory:
  each selector optimizes a different validation objective.

## Stage Selection

| Dataset | Variant | Mean RMSE-selected stage | Mean NLL-selected stage | RMSE/NLL cap hits |
|---|---|---:|---:|---:|
| Yacht | Natural | 1536.5 | 432.8 | 9/0 |
| Yacht | Ordinary | 2000.0 | 596.4 | 20/0 |
| Yacht | Homoscedastic | 1732.8 | 1732.8 | 7/7 |
| Kin8nm | Natural | 1998.6 | 1274.2 | 2/0 |
| Kin8nm | Ordinary | 1996.8 | 1950.0 | 2/2 |
| Kin8nm | Homoscedastic | 1997.6 | 1997.6 | 1/1 |

## Evidence

- Protocol: `reports/original-selector-sensitivity-protocol-v1.md`
- Frozen manifests: `data/manifests/yacht-original-selector-sensitivity-v1.json`
  and `data/manifests/kin8nm-original-selector-sensitivity-v1.json`
- Full runs: `artifacts/original-selector-sensitivity-v1/{yacht,kin8nm}/`
- Checked aggregate:
  `artifacts/original-selector-sensitivity-v1/comparison-s1-a01/`
- Presentation-ready table:
  `artifacts/original-selector-sensitivity-v1/comparison-s1-a01/comparison.md`

The aggregate and each of the six run directories contain a passing
`checksums.txt`. Smoke artifacts remain separate and non-reportable.
