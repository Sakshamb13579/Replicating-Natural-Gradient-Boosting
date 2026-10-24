# Original-Data Selector Sensitivity Comparison

**Reportable:** `true`

| Dataset | Variant | Paper T2 NLL | Paper T3 RMSE | Historical NLL/RMSE | RMSE-selector NLL/RMSE | NLL-selector NLL/RMSE | Delta NLL/RMSE | NLL gap change | Stages RMSE/NLL | Cap hits RMSE/NLL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yacht | natural | 0.2000 | 0.5000 | 2.1060/0.5576 | 1.9556/0.6131 | 0.4115/0.6241 | -1.5440/+0.0110 | -1.5440 | 1536.5/432.8 | 9/0 |
| yacht | ordinary | 3.2900 |  | 11.5473/13.8165 | 11.5617/13.8165 | 3.5133/14.0568 | -8.0484/+0.2403 | -8.0484 | 2000.0/596.4 | 20/0 |
| yacht | homoscedastic | 2.0200 |  | 13.2821/0.5508 | 12.7399/0.5603 | 12.7399/0.5603 | +0.0000/+0.0000 | +0.0000 | 1732.8/1732.8 | 7/7 |
| kin8nm | natural | -0.4900 | 0.1600 | -0.4191/0.1692 | -0.4186/0.1692 | -0.4196/0.1726 | -0.0010/+0.0035 | -0.0010 | 1998.6/1274.2 | 2/0 |
| kin8nm | ordinary | -0.5200 |  | -0.5078/0.1566 | -0.5083/0.1566 | -0.5075/0.1568 | +0.0007/+0.0002 | +0.0007 | 1996.8/1950.0 | 2/2 |
| kin8nm | homoscedastic | -0.3700 |  | -0.3623/0.1608 | -0.3685/0.1607 | -0.3685/0.1607 | +0.0000/+0.0000 | +0.0000 | 1997.6/1997.6 | 1/1 |

## Comparability Notes

- Paper Table 2 NLL values use the paper's best-validation-log-likelihood description; Table 3 RMSE is a direct benchmark only for natural NGBoost.
- Historical local results used released-script validation-MSE selection, unpreserved RNG-coupled inner splits, and remain a separate baseline.
- Controlled results use newly frozen inner splits and paired selectors on one inner trajectory and one outer refit trajectory per fold.
- Kin8nm controlled and historical evidence uses five folds; the paper used twenty, so precision is not equivalent.
- Ordinary and homoscedastic RMSE values are local diagnostics without matching published Table 3 ablation benchmarks.
- Homoscedastic scaling follows the released script's post-refit validation-residual rule, not the paper's stated training-residual definition.
- The paper's 2nd-Order arm is omitted because it is not implemented in the released library.
- Blank historical RMSE dispersion fields are unavailable in the authoritative preserved reports and were not fabricated.
