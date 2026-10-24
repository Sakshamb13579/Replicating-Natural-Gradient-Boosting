# 025-NG-P4 — Airfoil homoscedastic NGBoost + final three-way

## Verdict

Homoscedastic `NormalFixedVar` NGBoost (natural gradient + official val residual scale adjustment) completed all **five** frozen Airfoil folds. Mean test **NLL 2.2464**, mean test **RMSE 2.0727**.

**Final Airfoil three-way** (same folds; bounded to this pilot only):

| Rank by mean NLL | Variant |
|------------------|---------|
| 1 (best) | **natural** |
| 2 | homoscedastic |
| 3 | ordinary |

| Rank by mean RMSE | Variant |
|-------------------|---------|
| 1 (best) | **homoscedastic** |
| 2 | natural |
| 3 | ordinary |

This ranking is for **Airfoil v1 five-fold pilot only**. It is not a paper-wide table claim. Coverage near nominal can still reflect **wide** intervals — see width columns below.

## Homoscedastic protocol

| Item | Value |
|------|-------|
| Handoff | 025-NG-P4 |
| Runner | `--variant homoscedastic` (**not edited**) |
| Dist | `NormalFixedVar` + official post-fit scale from val residuals |
| Else | Same as 023/024: cap 2000, LR 0.01, MLE, default tree, frozen folds/seeds, val RMSE |
| Env | Unchanged |

## Standalone per-fold (homoscedastic)

| Fold | Best iter | Test NLL | Test RMSE | Cov50 | Cov80 | Cov95 | Width95 | Mean scale | Cap | Time |
|------|-----------|----------|-----------|-------|-------|-------|---------|------------|-----|------|
| 0 | 2000 | 2.070804 | 1.886991 | 0.447 | 0.773 | 0.933 | 6.531 | 1.6660 | True | 35.0 |
| 1 | 1955 | 2.264619 | 1.982547 | 0.460 | 0.687 | 0.887 | 5.453 | 1.3910 | False | 35.3 |
| 2 | 2000 | 2.150985 | 1.924286 | 0.447 | 0.700 | 0.887 | 5.847 | 1.4915 | True | 35.4 |
| 3 | 2000 | 2.213914 | 2.040525 | 0.500 | 0.760 | 0.927 | 6.161 | 1.5716 | True | 35.8 |
| 4 | 1981 | 2.531789 | 2.529040 | 0.440 | 0.707 | 0.887 | 6.804 | 1.7358 | False | 34.6 |

Aggregates: NLL mean 2.246422 (SE 0.078349); RMSE mean 2.072678 (SE 0.117031); cap hits 3/5; wall 196.7s. Scales positive/finite on all folds.

## Final three-way — per-fold NLL (lower better)

| Fold | Natural | Ordinary | Homoscedastic | Best |
|------|---------|----------|---------------|------|
| 0 | 2.223051 | 2.781104 | 2.070804 | homoscedastic |
| 1 | 2.174082 | 2.570530 | 2.264619 | natural |
| 2 | 2.144915 | 2.643631 | 2.150985 | natural |
| 3 | 2.188369 | 3.351480 | 2.213914 | natural |
| 4 | 2.359773 | 2.812578 | 2.531789 | natural |

NLL fold-win counts: natural 4 · ordinary 0 · homoscedastic 1

## Final three-way — per-fold RMSE (lower better)

| Fold | Natural | Ordinary | Homoscedastic | Best |
|------|---------|----------|---------------|------|
| 0 | 2.299805 | 5.189226 | 1.886991 | homoscedastic |
| 1 | 2.309458 | 4.506893 | 1.982547 | homoscedastic |
| 2 | 2.236127 | 5.025891 | 1.924286 | homoscedastic |
| 3 | 2.202265 | 5.217049 | 2.040525 | homoscedastic |
| 4 | 2.894497 | 5.084019 | 2.529040 | homoscedastic |

RMSE fold-win counts: natural 0 · ordinary 0 · homoscedastic 5

## Mean differences vs natural (Δ = other − natural; negative ⇒ other better)

| Contrast | Mean ΔNLL (SE) | Mean ΔRMSE (SE) |
|----------|----------------|-----------------|
| ordinary − natural | +0.6138 (0.1399) | +2.6162 (0.1762) |
| homoscedastic − natural | +0.0284 (0.0537) | -0.3158 (0.0423) |

## Coverage / width context (95%; coverage / mean width)

| Fold | Natural | Ordinary | Homoscedastic |
|------|---------|----------|---------------|
| 0 | 0.873/7.040 | 0.933/15.459 | 0.933/6.531 |
| 1 | 0.900/6.642 | 0.953/13.201 | 0.887/5.453 |
| 2 | 0.927/6.890 | 0.967/15.548 | 0.887/5.847 |
| 3 | 0.927/6.518 | 0.967/15.784 | 0.927/6.161 |
| 4 | 0.880/7.406 | 0.927/15.115 | 0.887/6.804 |

Fold-mean Cov95 / Width95:

| Variant | Cov95 | Width95 | Cap hits | Wall (s) |
|---------|-------|---------|----------|----------|
| natural | 0.9013 | 6.8992 | 4/5 | 226.9 |
| ordinary | 0.9493 | 15.0212 | 5/5 | 232.9 |
| homoscedastic | 0.9040 | 6.1590 | 3/5 | 196.7 |

Nominal coverage can be achieved with wider predictive intervals; compare mean_width_95 alongside coverage_95. Do not treat coverage alone as calibration success.

## Aggregate means (primary summary)

| Variant | Mean NLL | Mean RMSE |
|---------|----------|-----------|
| natural (023) | 2.218038 | 2.388430 |
| ordinary (024) | 2.831865 | 5.004616 |
| homoscedastic (025) | 2.246422 | 2.072678 |

## Claim boundary

Final **Airfoil** natural / ordinary / homoscedastic comparison on five frozen folds only. Not extrapolated to other NGBoost paper datasets. If ranking differs from paper narrative, that is reported as observed pilot evidence, not hidden.

## Verification

- Split hashes identical across 023/024/025.
- Homoscedastic official scale adjustment applied; scales > 0; predictions finite.
- Runner SHA unchanged; env/packages untouched.

## Artifacts

`artifacts/airfoil-self-noise/ngboost-homoscedastic/025-ng-p4-airfoil-homoscedastic-s1-a01/`

Includes `metrics.json`, `three-way-comparison.json`, `predictions.jsonl`/`.csv`, and full protocol files.

## Stop

025-NG-P4 complete. No STATUS/runner/env edit. No further task.
