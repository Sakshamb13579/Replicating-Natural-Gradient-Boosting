# Original-Data RMSE versus NLL Selector Sensitivity Protocol v1

## Question

Does selecting the NGBoost stage by validation NLL, as described by Duan et al.,
materially change the Yacht and Kin8nm results relative to selecting it by validation
squared error/RMSE, as implemented by the released `regression_exp.py`?

This is a controlled sensitivity analysis. It does not replace the historical
released-script reproduction and must not be described as the paper's unpublished
execution.

## Evidence being kept separate

| Evidence | Stage selector | Published/reported metrics |
|---|---|---|
| Duan et al. Table 2 | Best validation log-likelihood, equivalent to minimum validation NLL | Test NLL for NGBoost, 2nd-Order, Multiparameter, and Homoscedastic |
| Duan et al. Table 3 | Full NGBoost remains NLL-optimized; most point-prediction comparators are RMSE-optimized | Test RMSE; only full NGBoost overlaps the local three-arm ablation |
| Historical local released-script reproduction | Minimum validation squared error, equivalent to minimum validation RMSE | Test NLL and test RMSE; historical inner indices were not preserved |
| New controlled sensitivity | Both selectors applied to one shared fitted trajectory | Test NLL and test RMSE for each selected stage |

The historical results remain a separate baseline because their inner splits were
generated without an explicit `random_state` and were not preserved. The controlled
experiment freezes new inner indices and must not be presented as a bit-for-bit rerun
of those historical results.

## Datasets and scope

| Dataset | Rows x features | Controlled folds | Reason |
|---|---:|---:|---|
| Yacht Hydrodynamics | 308 x 6 | 20 | Matches the paper and historical local repetition count |
| Kin8nm | 8,192 x 8 | 5 | Matches the submitted proposal and historical bounded local scope; the paper used 20, so precision is not equivalent |

Use the exact preserved original data files and verify their existing SHA-256 values
before split generation. The target is the final column and all preceding columns are
features. No scaling, imputation, or feature transformation is added.

## Variants

Run the same three locally scoped Table 2 variants:

1. `natural`: Normal multiparameter NGBoost with `natural_gradient=True`.
2. `ordinary`: Normal multiparameter boosting with `natural_gradient=False`.
3. `homoscedastic`: `NormalFixedVar` location boosting with the released-script
   constant-scale adjustment.

The paper's 2nd-Order arm remains omitted because it is not implemented in the
released library. This omission is unchanged and must remain disclosed.

## Frozen split construction

For each dataset:

1. Generate outer repeated 90/10 row holdouts using the released script's sequential
   `RandomState(1).choice` construction.
2. Within each outer-training set, freeze one 80/20 train/validation partition using
   `random_state=10000+fold_index`.
3. Store absolute row indices and SHA-256 hashes for outer train, outer test, inner
   train, and inner validation.
4. Reuse those exact indices for all three variants and both selectors.
5. Verify disjointness, complete outer coverage, inner partitioning of outer train,
   and absence of test rows from inner splits.

This deliberately removes the released runner's RNG-coupled inner-split ambiguity.

## Fixed model configuration

- NGBoost commit: `672034041fc05d9d4b7a80c1e6163ea102579752`.
- Score: MLE.
- Distribution: `Normal`, or `NormalFixedVar` for homoscedastic.
- Base learner: NGBoost `default_tree_learner`.
- Learning rate: `0.01`.
- Estimator cap: `2,000`.
- Full batch: `minibatch_frac=1.0`.
- One deterministic stage-model seed and one deterministic refit seed per fold.
- The same per-fold seeds are reused across selectors; model variant is the only
  model-family change.

Any smoke run using fewer folds or estimators is non-reportable.

## Per-fold execution

For each dataset, fold, and variant:

1. Fit **one** 2,000-stage model on the frozen inner-training rows.
2. From that single stage trajectory, calculate at every stage:
   - validation squared error and its square root, RMSE;
   - validation NLL.
3. Record two selected stages:
   - `best_iter_rmse = argmin(validation squared error) + 1`;
   - `best_iter_nll = argmin(validation NLL) + 1`.
4. Fit **one** 2,000-stage refit trajectory on all outer-training rows.
5. Evaluate the untouched outer-test rows at both selected stages from that same
   refit trajectory.
6. Record test NLL and test RMSE for both selectors, plus selected stage and cap hit.

Using one stage trajectory and one refit trajectory ensures that the selector is the
only difference within each fold/variant comparison.

## Homoscedastic handling

To isolate the selector while retaining released-script comparator behavior:

1. Validation RMSE selects a location stage from the inner-training trajectory.
2. Validation NLL at each stage is computed after assigning that stage a constant
   residual-RMS scale from the frozen validation rows.
3. After the outer-training refit, each selected stage receives the released-script
   scale adjustment using residual RMS on the same frozen validation rows.
4. Apply that constant scale to the outer-test distribution and record NLL/RMSE.

This post-refit validation-scale calculation resembles the released script but is not
the paper's stated training-residual homoscedastic definition. That known comparator
difference remains a disclosure. The newer external-data pipeline uses a separately
declared leakage-free retained-scale rule and must not be merged with this original-
code sensitivity protocol.

## Metrics and comparisons

Record per fold and aggregate:

- selected stage for each selector;
- stage-cap hits;
- test NLL for each selector;
- test RMSE for each selector;
- paired `NLL-selected minus RMSE-selected` differences;
- mean, median, population SD, sample SD, and SE;
- fold-wise win/tie counts;
- runtime and finite-output checks.

### Direct paper comparisons

Table 2 test-NLL reference values:

| Dataset | Natural/NGBoost | Ordinary/Multiparameter | Homoscedastic |
|---|---:|---:|---:|
| Yacht | 0.20 +/- 0.26 | 3.29 +/- 0.20 | 2.02 +/- 0.21 |
| Kin8nm | -0.49 +/- 0.02 | -0.52 +/- 0.03 | -0.37 +/- 0.05 |

Table 3 full-NGBoost test-RMSE reference values:

| Dataset | Full NGBoost RMSE |
|---|---:|
| Yacht | 0.50 +/- 0.20 |
| Kin8nm | 0.16 +/- 0.00 |

Test NLL for all three local variants may be compared with Table 2. Test RMSE may be
compared with Table 3 only for the full natural-gradient NGBoost model, with the
controlled-protocol difference stated. Ordinary and homoscedastic RMSE have no
matching published Table 2 ablation benchmark and are local diagnostics only.

## Interpretation rules

1. Report whether NLL selection changes the selected stages and paired test metrics.
2. Report whether it reduces or increases the absolute test-NLL gap from Table 2.
3. Do not claim the selector discrepancy caused the Yacht mismatch unless the
   controlled result supports that conclusion.
4. If Yacht remains far from `0.20`, state that the selector does not explain the
   mismatch fully.
5. If Kin8nm remains close, state that result without treating five folds as equal in
   precision to the paper's twenty repetitions.
6. Keep historical released-script, controlled sensitivity, and published-paper
   values in separate table columns.

## Required artifacts

- Frozen processed-data and split manifests for Yacht and Kin8nm.
- Exact command, configuration, environment, logs, timing, and checksums.
- Per-fold selected stages and both-selector metrics.
- Per-row predictions for both selected stages where practical.
- A machine-readable comparison JSON/CSV.
- A presentation-ready table that visibly separates paper, historical released
  script, controlled RMSE selector, and controlled NLL selector.

## Gate before external full runs

The selector diagnostic is complete only when all three variants finish on the
frozen Yacht and Kin8nm folds, artifact checks pass, and the comparison table has
been reviewed. Only then should its conclusion be moved into the presentation. The
five external datasets are already prepared and smoke-tested, but their full runs
remain separate work.
