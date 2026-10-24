# External NGBoost protocol v1

## Scope and decision rule

This protocol evaluates external regression data that are absent from the selected
NGBoost paper tables. Every comparison is paired: natural-gradient, ordinary-gradient,
and homoscedastic variants use the same frozen outer/inner indices and the same
per-fold model seeds. The boosting stage is selected **only by validation negative
log likelihood (NLL)**. Test labels are unavailable to preprocessing and stage
selection.

The reportable configuration is `n_estimators=2000`, `learning_rate=0.01`, MLE,
the NGBoost default tree learner, full-batch fitting, and no feature scaling or
imputation. Natural and ordinary variants use `Normal`; they differ only in
`natural_gradient=True/False`. The homoscedastic variant uses `NormalFixedVar` for
the location trajectory. At every stage, its constant sigma is the RMS validation
residual of the stage model; validation Normal NLL selects the stage. That selected
sigma is retained after refitting location on outer train (inner train plus
validation). Sigma is **not** re-estimated after validation rows enter training.

`--max-folds` and `--n-estimators` are smoke-test controls only. Any use that changes
the frozen fold count or 2,000-estimator cap is marked `non_reportable` in artifacts
and cannot enter the primary comparison.

## Metrics and artifacts

Per fold and in aggregate, record test NLL, RMSE, empirical 50/80/95% interval
coverage, mean 50/80/95% interval width, selected stage, cap hit, sample counts,
and stage/refit/wall runtime. Preserve predictions, config, data/run manifests,
invocation, environment, timing, checksums, and log-friendly stdout in each new run
directory. The primary selector and metric are validation NLL and test NLL.

## Frozen split policies

| Dataset | Outer policy | Inner policy | Limitations |
|---|---|---|---|
| Airfoil Self-Noise | Existing v1 five repeated 90/10 row holdouts from sequential `RandomState(1).choice` permutations. Existing indices are reused unchanged, but NLL-sensitivity runs require new run IDs/directories. | Existing v1 80/20 row split of each outer train, seed `10000+fold`. | Row interpolation can place frequencies from the same aerodynamic condition on both sides. Existing v1 was originally preregistered with RMSE selection; these are explicitly separate NLL-sensitivity results. |
| Real Estate Valuation | 20 repeated 90/10 row holdouts, sequential `RandomState(1).choice`, matching paper style. | 80/20 row split of outer train, seed `10000+fold`. | About 41 test rows per repetition; coverage is noisy. Random rows mix months and repeated coordinates. `No` is excluded. |
| Parkinsons Telemonitoring | Shuffled five-fold partition of 42 subjects, seed 1; each subject is test exactly once. | Subject-disjoint 80/20 split of outer-training subjects, seed `10000+fold`; all recordings follow their subject. | Only 42 independent subjects. Recording-level sample size overstates independent evidence; coverage is exploratory. `subject#` and `total_UPDRS` are excluded from predictors. |
| Forest Fires | 20 repeated 90/10 row holdouts, sequential `RandomState(1).choice`. | 80/20 row split of outer train, seed `10000+fold`. | Retains four exact duplicate rows and permits spatial-cell overlap. Normal on `log1p(area)` remains approximate because 47.8% of raw targets are zero. Metrics are on log1p scale. |

## Superconductivity decision gate

The timing gate selected the proposal-authorized deterministic subset contingency.
A non-reportable 25-stage benchmark on all 21,263 `train.csv` rows projected about
**9.52 hours** for five folds across the three variants. No performance metrics were
computed or used in that decision. The benchmark is preserved at
`artifacts/superconductivity/benchmark-full-data-est25-a01.json`.

The frozen pilot therefore selects 5,000 source rows without replacement using
`RandomState(1)` before split generation, sorts the selected source indices for stable
storage, and preserves each `source_row_id`. It uses five paired repeated 90/10 row
holdouts with deterministic inner validation. `train.csv` is the sole feature source;
`unique_m.csv` and its material strings are not used. Rows selected by the fixed random
subset are retained without outcome-dependent duplicate filtering. These results are
explicitly subset-pilot evidence and must not be described as full-dataset performance.

## Interpretation boundaries

These datasets establish external sensitivity, not universal superiority. Repeated
row holdouts are dependent replicates; Parkinsons rows are clustered observations;
small test sets make interval estimates variable; and Normal likelihood assumptions
can be misspecified. Report paired fold differences, aggregate dispersion, cap hits,
and all dataset-specific limitations rather than requiring one variant to win every
fold.
