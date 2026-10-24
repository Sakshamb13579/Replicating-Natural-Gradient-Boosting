# External NGBoost Results v1

## Status

All 15 reportable runs are complete: five external datasets by three NGBoost
variants. Each run used validation-NLL stage selection, the frozen v1 data and split
manifest, 2,000 estimators, learning rate `0.01`, MLE, full batches, and one numerical
thread. All run checksum manifests pass.

The reviewed `a02` ledger contains exactly 15 aggregate rows and 15 paired
comparisons. It validates identical per-fold split hashes across variants and labels
its metrics as unweighted means of fold-level values.

An additional read-only data-curation check verified all five processed CSV hashes
and split-file hashes against their frozen manifests, every stored fold's index
hashes and train/test partitions, target/ID exclusion from model features, and
patient-disjoint Parkinsons inner/outer splits (42 unique subjects). The
Superconductivity subset has 5,000 unique in-range source-row IDs; its raw
`train.csv` was independently counted at 21,263 rows and 82 columns and its hash
matches the source manifest. Known repeated-condition/predictor-tuple overlap
remains a declared limitation of the chosen row-split estimands.

These are interim external-sensitivity results, not final or universal conclusions.

## Mean Fold Results

| Dataset | Variant | NLL | RMSE | Coverage 50/80/95 | Cap hits |
|---|---|---:|---:|---:|---:|
| Airfoil | Natural | 2.2470 | 2.5919 | 0.379/0.691/0.913 | 0 |
| Airfoil | Ordinary | 2.9465 | 5.4136 | 0.293/0.701/0.956 | 1 |
| Airfoil | Homoscedastic | **2.1567** | **2.0731** | 0.576/0.865/0.957 | 3 |
| Forest Fires | Natural | **1.7785** | 1.4097 | 0.270/0.867/0.935 | 0 |
| Forest Fires | Ordinary | 1.7826 | **1.4039** | 0.277/0.869/0.933 | 0 |
| Forest Fires | Homoscedastic | 1.7787 | 1.4149 | 0.310/0.861/0.935 | 0 |
| Parkinsons | Natural | **3.6301** | 8.5718 | 0.318/0.737/0.919 | 0 |
| Parkinsons | Ordinary | 3.6486 | **8.3576** | 0.398/0.699/0.924 | 0 |
| Parkinsons | Homoscedastic | 3.6662 | 9.0001 | 0.385/0.734/0.932 | 0 |
| Real Estate | Natural | **3.6441** | 8.0590 | 0.618/0.883/0.959 | 0 |
| Real Estate | Ordinary | 4.0367 | 13.5057 | 0.428/0.832/0.971 | 0 |
| Real Estate | Homoscedastic | 3.7928 | **7.9549** | 0.639/0.883/0.959 | 2 |
| Superconductivity subset | Natural | **3.5854** | 12.5533 | 0.416/0.762/0.936 | 0 |
| Superconductivity subset | Ordinary | 4.7301 | 33.7798 | 0.171/0.854/0.984 | 2 |
| Superconductivity subset | Homoscedastic | 3.8497 | **11.3164** | 0.705/0.870/0.941 | 0 |

Bold marks the lowest mean within a dataset and metric. These are mean-fold metrics,
not pooled-prediction metrics. NLL is the primary metric; RMSE evaluates only the
predictive mean and can rank variants differently.

## Paired NLL Review

Differences below are first variant minus second; negative values favor the first.
Win counts are first/tie/second with `1e-12` tie tolerance.

| Dataset | Natural - ordinary | Wins | Natural - homoscedastic | Wins | Ordinary - homoscedastic | Wins |
|---|---:|---:|---:|---:|---:|---:|
| Airfoil | -0.6995 | 5/0/0 | +0.0903 | 0/0/5 | +0.7898 | 0/0/5 |
| Forest Fires | -0.0041 | 13/0/7 | -0.0002 | 9/0/11 | +0.0039 | 8/0/12 |
| Parkinsons | -0.0185 | 4/0/1 | -0.0361 | 2/0/3 | -0.0176 | 3/0/2 |
| Real Estate | -0.3926 | 19/0/1 | -0.1487 | 15/0/5 | +0.2439 | 4/0/16 |
| Superconductivity subset | -1.1447 | 5/0/0 | -0.2643 | 5/0/0 | +0.8804 | 0/0/5 |

## Presentation-Safe Findings

- Airfoil strongly favors homoscedastic under this NLL-selector protocol: it beats
  natural and ordinary on NLL and RMSE in every fold. Three homoscedastic cap hits,
  one ordinary cap hit, and complete aerodynamic-condition overlap between train and
  test rows limit the claim to row-level interpolation under the frozen cap.
- The proposal's earlier Airfoil first run narrowly favored Natural over
  Homoscedastic in mean test NLL (`2.2180` versus `2.2464`); this revised protocol
  reverses that ordering (`2.2470` versus `2.1567`). The frozen splits and seeds
  match, but both stage selection and Homoscedastic scale handling changed.
  Causal attribution is not isolated. See
  `reports/airfoil-proposal-to-update-protocol-comparison-v1.md` for the exact
  code and artifact comparison.
- Superconductivity gives the clearest support for natural-gradient NLL: natural
  beats both alternatives on all five folds. Homoscedastic wins RMSE on all five
  natural-versus-homoscedastic comparisons, showing that distributional and point
  accuracy differ. This is a deterministic 5,000-row subset pilot, not full-data
  performance.
- Real Estate favors natural over ordinary strongly on NLL (19 of 20 folds). Natural
  also beats homoscedastic on mean NLL and 15 of 20 folds, but difficult folds have
  substantial influence and homoscedastic has slightly lower mean RMSE.
- Forest Fires is effectively tied. The largest mean NLL separation is only `0.0041`,
  and natural versus homoscedastic favors natural in mean but homoscedastic in 11 of
  20 folds. Do not rank the variants decisively.
- Parkinsons is inconclusive. Mean NLL differences are small, pairwise directions
  reverse across five subject-disjoint folds, and only 42 independent patients are
  represented.
- Interval calibration is not uniformly strong. In particular, nominal 50% coverage
  is frequently far from 0.50. Better NLL must not be described as uniformly better
  calibration.

## Review Limitations

- Eight stage-cap hits remain: Airfoil homoscedastic 3, Airfoil ordinary 1, Real
  Estate homoscedastic 2, and Superconductivity ordinary 2.
- Airfoil row splits share aerodynamic conditions across train and test. Real Estate,
  Forest Fires, and Superconductivity also contain repeated predictor tuples that can
  cross splits; this is interpolation evidence, not clean extrapolation evidence.
- Forest Fires uses `log1p(area)`, retains duplicates, permits spatial-cell overlap,
  and remains zero-inflated under an approximate Normal likelihood.
- Parkinsons metrics are unweighted fold means over subject-disjoint folds. The
  recording count must not be presented as the independent-patient sample size.
- The homoscedastic comparator uses validation labels both to estimate its retained
  constant sigma and to choose its stage. Natural and ordinary use validation labels
  only for stage selection. This is frozen and test-leakage-free, but asymmetric.
- Run artifacts preserve the exact runner hash, environment, data hashes, split
  hashes, command, predictions, and metrics. They identify NGBoost `0.5.11.dev0`
  from the repair worktree but do not record its Git commit directly. The current
  worktree is at `672034041fc05d9d4b7a80c1e6163ea102579752`; its only reported
  modification is the known original-example import repair.
- Artifacts preserve the selected stage and its validation NLL, not the complete
  2,000-stage validation trajectory. Independently checking the argmin requires a
  rerun.

## Evidence

- Frozen protocol: `reports/external-ngboost-protocol-v1.md`
- Full runs: `artifacts/external-ngboost-v1/<dataset>/<variant>/`
- Reviewed aggregate: `artifacts/external-ngboost-v1/ledger-s1-a02/`
- Main ledger: `external-ngboost-ledger.md` and `.csv`
- Paired ledger: `external-ngboost-ledger.paired.md` and `.paired.csv`
- Ledger checksums: `external-ngboost-ledger.checksums.txt`

The earlier `ledger-s1-a01` is superseded by `a02` because it did not include paired
fold comparisons or explicit mean-fold labeling. Its source run values were not
incorrect.
