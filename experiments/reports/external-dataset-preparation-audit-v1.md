# External Dataset Preparation Audit v1

> **Historical preparation checkpoint (before full training).** This document records
> the frozen dataset preparation and smoke-test status at that time. Statements below
> saying that selector work or full external runs are pending are no longer current.
> For completed results use `reports/original-selector-sensitivity-results-v1.md`,
> `reports/external-ngboost-results-v1.md`, and the checked external ledger at
> `artifacts/external-ngboost-v1/ledger-s1-a02/`. The frozen preparation decisions
> and data limitations below remain part of the evidence.

## Status

Data preparation and protocol freezing are complete for all five external datasets.
All datasets passed natural, ordinary, and homoscedastic one-fold/five-stage smoke
tests. Those smoke results are explicitly non-reportable. No full 2,000-stage
external-data comparison has started.

The common reportable protocol is documented in
`reports/external-ngboost-protocol-v1.md`: paired variants, validation-NLL stage
selection, 2,000-stage cap, learning rate 0.01, Normal MLE, default tree learner,
test NLL primary, and test RMSE plus 50/80/95 coverage and interval width secondary.

## Prepared datasets

| Dataset | Frozen rows x features | Changes from source | Frozen evaluation split | Remaining limitation |
|---|---:|---|---|---|
| Airfoil Self-Noise | 1,503 x 5 | Added stable `row_id`; preserved all five numeric inputs and dB target; no scaling or imputation | Existing five repeated 90/10 row holdouts with deterministic inner validation | Different frequencies from one aerodynamic condition can cross train/test; NLL selection is a sensitivity run distinct from the earlier RMSE-selected pilot |
| Real Estate Valuation | 414 x 6 | Removed unique transaction ID `No`; retained the six documented numeric predictors and house-price target; no scaling or imputation | Twenty repeated 90/10 row holdouts with deterministic inner validation | Small test sets; time periods and repeated coordinates can cross train/test |
| Parkinsons Telemonitoring | 5,875 x 19 | Target is `motor_UPDRS`; excluded `subject#` and alternate outcome `total_UPDRS` from model features; retained `subject#` only for grouping/evaluation | Five patient-disjoint outer folds and patient-disjoint inner validation | Only 42 independent patients; recording-level rows are not independent patient evidence |
| Forest Fires | 517 x 29 | Replaced month/day strings with fixed one-hot columns; transformed target to donor-aligned `log1p(area)`; retained all rows including four exact duplicates; no scaling or imputation | Twenty repeated 90/10 row holdouts with deterministic inner validation | Spatial cells can cross splits; zero inflation remains after transformation; metrics are on log1p scale |
| Superconductivity | 5,000 x 81 | Selected a deterministic 5,000-row subset from 21,263 rows using `RandomState(1)` before splitting; preserved `source_row_id`; used only `train.csv`; retained selected duplicates/collisions; no scaling or imputation | Five repeated 90/10 row holdouts with deterministic inner validation | Subset-pilot evidence only; repeated aggregated feature tuples can map to different targets |

## Why Superconductivity uses a subset

A non-reportable 25-stage benchmark on the complete 21,263-row table projected
approximately 9.52 hours for five folds across all three variants. The submitted
proposal explicitly permits fewer splits or a fixed row subset for this dataset.
The deterministic 5,000-row contingency was therefore selected before model-result
inspection. No benchmark performance metrics influenced the scope decision.

Benchmark:
`artifacts/superconductivity/benchmark-full-data-est25-a01.json`

## Data and manifests to inspect

### Airfoil Self-Noise

- Processed data: `data/processed/airfoil-self-noise-v1/airfoil-self-noise-v1.csv`
- Splits: `data/processed/airfoil-self-noise-v1/split-indices.json`
- Manifest: `data/manifests/airfoil-self-noise-v1.json`

### Real Estate Valuation

- Processed data: `data/processed/real-estate-valuation-v1/real-estate-valuation-v1.csv`
- Splits: `data/processed/real-estate-valuation-v1/split-indices.json`
- Manifest: `data/manifests/real-estate-valuation-v1.json`

### Parkinsons Telemonitoring

- Processed data: `data/processed/parkinsons-telemonitoring-v1/parkinsons-telemonitoring-v1.csv`
- Splits: `data/processed/parkinsons-telemonitoring-v1/split-indices.json`
- Manifest: `data/manifests/parkinsons-telemonitoring-v1.json`

### Forest Fires

- Processed data: `data/processed/forest-fires-v1/forest-fires-v1.csv`
- Splits: `data/processed/forest-fires-v1/split-indices.json`
- Manifest: `data/manifests/forest-fires-v1.json`

### Superconductivity

- Processed data: `data/processed/superconductivity-v1/superconductivity-v1.csv`
- Splits: `data/processed/superconductivity-v1/split-indices.json`
- Manifest: `data/manifests/superconductivity-v1.json`

All paths are relative to
`people/ash/work/paper-selection/ngboost/`.

## Integrity evidence

The preparation script verifies each raw source against the SHA-256 recorded in its
audit manifest. Processed CSV and split hashes are recorded in each frozen manifest.
Every split passed disjointness, coverage, index-range, and inner/outer partition
checks. Parkinsons additionally passed patient-group disjointness checks. The same
indices are reused by all three variants.

Preparation script: `src/prepare_external_datasets_v1.py`

Training runner: `src/run_external_ngboost.py`

Smoke artifacts and preliminary smoke ledger: `artifacts/smoke/`

## Work still pending

1. Run the controlled original-data RMSE-versus-NLL selector sensitivity for Yacht
   and Kin8nm. This is separate from external-data preparation and remains unfinished.
2. Run the full 2,000-stage, three-variant comparisons on all five external datasets.
3. Review per-fold outputs, aggregate paired comparisons, limitations, checksums,
   and runtime before promoting any number into the presentation ledger.

The historical Yacht/Kin8nm released-script runs did not preserve reusable frozen
inner indices. The selector diagnostic must therefore freeze a controlled split set
and compare RMSE and NLL selectors on the same stage trajectories; historical
released-script values remain a separate reproduction baseline.
