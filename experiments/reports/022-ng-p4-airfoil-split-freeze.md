# 022-NG-P4 — Airfoil Self-Noise pilot split freeze

## Verdict

Airfoil Self-Noise is **frozen** as NGBoost **Pilot 1** (`PILOT_READY`): all **1,503** rows / **5** features preserved (no scaling/imputation), plus **five** paired outer/inner splits with deterministic regeneration. No models fitted.

## Processed data

| Item | Value |
|------|-------|
| CSV | `ngboost/data/processed/airfoil-self-noise-v1/airfoil-self-noise-v1.csv` |
| CSV SHA-256 | `705a2a7f0e55fcc9a87700038487b12ce768f4a2e1692433a2d752309028546e` |
| Raw SHA-256 | `74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3` (matches audit) |
| Columns | `row_id` + 5 features + `scaled_sound_pressure_db` |
| Missing / non-finite | 0 / 0 |

## Five paired splits

Construction matches official `regression_exp.py` outer style (`RandomState(1)` sequential 90/10 permutations), bounded to **5** folds. Inner 80/20 uses `random_state=10000+fold_index` (017F1: independent of model RNG). Same indices for **natural / ordinary / homoscedastic**.

| Fold | Outer train/test | Inner train/val | Outer-train hash (prefix) |
|------|------------------|-----------------|---------------------------|
| 0 | 1353 / 150 | 1082 / 271 | `a8a15d3c…` |
| 1 | 1353 / 150 | 1082 / 271 | `9a863ebd…` |
| 2 | 1353 / 150 | 1082 / 271 | `e8b25886…` |
| 3 | 1353 / 150 | 1082 / 271 | `62a0c670…` |
| 4 | 1353 / 150 | 1082 / 271 | `04359f3c…` |

Integrity (every fold): outer disjoint + full coverage; inner partitions outer-train only; no test leakage; regeneration bit-identical.

Split file: `…/split-indices.json`  
SHA-256: `128c5375b2659dff369e153e1b35f2163fbd1c87b72cc5a183def469ce014c14`

## Split design disclosure

- **Primary pilot:** row-level paper-style **interpolation** (i.i.d. rows).
- **106-condition grouped holdout:** documented in audit; **not** mixed into this freeze.
- Row splits may place different frequencies from the same aerodynamic condition in train and test.

## Pre-registered protocol (no fitting yet)

| Item | Value |
|------|-------|
| Cap / LR | 2,000 / 0.01 |
| Dist / score / base | Normal / MLE / default tree |
| Stage selector | validation **RMSE** (official-script alignment; paper primary is NLL — disclosed) |
| Variants (same folds) | natural, ordinary, homoscedastic |
| Primary metric | test **NLL** |
| Secondary | RMSE; 50/80/95 coverage; mean widths |
| Reporting | No method must win every fold; report paired differences + aggregate direction |

## Manifest

`ngboost/data/manifests/airfoil-self-noise-v1.json` — `status: frozen`, `pilot_status: PILOT_READY`, Pilot 1.

## Feasibility matrix

Airfoil row → `PILOT_READY`; NGBoost Pilot 1 → `airfoil-self-noise-v1`. Other datasets untouched.

## Stop

022-NG-P4 complete. No STATUS edit. No model fit. No other dataset changes.
