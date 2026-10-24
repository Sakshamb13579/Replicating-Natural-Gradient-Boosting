# 023-NG-P4 — Airfoil natural-gradient NGBoost (five frozen folds)

## Verdict

Official-library **natural-gradient** NGBoost ran stably on all **five** frozen Airfoil v1 folds. Mean test **NLL 2.2180** (SE 0.0376); mean test **RMSE 2.3884**. Predictions/intervals finite with positive scales. This establishes natural-model feasibility only — **no superiority claim** vs ordinary/homoscedastic until those paired runs exist.

## Protocol

| Item | Value |
|------|-------|
| Handoff | 023-NG-P4 |
| Data / splits | `airfoil-self-noise-v1` (022 freeze; hashes verified) |
| Model | `NGBRegressor`, natural_gradient=True, Normal, MLE, default tree |
| Cap / LR / minibatch | 2000 / 0.01 / 1.0 |
| Stage selector | validation RMSE |
| Seeds | stage `20000+fold`, full `30000+fold` (independent of split RNG) |
| Threads | OMP/OPENBLAS/MKL/NUMEXPR = 1 |
| Env | `.RnD/work/ngboost/.venv` unchanged |

## Per-fold results

| Fold | Best iter | Test NLL | Test RMSE | Cov50 | Cov80 | Cov95 | Mean width95 | Cap hit | Time (s) |
|------|-----------|----------|-----------|-------|-------|-------|--------------|---------|----------|
| 0 | 2000 | 2.223051 | 2.299805 | 0.333 | 0.673 | 0.873 | 7.040 | True | 39.3 |
| 1 | 1995 | 2.174082 | 2.309458 | 0.407 | 0.707 | 0.900 | 6.642 | False | 42.9 |
| 2 | 2000 | 2.144915 | 2.236127 | 0.387 | 0.693 | 0.927 | 6.890 | True | 39.6 |
| 3 | 2000 | 2.188369 | 2.202265 | 0.400 | 0.700 | 0.927 | 6.518 | True | 40.4 |
| 4 | 2000 | 2.359773 | 2.894497 | 0.300 | 0.620 | 0.880 | 7.406 | True | 41.9 |

## Aggregates (n=5 folds)

| Metric | Mean | Median | Sample SD | SE |
|--------|------|--------|-----------|-----|
| Test NLL | 2.218038 | 2.188369 | 0.084073 | 0.037599 |
| Test RMSE | 2.388430 | 2.299805 | 0.286378 | 0.128072 |
| Best iter | 1999.0 | 2000.0 | 2.2 | 1.0 |

### Coverage / width (fold-mean)

| Level | Coverage mean (SE) | Mean width mean |
|-------|--------------------|-----------------|
| 50% | 0.3653 (0.0208) | 2.3742 |
| 80% | 0.6787 (0.0157) | 4.5111 |
| 95% | 0.9013 (0.0112) | 6.8992 |

Micro-average over 750 test rows: Cov50=0.3653, Cov80=0.6787, Cov95=0.9013.

Cap hits (best_iter == 2000): **4** / 5.

## Runtime

| | Seconds |
|--|---------|
| Wall | 226.9 |
| Sum fold totals | 204.1 |

## Verification

- CSV SHA `705a2a7f…` and splits SHA `128c5375…` match 022 manifest.
- Per-fold outer/inner index hashes match frozen `split-indices.json`.
- All predictive means/scales/interval bounds finite; scales > 0.
- Source status before/after recorded (repair-worktree commit `6720340…`). Pre-existing `M examples/experiments/regression_exp.py` unchanged by this run; no package edits.

## Claim boundary

Natural-model evidence only. Do not rank against ordinary or homoscedastic until those handoffs finish on the same five folds.

## Artifacts

`artifacts/airfoil-self-noise/ngboost-natural/023-ng-p4-airfoil-natural-s1-a01/`

Includes: `metrics.json`, `predictions.jsonl`, `predictions.csv`, `manifest.json`, `checksums.txt`, `config.json`, `command.txt`, `environment.txt`, `stdout.log`, `stderr.log`, `timing.json`, `source-status-before.txt`, `source-status-after.txt`.

## Stop

023-NG-P4 complete. No STATUS edit. No other variant/dataset.
