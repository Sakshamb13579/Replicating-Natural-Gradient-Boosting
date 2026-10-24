# 028-NG-P5 — Heteroscedastic natural-gradient NGBoost (regimes A/B/C)

## Verdict

Official-library **natural-gradient** NGBoost ran stably on all three frozen heteroscedastic regimes. Separate models per regime (2,400/800/800 train/val/test) with **validation-NLL** stage selection produced finite, positive predictive scales and near-nominal **95% coverage** (0.946–0.960). Regime **A** and **C** show strong predicted-scale tracking of true σ (Pearson ≈ 0.97–0.98); regime **B** reports `null` correlation as required (constant σ). This establishes natural-model feasibility on constructed ground-truth data only — **no superiority claim** vs ordinary/homoscedastic until those paired runs finish.

## Protocol

| Item | Value |
|------|-------|
| Handoff | 028-NG-P5 |
| Data | `heteroscedastic-v1` (007 freeze; CSV SHA verified) |
| Model | `NGBRegressor`, natural_gradient=True, Normal, MLE, default tree |
| Cap / LR / minibatch | 2000 / 0.01 / 1.0 |
| Stage selector | **validation NLL** (constructed uncertainty study; differs from official `regression_exp.py` RMSE selection) |
| Features | `x1`, `x2` only; `true_mu`, `true_sigma`, `noise_z` evaluation-only |
| Seeds | stage `20000+regime_idx`, full `30000+regime_idx` (A/B/C → 20000–20002 / 30000–30002) |
| Threads | OMP/OPENBLAS/MKL/NUMEXPR = 1 |
| Env | `.RnD/work/ngboost/.venv` unchanged |

## Per-regime results

| Regime | σ(x) role | Best iter | Cap hit | Test NLL | Oracle NLL | Test RMSE | Scale MAE | Cov50 | Cov80 | Cov95 | Pearson ρ(scale, σ) | Time (s) |
|--------|-----------|-----------|---------|----------|------------|-----------|-----------|-------|-------|-------|---------------------|----------|
| A | `0.1+0.5\|x1\|` | 377 | No | 0.7891 | 0.7310 | 0.6815 | 0.0602 | 0.510 | 0.803 | 0.948 | 0.969 | 58.3 |
| B | constant 0.6 | 300 | No | 0.9150 | 0.8850 | 0.5973 | 0.0463 | 0.528 | 0.811 | 0.960 | **null** | 57.9 |
| C | `2·(0.1+0.5\|x1\|)` | 300 | No | 1.4679 | 1.4242 | 1.3540 | 0.1086 | 0.515 | 0.811 | 0.946 | 0.975 | 58.2 |

Test NLL exceeds oracle NLL on all regimes (expected gap from finite-sample estimation). No cap hits (best iter well below 2000).

### Regime A — `|x1|` quantile bins (test, n=800)

| Bin | mean \|x1\| | mean true σ | mean pred scale | mean width95 |
|-----|--------------|-------------|-----------------|--------------|
| 1 | 0.257 | 0.228 | 0.269 | 1.053 |
| 2 | 0.738 | 0.469 | 0.455 | 1.785 |
| 3 | 1.244 | 0.722 | 0.708 | 2.776 |
| 4 | 1.738 | 0.969 | 0.897 | 3.514 |

Monotonic widening of predicted scale and 95% width with \|x1\| matches ground-truth heteroscedastic structure (supports pre-registered H2 directionally; full hypothesis test awaits three-way comparison).

### Regime B — homoscedastic control

True σ is constant (0.6). Scale correlation is **`null`** (not fabricated). Predicted scale shows mild \|x1\|-linked variation across bins (0.61–0.64) despite constant σ — a diagnostic for the ordinary/homoscedastic comparison, not a victory condition for natural gradient.

### Regime C — amplified heteroscedastic

Same functional \|x1\| dependence as A at doubled amplitude. Strong scale tracking (Spearman 0.979); bin summaries show wider intervals than A at comparable \|x1\|.

## Runtime

| | Seconds |
|--|---------|
| Wall | 183.8 |
| Sum regime totals | 174.3 |
| Shell elapsed | 188 |

## Verification

- CSV SHA `a39e58f…` matches 007 manifest; split `row_id` lists verified per regime.
- Ground truth excluded from model features.
- All predictive means/scales/interval bounds finite; scales > 0.
- Stage selection implemented via `staged_pred_dist` + validation NLL argmin.
- Source status before/after recorded (repair-worktree commit `6720340…`; pre-existing `M examples/experiments/regression_exp.py` unchanged).

## Claim boundary

Natural-gradient feasibility on constructed heteroscedastic regimes only. Do not rank against ordinary or homoscedastic variants until those handoffs finish on identical frozen splits.

## Artifacts

`artifacts/heteroscedastic-v1/ngboost-natural/028-ng-p5-heteroscedastic-natural-s0-a01/`

Includes: `metrics.json`, `predictions.jsonl`, `predictions.csv` (2,400 test rows with true/pred mean/scale and intervals), `manifest.json`, `checksums.txt`, `config.json`, `command.txt`, `environment.txt`, `stdout.log`, `stderr.log`, `timing.json`, `source-status-before.txt`, `source-status-after.txt`.

## Stop

028-NG-P5 complete. No STATUS edit. No other variant/dataset.
