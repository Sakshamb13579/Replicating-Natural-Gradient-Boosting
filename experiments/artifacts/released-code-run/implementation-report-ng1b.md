# NG-1B Implementation Report

**Agent:** Cursor Primary RnD and Implementation  
**Checkpoint:** NG-1B — full Yacht Table 2 slice  
**Commit:** `672034041fc05d9d4b7a80c1e6163ea102579752`  
**Worktree:** `.RnD/work/ngboost/repair-worktree/`  
**Date:** 2026-08-22

## Timing

| Run | Start (IST) | End (IST) | Wall time | Exit |
|-----|-------------|-----------|----------:|-----:|
| Homoscedastic 20-split | 20:58:xx | 21:09:xx | 658.4 s | 0 |
| Ordinary 20-split | 21:09:xx | 21:22:xx | 788.6 s | 0 |

Natural 20-split reused from NG-1A (864.3 s, 2026-08-22).

Timestamps in `raw/commands.log`.

## Commands

From `/mnt/hdd1/Projects/Research/Saksham/.RnD/work/ngboost/repair-worktree/` with `.venv` active:

```bash
python -m examples.experiments.regression_exp \
  --dataset=yacht --score=MLE --distn=NormalFixedVar --natural --n-splits=20

python -m examples.experiments.regression_exp \
  --dataset=yacht --score=MLE --n-splits=20
```

Logs: `yacht-homosk-repaired-split20.log`, `yacht-ordinary-repaired-split20.log`, `yacht-natural-repaired-split20.log`.

## Aggregate Summary (test NLL)

| Variant | Mean | Median | Pop. SD | Sample SD | SE | Mean RMSE | Median RMSE | Runtime (s) | BestIter=2000 | BestIter≥1999 |
|---------|-----:|-------:|--------:|----------:|---:|----------:|------------:|------------:|--------------:|--------------:|
| **Natural** | **2.1060** | **1.7924** | 2.3466 | 2.4075 | 0.5247 | 0.5576 | 0.5232 | 864.3 | 11/20 | 12/20 |
| Ordinary | 11.5473 | 6.7826 | 12.4147 | 12.7373 | 2.7760 | 13.8165 | 13.8304 | 788.6 | **19/20** | **20/20** |
| Homoscedastic | 13.2821 | 9.6021 | 11.6455 | 11.9481 | 2.6040 | 0.5508 | 0.4667 | 658.4 | 6/20 | 8/20 |

**Mean ordering:** Natural < Ordinary < Homoscedastic  
**Median ordering:** Natural < Ordinary < Homoscedastic

Paper Table 2 ordering: Natural < **Homoscedastic** < Ordinary.  
Our aggregate homoscedastic vs ordinary ordering is **inverted** relative to the paper, even though natural gradient wins overall.

## Table 2 Comparison

| Variant | Paper mean ± SE | Our mean | Our SE (computed) | Directional match? |
|---------|----------------:|---------:|------------------:|--------------------|
| Natural gradient | 0.20 ± 0.26 | 2.1060 | 0.5247 | Best variant — yes |
| Homoscedastic | 2.02 ± 0.21 | 13.2821 | 2.6040 | Exact value — no |
| Ordinary gradient | 3.29 ± 0.20 | 11.5473 | 2.7760 | Exact value — no |

Exact numerical reproduction: **not achieved** on any variant. Natural gradient remains the best of the three under this implementation.

## Paired Split Analysis (20 identical splits)

| Comparison | Natural wins | Ties | Mean paired Δ (nat − other) | Median paired Δ |
|------------|-------------:|-----:|----------------------------:|----------------:|
| Natural vs Homoscedastic | **20/20** | 0 | −11.1761 | −8.0955 |
| Natural vs Ordinary | **19/20** | 0 | −9.4413 | −5.9404 |

Per-split lowest-NLL winner: Natural **19**, Ordinary **1** (split 20), Homoscedastic **0**.

Split 20: natural NLL 10.9093, ordinary 4.5031, homoscedastic 23.7968 — natural upper-tail failure; only fold where ordinary beats natural.

## Per-Split Test NLL

| Split | Natural | Homoscedastic | Ordinary | Winner |
|------:|--------:|--------------:|---------:|:------:|
| 1 | 0.3801 | 11.3590 | 7.3588 | N |
| 2 | 1.0070 | 11.8524 | 3.2243 | N |
| 3 | 0.0973 | 4.6287 | 37.8738 | N |
| 4 | 2.2815 | 5.1721 | 3.3077 | N |
| 5 | 3.5413 | 22.2296 | 36.0346 | N |
| 6 | 0.5099 | 5.0678 | 4.1184 | N |
| 7 | −0.2473 | −0.0464 | 7.2129 | N |
| 8 | 2.3001 | 22.9376 | 3.5113 | N |
| 9 | 0.8746 | 6.6041 | 3.7585 | N |
| 10 | 0.0974 | 3.1404 | 6.8657 | N |
| 11 | 1.6838 | 2.0240 | 9.2503 | N |
| 12 | 3.2554 | 49.8338 | 18.6567 | N |
| 13 | 2.5615 | 23.6320 | 3.5407 | N |
| 14 | 3.0478 | 26.8310 | 5.6126 | N |
| 15 | 3.7344 | 14.6482 | 44.5550 | N |
| 16 | 1.1122 | 8.7988 | 6.9026 | N |
| 17 | 0.6090 | 4.1145 | 6.6994 | N |
| 18 | 2.4639 | 8.6123 | 3.4379 | N |
| 19 | 1.9010 | 10.4054 | 14.5222 | N |
| 20 | 10.9093 | 23.7968 | 4.5031 | O |

## Outlier and Estimator-Cap Analysis

### Folds with test NLL > 10

| Variant | Folds (split, NLL) |
|---------|-------------------|
| Natural | (20, 10.91) |
| Homoscedastic | (1, 11.36), (2, 11.85), (5, 22.23), (8, 22.94), (12, **49.83**), (13, 23.63), (14, 26.83), (15, 14.65), (19, 10.41), (20, 23.80) |
| Ordinary | (3, 37.87), (5, 36.03), (12, 18.66), (15, 44.56), (19, 14.52) |

### Mean sensitivity (excluding NLL > 10)

| Variant | Full mean | Trimmed mean (NLL ≤ 10) | Folds retained |
|---------|----------:|--------------------------:|---------------:|
| Natural | 2.1060 | 1.6427 | 19/20 |
| Homoscedastic | 13.2821 | 4.8116 | 10/20 |
| Ordinary | 11.5473 | 5.2869 | 15/20 |

Trimming reduces homoscedastic/ordinary means substantially but **does not change** natural being best. Homoscedastic remains the noisiest variant with the most catastrophic failures (split 12: 49.83).

### Estimator-cap saturation

- Ordinary: 19/20 folds at exactly 2000 — near-total cap saturation.
- Natural: 11/20 at cap — moderate.
- Homoscedastic: 6/20 at cap — lower cap rate but worse tail failures.

## Repairs and Source Status

- No new code changes; import-only diff unchanged (`raw/repair-imports.diff`).
- `.RnD/source/ngboost/`: clean @ `6720340`.

## Recommended Classification

### **Directional PASS**

Rationale against STATUS verdict rules:

| Criterion | Result |
|-----------|--------|
| Natural lowest mean NLL | Yes (2.11 vs 11.55 vs 13.28) |
| Natural lowest median NLL | Yes (1.79 vs 6.78 vs 9.60) |
| Clear majority vs homoscedastic | Yes — **20/20** |
| Clear majority vs ordinary | Yes — **19/20** |
| Natural consistently loses to an ablation | No |

Caveats (prevent strong numerical pass):

- All three means diverge sharply from paper Table 2 absolute values.
- Paper homoscedastic-vs-ordinary ordering is **not** reproduced (we get Ordinary < Homoscedastic).
- RMSE-based stage selection vs paper's described NLL selection remains unresolved.
- Upper-tail NLL failures, especially homoscedastic split 12 (49.83), inflate aggregate statistics.
- sklearn 1.6.1 / numpy 2.2.6 environment differs from paper era.

**Central directional claim supported:** natural-gradient NGBoost achieves lower test NLL than both official ablations on paired Yacht splits under the current official script.

**Not supported:** exact Table 2 values or full three-way ordering identical to the paper.

## Stop Compliance

- Did not start NG-1C, MiniRocket, or ADBench.
- Did not modify stage selection or any research logic.

## Recommendation for Lead

Authorize **NG-1C** (Wine or Kin8nm confirmation) if Directional PASS is accepted. Proceed with MiniRocket half-day backup per roadmap before final paper lock. NGBoost remains viable as proposal paper under a **directional reproduction** narrative with documented protocol deviations.
