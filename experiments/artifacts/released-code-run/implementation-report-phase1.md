# Phase 1A Implementation Report

**Agent:** Cursor Primary RnD and Implementation  
**Commit:** `672034041fc05d9d4b7a80c1e6163ea102579752`  
**Worktree:** `.RnD/work/ngboost/repair-worktree/` (import-only repair unchanged)  
**Date:** 2026-08-22

## Timing

| Run | Start (approx IST) | Wall time | Exit |
|-----|-------------------|----------:|-----:|
| Homoscedastic 2-split | ~20:21 | 68.4 s | 0 |
| Ordinary 2-split | ~20:22 | 82.0 s | 0 |
| Natural 20-split | ~20:24 | 864.3 s (14.4 min) | 0 |

Commands logged in `raw/commands.log`.

## Environment

Unchanged from Phase 0 — see `raw/environment.txt`. Editable install points to `repair-worktree/`.

## Exact Commands

All executed from `/mnt/hdd1/Projects/Research/Saksham/.RnD/work/ngboost/repair-worktree/` with `.RnD/work/ngboost/.venv` activated.

```bash
python -m examples.experiments.regression_exp \
  --dataset=yacht --score=MLE --distn=NormalFixedVar --natural --n-splits=2

python -m examples.experiments.regression_exp \
  --dataset=yacht --score=MLE --n-splits=2

python -m examples.experiments.regression_exp \
  --dataset=yacht --score=MLE --natural --n-splits=20
```

Logs:

- `raw/yacht-homosk-repaired-split2.log`
- `raw/yacht-ordinary-repaired-split2.log`
- `raw/yacht-natural-repaired-split20.log`

## Per-Split Tables

### Homoscedastic (`NormalFixedVar`, natural), 2 splits

| Split | BestIter | Val RMSE | Test RMSE | Test NLL |
|------:|---------:|---------:|----------:|---------:|
| 1 | 1999 | 0.4570 | 0.4911 | 10.0080 |
| 2 | 1356 | 0.4123 | 0.4076 | 5.7849 |

### Ordinary gradient, 2 splits

| Split | BestIter | Val RMSE | Test RMSE | Test NLL |
|------:|---------:|---------:|----------:|---------:|
| 1 | 2000 | 12.4460 | 14.5827 | 7.3588 |
| 2 | 2000 | 14.6781 | 13.3965 | 3.2243 |

### Natural gradient, 20 splits

| Split | BestIter | Val RMSE | Test RMSE | Test NLL |
|------:|---------:|---------:|----------:|---------:|
| 1 | 1985 | 0.6012 | 0.5265 | 0.3801 |
| 2 | 2000 | 1.0050 | 0.4566 | 1.0070 |
| 3 | 2000 | 1.0525 | 0.3318 | 0.0973 |
| 4 | 2000 | 0.8565 | 0.4050 | 2.2815 |
| 5 | 2000 | 0.5738 | 0.6073 | 3.5413 |
| 6 | 1786 | 0.4285 | 0.5200 | 0.5099 |
| 7 | 2000 | 0.6311 | 0.2338 | -0.2473 |
| 8 | 2000 | 0.5885 | 0.7739 | 2.3001 |
| 9 | 1722 | 0.8041 | 0.4799 | 0.8746 |
| 10 | 436 | 0.5747 | 0.4791 | 0.0974 |
| 11 | 2000 | 0.9313 | 0.4523 | 1.6838 |
| 12 | 1977 | 0.4392 | 0.8036 | 3.2554 |
| 13 | 1813 | 0.4185 | 0.7035 | 2.5615 |
| 14 | 2000 | 1.0019 | 0.5835 | 3.0478 |
| 15 | 1999 | 0.6093 | 0.9425 | 3.7344 |
| 16 | 1249 | 0.4088 | 0.6179 | 1.1122 |
| 17 | 2000 | 1.0632 | 0.4025 | 0.6090 |
| 18 | 2000 | 0.6982 | 0.5529 | 2.4639 |
| 19 | 2000 | 0.7790 | 0.4750 | 1.9010 |
| 20 | 1975 | 0.6986 | 0.8048 | 10.9093 |

## Summary Statistics

Dispersion naming follows lead guidance: script `np.std` default is **population SD** (`ddof=0`). Standard error reported as `population_sd / sqrt(n)`.

| Run | n | Mean test NLL | Pop. SD | Sample SD | SE | Mean test RMSE | Runtime (s) |
|-----|--:|--------------:|--------:|----------:|---:|---------------:|------------:|
| Natural 2-split (Phase 0) | 2 | 0.6555 | 0.3665 | 0.5183 | 0.2592 | 0.4934 | 88.2 |
| Homoscedastic 2-split | 2 | 7.8965 | 2.1115 | 2.9862 | 1.4931 | 0.4494 | 68.4 |
| Ordinary 2-split | 2 | 5.2916 | 2.0673 | 2.9235 | 1.4618 | 13.9896 | 82.0 |
| **Natural 20-split** | 20 | **2.1060** | **2.3466** | 2.4075 | **0.5247** | 0.5576 | 864.3 |

Script summary line for 20-split natural: `NLL NGB=2.1060 +/ 2.3466` — the `+/` value matches **population SD**, not standard error.

## Estimator-Cap Saturation (`n_estimators=2000`)

| Run | BestIter == 2000 | BestIter >= 1999 | Notes |
|-----|-----------------:|-----------------:|-------|
| Natural 2-split | 1/2 | 1/2 | |
| Homoscedastic 2-split | 0/2 | 1/2 | split 1 stopped at 1999 |
| Ordinary 2-split | **2/2** | **2/2** | full cap saturation |
| Natural 20-split | **11/20** | **12/20** | 55% hit exact cap; 60% within 1 iter |

High cap selection suggests RMSE-based early stopping may be under-resolved for NLL comparison on some folds. Lead should factor this into Phase 1B verdict.

## Two-Split Exploratory Ordering (non-decisive)

Mean test NLL (low → high):

1. Natural — 0.6555
2. Ordinary — 5.2916
3. Homoscedastic — 7.8965

Paper Table 2 target ordering: **Natural < Homoscedastic < Ordinary**.

Two-split result: Natural wins, but **Homoscedastic is worst**, not middle. Per lead instruction, this cannot reject the candidate.

## Twenty-Split Natural vs Paper Table 2

| | Paper Yacht test NLL | This run (20-split natural) |
|--|---------------------:|----------------------------:|
| Mean | 0.20 | 2.1060 |
| Dispersion reported | ± 0.26 (**standard error**) | ± 2.3466 (**population SD** in script) |
| Standard error (computed) | 0.26 (paper) | 0.5247 (`pop_sd / sqrt(20)`) |

Numerical gap is large. Plausible contributing factors (not altered in this phase):

- Paper describes validation **NLL** stage selection; script selects by validation **RMSE**
- Package/version drift (sklearn 1.6.1, numpy 2.2.6 vs paper era)
- Estimator cap saturation on 55% of natural folds
- Paper SE vs script population SD reporting mismatch

Execution remained faithful to the official script after import-only repair.

## Repairs and Source Status

- No new code changes beyond Phase 0 import-only diff (`raw/repair-imports.diff`)
- `.RnD/source/ngboost/`: clean @ `6720340`
- Worktree: detached HEAD @ `6720340`

## Warnings

- Pandas `delim_whitespace` FutureWarning on Yacht load — non-blocking
- One natural fold (split 7) returned negative test NLL (-0.2473); valid for log-density but increases variance
- Split 20 natural NLL spike (10.9093) drives upper tail

## Stop Conditions

- No new compatibility failures
- 20-split natural completed under 45-minute limit (14.4 min)
- **Did not** run 20-split homoscedastic or ordinary (awaiting Phase 1B handoff)

## Recommendation for Lead

**Proceed to Phase 1B** — all Phase 1A runs completed without further repair. Numerical reproduction is **not yet validated**; full three-variant 20-split comparison is required before strong/directional pass classification.

Functional execution: **PASS**. Table 2 numerical match: **open — likely directional at best pending full ablation runs**.
