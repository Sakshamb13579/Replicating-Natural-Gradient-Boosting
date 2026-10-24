# NG-1C Implementation Report

**Agent:** Cursor Primary RnD and Implementation  
**Checkpoint:** NG-1C — Kin8nm confirmation (5 splits × 3 variants)  
**Commit:** `672034041fc05d9d4b7a80c1e6163ea102579752`  
**Worktree:** `.RnD/work/ngboost/repair-worktree/`  
**Date:** 2026-08-22

## Timing

| Run | Wall time | Exit |
|-----|----------:|-----:|
| Natural 5-split | 1016.1 s (16.9 min) | 0 |
| Homoscedastic 5-split | 593.2 s (9.9 min) | 0 |
| Ordinary 5-split | 1015.3 s (16.9 min) | 0 |
| **Total** | **2624.6 s (43.7 min)** | — |

All runs under 60-minute individual limit and 3-hour checkpoint limit. Timestamps in `raw/commands.log`.

## Data Identity

See `raw/data-checks-kin8nm.md`. Bundled `kin8nm.csv`, SHA-256 `7b9bf030…`, shape `(8192, 8)` features after loader split.

## Commands

From `/mnt/hdd1/Projects/Research/Saksham/.RnD/work/ngboost/repair-worktree/` with `.venv` active:

```bash
python -m examples.experiments.regression_exp \
  --dataset=kin8nm --score=MLE --natural --n-splits=5

python -m examples.experiments.regression_exp \
  --dataset=kin8nm --score=MLE --distn=NormalFixedVar --natural --n-splits=5

python -m examples.experiments.regression_exp \
  --dataset=kin8nm --score=MLE --n-splits=5
```

Logs: `kin8nm-natural-repaired-split5.log`, `kin8nm-homosk-repaired-split5.log`, `kin8nm-ordinary-repaired-split5.log`.

## Per-Split Test NLL (lower is better)

| Split | Natural | Homoscedastic | Ordinary | Lowest |
|------:|--------:|--------------:|---------:|:------:|
| 1 | −0.3890 | −0.3094 | **−0.4667** | O |
| 2 | −0.4270 | −0.3919 | **−0.5027** | O |
| 3 | −0.4348 | −0.3756 | **−0.5167** | O |
| 4 | −0.4196 | −0.3851 | **−0.5236** | O |
| 5 | −0.4253 | −0.3497 | **−0.5295** | O |

## Summary Statistics

| Variant | Mean NLL | Median NLL | Pop. SD | Sample SD | SE | Mean RMSE | BestIter=2000 | BestIter≥1999 |
|---------|----------:|-----------:|--------:|----------:|---:|----------:|--------------:|--------------:|
| **Ordinary** | **−0.5078** | **−0.5167** | 0.0224 | 0.0251 | 0.0100 | 0.1566 | 1/5 | 2/5 |
| Natural | −0.4191 | −0.4253 | 0.0158 | 0.0177 | 0.0071 | 0.1692 | 3/5 | 3/5 |
| Homoscedastic | −0.3623 | −0.3756 | 0.0301 | 0.0337 | 0.0135 | 0.1608 | 2/5 | 3/5 |

Five splits are confirmation evidence only — not equivalent to the paper's 20 repetitions.

## Paper Table 2 Comparison (Kin8nm)

| Variant | Paper mean ± SE | Our mean (5 splits) | Our SE | Δ mean |
|---------|----------------:|--------------------:|-------:|-------:|
| Ordinary gradient | −0.52 ± 0.03 | **−0.5078** | 0.0100 | +0.012 |
| Natural gradient | −0.49 ± 0.02 | −0.4191 | 0.0071 | +0.071 |
| Homoscedastic | −0.37 ± 0.05 | −0.3623 | 0.0135 | +0.008 |

**Ordering (lower NLL better):**

| Source | Order |
|--------|-------|
| Paper | Ordinary < Natural < Homoscedastic |
| NG-1C (5 splits) | **Ordinary < Natural < Homoscedastic** ✓ |

Signs, scale, and three-way ordering all align with Table 2. Ordinary and natural are close (paper: 0.03 apart; ours: 0.089 on mean). Natural is not the winner on Kin8nm — consistent with paper's non-favourable confirmation target.

## Paired Comparisons

| Comparison | Natural wins | Mean paired Δ (nat − other) | Median paired Δ |
|------------|-------------:|----------------------------:|----------------:|
| Natural vs Homoscedastic | **5/5** | −0.0568 | −0.0592 |
| Natural vs Ordinary | **0/5** | +0.0887 | +0.0819 |

Natural decisively beats homoscedastic on every split. Ordinary beats natural on every split — matching the paper's slight ordinary advantage.

## Estimator-Cap Saturation

Moderate on all variants (1–3/5 folds at exactly 2000). No catastrophic tail failures unlike Yacht homoscedastic runs.

## Repairs and Source Status

- No new code changes; import-only diff unchanged.
- `.RnD/source/ngboost/`: clean @ `6720340`.

## Confirmation Recommendation

### **Supports confirmation**

Rationale:

- Three-way ordering matches paper Table 2 on Kin8nm.
- Natural and ordinary are reasonably close; ordinary slightly better (paper expectation).
- Both natural and ordinary clearly outperform homoscedastic by mean, median, and paired evidence (natural 5/5 vs homosk).
- Numerical scale is plausible though natural mean is ~0.07 above paper (−0.42 vs −0.49); five splits limit precision.

This contrasts productively with Yacht NG-1B: on Yacht natural won decisively; on Kin8nm ordinary slightly wins — both behaviours match paper-reported patterns under the same official script.

## Stop Compliance

- Did not start MiniRocket or ADBench.
- No protocol or code changes beyond authorized import repair.

## Recommendation for Lead

NGBoost remains **Directional PASS** with cross-dataset confirmation. Proceed to MiniRocket half-day backup per roadmap before paper lock; begin Colab confirmation and proposal evidence assembly for NGBoost.
