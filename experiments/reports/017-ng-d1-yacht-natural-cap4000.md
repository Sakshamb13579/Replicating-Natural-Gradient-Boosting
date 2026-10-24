# 017-NG-D1 Yacht Natural Cap-4000 Diagnostic

**Handoff:** `017-NG-D1`  
**Dataset:** Yacht only  
**Model:** Natural-gradient NGBoost only  
**Completed:** 2026-08-23  
**Artifacts:** `ngboost/artifacts/yacht/ngboost-natural/017-ng-d1-yacht-natural-s1-a01/`  
**Prior reference:** `.RnD/work/ngboost/raw/yacht-natural-repaired-split20.log` (cap 2000, NG-1A/NG-1B)

## 1. Verdict

**Diagnostic complete.** Raising `n_estimators` from **2,000 → 4,000** on the same 20 Yacht splits **materially changes** stage selection and **worsens aggregate test NLL** (mean 2.85 vs 2.11). This supports **retaining cap 2,000 as the protocol-faithful default** and documenting cap-4000 as a **negative sensitivity check**, not a substitute replication. The prior directional finding (natural gradient best among the three Yacht variants under cap 2000) is **not overturned by this run alone**, but split-level NLL instability at higher caps (especially split 8) warrants a sensitivity footnote.

## 2. Scope and Command

| Field | Value |
|---|---|
| Worktree | `.RnD/work/ngboost/repair-worktree/` @ `672034041fc05d9d4b7a80c1e6163ea102579752` |
| Environment | `.RnD/work/ngboost/.venv/` (Python 3.10.0, ngboost 0.5.11.dev0) |
| **Only change from prior natural run** | `--n-est=4000` (prior default 2000) |
| Preserved | `--dataset=yacht --score=MLE --natural --n-splits=20`; distn=Normal; lr=0.01; tree base; minibatch_frac=1.0; RMSE stage selection; split seed sequence |

```bash
python -m examples.experiments.regression_exp \
  --dataset=yacht --score=MLE --natural --n-splits=20 --n-est=4000
```

Thread limits: `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1`. `PYTHONDONTWRITEBYTECODE=1`.

## 3. Source Cleanliness

| Check | Before | After |
|---|---|---|
| Commit | `672034041fc05d9d4b7a80c1e6163ea102579752` | **Unchanged** |
| Working tree diff | Import repair only (`Normal`, `NormalFixedVar`, `MLE` imports in `regression_exp.py`) | **Identical** — no new edits |

Evidence: `source-status-before.txt`, `source-status-after.txt` in artifact directory.

## 4. Runtime

| Run | Wall time (s) | Exit |
|---|---:|---:|
| Prior cap 2000 | 864.31 | 0 |
| **Current cap 4000** | **2352** | **0** |

~2.7× longer runtime for cap doubling (expected; not all folds hit cap).

## 5. Aggregate Comparison

### Test NLL

| Statistic | Cap 2000 (prior) | Cap 4000 (current) | Δ (4000 − 2000) |
|---|---:|---:|---:|
| Mean | 2.1060 | **2.8485** | **+0.7425** |
| Median | 1.7924 | 1.7924 | 0.0000 |
| Pop. SD | 2.3466 | **3.5581** | +1.2115 |
| Sample SD | 2.4075 | 3.6505 | — |
| SE | 0.5383 | 0.8163 | — |

### Test RMSE

| Statistic | Cap 2000 | Cap 4000 | Δ |
|---|---:|---:|---:|
| Mean | 0.5576 | 0.5599 | +0.0023 |
| Median | 0.5232 | 0.5225 | −0.0007 |
| Pop. SD | 0.1718 | 0.1648 | −0.0070 |

RMSE barely moves; **NLL degradation is not driven by mean RMSE change**.

### Best iteration / cap hits

| Metric | Cap 2000 | Cap 4000 |
|---|---:|---:|
| Mean BestIter | 1847.1 | **3196.0** |
| Median BestIter | 2000.0 | **3826.0** |
| Exact cap hits | 11 @ 2000 | **9 @ 4000** |
| Near cap (≥3990) | — | **10** |
| Prior-cap folds now >2000 | — | **9 splits** (4,5,7,8,11,14,17,18,19) |

Stage selection shifts upward when headroom increases; RMSE-based early stopping selects deeper models on many previously 2000-capped folds.

### Validation RMSE (stage-selection metric)

| Statistic | Cap 2000 | Cap 4000 |
|---|---:|---:|
| Mean | 0.7082 | 0.6693 |
| Median | 0.6647 | 0.6497 |

Lower mean val RMSE at cap 4000 reflects access to more iterations, but **does not translate to better test NLL**.

**Note:** Upstream `regression_exp.py` computes validation NLL internally but **does not print it**; val RMSE and test NLL are parsed from stdout for both runs.

## 6. Paired Per-Split Comparison (20 splits)

| Outcome | Test NLL | Test RMSE |
|---|---:|---:|
| Cap 4000 wins | **4** | 12 |
| Ties | 0 | 1 |
| Cap 4000 losses | **16** | 7 |

| Paired summary | Value |
|---|---:|
| Mean Δ test NLL (4000 − 2000) | **+0.7425** |
| Median Δ test NLL | +0.0297 |
| Mean Δ BestIter | +1348.9 |

### Per-split table

| Split | BestIter 2000 | BestIter 4000 | Δ Iter | Val RMSE 2000 | Val RMSE 4000 | Test NLL 2000 | Test NLL 4000 | Δ NLL | Prior cap@2000 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 1985 | 2026 | +41 | 0.6012 | 0.6012 | 0.3801 | 0.3802 | +0.0001 | |
| 2 | 2000 | 1874 | −126 | 1.0050 | 0.8551 | 1.0070 | 1.0279 | +0.0209 | ✓ |
| 3 | 2000 | 300 | −1700 | 1.0525 | 0.6468 | 0.0973 | 0.9825 | **+0.8852** | ✓ |
| 4 | 2000 | 3542 | +1542 | 0.8565 | 0.9412 | 2.2815 | 2.4691 | +0.1876 | ✓ |
| 5 | 2000 | 3381 | +1381 | 0.5738 | 0.6850 | 3.5413 | 3.7299 | +0.1886 | ✓ |
| 6 | 1786 | 4000 | +2214 | 0.4285 | 0.5282 | 0.5099 | 0.5415 | +0.0316 | |
| 7 | 2000 | 4000 | +2000 | 0.6311 | 0.9247 | −0.2473 | −0.3016 | −0.0543 | ✓ |
| 8 | 2000 | 4000 | +2000 | 0.5885 | 0.7256 | 2.3001 | **14.4738** | **+12.1737** | ✓ |
| 9 | 1722 | 4000 | +2278 | 0.8041 | 1.0236 | 0.8746 | 1.6498 | +0.7752 | |
| 10 | 436 | 4000 | +3564 | 0.5747 | 0.6913 | 0.0974 | 0.4876 | +0.3902 | |
| 11 | 2000 | 4000 | +2000 | 0.9313 | 0.3161 | 1.6838 | 1.6889 | +0.0051 | ✓ |
| 12 | 1977 | 3261 | +1284 | 0.4392 | 0.4068 | 3.2554 | 2.6688 | **−0.5866** | |
| 13 | 1813 | 3998 | +2185 | 0.4185 | 0.6525 | 2.5615 | 2.5894 | +0.0279 | |
| 14 | 2000 | 4000 | +2000 | 1.0019 | 0.5209 | 3.0478 | 3.0650 | +0.0172 | ✓ |
| 15 | 1999 | 4000 | +2001 | 0.6093 | 0.7233 | 3.7344 | 3.7422 | +0.0078 | |
| 16 | 1249 | 722 | −527 | 0.4088 | 0.4968 | 1.1122 | 0.9081 | **−0.2041** | |
| 17 | 2000 | 2593 | +593 | 1.0632 | 0.6079 | 0.6090 | 0.8106 | +0.2016 | ✓ |
| 18 | 2000 | 3654 | +1654 | 0.6982 | 0.4245 | 2.4639 | 2.8933 | +0.4294 | ✓ |
| 19 | 2000 | 4000 | +2000 | 0.7790 | 0.2896 | 1.9010 | 1.8959 | −0.0051 | ✓ |
| 20 | 1975 | 2569 | +594 | 0.6986 | 1.3242 | 10.9093 | 11.2668 | +0.3575 | |

**Cap-4000 wins (NLL):** splits 7, 12, 16, 19 (4/20).  
**Largest NLL harm:** split **8** (+12.17) when BestIter moves from 2000 → 4000.  
**Previously 2000-capped folds moving beyond 2000:** 9 splits; 8 of 9 have higher test NLL (exception: split 19, −0.005).

## 7. Interpretation

### Does cap saturation materially change stage selection?

**Yes.** Mean BestIter rises from 1847 → 3196; nine prior 2000-capped folds select >2000 iterations under cap 4000; nine folds hit the new cap exactly.

### Does it materially change test NLL?

**Yes.** Mean test NLL worsens by **+0.74** (35% relative increase); population SD widens from 2.35 → 3.56. Median unchanged (1.79) because degradation is concentrated in a few heavy-tail folds (8, 3, 10, 9).

### Does it change the directional result?

**Not directly tested** (no ordinary/homoscedastic re-run at cap 4000). Under cap 2000, natural gradient had mean test NLL 2.11 vs ordinary 11.55 and homoscedastic 13.28. Cap 4000 natural mean 2.85 remains far below those baselines, so **natural-as-best-variant direction likely holds**, but **absolute NLL reporting at cap 4000 would be misleadingly worse** and **less stable**.

### Recommendation

| Option | Supported? |
|---|---|
| Retain **cap 2000** as protocol-faithful default | **Yes** — matches upstream default, better aggregate NLL, faster |
| Sensitivity note on cap / stage depth | **Yes** — document split 8 instability and 16/20 NLL losses |
| Substitute cap 4000 as official replication | **No** |
| Claim cap saturation explains all paper gap | **No** — diagnostic only |

## 8. Artifacts

Directory: `ngboost/artifacts/yacht/ngboost-natural/017-ng-d1-yacht-natural-s1-a01/`

| File | Purpose |
|---|---|
| `manifest.json` | Run metadata |
| `config.json` | Frozen args and preserved defaults |
| `command.txt` | Exact shell command |
| `environment.txt` | Python + pip freeze |
| `stdout.log` / `stderr.log` | Raw run output |
| `timing.json` | Wall time 2352 s, exit 0 |
| `metrics.json` | Parsed per-split + aggregates + paired comparison |
| `checksums.txt` | SHA-256 of all artifact files |
| `source-status-before.txt` / `source-status-after.txt` | Git cleanliness evidence |

## 9. Compliance

- **Exactly one dataset/model executed** (Yacht natural NGBoost).
- **Same 20-split sequence**; only cap changed.
- **Source unchanged** before/after (import repair only).
- **No environment install, no source edit, no other model/dataset.**
- **STATUS not edited.**

---

**End of handoff 017-NG-D1. Cursor stops here.**
