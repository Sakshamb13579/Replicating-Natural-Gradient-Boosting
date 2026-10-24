# 017F1-NG-D1 Controlled Yacht Cap Diagnostic

**Handoff:** `017F1-NG-D1`  
**Parent:** `017-NG-D1`  
**Dataset:** Yacht only  
**Model:** Natural-gradient NGBoost only  
**Completed:** 2026-08-23  
**Artifacts:** `ngboost/artifacts/yacht/ngboost-natural/017-ng-d1-yacht-natural-s1-a02/`  
**Harness:** `ngboost/src/controlled_yacht_cap_diagnostic.py`  
**a01 (preserved):** `…/017-ng-d1-yacht-natural-s1-a01/` — **not overwritten**

## 1. Verdict

**Controlled diagnostic complete.** With frozen outer/inner splits and **one shared 4,000-stage trajectory per fit**, selecting stages under a 2,000-stage ceiling vs a 4,000-stage ceiling still **worsens mean test NLL** (+0.168; cap-4000 wins 2 / ties 5 / loses 13). Extra headroom itself tends to **harm** probabilistic performance under RMSE stage selection. Retain **cap 2,000** as the protocol-faithful default; treat a01’s larger paired gap as **confounded**, not as a pure cap effect.

## 2. a01 Defect (Preserved, Not Used as Pure Cap Evidence)

Official `regression_exp.py` precomputes outer 90/10 folds, but calls `train_test_split(...)` **without `random_state`** inside the fold loop. Fitting to 4,000 stages consumes more of NumPy’s global RNG than fitting to 2,000, so **later folds receive different inner validation partitions**. Therefore a01 cannot isolate estimator headroom.

| a01 claim | Status after 017F1 |
|---|---|
| Cap 4000 vs 2000 under official script | **Confounded** by inner-split RNG coupling |
| a01 artifacts | **Preserved unchanged** |
| Use a01 as pure headroom sensitivity | **Do not** |

## 3. Controlled Design (This Run)

For each of 20 outer folds (same construction seed as official: `np.random.RandomState(1)` permutations):

1. Freeze inner 80/20 with `random_state = 10000 + fold_idx` (**independent of fitting**).
2. Seed stage-selection model with `20000 + fold_idx`; full-train model with `30000 + fold_idx`.
3. Fit **one** natural NGBoost to **4,000** stages on inner train.
4. Choose `best_iter_cap2000` from stages 1–2000 and `best_iter_cap4000` from stages 1–4000 via **validation RMSE** on that same staged path.
5. Fit **one** 4,000-stage natural model on full outer train.
6. Evaluate the **same** full-train model on the outer test set at both selected iteration counts (`pred_dist(..., max_iter=…)`).

This is a **diagnostic harness**, not an official-script replication.

## 4. Source Cleanliness

| Check | Result |
|---|---|
| Commit | `672034041fc05d9d4b7a80c1e6163ea102579752` |
| Before/after status | **Identical** (import repair only; no new upstream edits) |

## 5. Shared-Trajectory Verification

| Check | Result |
|---|---|
| Design | Cap views from one 4000-stage full-train model via `max_iter` |
| In-run flag bug | Incorrectly called `.mean()` on staged **prediction vectors** (scalarized rows) |
| Post-hoc fold-1 re-fit | `best_iter` matches stored; `predict(max_iter=k)` equals `staged_predict[k-1]` for selected k and spot checks 1/500/1000/2000 |
| NLL re-match fold 1 | Stored vs re-fit agree to ~1e-14 |

Evidence: `posthoc-shared-trajectory-check.json`. Harness verification code corrected for future runs; metrics retain historical in-run flags plus corrected post-hoc confirmation.

## 6. Aggregate Results (Controlled)

### Test NLL

| Statistic | Cap view ≤2000 | Cap view ≤4000 | Δ (4000−2000) |
|---|---:|---:|---:|
| Mean | **2.0351** | **2.2029** | **+0.1678** |
| Median | 1.2554 | 1.4770 | +0.2216 |
| Pop. SD | 2.3493 | 2.5334 | +0.1840 |
| Sample SD | 2.4104 | 2.5992 | — |
| SE | 0.5390 | 0.5812 | — |

### Test RMSE

| Statistic | ≤2000 | ≤4000 | Δ |
|---|---:|---:|---:|
| Mean | 0.6209 | 0.6179 | −0.0031 |
| Median | 0.5534 | 0.5559 | +0.0025 |
| Pop. SD | 0.2188 | 0.2212 | +0.0023 |

RMSE slightly improves on average while **NLL worsens** — stage selection on val RMSE does not protect test NLL when more stages are allowed.

### Best iteration / cap hits

| Metric | ≤2000 view | ≤4000 view |
|---|---:|---:|
| Mean BestIter | 1535.9 | 2973.9 |
| Median BestIter | 1974.5 | 3998.0 |
| Exact ceiling hits | 9 @ 2000 | 10 @ 4000 |
| Near 4000 (≥3990) | — | 11 |
| Folds where 4000 view selects >2000 | **15** | — |

### Paired outcomes

| Outcome | Test NLL | Test RMSE |
|---|---:|---:|
| Cap-4000 wins | **2** | 12 |
| Ties | **5** | 5 |
| Cap-4000 losses | **13** | 3 |

Mean paired Δ NLL = **+0.168**. Largest harms: fold 20 (+1.12), fold 9 (+0.77), fold 10 (+0.36).

### Runtime

| Item | Value |
|---|---:|
| Wall time | **2425 s** (~40.4 min) |
| Exit | 0 |

## 7. Per-Fold Table

| Fold | BestIter≤2k | BestIter≤4k | Test NLL≤2k | Test NLL≤4k | Δ NLL | Test RMSE≤2k | Test RMSE≤4k |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1852 | 4000 | 0.3791 | 0.3912 | +0.0121 | 0.5265 | 0.5261 |
| 2 | 1990 | 4000 | 1.0088 | 1.2692 | +0.2604 | 0.4572 | 0.4472 |
| 3 | 232 | 232 | 1.5686 | 1.5686 | 0 | 0.4657 | 0.4657 |
| 4 | 2000 | 4000 | 2.2609 | 2.4846 | +0.2237 | 0.4067 | 0.4020 |
| 5 | 2000 | 4000 | 4.1084 | 4.1608 | +0.0524 | 0.6790 | 0.6806 |
| 6 | 1758 | 3896 | 0.5272 | 0.5415 | +0.0143 | 0.5200 | 0.5184 |
| 7 | 2000 | 4000 | −0.2434 | −0.3005 | −0.0571 | 0.2338 | 0.2301 |
| 8 | 289 | 289 | 1.1254 | 1.1254 | 0 | 1.0979 | 1.0979 |
| 9 | 2000 | 4000 | 0.8766 | 1.6494 | +0.7728 | 0.4799 | 0.4638 |
| 10 | 2000 | 4000 | 0.1211 | 0.4799 | +0.3589 | 0.4393 | 0.4233 |
| 11 | 169 | 169 | 2.2642 | 2.2642 | 0 | 1.0343 | 1.0343 |
| 12 | 1959 | 2474 | 3.0531 | 3.0503 | −0.0028 | 0.7848 | 0.7848 |
| 13 | 2000 | 3996 | 2.5677 | 2.5799 | +0.0122 | 0.7035 | 0.7039 |
| 14 | 410 | 410 | 0.7229 | 0.7229 | 0 | 0.7146 | 0.7146 |
| 15 | 1930 | 3981 | 3.7342 | 3.7422 | +0.0080 | 0.9426 | 0.9413 |
| 16 | 2000 | 3774 | 1.1139 | 1.1310 | +0.0171 | 0.6179 | 0.6150 |
| 17 | 2000 | 4000 | 0.6032 | 0.8773 | +0.2741 | 0.4025 | 0.3944 |
| 18 | 2000 | 4000 | 2.5955 | 2.8847 | +0.2892 | 0.5642 | 0.5691 |
| 19 | 257 | 257 | 1.3853 | 1.3853 | 0 | 0.5426 | 0.5426 |
| 20 | 1871 | 4000 | 10.9296 | 12.0500 | +1.1205 | 0.8056 | 0.8019 |

## 8. Comparison to a01 (Illustrative Only)

| Item | a01 (confounded) | a02 (controlled) |
|---|---:|---:|
| Mean Δ NLL (4000−2000) | +0.742 | **+0.168** |
| NLL wins / ties / losses for 4000 | 4 / 0 / 16 | **2 / 5 / 13** |
| Isolates cap headroom? | No | **Yes** |

Direction agrees (extra headroom harms mean NLL); a01 **overstated** the magnitude because of split coupling.

## 9. Interpretation

Does extra headroom itself help or harm under controlled stage selection?

- **Harms mean/median test NLL** when the selector may use stages beyond 2,000.
- Often selects deeper models (15/20 folds pick >2000 when allowed) that fit val RMSE slightly better but degrade test NLL.
- Five folds have identical best iters under both ceilings (early stopping inside 1–2000); those are exact NLL ties by construction.

Recommendation: keep **n_estimators=2000** as the Yacht natural protocol default; cite a02 for sensitivity; label a01 as confounded.

## 10. Artifacts (a02)

| File | Purpose |
|---|---|
| `controlled_yacht_cap_diagnostic.py` | Harness snapshot |
| `command.txt`, `config.json`, `environment.txt` | Repro metadata |
| `stdout.log`, `stderr.log`, `harness-wrapper.log` | Raw logs |
| `timing.json` | Wall 2425 s, exit 0 |
| `metrics.json` | Per-fold + aggregates |
| `split-indices.json` | Explicit outer/inner index lists + hashes |
| `posthoc-shared-trajectory-check.json` | Corrected shared-path verification |
| `manifest.json`, `checksums.txt` | Inventory |
| `source-status-before.txt`, `source-status-after.txt` | Upstream cleanliness |

Also written: `ngboost/src/controlled_yacht_cap_diagnostic.py` (canonical harness).

## 11. Compliance

- **a01 preserved**; confound documented.
- **One dataset / natural model only.**
- Both cap views from **identical 4,000-stage fits**.
- Split identities/hashes stored.
- **No STATUS / upstream / environment edits;** no other model or dataset.

---

**End of handoff 017F1-NG-D1. Cursor stops here.**
