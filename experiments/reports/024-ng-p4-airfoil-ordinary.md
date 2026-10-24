# 024-NG-P4 — Airfoil ordinary-gradient NGBoost (five frozen folds)

## Verdict

Ordinary-gradient multiparameter NGBoost (`natural_gradient=False`, Normal/MLE) completed all **five** frozen Airfoil folds. Mean test **NLL 2.8319** (SE 0.1372); mean test **RMSE 5.0046**.

**Provisional two-way vs 023 natural** (identical folds): mean ΔNLL (ordinary − natural) = **+0.6138** (natural lower NLL on **5/5** folds). This is **not** a final three-way ablation verdict — homoscedastic still pending.

## Protocol

| Item | Value |
|------|-------|
| Handoff | 024-NG-P4 |
| Runner | `run_airfoil_ngboost_variant.py --variant ordinary` (**not edited**) |
| Only change vs 023 | `natural_gradient=False` |
| Cap / LR / Dist / Score / tree | 2000 / 0.01 / Normal / MLE / default |
| Folds / seeds / stage selector | Exact 022 freeze; stage `20000+f`, full `30000+f`; val RMSE |
| Env / packages | Unchanged |

## Standalone per-fold (ordinary)

| Fold | Best iter | Test NLL | Test RMSE | Cov50 | Cov80 | Cov95 | Cap hit | Time (s) |
|------|-----------|----------|-----------|-------|-------|-------|---------|----------|
| 0 | 2000 | 2.781104 | 5.189226 | 0.273 | 0.733 | 0.933 | True | 41.0 |
| 1 | 2000 | 2.570530 | 4.506893 | 0.347 | 0.660 | 0.953 | True | 42.5 |
| 2 | 2000 | 2.643631 | 5.025891 | 0.307 | 0.740 | 0.967 | True | 41.4 |
| 3 | 2000 | 3.351480 | 5.217049 | 0.280 | 0.647 | 0.967 | True | 43.1 |
| 4 | 2000 | 2.812578 | 5.084019 | 0.300 | 0.693 | 0.927 | True | 41.8 |

## Aggregates (ordinary, n=5)

| Metric | Mean | Median | Sample SD | SE |
|--------|------|--------|-----------|-----|
| Test NLL | 2.831865 | 2.781104 | 0.306870 | 0.137236 |
| Test RMSE | 5.004616 | 5.084019 | 0.288832 | 0.129169 |

Cap hits: **5**/5. Wall: **232.9 s**.

Coverage fold-mean: 50% 0.301; 80% 0.695; 95% 0.949.

## Provisional paired comparison vs 023 natural

Label: **two-way provisional** (same split hashes verified). Δ = ordinary − natural (positive ⇒ natural better).

| Fold | Nat NLL | Ord NLL | ΔNLL | Nat RMSE | Ord RMSE | ΔRMSE |
|------|---------|---------|------|----------|----------|-------|
| 0 | 2.223051 | 2.781104 | +0.558052 | 2.299805 | 5.189226 | +2.889422 |
| 1 | 2.174082 | 2.570530 | +0.396448 | 2.309458 | 4.506893 | +2.197435 |
| 2 | 2.144915 | 2.643631 | +0.498716 | 2.236127 | 5.025891 | +2.789764 |
| 3 | 2.188369 | 3.351480 | +1.163111 | 2.202265 | 5.217049 | +3.014784 |
| 4 | 2.359773 | 2.812578 | +0.452805 | 2.894497 | 5.084019 | +2.189522 |

| Aggregate Δ | Mean | SE |
|-------------|------|-----|
| ΔNLL | +0.613826 | 0.139867 |
| ΔRMSE | +2.616185 | 0.176219 |

NLL win counts: natural 5 · ordinary 0 · ties 0.

Reference 023 means: NLL 2.2180, RMSE 2.3884.

## Claim boundary

Do **not** treat this as the final natural vs ordinary vs homoscedastic ranking. Homoscedastic handoff still required on the same folds.

## Verification

- Split hashes match 022/023; predictions finite; scales > 0.
- Runner SHA unchanged; source-status before/after identical for this run; packages untouched.

## Artifacts

`artifacts/airfoil-self-noise/ngboost-ordinary/024-ng-p4-airfoil-ordinary-s1-a01/`

Includes `metrics.json`, `paired-comparison-to-023.json`, `predictions.jsonl`/`.csv`, and full protocol files.

## Stop

024-NG-P4 complete. No STATUS/runner/env edit. No homoscedastic or further task.
