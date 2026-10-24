# 029-NG-P5 — Heteroscedastic ordinary-gradient NGBoost (regimes A/B/C)

## Verdict

Ordinary-gradient multiparameter NGBoost ran stably on all three frozen heteroscedastic regimes under the **exact 028 protocol** (`natural_gradient=False` only). Finite, positive scales; 95% coverage 0.930–0.949. **Provisional** paired comparison vs reviewed natural (028): natural wins test NLL on **all three** regimes (mean ΔNLL ordinary−natural ≈ **+0.036**). Final three-way verdict waits for homoscedastic.

## Protocol

| Item | Value |
|------|-------|
| Handoff | 029-NG-P5 |
| Data | `heteroscedastic-v1` (007 freeze; same as 028) |
| Model | `NGBRegressor`, **natural_gradient=False**, Normal, MLE, default tree |
| Cap / LR / minibatch | 2000 / 0.01 / 1.0 |
| Stage selector | validation **NLL** (same as 028) |
| Features | `x1`, `x2` only |
| Seeds | stage `20000+regime_idx`, full `30000+regime_idx` |
| Threads | 1 |
| Env | `.RnD/work/ngboost/.venv` unchanged |

Only difference from 028: gradient flag.

## Per-regime results (ordinary)

| Regime | Best iter | Cap hit | Test NLL | Oracle NLL | Test RMSE | Scale MAE | Cov50 | Cov80 | Cov95 | Pearson ρ | Time (s) |
|--------|-----------|---------|----------|------------|-----------|-----------|-------|-------|-------|-----------|----------|
| A | 577 | No | 0.8349 | 0.7310 | 0.6956 | 0.0911 | 0.494 | 0.776 | 0.930 | 0.898 | 55.9 |
| B | 404 | No | 0.9433 | 0.8850 | 0.6082 | 0.0906 | 0.483 | 0.783 | 0.934 | **null** | 56.0 |
| C | 383 | No | 1.5019 | 1.4242 | 1.4080 | 0.1405 | 0.473 | 0.784 | 0.949 | 0.938 | 56.6 |

### Regime A bins (test)

Monotonic widening of pred scale / width95 with |x1| retained (bins 1→4: pred scale 0.253→0.954).

### Regime B

Constant true σ → Pearson/Spearman **null**. Mild |x1|-linked scale variation remains (bins ≈ 0.56–0.64).

## Provisional paired comparison vs 028 (natural)

| Regime | Ord NLL | Nat NLL | ΔNLL (O−N) | Ord RMSE | Nat RMSE | ΔRMSE | Ord ScaleMAE | Nat ScaleMAE | ΔMAE | Ord ρ | Nat ρ | ΔCov95 | Winner (NLL) |
|--------|---------|---------|------------|----------|----------|-------|--------------|--------------|------|-------|-------|--------|--------------|
| A | 0.8349 | 0.7891 | **+0.0458** | 0.6956 | 0.6815 | +0.0141 | 0.0911 | 0.0602 | +0.0309 | 0.898 | 0.969 | −0.0175 | natural |
| B | 0.9433 | 0.9150 | **+0.0283** | 0.6082 | 0.5973 | +0.0109 | 0.0906 | 0.0463 | +0.0443 | null | null | −0.0263 | natural |
| C | 1.5019 | 1.4679 | **+0.0340** | 1.4080 | 1.3540 | +0.0540 | 0.1405 | 0.1086 | +0.0319 | 0.938 | 0.975 | +0.0025 | natural |

| Aggregate | Ordinary | Natural | Δ (O−N) |
|-----------|----------|---------|---------|
| Mean test NLL | 1.0934 | 1.0573 | +0.0361 |
| Mean test RMSE | 0.9039 | 0.8776 | +0.0263 |
| Wall (s) | 178.5 | 183.8 | −5.4 |
| NLL wins (regimes) | 0 | **3** | — |

Natural also tracks true σ more tightly (lower scale MAE; higher Pearson on A/C). Coverage is close; ordinary slightly undercovers at 95% on A/B.

**Provisional only.** Do not treat as final superiority; homoscedastic still outstanding.

## Runtime

| | Seconds |
|--|---------|
| Wall | 178.5 |
| Sum regime totals | 168.5 |
| Shell elapsed | 182 |

## Verification

- CSV SHA `a39e58f…`; split row_ids verified; GT excluded from features.
- All preds finite; scales > 0; regime B correlations `null`.
- Source status before/after: repair-worktree `6720340…`; pre-existing `M examples/experiments/regression_exp.py` unchanged.
- Env/packages untouched.

## Claim boundary

Ordinary multiparameter evidence + provisional 028 pairing only. Final three-way waits for homoscedastic.

## Artifacts

`artifacts/heteroscedastic-v1/ngboost-ordinary/029-ng-p5-heteroscedastic-ordinary-s0-a01/`

Includes: `metrics.json`, `predictions.jsonl`, `predictions.csv`, `paired-comparison-to-028.json`, `manifest.json`, `checksums.txt`, logs, timing, source-status.

## Stop

029-NG-P5 complete. No STATUS edit. No natural rerun, homoscedastic, or other dataset.
