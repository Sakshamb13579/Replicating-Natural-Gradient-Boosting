# 030-NG-P5 — Heteroscedastic fixed-variance NGBoost + final three-way

## Verdict

Homoscedastic `NormalFixedVar` NGBoost (natural gradient + **leakage-free** held-out validation RMS scale) completed all three frozen regimes. Constant positive scales; Pearson/Spearman **null** as required.

**Final constructed three-way** (028 natural / 029 ordinary / 030 homoscedastic):

| Rank (mean test NLL) | Variant |
|----------------------|---------|
| 1 (best) | **natural** 1.0573 |
| 2 | ordinary 1.0934 |
| 3 | homoscedastic 1.2211 |

NLL regime wins: natural **2** (A, C) · ordinary 0 · homoscedastic **1** (B).  
Pre-registered hypotheses **H1–H4 all pass**. Natural does **not** win every regime (wins B would violate the control framing — and it does not).

## Protocol (homoscedastic)

| Item | Value |
|------|-------|
| Handoff | 030-NG-P5 |
| Dist | `NormalFixedVar`, natural_gradient=True |
| Cap / LR / seeds | same as 028/029 |
| Stage selection | per stage: σ = RMS(stage-model val residuals); pick argmin Normal NLL |
| Test scale | **retain** that held-out σ; full location fit on train+val; **no** re-estimate after val enters training |
| Correlations | null (constant scale within regime) |

## Standalone per-regime (homoscedastic)

| Regime | Best iter | Fixed σ | Test NLL | Oracle NLL | Test RMSE | Scale MAE | Cov50 | Cov80 | Cov95 | Time (s) |
|--------|-----------|---------|----------|------------|-----------|-----------|-------|-------|-------|----------|
| A | 285 | 0.7085 | 1.0357 | 0.7310 | 0.6806 | 0.2604 | 0.625 | 0.849 | 0.949 | 44.7 |
| B | 346 | 0.6301 | 0.9038 | 0.8850 | 0.5956 | 0.0301 | 0.529 | 0.829 | 0.965 | 43.8 |
| C | 189 | 1.4094 | 1.7238 | 1.4242 | 1.3544 | 0.5192 | 0.628 | 0.841 | 0.948 | 44.7 |

Wall ≈ 141 s. Cap hits: 0.

## Final three-way — test NLL (lower better)

| Regime | Natural | Ordinary | Homoscedastic | Winner |
|--------|---------|----------|---------------|--------|
| A | **0.7891** | 0.8349 | 1.0357 | natural |
| B | 0.9150 | 0.9433 | **0.9038** | homoscedastic |
| C | **1.4679** | 1.5019 | 1.7238 | natural |

ΔNLL (homo − natural): A **+0.247**, B **−0.011**, C **+0.256**.

## Final three-way — test RMSE

| Regime | Natural | Ordinary | Homoscedastic | Winner |
|--------|---------|----------|---------------|--------|
| A | 0.6815 | 0.6956 | **0.6806** | homoscedastic |
| B | 0.5973 | 0.6082 | **0.5956** | homoscedastic |
| C | **1.3540** | 1.4080 | 1.3544 | natural |

RMSE wins: natural 1 · ordinary 0 · homoscedastic 2 (location-only model competitive on point error).

## Scale recovery

| Regime | Nat ScaleMAE | Ord ScaleMAE | Homo ScaleMAE | Nat ρ | Ord ρ | Homo ρ |
|--------|--------------|--------------|---------------|-------|-------|--------|
| A | 0.060 | 0.091 | 0.260 | 0.969 | 0.898 | null |
| B | 0.046 | 0.091 | **0.030** | null | null | null |
| C | 0.109 | 0.140 | 0.519 | 0.975 | 0.938 | null |

Input-dependent models track true σ on A/C; fixed variance is closest to constant σ=0.6 on B.

## Pre-registered hypothesis outcomes

| ID | Result | Summary |
|----|--------|---------|
| **H1** (A: multiparameter beats fixed-var NLL) | **PASS** | Nat 0.789 & Ord 0.835 < Homo 1.036 |
| **H2** (A: width↑ with \|x1\|) | **PASS** | Nat/Ord bin widths monotonic; homo width constant |
| **H3** (B: control; fixed-var need not lose) | **PASS** | Homo **wins** NLL on B (0.904 vs 0.915 / 0.943) |
| **H4** (C: input-dependent still tracks) | **PASS** | Nat/Ord Pearson ≈0.97/0.94; both beat homo NLL |
| Non-req: natural need not win every regime | **satisfied** | Natural wins A+C only; homo wins B |

## Runtime

| Variant | Wall (s) |
|---------|----------|
| natural (028) | 183.8 |
| ordinary (029) | 178.5 |
| homoscedastic (030) | 140.8 |

## Verification

- CSV SHA `a39e58f…`; splits verified; GT excluded from features.
- Constant positive scales; correlations null; sigma not re-estimated after full fit.
- Env/packages untouched; source status before/after recorded.

## Claim boundary

Final **constructed heteroscedastic** three-way on frozen regimes A/B/C only. Not extrapolated to Airfoil or other NGBoost paper tables.

## Artifacts

`artifacts/heteroscedastic-v1/ngboost-homoscedastic/030-ng-p5-heteroscedastic-homoscedastic-s0-a01/`

Includes `metrics.json`, `three-way-comparison.json`, predictions, `manifest.json`, `checksums.txt`, protocol logs.

## Stop

030-NG-P5 complete. No STATUS edit. No other model/dataset.
