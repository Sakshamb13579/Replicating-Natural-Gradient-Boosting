# 007-NG-CD1 Heteroscedastic Constructed Dataset — Generator Freeze

**Handoff:** `007-NG-CD1`  
**Dataset:** Constructed heteroscedastic regression v1 only  
**Model:** none (data generation and protocol freeze only)  
**Completed:** 2026-08-23  
**Manifest:** `ngboost/data/manifests/heteroscedastic-v1.json`  
**Generator:** `ngboost/src/generate_heteroscedastic_v1.py`

## 1. Verdict

**FROZEN** — Deterministic generator, paired three-regime dataset, and 60/20/20 split assignments are fixed with checksums. Pre-registered pilot hypotheses and metrics are documented without model execution. Eligible for Phase 5 constructed-data pilots under `AUDIT_FROZEN`.

## 2. Why This Is Constructed Evidence (Not Arbitrary Synthetic Rows)

NGBoost’s core claim is **input-dependent conditional uncertainty** (multiparameter boosting with natural gradients). This generator holds the mean function fixed and varies only the noise scale across three **paired** regimes on the same `(x1, x2, z)` draws:

| Regime | σ(x) | Role |
|---|---|---|
| **A** | `0.1 + 0.5·|x1|` | Primary heteroscedastic test aligned with shared plan §3.2 |
| **B** | `0.6` constant | Homoscedastic **control** — isolates whether models overfit input-dependent width when none is needed |
| **C** | `2·(0.1 + 0.5·|x1|)` | Amplified heteroscedastic variant; same functional dependence on `|x1|` as A |

Ground-truth `true_mu` and `true_sigma` are known for every row, so pilots can evaluate NLL, coverage, and width **against the generative law** without label noise ambiguity. Regime pairing ensures differences in metrics reflect σ-structure, not resampling noise.

## 3. Frozen Generator Specification

```text
x1, x2 ~ Uniform(-2, 2) independently
mu(x)  = sin(x1) + x2^2
z      ~ Normal(0, 1)

regime A: sigma(x) = 0.1 + 0.5 * abs(x1)
regime B: sigma(x) = 0.6
regime C: sigma(x) = 2 * (0.1 + 0.5 * abs(x1))

y = mu(x) + sigma(x) * z
```

| Parameter | Value |
|---|---|
| Base rows `n` | 4,000 |
| Regime rows | 12,000 (4,000 × 3) |
| Data RNG seed | `0` (`numpy.random.default_rng`) |
| Split RNG seed | `1` |
| RNG algorithm | PCG64 |
| Python (generation host) | 3.10.0 |
| NumPy (generation host) | 2.0.2 |

**Determinism:** Generator uses explicit `default_rng` instances only; no global `np.random` state.

## 4. Output Artifacts and Checksums

| Path | SHA-256 | Rows |
|---|---|---:|
| `ngboost/src/generate_heteroscedastic_v1.py` | `2ac6fe06d86ce368eb9e27287fa601814fcd494ffc8a3398fa9efd5b2302c126` | — |
| `ngboost/data/processed/heteroscedastic-v1/base-rows-v1.csv` | `7ba29dc8d50d08f82a53d5f399cb67dab9839d0daa742010760a6c08e5337cd9` | 4,000 |
| `ngboost/data/processed/heteroscedastic-v1/heteroscedastic-v1.csv` | `a39e58f617b5e595595fdd011af9284b40a5d2d452a427a3451047a50f6a181f` | 12,000 |
| `ngboost/data/processed/heteroscedastic-v1/generation-meta.json` | `fa6a000ca0c24329096f84a726c11062df64fe53e2dae63cccd7bc083a504138` | — |
| `ngboost/data/manifests/heteroscedastic-v1.json` | (see manifest at freeze time) | — |

### Column schema (`heteroscedastic-v1.csv`)

`row_id`, `regime`, `split`, `x1`, `x2`, `true_mu`, `true_sigma`, `noise_z`, `y`

Base companion file (`base-rows-v1.csv`): `row_id`, `x1`, `x2`, `noise_z`, `true_mu`, `split` — same underlying draws without regime expansion.

## 5. Split Freeze

| Level | Train | Validation | Test |
|---|---:|---:|---:|
| Base `row_id` | 2,400 | 800 | 800 |
| All regimes combined | 7,200 | 2,400 | 2,400 |
| Per regime (A/B/C) | 2,400 | 800 | 800 |

Assignment: single permutation of 4,000 base IDs via split seed `1`; **identical** `split` label for each `row_id` in regimes A, B, and C. Full `row_id` lists are stored in the manifest; pilots must not regenerate or alter splits.

## 6. Data-Only Sanity Checks (No Model Fit)

| Check | Result |
|---|---|
| All numeric fields finite | Pass |
| Unique base `row_id` | 4,000 |
| `x1` range | [−1.999, 1.998] ⊂ [−2, 2] |
| `x2` range | [−1.999, 2.000] ⊂ [−2, 2] |
| Paired `x1`, `x2`, `noise_z`, `split` across A/B/C | Pass (exact match by `row_id`) |
| Formula `y = true_mu + true_sigma * noise_z` | Pass (spot checks + construction) |

### Residual diagnostics (population sample, n=4,000 per regime)

| Regime | Residual mean | Normalized residual mean | Normalized residual SD | σ min | σ max | corr(\|x1\|, σ) |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.0074 | 0.0216 | 0.994 | 0.100 | 1.100 | **1.000** |
| B | 0.0129 | 0.0216 | 0.994 | 0.600 | 0.600 | ~0 |
| C | 0.0148 | 0.0216 | 0.994 | 0.200 | 2.199 | **1.000** |

Normalized residuals share the same mean across regimes because `(x1, x2, z)` are paired; SD ≈ 1 confirms unit-scale noise draw. Regime B correlation ≈ 0 confirms constant σ control.

### Regime A — binned `|x1|` vs `true_sigma` (quantile bins)

| Bin | Mean \|x1\| | Mean true σ |
|---:|---:|---:|
| 1 | 0.251 | 0.226 |
| 2 | 0.742 | 0.471 |
| 3 | 1.251 | 0.726 |
| 4 | 1.748 | 0.974 |

Monotonic widening confirms the intended heteroscedastic structure for pilot slice analysis.

## 7. Pre-Registered Pilot Protocol (Not Executed)

### Models (future handoff)

1. Natural-gradient multiparameter NGBoost (`Normal`)
2. Ordinary-gradient multiparameter NGBoost (`Normal`)
3. Homoscedastic NGBoost (`NormalFixedVar`)

### Metrics

| Tier | Metric |
|---|---|
| **Primary** | Test NLL (Normal predictive distribution) |
| **Secondary** | RMSE of location vs `y`; empirical coverage at 50 / 80 / 95%; mean interval width at same levels |
| **Diagnostic** | Predicted scale or width vs `true_sigma`; binned by `|x1|` quantiles |

### Hypotheses (scientifically bounded)

- **H1 (regime A):** Multiparameter natural/ordinary NGBoost beats homoscedastic on test NLL when σ depends on `|x1|`.
- **H2 (regime A):** Predicted width increases with `|x1|` in the same direction as `true_sigma`.
- **H3 (regime B):** Homoscedastic control — natural gradient **not required** to win; absence of input-dependent σ is the point.
- **H4 (regime C):** Input-dependent models should still track doubled σ(|x1|); effect size differs from A by construction.

**Non-requirement:** Natural gradient need not dominate every regime.

## 8. Created or Modified Paths

| Path | Action |
|---|---|
| `ngboost/src/generate_heteroscedastic_v1.py` | Created |
| `ngboost/data/processed/heteroscedastic-v1/base-rows-v1.csv` | Created |
| `ngboost/data/processed/heteroscedastic-v1/heteroscedastic-v1.csv` | Created |
| `ngboost/data/processed/heteroscedastic-v1/generation-meta.json` | Created |
| `ngboost/data/manifests/heteroscedastic-v1.json` | Created (`status: frozen`) |
| `ngboost/reports/007-ng-cd1-heteroscedastic-generator.md` | Created |
| `comparison/feasibility-matrix.md` | Modified — constructed row → `AUDIT_FROZEN` |

## 9. Acceptance Checklist

- [x] Generator deterministic and hashed
- [x] 4,000 base and 12,000 regime rows
- [x] Same base inputs/noise/split reused across A/B/C
- [x] Split counts 2,400/800/800 base; 7,200/2,400/2,400 all-regime totals
- [x] Sanity checks demonstrate ground truth without model fitting
- [x] Hypotheses and metrics predeclared and bounded
- [x] Only allowed paths changed; no model or follow-on task

## Compliance

- **Write boundary:** five handoff-allowed locations only.
- **No package installation, environment creation, or model execution.**
- **No changes to existing UCI desk audits or STATUS.**

---

**End of handoff 007-NG-CD1. Cursor stops here.**
