# 005-NG-FD4 Parkinsons Telemonitoring Data Feasibility Audit

**Handoff:** `005-NG-FD4` (fix: `005F1-NG-FD4`)  
**Dataset:** Parkinsons Telemonitoring (UCI 189) only  
**Model:** none  
**Completed:** 2026-08-23  
**Manifest:** `ngboost/data/manifests/parkinsons-telemonitoring-audit-v1.json`  
**UCI metadata:** `ngboost/data/raw/parkinsons-telemonitoring/uci-189-metadata.json`

## 1. Verdict

**MODERATE** — Authoritative UCI clinical voice dataset with CC BY 4.0 licence, clean repeated-measure structure (5875 recordings, **42 patient clusters**), and a well-justified **`motor_UPDRS`** target. **Patient-disjoint, cluster-aware evaluation is mandatory.** Recordings are **not** independent test units. Excluding `total_UPDRS` and `subject#` from predictors controls major leakage. Usable for NGBoost uncertainty evaluation with explicit clinical, clustering, and small-cohort caveats.

## 2. Source, Licence, Citation, and Acquisition

| Field | Evidence |
|---|---|
| UCI ID | 189 |
| Dataset page | https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring |
| DOI | `10.24432/C5ZS3N` |
| Licence | CC BY 4.0 |
| Introductory paper | Tsanas, Little, McSharry & Ramig (2009), *IEEE Trans. Biomedical Engineering* |
| Citation (UCI APA) | Tsanas, A., Little, M. A., McSharry, P. E., & Ramig, L. O. (2009). Parkinsons Telemonitoring [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5ZS3N. |
| Download URL | https://archive.ics.uci.edu/static/public/189/parkinsons+telemonitoring.zip |
| Access date | 2026-08-23 |
| Download time (UTC) | 2026-08-23T15:00:22Z |
| Preserved metadata | `ngboost/data/raw/parkinsons-telemonitoring/uci-189-metadata.json` |
| Metadata SHA-256 | `9e62ea97c20b358a33dc69740e25e0ba7161b7025fcc56a39adfacee2fa2b39d` |

## 3. Raw Files and Direct Data Checks

| File | SHA-256 | Size |
|---|---|---|
| `parkinsons-telemonitoring.zip` | `2f82bb0ef96fa7d8d7edf4d97b89173e07c2cca7723440a64bc38918a83345df` | 292,221 bytes |
| `parkinsons_updrs.data` | `f2c7d5025dec4e92e7feae367a5f7ccf58789a10ac6b54bdf15976c599f9dd39` | 911,261 bytes |
| `parkinsons_updrs.names` | `744dc2835eab5c0fc744c638e5e6b05e656788f60f8aa2aba9d48e21c629f7c2` | 4,423 bytes |

**Format:** comma-separated ASCII; **header row present** in current UCI file; 5875 data rows × 22 columns (matches UCI instance count).

| Check | Result |
|---|---|
| Missing / NaN / Inf | 0 |
| Duplicate full rows | 0 |
| Subjects (`subject#`) | 42 (IDs 1–42) |
| Recordings per subject | min 101, median 141, mean 140, max 168 |
| `test_time` range (days since recruitment) | 0 – 209.5 |

## 4. Column Roles and Leakage

| Column | Role | Pilot use |
|---|---|---|
| `subject#` | **Patient ID** | **Exclude from features**; defines split groups |
| `age`, `sex` | Demographics | Allowed predictors |
| `test_time` | Days since recruitment | Allowed predictor when subject stays in one split |
| `motor_UPDRS` | **Primary target** | Predict from voice + demographics |
| `total_UPDRS` | Alternate outcome | **Exclude as predictor** (r=0.947 with motor) |
| 16 voice measures (Jitter…PPE) | Biomarkers | Allowed predictors |

**19 modeling features** (UCI feature count): age, sex, test_time, and 16 voice columns.

### Target choice: `motor_UPDRS`

Voice features were designed for motor speech impairment tracking. **Motor UPDRS** is the direct motor severity scale; **total UPDRS** subsumes non-motor components and is nearly collinear with motor (Pearson **r = 0.947**). `total_UPDRS` remains a documented alternate outcome only.

### Correlations (informative, no model fit)

| Pair | r |
|---|---:|
| motor_UPDRS vs total_UPDRS | 0.947 |
| motor_UPDRS vs age | 0.274 |
| motor_UPDRS vs test_time | 0.068 |

## 5. Target Summaries

| Target | Min | Max | Mean | Skew | Unique values |
|---|---:|---:|---:|---:|---:|
| motor_UPDRS | 5.04 | 39.51 | 21.30 | 0.075 | 1080 |
| total_UPDRS | 7.00 | 54.99 | 29.02 | 0.271 | 1129 |

## 6. Draft Cluster-Aware Patient-Disjoint Evaluation (not frozen)

**Mandatory:** no `subject#` may appear in both train and test within an evaluation fold. **5875 recordings are clustered repeated measures** — metrics must distinguish recording-level scores from **subject-macro** summaries.

### Draft split / CV protocol

| Item | Draft proposal |
|---|---|
| Primary protocol | **5-fold subject-group cross-validation** on 42 subjects (~**8–9 held-out subjects** per outer fold; ~1100–1200 held-out recordings per fold) |
| Leakage control | Whole subjects held out; optional subject-disjoint inner validation on training subjects only |
| Paired comparison | Identical subject-fold assignments reused across natural / ordinary / homoscedastic variants |
| Repetitions | Five outer folds (each subject held out once) — **not** row-level 90/10 paper repetitions |
| Indices | **Not generated** in this audit |

### Metric layers (draft)

| Layer | Definition | Role |
|---|---|---|
| **Recording-level** | NLL / RMSE / coverage computed over all held-out recordings in a fold | Standard probabilistic scores on all rows |
| **Subject-macro** | Per-subject mean NLL (or RMSE/coverage) **then** averaged across held-out subjects in the fold | Primary generalization summary — **~8–9 units per fold** |
| **Fold aggregation** | Mean ± SD (or subject-level bootstrap CI) of subject-macro metrics across outer folds | Cluster-aware uncertainty — fold-to-fold variation |

### Calibration and coverage claims

- Recording-level empirical coverage from ~1100 rows in a fold is **not** equivalent to ~1100 independent patients.
- Subject-macro coverage from **~8–9 patients** is **exploratory only** and **cannot** establish clinical calibration or population interval honesty.
- Variation across outer folds or subject-level bootstrap resampling may summarize uncertainty without over-claiming inference.

**Small-cohort note:** The operative test count is **patients per fold**, not recording count.

## 7. Paper-Overlap and Bounded Prior-Application Findings

- **NGBoost paper:** not among ten Table 1–3 datasets.
- **Official repo:** no Parkinsons loader in `regression_exp.py`.
- **Prior search (§4.4, 2026-08-23):** UPDRS work with other boosters; **no NGBoost hit** on this UCI table in bounded search.

**Limitation:** Non-exhaustive; not global proof of non-use.

## 8. Compute, Storage, Ethics

| Dimension | Assessment |
|---|---|
| Storage | ~0.9 MB data — trivial |
| RAM | ~0.7 MB float64 feature matrix — trivial |
| CPU | Trivial for NGBoost at n=5875, p=19 |
| Ethics | Clinical PD telemonitoring data; numeric subject IDs only; aggregate research use; cite Tsanas et al.; **no re-identification**; disclose 42-patient cohort and clinical context in proposal |

## 9. Scientific Value for NGBoost Uncertainty

- Repeated-measure clinical regression tests probabilistic predictions on **real-world biomarker noise**.
- Motor severity heteroscedasticity and measurement uncertainty are clinically meaningful.
- **Cluster-aware subject-disjoint evaluation** is scientifically defensible though it **differs from Yacht row splits** — must be stated clearly.
- Subject-macro metrics and fold-to-fold variation are the appropriate uncertainty summaries; recording counts alone must not inflate claims.
- Method victory not required.

## 10. Supported, Unsupported, and Unresolved Conclusions

### Supported

- UCI download with checksums; preserved API metadata for dataset 189.
- 5875 recordings, 42 subjects, zero missingness, quantified repeated-measure structure.
- `motor_UPDRS` target justified; `total_UPDRS` and `subject#` leakage paths documented.
- Patient-disjoint **5-fold subject-group CV** drafted with recording-level vs subject-macro metric distinction.
- Cluster-aware uncertainty framing (outer-fold variation; exploratory coverage only).
- Paper/repo absence and bounded prior-search wording preserved.

### Unsupported

- Treating ~550–1100 held-out **recordings** as independent patients.
- Row-level 90/10 paper protocol without patient leakage.
- Clinical calibration or population coverage claims from ~8–9 held-out subjects per fold.
- Strong external validity from 42-subject cohort alone.
- Global proof that NGBoost never used this table.

### Unresolved

- Include vs withhold `test_time` as predictor under subject holdout.
- Exact inner-validation policy within training subjects at pilot freeze.
- Gate 2 pilot priority vs Airfoil/Superconductivity.

## 11. Created or Modified Paths

| Path | Action |
|---|---|
| `ngboost/data/raw/parkinsons-telemonitoring/parkinsons-telemonitoring.zip` | Created |
| `ngboost/data/raw/parkinsons-telemonitoring/parkinsons_updrs.data` | Created |
| `ngboost/data/raw/parkinsons-telemonitoring/parkinsons_updrs.names` | Created |
| `ngboost/data/raw/parkinsons-telemonitoring/uci-189-metadata.json` | Created |
| `ngboost/data/manifests/parkinsons-telemonitoring-audit-v1.json` | Created; updated (005F1-NG-FD4) |
| `ngboost/reports/005-ng-fd4-parkinsons-data-audit.md` | Created; updated (005F1-NG-FD4) |
| `comparison/feasibility-matrix.md` | Modified — Parkinsons row (005F1-NG-FD4) |

## 12. Recommendation for Lead Review

1. Accept **MODERATE** and `AUDIT_DRAFT`.
2. Lock **`motor_UPDRS`** primary target and forbid `total_UPDRS` as predictor at pilot handoff.
3. Require **5-fold subject-group CV manifest** with recording-level and subject-macro metric reporting before training authorization.
4. Disclose clinical/small-cohort/clustering limits in proposal prose.

## Compliance

- **Write boundary:** only the four handoff-allowed locations changed.
- **No package installation, environment creation, or model execution.**
- **No other dataset audited.**

---

**End of handoff 005-NG-FD4. Fix pass 005F1-NG-FD4 appended below.**

## Fix Review Response (005F1-NG-FD4)

**Parent checkpoint:** `005-NG-FD4`  
**Completed:** 2026-08-23

### Changes made

| Item | Fix |
|---|---|
| Cluster vs independence | Removed implication that ~550 held-out recordings equate to hundreds of independent test units; recordings are repeated measures within ~8–9 patients per fold |
| Draft split protocol | Replaced 80/10/10 example with **5-fold subject-group CV** (~8–9 held-out subjects per outer fold; subject-disjoint inner val optional) |
| Metric layers | Added **recording-level** vs **subject-macro** distinction and fold-aggregated uncertainty |
| Calibration bounds | Stated that empirical coverage from few patients is **exploratory** and cannot establish clinical calibration |
| Manifest | Added `evaluation_protocol`; updated `proposed_split`; added `fix_handoff_id`, `updated_at` |
| Feasibility matrix | Parkinsons notes updated for cluster-aware evaluation |
| Unchanged | Verified hashes, row/subject counts, target summaries, correlations |

### MODERATE verdict

**Remains justified.** The fix tightens evaluation semantics without changing data-quality findings. Parkinsons remains usable with mandatory patient-disjoint, cluster-aware reporting.

### Compliance

- Only the three allowed files changed.
- No split indices generated; no package install or model run.

**End of fix handoff 005F1-NG-FD4. Cursor stops here.**
