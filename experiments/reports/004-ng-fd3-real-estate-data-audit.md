# 004-NG-FD3 Real Estate Valuation Data Feasibility Audit

**Handoff:** `004-NG-FD3`  
**Dataset:** Real Estate Valuation (UCI 477) only  
**Model:** none  
**Completed:** 2026-08-23  
**Manifest:** `ngboost/data/manifests/real-estate-valuation-audit-v1.json`  
**UCI metadata:** `ngboost/data/raw/real-estate-valuation/uci-477-metadata.json`

## 1. Verdict

**WEAK** — Authoritative UCI source (CC BY 4.0), clean single-sheet workbook, and clear probabilistic regression story (location/MRT/price heteroscedasticity), but **n=414** makes per-split NLL and coverage comparisons **high-variance** (~41 test rows per 90/10 holdout). Suitable as a **secondary** new-data listing and low-cost ablation check, not a primary uncertainty-evidence dataset without explicit small-sample safeguards.

## 2. Source, Licence, Citation, and Acquisition

| Field | Evidence |
|---|---|
| UCI ID | 477 |
| Dataset page | https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set |
| DOI | `10.24432/C5J30W` |
| Licence | CC BY 4.0 |
| Dataset citation (APA) | Yeh, I-Cheng (2018). Real Estate Valuation [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5J30W. |
| Introductory paper | Yeh & Hsu (2018), *Applied Soft Computing* — building real estate valuation models |
| Geography | Sindian Dist., New Taipei City, Taiwan (UCI abstract) |
| Download URL | https://archive.ics.uci.edu/static/public/477/real+estate+valuation+data+set.zip |
| Access date | 2026-08-23 |
| Download time (UTC) | 2026-08-23T14:57:12Z |
| Preserved metadata | `ngboost/data/raw/real-estate-valuation/uci-477-metadata.json` |
| Metadata SHA-256 | `a59f0b33b1b80933df518a4a69b140594ec57203abeb040cb2bf0b05fb81ea93` |
| Metadata retrieval (UTC) | 2026-08-23T14:57:36Z |

Acquisition used the UCI static zip route only.

## 3. Workbook Structure and Raw-File Checks

| File | SHA-256 | Size |
|---|---|---|
| `real-estate-valuation.zip` | `aa437bdac3ca23200258a0a58d251c0af90651aeaf7c7a07aaa714f0f9f25e2c` | 32,885 bytes |
| `Real estate valuation data set.xlsx` | `597d72fcc6c0539e6035a033ddb387db48fff3fb1f3c98fee31fe081c64a9059` | 32,717 bytes |

**Workbook (parsed with stdlib zip+xml; no package install):**

| Check | Result |
|---|---|
| Sheets | 1 (`sheet1` only) |
| Header row | Present |
| Data rows | 414 (matches UCI) |
| Columns | 8 |
| Missing / non-finite | 0 |
| Exact duplicate rows | 0 |
| Constant feature columns | 0 |

### Column map

| Column | Role | Notes |
|---|---|---|
| `No` | **ID** | 414 unique values (1–414) — **exclude from features** |
| `X1 transaction date` | Feature | 12 unique decimal-year periods (2012.667–2013.583) |
| `X2 house age` | Feature | 0–43.8 years |
| `X3 distance to the nearest MRT station` | Feature | 23.4–6488.0 m |
| `X4 number of convenience stores` | Feature | 0–10 |
| `X5 latitude` | Feature | 24.932–25.015° |
| `X6 longitude` | Feature | 121.474–121.566° |
| `Y house price of unit area` | **Target** | 10,000 NTD/Ping (1 Ping = 3.3 m²) |

UCI counts **6 features** (X1–X6) plus ID and target.

## 4. Target and Feature Summary

**Target `Y house price of unit area`:** min 7.6, max 117.5, mean 37.98, pop. stdev 13.59, skew **0.598** (right-skewed), 270 unique values.

## 5. Leakage, Identifiers, and Grouping

| Field | Finding | Risk / action |
|---|---|---|
| `No` | Unique transaction index | **Do not use as feature** |
| `X1 transaction date` | 12 monthly cohorts | Random row split mixes months; optional **temporal holdout by X1** |
| `(X5, X6)` lat/lon | 259 unique pairs; up to **13** transactions per pair | Random split can leak same-location rows; optional **location group holdout** (259 groups) |

UCI notes the original authors used a random 2/3–1/3 train/test partition; this audit does not adopt that split and proposes paper-aligned holdouts as draft only.

## 6. Paper-Overlap and Bounded Prior-Application Findings

- **NGBoost paper:** Real Estate not among the ten Table 1–3 datasets.
- **Official repo:** no real-estate loader in `regression_exp.py`.
- **Prior search (evidence pack §4.3, 2026-08-23):** `"Real Estate Valuation" NGBoost`, `"house price of unit area" NGBoost`, `Yeh Hsu NGBoost`, `Sindian NGBoost` — **no NGBoost hit** on this UCI table in bounded search.

**Limitation:** Non-exhaustive; cannot claim global non-use.

## 7. Small-Sample Reliability for Uncertainty Metrics

| Item | Implication |
|---|---|
| n = 414 | Smallest of the five NGBoost new-data candidates |
| ~41 test rows per 90/10 split | NLL SE over repetitions is estimable; **coverage/interval metrics remain noisy** |
| 20 repetitions | Helps aggregate means; does not increase per-split test support |
| Recommendation | Report mean NLL ± SE across splits; treat coverage as exploratory; prefer Airfoil/Superconductivity for primary uncertainty narrative |

## 8. Scientific Value for NGBoost

- Location, MRT distance, and neighborhood amenities plausibly induce **heteroscedastic price uncertainty** — aligns with probabilistic regression motivation.
- Very low compute cost enables quick three-way ablation (natural / ordinary / homoscedastic).
- Best framed as **secondary** evidence alongside larger datasets; method victory not required.

## 9. Preprocessing and Draft Split Proposal (not frozen)

### Preprocessing (draft)

1. Read sheet1 from xlsx; drop `No`.
2. Features: X1–X6; target: Y.
3. No mandatory scaling for tree NGBoost.

### Draft split options

| Option | Protocol | Notes |
|---|---|---|
| **A — primary, paper-aligned** | Row-level 90/10, `seed=1`, 20 reps; shared across variants | Matches NGBoost feasibility protocol; location/temporal leakage possible |
| **B — optional temporal** | Hold out whole `X1` month cohorts (12 groups) | Robustness; smaller effective train/test |
| **C — optional location** | Hold out `(X5, X6)` groups (259 groups) | Robustness; many singleton groups |

No split indices generated.

## 10. Compute, Storage, Ethics

| Dimension | Assessment |
|---|---|
| Storage | ~33 KB — trivial |
| RAM | ~0.03 MB float64 matrix — trivial |
| CPU | **Trivial** — full 20×3 protocol feasible quickly |
| Ethics | Aggregate property data with coordinates; one district; attribute UCI/Yeh; no personal identifiers named |

## 11. Supported, Unsupported, and Unresolved Conclusions

### Supported

- UCI download with checksums; preserved API metadata for dataset 477.
- Single-sheet structure; 414×6+ID+target; zero missingness.
- Identifier, temporal, and location leakage paths documented.
- Paper/repo absence; bounded prior-search wording preserved.
- Small-sample limits for uncertainty metrics stated explicitly.

### Unsupported

- Reliable per-split coverage conclusions at n≈41 test points.
- Global proof of non-use by NGBoost elsewhere.
- That random splits remove location or temporal dependence.

### Unresolved

- Whether to retain X1 under row-level splits vs temporal holdout.
- Primary vs secondary pilot role at Gate 2.
- Handling of repeated lat/lon groups in proposal prose.

## 12. Created or Modified Paths

| Path | Action |
|---|---|
| `ngboost/data/raw/real-estate-valuation/real-estate-valuation.zip` | Created |
| `ngboost/data/raw/real-estate-valuation/Real estate valuation data set.xlsx` | Created |
| `ngboost/data/raw/real-estate-valuation/uci-477-metadata.json` | Created |
| `ngboost/data/manifests/real-estate-valuation-audit-v1.json` | Created |
| `ngboost/reports/004-ng-fd3-real-estate-data-audit.md` | Created |
| `comparison/feasibility-matrix.md` | Modified — Real Estate row + status summary |

## 13. Recommendation for Lead Review

1. Accept **WEAK** verdict and `AUDIT_DRAFT`.
2. Keep in five-dataset proposal list with **small-n and leakage caveats**.
3. Do not promote to `PILOT_READY` as primary uncertainty dataset without explicit scope.
4. If piloted, authorize split option (A/B/C) and require aggregate NLL reporting with SE.

## Compliance

- **Write boundary:** only the four handoff-allowed locations changed.
- **No package installation, environment creation, or model execution.**
- **No other dataset audited.**

---

**End of handoff 004-NG-FD3. Cursor stops here.**
