# 003-NG-FD2 Superconductivity Data Feasibility Audit

**Handoff:** `003-NG-FD2`  
**Dataset:** Superconductivity (UCI 464) only  
**Model:** none  
**Completed:** 2026-08-23  
**Manifest:** `ngboost/data/manifests/superconductivity-audit-v1.json`  
**UCI metadata:** `ngboost/data/raw/superconductivity/uci-464-metadata.json`

## 1. Verdict

**MODERATE** — Authoritative UCI source with CC BY 4.0 licence; clean primary table (`train.csv`: 21263×81+Tc); absent from NGBoost paper/repo; strong scientific value as a higher-dimensional materials regression benchmark. Compute for a full paper-aligned 20-split × 3-variant pilot is materially heavier than Yacht/Airfoil and requires a **predeclared subset or fewer splits** for actual execution, though the **full dataset remains valid for proposal listing**.

## 2. Source, Licence, Citation, and Acquisition

| Field | Evidence |
|---|---|
| UCI ID | 464 |
| Canonical page | https://archive.ics.uci.edu/dataset/464/superconductivty+data |
| Common alias (same page) | https://archive.ics.uci.edu/dataset/464/superconductivity+data |
| DOI | `10.24432/C53P47` (verified in preserved UCI API metadata) |
| Licence | CC BY 4.0 |
| Dataset citation (APA) | Hamidieh, K. (2018). Superconductivty Data [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53P47. |
| Introductory work | Hamidieh (2018), *Computational Materials Science*; arXiv:1803.10260 |
| Underlying data | NIMS Superconductor Database — http://supercon.nims.go.jp/index_en.html (public) |
| Download URL | https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip |
| Access date | 2026-08-23 |
| Download time (UTC) | 2026-08-23T14:54:12Z |
| Preserved metadata | `ngboost/data/raw/superconductivity/uci-464-metadata.json` |
| Metadata SHA-256 | `fa66e7acc67f2065a7a9d53ba19542b90e279f85d8e80a1abe3795957b9a427d` |
| Metadata retrieval (UTC) | 2026-08-23T14:54:44Z |

**Slug note:** UCI’s canonical slug is misspelled `superconductivty+data` (also in dataset title). Both `superconductivty+data` and `superconductivity+data` URLs resolve; download uses the API `href` path.

Acquisition used the UCI static zip route only (not third-party mirrors).

## 3. Raw Files, Checksums, and Direct Data Checks

| File | SHA-256 | Size |
|---|---|---|
| `superconductivty-data.zip` | `87f4490d73390ff94ee01dbf0d7d32abc80b22f2c803d471765cfc46a9f6371e` | 8,300,005 bytes |
| `train.csv` | `4dfb6e3a1f6ffd969e5a5e42f093c4800d1e2a6c8b1e309f8fcd9f23d86952f3` | 23,859,780 bytes |
| `unique_m.csv` | `b68ae6b55ea8581eff8b1ffba073a899db7e2d2f7f3b781bb0802f643f51e5f7` | 4,291,603 bytes |

**Direct checks on `train.csv` (stdlib Python, no model fitting):**

| Check | Result |
|---|---|
| Rows | 21263 (matches UCI) |
| Columns | 82 (81 features + `critical_temp`) |
| Header | Present |
| Missing / NaN / Inf | 0 |
| Malformed rows | 0 |
| Constant feature columns | 0 |
| Exact duplicate full rows | 66 |
| Unique feature tuples (excl. target) | 15170 |
| Feature tuples with multiple Tc values | 2374 |

**Target `critical_temp` (K):** min 0.00021, max 185.0, mean 34.42, pop. stdev 34.25, skew **0.860** (right-skewed).

**Feature range samples:** `number_of_elements` 1–9; `mean_atomic_mass` 6.941–208.98; `wtd_std_Valence` 0–3.

## 4. File Relationship and Leakage Controls

UCI descriptive text (preserved in metadata): **`train.csv`** holds 81 features + critical temperature; **`unique_m.csv`** holds the same 21263 superconductors with chemical formula decomposed into elemental fractions, plus `critical_temp` and `material`.

| Aspect | Finding |
|---|---|
| Row counts | Both 21263 |
| Row-order Tc alignment | 21263/21263 exact matches by index |
| `train.csv` role | **Primary modeling table** — aggregated descriptors |
| `unique_m.csv` role | Parallel composition view; 86 element columns + Tc + material formula string |
| `material` column | 15542 unique formula strings — **identifier; do not use as feature** |

**Leakage / misalignment risks:**

1. **Do not** concatenate `unique_m` element columns with `train.csv` aggregated features without deduplication — double-counts chemical information.
2. **Do not** use `material` strings as model inputs.
3. **Do not** sort/join files independently; row index alignment verified on Tc but pilot should use **`train.csv` only** unless a separately authorized composition-based experiment is defined.
4. Duplicate aggregated features mapping to different Tc (2374 tuples) reflect distinct materials with similar statistics — not target leakage, but affects interpretability.

## 5. Paper-Overlap and Bounded Prior-Application Findings

### NGBoost paper (ICML 2020)

Ten datasets in Tables 1–3: Boston, Concrete, Energy, Kin8nm, Naval, Power, Protein, Wine, Yacht, Year MSD. **Superconductivity is not among them.**

### Official repository

`regression_exp.py` loaders exclude superconductivity. Grep under `.RnD/source/ngboost` for `superconduct`: **no matches**.

### Prior-application search (bounded, reused from evidence pack §4.2)

Queries: `"Superconductivity" NGBoost`, `superconductor "Natural Gradient Boosting"`, `Hamidieh NGBoost`, repo grep. **Finding:** other models applied to Tc prediction; **no documented NGBoost application on this UCI table** in the bounded 2026-08-23 search.

**Limitation:** Non-exhaustive — cannot claim global non-use.

## 6. Scientific Value for NGBoost

- **Higher-dimensional tabular regression** (81 features) tests scalability beyond Yacht/Airfoil while staying in materials science.
- **Probabilistic Tc prediction** supports NLL, intervals, and natural vs homoscedastic ablations on a substantive domain task.
- **Proposal role:** Strong candidate to **list** among five new UCI datasets; pilot scope should be **bounded** (see compute section).
- Method victory not required; honest uncertainty quantification on skewed Tc distribution is scientifically meaningful.

## 7. Preprocessing and Draft Split Proposal (not frozen)

### Preprocessing (draft)

1. Load `train.csv` with header; target = `critical_temp`.
2. No mandatory scaling for tree NGBoost.
3. Exclude `unique_m.csv` from default pilot pipeline.
4. Lead decision before freeze: retain or drop 66 exact duplicate rows; document 2374 feature-collision tuples.

### Draft split options

| Option | Protocol | Notes |
|---|---|---|
| **A — paper-aligned (full data)** | Row-level 90/10, `np.random.seed(1)`, 20 reps; shared across variants | Matches NGBoost script convention; **compute-heavy** at n=21263, p=81 |
| **A′ — scoped full data** | Same protocol with **5 reps** (Kin8nm-style) | Recommended default for pilot unless timing handoff authorizes 20 |
| **B — subset contingency** | Fixed random row subset (e.g. n=5000, seed documented) then Option A | For strict timeboxes; must be predeclared in pilot handoff |

No split indices generated in this audit.

## 8. Compute, Storage, and Ethics

| Dimension | Assessment |
|---|---|
| Storage | ~28 MB extracted; ~8.3 MB zip — low |
| RAM (train matrix, float64) | ~13 MB — trivial |
| CPU (qualitative) | **Moderate–high** for 2000-estimator NGBoost × 3 variants × 20 splits on full data; reference Yacht natural 20-split ~864 s at n=308, p=6 |
| Proposal listing | Full 21263×81 dataset appropriate to cite |
| Pilot execution | Predeclare fewer splits and/or subset in pilot handoff |
| Ethics | Public materials data; no personal identifiers; attribute UCI/NIMS/Hamidieh |

No model timing was performed in this audit.

## 9. Supported, Unsupported, and Unresolved Conclusions

### Supported

- Authoritative UCI download with recorded checksums.
- Preserved UCI API metadata JSON validates dataset 464.
- `train.csv` shape, cleanliness, and target summary as recorded.
- File roles and row-aligned relationship documented with leakage controls.
- Paper-absent and repo-absent; bounded prior search wording preserved.
- CC BY 4.0 licence from UCI metadata.

### Unsupported

- Global proof that NGBoost was never applied to this table.
- That full 20-split × 3-variant pilot completes within Airfoil/Yacht-class timeboxes without scoping.
- That merging both CSV representations is safe without a separate protocol.

### Unresolved

- Primary pilot scope: full data + 5 splits vs subset + 20 splits.
- Handling of duplicate rows and feature-collision tuples at freeze.
- Final selection among five NGBoost new-data candidates (Gate 2).

## 10. Created or Modified Paths

| Path | Action |
|---|---|
| `ngboost/data/raw/superconductivity/superconductivty-data.zip` | Created |
| `ngboost/data/raw/superconductivity/train.csv` | Created |
| `ngboost/data/raw/superconductivity/unique_m.csv` | Created |
| `ngboost/data/raw/superconductivity/uci-464-metadata.json` | Created |
| `ngboost/data/manifests/superconductivity-audit-v1.json` | Created |
| `ngboost/reports/003-ng-fd2-superconductivity-data-audit.md` | Created |
| `comparison/feasibility-matrix.md` | Modified — Superconductivity row + status summary |

## 11. Recommendation for Lead Review

1. Accept **MODERATE** verdict and `AUDIT_DRAFT` status.
2. For proposal text: cite full dataset dimensions and Hamidieh/NIMS provenance.
3. For pilot handoff: authorize **`train.csv` only**, split count or subset policy, and duplicate-row handling before freeze.
4. Do not promote to `AUDIT_FROZEN` until compute scope is explicit.

## Compliance

- **Write boundary:** only the four handoff-allowed locations changed.
- **No package installation, environment creation, or model execution.**
- **No other dataset audited.**

---

**End of handoff 003-NG-FD2. Cursor stops here.**
