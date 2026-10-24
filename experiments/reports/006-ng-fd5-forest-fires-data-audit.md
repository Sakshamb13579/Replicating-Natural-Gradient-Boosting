# 006-NG-FD5 Forest Fires Data Feasibility Audit

**Handoff:** `006-NG-FD5`  
**Dataset:** Forest Fires (UCI 162) only  
**Model:** none  
**Completed:** 2026-08-23  
**Manifest:** `ngboost/data/manifests/forest-fires-audit-v1.json`  
**UCI metadata:** `ngboost/data/raw/forest-fires/uci-162-metadata.json`

## 1. Verdict

**MODERATE** — Authoritative UCI Montesinho wildfire table (CC BY 4.0, Cortez & Morais 2007) with clean structure (517×12+target) and high value as an **NGBoost stress test** for zero inflation, skew, and Normal misspecification. **Not suitable as primary uncertainty/calibration evidence on raw burned area.** Donor-documented **`log1p(area)`** target is the defensible modeling scope; back-transformed Normal intervals are **asymmetric on hectares** and must not be reported as symmetric raw-scale bands.

## 2. Source, Licence, Citation, and Acquisition

| Field | Evidence |
|---|---|
| UCI ID | 162 |
| Dataset page | https://archive.ics.uci.edu/dataset/162/forest+fires |
| DOI | `10.24432/C5D88D` |
| Licence | CC BY 4.0 |
| Source paper | Cortez & Morais (2007), EPIA — *New Trends in Artificial Intelligence*, pp. 512–523 |
| Geography | Montesinho Natural Park, Portugal |
| Download URL | https://archive.ics.uci.edu/static/public/162/forest+fires.zip |
| Access date | 2026-08-23 |
| Download time (UTC) | 2026-08-23T15:04:48Z |
| Preserved metadata | `ngboost/data/raw/forest-fires/uci-162-metadata.json` |
| Metadata SHA-256 | `6d1ceaae537d9ae2f46a5084ec12704807a71d233bd5e9b928b181f82d56e5a1` |

## 3. Raw Files and Direct Data Checks

| File | SHA-256 | Size |
|---|---|---|
| `forest-fires.zip` | `abd40d6142b5e30fea73ff7292f91dcdd8b64bf3b3f36e20f96c37f5d14f06fb` | 8,932 bytes |
| `forestfires.csv` | `0d6586a1fa52f55bef48578aef14eb97273f1e9330e1a53423df497a77065253` | 25,478 bytes |
| `forestfires.names` | `476117404b73befcdb300a0128d9542b792fac8547a95061dfa8389a73d303bc` | 3,420 bytes |

**Format:** CSV with header; 517 rows; 12 features + `area` target.

| Check | Result |
|---|---|
| Missing / NaN / Inf | 0 |
| Exact duplicate rows | 4 |
| Months present | 12 (`jan`–`dec`) |
| Days present | 7 (`mon`–`sun`) |

## 4. Target Distribution: Raw vs log1p

### Raw `area` (ha)

| Statistic | Value |
|---|---:|
| Zeros | **247 / 517 (47.8%)** |
| Positive burns | 270 |
| Min / max | 0.0 / 1090.84 |
| Mean / median | 12.85 / 0.52 |
| Q1 / Q3 | 0.0 / 6.58 |
| Skewness | **12.81** |
| > 100 ha | 11 rows |

### Donor transform (authoritative)

`forestfires.names` Past Usage and Cortez & Morais (2007): model **`ln(area + 1)`**, then **inverse transform** (`exp(y)−1`) for reporting MAD/RMSE on raw ha.

### `log1p(area)` (all rows)

| Statistic | Value |
|---|---:|
| Skewness | **1.21** (reduced vs raw) |
| Note | Zeros map to 0.0 on log1p scale |

### Scientific assessment (no model run)

| Question | Answer |
|---|---|
| Normal NGBoost on **raw area**? | **Misspecified / not recommended** — mass at zero + heavy tail; NLL on raw ha is computable but not a defensible primary metric |
| **`log1p` target** defensible? | **Yes** for donor-aligned stress test; still approximate (Normal on log1p with point mass at 0) |
| Primary probabilistic metrics? | **NLL and interval coverage on log1p(area)** — not raw ha |
| Inverse-transformed intervals? | Back-transform of symmetric log-scale intervals yields **asymmetric** raw ha bounds — **do not** claim symmetric raw-scale predictive intervals |

### Back-transform point summaries (Normal on z = log1p(area))

For predictive Normal on the log1p scale with mean `mu_z` and scale `sigma_z`:

| Summary | Formula | Role |
|---|---|---|
| Raw-scale **median** | `expm1(mu_z)` | Median of the implied lognormal on positive support; **not** the raw-scale expected value |
| Raw-scale **mean** (lognormal) | `exp(mu_z + sigma_z^2/2) - 1` | Expected value **if** z were exactly lognormal with no zero inflation |
| Zero inflation | 247/517 exact zeros | Both summaries are **model-based approximations**; neither matches population mean/median of raw `area` |

Any optional **raw-scale RMSE/MAE** must **predeclare** which back-transformed point summary is used before comparing to observed hectares.

## 5. Features, Spatial Structure, and Preprocessing (draft)

| Feature group | Columns | Draft handling |
|---|---|---|
| Spatial grid | `X`, `Y` (1–9 / 2–9) | Numeric; 36 unique cells; up to 52 rows per cell |
| Calendar | `month`, `day` | Categorical — encode (one-hot or ordinal) for trees |
| Fire weather | `FFMC`, `DMC`, `DC`, `ISI` | Numeric |
| Weather | `temp`, `RH`, `wind`, `rain` | Numeric (`rain` often 0) |

**Spatial dependence:** 322 unique `(month, day, X, Y)` tuples; repeated park cells across dates. Optional **(X,Y) group holdout** (36 groups) for robustness; row-level splits may leak spatial cells.

**Duplicates:** 4 exact duplicate rows — disclose or drop at pilot freeze.

## 6. Draft Split Protocol (not frozen)

| Option | Protocol | Notes |
|---|---|---|
| **A — primary** | Row-level 90/10, `seed=1`, 5–20 reps on **log1p(area)**; shared across variants | Primary **NLL and coverage on log1p**; paper-feasibility scale; spatial leakage possible |
| **B — optional** | Hold out whole `(X,Y)` cells | 36 groups; different estimand |

No split indices generated.

## 7. Paper-Overlap and Bounded Prior-Application Findings

- **NGBoost paper:** not among ten Table 1–3 datasets.
- **Official repo:** no forest-fires loader.
- **Prior search (§4.5, 2026-08-23):** wildfire+NGBoost literature exists elsewhere; **no hit on this UCI Montesinho table** in bounded search.

**Limitation:** Dataset-specific; literature search non-exhaustive.

## 8. Compute, Ethics, and Stress-Test Role

| Dimension | Assessment |
|---|---|
| Storage / CPU | Trivial (~25 KB; n=517) |
| Ethics / framing | Environmental burned-area data; no PII; cite Cortez/Morais; **not** operational fire forecasting |
| Stress-test role | **Strong** for homoscedastic vs natural-gradient under misspecification |
| Primary evidence role | **Weak** — use Airfoil/Superconductivity for main uncertainty narrative |

## 9. Supported, Unsupported, and Unresolved Conclusions

### Supported

- UCI download with checksums; preserved metadata for dataset 162.
- Zero inflation, skew, and extremes quantified on raw and log1p scales.
- Donor log1p practice verified from `forestfires.names`.
- Raw vs transformed evaluation clearly separated.
- Paper/repo absence and bounded prior-search wording preserved.

### Unsupported

- Normal NGBoost on raw ha as **primary** probabilistic protocol (misspecified; not mathematically impossible).
- Symmetric raw-scale prediction intervals from back-transformed Normal log1p intervals.
- Raw-scale RMSE/MAE without predeclared median vs mean back-transform.
- Global proof of non-use by all NGBoost wildfire studies.

### Unresolved

- Exact categorical encoding at pilot freeze.
- Drop vs retain 4 duplicate rows.
- Row-level vs spatial-group split for proposal pilots.

## 10. Created or Modified Paths

| Path | Action |
|---|---|
| `ngboost/data/raw/forest-fires/forest-fires.zip` | Created |
| `ngboost/data/raw/forest-fires/forestfires.csv` | Created |
| `ngboost/data/raw/forest-fires/forestfires.names` | Created |
| `ngboost/data/raw/forest-fires/uci-162-metadata.json` | Created |
| `ngboost/data/manifests/forest-fires-audit-v1.json` | Created |
| `ngboost/reports/006-ng-fd5-forest-fires-data-audit.md` | Created |
| `comparison/feasibility-matrix.md` | Modified — Forest Fires row + status summary |

## 11. Recommendation for Lead Review

1. Accept **MODERATE** and `AUDIT_DRAFT` as **stress-test** candidate.
2. Freeze **`log1p(area)`** as pilot target; primary **NLL and coverage on log1p scale**.
3. Require explicit back-transform interval disclaimer and median-vs-mean disclosure for any raw ha point metric.
4. Complete Gate 2 with all five NGBoost dataset audits before pilot selection.

## Compliance

- **Write boundary:** only the four handoff-allowed locations changed.
- **No package installation, environment creation, or model execution.**
- **No other dataset audited.**

---

**End of handoff 006-NG-FD5. Cursor stops here.**

---

## Fix Review Response — 006F1-NG-FD5

**Parent:** `006-NG-FD5`  
**Fix handoff:** `006F1-NG-FD5`  
**Completed:** 2026-08-23

### Defect addressed

Review noted that for `z = log1p(area)` with Normal NGBoost, `expm1(mu_z)` is the **raw-scale median** implied by the transformed model, not its raw-scale expected value; the lognormal-style mean is `exp(mu_z + sigma_z^2/2) - 1`. Zero inflation makes both summaries approximate. Raw-area Normal NLL is computable but scientifically misspecified, not mathematically invalid.

### Changes made

| Location | Change |
|---|---|
| `forest-fires-audit-v1.json` | Added `metric_protocol` (primary NLL/coverage on log1p; optional raw RMSE/MAE with predeclared back-transform). Split `inverse_for_reporting` into median vs lognormal mean formulas. Replaced "invalid" with misspecification language for raw-area Normal modeling. |
| This report | Added back-transform table (median vs mean); clarified primary metrics on log1p scale; updated unsupported/recommendation wording. |
| `feasibility-matrix.md` | Forest Fires row: "raw Normal misspecified" replaces "raw Normal invalid". |

### Acceptance checklist

- [x] Mean versus median semantics are correct (`expm1(mu_z)` vs `exp(mu_z + sigma_z^2/2) - 1`).
- [x] Log-scale primary metrics and optional raw-scale point metrics are separated.
- [x] Raw Normal modeling is called misspecified, not mathematically impossible.
- [x] Zero-inflation limitation remains explicit.
- [x] Only allowed files changed; no execution or next task.

### Compliance

- **Write boundary:** manifest, this report, feasibility matrix Forest Fires row only.
- **No package installation, environment creation, model execution, or STATUS edit.**
- **Verified data statistics unchanged** (517 rows, 247 zeros, checksums preserved).

---

**End of fix handoff 006F1-NG-FD5. Cursor stops here.**
