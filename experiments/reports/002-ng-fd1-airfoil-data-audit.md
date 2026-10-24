# 002-NG-FD1 Airfoil Self-Noise Data Feasibility Audit

**Handoff:** `002-NG-FD1` (fix: `002F1-NG-FD1`)  
**Dataset:** Airfoil Self-Noise (UCI 291) only  
**Model:** none  
**Completed:** 2026-08-23 (updated 2026-08-23 fix pass)  
**Manifest:** `ngboost/data/manifests/airfoil-self-noise-audit-v1.json`  
**UCI metadata:** `ngboost/data/raw/airfoil-self-noise/uci-291-metadata.json`

## 1. Verdict

**STRONG** — Authoritative UCI source verified; CC BY 4.0 licence confirmed; clean tabular regression table (1503×5+target); absent from the selected NGBoost paper’s ten datasets and from the official repo loaders; low preparation cost and clear relevance to probabilistic regression with interpretable physics features.

## 2. Source, Licence, Citation, and Acquisition

| Field | Evidence |
|---|---|
| UCI ID | 291 |
| Dataset page | https://archive.ics.uci.edu/dataset/291/airfoil+self+noise |
| DOI | `10.24432/C5VW2C` (confirmed in preserved UCI API metadata) |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Citation (APA) | Brooks, T., Pope, D., & Marcolini, M. (1989). Airfoil Self-Noise [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5VW2C. |
| Underlying source | NASA aerodynamic/acoustic wind-tunnel experiments; Brooks, Pope & Marcolini, NASA RP-1218 (1989) per UCI donation metadata and shared evidence pack §4.1 |
| Download URL | https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip |
| Access date | 2026-08-23 |
| Download time (UTC) | 2026-08-23T14:45:21Z |
| Preserved metadata | `ngboost/data/raw/airfoil-self-noise/uci-291-metadata.json` |
| Metadata retrieval URL | https://archive.ics.uci.edu/api/trpc/donated_datasets.findById?batch=1&input=%7B%220%22%3A%7B%22json%22%3A291%7D%7D |
| Metadata retrieval time (UTC) | 2026-08-23T14:47:54Z |
| Metadata SHA-256 | `913da48b6ea2680afed32893844ef36dc71c08772ac665b3de695e717c0e71a1` |

The saved JSON validates `dataset.id == 291` and records DOI, licence, instance/feature counts, variable table, and citation strings from the UCI API (not HTML).

## 3. Raw-File Identity, Checksum, Format, and Direct Checks

| File | SHA-256 | Size |
|---|---|---|
| `airfoil-self-noise.zip` | `5c7767ba53ad827d3f48ba1eb9434117f4892df8f10bc4c99e118a9e8a7ae07c` | 60,126 bytes |
| `airfoil_self_noise.dat` | `74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3` | 59,984 bytes |

**Format:** tab-separated, no header, six numeric columns per row.

**Direct checks (stdlib Python 3, no package install):**

| Check | Result |
|---|---|
| Row count | 1503 (matches UCI `# Instances`) |
| Columns per row | 6 (consistent) |
| Missing / NaN / Inf | 0 |
| Duplicate rows | 0 |
| Delimiter | Tab only |

Stored under: `ngboost/data/raw/airfoil-self-noise/`

## 4. Target and Feature Summary

**Target:** `scaled-sound-pressure` — scaled sound pressure level (dB).

| Column | UCI name | Min | Max | Mean | Pop. stdev |
|---|---|---:|---:|---:|---:|
| 1 | frequency (Hz) | 200 | 20,000 | 2,886.4 | 3,151.5 |
| 2 | attack-angle (deg) | 0.0 | 22.2 | 6.78 | 5.92 |
| 3 | chord-length (m) | 0.0254 | 0.3048 | 0.137 | 0.094 |
| 4 | free-stream-velocity (m/s) | 31.7 | 71.3 | 50.9 | 15.6 |
| 5 | suction-side-displacement-thickness (m) | 0.000401 | 0.0584 | 0.0111 | 0.0131 |
| 6 (target) | scaled-sound-pressure (dB) | 103.38 | 140.99 | 124.84 | 6.90 |

**Target distribution:** median 125.72; Q1 120.19; Q3 130.00; population skewness **−0.419** (mild left skew, not extreme).

**Metadata note:** Preserved UCI variable table lists attack-angle type as **Binary**, but the raw file contains **27 distinct** angle values from 0.0° to 22.2°. This discrepancy is preserved explicitly; treat attack-angle as numeric experimental setting, not a binary feature.

## 5. Paper-Overlap and Bounded Prior-Application Findings

### Selected NGBoost paper (ICML 2020)

Per `shared/repo/docs/ngboost-proposal-evidence.md` §2, Tables 1–3 use ten datasets: Boston, Concrete, Energy, Kin8nm, Naval, Power, Protein, Wine, Yacht, Year MSD. **Airfoil Self-Noise is not among them.**

### Official repository (read-only)

`regression_exp.py` loaders: housing, concrete, wine, kin8nm, naval, power, energy, protein, yacht, msd. **No airfoil loader or data path.** Repo grep for `airfoil` under `.RnD/source/ngboost`: no matches.

### Prior-application search (bounded, reused)

From evidence pack §4.1 (2026-08-23): queries `"Airfoil Self-Noise" NGBoost`, `airfoil "Natural Gradient Boosting"`, and repo grep. **Finding:** other models (XGBoost/RF) appear on airfoil noise tasks; **no NGBoost application on this UCI table** found in the documented search.

**Limitation:** Search is **non-exhaustive**. We can claim paper-absence and no hit in the bounded search; we **cannot** claim global proof of non-use.

## 6. Repeated Condition Groups and Draft Split Protocol

### Condition-group structure (quantified from raw `.dat`)

Each row is one frequency sample. Grouping by the four non-frequency inputs `(attack-angle, chord-length, free-stream-velocity, suction-side-displacement-thickness)`:

| Metric | Value |
|---|---:|
| Distinct condition groups | 106 |
| Rows per group (min / median / mean / max) | 8 / 14 / 14.18 / 19 |
| Group size distribution | 8→3, 9→2, 10→2, 11→7, 12→4, 13→15, 14→23, 15→15, 16→25, 17→7, 18→2, 19→1 |
| Total rows (check) | 1503 |

Frequency is the within-condition sweep variable: each group holds 8–19 frequency measurements at fixed aerodynamic settings.

**Condition-group holdout is technically possible** (106 groups). It would test generalization to unseen wind-tunnel settings rather than unseen frequencies within seen settings.

### Preprocessing (draft)

1. Read `airfoil_self_noise.dat` as tab-separated, headerless.
2. Assign UCI variable names; use last column as target.
3. No mandatory scaling for tree-based NGBoost.
4. No imputation required (zero missing values observed).

### Draft split options (not frozen — no indices written)

| Option | Protocol | Scientific question | Leakage note |
|---|---|---|---|
| **A — primary, paper-aligned** | Row-level random **90/10** × 20 reps, `np.random.seed(1)`; same draws for natural / ordinary / homoscedastic | Matches official NGBoost regression feasibility protocol (Yacht/Kin8nm path) | Frequency points from the same condition may appear in both train and test → moderate within-condition leakage |
| **B — optional grouped robustness** | Hold out whole condition groups (e.g. 80/20 or 90/10 by group, fixed seed) | Generalization to unseen aerodynamic settings | Lower condition leakage; only 106 test units; different estimand than Option A |

Neither option is frozen. Lead review should pick primary vs optional robustness before pilot handoff.

## 7. Compute, Storage, Ethics, and Leakage

| Dimension | Assessment |
|---|---|
| Storage | ~60 KB raw; negligible |
| CPU | Trivial for NGBoost at n=1503, p=5; suitable for local pilot and proposal scope |
| Ethics / privacy | Public experimental NASA data; no personal identifiers |
| Leakage | 106 repeated condition groups quantified; row-level split (Option A) carries within-condition frequency leakage; optional group holdout (Option B) changes scientific question |
| Licence | CC BY 4.0 — compatible with proposal use with attribution |

## 8. Scientific Value for NGBoost Uncertainty Evaluation

- **Task fit:** Continuous regression with physically interpretable inputs and smooth acoustic response — suitable for Normal-distribution NGBoost and NLL/coverage analysis.
- **Uncertainty angle:** Aerodynamic noise may vary with operating conditions (angle, velocity, thickness); heteroscedastic predictive intervals are scientifically meaningful even if not guaranteed.
- **Proposal role:** Strong **existing-data pilot** candidate alongside a stress-test dataset (e.g. Forest Fires); complements constructed heteroscedastic data without heavy preprocessing.
- **Not required:** Method victory; negative or mixed ablation ordering remains useful if protocol is fair.

## 9. Supported, Unsupported, and Unresolved Conclusions

### Supported

- Authoritative UCI download with recorded SHA-256.
- Preserved UCI API metadata JSON for dataset 291 with recorded checksum and retrieval URL.
- 1503 rows, 5 features, target in dB; zero missing/non-finite/duplicate rows in raw file (unchanged from parent audit).
- 106 condition groups quantified with group-size distribution.
- Absent from NGBoost paper’s ten datasets and official repo loaders.
- CC BY 4.0 licence from preserved UCI metadata.
- Low compute/storage burden; draft row-level split remains paper-aligned; grouped robustness option documented.

### Unsupported

- That NGBoost has never been applied to this dataset globally.
- That a specific pilot will show natural-gradient wins (no model was run).
- That row-level splits (Option A) eliminate condition-level dependence.
- That grouped holdout (Option B) is equivalent to the paper’s Yacht/Kin8nm protocol without changing the estimand.

### Unresolved

- UCI “Binary” typing for attack-angle vs 27 observed levels (discrepancy preserved).
- Which split option is primary for Airfoil pilot (row-level vs condition-group robustness).
- Final pilot selection among the five NGBoost new-data candidates (Gate 2).

## 10. Created or Modified Paths

| Path | Action |
|---|---|
| `ngboost/data/raw/airfoil-self-noise/airfoil-self-noise.zip` | Created (002-NG-FD1) |
| `ngboost/data/raw/airfoil-self-noise/airfoil_self_noise.dat` | Created (002-NG-FD1) |
| `ngboost/data/manifests/airfoil-self-noise-audit-v1.json` | Created; updated (002F1-NG-FD1) |
| `ngboost/reports/002-ng-fd1-airfoil-data-audit.md` | Created; updated (002F1-NG-FD1) |
| `comparison/feasibility-matrix.md` | Modified — Airfoil row; header status (002F1-NG-FD1) |
| `ngboost/data/raw/airfoil-self-noise/uci-291-metadata.json` | Created (002F1-NG-FD1) |

## 11. Recommendation for Lead Review

1. Accept **STRONG** desk-audit verdict and `AUDIT_DRAFT` status pending OpenCode verification of checksums and UCI metadata.
2. Promote to `AUDIT_FROZEN` only after lead picks primary split option (row-level Option A vs grouped Option B) for Airfoil pilots.
3. When authorizing a pilot handoff, freeze split indices in a separate `airfoil-self-noise-v1.json` manifest — do not reuse this audit manifest as a split file.
4. Note attack-angle metadata discrepancy in proposal dataset description.

## Compliance

- **Write boundary:** only the four handoff-allowed locations changed.
- **No package installation, environment creation, or model execution.**
- **No other dataset audited.**

---

**End of handoff 002-NG-FD1. Fix pass 002F1-NG-FD1 appended below.**

## Fix Review Response (002F1-NG-FD1)

**Parent checkpoint:** `002-NG-FD1`  
**Completed:** 2026-08-23

### Changes made

| Item | Fix |
|---|---|
| Feasibility matrix header | Updated from “no desk audits executed” to acknowledge one `AUDIT_DRAFT` (Airfoil) |
| UCI metadata evidence | Fetched and saved `uci-291-metadata.json` from UCI API (`dataset.id=291` validated); SHA-256 `913da48b6ea2680afed32893844ef36dc71c08772ac665b3de695e717c0e71a1`; retrieval UTC `2026-08-23T14:47:54Z` |
| Manifest | Added `uci_metadata` block, `condition_groups` quantification, dual draft split options, `fix_handoff_id`, `updated_at` |
| Report | Added preserved-metadata citations, condition-group table, split Options A/B, updated supported/unresolved conclusions and path list |
| Airfoil matrix row | Group split → `Optional (106 cond.)`; notes updated for condition groups and metadata preservation |
| Unchanged (OpenCode-verified) | Raw zip/dat SHA-256, row count 1503, feature count 5, missingness 0, duplicate rows 0, target summary statistics |

### STRONG verdict

**Remains justified.** Source, licence, cleanliness, paper-absence, and compute feasibility are unchanged and now backed by preserved API metadata. Condition-group analysis refines split guidance but does not undermine dataset suitability; it clarifies that row-level splits are paper-aligned while grouped holdout is an optional robustness variant with a different scientific question.

### Compliance

- Only the four allowed files changed.
- No package installation, environment creation, model execution, or second dataset.
- No split indices generated or frozen.

**End of fix handoff 002F1-NG-FD1. Cursor stops here.**
