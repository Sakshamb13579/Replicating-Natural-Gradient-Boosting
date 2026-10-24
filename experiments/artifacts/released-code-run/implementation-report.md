# Phase 0 Implementation Report

**Agent:** Cursor Primary RnD and Implementation  
**Task:** STATUS.md bounded Yacht functional check  
**Commit:** `672034041fc05d9d4b7a80c1e6163ea102579752`

## Timing

| Stage | Start (IST) | End (IST) | Wall time |
|-------|-------------|-----------|-----------|
| Environment setup | 2026-08-22 ~20:11 | ~20:12 | ~82 s install |
| Unmodified run | 2026-08-22 20:12:25 | 20:12:29 | 3.6 s (failed) |
| Repair + reinstall | 2026-08-22 ~20:13 | ~20:14 | ~1 min |
| Repaired run | 2026-08-22 ~20:14 | ~20:15 | 88.2 s |

## Environment

- Python: 3.10.0
- Venv: `.RnD/work/ngboost/.venv`
- Editable install (after repair): `repair-worktree/`
- Key packages: ngboost 0.5.11.dev0, scikit-learn 1.6.1, numpy 2.2.6, pandas 2.3.3, scipy 1.15.3
- Extra experiment deps: pandas, openpyxl, xlrd
- Full freeze: `raw/environment.txt`

## Commands

```bash
# Environment
cd /mnt/hdd1/Projects/Research/Saksham/.RnD/work/ngboost
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install -e ../../source/ngboost pandas openpyxl xlrd

# Unmodified (from clean source clone)
cd /mnt/hdd1/Projects/Research/Saksham/.RnD/source/ngboost
python -m examples.experiments.regression_exp \
  --dataset=yacht --score=MLE --natural --n-splits=2

# Worktree + import-only repair
git worktree add ../work/ngboost/repair-worktree 672034041fc05d9d4b7a80c1e6163ea102579752
# patch: add Normal, NormalFixedVar, MLE imports — see raw/repair-imports.diff
pip install -e repair-worktree

# Repaired run (from worktree)
cd /mnt/hdd1/Projects/Research/Saksham/.RnD/work/ngboost/repair-worktree
python -m examples.experiments.regression_exp \
  --dataset=yacht --score=MLE --natural --n-splits=2
```

## Unmodified outcome

- **Exit code:** 1
- **Yacht loader:** succeeded (`X.shape=(308, 6)`)
- **Failure:** `NameError: name 'Normal' is not defined` at `Dist=eval(args.distn)`
- **Log:** `raw/yacht-natural-unmodified-split2.log`

## Repair

- **Location:** `.RnD/work/ngboost/repair-worktree/` (detached worktree)
- **Diff:** `raw/repair-imports.diff` — two import lines only
- **No changes** to splits, metrics, LR, n_estimators, early stopping, or model logic

## Repaired outcome

- **Exit code:** 0
- **Runtime:** 88.2 s wall
- **Log:** `raw/yacht-natural-repaired-split2.log`

| Split | BestIter | Val RMSE | Test RMSE | Test NLL |
|------:|---------:|---------:|----------:|---------:|
| 1 | 1416 | 0.4903 | 0.5288 | 0.2890 |
| 2 | 2000 | 0.4935 | 0.4579 | 1.0220 |

Summary line: mean test NLL **0.6555 ± 0.3665** (script prints sample std, not SE).

## Source-clone clean status

- `.RnD/source/ngboost/`: clean working tree on `master` @ `6720340`
- Repair isolated to `repair-worktree/` only

## Blockers / notes

- `FutureWarning` on `delim_whitespace` in pandas — non-blocking
- Paper Table 2 uses 20 splits; Phase 0 used 2 per task spec
- Numerical comparison to paper deferred to lead review (Phase 2)

## Artifacts

| File | Purpose |
|------|---------|
| `raw/environment.txt` | pip freeze + key versions |
| `raw/commands.log` | timestamps |
| `raw/yacht-natural-unmodified-split2.log` | expected failure |
| `raw/yacht-natural-repaired-split2.log` | successful 2-split run |
| `raw/repair-imports.diff` | import-only patch |
| `raw/data-checks.md` | Yacht shape / URL |
| `raw/implementation-report.md` | this report |
