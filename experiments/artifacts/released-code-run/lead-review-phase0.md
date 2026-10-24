# Lead Review: NGBoost Phase 0

## Verdict

**PASS (functional only).**

The official Yacht experiment path loads original data and completes natural-gradient training after a documented import-only compatibility repair. This phase does not establish numerical reproduction of Table 2.

## Verified Evidence

- Clean source clone at `672034041fc05d9d4b7a80c1e6163ea102579752`.
- Detached repair worktree at the same commit.
- Repair diff contains exactly two added import statements.
- Unmodified run reaches Yacht loading and fails at the expected undefined `Normal` symbol.
- Repaired two-split natural run exits successfully.
- No method, data, split, metric, stage-selection, learner, estimator-count, or learning-rate logic changed.
- Runtime was 88.2 seconds for two splits.

## Numerical Observation

The two natural-gradient fold NLL values are 0.2890 and 1.0220. Their mean is 0.6555.

The script reports `np.std`, whose default `ddof=0` is the **population standard deviation**, 0.3665. The implementation report incorrectly labels this as sample standard deviation. With only two folds, this result is functional evidence only and should not be compared strongly with the paper's 20-repetition standard error.

## Evidence Corrections Needed

- Correct `sample std` to `population std` in future reports. Also report standard error as `population_std / sqrt(n)` when comparing with the paper.
- `data-checks.md` says there is no local cache, but the previously downloaded file exists at `.RnD/work/ngboost/data/yacht_hydrodynamics.data`, SHA-256 `00dfecc0fc01ddd4c90b558a3ac11b246df8ebcfea130724223475a9a67f0ea1`. The experiment itself did use the official runtime URL, which is acceptable.
- The implementation report's displayed relative `git worktree add ../work/...` command does not resolve correctly from `source/ngboost/`. The actual worktree is correctly located under `.RnD/work/ngboost/`; future command records should use absolute paths.

## Decision

Proceed to Phase 1. Do not lock NGBoost yet. Numerical validation requires all three variants over the paper's 20 splits.
