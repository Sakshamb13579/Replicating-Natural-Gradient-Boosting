# Lead Review: NG-1A

## Verdict

**PASS for execution; CONDITIONAL for numerical reproduction. Proceed to NG-1B.**

## Verified Evidence

- Homoscedastic and ordinary-gradient two-split runs completed without new repairs.
- Natural-gradient 20-split run completed in 864.3 seconds.
- Source clone remains clean at `672034041fc05d9d4b7a80c1e6163ea102579752`.
- Detached worktree still differs by exactly the two import statements authorized in NG-0.
- Original Yacht runtime loader, official split construction, MLE score, 2,000-estimator cap, learning rate, and RMSE-based stage selection remained unchanged.
- Summary statistics in the implementation report agree with the raw logs.

## Finding Status

Natural-gradient NGBoost produced mean test NLL 2.1060, population SD 2.3466, and computed SE 0.5247 over 20 splits. The paper reports 0.20 +/- 0.26 SE. This is a substantial numerical mismatch, not a close reproduction.

The mismatch does not yet reject the central directional finding because:

- the two official ablation paths have only been run for two splits;
- natural gradient is still best in the exploratory two-split comparison;
- 11/20 natural folds hit the 2,000-estimator cap and another fold stopped at 1,999;
- one natural fold has NLL 10.9093, making the mean highly sensitive to an upper-tail failure;
- the current official script selects stages by validation RMSE despite the paper describing validation log-likelihood selection;
- the execution environment is materially newer than the paper environment.

## Decision

Authorize full 20-split homoscedastic and ordinary-gradient runs. These runs are necessary to test whether the central Table 2 ordering survives under one identical current official implementation.

NGBoost can no longer receive a `Strong PASS` from Yacht exact-value reproduction on the current commit. It may receive a `Directional PASS` if natural gradient clearly outperforms both ablations over the full paired splits. Otherwise it becomes `CONDITIONAL` or `DROP`.
