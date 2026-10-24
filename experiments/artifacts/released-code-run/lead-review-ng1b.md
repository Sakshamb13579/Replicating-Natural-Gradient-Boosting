# Lead Review: NG-1B

## Verdict

**Directional PASS. Proceed to NG-1C on Kin8nm.**

## Verified Evidence

- Natural, homoscedastic, and ordinary-gradient variants completed over 20 identical Yacht splits.
- Source clone remains clean at `672034041fc05d9d4b7a80c1e6163ea102579752`.
- Detached worktree still contains only the two authorized import statements.
- Raw logs support the report's aggregate and paired statistics.
- No model, split, metric, data, learning-rate, estimator-count, or stage-selection logic changed.

## Supported Finding

Under the current official experiment implementation, natural-gradient NGBoost is decisively better than both available official ablations on Yacht test NLL:

- lowest mean NLL: 2.1060;
- lowest median NLL: 1.7924;
- 20/20 paired wins against homoscedastic;
- 19/20 paired wins against ordinary gradient.

This supports the paper's central directional claim that natural-gradient updates can materially improve multiparameter probabilistic boosting.

## Not Reproduced

- No variant matches its paper Table 2 absolute NLL value.
- Our ordering is `Natural < Ordinary < Homoscedastic`; the paper reports `Natural < Homoscedastic < Ordinary` on Yacht.
- Natural mean NLL is 2.1060 rather than 0.20.
- Ordinary saturates the 2,000-estimator cap on 19/20 folds; natural does so on 11/20.
- Extreme upper-tail NLL failures materially inflate all means.

The result must not be described as an exact or complete Table 2 reproduction. The defensible wording is: the current official code reproduces the natural-gradient advantage direction on Yacht, while absolute results and the ordering of the two ablations diverge.

## Confirmation Choice

Use Kin8nm rather than Wine.

- Kin8nm is bundled in the repository and has an unambiguous 8,192-row identity.
- Wine is described ambiguously in the PDF and its reported sample count does not clearly identify the exact construction.
- Kin8nm tests a substantially larger data regime.
- This is a non-cherry-picked target: paper Table 2 reports ordinary gradient slightly better than natural gradient on Kin8nm, within uncertainty.

Paper Kin8nm targets:

| Variant | Test NLL, mean +/- SE |
|---|---:|
| Natural gradient | -0.49 +/- 0.02 |
| Ordinary gradient | -0.52 +/- 0.03 |
| Homoscedastic | -0.37 +/- 0.05 |

The expected directional pattern is `Ordinary approximately Natural < Homoscedastic`, not a forced natural-gradient win.

## Decision

Authorize a timeboxed five-split Kin8nm confirmation across all three variants. NGBoost remains viable but cannot be locked until this confirmation and the independent MiniRocket smoke test are reviewed.
