# Lead Review: NG-1C

## Verdict

**Supports confirmation. NGBoost retains a Directional PASS with cross-dataset evidence.**

## Verified Evidence

- Kin8nm is the official bundled 8,192-row dataset with eight features and one target.
- Dataset checksum and missing-value checks are recorded.
- Natural, homoscedastic, and ordinary-gradient variants completed over five identical splits.
- Source clone remains clean at `672034041fc05d9d4b7a80c1e6163ea102579752`.
- Detached worktree still contains only the two authorized import statements.
- No new compatibility or protocol changes were introduced.
- Reported statistics agree with raw logs.

## Confirmation Result

| Variant | Paper mean NLL | Our five-split mean NLL |
|---|---:|---:|
| Ordinary gradient | -0.52 | -0.5078 |
| Natural gradient | -0.49 | -0.4191 |
| Homoscedastic | -0.37 | -0.3623 |

The complete ordering `Ordinary < Natural < Homoscedastic` matches paper Table 2. Ordinary beats natural on all five paired splits; natural beats homoscedastic on all five. This was a predefined non-favourable target for natural gradient and therefore provides useful protection against cherry-picking.

Five splits are not equivalent to the paper's 20 repetitions, but the signs, scale, ordering, and paired behaviour support confirmation.

## Overall NGBoost Status

- Functional official-code path: PASS after import-only repair.
- Yacht exact values: not reproduced.
- Yacht central natural-gradient advantage: directionally reproduced.
- Kin8nm ordering and scale: confirmed on five splits.
- Candidate classification: **Directional PASS**.

NGBoost is proposal-viable. The defensible project narrative includes both the reproduced cross-dataset behaviour and the unresolved Yacht numerical/protocol discrepancy. It must not claim exact reproduction of all Table 2 values.

## Decision

Close local NGBoost candidate validation for now. Proceed to the independent MiniRocket `MR-0` backup test before paper lock. Colab confirmation remains required if NGBoost is selected.
