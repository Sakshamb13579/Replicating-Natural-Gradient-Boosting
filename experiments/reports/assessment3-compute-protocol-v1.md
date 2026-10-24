# Assignment 3 compute protocol v1

## Authorisation and scope

Active person: ash. Authorised in chat on 17 October 2026 to start the compute-heavy
Assignment 3 work while the user writes report sections with Claude.

This protocol covers full-data Superconductivity, grouped-condition Airfoil, and
ten fresh constructed-data replicates. Existing Assignment 2 artifacts remain
historical evidence. Writing is being handled separately by the user and Claude.

## Frozen batch

All paths below are relative to `people/ash/work/paper-selection/ngboost/`.

- Batch: `artifacts/assessment3-compute-v1/`
- Scientific decisions: `protocol.json`
- Dataset/source/code hashes and task matrix: `batch.json`
- Frozen inputs: `inputs/<dataset>/{data.csv,splits.json,manifest.json}`
- Run outputs: `runs/<dataset>/<variant>/fold-XX/`
- Live scheduler status: `status.json`
- Logs: `logs/<dataset>--<variant>.log` and `supervisor.log`
- Validated final aggregate: `aggregate/{summary.md,summary.json,fold-ledger.csv}`

There are **96 dataset-variant tasks and 120 variant-fold evaluations**:

| Experiment | Design | Evaluations |
|---|---|---:|
| Superconductivity full | 3 variants x 5 repeated row holdouts | 15 |
| Airfoil grouped | 3 variants x 5 condition folds | 15 |
| Constructed fresh replicates | 10 independent data/split draws x A/B/C x 3 variants | 90 |

## Shared model protocol

- Natural and Ordinary use Normal; Homoscedastic uses NormalFixedVar.
- MLE, default tree learner, learning rate 0.01, full batches, cap 2,000 stages.
- Select the first minimum validation-NLL stage without test-label access.
- Fit the complete refit trajectory on outer training rows and predict its selected
  prefix, matching the existing external and constructed runners.
- Homoscedastic sigma is estimated from held-out inner-validation residuals at
  each stage. Retain the selected sigma after refitting the location model.
- This comparator uses validation outcomes for both stage and sigma, unlike the
  other two variants; disclose the asymmetry.
- External seeds: stage `20000+fold`, refit `30000+fold`.
- No scaling, imputation, outcome-driven row filtering, or test-based selection.

Every fold saves its complete validation-NLL trajectory, selected stage, retained
sigma if applicable, test predictions, interval bounds, recomputed metrics, timing,
and checksums. Environment records include the NGBoost commit and worktree status.
Completed folds are validated and reused on restart; an interrupted incomplete fold
is rerun. This is fold-level resumability, not stage-level model checkpointing.

## Full-data Superconductivity

- Source: audited `train.csv`, 21,263 rows and 81 model features; critical temperature
  is the target. No material strings or `unique_m.csv` enter the features.
- Preserve source order and all rows, consistent with the subset pilot.
- Five repeated 90/10 row holdouts using sequential `RandomState(1).choice`;
  inner 80/20 row holdout with seed `10000+fold`.
- Verified source totals: 66 exact duplicate extra rows, 6,093 repeated-feature extra
  rows, and 2,374 feature tuples mapping to multiple distinct targets.
- The input manifest records exact-row and feature-tuple overlap per outer and
  inner split. These are row-interpolation results, not a material-disjoint study.
- Full data is the primary result; do not pool it with the 5,000-row pilot.

## Grouped-condition Airfoil

- Reuse the checksum-verified 1,503-row processed source and five original features.
- Condition = exact tuple of angle, chord, velocity, and suction-side displacement
  thickness; frequency varies within a condition. Verified 106 unique conditions.
- Assign sorted condition IDs to five shuffled KFold partitions, seed 1, each
  condition tested exactly once. Groups, not rows, are the outer partition units.
- Inner validation holds out 20% of outer-training groups, seed `10000+fold`.
- Both levels are checked for group disjointness and complete row partitioning.
- Group IDs are evaluation-only and are excluded from model predictors.
- Compare to the earlier row-split study descriptively. Different folds, training
  proportions and generalisation targets do not support a matched row/group test
  or a claim that all performance change is solely caused by grouping.

## Ten fresh constructed-data replicates

- Import the existing frozen generator functions; do not run its writing entrypoint.
- Replicates `r=1..10`: PCG64 data seed `r`, split seed `1000+r`.
- Synthetic seed provenance is authoritative in each input manifest's `source`
  object. The reused split-integrity helper also emits legacy informational fields
  `outer_construction_seed=1` and `inner_random_state=10000`; those fields do not
  describe synthetic generation and are not read by the runner. Synthetic split
  labels come from the single PCG64 permutation with seed `1000+r`; there is no
  separate inner random split. Frozen row indices and hashes identify the exact
  split irrespective of those inherited helper labels.
- Each has 4,000 base rows: 2,400 training, 800 validation, 800 test.
- The existing formulas are unchanged: `mu=sin(x1)+x2^2`, A sigma
  `0.1+0.5*abs(x1)`, B sigma `0.6`, C sigma twice A, and shared standard-normal z.
- Within a replicate, all regimes share base inputs, z and split labels. All
  variants and regimes share model seeds: stage `20000+100*r`, refit `30000+100*r`.
  This is a prospective seed policy, differing from v1's regime-index model offsets.
- Only x1/x2 are features. Ground truth is evaluation-only.
- Evaluate NLL, RMSE, oracle NLL, scale MAE, scale correlation when defined,
  marginal intervals and conditional intervals in fixed |x1| bins
  `[0,.5), [.5,1), [1,1.5), [1.5,2)`.
- Report paired differences, means, sample SD/SE and win counts over the ten
  independent replicates, separately by regime. Pearson values within an 800-row
  test set are not a count of independent seeds.
- Keep v1 separate. Do not rewrite its hypotheses or require Homoscedastic to win
  B. No new equivalence tolerance or mandatory winner has been introduced.

## Validation before launch

`validation/preflight.json` records nine 25-stage nonreportable smoke runs (three
experiment types x three variants), successful independent prediction comparisons
against the old external runner for all variants, and paired-input checks for all
ten fresh replicates. Smoke metrics never enter the reportable aggregate.

The historical constructed-data audit is at
`historical-constructed-audit/{audit.md,audit.json}`. It recomputed all nine saved
metric sets from 7,200 predictions and verified data/source/prediction/metrics
hashes. Two old manifest.json checksums do not match their self-containing final
manifests. Those exceptions are recorded explicitly; historical files were not
rewritten. This audit cannot independently recover unsaved validation trajectories.

## Execution and monitoring

The existing environment is `.RnD/work/ngboost/.venv/bin/python`; no packages were
installed or changed. All numerical thread variables must be set **before Python
starts**. Four workers are used on the eight-logical-CPU host. Three start the long
Superconductivity variant tasks, and the remaining slot starts grouped Airfoil and
then fresh synthetic replicates. Worker slots are reused as tasks finish.

From the workspace root, to launch or resume after the supervisor has stopped:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .RnD/work/ngboost/.venv/bin/python -B \
  people/ash/work/paper-selection/ngboost/src/run_assessment3_compute.py launch --workers 4
```

The supervisor is detached from the terminal and holds an exclusive lock. It does
not require the chat to stay open, but the host must remain running. It writes
live state every five seconds. Final aggregation is automatic after tasks finish.
Any failed tasks remain visible in status and logs and can be retried by resuming.

To validate and aggregate currently completed tasks without declaring completion:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .RnD/work/ngboost/.venv/bin/python -B \
  people/ash/work/paper-selection/ngboost/src/aggregate_assessment3_compute.py --allow-partial
```

Do not edit frozen execution scripts or inputs during this batch. Any later
scientific protocol change needs a separately named/versioned batch. Results are
not final until all expected tasks, fold checksums and metric recomputations pass.
