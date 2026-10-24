# Airfoil: Proposal First Run versus Project Update Protocol

## Scope and result

The submitted proposal's Table 3 explicitly called its Airfoil numbers **first-run,
validation-RMSE-selected** results and promised a validation-NLL rerun (D4/D5).
That rerun is complete. The mean test-NLL ordering reversed: Natural narrowly led
Homoscedastic in the proposal run, whereas Homoscedastic led Natural in all five
paired folds of the new run. These are two completed protocols on the same frozen
Airfoil data and splits, **not** an isolated experiment changing only the selector.

| Protocol | Natural mean test NLL | Homoscedastic mean test NLL | Ordinary mean test NLL | Natural mean fold RMSE | Homoscedastic mean fold RMSE | Ordinary mean fold RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Proposal first run: validation RMSE, released-style post-refit Homoscedastic scale | **2.218038** | 2.246422 | 2.831865 | 2.388430 | **2.072678** | 5.004616 |
| Project Update: validation NLL, retained inner-validation Homoscedastic scale | 2.247040 | **2.156740** | 2.946515 | 2.591912 | **2.073056** | 5.413560 |

Values are unweighted means of the five fold-level metrics. In the proposal first
run, Natural's NLL advantage over Homoscedastic was only `0.028384` (the preserved
paired SE was about `0.054`). Under the revised protocol, Homoscedastic's mean-NLL
advantage is `0.090299`, and it beats Natural on NLL and RMSE on **5/5** paired
folds. This is a reversal between protocols, not a claim of statistical significance.

## Matched data and model inputs

- Both runs used the same frozen `airfoil-self-noise-v1.csv` (SHA-256
  `705a2a7f0e55fcc9a87700038487b12ce768f4a2e1692433a2d752309028546e`)
  and `split-indices.json` (SHA-256
  `128c5375b2659dff369e153e1b35f2163fbd1c87b72cc5a183def469ce014c14`).
- The old and new per-fold metrics have identical outer-train, outer-test,
  inner-train, and inner-validation index hashes on **all five folds** for each
  variant. For example, fold 0's hashes are `a8a15d3c…` (outer train),
  `da78dec0…` (outer test), `b4eb1d2e…` (inner train), and `db2cecad…`
  (inner validation) in both sets of artifacts.
- The per-fold stage-model and full-model seeds also match: `20000+fold` and
  `30000+fold`. Both runs declare 2,000 estimators, learning rate `0.01`, MLE,
  the default tree learner, and full-batch fitting.

These checks rule out a different frozen row split as the explanation of the
observed aggregate reversal. They do not make the two runner protocols identical.
The first-run Homoscedastic config records runner SHA-256
`7b959ac78d3864d65fe5a63b76baf03f7bcb6ef4ed4b8d2e1e2e3230739002d9`;
the revised Homoscedastic run manifest records runner SHA-256
`63bcdd34f3b521d8896d54e3417f302086cbd7da5e5276fd00156b94dfcbe35d`.
Both hashes match the current preserved runner files inspected for this comparison.

## Confirmed differences in code and artifacts

1. **Stage selection:** `src/run_airfoil_ngboost_variant.py` chooses the minimum
   validation RMSE (`val_rmse`, `best_iter` at lines 203–209). The newer
   `src/run_external_ngboost.py` chooses minimum validation NLL (`select_stage`,
   lines 124–144). Natural selected a different stage on all five folds;
   Ordinary also changed on four of five.
2. **Homoscedastic scale:** the first runner evaluates the selected outer-refit
   model on the held-out inner-validation rows, takes its residual RMS, and
   multiplies that factor by `forecast.var` for test prediction (`run_airfoil_ngboost_variant.py`,
   lines 222–238). The new runner computes a validation-residual RMS sigma for
   each candidate inner-training stage, retains the sigma of the selected stage,
   refits the location trajectory, and uses that retained sigma on test rows
   without re-estimating it (`run_external_ngboost.py`, lines 124–144, 168–183).
3. **Homoscedastic stages:** despite the changed selection procedure, the five
   selected stages are **identical in both runs**: `2000, 1955, 2000, 2000, 1981`.
   Its change in NLL therefore cannot be explained by a different chosen stage.
   The mean-fold RMSEs are very close (`2.072678` and `2.073056`).

| Fold | Homoscedastic selected stage, both runs | First-run test scale | Revised test scale | First-run test NLL | Revised test NLL |
|---:|---:|---:|---:|---:|---:|
| 0 | 2000 | 1.666037 | 2.439066 | 2.070804 | 2.109823 |
| 1 | 1955 | 1.391049 | 1.988558 | 2.264619 | 2.103330 |
| 2 | 2000 | 1.491505 | 2.052806 | 2.150985 | 2.077368 |
| 3 | 2000 | 1.571614 | 2.053454 | 2.213914 | 2.133197 |
| 4 | 1981 | 1.735839 | 2.264461 | 2.531789 | 2.359984 |

The new scale is larger on every fold; the NLL improves on four folds and worsens
on fold 0. These are descriptive observations from preserved runs, not an
independently fitted scale-rule ablation.

## Interpretation and limitation

**Causal attribution is not isolated.** Stage selection and Homoscedastic scale
handling changed together between the first and revised Airfoil runs. The matching
Homoscedastic stages and changed stored scales show that its NLL change was not
caused by a stage-count change, but the full Natural-versus-Homoscedastic ranking
reversal has not been decomposed into selector-only and scale-only effects. Do not
say the proposal's Natural edge was *caused solely* by using the RMSE selector or
that the revised protocol proves Homoscedastic universally superior.

Airfoil also retains the proposal's D4 limitation: the frozen outer splits are
row-level holdouts across repeated aerodynamic conditions. They assess
interpolation among rows and do not test generalisation to entirely unseen
wind-tunnel conditions. Homoscedastic hit the 2,000-stage cap on three of five
revised folds; robustness and grouped-condition sensitivity remain follow-up work.

## Sources

- Submitted proposal: `shared/607773723_Project_Proposal.pdf`, Table 3 and D4/D5.
- First-run review: `reports/025-ng-p4-airfoil-three-way-lead-review.md`.
- First-run artifacts: `artifacts/airfoil-self-noise/ngboost-{natural,ordinary,homoscedastic}/`.
- Revised protocol: `reports/external-ngboost-protocol-v1.md`.
- Revised artifacts: `artifacts/external-ngboost-v1/airfoil-self-noise/{natural,ordinary,homoscedastic}/`.
- Checked current ledgers: `artifacts/external-ngboost-v1/ledger-s1-a02/external-ngboost-ledger.md`
  and `external-ngboost-ledger.paired.md` in the same directory.

All relative `reports/` and `artifacts/` paths above are under
`people/ash/work/paper-selection/ngboost/`. The first-run and revised artifact
sets each preserve their own commands, configuration, metrics, predictions, and
checksum manifests.
