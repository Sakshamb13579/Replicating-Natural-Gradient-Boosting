# Replicating NGBoost: Probabilistic Regression on Existing and Constructed Data

Code, data and records for a COMP8240 replication of Natural Gradient
Boosting (Duan et al., ICML 2020). The report is
[`report/report.pdf`](report/report.pdf).

The project:

- runs the released NGBoost code on Yacht and Kin8nm;
- tests the paper's stage-selection rule in a controlled experiment;
- extends the three-variant comparison (Natural, Ordinary, Homoscedastic) to
  five further UCI datasets and to synthetic data with known noise;
- evaluates written explanations of predictions with a human annotator and an
  LLM judge.

## Layout

```
report/                 the report: PDF, LaTeX source, ACL style, build script
upstream/
  ngboost/              official NGBoost, git submodule pinned at 6720340 (Apache 2.0)
  patches/              the two-line import repair, kept separate from upstream
  CHANGES.md            what the repair changes and why
experiments/            everything written by this project
  src/                  data preparation, generators, experiment runners, aggregation
  tests/                unit tests for the judge code
  data/
    original/           Yacht (Kin8nm is not included, see the README there)
    processed/          prepared public-data CSVs, synthetic v1 data, split indices
    manifests/          per-dataset audits and manifests with SHA-256 values
    raw/                UCI metadata only; archives are fetched by a script
    judge/              smoke-test cases for the judge
    LICENCES.md         sources, licences and changes for every dataset
  reports/              protocols, data audits and per-step result reports
  artifacts/            saved runs: configs, commands, predictions, metrics, checksums
LICENSE                 MIT, for project-authored files only
INTEGRITY.md            the 9 recorded checksums that intentionally or historically differ
requirements.txt        pinned Python packages
```

## Where the report's evidence is stored

| Report | What | Path under `experiments/` |
|---|---|---|
| Sec. 3, Table 1 "Released code" | Logs, commands and environment of the repaired released script | `artifacts/released-code-run/` |
| Sec. 3, Table 1 "Controlled" | Selector experiment runs and comparison table | `artifacts/original-selector-sensitivity-v1/` (table: `comparison-s1-a01/`) |
| Sec. 3, Table 1 | Yacht and Kin8nm prepared inputs, split indices, manifests | `data/processed/yacht-*`, `data/manifests/{yacht,kin8nm}-original-selector-sensitivity-v1.json` |
| Sec. 3, cap diagnostic | 2,000 against 4,000 stages for Yacht Natural | `artifacts/yacht/ngboost-natural/017-ng-d1-yacht-natural-s1-a02/` |
| Sec. 3, protocol | Selector protocol and results report | `reports/original-selector-sensitivity-{protocol,results}-v1.md` |
| Sec. 4, App. B | Public-data audits (UCI record, citation, file hashes, column roles) | `data/manifests/*-audit-v1.json`, `reports/00[2-6]-*-data-audit.md`, `reports/external-dataset-preparation-audit-v1.md` |
| Sec. 4, App. B | Prepared public-data CSVs and saved split indices | `data/processed/<dataset>-v1/` |
| Sec. 4, App. B | Full-data Superconductivity and grouped Airfoil inputs | `artifacts/assessment3-compute-v1/inputs/{superconductivity-full,airfoil-grouped}/` |
| Sec. 4, App. B | Synthetic generator | `src/generate_heteroscedastic_v1.py`, `src/prepare_assessment3_compute.py` |
| Sec. 4, App. B | Ten synthetic replicates (A, B, C), with generator hash, data seed and split seed | `artifacts/assessment3-compute-v1/inputs/synthetic-r*-*/` (`data.csv`, `splits.json`, `manifest.json`) |
| Sec. 4 | Earlier v1 synthetic study, kept as a historical reference | `data/processed/heteroscedastic-v1/`, `artifacts/heteroscedastic-v1/` |
| Sec. 5, App. B | Public-data results, all five datasets | `artifacts/external-ngboost-v1/` (ledger: `ledger-s1-a02/`) |
| Sec. 5, App. B | Full-data, grouped and synthetic results | `artifacts/assessment3-compute-v1/runs/`, summary in `aggregate/` |
| Sec. 5, App. C | Judge protocol, cases and explanation set | `artifacts/llm-judge-evaluation-v1/{protocol,cases,construction-key}.json` |
| App. C | Raw prompts and responses (60 requests) with checksums | `artifacts/llm-judge-evaluation-v1/run/` |
| App. C | Frozen judge code and model configuration | `artifacts/llm-judge-evaluation-v1/code/`, `run/manifest.json` |
| App. C | Blank and completed annotation sheets, instructions | `artifacts/llm-judge-evaluation-v1/human/` |
| App. C | Agreement and judge analysis | `artifacts/llm-judge-evaluation-v1/analysis/` |
| App. C | Three-case development smoke test | `artifacts/llm-judge-smoke-v1/`, `data/judge/` |
| Supplementary | Author-reviewed rating pass, not used in the reported agreement | `artifacts/llm-judge-evaluation-v1/supplementary-llm/` |
| Historical | Airfoil pilot, run before the grouped design | `artifacts/airfoil-self-noise/`, `data/processed/airfoil-self-noise-v1/` |

Every run directory holds its `command.txt`, `config.json`, environment
record, outputs and a `checksums.txt`. These records are kept exactly as
written, so they contain the absolute paths of the machine they ran on. Paths
in the scripts themselves are relative to this repository.

## Setup

```bash
git clone --recurse-submodules https://github.com/Sakshamb13579/Replicating-Natural-Gradient-Boosting.git
cd Replicating-Natural-Gradient-Boosting
git -C upstream/ngboost apply ../patches/repair-imports.diff
python3.10 -m venv .venv && . .venv/bin/activate
python -m pip install --upgrade pip   # pip 21 cannot install NGBoost in editable mode
pip install -r requirements.txt
pip install --no-deps -e upstream/ngboost   # the pinned, patched NGBoost; dependencies come from requirements.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd experiments
python src/fetch_uci_data.py          # raw UCI archives, checked against the audits
python src/kin8nm_targets.py restore  # Kin8nm targets from the submodule, see below
```

The controlled experiments used Python 3.10.0 and single-threaded numerical
libraries, as set above. Applying the patch leaves
`examples/experiments/regression_exp.py` marked as modified inside the
submodule; that is expected. The commands below use Bash; brace expansion such
as `{natural,ordinary,homoscedastic}` also works in zsh but not in plain `sh`.

## Checking the reported numbers from saved results

These commands rebuild the summary tables from the saved predictions without
refitting any model, and finish in a few minutes. Run them from
`experiments/`.

```bash
# Section 3 controlled selector table
A=artifacts/original-selector-sensitivity-v1
python src/aggregate_original_selector_sensitivity.py --out-dir $A/check \
  $A/yacht/ngboost-{natural,ordinary,homoscedastic}/* \
  $A/kin8nm/ngboost-{natural,ordinary,homoscedastic}/*

# Section 5 public-data ledger
L=artifacts/external-ngboost-v1
python src/aggregate_external_ngboost_ledger.py --output-prefix $L/check/external-ngboost-ledger \
  $L/airfoil-self-noise $L/forest-fires $L/parkinsons-telemonitoring \
  $L/real-estate-valuation $L/superconductivity

# Section 5 full-data, grouped and synthetic summary (rewrites aggregate/)
python src/aggregate_assessment3_compute.py

# Section 5 and Appendix C judge summary, from the saved responses; no model needed (rewrites analysis/)
python src/aggregate_full_judge_v1.py

# Appendix C human-judge agreement (rewrites analysis/)
python src/aggregate_judge_human_agreement.py
```

The new outputs match the committed ones, apart from columns that record the
absolute run-directory path. Run `kin8nm_targets.py restore` first (setup does), because the selector
table needs the Kin8nm targets. `INTEGRITY.md` lists the few recorded
checksums that do not match, and why.

## Rerunning the experiments

Each command writes a new run directory. Use a new output path so that the
committed runs stay untouched.

```bash
# Released-code run (Section 3), from upstream/ngboost with the patch applied.
# The script downloads Yacht itself and reads Kin8nm from data/uci/.
cd ../upstream/ngboost   # from experiments/, where setup ends
python -m examples.experiments.regression_exp --dataset=yacht --score=MLE --natural --n-splits=20
python -m examples.experiments.regression_exp --dataset=yacht --score=MLE --n-splits=20
python -m examples.experiments.regression_exp --dataset=yacht --score=MLE --distn=NormalFixedVar --natural --n-splits=20
python -m examples.experiments.regression_exp --dataset=kin8nm --score=MLE --natural --n-splits=5
python -m examples.experiments.regression_exp --dataset=kin8nm --score=MLE --n-splits=5
python -m examples.experiments.regression_exp --dataset=kin8nm --score=MLE --distn=NormalFixedVar --natural --n-splits=5
cd ../../experiments

# Controlled selector experiment (Section 3): prepare inputs, then one run per dataset and variant
python src/prepare_original_selector_sensitivity_v1.py
python src/run_original_selector_sensitivity.py --dataset yacht --variant natural \
  artifacts/original-selector-sensitivity-v1/yacht/ngboost-natural/yacht-natural-rerun
#   datasets: yacht, kin8nm; variants: natural, ordinary, homoscedastic

# Cap diagnostic (Section 3)
python src/controlled_yacht_cap_diagnostic.py artifacts/yacht/ngboost-natural/cap-rerun

# Public-data runs (Section 5): one run per dataset and variant
python src/prepare_external_datasets_v1.py
python src/run_external_ngboost.py --dataset forest-fires --variant natural \
  artifacts/external-ngboost-v1/forest-fires/natural/forest-fires-natural-rerun

# Full-data Superconductivity, grouped Airfoil and the ten synthetic replicates
python src/prepare_assessment3_compute.py --batch artifacts/assessment3-compute-rerun
python src/run_assessment3_compute.py launch --batch artifacts/assessment3-compute-rerun --workers 4
python src/aggregate_assessment3_compute.py --batch artifacts/assessment3-compute-rerun

# Judge unit tests
python -m unittest discover tests
```

**The LLM-judge evaluation is a frozen, completed run and cannot be relaunched
in place.** `src/run_full_judge_v1.py` always writes to
`artifacts/llm-judge-evaluation-v1/` and refuses to launch once that run is
complete, so that the reported verdicts cannot be overwritten. What can be
rerun from the repository is the analysis: the 60 saved prompts and raw
responses in `run/requests/` are re-scored by `src/aggregate_full_judge_v1.py`
and compared with the human labels by `src/aggregate_judge_human_agreement.py`.
The run used Ollama 0.9.6 serving the Q8_0 GGUF identified in report
Appendix C under the model name `minicpm4.1-8b-q8_0-judge`; the request
options and server record are in `run/manifest.json`. Running the same prompts
on a new judge would need a copy of the script with a different output folder.

## What is not included, and where to get it

| Item | Why not | Where to get it |
|---|---|---|
| NGBoost source | Kept as upstream, not copied | Submodule `upstream/ngboost`, <https://github.com/stanfordmlgroup/ngboost> at `672034041fc05d9d4b7a80c1e6163ea102579752` |
| Kin8nm data | No separate licence statement located, so cited rather than redistributed | `upstream/ngboost/data/uci/kin8nm.csv`; description at <https://www.cs.toronto.edu/~delve/data/kin/desc.html>. The saved Kin8nm predictions have `y_true` removed; `src/kin8nm_targets.py restore` puts it back and checks the original checksums |
| Raw UCI archives | Duplicate the prepared CSVs | `src/fetch_uci_data.py`, or the UCI pages listed in `experiments/data/LICENCES.md` |
| Judge model weights (GGUF) | Third-party quantisation, referenced rather than redistributed | Hugging Face `mradermacher/MiniCPM4.1-8B-GGUF`, Q8_0 file; revision and SHA-256 in report Appendix C. The base model is `openbmb/MiniCPM4.1-8B` (Apache 2.0) |

## Building the report

```sh
cd report && python build.py
```

The build uses Podman with the `docker.io/texlive/texlive` image. It fails on
missing references, overfull boxes and en or em dashes.

## Licences

- Project-authored files: MIT (`LICENSE`).
- NGBoost: Apache License 2.0, in `upstream/ngboost/LICENSE`. Upstream has no
  NOTICE file.
- UCI datasets: CC BY 4.0, with credit and changes listed in
  `experiments/data/LICENCES.md`.
- ACL style files in `report/acl-style/`: distributed by the ACL with its
  paper template.
