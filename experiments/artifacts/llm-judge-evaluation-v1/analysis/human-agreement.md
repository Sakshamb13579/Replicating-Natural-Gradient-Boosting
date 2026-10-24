# Judge agreement against human ratings

Human reference ratings are **complete** for 1 rater(s), 30 items and 90 criterion verdicts each.
Human ratings are the reference and the judge verdicts are compared against them.

## Judge versus `anonymous-human-01`

Sheet: `ratings-anonymous-human-01.csv`.

| Criterion | n | Agreement | Cohen's kappa | Reference majority baseline |
|---|---:|---:|---:|---:|
| clarity | 30 | 90.0% | 0.000 | 90.0% |
| interval_consistency | 30 | 86.7% | 0.636 | 83.3% |
| evidence_support | 30 | 93.3% | 0.867 | 50.0% |
| pooled | 90 | 90.0% | 0.748 | 74.4% |

Items differing on any criterion: 6 of 30.

| Item | Construction intent | Stratum | Differing criteria | Human | Judge |
|---|---|---|---|---|---|
| `75cd8eb88551` | unsupported_cause | real-estate | interval_consistency | interval_consistency=1 | interval_consistency=0 |
| `85672ac90936` | obscuring_jargon | synthetic-B | clarity, interval_consistency | clarity=0, interval_consistency=1 | clarity=1, interval_consistency=0 |
| `90319537a775` | faithful_description | real-estate | evidence_support | evidence_support=1 | evidence_support=0 |
| `b1d5468a8547` | obscuring_jargon | airfoil | clarity, interval_consistency | clarity=0, interval_consistency=1 | clarity=1, interval_consistency=0 |
| `c9e45ed53ce7` | obscuring_jargon | real-estate | clarity, interval_consistency | clarity=0, interval_consistency=1 | clarity=1, interval_consistency=0 |
| `fcdc49267cf8` | faithful_description | real-estate | evidence_support | evidence_support=1 | evidence_support=0 |

Rater pass counts by construction intent, for description only:

| Construction intent | n | Clarity pass | Interval consistency pass | Evidence support pass |
|---|---:|---:|---:|---:|
| faithful_description | 15 | 15 | 15 | 15 |
| interval_interpretation | 5 | 5 | 0 | 0 |
| unsupported_cause | 5 | 5 | 5 | 0 |
| obscuring_jargon | 5 | 2 | 5 | 0 |

## Interpretation

- Human ratings are the reference; judge verdicts are the predictions compared against them
- Construction intent is an authored design category and is never used as a reference label
- The 30 items instantiate five explanation templates, so the items are not independent
- A single rater cannot separate judge error from one rater's idiosyncrasy
- No human-versus-human agreement is available with one sheet
