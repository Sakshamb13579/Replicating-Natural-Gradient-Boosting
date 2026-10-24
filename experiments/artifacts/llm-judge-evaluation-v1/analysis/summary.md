# Full LLM-judge inference results v1

Inference attempts are complete; **human reference ratings are complete for 1 rater(s)**.

## Scope

15 held-out prediction cases and 30 authored explanations, with 90 pointwise criterion verdicts and 30 order-specific pairwise verdicts planned. Invalid outputs are counted separately below.
The same smoke-tested MiniCPM4.1-8B Q8_0 prompt was frozen before evaluation; no prompt selection used these outputs.

## Pointwise pass counts

These are model pass counts grouped by construction intent, **not accuracies against human gold**.

| Construction intent | Planned | Valid outputs | Clarity pass | Interval consistency pass | Evidence support pass |
|---|---:|---:|---:|---:|---:|
| faithful_description | 15 | 15 | 15 | 15 | 13 |
| interval_interpretation | 5 | 5 | 5 | 0 | 0 |
| unsupported_cause | 5 | 5 | 5 | 4 | 0 |
| obscuring_jargon | 5 | 5 | 5 | 2 | 0 |

## Pairwise order effects

- Same underlying verdict in both orders: **4/15** valid paired cases (15 planned).
- Order-sensitive: **11** valid paired cases (15 planned).
- Invalid request records: **0/60**; paired cases missing a valid order: 0/15.
- Literal label counts across 30 valid order-specific comparisons (30 planned): {'B': 20, 'Tie': 6, 'A': 4}.
- Ordered winner construction categories: {'interval_interpretation': 5, 'faithful_description': 19, 'Tie': 6}.
- Consistent winner construction categories: {'faithful_description': 4}.

| Stratum | Cases | Consistent | Order-sensitive | Consistent explicit ties |
|---|---:|---:|---:|---:|
| synthetic-A | 3 | 1 | 2 | 0 |
| synthetic-B | 3 | 1 | 2 | 0 |
| synthetic-C | 3 | 1 | 2 | 0 |
| airfoil | 3 | 0 | 3 | 0 |
| real-estate | 3 | 1 | 2 | 0 |

## Timing

Completed-request wall-time sum: 78.4 minutes.
Aggregate decode throughput: 1.97 tokens/second.

## Human ratings

Completed sheets: ratings-anonymous-human-01.csv.
Judge-versus-human agreement, with the human ratings as the reference:

| Rater | Criterion | Agreement | Cohen's kappa |
|---|---|---:|---:|
| anonymous-human-01 | clarity | 90.0% | 0.000 |
| anonymous-human-01 | interval_consistency | 86.7% | 0.636 |
| anonymous-human-01 | evidence_support | 93.3% | 0.867 |
| anonymous-human-01 | pooled | 90.0% | 0.748 |

Per-item disagreements and the rater pass counts are in `analysis/human-agreement.md`.

Raw outputs, settings and checksums remain in `run/`; source provenance and code snapshots were frozen before launch.
