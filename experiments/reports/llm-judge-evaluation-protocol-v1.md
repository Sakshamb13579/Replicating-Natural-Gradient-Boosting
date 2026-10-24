# Full LLM-judge evaluation protocol v1

The user authorised the full inference run after seeing the smoke-test reliability
findings. The tested prompt is retained unchanged. This run measures that judge's
behaviour; it does not assume the smoke findings have been resolved.

## Frozen scope

Batch: `artifacts/llm-judge-evaluation-v1/`, relative to the NGBoost work directory.

- 15 held-out prediction cases, three each from fresh synthetic A/B/C, grouped
  Airfoil and Real Estate.
- Natural-variant predictions supply the evidence packets. The LLM is not choosing
  between NGBoost variants, and this is not an exhaustive audit of every test row.
- 30 authored explanations: one intended-faithful description and one controlled
  edit per case. Five edits focus on interval interpretation, five on unsupported
  causes and five on obscuring jargon. These are construction intents, not gold labels.
- 30 pointwise requests, each producing clarity, interval-consistency and
  evidence-support verdicts; 30 pairwise requests covering both answer orders.
- Total: 60 model requests and 90 pointwise criterion verdicts.

## Selection and independence

Selection seed is 20261018. Synthetic cases use saved held-out predictions from
replicates 2, 5 and 8; public cases use fold indices 0, 1 and 2. Selection is from
sorted saved test rows by seeded uniform sampling, excluding duplicate latent
synthetic observations across regimes and repeated public row IDs across cases.
Observed outcomes, prediction errors and judge scores do not determine selection.
Every selected row is verified against its saved outer-test membership, source
target and prediction checksum. Saved interval bounds are checked against the
Normal parameters before display rounding.

The six authored explanations in the earlier three-case smoke test are development
material. They are not included in this evaluation. The originally suggested larger
development set was not executed; the actual development history is the recorded
smoke test. Evaluation explanations still share templates and edit types with
development, which limits claims of broad task generalisation.

The unit for paired answer-order consistency is the 15 evidence cases, not the
30 displayed orders. Likewise, 30 explanations do not represent 30 independent
prediction cases. Synthetic source replicates are shared across regimes, although
the selected latent rows differ; no independent-sample inferential claim is made.

## Evidence and blinding

Packets contain the predicted Normal parameters, nominal 95% interval, units and
explicitly available explanatory evidence. Synthetic packets also provide their
known generating law and true conditional scale. Observed test outcomes and
empirical calibration claims are not supplied. Numeric displays use six significant
digits consistently in packets and explanations.

Candidate IDs are opaque and initial order is seeded/shuffled. Construction intent,
source row keys and authored categories are excluded from model prompts. Human
rating rows are separately shuffled and contain no judge output or construction key.
Human agreement remains pending until actual independent ratings are provided.

## Frozen inference

The batch stores copies of `llm_judge.py`, `run_llm_judge.py` and the preparation/
supervision script, plus frozen checksums, source hashes and model configuration.
The worker imports the frozen client/runner copies. MiniCPM4.1-8B Q8_0 is served
by the existing bare-metal Ollama service. Temperature is zero, seed zero, context
2,048 tokens, four CPU threads and 160 maximum output tokens. The non-reasoning
ChatML prefix and JSON schemas match the smoke test. Scoring emits discrete
verdicts, not calibrated label probabilities.

Invalid, truncated or malformed outputs remain recorded failures rather than
being rewritten. Completed requests are cached under a configuration guard.
The worker runs independently of the terminal, holds an exclusive lock and unloads
the model after finishing while preserving its registered weights and service.

## Commands

From the workspace root:

```bash
python -B people/ash/work/paper-selection/ngboost/src/run_full_judge_v1.py prepare
python -B people/ash/work/paper-selection/ngboost/src/run_full_judge_v1.py launch
python -B people/ash/work/paper-selection/ngboost/src/run_full_judge_v1.py status
python -B people/ash/work/paper-selection/ngboost/src/aggregate_full_judge_v1.py --detach
python -B people/ash/work/paper-selection/ngboost/src/aggregate_full_judge_v1.py
```

Preparation is immutable: differing inputs require a new batch version. Monitor
`status.json`, `worker.log` and the persisted request records. The final analysis
separates inference completion from pending human-reference evaluation. Only the
`human/` directory should be distributed to raters, each with a separate copy of
the blank sheet.

The detached analysis monitor waits for the inference worker to finish, independently
checks every frozen request payload and verdict, verifies checksums and writes
`analysis/summary.{json,md}`. Invalid outputs remain visible in the final rates;
construction intents are not substituted for human gold labels. The strict worker
may mark its validation failed if any response is invalid even after all 60 attempts
have completed; the analysis distinguishes that state from an interrupted/incomplete
batch. `analysis-monitor.json` records automatic postprocessing success or failure.
