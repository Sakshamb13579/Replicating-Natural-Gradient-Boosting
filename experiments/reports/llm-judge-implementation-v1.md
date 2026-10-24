# LLM-judge implementation and smoke protocol v1

## Scope

The selected judge is MiniCPM4.1-8B at Q8_0, served by bare-metal Ollama.
Host/model setup is in `people/ash/work/local-llm/`, outside the research code
and result directories. Project requests use the existing loopback HTTP service.
This implementation phase runs authored miniature cases only; no held-out study
or human agreement evaluation has been launched.

## Project files

- `src/llm_judge.py`: standard-library HTTP client, prompts, strict verdict parsing,
  pairwise answer-identity mapping and binary agreement calculations.
- `src/run_llm_judge.py`: configurable runner, raw request/response preservation,
  configuration-checked resumption, summary and checksums.
- `data/judge/judge-smoke-v1.json`: three authored engineering cases, each with
  two explanations, explicitly marked non-reportable.
- `tests/test_llm_judge.py`: protocol tests independent of the model runtime.

Paths above are relative to `people/ash/work/paper-selection/ngboost/`.

## Scoring method

The client submits each explanation with its evidence packet and requests three
independent boolean verdicts: clarity, interval consistency and evidence support,
plus a brief reason. The prompt separates readability from correctness, forbids
invented causes and distinguishes prediction intervals from confidence intervals.
Candidate texts and packets are treated as data, and authored expectations are
excluded from the actual prompts.

Ollama's native JSON-schema mode constrains the response format. The client
rejects malformed JSON, string booleans, missing/extra fields, overlong reasons,
truncated generations and insufficient context headroom. It does not repair
outputs or silently extract a convenient substring.

This is **deterministic schema-constrained verdict generation**, replacing the
earlier tentative token-likelihood approach. It does not expose or fabricate
normalised label probabilities. All runtime, template, quantization and prompt
details must accompany the eventual judge results.

## Smoke requests

- Six pointwise explanation judgements.
- Six pairwise judgements: each of the three cases is shown in both A/B orders.
- One repeat of the first pointwise request to inspect deterministic score stability.
- Total: 13 requests, sequential, no NGBoost fitting.

The examples target a prediction/mean-interval confusion, an invented cause of
interval width, and obscuring jargon. Their labels are **authored engineering
expectations**, not independent human ratings or final-study gold labels.

The swap analysis compares underlying candidate IDs, not the literal A/B letters.
Disagreement between orders is recorded as order sensitivity, distinct from an
explicit substantive tie. Exact verdict equality and score equality are checked
separately for the repeated request.

## Commands

From the workspace root:

```bash
python -B -m unittest discover -s people/ash/work/paper-selection/ngboost/tests -p test_llm_judge.py -v
python -B people/ash/work/paper-selection/ngboost/src/run_llm_judge.py \
  --cases people/ash/work/paper-selection/ngboost/data/judge/judge-smoke-v1.json \
  --output people/ash/work/paper-selection/ngboost/artifacts/llm-judge-smoke-v1 \
  --repeat-first
```

The client needs only Python's standard library; no changes to the NGBoost
virtual environment or PyTorch installation are required.

## Acceptance and interpretation

Implementation readiness requires registered Q8_0 provenance, valid complete
structured outputs, correct order mapping, persisted raw evidence and readable
timing/failure records. A mismatch with an authored expectation is a judge-behaviour
finding, not a reason to rewrite a test label or hide an unsuccessful request.
Even perfect smoke behaviour does not establish human agreement or validate the
judge for the final benchmark. That requires development/test separation, frozen
case selection and prompts, and actual independent human ratings.

## Verification and real-model smoke results

All six protocol tests pass: answer-identity swapping, degenerate/imperfect
kappa, strict booleans/fields, truncation/unstructured-output rejection,
exclusion of authored expectations from prompts, and a real local HTTP transport
test exercising persistence, configuration guards and cache-only resumption.

The 8,699,309,920-byte model matched its published SHA-256. The existing bare-metal
Ollama 0.9.6 imported it successfully; no reinstallation or privileged service
change was required. Ollama identifies the model as `minicpm`, 8.2B, Q8_0, with
digest `54b5e0330160cbe553d56b81abd019722f1b871a5573ec62d9a8c561f4b0ffcf`.
A preliminary structured readiness probe also passed.

Actual outputs are preserved at `artifacts/llm-judge-smoke-v1/`, including the
configuration/model manifest, all 13 completed raw request/response records,
`summary.json` and passing checksums. These remain non-reportable engineering
examples rather than the final study.

| Check | Observed result |
|---|---|
| Valid pointwise outputs | 6 of 6 explanations |
| Valid pairwise outputs | 6 of 6 order-specific comparisons |
| Repeated first pointwise judgement | Identical scores and complete verdict |
| Swap consistency | 1 of 3 cases |
| Clarity vs authored expectations | 6 of 6 |
| Interval consistency vs authored expectations | 5 of 6 |
| Evidence support vs authored expectations | 4 of 6 |

The model correctly rejected the false coverage guarantee and the invented
training scarcity when these were scored individually. However:

1. It penalised a valid statement that the packet did not establish a cause of
   interval width, despite the rubric explicitly permitting accurate description
   without a causal claim.
2. It correctly found the jargon unclear but additionally invented confidence-
   interval and causal claims that the candidate did not make.
3. In one pairwise order, it treated the candidate's invented count of two training
   houses as if that count appeared in the authoritative evidence packet.
4. The interval-interpretation pair changed from a tie to the faithful explanation
   winning when reversed. The scarcity pair selected the literal label B in both
   orders, thereby selecting opposite underlying explanations.

Consequently, the transport/scoring implementation works, but this development
prompt/model combination is **not yet ready for the held-out evaluation**.
Next development work should strengthen evidence/candidate separation and criterion
independence, retest both orders, then freeze the rubric and assess actual human
agreement. These smoke expectations must not be relabelled to make the judge agree.

## Timing and resource observations

- The sum of completed-request wall times was 911.7 seconds (15.2 minutes), with
  aggregate decode throughput approximately 1.69 tokens/second on the shared CPU.
- Recorded successful responses contain 6,014 prompt tokens and 794 generated
  tokens in total. Prompt counts describe processing records, not independent cases.
- The terminal's 15-minute command limit interrupted the outstanding last request.
  Resuming reused the first 12 completed records and reran the final request.
  The timing sum above excludes that interrupted attempt and the separate readiness
  probe; it is not the total elapsed time of all attempts.
- While loaded, Ollama reported 9.1 GB and 100% CPU. The model was unloaded after
  the smoke test to release memory, while the service and disk model remain available.
- A full benchmark should be budgeted from these measured CPU timings and its
  final prompt sizes. Three criterion verdicts are bundled into each pointwise
  request, rather than requiring three separate model calls per explanation.
