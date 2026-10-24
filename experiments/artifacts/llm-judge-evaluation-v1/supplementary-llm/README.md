# Supplementary LLM ratings

## Current status: draft removed; fresh user rating requested

At the user's request on 2026-10-18, the separate `ratings-gpt-6.1-sol.csv`
draft and its explicit-decision exporter `export_ratings.py` were deleted.
Descriptions below document their historical provenance, not currently available
files. Historical exporters that depend on the deleted sheet are no longer
runnable without that source. The earlier user-endorsed copies remain historical
review records, not fresh independently entered ratings.

The user has requested a new item-by-item session in which the assistant displays
only evidence and explanation, supplies no scores, and transcribes the user's
own choices. These new choices may be described as human-entered, with assistant
transcription, rather than model-generated. Prior exposure to model scores remains
part of the session provenance, so they must not be described as blinded or
independent of that exposure. No fresh user scores have been supplied yet.

## Provenance

- Rating date: 2026-10-18.
- Rater model: `openai/gpt-6.1-sol`, acting in the existing OpenCode conversation.
- CSV rater ID: `openai-gpt-6.1-sol-supplementary-nonblinded`.
- Output: `ratings-gpt-6.1-sol.csv`, containing all 30 items in the original order.
- Input: `../human/ratings-blank.csv`, with the rubric from `../human/INSTRUCTIONS.md`.
- Requested by the user as a separate copy containing this assistant's assessments.

These are AI-generated ratings, not human annotations or human gold labels.
Human-reference ratings remain pending. Model size alone does not establish
that these assessments are correct or superior to MiniCPM's assessments.

The conversation already contained MiniCPM aggregate findings, construction
descriptions and individual verdicts, including the verdict for `7bf12556637c`.
Consequently these ratings are not fully blinded or an independent replication.
The individual judgements were made from the supplied evidence/explanation pairs
under the stated rubric; no additional model API was called for this sheet.

## Format and decisions

The original nine columns are retained. `item_id`, `evidence_json` and
`explanation` are copied verbatim as decoded CSV fields. All three criterion
columns contain `1` (pass) or `0` (fail), with a rationale in `notes`.

- **Clarity:** understandable, unambiguous wording. A clearly worded false claim
  can pass this criterion.
- **Interval consistency:** numeric values, units and interpretation must agree
  with the packet. Calling an outcome prediction interval a confidence interval
  for the conditional mean, or guaranteeing the next outcome, fails.
- **Evidence support:** descriptive statements can pass without explaining a
  causal mechanism. Invented neighbour counts, local variation or guarantees fail.
- The simulated conditional noise scale is distinct from fitted predictive
  standard deviation. Reporting the known generating scale does not claim the
  fitted interval is calibrated.

For the five jargon-heavy items, clarity fails and interval consistency passes:
the supplied numeric and nominal-interval statements remain correct. Evidence
support fails because the additional undefined technical-process assertion is
not substantiated by the packet, rather than because confusing language
automatically invalidates every other criterion. This last decision involves
interpretation: a rater treating that sentence as empty stylistic padding might
reasonably assign a different evidence-support label. The row notes retain the
reason for this assistant's judgement.

## Export and verification

`export_ratings.py` records the 30 explicit item-level judgements and serialises
them into a separate CSV. It does not infer labels from construction categories,
read MiniCPM outputs, or perform inference. Running it a second time refuses to
overwrite the existing CSV.

The exporter checks the complete ID set, binary values, populated rater IDs and
notes, original field contents/order, CSV round-trip equality and the original
blank sheet's byte-for-byte preservation. It prints the input hash and pass
counts. This supplementary directory is outside the frozen input manifest;
the frozen benchmark and original human-rating sheet are preserved.

## User-reviewed copy

On 2026-10-18, the active user (`ash`) stated that they had checked the LLM
ratings and that their ratings agreed on every item, and requested a copy under
their identity. `ratings-ash-reviewed.csv` records that reported review and
endorsement. The current rater ID is `ash`; the initial export used
`ash-reviewed-ai-assisted-nonblinded`.

The values are copied from the supplementary LLM sheet at the user's request;
no separate pre-exposure user rating sheet was supplied. Each row discloses the
non-blinded review. Copied LLM rationales have been removed from the user sheets;
they remain in the separate original LLM sheet, attributed as such. This is
AI-assisted, nonblinded user review, not a second independent rater or independent
human gold. The reported complete agreement must not be presented as blinded
human–LLM validation or used to estimate independent inter-rater reliability.
Independent human-reference ratings remain pending.

`export_reviewed_ratings.py` creates and verifies this separate copy without
changing the LLM sheet, frozen benchmark, original human sheet or main analysis.

### Subsequent update to the original sheet

The user subsequently explicitly requested that the reviewed ratings also fill
`../human/ratings-blank.csv`. `apply_user_review.py` performs that update while
archiving the original blank bytes in `ratings-blank-before-user-review.csv`.
The original path is therefore now populated, superseding the earlier blank-sheet
status. See `../human/RATINGS-UPDATE.md` for attribution and the effect on the
historical frozen checksum entry. The main analysis remains unchanged.

### Concise user-review notes

At the user's request, `simplify_review_notes.py` replaces the long rater ID with
`ash` and removes copied LLM rationales from both user-review sheets. Each row
retains a concise factual note: Ash reviewed the items and reports agreement with
LLM ratings already seen; the review was non-blinded. Source-model details remain
in this provenance record and the separate LLM sheet. All criterion values,
evidence, explanations and item order are unchanged. User review is complete;
analysis and drafting can proceed without representing it as independent
human-reference validation.
