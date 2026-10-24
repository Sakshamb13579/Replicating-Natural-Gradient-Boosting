# User-requested rating-sheet update — 2026-10-18

At the user's explicit request, `ratings-blank.csv` now contains all 30 reviewed
ratings from `../supplementary-llm/ratings-ash-reviewed.csv`. Despite its historical
filename, this sheet is no longer blank. Item IDs, evidence, explanations, column
names and row order are preserved.

Rater: `ash` (initial export: `ash-reviewed-ai-assisted-nonblinded`). The user reported reviewing the
assistant's ratings and agreeing on all items. These are AI-assisted, nonblinded
user-reviewed ratings, not independent human-reference annotations. Copied
LLM rationales were subsequently removed from the user rows at the user's request;
they remain in the separate LLM sheet. Row notes retain the non-blinded-review
fact and point to source provenance. User review is complete and does not block
continued analysis or report drafting. No independent human
agreement estimate has been added to the main analysis.

The exact original blank sheet is archived at:

`../supplementary-llm/ratings-blank-before-user-review.csv`

Its SHA256 is:
`53b8914a09c93570eed30e37ad5e4f515e6072e2d2e467e1649769278497a6ec`.

The historical `../frozen-checksums.txt` is retained as the pre-rating manifest.
Its `human/ratings-blank.csv` entry now refers to the archived blank bytes and
will not match the filled sheet at its old path. This is a documented
post-evaluation annotation update, not a change to the inference inputs or raw
judge outputs. Use the archived blank copy when verifying that original entry.
