"""Simplify user-review labels while retaining truthful exposure provenance."""

import csv
import io
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PATHS = (ROOT / "ratings-ash-reviewed.csv", ROOT.parent / "human/ratings-blank.csv")
NOTE = (
    "Ash reviewed all items and reports that their own ratings agree with the "
    "LLM ratings they had already seen; non-blinded review. "
    "Source provenance: ../supplementary-llm/README.md."
)


def main():
    originals = [path.read_bytes() for path in PATHS]
    assert originals[0] == originals[1], "Review sheets differ; inspect before editing."
    reader = csv.DictReader(io.StringIO(originals[0].decode("utf-8")))
    fields = reader.fieldnames
    rows = list(reader)
    assert len(rows) == len({row["item_id"] for row in rows}) == 30
    updated = [{**row, "rater_id": "ash", "notes": NOTE} for row in rows]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(updated)
    data = buffer.getvalue().encode("utf-8")
    assert list(csv.DictReader(io.StringIO(data.decode("utf-8")))) == updated
    for before, after in zip(rows, updated):
        assert all(before[key] == after[key] for key in fields
                   if key not in ("rater_id", "notes"))
    for path, original in zip(PATHS, originals):
        assert path.read_bytes() == original
        path.write_bytes(data)
        assert path.read_bytes() == data
    print("Updated both user-review sheets: 30 items each; rater ash; concise non-blinded review note.")
    print("Verified unchanged item contents, ratings, and ordering.")


if __name__ == "__main__":
    main()
