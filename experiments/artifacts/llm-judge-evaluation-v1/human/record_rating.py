"""Transcribe explicitly supplied anonymous-rater scores, never infer them."""

import argparse
import csv
import io
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "supplementary-llm/ratings-blank-before-user-review.csv"
OUTPUT = ROOT / "ratings-anonymous-human-01.csv"
CRITERIA = ("clarity", "interval_consistency", "evidence_support")


def read_sheet(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("item_id")
    for criterion in CRITERIA:
        parser.add_argument(criterion, choices=("0", "1"))
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    fields, source = read_sheet(SOURCE)
    if OUTPUT.exists():
        output_fields, rows = read_sheet(OUTPUT)
        assert output_fields == fields
    else:
        rows = [dict(row) for row in source]
    assert len(rows) == len(source) == 30
    for original, current in zip(source, rows):
        assert all(original[key] == current[key] for key in
                   ("item_id", "evidence_json", "explanation"))
    matches = [row for row in rows if row["item_id"] == args.item_id]
    assert len(matches) == 1, "Unknown or duplicated item ID"
    row = matches[0]
    assert all(not row[key] for key in CRITERIA), "Already rated; inspect before changing"
    row.update({key: getattr(args, key) for key in CRITERIA})
    row.update(rater_id="anonymous-human-01", notes=args.notes)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    assert list(csv.DictReader(io.StringIO(buffer.getvalue()))) == rows
    OUTPUT.write_text(buffer.getvalue(), encoding="utf-8", newline="")
    assert read_sheet(OUTPUT)[1] == rows
    count = sum(all(row[key] in ("0", "1") for key in CRITERIA) for row in rows)
    print(f"Recorded {args.item_id}; {count}/30 items rated in {OUTPUT.name}.")


if __name__ == "__main__":
    main()
