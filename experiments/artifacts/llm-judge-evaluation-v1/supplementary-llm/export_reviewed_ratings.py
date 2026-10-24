"""Record the user's reported review/agreement, preserving assisted provenance."""

import csv
import hashlib
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "ratings-gpt-6.1-sol.csv"
OUTPUT = ROOT / "ratings-ash-reviewed.csv"


def main():
    original = SOURCE.read_bytes()
    reader = csv.DictReader(io.StringIO(original.decode("utf-8")))
    fields = reader.fieldnames
    rows = list(reader)
    assert len(rows) == len({row["item_id"] for row in rows}) == 30
    reviewed = []
    for row in rows:
        assert all(row[key] in ("0", "1") for key in (
            "clarity", "interval_consistency", "evidence_support"
        ))
        reviewed.append({
            **row,
            "rater_id": "ash-reviewed-ai-assisted-nonblinded",
            "notes": (
                "User reports reviewing the LLM ratings and agreeing with this "
                "item; AI-assisted, nonblinded endorsement. Original LLM rationale: "
                + row["notes"]
            ),
        })
    with OUTPUT.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reviewed)
    with OUTPUT.open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    assert saved == reviewed
    for before, after in zip(rows, saved):
        assert all(before[key] == after[key] for key in fields
                   if key not in ("rater_id", "notes"))
    assert SOURCE.read_bytes() == original
    print(json.dumps({
        "output": str(OUTPUT),
        "items": len(saved),
        "source_sha256": hashlib.sha256(original).hexdigest(),
        "source_unchanged": True,
        "provenance": "user-reported review and agreement; AI-assisted, nonblinded",
        "independent_human_reference": False,
    }, indent=2))


if __name__ == "__main__":
    main()
