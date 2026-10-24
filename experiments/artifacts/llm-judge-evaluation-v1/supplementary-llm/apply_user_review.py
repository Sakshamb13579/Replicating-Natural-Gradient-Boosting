"""Apply the user's reviewed ratings to the requested original sheet."""

import csv
import hashlib
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT.parent / "human" / "ratings-blank.csv"
REVIEWED = ROOT / "ratings-ash-reviewed.csv"
BACKUP = ROOT / "ratings-blank-before-user-review.csv"


def parse(data):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
    return reader.fieldnames, list(reader)


def main():
    original = TARGET.read_bytes()
    replacement = REVIEWED.read_bytes()
    fields, blank = parse(original)
    new_fields, ratings = parse(replacement)
    assert fields == new_fields and len(blank) == len(ratings) == 30
    assert len({row["item_id"] for row in ratings}) == 30
    for before, after in zip(blank, ratings):
        assert all(before[key] == after[key] for key in fields[:3])
        assert all(before[key] == "" for key in fields[3:])
        assert all(after[key] in ("0", "1") for key in fields[3:6])
        assert after["rater_id"] == "ash-reviewed-ai-assisted-nonblinded"
        assert "AI-assisted, nonblinded" in after["notes"]
    original_hash = hashlib.sha256(original).hexdigest()
    assert original_hash == "53b8914a09c93570eed30e37ad5e4f515e6072e2d2e467e1649769278497a6ec"
    with BACKUP.open("xb") as handle:
        handle.write(original)
    TARGET.write_bytes(replacement)
    assert TARGET.read_bytes() == replacement
    assert BACKUP.read_bytes() == original
    print(json.dumps({
        "updated": str(TARGET), "rated_items": len(ratings),
        "blank_backup": str(BACKUP), "original_sha256": original_hash,
        "updated_sha256": hashlib.sha256(replacement).hexdigest(),
        "item_fields_and_order_preserved": True,
    }, indent=2))


if __name__ == "__main__":
    main()
