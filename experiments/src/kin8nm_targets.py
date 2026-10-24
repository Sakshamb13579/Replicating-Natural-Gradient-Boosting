#!/usr/bin/env python3
"""Remove or restore the Kin8nm target column in saved selector predictions.

Kin8nm is cited rather than redistributed, so the published prediction files
omit ``y_true``. ``restore`` rejoins it by ``row_id`` (the 0-based row index)
from the CSV distributed with NGBoost and checks every restored file against
the run's original ``checksums.txt``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts/original-selector-sensitivity-v1/kin8nm"
UPSTREAM_CSV = ROOT.parent / "upstream/ngboost/data/uci/kin8nm.csv"
UPSTREAM_SHA256 = "7b9bf0301ac936d88122557a151e1ba8f1ebc278fcf46d9f3c6d462debdbc8ad"
TARGET_POSITION = 3  # fold_index, selector, row_id, y_true, ...


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> tuple[list[list[str]], str]:
    text = path.read_bytes().decode()
    newline = "\r\n" if "\r\n" in text else "\n"
    return [line.split(",") for line in text.split(newline) if line], newline


def write_rows(path: Path, rows: list[list[str]], newline: str) -> None:
    path.write_bytes("".join(",".join(r) + newline for r in rows).encode())


def run_dirs(runs: Path) -> list[Path]:
    return sorted(path.parent for path in runs.glob("*/*/predictions.csv"))


def strip(runs: Path) -> None:
    for directory in run_dirs(runs):
        csv_path = directory / "predictions.csv"
        rows, newline = read_rows(csv_path)
        if rows[0][TARGET_POSITION] == "y_true":
            write_rows(csv_path, [r[:TARGET_POSITION] + r[TARGET_POSITION + 1:] for r in rows], newline)
        jsonl_path = directory / "predictions.jsonl"
        records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
        jsonl_path.write_text("".join(
            json.dumps({k: v for k, v in record.items() if k != "y_true"}) + "\n" for record in records))
        print(f"stripped {directory.name}")


def restore(runs: Path, upstream_csv: Path) -> None:
    if sha256_file(upstream_csv) != UPSTREAM_SHA256:
        raise ValueError(f"unexpected Kin8nm checksum: {upstream_csv}")
    target = pd.read_csv(upstream_csv, float_precision="round_trip").iloc[:, -1].to_numpy()
    for directory in run_dirs(runs):
        csv_path = directory / "predictions.csv"
        rows, newline = read_rows(csv_path)
        if rows[0][TARGET_POSITION] != "y_true":
            row_id = rows[0].index("row_id")
            rows[0].insert(TARGET_POSITION, "y_true")
            for r in rows[1:]:
                r.insert(TARGET_POSITION, repr(float(target[int(r[row_id])])))
            write_rows(csv_path, rows, newline)
        jsonl_path = directory / "predictions.jsonl"
        lines = []
        for line in jsonl_path.read_text().splitlines():
            record = json.loads(line)
            if "y_true" not in record:
                items = list(record.items())
                items.insert(TARGET_POSITION, ("y_true", float(target[record["row_id"]])))
                record = dict(items)
            lines.append(json.dumps(record) + "\n")
        jsonl_path.write_text("".join(lines))
        expected = dict(reversed(line.split()) for line in (directory / "checksums.txt").read_text().splitlines())
        for name in ("predictions.csv", "predictions.jsonl"):
            if sha256_file(directory / name) != expected[name]:
                raise ValueError(f"restored file does not match recorded checksum: {directory / name}")
        print(f"restored and verified {directory.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["strip", "restore"])
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--upstream-csv", type=Path, default=UPSTREAM_CSV)
    args = parser.parse_args()
    if args.action == "strip":
        strip(args.runs)
    else:
        restore(args.runs, args.upstream_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
