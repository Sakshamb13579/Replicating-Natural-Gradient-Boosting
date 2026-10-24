#!/usr/bin/env python3
"""Download the five UCI archives into data/raw and verify the audited checksums.

Each ``data/manifests/<dataset>-audit-v1.json`` records the archive URL and the
SHA-256 of the archive and of every extracted file used by the preparation
scripts. Only the prepared CSVs are committed; this script recreates the raw
inputs those CSVs were built from.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["airfoil-self-noise", "forest-fires", "parkinsons-telemonitoring",
            "real-estate-valuation", "superconductivity"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(dataset: str) -> None:
    audit = json.loads((ROOT / "data/manifests" / f"{dataset}-audit-v1.json").read_text())
    download = audit["download"]
    files = {Path(entry["path"]).name: entry["sha256"] for entry in download["files"]}
    archive_name = next(name for name in files if name.endswith(".zip"))
    raw_dir = ROOT / "data/raw" / dataset
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / archive_name
    if not archive.exists():
        urllib.request.urlretrieve(download["url"], archive)
    if sha256_file(archive) != files[archive_name]:
        raise ValueError(f"{archive}: checksum differs from the audit; UCI may have changed the archive")
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            name = Path(member).name
            if name in files:
                (raw_dir / name).write_bytes(bundle.read(member))
    for name, expected in files.items():
        if sha256_file(raw_dir / name) != expected:
            raise ValueError(f"{raw_dir / name}: checksum differs from the audit")
    print(f"{dataset}: {len(files)} files verified")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", help=f"default: all of {', '.join(DATASETS)}")
    datasets = parser.parse_args().datasets or DATASETS
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(unknown)}")
    for dataset in datasets:
        fetch(dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
