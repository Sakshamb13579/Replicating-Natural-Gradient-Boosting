#!/usr/bin/env python3
"""Freeze original-data tables and controlled selector-sensitivity v1 splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent / "upstream/ngboost"
OUTER_SEED = 1
INNER_SEED_BASE = 10_000
VARIANTS = ["natural", "ordinary", "homoscedastic"]

DATASETS = {
    "yacht": {
        "name": "Yacht Hydrodynamics",
        "slug": "yacht-original-selector-sensitivity-v1",
        "source": ROOT / "data/original/yacht_hydrodynamics.data",
        "sha256": "00dfecc0fc01ddd4c90b558a3ac11b246df8ebcfea130724223475a9a67f0ea1",
        "shape": (308, 7),
        "folds": 20,
        "columns": [
            "longitudinal_position",
            "prismatic_coefficient",
            "length_displacement_ratio",
            "beam_draught_ratio",
            "length_beam_ratio",
            "froude_number",
            "residuary_resistance",
        ],
        "loader": "pandas.read_csv(header=None, delim_whitespace=True)",
    },
    "kin8nm": {
        "name": "Kin8nm",
        "slug": "kin8nm-original-selector-sensitivity-v1",
        "source": UPSTREAM / "data/uci/kin8nm.csv",
        "sha256": "7b9bf0301ac936d88122557a151e1ba8f1ebc278fcf46d9f3c6d462debdbc8ad",
        "shape": (8192, 9),
        "folds": 5,
        "columns": ["theta1", "theta2", "theta3", "theta4", "theta5", "theta6", "theta7", "theta8", "y"],
        "loader": "pandas.read_csv(default comma-separated header handling)",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def index_hash(indices: np.ndarray) -> str:
    return sha256_bytes(np.asarray(indices, dtype=np.int64).tobytes())


def write_immutable(path: Path, content: bytes) -> str:
    """Create a frozen file, or verify an existing byte-identical file."""
    if path.exists():
        existing = path.read_bytes()
        if existing != content:
            raise FileExistsError(f"refusing to alter non-identical frozen file: {path}")
        status = "verified_unchanged"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        status = "created"
    print(f"{status}: {path} sha256={sha256_bytes(content)}", flush=True)
    return sha256_bytes(content)


def load_official(dataset: str, spec: dict) -> pd.DataFrame:
    source = spec["source"]
    actual_hash = sha256_file(source)
    if actual_hash != spec["sha256"]:
        raise ValueError(f"{dataset} preserved raw hash mismatch: {actual_hash}")
    if dataset == "yacht":
        # This is intentionally the historical loader call, including whitespace parsing.
        frame = pd.read_csv(source, header=None, delim_whitespace=True)
        frame.columns = spec["columns"]
    else:
        frame = pd.read_csv(source)
        if list(frame.columns) != spec["columns"]:
            raise ValueError(f"{dataset} official CSV columns changed")
    if frame.shape != spec["shape"]:
        raise ValueError(f"{dataset} official-loader shape {frame.shape} != {spec['shape']}")
    values = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{dataset} contains non-finite values")
    return frame


def make_folds(n_rows: int, n_folds: int) -> list[dict]:
    rng = np.random.RandomState(OUTER_SEED)
    folds = []
    universe = set(range(n_rows))
    for fold_index in range(n_folds):
        permutation = rng.choice(range(n_rows), n_rows, replace=False)
        end_train = round(n_rows * 9.0 / 10)
        outer_train = permutation[:end_train]
        outer_test = permutation[end_train:]
        positions = np.arange(len(outer_train))
        inner_train_pos, inner_val_pos = train_test_split(
            positions, test_size=0.2, random_state=INNER_SEED_BASE + fold_index
        )
        inner_train = outer_train[inner_train_pos]
        inner_val = outer_train[inner_val_pos]

        outer_train_set = set(outer_train.tolist())
        outer_test_set = set(outer_test.tolist())
        inner_train_set = set(inner_train.tolist())
        inner_val_set = set(inner_val.tolist())
        checks = {
            "outer_disjoint": not (outer_train_set & outer_test_set),
            "outer_complete": outer_train_set | outer_test_set == universe,
            "inner_disjoint": not (inner_train_set & inner_val_set),
            "inner_partitions_outer_train": inner_train_set | inner_val_set == outer_train_set,
            "no_outer_test_in_inner": not ((inner_train_set | inner_val_set) & outer_test_set),
        }
        if not all(checks.values()):
            raise AssertionError((fold_index, checks))
        fold = {
            "fold_index": fold_index,
            "outer_construction_seed": OUTER_SEED,
            "inner_random_state": INNER_SEED_BASE + fold_index,
            "n_outer_train": len(outer_train),
            "n_outer_test": len(outer_test),
            "n_inner_train": len(inner_train),
            "n_inner_val": len(inner_val),
            "integrity_checks": checks,
        }
        for name, indices in (
            ("outer_train", outer_train),
            ("outer_test", outer_test),
            ("inner_train", inner_train),
            ("inner_val", inner_val),
        ):
            fold[f"{name}_hash"] = index_hash(indices)
            fold[f"{name}_indices"] = indices.astype(int).tolist()
        folds.append(fold)
    return folds


def prepare(dataset: str, spec: dict) -> dict:
    frame = load_official(dataset, spec)
    slug = spec["slug"]
    processed_dir = ROOT / "data/processed" / slug
    csv_path = processed_dir / f"{slug}.csv"
    split_path = processed_dir / "split-indices.json"
    manifest_path = ROOT / "data/manifests" / f"{slug}.json"

    processed = frame.copy()
    processed.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
    csv_bytes = processed.to_csv(index=False, lineterminator="\n").encode("utf-8")
    csv_sha = write_immutable(csv_path, csv_bytes)
    reloaded = pd.read_csv(csv_path)
    if list(reloaded.columns) != list(processed.columns):
        raise AssertionError("processed column mismatch")
    if reloaded["row_id"].tolist() != list(range(len(frame))):
        raise AssertionError("processed row_id mismatch")
    if not np.allclose(reloaded.iloc[:, 1:].to_numpy(), frame.to_numpy(), rtol=0, atol=0):
        raise AssertionError("processed values do not exactly match official-loader values")

    folds = make_folds(len(frame), spec["folds"])
    splits = {
        "schema_version": 1,
        "status": "frozen",
        "dataset_slug": slug,
        "n_rows": len(frame),
        "n_folds": spec["folds"],
        "outer_policy": "Sequential repeated 90/10 holdouts from RandomState(1).choice without replacement; official regression_exp.py construction.",
        "outer_seed": OUTER_SEED,
        "inner_policy": "train_test_split(test_size=0.2, random_state=10000+fold_index) on outer-train positions; stored as absolute row indices.",
        "inner_seed_base": INNER_SEED_BASE,
        "paired_reuse": VARIANTS,
        "folds": folds,
    }
    split_bytes = (json.dumps(splits, indent=2) + "\n").encode("utf-8")
    split_sha = write_immutable(split_path, split_bytes)

    manifest = {
        "schema_version": 1,
        "status": "frozen",
        "manifest_kind": "original_data_selector_sensitivity_v1",
        "dataset": dataset,
        "dataset_name": spec["name"],
        "dataset_slug": slug,
        "source": {
            "preserved_raw_path": str(spec["source"]),
            "raw_sha256": spec["sha256"],
            "official_loader": spec["loader"],
            "official_loaded_shape": list(spec["shape"]),
            "official_target_rule": "final column",
            "official_feature_rule": "all columns preceding final column",
        },
        "processed": {
            "csv_path": str(csv_path.relative_to(ROOT)),
            "csv_sha256": csv_sha,
            "columns": list(processed.columns),
            "id_column": "row_id",
            "feature_columns": spec["columns"][:-1],
            "target_column": spec["columns"][-1],
            "n_rows": len(frame),
            "n_features": len(spec["columns"]) - 1,
            "scaling": False,
            "imputation": False,
            "feature_transformation": False,
        },
        "splits": {
            "path": str(split_path.relative_to(ROOT)),
            "sha256": split_sha,
            "n_folds": spec["folds"],
            "outer_seed": OUTER_SEED,
            "inner_seed_base": INNER_SEED_BASE,
            "fold_hashes": [
                {key: fold[key] for key in ("fold_index", "outer_train_hash", "outer_test_hash", "inner_train_hash", "inner_val_hash")}
                for fold in folds
            ],
        },
        "protocol": "reports/original-selector-sensitivity-protocol-v1.md",
        "upstream_commit": "672034041fc05d9d4b7a80c1e6163ea102579752",
        "immutability": "Preparation refuses to replace any non-identical existing CSV, split manifest, or dataset manifest.",
    }
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    manifest_sha = write_immutable(manifest_path, manifest_bytes)
    return {
        "dataset": dataset,
        "slug": slug,
        "raw_sha256": spec["sha256"],
        "official_shape": list(frame.shape),
        "processed_csv_sha256": csv_sha,
        "split_manifest_sha256": split_sha,
        "dataset_manifest_sha256": manifest_sha,
        "n_folds": len(folds),
        "fold_counts": {key: folds[0][key] for key in ("n_outer_train", "n_outer_test", "n_inner_train", "n_inner_val")},
    }


def main() -> int:
    summaries = [prepare(dataset, spec) for dataset, spec in DATASETS.items()]
    print(json.dumps({"status": "frozen_or_verified", "datasets": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
