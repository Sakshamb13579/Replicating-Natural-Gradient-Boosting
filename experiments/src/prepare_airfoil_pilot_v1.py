#!/usr/bin/env python3
"""022-NG-P4: Freeze Airfoil Self-Noise processed table + five paired pilot splits.

No model fitting. Outer folds match official NGBoost regression_exp.py style
(RandomState(1) sequential 90/10 permutations) but bounded to five folds.
Inner 80/20 uses random_state=10000+fold_index (017F1 lesson: independent of
model RNG). Same indices reused for natural / ordinary / homoscedastic runs.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "airfoil-self-noise" / "airfoil_self_noise.dat"
AUDIT = ROOT / "data" / "manifests" / "airfoil-self-noise-audit-v1.json"
OUT_DIR = ROOT / "data" / "processed" / "airfoil-self-noise-v1"
CSV_PATH = OUT_DIR / "airfoil-self-noise-v1.csv"
SPLITS_PATH = OUT_DIR / "split-indices.json"
MANIFEST_PATH = ROOT / "data" / "manifests" / "airfoil-self-noise-v1.json"

EXPECTED_RAW_SHA = "74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3"
N_ROWS = 1503
N_FEATURES = 5
N_FOLDS = 5
OUTER_SEED = 1
INNER_SEED_BASE = 10_000
OUTER_TRAIN_FRAC = 9.0 / 10
INNER_VAL_FRAC = 0.2

FEATURE_COLS = [
    "frequency_hz",
    "angle_deg",
    "chord_m",
    "velocity_ms",
    "suction_side_displacement_thickness_m",
]
TARGET_COL = "scaled_sound_pressure_db"
ID_COL = "row_id"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def index_hash(indices: np.ndarray) -> str:
    arr = np.asarray(indices, dtype=np.int64)
    return sha256_bytes(arr.tobytes())


def make_outer_folds(n: int, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Match official regression_exp.py outer 90/10 construction (bounded n_splits)."""
    rng = np.random.RandomState(seed)
    folds = []
    for _ in range(n_splits):
        permutation = rng.choice(range(n), n, replace=False)
        end_train = round(n * OUTER_TRAIN_FRAC)
        train_index = permutation[0:end_train]
        test_index = permutation[end_train:n]
        folds.append((train_index, test_index))
    return folds


def make_inner_split(
    outer_train: np.ndarray, fold_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """80/20 of outer train; absolute row indices; seed independent of model RNG."""
    inner_seed = INNER_SEED_BASE + fold_index
    pos = np.arange(len(outer_train))
    pos_train, pos_val = train_test_split(
        pos, test_size=INNER_VAL_FRAC, random_state=inner_seed
    )
    return outer_train[pos_train], outer_train[pos_val]


def load_raw(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter="\t")
    assert data.ndim == 2 and data.shape == (N_ROWS, N_FEATURES + 1), data.shape
    assert np.isfinite(data).all()
    return data


def write_processed_csv(data: np.ndarray, path: Path) -> str:
    header = ",".join([ID_COL] + FEATURE_COLS + [TARGET_COL])
    lines = [header]
    for i in range(N_ROWS):
        row = data[i]
        # Preserve raw numeric text fidelity via float repr; no scaling.
        cells = [str(i)] + [repr(float(row[j])) for j in range(N_FEATURES + 1)]
        lines.append(",".join(cells))
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return sha256_bytes(text.encode("utf-8"))


def verify_fold(
    fold_idx: int,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    inner_train: np.ndarray,
    inner_val: np.ndarray,
) -> dict:
    ot, oe = set(outer_train.tolist()), set(outer_test.tolist())
    it, iv = set(inner_train.tolist()), set(inner_val.tolist())
    checks = {
        "outer_disjoint": len(ot & oe) == 0,
        "outer_full_coverage": ot | oe == set(range(N_ROWS)),
        "outer_counts": (len(outer_train), len(outer_test)) == (1353, 150),
        "inner_subset_of_outer_train": it | iv == ot and it & iv == set(),
        "inner_counts": (len(inner_train), len(inner_val)) == (1082, 271),
        "inner_disjoint": len(it & iv) == 0,
        "no_test_in_inner": len((it | iv) & oe) == 0,
    }
    assert all(checks.values()), (fold_idx, checks)
    return checks


def build_folds() -> list[dict]:
    folds_raw = make_outer_folds(N_ROWS, N_FOLDS, OUTER_SEED)
    out = []
    for fold_idx, (outer_train, outer_test) in enumerate(folds_raw):
        inner_train, inner_val = make_inner_split(outer_train, fold_idx)
        checks = verify_fold(fold_idx, outer_train, outer_test, inner_train, inner_val)
        out.append(
            {
                "fold_index": fold_idx,
                "outer_construction_seed": OUTER_SEED,
                "inner_random_state": INNER_SEED_BASE + fold_idx,
                "n_outer_train": int(len(outer_train)),
                "n_outer_test": int(len(outer_test)),
                "n_inner_train": int(len(inner_train)),
                "n_inner_val": int(len(inner_val)),
                "outer_train_hash": index_hash(outer_train),
                "outer_test_hash": index_hash(outer_test),
                "inner_train_hash": index_hash(inner_train),
                "inner_val_hash": index_hash(inner_val),
                "outer_train_indices": outer_train.astype(int).tolist(),
                "outer_test_indices": outer_test.astype(int).tolist(),
                "inner_train_indices": inner_train.astype(int).tolist(),
                "inner_val_indices": inner_val.astype(int).tolist(),
                "integrity_checks": checks,
            }
        )
    return out


def regenerate_and_compare(frozen: list[dict]) -> bool:
    again = build_folds()
    for a, b in zip(frozen, again):
        for key in (
            "outer_train_hash",
            "outer_test_hash",
            "inner_train_hash",
            "inner_val_hash",
            "outer_train_indices",
            "outer_test_indices",
            "inner_train_indices",
            "inner_val_indices",
        ):
            if a[key] != b[key]:
                return False
    return True


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_sha = sha256_file(RAW)
    assert raw_sha == EXPECTED_RAW_SHA, (raw_sha, EXPECTED_RAW_SHA)

    data = load_raw(RAW)
    csv_sha = write_processed_csv(data, CSV_PATH)

    # Reload processed and confirm identity with raw numeric matrix.
    reloaded = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1)
    assert reloaded.shape == (N_ROWS, N_FEATURES + 2)
    assert np.array_equal(reloaded[:, 0].astype(int), np.arange(N_ROWS))
    assert np.allclose(reloaded[:, 1:], data)

    folds = build_folds()
    assert regenerate_and_compare(folds)

    splits_doc = {
        "handoff": "022-NG-P4",
        "dataset_slug": "airfoil-self-noise-v1",
        "n_rows": N_ROWS,
        "n_folds": N_FOLDS,
        "outer_policy": (
            "Repeated 90/10 row holdouts via RandomState(1).choice permutations; "
            "same construction as official regression_exp.py, bounded to 5 folds."
        ),
        "outer_seed": OUTER_SEED,
        "inner_policy": (
            "train_test_split(test_size=0.2, random_state=10000+fold_index) on "
            "outer-train positions; independent of model RNG (017F1)."
        ),
        "inner_seed_base": INNER_SEED_BASE,
        "paired_reuse": [
            "natural",
            "ordinary",
            "homoscedastic",
        ],
        "folds": folds,
        "regeneration_deterministic": True,
    }
    splits_text = json.dumps(splits_doc, indent=2) + "\n"
    SPLITS_PATH.write_text(splits_text)
    splits_sha = sha256_bytes(splits_text.encode("utf-8"))

    script_sha = sha256_file(Path(__file__).resolve())
    audit = json.loads(AUDIT.read_text()) if AUDIT.is_file() else {}

    missing = int(np.isnan(data).sum())
    assert missing == 0

    manifest = {
        "status": "frozen",
        "pilot_status": "PILOT_READY",
        "pilot_slot": "NGBoost Pilot 1",
        "manifest_kind": "processed_pilot_split_freeze",
        "dataset_name": "Airfoil Self-Noise",
        "dataset_slug": "airfoil-self-noise-v1",
        "version": 1,
        "handoff_that_froze": "022-NG-P4",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "raw_path": str(RAW.relative_to(ROOT.parent)),
            "raw_sha256": raw_sha,
            "audit_manifest": "ngboost/data/manifests/airfoil-self-noise-audit-v1.json",
            "audit_verdict": audit.get("audit_verdict"),
            "uci_id": 291,
            "doi": "10.24432/C5VW2C",
            "licence": "CC BY 4.0",
        },
        "prep": {
            "script": "ngboost/src/prepare_airfoil_pilot_v1.py",
            "script_sha256": script_sha,
            "scaling": False,
            "imputation": False,
            "rows_preserved": N_ROWS,
            "features_preserved": N_FEATURES,
            "missing_cells": 0,
            "non_finite": 0,
        },
        "processed": {
            "csv_path": "ngboost/data/processed/airfoil-self-noise-v1/airfoil-self-noise-v1.csv",
            "csv_sha256": csv_sha,
            "columns": [ID_COL] + FEATURE_COLS + [TARGET_COL],
            "id_column": ID_COL,
            "feature_columns": FEATURE_COLS,
            "target_column": TARGET_COL,
            "n_rows": N_ROWS,
            "n_features": N_FEATURES,
        },
        "splits": {
            "path": "ngboost/data/processed/airfoil-self-noise-v1/split-indices.json",
            "sha256": splits_sha,
            "n_folds": N_FOLDS,
            "outer_seed": OUTER_SEED,
            "inner_seed_base": INNER_SEED_BASE,
            "expected_counts": {
                "outer_train": 1353,
                "outer_test": 150,
                "inner_train": 1082,
                "inner_val": 271,
            },
            "fold_hashes": [
                {
                    "fold_index": f["fold_index"],
                    "outer_train_hash": f["outer_train_hash"],
                    "outer_test_hash": f["outer_test_hash"],
                    "inner_train_hash": f["inner_train_hash"],
                    "inner_val_hash": f["inner_val_hash"],
                }
                for f in folds
            ],
            "regeneration_deterministic": True,
            "paired_for_variants": ["natural", "ordinary", "homoscedastic"],
        },
        "split_design": {
            "primary": "row_level_paper_style_interpolation",
            "description": (
                "Five repeated outer 90/10 row holdouts; primary estimand is "
                "i.i.d.-row interpolation aligned with official NGBoost script style."
            ),
            "grouped_alternative_documented_not_mixed": {
                "num_condition_groups": 106,
                "definition": (
                    "Unique (angle, chord, velocity, suction thickness); "
                    "frequency is within-condition sweep."
                ),
                "status": "documented_in_audit_only; not used in this pilot freeze",
            },
            "leakage_disclosure": (
                "Row-level splits may place different frequencies from the same "
                "aerodynamic condition in both train and test (interpolation bias)."
            ),
        },
        "protocol_preregistered": {
            "n_estimators_cap": 2000,
            "learning_rate": 0.01,
            "distribution": "Normal",
            "score": "MLE",
            "base_learner": "default_tree_learner",
            "stage_selector": "validation_RMSE",
            "stage_selector_note": (
                "Official-script alignment uses validation RMSE for early stage "
                "selection; paper primary metric is NLL — discrepancy disclosed."
            ),
            "variants_same_folds": ["natural", "ordinary", "homoscedastic"],
            "metrics": {
                "primary": "test_NLL",
                "secondary": [
                    "RMSE",
                    "coverage_50",
                    "coverage_80",
                    "coverage_95",
                    "mean_interval_width_50",
                    "mean_interval_width_80",
                    "mean_interval_width_95",
                ],
            },
            "reporting_rule": (
                "No method is required to win every fold; report paired "
                "fold-wise differences and aggregate direction across five folds."
            ),
        },
        "integrity": {
            "raw_sha_match_audit": raw_sha == EXPECTED_RAW_SHA,
            "all_fold_checks_passed": True,
            "regeneration_match": True,
        },
    }

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")

    summary = {
        "csv_sha256": csv_sha,
        "splits_sha256": splits_sha,
        "script_sha256": script_sha,
        "raw_sha256": raw_sha,
        "fold_counts": [
            {
                "fold": f["fold_index"],
                "outer_train": f["n_outer_train"],
                "outer_test": f["n_outer_test"],
                "inner_train": f["n_inner_train"],
                "inner_val": f["n_inner_val"],
            }
            for f in folds
        ],
        "fold_hashes": manifest["splits"]["fold_hashes"],
        "pilot_status": "PILOT_READY",
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
