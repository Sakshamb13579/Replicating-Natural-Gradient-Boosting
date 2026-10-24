#!/usr/bin/env python3
"""Prepare and freeze v1 external NGBoost datasets without model fitting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "manifests"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTER_SEED = 1
INNER_SEED_BASE = 10_000
SUPERCONDUCTIVITY_SUBSET_SEED = 1
SUPERCONDUCTIVITY_SUBSET_ROWS = 5_000
VARIANTS = ["natural", "ordinary", "homoscedastic"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_hash(indices: np.ndarray) -> str:
    return sha256_bytes(np.asarray(indices, dtype=np.int64).tobytes())


def immutable_write(path: Path, content: bytes) -> str:
    """Create an artifact, accept an identical artifact, and never replace one."""
    expected = sha256_bytes(content)
    if path.exists():
        actual = sha256_file(path)
        if actual != expected:
            raise FileExistsError(
                f"refusing to overwrite differing artifact {path}: {actual} != {expected}"
            )
        return actual
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return expected


def audit_raw(dataset: str, raw_name: str) -> tuple[Path, dict, str]:
    audit_path = MANIFEST_DIR / f"{dataset}-audit-v1.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    entries = audit["download"]["files"]
    matches = [entry for entry in entries if Path(entry["path"]).name == raw_name]
    if len(matches) != 1:
        raise ValueError(f"audit does not identify exactly one {raw_name}")
    raw_path = ROOT / "data" / "raw" / dataset / raw_name
    actual = sha256_file(raw_path)
    expected = matches[0]["sha256"]
    if actual != expected:
        raise ValueError(f"raw hash mismatch for {raw_path}: {actual} != {expected}")
    return raw_path, audit, actual


def repeated_row_folds(n_rows: int, n_folds: int = 20) -> list[dict]:
    rng = np.random.RandomState(OUTER_SEED)
    folds = []
    for fold_index in range(n_folds):
        permutation = rng.choice(range(n_rows), n_rows, replace=False)
        boundary = round(n_rows * 0.9)
        outer_train = permutation[:boundary]
        outer_test = permutation[boundary:]
        positions = np.arange(len(outer_train))
        train_pos, val_pos = train_test_split(
            positions, test_size=0.2, random_state=INNER_SEED_BASE + fold_index
        )
        folds.append(
            make_fold(
                fold_index,
                outer_train,
                outer_test,
                outer_train[train_pos],
                outer_train[val_pos],
                n_rows,
                None,
            )
        )
    return folds


def make_fold(
    fold_index: int,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    inner_train: np.ndarray,
    inner_val: np.ndarray,
    n_rows: int,
    groups: np.ndarray | None,
) -> dict:
    arrays = [np.asarray(x, dtype=np.int64) for x in (outer_train, outer_test, inner_train, inner_val)]
    outer_train, outer_test, inner_train, inner_val = arrays
    ot, oe, it, iv = map(lambda x: set(x.tolist()), arrays)
    checks = {
        "outer_disjoint": not bool(ot & oe),
        "outer_full_coverage": ot | oe == set(range(n_rows)),
        "inner_partitions_outer_train": it | iv == ot and not bool(it & iv),
        "no_test_in_inner": not bool((it | iv) & oe),
        "indices_in_range": all(
            bool((a >= 0).all() and (a < n_rows).all()) for a in arrays
        ),
    }
    if groups is not None:
        outer_train_groups = set(groups[outer_train].tolist())
        outer_test_groups = set(groups[outer_test].tolist())
        inner_train_groups = set(groups[inner_train].tolist())
        inner_val_groups = set(groups[inner_val].tolist())
        checks.update(
            {
                "outer_groups_disjoint": not bool(outer_train_groups & outer_test_groups),
                "inner_groups_disjoint": not bool(inner_train_groups & inner_val_groups),
                "test_groups_absent_from_inner": not bool(
                    outer_test_groups & (inner_train_groups | inner_val_groups)
                ),
            }
        )
    if not all(checks.values()):
        raise AssertionError((fold_index, checks))
    result = {
        "fold_index": fold_index,
        "outer_construction_seed": OUTER_SEED,
        "inner_random_state": INNER_SEED_BASE + fold_index,
        "n_outer_train": len(outer_train),
        "n_outer_test": len(outer_test),
        "n_inner_train": len(inner_train),
        "n_inner_val": len(inner_val),
        "integrity_checks": checks,
    }
    for name, values in (
        ("outer_train", outer_train),
        ("outer_test", outer_test),
        ("inner_train", inner_train),
        ("inner_val", inner_val),
    ):
        result[f"{name}_hash"] = index_hash(values)
        result[f"{name}_indices"] = values.tolist()
    if groups is not None:
        result["outer_train_groups"] = sorted(int(x) for x in set(groups[outer_train]))
        result["outer_test_groups"] = sorted(int(x) for x in set(groups[outer_test]))
        result["inner_train_groups"] = sorted(int(x) for x in set(groups[inner_train]))
        result["inner_val_groups"] = sorted(int(x) for x in set(groups[inner_val]))
    return result


def subject_folds(groups: np.ndarray) -> list[dict]:
    subjects = np.asarray(sorted(set(groups.tolist())), dtype=np.int64)
    outer = KFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
    folds = []
    for fold_index, (train_subject_pos, test_subject_pos) in enumerate(outer.split(subjects)):
        train_subjects = subjects[train_subject_pos]
        test_subjects = subjects[test_subject_pos]
        inner_train_subjects, inner_val_subjects = train_test_split(
            train_subjects,
            test_size=0.2,
            random_state=INNER_SEED_BASE + fold_index,
        )
        outer_train = np.flatnonzero(np.isin(groups, train_subjects))
        outer_test = np.flatnonzero(np.isin(groups, test_subjects))
        inner_train = np.flatnonzero(np.isin(groups, inner_train_subjects))
        inner_val = np.flatnonzero(np.isin(groups, inner_val_subjects))
        folds.append(
            make_fold(
                fold_index,
                outer_train,
                outer_test,
                inner_train,
                inner_val,
                len(groups),
                groups,
            )
        )
    test_subjects = [subject for fold in folds for subject in fold["outer_test_groups"]]
    if sorted(test_subjects) != subjects.tolist():
        raise AssertionError("each subject must occur in exactly one outer test fold")
    return folds


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def freeze_dataset(
    dataset: str,
    dataset_name: str,
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    eval_columns: list[str],
    folds: list[dict],
    raw_path: Path,
    raw_sha: str,
    audit: dict,
    split_policy: str,
    preprocessing: dict,
    limitations: list[str],
) -> dict:
    slug = f"{dataset}-v1"
    out_dir = PROCESSED_DIR / slug
    csv_path = out_dir / f"{slug}.csv"
    split_path = out_dir / "split-indices.json"
    manifest_path = MANIFEST_DIR / f"{slug}.json"

    expected_columns = ["row_id"] + eval_columns + feature_columns + [target_column]
    if list(frame.columns) != expected_columns:
        raise AssertionError((list(frame.columns), expected_columns))
    if frame["row_id"].tolist() != list(range(len(frame))):
        raise AssertionError("row_id must equal immutable zero-based row position")
    if not np.isfinite(frame[feature_columns + [target_column]].to_numpy(dtype=float)).all():
        raise AssertionError("model columns contain missing or non-finite values")

    csv_sha = immutable_write(csv_path, csv_bytes(frame))
    split_doc = {
        "schema_version": 1,
        "dataset_slug": slug,
        "n_rows": len(frame),
        "n_folds": len(folds),
        "outer_seed": OUTER_SEED,
        "inner_seed_base": INNER_SEED_BASE,
        "outer_policy": split_policy,
        "paired_reuse": VARIANTS,
        "folds": folds,
    }
    split_bytes = (json.dumps(split_doc, indent=2) + "\n").encode("utf-8")
    split_sha = immutable_write(split_path, split_bytes)

    manifest = {
        "status": "frozen",
        "manifest_kind": "external_processed_split_freeze",
        "schema_version": 1,
        "dataset_name": dataset_name,
        "dataset_slug": slug,
        "created_at": audit.get("created_at"),
        "source": {
            "raw_path": str(raw_path.relative_to(ROOT)),
            "raw_sha256": raw_sha,
            "audit_manifest": str((MANIFEST_DIR / f"{dataset}-audit-v1.json").relative_to(ROOT)),
            "audit_recorded_sha256_verified": True,
            "uci_id": audit.get("uci_id"),
            "doi": audit.get("doi"),
            "licence": audit.get("licence"),
        },
        "processed": {
            "csv_path": str(csv_path.relative_to(ROOT)),
            "csv_sha256": csv_sha,
            "columns": expected_columns,
            "id_column": "row_id",
            "evaluation_columns": eval_columns,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "n_rows": len(frame),
            "n_features": len(feature_columns),
        },
        "splits": {
            "path": str(split_path.relative_to(ROOT)),
            "sha256": split_sha,
            "n_folds": len(folds),
            "outer_seed": OUTER_SEED,
            "inner_seed_base": INNER_SEED_BASE,
            "paired_for_variants": VARIANTS,
        },
        "split_policy": split_policy,
        "preprocessing": preprocessing,
        "limitations": limitations,
        "protocol": {
            "stage_selector": "validation_NLL",
            "n_estimators_cap": 2000,
            "learning_rate": 0.01,
            "score": "MLE",
            "base_learner": "default_tree_learner",
        },
        "integrity": {
            "raw_sha_match_audit": True,
            "all_fold_checks_passed": all(
                all(fold["integrity_checks"].values()) for fold in folds
            ),
            "processed_csv_sha256": csv_sha,
            "split_indices_sha256": split_sha,
        },
    }
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    manifest_sha = immutable_write(manifest_path, manifest_bytes)
    return {
        "dataset": dataset,
        "rows": len(frame),
        "features": len(feature_columns),
        "folds": len(folds),
        "raw_sha256": raw_sha,
        "csv_sha256": csv_sha,
        "splits_sha256": split_sha,
        "manifest_sha256": manifest_sha,
    }


def prepare_real_estate() -> dict:
    dataset = "real-estate-valuation"
    raw, audit, raw_sha = audit_raw(dataset, "Real estate valuation data set.xlsx")
    workbook = pd.ExcelFile(raw)
    if len(workbook.sheet_names) != 1:
        raise AssertionError(f"expected one worksheet, found {workbook.sheet_names}")
    source = pd.read_excel(workbook, sheet_name=workbook.sheet_names[0])
    source.columns = [str(column).strip() for column in source.columns]
    source = source.rename(columns={source.columns[-1]: "Y"})
    features = [
        "X1 transaction date",
        "X2 house age",
        "X3 distance to the nearest MRT station",
        "X4 number of convenience stores",
        "X5 latitude",
        "X6 longitude",
    ]
    if list(source.columns[:7]) != ["No"] + features or len(source) != 414:
        raise AssertionError((source.columns.tolist(), source.shape))
    frame = source[features + ["Y"]].copy()
    frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
    return freeze_dataset(
        dataset,
        "Real Estate Valuation",
        frame,
        features,
        "Y",
        [],
        repeated_row_folds(len(frame)),
        raw,
        raw_sha,
        audit,
        "20 repeated 90/10 row holdouts via sequential RandomState(1).choice; inner 80/20 rows with seed 10000+fold",
        {
            "worksheet": workbook.sheet_names[0],
            "audit_sheet_label_note": "Audit calls the sole worksheet sheet1; workbook label is preserved here",
            "dropped_columns": ["No"],
            "scaling": False,
            "imputation": False,
        },
        ["Small test sets make coverage high variance", "Row splits mix time and repeated coordinates"],
    )


def prepare_parkinsons() -> dict:
    dataset = "parkinsons-telemonitoring"
    raw, audit, raw_sha = audit_raw(dataset, "parkinsons_updrs.data")
    source = pd.read_csv(raw)
    features = list(audit["column_roles"]["allowed_predictors"])
    if len(features) != 19 or set(features) & {"subject#", "motor_UPDRS", "total_UPDRS"}:
        raise AssertionError("audit predictor allowlist is invalid")
    if len(source) != 5875 or set(source["subject#"].unique()) != set(range(1, 43)):
        raise AssertionError(source.shape)
    frame = source[["subject#"] + features + ["motor_UPDRS"]].copy()
    frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
    groups = frame["subject#"].to_numpy(dtype=np.int64)
    return freeze_dataset(
        dataset,
        "Parkinsons Telemonitoring",
        frame,
        features,
        "motor_UPDRS",
        ["subject#"],
        subject_folds(groups),
        raw,
        raw_sha,
        audit,
        "Shuffled 5-fold subject partition seed 1; subject-disjoint inner validation from outer-training subjects with seed 10000+fold",
        {
            "allowed_predictors_exactly": features,
            "excluded_from_features": ["subject#", "total_UPDRS"],
            "evaluation_group_column": "subject#",
            "scaling": False,
            "imputation": False,
        },
        ["Only 42 independent subjects", "Recording-level metrics are not independent-patient evidence"],
    )


def prepare_forest_fires() -> dict:
    dataset = "forest-fires"
    raw, audit, raw_sha = audit_raw(dataset, "forestfires.csv")
    source = pd.read_csv(raw)
    months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    days = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    if len(source) != 517 or int(source.duplicated().sum()) != 4:
        raise AssertionError((source.shape, source.duplicated().sum()))
    if not set(source["month"]).issubset(months) or not set(source["day"]).issubset(days):
        raise AssertionError("unexpected month/day category")
    numeric = ["X", "Y", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]
    encoded = source[numeric].copy()
    for category in months:
        encoded[f"month_{category}"] = (source["month"] == category).astype(np.int8)
    for category in days:
        encoded[f"day_{category}"] = (source["day"] == category).astype(np.int8)
    features = encoded.columns.tolist()
    encoded["log1p_area"] = np.log1p(source["area"].to_numpy(dtype=float))
    encoded.insert(0, "row_id", np.arange(len(encoded), dtype=np.int64))
    return freeze_dataset(
        dataset,
        "Forest Fires",
        encoded,
        features,
        "log1p_area",
        [],
        repeated_row_folds(len(encoded)),
        raw,
        raw_sha,
        audit,
        "20 repeated 90/10 row holdouts via sequential RandomState(1).choice; inner 80/20 rows with seed 10000+fold",
        {
            "rows_retained": 517,
            "exact_duplicate_rows_retained": 4,
            "target_transform": "log1p(area)",
            "one_hot_categories": {"month": months, "day": days},
            "numeric_features": numeric,
            "scaling": False,
            "imputation": False,
        },
        ["Four exact duplicate rows retained", "Row splits can overlap spatial cells", "Normal likelihood remains approximate under zero inflation"],
    )


def prepare_superconductivity() -> dict:
    dataset = "superconductivity"
    raw, audit, raw_sha = audit_raw(dataset, "train.csv")
    source = pd.read_csv(raw)
    if source.shape != (21_263, 82) or source.columns[-1] != "critical_temp":
        raise AssertionError((source.shape, source.columns[-1]))
    features = source.columns[:-1].tolist()
    rng = np.random.RandomState(SUPERCONDUCTIVITY_SUBSET_SEED)
    source_indices = np.sort(
        rng.choice(
            len(source), SUPERCONDUCTIVITY_SUBSET_ROWS, replace=False
        ).astype(np.int64)
    )
    selected = source.iloc[source_indices].reset_index(drop=True)
    frame = selected[features + ["critical_temp"]].copy()
    frame.insert(0, "source_row_id", source_indices)
    frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
    duplicate_full_rows = int(selected.duplicated().sum())
    duplicate_feature_rows = int(selected.duplicated(subset=features, keep=False).sum())
    return freeze_dataset(
        dataset,
        "Superconductivity Data (deterministic 5,000-row pilot subset)",
        frame,
        features,
        "critical_temp",
        ["source_row_id"],
        repeated_row_folds(len(frame), n_folds=5),
        raw,
        raw_sha,
        audit,
        "Deterministic 5000-row subset selected before split generation with RandomState(1); five repeated 90/10 row holdouts via sequential RandomState(1).choice; inner 80/20 rows with seed 10000+fold",
        {
            "source_table_rows": len(source),
            "subset_rows": SUPERCONDUCTIVITY_SUBSET_ROWS,
            "subset_seed": SUPERCONDUCTIVITY_SUBSET_SEED,
            "subset_selection": "RandomState(1).choice without replacement, then sorted by source row",
            "source_row_id_preserved": True,
            "primary_source_only": "train.csv",
            "unique_m_used": False,
            "duplicate_policy": "Retain all rows selected by the predeclared random subset; no outcome-dependent filtering",
            "selected_full_duplicate_rows": duplicate_full_rows,
            "selected_rows_in_repeated_feature_tuples": duplicate_feature_rows,
            "scaling": False,
            "imputation": False,
            "scope_gate": {
                "benchmark": "artifacts/superconductivity/benchmark-full-data-est25-a01.json",
                "full_data_projected_hours_5_folds_3_variants": 9.52,
                "decision": "Use proposal-authorized deterministic 5000-row subset contingency",
            },
        },
        [
            "Pilot uses a deterministic 5000-row subset rather than all 21263 rows because the full five-fold three-variant run projected to about 9.5 hours",
            "Aggregated feature tuples can repeat and can map to differing critical temperatures",
            "Subset results are pilot evidence and do not estimate full-dataset performance exactly",
        ],
    )


def main() -> int:
    results = [
        prepare_real_estate(),
        prepare_parkinsons(),
        prepare_forest_fires(),
        prepare_superconductivity(),
    ]
    print(json.dumps({"status": "prepared_or_identical", "datasets": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
