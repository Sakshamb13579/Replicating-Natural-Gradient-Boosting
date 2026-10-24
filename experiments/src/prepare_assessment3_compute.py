#!/usr/bin/env python3
"""Freeze Assignment 3 inputs and paired splits without fitting any models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

import generate_heteroscedastic_v1 as generator
from prepare_external_datasets_v1 import (
    audit_raw, csv_bytes, immutable_write, make_fold, repeated_row_folds, sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = ROOT / "artifacts" / "assessment3-compute-v1"
VARIANTS = ["natural", "ordinary", "homoscedastic"]


def freeze_json(path: Path, obj: object) -> None:
    immutable_write(path, (json.dumps(obj, indent=2, allow_nan=False) + "\n").encode())


def collision_audit(frame: pd.DataFrame, features: list[str], target: str, folds: list[dict]) -> dict:
    feature_ids = pd.util.hash_pandas_object(frame[features], index=False).to_numpy()
    exact_ids = pd.util.hash_pandas_object(frame[features + [target]], index=False).to_numpy()
    rows = []
    for fold in folds:
        item = {"fold_index": fold["fold_index"]}
        for label, left, right in (
            ("outer", "outer_train", "outer_test"), ("inner", "inner_train", "inner_val")
        ):
            for kind, ids in (("feature", feature_ids), ("exact", exact_ids)):
                a = ids[fold[left + "_indices"]]
                b = ids[fold[right + "_indices"]]
                shared = np.intersect1d(a, b)
                item[f"{label}_{kind}_shared_tuples"] = len(shared)
                item[f"{label}_{kind}_right_rows_with_left_match"] = int(np.isin(b, a).sum())
        rows.append(item)
    return {
        "method": "pandas 64-bit row hashes; counts checked with DataFrame.duplicated for totals",
        "exact_duplicate_extra_rows": int(frame.duplicated(features + [target]).sum()),
        "repeated_feature_extra_rows": int(frame.duplicated(features).sum()),
        "feature_tuples_with_multiple_targets": int(
            (frame.groupby(features, dropna=False)[target].nunique() > 1).sum()
        ),
        "per_fold": rows,
    }


def freeze_dataset(batch: Path, name: str, frame: pd.DataFrame, features: list[str],
                   target: str, folds: list[dict], source: dict, **extra) -> dict:
    if not np.isfinite(frame[features + [target]].to_numpy(dtype=float)).all():
        raise ValueError(f"nonfinite model data: {name}")
    if frame.row_id.tolist() != list(range(len(frame))):
        raise ValueError(f"unexpected row IDs: {name}")
    directory = batch / "inputs" / name
    csv_path = directory / "data.csv"
    split_path = directory / "splits.json"
    immutable_write(csv_path, csv_bytes(frame))
    freeze_json(split_path, {"folds": folds})
    manifest = {
        "status": "frozen", "dataset": name, "rows": len(frame),
        "features": features, "target": target, "columns": frame.columns.tolist(),
        "csv_sha256": sha256_file(csv_path), "splits_sha256": sha256_file(split_path),
        "source": source, **extra,
    }
    freeze_json(directory / "manifest.json", manifest)
    return {"dataset": name, "manifest_sha256": sha256_file(directory / "manifest.json"),
            "folds": len(folds), "rows": len(frame)}


def prepare_superconductivity(batch: Path) -> dict:
    path, _, raw_sha = audit_raw("superconductivity", "train.csv")
    frame = pd.read_csv(path)
    if frame.shape != (21263, 82) or frame.columns[-1] != "critical_temp":
        raise ValueError(frame.shape)
    features = frame.columns[:-1].tolist()
    frame.insert(0, "row_id", np.arange(len(frame)))
    folds = repeated_row_folds(len(frame), n_folds=5)
    return freeze_dataset(
        batch, "superconductivity-full", frame, features, "critical_temp", folds,
        {"raw_path": str(path.relative_to(ROOT)), "raw_sha256": raw_sha},
        split_policy="Five repeated 90/10 row holdouts, sequential RandomState(1).choice; inner 80/20 seed 10000+fold",
        duplicate_policy="Retain every source row, consistent with subset pilot; disclose collisions",
        collision_audit=collision_audit(frame, features, "critical_temp", folds),
        limitations=["Repeated holdouts overlap and are not independent replicates",
                     "Predictor tuples and exact duplicate rows may cross splits",
                     "Full-data primary result; 5000-row pilot remains separate"],
    )


def prepare_airfoil(batch: Path) -> dict:
    old_path = ROOT / "data/manifests/airfoil-self-noise-v1.json"
    old = json.loads(old_path.read_text())
    csv_path = ROOT / "data/processed/airfoil-self-noise-v1/airfoil-self-noise-v1.csv"
    if sha256_file(csv_path) != old["processed"]["csv_sha256"]:
        raise ValueError("Airfoil frozen source hash mismatch")
    raw, _, raw_sha = audit_raw("airfoil-self-noise", "airfoil_self_noise.dat")
    frame = pd.read_csv(csv_path)
    features = old["processed"]["feature_columns"]
    target = old["processed"]["target_column"]
    condition = [col for col in features if col != "frequency_hz"]
    group_index = pd.MultiIndex.from_frame(frame[condition])
    groups, _ = pd.factorize(group_index, sort=True)
    unique = np.unique(groups)
    if len(frame) != 1503 or len(unique) != 106:
        raise ValueError((len(frame), len(unique)))
    frame.insert(1, "condition_id", groups)
    folds = []
    for index, (a, b) in enumerate(KFold(5, shuffle=True, random_state=1).split(unique)):
        train_groups, test_groups = unique[a], unique[b]
        inner_train_groups, val_groups = train_test_split(
            train_groups, test_size=0.2, random_state=10000 + index
        )
        arrays = [np.flatnonzero(np.isin(groups, ids)) for ids in
                  (train_groups, test_groups, inner_train_groups, val_groups)]
        folds.append(make_fold(index, *arrays, len(frame), groups))
    if sorted(i for fold in folds for i in fold["outer_test_indices"]) != list(range(len(frame))):
        raise ValueError("each Airfoil row must be tested once")
    return freeze_dataset(
        batch, "airfoil-grouped", frame, features, target, folds,
        {"processed_manifest_sha256": sha256_file(old_path), "raw_sha256": raw_sha,
         "raw_path": str(raw.relative_to(ROOT))},
        group_column="condition_id", group_definition=condition, n_groups=len(unique),
        split_policy="Five shuffled folds of unique conditions, seed 1; group-disjoint inner 80/20 seed 10000+fold",
        limitations=["Held-out condition combinations, not guaranteed unseen feature levels",
                     "Different estimand and fold sizes from repeated row splits; no paired row/group fold test"],
    )


def prepare_synthetic(batch: Path, replicate: int) -> dict:
    base = generator.generate_base(
        np.random.default_rng(replicate), np.random.default_rng(1000 + replicate)
    )
    expanded = generator.expand_regimes(base)
    summaries = []
    for regime_index, regime in enumerate(generator.REGIMES):
        mask = expanded["regime"] == regime
        frame = pd.DataFrame({key: value[mask] for key, value in expanded.items()})
        train = np.flatnonzero(frame.split.to_numpy() == "train")
        val = np.flatnonzero(frame.split.to_numpy() == "validation")
        test = np.flatnonzero(frame.split.to_numpy() == "test")
        fold = make_fold(0, np.concatenate([train, val]), test, train, val, len(frame), None)
        # Matched RNG across A/B/C and all variants in each independent replicate.
        fold["stage_model_seed"] = 20000 + 100 * replicate
        fold["full_model_seed"] = 30000 + 100 * replicate
        if not np.allclose(frame.y, frame.true_mu + frame.true_sigma * frame.noise_z, rtol=0, atol=1e-14):
            raise ValueError("synthetic formula mismatch")
        summaries.append(freeze_dataset(
            batch, f"synthetic-r{replicate:02d}-{regime}", frame, ["x1", "x2"], "y", [fold],
            {"generator_sha256": sha256_file(Path(generator.__file__)),
             "data_seed": replicate, "split_seed": 1000 + replicate,
             "rng": "PCG64", "numpy": np.__version__},
            replicate=replicate, regime=regime, regime_index=regime_index,
            evaluation_only=["true_mu", "true_sigma", "noise_z", "regime", "split"],
            split_policy="2400 train / 800 validation / 800 test; paired A/B/C inputs, noise, splits",
            limitations=["Known Gaussian synthetic law; does not establish real-data superiority",
                         "Fresh data and split per replicate; v1 excluded from replicate aggregate"],
        ))
    return {"replicate": replicate, "datasets": summaries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    args = parser.parse_args()
    batch = args.batch.resolve()
    allowed = ROOT.resolve() / "artifacts"
    if not batch.is_relative_to(allowed):
        raise ValueError("batch must be under this person's NGBoost artifacts tree")
    # Freeze scientific decisions before any fit or outcome is inspected.
    protocol = {
        "version": 1, "variants": VARIANTS, "n_estimators": 2000,
        "learning_rate": 0.01, "stage_selector": "validation_NLL",
        "homoscedastic_sigma": "RMS of inner-validation residuals at each stage; retain selected sigma after refit",
        "refit": "Fit all 2000 stages on outer train; predict using selected prefix, matching existing runners",
        "numerical_threads_per_process": 1, "max_concurrent_processes": 4,
        "synthetic_replicates": list(range(1, 11)),
        "synthetic_model_seed_policy": "Same model seeds across A/B/C and variants within replicate; stage 20000+100*r, refit 30000+100*r",
        "external_model_seed_policy": "stage 20000+fold, refit 30000+fold",
        "synthetic_reporting": [
            "Per regime: paired NLL differences, mean, sample SD, SE over 10 fresh replicates, and win/tie/loss counts",
            "Scale MAE, Pearson/Spearman where nonconstant, marginal and conditional interval coverage and widths",
            "Descriptive replicate support for original H1/H2/H4; H3 remains a constant-noise control with no required winner",
            "Retain v1 hypotheses and results without rewriting or pooling; no new equivalence threshold",
        ],
        "aggregation": "Unweighted means of fold metrics; paired comparisons only within identical splits; no iid interpretation of overlapping holdouts",
    }
    freeze_json(batch / "protocol.json", protocol)
    external = [prepare_superconductivity(batch), prepare_airfoil(batch)]
    synthetic = [prepare_synthetic(batch, r) for r in range(1, 11)]
    datasets = external + [d for rep in synthetic for d in rep["datasets"]]
    # Three long tasks occupy at most three slots; the fourth can service auxiliary work.
    tasks = [{"dataset": d["dataset"], "variant": v} for d in datasets for v in VARIANTS]
    source_files = [Path(__file__), Path(generator.__file__),
                    ROOT / "src/prepare_external_datasets_v1.py",
                    ROOT / "src/run_external_ngboost.py",
                    ROOT / "src/run_assessment3_compute.py",
                    ROOT / "src/aggregate_assessment3_compute.py"]
    summary = {
        "status": "frozen", "protocol_sha256": sha256_file(batch / "protocol.json"),
        "datasets": datasets, "tasks": tasks,
        "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in source_files},
        "total_tasks": len(tasks), "total_variant_fold_evaluations": sum(d["folds"] * 3 for d in datasets),
    }
    freeze_json(batch / "batch.json", summary)
    print(json.dumps({key: summary[key] for key in ("status", "total_tasks", "total_variant_fold_evaluations")}, indent=2))
    print(batch)


if __name__ == "__main__":
    main()
