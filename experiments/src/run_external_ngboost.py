#!/usr/bin/env python3
"""Run paired three-variant NGBoost experiments from frozen dataset manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_squared_error

from ngboost import NGBRegressor
from ngboost.distns import Normal, NormalFixedVar
from ngboost.learners import default_tree_learner
from ngboost.scores import MLE

ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    "airfoil-self-noise",
    "real-estate-valuation",
    "parkinsons-telemonitoring",
    "forest-fires",
    "superconductivity",
)
VARIANTS = ("natural", "ordinary", "homoscedastic")
FULL_N_ESTIMATORS = 2000
LEARNING_RATE = 0.01
STAGE_SEED_BASE = 20_000
FULL_SEED_BASE = 30_000
INTERVALS = {"50": (0.25, 0.75), "80": (0.10, 0.90), "95": (0.025, 0.975)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_hash(indices: list[int] | np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "pop_sd": float(array.std(ddof=0)),
        "sample_sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "se": float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0,
        "n": len(array),
    }


def load_dataset(dataset: str) -> tuple[pd.DataFrame, dict, dict, Path, Path]:
    manifest_path = ROOT / "data" / "manifests" / f"{dataset}-v1.json"
    if not manifest_path.is_file():
        if dataset == "superconductivity":
            raise FileNotFoundError(
                "Superconductivity mode is TBD; benchmark and freeze a v1 manifest first"
            )
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen":
        raise ValueError(f"manifest is not frozen: {manifest_path}")

    if dataset == "airfoil-self-noise":
        csv_path = ROOT / "data" / "processed" / "airfoil-self-noise-v1" / "airfoil-self-noise-v1.csv"
        split_path = ROOT / "data" / "processed" / "airfoil-self-noise-v1" / "split-indices.json"
        limitations = [
            "Row splits can share aerodynamic conditions",
            "NLL selection is a new sensitivity analysis distinct from the original RMSE-selector preregistration",
        ]
        manifest.setdefault("limitations", limitations)
    else:
        csv_path = ROOT / manifest["processed"]["csv_path"]
        split_path = ROOT / manifest["splits"]["path"]

    if sha256_file(csv_path) != manifest["processed"]["csv_sha256"]:
        raise ValueError(f"processed CSV hash mismatch: {csv_path}")
    if sha256_file(split_path) != manifest["splits"]["sha256"]:
        raise ValueError(f"split manifest hash mismatch: {split_path}")
    frame = pd.read_csv(csv_path)
    if list(frame.columns) != manifest["processed"]["columns"]:
        raise ValueError("processed columns do not match manifest")
    if frame[manifest["processed"]["id_column"]].tolist() != list(range(len(frame))):
        raise ValueError("row IDs do not equal frozen zero-based positions")
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    if len(splits["folds"]) != manifest["splits"]["n_folds"]:
        raise ValueError("split fold count mismatch")
    for fold in splits["folds"]:
        for name in ("outer_train", "outer_test", "inner_train", "inner_val"):
            if index_hash(fold[f"{name}_indices"]) != fold[f"{name}_hash"]:
                raise ValueError(f"fold {fold['fold_index']} {name} hash mismatch")
    return frame, manifest, splits, manifest_path, csv_path


def make_model(variant: str, random_state: int, n_estimators: int) -> NGBRegressor:
    return NGBRegressor(
        Base=default_tree_learner,
        Dist=NormalFixedVar if variant == "homoscedastic" else Normal,
        Score=MLE,
        n_estimators=n_estimators,
        learning_rate=LEARNING_RATE,
        natural_gradient=variant != "ordinary",
        minibatch_frac=1.0,
        verbose=False,
        random_state=random_state,
    )


def select_stage(
    variant: str, model: NGBRegressor, X_val: np.ndarray, y_val: np.ndarray
) -> tuple[int, float | None, np.ndarray]:
    distributions = list(model.staged_pred_dist(X_val))
    nlls = np.empty(len(distributions), dtype=np.float64)
    sigmas = np.empty(len(distributions), dtype=np.float64)
    for index, distribution in enumerate(distributions):
        if variant == "homoscedastic":
            location = np.asarray(distribution.loc, dtype=np.float64).reshape(-1)
            sigma = float(np.sqrt(np.mean((location - y_val) ** 2)))
            if not np.isfinite(sigma) or sigma <= 0:
                raise FloatingPointError(f"invalid validation residual sigma at stage {index + 1}")
            sigmas[index] = sigma
            nlls[index] = float(-norm.logpdf(y_val, loc=location, scale=sigma).mean())
        else:
            nlls[index] = float(-distribution.logpdf(y_val).mean())
    if not np.isfinite(nlls).all():
        raise FloatingPointError("validation NLL contains non-finite values")
    best_index = int(np.argmin(nlls))
    sigma = float(sigmas[best_index]) if variant == "homoscedastic" else None
    return best_index + 1, sigma, nlls


def run_fold(
    variant: str,
    fold: dict,
    row_ids: np.ndarray,
    groups: np.ndarray | None,
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int,
) -> tuple[dict, list[dict]]:
    fold_index = int(fold["fold_index"])
    indices = {
        name: np.asarray(fold[f"{name}_indices"], dtype=np.int64)
        for name in ("outer_train", "outer_test", "inner_train", "inner_val")
    }
    stage_seed = STAGE_SEED_BASE + fold_index
    full_seed = FULL_SEED_BASE + fold_index

    stage_start = time.perf_counter()
    stage_model = make_model(variant, stage_seed, n_estimators)
    stage_model.fit(X[indices["inner_train"]], y[indices["inner_train"]])
    stage_fit_seconds = time.perf_counter() - stage_start
    best_iter, retained_sigma, val_nll = select_stage(
        variant, stage_model, X[indices["inner_val"]], y[indices["inner_val"]]
    )

    refit_start = time.perf_counter()
    full_model = make_model(variant, full_seed, n_estimators)
    full_model.fit(X[indices["outer_train"]], y[indices["outer_train"]])
    refit_seconds = time.perf_counter() - refit_start
    test_indices = indices["outer_test"]
    forecast = full_model.pred_dist(X[test_indices], max_iter=best_iter)
    mean = np.asarray(forecast.mean(), dtype=np.float64).reshape(-1)
    if variant == "homoscedastic":
        mean = np.asarray(forecast.loc, dtype=np.float64).reshape(-1)
        scale = np.full(len(mean), retained_sigma, dtype=np.float64)
        test_nll = float(-norm.logpdf(y[test_indices], loc=mean, scale=scale).mean())
        quantile = lambda q: norm.ppf(q, loc=mean, scale=scale)
    else:
        scale = np.asarray(forecast.scale, dtype=np.float64).reshape(-1)
        test_nll = float(-forecast.logpdf(y[test_indices]).mean())
        quantile = lambda q: np.asarray(forecast.ppf(q), dtype=np.float64).reshape(-1)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or not (scale > 0).all():
        raise FloatingPointError("non-finite prediction or non-positive scale")

    bounds = {level: (quantile(lo), quantile(hi)) for level, (lo, hi) in INTERVALS.items()}
    coverage = {
        level: float(((y[test_indices] >= lo) & (y[test_indices] <= hi)).mean())
        for level, (lo, hi) in bounds.items()
    }
    width = {level: float(np.mean(hi - lo)) for level, (lo, hi) in bounds.items()}
    predictions = []
    for position, row_index in enumerate(test_indices):
        row = {
            "fold_index": fold_index,
            "row_id": int(row_ids[row_index]),
            "y_true": float(y[row_index]),
            "pred_mean": float(mean[position]),
            "pred_scale": float(scale[position]),
            "best_iter": best_iter,
        }
        if groups is not None:
            row["group_id"] = int(groups[row_index])
        for level, (lo, hi) in bounds.items():
            row[f"interval_{level}_lo"] = float(lo[position])
            row[f"interval_{level}_hi"] = float(hi[position])
        predictions.append(row)

    metrics = {
        "fold_index": fold_index,
        "split_id": f"fold-{fold_index:02d}",
        "seeds": {"stage_model": stage_seed, "full_model": full_seed},
        "split_hashes": {
            f"{name}_hash": fold[f"{name}_hash"] for name in indices
        },
        "n_outer_train": len(indices["outer_train"]),
        "n_outer_test": len(test_indices),
        "n_inner_train": len(indices["inner_train"]),
        "n_inner_val": len(indices["inner_val"]),
        "best_iter": best_iter,
        "cap_hit": best_iter == n_estimators,
        "val_nll_at_best": float(val_nll[best_iter - 1]),
        "fixed_sigma_from_stage_val": retained_sigma,
        "sigma_reestimated_after_refit": False if variant == "homoscedastic" else None,
        "test_nll": test_nll,
        "test_rmse": float(np.sqrt(mean_squared_error(y[test_indices], mean))),
        "coverage": coverage,
        "mean_width": width,
        "pred_scale_min": float(scale.min()),
        "pred_scale_max": float(scale.max()),
        "pred_scale_mean": float(scale.mean()),
        "timing_sec": {
            "stage_fit": stage_fit_seconds,
            "full_refit": refit_seconds,
            "fold_total": stage_fit_seconds + refit_seconds,
        },
    }
    return metrics, predictions


def package_versions() -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": __import__("scipy").__version__,
        "sklearn": __import__("sklearn").__version__,
        "ngboost": getattr(__import__("ngboost"), "__version__", "unknown"),
        "ngboost_file": __import__("ngboost").__file__,
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--n-estimators", type=int, default=FULL_N_ESTIMATORS)
    parser.add_argument("out_dir")
    args = parser.parse_args()
    if args.n_estimators < 1 or (args.max_folds is not None and args.max_folds < 1):
        parser.error("smoke overrides must be positive")

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"output directory already exists; refusing overwrite: {out_dir}")
    out_dir.mkdir(parents=True)
    log_path = out_dir / "stdout.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    frame, data_manifest, splits, manifest_path, csv_path = load_dataset(args.dataset)
    full_fold_count = len(splits["folds"])
    selected_folds = splits["folds"][: args.max_folds] if args.max_folds else splits["folds"]
    reportable = args.n_estimators == FULL_N_ESTIMATORS and len(selected_folds) == full_fold_count
    non_reportable_reasons = []
    if args.n_estimators != FULL_N_ESTIMATORS:
        non_reportable_reasons.append(f"n_estimators override {args.n_estimators} != 2000")
    if len(selected_folds) != full_fold_count:
        non_reportable_reasons.append(f"max_folds override ran {len(selected_folds)} of {full_fold_count}")

    processed = data_manifest["processed"]
    features = processed["feature_columns"]
    target = processed["target_column"]
    evaluation_columns = processed.get("evaluation_columns", [])
    row_ids = frame[processed["id_column"]].to_numpy(dtype=np.int64)
    groups = (
        frame[evaluation_columns[0]].to_numpy(dtype=np.int64)
        if evaluation_columns
        else None
    )
    X = np.ascontiguousarray(frame[features].to_numpy(dtype=np.float64))
    y = np.ascontiguousarray(frame[target].to_numpy(dtype=np.float64))
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("non-finite model data")

    command = " ".join(shlex.quote(value) for value in [sys.executable] + sys.argv)
    config = {
        "dataset": args.dataset,
        "dataset_manifest": str(manifest_path),
        "variant": args.variant,
        "n_estimators_cap": args.n_estimators,
        "learning_rate": LEARNING_RATE,
        "distribution": "NormalFixedVar" if args.variant == "homoscedastic" else "Normal",
        "score": "MLE",
        "base_learner": "default_tree_learner",
        "natural_gradient": args.variant != "ordinary",
        "stage_selector": "validation_NLL",
        "full_protocol_n_estimators": FULL_N_ESTIMATORS,
        "full_protocol_folds": full_fold_count,
        "folds_run": len(selected_folds),
        "reportable": reportable,
        "non_reportable_reasons": non_reportable_reasons,
        "stage_seed_base": STAGE_SEED_BASE,
        "full_seed_base": FULL_SEED_BASE,
    }
    write_json(out_dir / "config.json", config)
    (out_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    environment = {
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "versions": package_versions(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
    }
    write_json(out_dir / "environment.json", environment)
    (out_dir / "environment.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in environment["versions"].items()) + "\n",
        encoding="utf-8",
    )
    run_manifest = {
        "artifact_schema_version": 1,
        "status": "running",
        "reportable": reportable,
        "data_manifest_path": str(manifest_path),
        "data_manifest_sha256": sha256_file(manifest_path),
        "processed_csv_path": str(csv_path),
        "processed_csv_sha256": sha256_file(csv_path),
        "split_manifest_sha256": data_manifest["splits"]["sha256"],
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json(out_dir / "manifest.json", run_manifest)

    wall_start = time.perf_counter()
    per_fold = []
    predictions = []
    for run_number, fold in enumerate(selected_folds, start=1):
        fold_metrics, fold_predictions = run_fold(
            args.variant, fold, row_ids, groups, X, y, args.n_estimators
        )
        per_fold.append(fold_metrics)
        predictions.extend(fold_predictions)
        log(
            "[%d/%d] fold=%d best_iter=%d val_nll=%.6f test_nll=%.6f "
            "rmse=%.6f cov50=%.3f cov80=%.3f cov95=%.3f cap_hit=%s time=%.3fs"
            % (
                run_number,
                len(selected_folds),
                fold_metrics["fold_index"],
                fold_metrics["best_iter"],
                fold_metrics["val_nll_at_best"],
                fold_metrics["test_nll"],
                fold_metrics["test_rmse"],
                fold_metrics["coverage"]["50"],
                fold_metrics["coverage"]["80"],
                fold_metrics["coverage"]["95"],
                fold_metrics["cap_hit"],
                fold_metrics["timing_sec"]["fold_total"],
            )
        )
    wall_seconds = time.perf_counter() - wall_start

    aggregate = {
        "test_nll": stats([fold["test_nll"] for fold in per_fold]),
        "test_rmse": stats([fold["test_rmse"] for fold in per_fold]),
        "best_iter": stats([float(fold["best_iter"]) for fold in per_fold]),
        "cap_hits": sum(bool(fold["cap_hit"]) for fold in per_fold),
        "coverage_per_fold": {
            level: stats([fold["coverage"][level] for fold in per_fold]) for level in INTERVALS
        },
        "mean_width_per_fold": {
            level: stats([fold["mean_width"][level] for fold in per_fold]) for level in INTERVALS
        },
    }
    result = {
        "artifact_schema_version": 1,
        "dataset": args.dataset,
        "dataset_slug": data_manifest["dataset_slug"],
        "variant": args.variant,
        "split_policy": data_manifest.get("split_policy", splits.get("outer_policy")),
        "selector": "validation_NLL",
        "reportable": reportable,
        "non_reportable_reasons": non_reportable_reasons,
        "protocol": config,
        "limitations": data_manifest.get("limitations", []),
        "per_fold": per_fold,
        "aggregate": aggregate,
        "sample_counts": {
            "dataset_rows": len(frame),
            "folds_run": len(per_fold),
            "test_predictions": len(predictions),
            "test_rows_per_fold": [fold["n_outer_test"] for fold in per_fold],
        },
        "timing_sec": {
            "wall": wall_seconds,
            "sum_fold_totals": float(sum(fold["timing_sec"]["fold_total"] for fold in per_fold)),
        },
        "versions": environment["versions"],
    }
    write_json(out_dir / "metrics.json", result)
    fields = [
        "fold_index", "row_id", "group_id", "y_true", "pred_mean", "pred_scale", "best_iter",
        "interval_50_lo", "interval_50_hi", "interval_80_lo", "interval_80_hi",
        "interval_95_lo", "interval_95_hi",
    ]
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(predictions)
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row) + "\n")
    timing = {"wall_seconds": wall_seconds, "folds": [fold["timing_sec"] for fold in per_fold]}
    write_json(out_dir / "timing.json", timing)
    run_manifest["status"] = "complete"
    run_manifest["metrics_sha256"] = sha256_file(out_dir / "metrics.json")
    write_json(out_dir / "manifest.json", run_manifest)
    log(json.dumps({"status": "complete", "reportable": reportable, "aggregate": aggregate}, indent=2))
    checksum_files = sorted(path for path in out_dir.iterdir() if path.name != "checksums.txt")
    (out_dir / "checksums.txt").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_files), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
