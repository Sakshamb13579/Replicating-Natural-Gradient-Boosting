#!/usr/bin/env python3
"""Run controlled RMSE-vs-NLL stage selection on frozen original data."""

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
from datetime import datetime, timezone
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
DATASETS = ("yacht", "kin8nm")
VARIANTS = ("natural", "ordinary", "homoscedastic")
SLUGS = {name: f"{name}-original-selector-sensitivity-v1" for name in DATASETS}
SELECTORS = ("rmse", "nll")
FULL_N_ESTIMATORS = 2000
LEARNING_RATE = 0.01
STAGE_SEED_BASE = 20_000
REFIT_SEED_BASE = 30_000
TIE_ATOL = 1e-12


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_hash(indices: list[int] | np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


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


def checked_vector(value: object, expected: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(array) != expected:
        raise ValueError(f"{label} shape mismatch after flatten: {array.shape}, expected {expected}")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{label} contains non-finite values")
    return array


def load_frozen(dataset: str) -> tuple[pd.DataFrame, dict, dict, Path, Path, Path]:
    slug = SLUGS[dataset]
    manifest_path = ROOT / "data/manifests" / f"{slug}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen" or manifest.get("dataset_slug") != slug:
        raise ValueError(f"invalid frozen manifest: {manifest_path}")
    csv_path = ROOT / manifest["processed"]["csv_path"]
    split_path = ROOT / manifest["splits"]["path"]
    if sha256_file(csv_path) != manifest["processed"]["csv_sha256"]:
        raise ValueError("processed CSV hash mismatch")
    if sha256_file(split_path) != manifest["splits"]["sha256"]:
        raise ValueError("split manifest hash mismatch")
    if sha256_file(Path(manifest["source"]["preserved_raw_path"])) != manifest["source"]["raw_sha256"]:
        raise ValueError("preserved raw hash mismatch")
    frame = pd.read_csv(csv_path)
    if list(frame.columns) != manifest["processed"]["columns"]:
        raise ValueError("processed columns mismatch")
    if frame["row_id"].tolist() != list(range(len(frame))):
        raise ValueError("row_id is not stable zero-based source order")
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    if splits.get("status") != "frozen" or len(splits["folds"]) != manifest["splits"]["n_folds"]:
        raise ValueError("invalid split manifest")
    for fold in splits["folds"]:
        for name in ("outer_train", "outer_test", "inner_train", "inner_val"):
            if index_hash(fold[f"{name}_indices"]) != fold[f"{name}_hash"]:
                raise ValueError(f"fold {fold['fold_index']} {name} hash mismatch")
    return frame, manifest, splits, manifest_path, csv_path, split_path


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


def validation_trajectory(
    variant: str, model: NGBRegressor, X_val: np.ndarray, y_val: np.ndarray
) -> dict:
    mse_values, rmse_values, nll_values, sigma_values = [], [], [], []
    for stage, forecast in enumerate(model.staged_pred_dist(X_val), start=1):
        mean = checked_vector(forecast.loc if variant == "homoscedastic" else forecast.mean(), len(y_val), f"stage {stage} validation mean")
        mse = float(mean_squared_error(y_val, mean))
        rmse = float(np.sqrt(mse))
        if variant == "homoscedastic":
            sigma = rmse
            if sigma <= 0:
                raise FloatingPointError(f"stage {stage} non-positive validation residual RMS")
            logpdf = checked_vector(norm.logpdf(y_val, loc=mean, scale=sigma), len(y_val), f"stage {stage} validation fixed-scale logpdf")
            sigma_values.append(sigma)
        else:
            logpdf = checked_vector(forecast.logpdf(y_val.reshape(-1)), len(y_val), f"stage {stage} validation Normal logpdf")
            sigma_values.append(None)
        mse_values.append(mse)
        rmse_values.append(rmse)
        nll_values.append(float(-logpdf.mean()))
    return {"mse": mse_values, "rmse": rmse_values, "nll": nll_values, "homoscedastic_residual_rms": sigma_values}


def evaluate_stage(
    variant: str,
    model: NGBRegressor,
    stage: int,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    forecast = model.pred_dist(X_test, max_iter=stage)
    mean = checked_vector(forecast.loc if variant == "homoscedastic" else forecast.mean(), len(y_test), "test mean")
    scale_adjustment = None
    fixed_var = None
    if variant == "homoscedastic":
        forecast_val = model.pred_dist(X_val, max_iter=stage)
        val_loc = checked_vector(forecast_val.loc, len(y_val), "post-refit validation location")
        scale_adjustment = float(np.sqrt(np.mean((val_loc - y_val) ** 2)))
        if not np.isfinite(scale_adjustment) or scale_adjustment <= 0:
            raise FloatingPointError("invalid post-refit validation residual RMS")
        fixed_var = checked_vector(forecast.var, len(y_test), "NormalFixedVar test variance")
        scale = fixed_var * scale_adjustment
        if not np.allclose(scale, scale[0], rtol=0, atol=1e-15):
            raise ValueError("NormalFixedVar released-script scale is not constant")
        logpdf = checked_vector(norm.logpdf(y_test, loc=mean, scale=scale), len(y_test), "test fixed-scale logpdf")
    else:
        scale = checked_vector(forecast.scale, len(y_test), "test Normal scale")
        logpdf = checked_vector(forecast.logpdf(y_test.reshape(-1)), len(y_test), "test Normal logpdf")
    if not (scale > 0).all():
        raise FloatingPointError("test scale is non-positive")
    metrics = {
        "selected_stage": stage,
        "test_nll": float(-logpdf.mean()),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, mean))),
        "post_refit_validation_residual_rms": scale_adjustment,
        "normal_fixed_var_min": float(fixed_var.min()) if fixed_var is not None else None,
        "normal_fixed_var_max": float(fixed_var.max()) if fixed_var is not None else None,
        "pred_scale_min": float(scale.min()),
        "pred_scale_max": float(scale.max()),
        "pred_scale_mean": float(scale.mean()),
        "finite_output": True,
        "positive_scale": True,
    }
    return metrics, mean, scale


def run_fold(variant: str, fold: dict, row_ids: np.ndarray, X: np.ndarray, y: np.ndarray, n_estimators: int) -> tuple[dict, list[dict]]:
    fold_index = int(fold["fold_index"])
    indices = {name: np.asarray(fold[f"{name}_indices"], dtype=np.int64) for name in ("outer_train", "outer_test", "inner_train", "inner_val")}
    stage_seed = STAGE_SEED_BASE + fold_index
    refit_seed = REFIT_SEED_BASE + fold_index

    stage_start = time.perf_counter()
    stage_model = make_model(variant, stage_seed, n_estimators)
    stage_model.fit(X[indices["inner_train"]], y[indices["inner_train"]])
    stage_fit_seconds = time.perf_counter() - stage_start
    trajectory = validation_trajectory(variant, stage_model, X[indices["inner_val"]], y[indices["inner_val"]])
    if len(trajectory["mse"]) != n_estimators:
        raise AssertionError("inner trajectory stage count mismatch")
    selected = {"rmse": int(np.argmin(trajectory["mse"])) + 1, "nll": int(np.argmin(trajectory["nll"])) + 1}

    refit_start = time.perf_counter()
    refit_model = make_model(variant, refit_seed, n_estimators)
    refit_model.fit(X[indices["outer_train"]], y[indices["outer_train"]])
    refit_seconds = time.perf_counter() - refit_start

    selector_metrics = {}
    predictions = []
    for selector in SELECTORS:
        stage = selected[selector]
        evaluated, mean, scale = evaluate_stage(
            variant, refit_model, stage, X[indices["inner_val"]], y[indices["inner_val"]], X[indices["outer_test"]], y[indices["outer_test"]]
        )
        evaluated.update({
            "cap_hit": stage == n_estimators,
            "validation_mse_at_selected_stage": trajectory["mse"][stage - 1],
            "validation_rmse_at_selected_stage": trajectory["rmse"][stage - 1],
            "validation_nll_at_selected_stage": trajectory["nll"][stage - 1],
        })
        selector_metrics[selector] = evaluated
        for position, absolute_index in enumerate(indices["outer_test"]):
            predictions.append({
                "fold_index": fold_index,
                "selector": selector,
                "row_id": int(row_ids[absolute_index]),
                "y_true": float(y[absolute_index]),
                "pred_mean": float(mean[position]),
                "pred_scale": float(scale[position]),
                "selected_stage": stage,
            })
    delta = {
        "selected_stage_nll_minus_rmse": selected["nll"] - selected["rmse"],
        "test_nll_nll_selector_minus_rmse_selector": selector_metrics["nll"]["test_nll"] - selector_metrics["rmse"]["test_nll"],
        "test_rmse_nll_selector_minus_rmse_selector": selector_metrics["nll"]["test_rmse"] - selector_metrics["rmse"]["test_rmse"],
    }
    metrics = {
        "fold_index": fold_index,
        "split_id": f"fold-{fold_index:02d}",
        "seeds": {"stage_model": stage_seed, "refit_model": refit_seed, "inner_random_state_frozen": fold["inner_random_state"]},
        "fit_counts": {"inner_stage_trajectory": 1, "outer_refit_trajectory": 1},
        "split_hashes": {f"{name}_hash": fold[f"{name}_hash"] for name in indices},
        "sample_counts": {f"n_{name}": len(value) for name, value in indices.items()},
        "validation_trajectory": trajectory,
        "selectors": selector_metrics,
        "paired_delta": delta,
        "timing_sec": {"stage_fit": stage_fit_seconds, "outer_refit": refit_seconds, "fold_total": stage_fit_seconds + refit_seconds},
    }
    return metrics, predictions


def wins(values: list[float]) -> dict:
    nll_wins = sum(value < -TIE_ATOL for value in values)
    rmse_wins = sum(value > TIE_ATOL for value in values)
    return {"nll_selector_wins": nll_wins, "rmse_selector_wins": rmse_wins, "ties": len(values) - nll_wins - rmse_wins, "tie_atol": TIE_ATOL}


def aggregate(per_fold: list[dict]) -> dict:
    output = {"selectors": {}, "paired_deltas": {}, "win_tie_counts": {}}
    for selector in SELECTORS:
        output["selectors"][selector] = {
            "test_nll": stats([fold["selectors"][selector]["test_nll"] for fold in per_fold]),
            "test_rmse": stats([fold["selectors"][selector]["test_rmse"] for fold in per_fold]),
            "selected_stage": stats([float(fold["selectors"][selector]["selected_stage"]) for fold in per_fold]),
            "cap_hits": sum(fold["selectors"][selector]["cap_hit"] for fold in per_fold),
        }
    for metric in ("selected_stage", "test_nll", "test_rmse"):
        key = (
            "selected_stage_nll_minus_rmse"
            if metric == "selected_stage"
            else f"{metric}_nll_selector_minus_rmse_selector"
        )
        values = [fold["paired_delta"][key] for fold in per_fold]
        output["paired_deltas"][metric] = stats([float(value) for value in values])
        output["win_tie_counts"][metric] = wins(values)
    output["selected_stage_changed_folds"] = sum(
        fold["selectors"]["rmse"]["selected_stage"] != fold["selectors"]["nll"]["selected_stage"] for fold in per_fold
    )
    return output


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--max-folds", type=int, help="smoke only; makes output non-reportable")
    parser.add_argument("--n-estimators", type=int, default=FULL_N_ESTIMATORS, help="values other than 2000 are smoke only")
    parser.add_argument("out_dir")
    args = parser.parse_args()
    if args.n_estimators < 1 or (args.max_folds is not None and args.max_folds < 1):
        parser.error("smoke overrides must be positive")

    out_dir = Path(args.out_dir).resolve()
    if not out_dir.is_relative_to(ROOT):
        parser.error(f"output directory must be under {ROOT}")
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    stdout_path, stderr_path = out_dir / "stdout.log", out_dir / "stderr.log"
    stderr_path.write_text("", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        with stdout_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    frame, data_manifest, splits, manifest_path, csv_path, split_path = load_frozen(args.dataset)
    all_folds = splits["folds"]
    selected_folds = all_folds[: args.max_folds] if args.max_folds is not None else all_folds
    reasons = []
    if args.n_estimators != FULL_N_ESTIMATORS:
        reasons.append(f"n_estimators override {args.n_estimators} != {FULL_N_ESTIMATORS}")
    if len(selected_folds) != len(all_folds):
        reasons.append(f"max_folds override ran {len(selected_folds)} of {len(all_folds)}")
    reportable = not reasons
    started_at = utc_now()
    command = " ".join(shlex.quote(value) for value in [sys.executable] + sys.argv)
    command_record = f"working_directory={Path.cwd()}\ncommand={command}\n"
    (out_dir / "command.txt").write_text(command_record, encoding="utf-8")
    config = {
        "dataset": args.dataset,
        "dataset_slug": data_manifest["dataset_slug"],
        "variant": args.variant,
        "distribution": "NormalFixedVar" if args.variant == "homoscedastic" else "Normal",
        "score": "MLE",
        "base_learner": "default_tree_learner",
        "natural_gradient": args.variant != "ordinary",
        "learning_rate": LEARNING_RATE,
        "minibatch_frac": 1.0,
        "n_estimators_cap": args.n_estimators,
        "full_protocol_n_estimators": FULL_N_ESTIMATORS,
        "full_protocol_folds": len(all_folds),
        "folds_run": len(selected_folds),
        "selectors": {"rmse": "argmin validation MSE + 1", "nll": "argmin validation NLL + 1"},
        "homoscedastic_validation_scale": "per-stage validation residual RMS",
        "homoscedastic_post_refit_scale": "NormalFixedVar forecast.var multiplied by post-refit frozen-validation residual RMS, separately per selected stage",
        "stage_seed_base": STAGE_SEED_BASE,
        "refit_seed_base": REFIT_SEED_BASE,
        "reportable": reportable,
        "non_reportable_reasons": reasons,
    }
    write_json(out_dir / "config.json", config)
    environment = {
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "versions": package_versions(),
        "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
    }
    write_json(out_dir / "environment.json", environment)
    (out_dir / "environment.txt").write_text("\n".join(f"{key}={value}" for key, value in environment["versions"].items()) + "\n", encoding="utf-8")
    run_manifest = {
        "artifact_schema_version": 1,
        "status": "running",
        "handoff_id": "original-selector-sensitivity-v1",
        "run_id": out_dir.name,
        "dataset_slug": data_manifest["dataset_slug"],
        "model_slug": f"ngboost-{args.variant}",
        "seed": {"stage_base": STAGE_SEED_BASE, "refit_base": REFIT_SEED_BASE},
        "attempt": 1,
        "created_at": started_at,
        "reportable": reportable,
        "split_manifest": str(split_path),
        "upstream_sources": [{"commit": "672034041fc05d9d4b7a80c1e6163ea102579752", "ngboost_file": environment["versions"]["ngboost_file"]}],
        "data_checksums": {"raw": data_manifest["source"]["raw_sha256"], "processed_csv": sha256_file(csv_path), "dataset_manifest": sha256_file(manifest_path), "split_manifest": sha256_file(split_path)},
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json(out_dir / "manifest.json", run_manifest)

    processed = data_manifest["processed"]
    row_ids = frame[processed["id_column"]].to_numpy(dtype=np.int64)
    X = np.ascontiguousarray(frame[processed["feature_columns"]].to_numpy(dtype=np.float64))
    y = np.ascontiguousarray(frame[processed["target_column"]].to_numpy(dtype=np.float64).reshape(-1))
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("non-finite model input")

    wall_start = time.perf_counter()
    per_fold, predictions = [], []
    for run_number, fold in enumerate(selected_folds, start=1):
        fold_metrics, fold_predictions = run_fold(args.variant, fold, row_ids, X, y, args.n_estimators)
        per_fold.append(fold_metrics)
        predictions.extend(fold_predictions)
        rmse_result, nll_result = fold_metrics["selectors"]["rmse"], fold_metrics["selectors"]["nll"]
        log(
            f"[{run_number}/{len(selected_folds)}] fold={fold['fold_index']} "
            f"rmse_stage={rmse_result['selected_stage']} nll_stage={nll_result['selected_stage']} "
            f"rmse_sel(test_nll={rmse_result['test_nll']:.6f},test_rmse={rmse_result['test_rmse']:.6f}) "
            f"nll_sel(test_nll={nll_result['test_nll']:.6f},test_rmse={nll_result['test_rmse']:.6f}) "
            f"time={fold_metrics['timing_sec']['fold_total']:.3f}s"
        )
    wall_seconds = time.perf_counter() - wall_start
    result = {
        "artifact_schema_version": 1,
        "dataset": args.dataset,
        "dataset_slug": data_manifest["dataset_slug"],
        "variant": args.variant,
        "reportable": reportable,
        "non_reportable_reasons": reasons,
        "protocol": config,
        "per_fold": per_fold,
        "aggregate": aggregate(per_fold),
        "sample_counts": {"dataset_rows": len(frame), "folds_run": len(per_fold), "prediction_rows_both_selectors": len(predictions)},
        "timing_sec": {"wall": wall_seconds, "sum_fold_totals": float(sum(fold["timing_sec"]["fold_total"] for fold in per_fold))},
        "versions": environment["versions"],
        "comparability": "Controlled frozen-inner-split sensitivity; not a bit-for-bit rerun of historical released-script results.",
    }
    write_json(out_dir / "metrics.json", result)
    prediction_fields = ["fold_index", "selector", "row_id", "y_true", "pred_mean", "pred_scale", "selected_stage"]
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        writer.writerows(predictions)
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row) + "\n")
    ended_at = utc_now()
    write_json(out_dir / "timing.json", {"started_at": started_at, "ended_at": ended_at, "wall_seconds": wall_seconds, "folds": [fold["timing_sec"] for fold in per_fold]})
    run_manifest.update({"status": "complete", "completed_at": ended_at, "metrics_sha256": sha256_file(out_dir / "metrics.json"), "predictions_csv_sha256": sha256_file(out_dir / "predictions.csv")})
    write_json(out_dir / "manifest.json", run_manifest)
    log(json.dumps({"status": "complete", "reportable": reportable, "aggregate": result["aggregate"]}))
    checksum_paths = sorted(path for path in out_dir.iterdir() if path.is_file() and path.name != "checksums.txt")
    (out_dir / "checksums.txt").write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
