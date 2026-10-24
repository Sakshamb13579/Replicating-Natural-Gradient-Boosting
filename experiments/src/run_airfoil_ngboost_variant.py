#!/usr/bin/env python3
"""023-NG-P4 (+ future variant handoffs): NGBoost on frozen Airfoil v1 folds.

Usage:
  run_airfoil_ngboost_variant.py --variant natural OUT_DIR

Natural-gradient only for 023. Ordinary/homoscedastic are separate handoffs.
Uses exact frozen outer/inner indices from 022; does not regenerate splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error

from ngboost import NGBRegressor
from ngboost.distns import Normal, NormalFixedVar
from ngboost.learners import default_tree_learner
from ngboost.scores import MLE

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "airfoil-self-noise-v1.json"
CSV_PATH = ROOT / "data" / "processed" / "airfoil-self-noise-v1" / "airfoil-self-noise-v1.csv"
SPLITS_PATH = (
    ROOT / "data" / "processed" / "airfoil-self-noise-v1" / "split-indices.json"
)

N_EST = 2000
LR = 0.01
STAGE_MODEL_SEED_BASE = 20_000
FULL_MODEL_SEED_BASE = 30_000

INTERVAL_LEVELS = {
    "50": (0.25, 0.75),
    "80": (0.10, 0.90),
    "95": (0.025, 0.975),
}

EXPECTED_CSV_SHA = "705a2a7f0e55fcc9a87700038487b12ce768f4a2e1692433a2d752309028546e"
EXPECTED_SPLITS_SHA = "128c5375b2659dff369e153e1b35f2163fbd1c87b72cc5a183def469ce014c14"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def index_hash(indices) -> str:
    arr = np.asarray(indices, dtype=np.int64)
    return sha256_bytes(arr.tobytes())


def stats(vals: list[float]) -> dict:
    arr = np.asarray(vals, dtype=np.float64)
    n = len(arr)
    mean = float(arr.mean())
    median = float(np.median(arr))
    pop_sd = float(arr.std(ddof=0))
    sample_sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = sample_sd / (n**0.5) if n > 0 else 0.0
    return {
        "mean": mean,
        "median": median,
        "pop_sd": pop_sd,
        "sample_sd": sample_sd,
        "se": se,
        "n": n,
    }


def load_frozen() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    manifest = json.loads(MANIFEST.read_text())
    csv_sha = sha256_file(CSV_PATH)
    splits_sha = sha256_file(SPLITS_PATH)
    assert csv_sha == EXPECTED_CSV_SHA == manifest["processed"]["csv_sha256"], (
        csv_sha,
        EXPECTED_CSV_SHA,
    )
    assert splits_sha == EXPECTED_SPLITS_SHA == manifest["splits"]["sha256"], (
        splits_sha,
        EXPECTED_SPLITS_SHA,
    )

    data = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1)
    row_ids = data[:, 0].astype(int)
    X = np.ascontiguousarray(data[:, 1:-1], dtype=np.float64)
    y = np.ascontiguousarray(data[:, -1], dtype=np.float64)
    assert X.shape == (1503, 5)
    assert not np.isnan(X).any() and not np.isnan(y).any()

    splits = json.loads(SPLITS_PATH.read_text())
    assert splits["n_folds"] == 5
    for f, expected in zip(splits["folds"], manifest["splits"]["fold_hashes"]):
        assert f["fold_index"] == expected["fold_index"]
        for key in (
            "outer_train_hash",
            "outer_test_hash",
            "inner_train_hash",
            "inner_val_hash",
        ):
            assert f[key] == expected[key], (f["fold_index"], key)
            assert index_hash(f[key.replace("_hash", "_indices")]) == f[key]

    return row_ids, X, y, manifest, splits


def make_ngb(variant: str, random_state: int) -> NGBRegressor:
    if variant == "natural":
        return NGBRegressor(
            Base=default_tree_learner,
            Dist=Normal,
            Score=MLE,
            n_estimators=N_EST,
            learning_rate=LR,
            natural_gradient=True,
            minibatch_frac=1.0,
            verbose=False,
            random_state=random_state,
        )
    if variant == "ordinary":
        return NGBRegressor(
            Base=default_tree_learner,
            Dist=Normal,
            Score=MLE,
            n_estimators=N_EST,
            learning_rate=LR,
            natural_gradient=False,
            minibatch_frac=1.0,
            verbose=False,
            random_state=random_state,
        )
    if variant == "homoscedastic":
        return NGBRegressor(
            Base=default_tree_learner,
            Dist=NormalFixedVar,
            Score=MLE,
            n_estimators=N_EST,
            learning_rate=LR,
            natural_gradient=True,
            minibatch_frac=1.0,
            verbose=False,
            random_state=random_state,
        )
    raise ValueError(f"unknown variant: {variant}")


def coverage_and_width(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    inside = (y >= lo) & (y <= hi)
    return {
        "coverage": float(inside.mean()),
        "mean_width": float(np.mean(hi - lo)),
        "n": int(len(y)),
    }


def run_fold(
    variant: str,
    fold: dict,
    row_ids: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[dict, list[dict]]:
    fold_idx = int(fold["fold_index"])
    outer_train = np.asarray(fold["outer_train_indices"], dtype=np.int64)
    outer_test = np.asarray(fold["outer_test_indices"], dtype=np.int64)
    inner_train = np.asarray(fold["inner_train_indices"], dtype=np.int64)
    inner_val = np.asarray(fold["inner_val_indices"], dtype=np.int64)

    assert index_hash(outer_train) == fold["outer_train_hash"]
    assert index_hash(outer_test) == fold["outer_test_hash"]
    assert index_hash(inner_train) == fold["inner_train_hash"]
    assert index_hash(inner_val) == fold["inner_val_hash"]

    X_tr, y_tr = X[inner_train], y[inner_train]
    X_va, y_va = X[inner_val], y[inner_val]
    X_all, y_all = X[outer_train], y[outer_train]
    X_te, y_te = X[outer_test], y[outer_test]
    test_ids = row_ids[outer_test]

    stage_seed = STAGE_MODEL_SEED_BASE + fold_idx
    full_seed = FULL_MODEL_SEED_BASE + fold_idx

    t0 = time.perf_counter()
    ngb_stage = make_ngb(variant, stage_seed)
    ngb_stage.fit(X_tr, y_tr)
    stage_fit_sec = time.perf_counter() - t0

    y_preds = list(ngb_stage.staged_predict(X_va))
    assert len(y_preds) == N_EST
    val_rmse = np.asarray(
        [np.sqrt(mean_squared_error(yp, y_va)) for yp in y_preds], dtype=np.float64
    )
    best_iter = int(np.argmin(val_rmse) + 1)
    cap_hit = best_iter == N_EST

    t1 = time.perf_counter()
    ngb_full = make_ngb(variant, full_seed)
    ngb_full.fit(X_all, y_all)
    full_fit_sec = time.perf_counter() - t1

    forecast = ngb_full.pred_dist(X_te, max_iter=best_iter)
    mean = np.asarray(forecast.mean(), dtype=np.float64).reshape(-1)
    scale = np.asarray(forecast.scale, dtype=np.float64).reshape(-1)
    assert np.isfinite(mean).all() and np.isfinite(scale).all()
    assert (scale > 0).all()

    # Homoscedastic post-scale matches official regression_exp.py NormalFixedVar path.
    if variant == "homoscedastic":
        from scipy.stats import norm as norm_dist

        forecast_val = ngb_full.pred_dist(X_va, max_iter=best_iter)
        loc = np.asarray(forecast.loc, dtype=np.float64).reshape(-1)
        scale_factor = float(
            ((forecast_val.loc - y_va.flatten()) ** 2).mean() ** 0.5
        )
        scale = np.asarray(forecast.var, dtype=np.float64).reshape(-1) * scale_factor
        mean = loc
        interval_bounds = {}
        for name, (lo_q, hi_q) in INTERVAL_LEVELS.items():
            lo = norm_dist.ppf(lo_q, loc=loc, scale=scale)
            hi = norm_dist.ppf(hi_q, loc=loc, scale=scale)
            interval_bounds[name] = (lo, hi)
        nll = float(-norm_dist.logpdf(y_te.flatten(), loc=loc, scale=scale).mean())
    else:
        interval_bounds = {}
        for name, (lo_q, hi_q) in INTERVAL_LEVELS.items():
            lo = np.asarray(forecast.ppf(lo_q), dtype=np.float64).reshape(-1)
            hi = np.asarray(forecast.ppf(hi_q), dtype=np.float64).reshape(-1)
            interval_bounds[name] = (lo, hi)
        nll = float(-forecast.logpdf(y_te.flatten()).mean())

    rmse = float(np.sqrt(mean_squared_error(mean, y_te)))

    cov = {}
    for name, (lo, hi) in interval_bounds.items():
        assert np.isfinite(lo).all() and np.isfinite(hi).all()
        cov[name] = coverage_and_width(y_te, lo, hi)

    pred_rows = []
    for i in range(len(y_te)):
        row = {
            "fold_index": fold_idx,
            "row_id": int(test_ids[i]),
            "y_true": float(y_te[i]),
            "pred_mean": float(mean[i]),
            "pred_scale": float(scale[i]),
            "best_iter": best_iter,
        }
        for name, (lo, hi) in interval_bounds.items():
            row[f"interval_{name}_lo"] = float(lo[i])
            row[f"interval_{name}_hi"] = float(hi[i])
        pred_rows.append(row)

    fold_metrics = {
        "fold_index": fold_idx,
        "seeds": {
            "inner_random_state_frozen": fold["inner_random_state"],
            "stage_model_seed": stage_seed,
            "full_model_seed": full_seed,
        },
        "split_hashes_verified": {
            "outer_train_hash": fold["outer_train_hash"],
            "outer_test_hash": fold["outer_test_hash"],
            "inner_train_hash": fold["inner_train_hash"],
            "inner_val_hash": fold["inner_val_hash"],
        },
        "n_outer_train": int(len(outer_train)),
        "n_outer_test": int(len(outer_test)),
        "n_inner_train": int(len(inner_train)),
        "n_inner_val": int(len(inner_val)),
        "best_iter": best_iter,
        "cap_hit": cap_hit,
        "val_rmse_at_best": float(val_rmse[best_iter - 1]),
        "test_nll": nll,
        "test_rmse": rmse,
        "coverage": {k: v["coverage"] for k, v in cov.items()},
        "mean_width": {k: v["mean_width"] for k, v in cov.items()},
        "pred_scale_min": float(scale.min()),
        "pred_scale_max": float(scale.max()),
        "pred_scale_mean": float(scale.mean()),
        "finite_preds": True,
        "positive_scales": bool((scale > 0).all()),
        "timing_sec": {
            "stage_fit": stage_fit_sec,
            "full_fit": full_fit_sec,
            "fold_total": stage_fit_sec + full_fit_sec,
        },
    }
    return fold_metrics, pred_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["natural", "ordinary", "homoscedastic"])
    parser.add_argument("out_dir")
    args = parser.parse_args()

    # 023 authorizes natural only; keep other variants callable for later handoffs.
    if args.variant != "natural":
        # Still allow code path, but 023 invocation must pass natural.
        pass

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    row_ids, X, y, manifest, splits = load_frozen()

    wall0 = time.perf_counter()
    per_fold = []
    all_preds: list[dict] = []

    for fold in splits["folds"]:
        fm, preds = run_fold(args.variant, fold, row_ids, X, y)
        per_fold.append(fm)
        all_preds.extend(preds)
        print(
            "[%d/5] BestIter=%d TestNLL=%.6f TestRMSE=%.6f "
            "Cov50=%.3f Cov80=%.3f Cov95=%.3f CapHit=%s Time=%.1fs"
            % (
                fm["fold_index"] + 1,
                fm["best_iter"],
                fm["test_nll"],
                fm["test_rmse"],
                fm["coverage"]["50"],
                fm["coverage"]["80"],
                fm["coverage"]["95"],
                fm["cap_hit"],
                fm["timing_sec"]["fold_total"],
            ),
            flush=True,
        )

    wall = time.perf_counter() - wall0

    nlls = [f["test_nll"] for f in per_fold]
    rmses = [f["test_rmse"] for f in per_fold]
    best_iters = [f["best_iter"] for f in per_fold]
    cap_hits = int(sum(1 for f in per_fold if f["cap_hit"]))

    agg_cov = {}
    agg_width = {}
    for level in INTERVAL_LEVELS:
        agg_cov[level] = stats([f["coverage"][level] for f in per_fold])
        agg_width[level] = stats([f["mean_width"][level] for f in per_fold])

    # Also micro-average coverage across all test predictions.
    micro = {}
    for level in INTERVAL_LEVELS:
        lo = np.array([p[f"interval_{level}_lo"] for p in all_preds])
        hi = np.array([p[f"interval_{level}_hi"] for p in all_preds])
        yt = np.array([p["y_true"] for p in all_preds])
        micro[level] = coverage_and_width(yt, lo, hi)

    result = {
        "handoff": "023-NG-P4" if args.variant == "natural" else f"airfoil-{args.variant}",
        "dataset": "airfoil-self-noise-v1",
        "variant": args.variant,
        "claim_boundary": (
            "Natural-model feasibility/performance only; no superiority claim "
            "until ordinary and homoscedastic runs on identical folds."
        ),
        "protocol": {
            "n_estimators_cap": N_EST,
            "learning_rate": LR,
            "distn": "Normal" if args.variant != "homoscedastic" else "NormalFixedVar",
            "score": "MLE",
            "base": "default_tree_learner",
            "natural_gradient": args.variant != "ordinary",
            "minibatch_frac": 1.0,
            "stage_selector": "validation_RMSE",
            "n_folds": 5,
            "stage_model_seed_base": STAGE_MODEL_SEED_BASE,
            "full_model_seed_base": FULL_MODEL_SEED_BASE,
            "scaling": False,
            "imputation": False,
        },
        "data_verification": {
            "csv_sha256": EXPECTED_CSV_SHA,
            "splits_sha256": EXPECTED_SPLITS_SHA,
            "manifest_status": manifest["status"],
            "split_hashes_match_022": True,
        },
        "per_fold": per_fold,
        "aggregate": {
            "test_nll": stats(nlls),
            "test_rmse": stats(rmses),
            "best_iter": stats([float(x) for x in best_iters]),
            "cap_hits": cap_hits,
            "coverage_per_fold": agg_cov,
            "mean_width_per_fold": agg_width,
            "coverage_micro": micro,
        },
        "timing_sec": {
            "wall": wall,
            "sum_fold_totals": float(sum(f["timing_sec"]["fold_total"] for f in per_fold)),
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "sklearn": __import__("sklearn").__version__,
            "ngboost": getattr(__import__("ngboost"), "__version__", "unknown"),
            "ngboost_file": __import__("ngboost").__file__,
        },
        "thread_env": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "n_predictions": len(all_preds),
    }

    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    # Machine-readable predictions (JSONL).
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in all_preds:
            f.write(json.dumps(row) + "\n")

    # Compact CSV for intervals.
    import csv

    fields = [
        "fold_index",
        "row_id",
        "y_true",
        "pred_mean",
        "pred_scale",
        "best_iter",
        "interval_50_lo",
        "interval_50_hi",
        "interval_80_lo",
        "interval_80_hi",
        "interval_95_lo",
        "interval_95_hi",
    ]
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in all_preds:
            w.writerow({k: row[k] for k in fields})

    print(json.dumps({"aggregate": result["aggregate"], "timing_sec": result["timing_sec"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
