#!/usr/bin/env python3
"""029-NG-P5: Ordinary-gradient NGBoost on frozen heteroscedastic-v1 regimes A/B/C.

Exact 028 protocol/harness; only natural_gradient=False differs.
Stage selection uses validation NLL; ground-truth columns are evaluation-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.learners import default_tree_learner
from ngboost.scores import MLE

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "heteroscedastic-v1.json"
CSV_PATH = ROOT / "data" / "processed" / "heteroscedastic-v1" / "heteroscedastic-v1.csv"

N_EST = 2000
LR = 0.01
REGIMES = ("A", "B", "C")
STAGE_SEED_BASE = 20_000
FULL_SEED_BASE = 30_000

INTERVAL_LEVELS = {
    "50": (0.25, 0.75),
    "80": (0.10, 0.90),
    "95": (0.025, 0.975),
}

EXPECTED_CSV_SHA = "a39e58f617b5e595595fdd011af9284b40a5d2d452a427a3451047a50f6a181f"
FEATURE_COLS = ("x1", "x2")
ABS_X1_BINS = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coverage_and_width(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    inside = (y >= lo) & (y <= hi)
    return {
        "coverage": float(inside.mean()),
        "mean_width": float(np.mean(hi - lo)),
        "n": int(len(y)),
    }


def load_frozen() -> tuple[pd.DataFrame, dict]:
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["status"] == "frozen"
    csv_sha = sha256_file(CSV_PATH)
    assert csv_sha == EXPECTED_CSV_SHA == manifest["checksums"]["heteroscedastic-v1.csv"], (
        csv_sha,
        EXPECTED_CSV_SHA,
    )

    df = pd.read_csv(CSV_PATH)
    expected_cols = manifest["columns"]["heteroscedastic-v1.csv"]
    assert list(df.columns) == expected_cols
    assert len(df) == 12_000
    assert set(df["regime"].unique()) == set(REGIMES)
    assert set(df["split"].unique()) == {"train", "validation", "test"}

    split_ids = manifest["split"]
    for split_name, id_key in (
        ("train", "train_row_ids"),
        ("validation", "validation_row_ids"),
        ("test", "test_row_ids"),
    ):
        expected_ids = set(split_ids[id_key])
        for regime in REGIMES:
            got = set(df.loc[df["regime"] == regime, "row_id"].tolist())
            split_mask = df["regime"] == regime
            got_split = set(df.loc[split_mask & (df["split"] == split_name), "row_id"].tolist())
            assert got_split == expected_ids, (regime, split_name)
            assert len(got_split) == split_ids["counts_base"][split_name]

    numeric_cols = ["x1", "x2", "true_mu", "true_sigma", "noise_z", "y"]
    assert np.isfinite(df[numeric_cols].to_numpy(dtype=np.float64)).all()
    return df, manifest


def make_ngb(random_state: int) -> NGBRegressor:
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


def scale_correlations(
    pred_scale: np.ndarray, true_sigma: np.ndarray
) -> dict[str, float | None]:
    if float(np.std(true_sigma)) < 1e-12:
        return {"pearson": None, "spearman": None, "note": "constant true_sigma"}
    return {
        "pearson": float(pearsonr(pred_scale, true_sigma)[0]),
        "spearman": float(spearmanr(pred_scale, true_sigma)[0]),
        "note": None,
    }


def abs_x1_bin_summary(
    abs_x1: np.ndarray,
    true_sigma: np.ndarray,
    pred_scale: np.ndarray,
    width_95: np.ndarray,
) -> list[dict]:
    edges = np.quantile(abs_x1, np.linspace(0, 1, ABS_X1_BINS + 1))
    edges[-1] += 1e-9
    rows = []
    for i in range(ABS_X1_BINS):
        lo, hi = edges[i], edges[i + 1]
        mask = (abs_x1 >= lo) & (abs_x1 < hi)
        if not mask.any():
            continue
        rows.append(
            {
                "bin_index": i + 1,
                "n": int(mask.sum()),
                "mean_abs_x1": float(abs_x1[mask].mean()),
                "mean_true_sigma": float(true_sigma[mask].mean()),
                "mean_pred_scale": float(pred_scale[mask].mean()),
                "mean_width_95": float(width_95[mask].mean()),
            }
        )
    return rows


def run_regime(regime: str, df: pd.DataFrame) -> tuple[dict, list[dict]]:
    regime_idx = REGIMES.index(regime)
    stage_seed = STAGE_SEED_BASE + regime_idx
    full_seed = FULL_SEED_BASE + regime_idx

    sub = df[df["regime"] == regime].copy()
    train = sub[sub["split"] == "train"]
    val = sub[sub["split"] == "validation"]
    test = sub[sub["split"] == "test"]
    assert len(train) == 2400 and len(val) == 800 and len(test) == 800

    X_tr = np.ascontiguousarray(train[list(FEATURE_COLS)], dtype=np.float64)
    y_tr = np.ascontiguousarray(train["y"], dtype=np.float64)
    X_va = np.ascontiguousarray(val[list(FEATURE_COLS)], dtype=np.float64)
    y_va = np.ascontiguousarray(val["y"], dtype=np.float64)
    X_all = np.ascontiguousarray(
        pd.concat([train, val])[list(FEATURE_COLS)], dtype=np.float64
    )
    y_all = np.ascontiguousarray(pd.concat([train, val])["y"], dtype=np.float64)
    X_te = np.ascontiguousarray(test[list(FEATURE_COLS)], dtype=np.float64)
    y_te = np.ascontiguousarray(test["y"], dtype=np.float64)
    true_mu_te = np.ascontiguousarray(test["true_mu"], dtype=np.float64)
    true_sigma_te = np.ascontiguousarray(test["true_sigma"], dtype=np.float64)
    abs_x1_te = np.abs(test["x1"].to_numpy(dtype=np.float64))

    t0 = time.perf_counter()
    ngb_stage = make_ngb(stage_seed)
    ngb_stage.fit(X_tr, y_tr)
    stage_fit_sec = time.perf_counter() - t0

    staged_dists = ngb_stage.staged_pred_dist(X_va)
    val_nll = np.asarray(
        [-dist.logpdf(y_va.flatten()).mean() for dist in staged_dists], dtype=np.float64
    )
    assert len(val_nll) == N_EST
    best_iter = int(np.argmin(val_nll) + 1)
    cap_hit = best_iter == N_EST

    t1 = time.perf_counter()
    ngb_full = make_ngb(full_seed)
    ngb_full.fit(X_all, y_all)
    full_fit_sec = time.perf_counter() - t1

    forecast = ngb_full.pred_dist(X_te, max_iter=best_iter)
    mean = np.asarray(forecast.mean(), dtype=np.float64).reshape(-1)
    scale = np.asarray(forecast.scale, dtype=np.float64).reshape(-1)
    assert np.isfinite(mean).all() and np.isfinite(scale).all()
    assert (scale > 0).all()

    interval_bounds = {}
    for name, (lo_q, hi_q) in INTERVAL_LEVELS.items():
        lo = np.asarray(forecast.ppf(lo_q), dtype=np.float64).reshape(-1)
        hi = np.asarray(forecast.ppf(hi_q), dtype=np.float64).reshape(-1)
        interval_bounds[name] = (lo, hi)

    test_nll = float(-forecast.logpdf(y_te.flatten()).mean())
    test_rmse = float(np.sqrt(mean_squared_error(mean, y_te)))
    oracle_nll = float(-norm.logpdf(y_te, loc=true_mu_te, scale=true_sigma_te).mean())
    scale_mae = float(np.mean(np.abs(scale - true_sigma_te)))

    cov = {}
    for name, (lo, hi) in interval_bounds.items():
        assert np.isfinite(lo).all() and np.isfinite(hi).all()
        cov[name] = coverage_and_width(y_te, lo, hi)

    width_95 = interval_bounds["95"][1] - interval_bounds["95"][0]
    corr = scale_correlations(scale, true_sigma_te)
    bins = abs_x1_bin_summary(abs_x1_te, true_sigma_te, scale, width_95)

    pred_rows = []
    test_df = test.reset_index(drop=True)
    for i in range(len(y_te)):
        row = {
            "regime": regime,
            "row_id": int(test_df.loc[i, "row_id"]),
            "split": "test",
            "x1": float(test_df.loc[i, "x1"]),
            "x2": float(test_df.loc[i, "x2"]),
            "y_true": float(y_te[i]),
            "true_mu": float(true_mu_te[i]),
            "true_sigma": float(true_sigma_te[i]),
            "pred_mean": float(mean[i]),
            "pred_scale": float(scale[i]),
            "best_iter": best_iter,
        }
        for name, (lo, hi) in interval_bounds.items():
            row[f"interval_{name}_lo"] = float(lo[i])
            row[f"interval_{name}_hi"] = float(hi[i])
        pred_rows.append(row)

    metrics = {
        "regime": regime,
        "seeds": {
            "stage_model_seed": stage_seed,
            "full_model_seed": full_seed,
        },
        "split_counts": {
            "train": int(len(train)),
            "validation": int(len(val)),
            "test": int(len(test)),
        },
        "best_iter": best_iter,
        "cap_hit": cap_hit,
        "val_nll_at_best": float(val_nll[best_iter - 1]),
        "val_nll_min": float(val_nll.min()),
        "test_nll": test_nll,
        "test_rmse": test_rmse,
        "oracle_nll": oracle_nll,
        "scale_mae_vs_true_sigma": scale_mae,
        "scale_correlation": corr,
        "coverage": {k: v["coverage"] for k, v in cov.items()},
        "mean_width": {k: v["mean_width"] for k, v in cov.items()},
        "abs_x1_bins_test": bins,
        "pred_scale_min": float(scale.min()),
        "pred_scale_max": float(scale.max()),
        "pred_scale_mean": float(scale.mean()),
        "true_sigma_min": float(true_sigma_te.min()),
        "true_sigma_max": float(true_sigma_te.max()),
        "finite_preds": True,
        "positive_scales": bool((scale > 0).all()),
        "timing_sec": {
            "stage_fit": stage_fit_sec,
            "full_fit": full_fit_sec,
            "regime_total": stage_fit_sec + full_fit_sec,
        },
    }
    return metrics, pred_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, manifest = load_frozen()

    wall0 = time.perf_counter()
    per_regime = []
    all_preds: list[dict] = []

    for regime in REGIMES:
        rm, preds = run_regime(regime, df)
        per_regime.append(rm)
        all_preds.extend(preds)
        print(
            "[%s] BestIter=%d ValNLL=%.6f TestNLL=%.6f OracleNLL=%.6f "
            "TestRMSE=%.6f ScaleMAE=%.6f Cov95=%.3f CapHit=%s Time=%.1fs"
            % (
                regime,
                rm["best_iter"],
                rm["val_nll_at_best"],
                rm["test_nll"],
                rm["oracle_nll"],
                rm["test_rmse"],
                rm["scale_mae_vs_true_sigma"],
                rm["coverage"]["95"],
                rm["cap_hit"],
                rm["timing_sec"]["regime_total"],
            ),
            flush=True,
        )

    wall = time.perf_counter() - wall0

    result = {
        "handoff": "029-NG-P5",
        "dataset": "heteroscedastic-v1",
        "variant": "ordinary",
        "claim_boundary": (
            "Ordinary-gradient multiparameter evidence on constructed heteroscedastic regimes A/B/C; "
            "provisional paired comparison to 028 only — final three-way verdict waits for homoscedastic."
        ),
        "protocol": {
            "n_estimators_cap": N_EST,
            "learning_rate": LR,
            "distn": "Normal",
            "score": "MLE",
            "base": "default_tree_learner",
            "natural_gradient": False,
            "minibatch_frac": 1.0,
            "stage_selector": "validation_NLL",
            "stage_selector_note": (
                "Validation NLL for constructed uncertainty study; differs from official "
                "regression_exp.py RMSE selection and aligns with probabilistic objective."
            ),
            "features": list(FEATURE_COLS),
            "target": "y",
            "ground_truth_eval_only": ["true_mu", "true_sigma", "noise_z"],
            "regimes": list(REGIMES),
            "split_policy": "frozen 60/20/20 per regime from 007",
            "stage_model_seed_base": STAGE_SEED_BASE,
            "full_model_seed_base": FULL_SEED_BASE,
        },
        "data_verification": {
            "csv_sha256": EXPECTED_CSV_SHA,
            "manifest_status": manifest["status"],
            "split_row_ids_verified": True,
            "ground_truth_excluded_from_features": True,
        },
        "per_regime": per_regime,
        "aggregate": {
            "test_nll_mean": float(np.mean([r["test_nll"] for r in per_regime])),
            "test_rmse_mean": float(np.mean([r["test_rmse"] for r in per_regime])),
            "oracle_nll_mean": float(np.mean([r["oracle_nll"] for r in per_regime])),
            "cap_hits": int(sum(1 for r in per_regime if r["cap_hit"])),
        },
        "timing_sec": {
            "wall": wall,
            "sum_regime_totals": float(
                sum(r["timing_sec"]["regime_total"] for r in per_regime)
            ),
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
            "scipy": __import__("scipy").__version__,
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
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in all_preds:
            f.write(json.dumps(row) + "\n")

    fields = [
        "regime",
        "row_id",
        "split",
        "x1",
        "x2",
        "y_true",
        "true_mu",
        "true_sigma",
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
