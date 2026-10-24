#!/usr/bin/env python3
"""Controlled Yacht natural-gradient cap diagnostic (017F1-NG-D1).

Isolates estimator headroom by deriving cap-2000 and cap-4000 views from
identical 4000-stage trajectories on frozen outer/inner splits.

This is a diagnostic harness, not the official regression_exp replication.
a01 (official script) is confounded by inner train_test_split RNG coupling.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.learners import default_tree_learner
from ngboost.scores import MLE

N_SPLITS = 20
N_EST = 4000
CAP_LOW = 2000
LR = 0.01
OUTER_SEED = 1
# Inner splits frozen independently of model fitting / global RNG consumption.
INNER_SEED_BASE = 10_000
# Explicit per-fold model seeds (stage-selection fit and full-train fit).
STAGE_MODEL_SEED_BASE = 20_000
FULL_MODEL_SEED_BASE = 30_000

YACHT_LOCAL = Path(__file__).resolve().parents[1] / "data/original/yacht_hydrodynamics.data"
YACHT_URL = (
    "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/"
    "yacht_hydrodynamics.data"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def index_hash(indices: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(indices.astype(np.int64)).tobytes())


def load_yacht() -> tuple[np.ndarray, np.ndarray, dict]:
    if YACHT_LOCAL.is_file():
        path = YACHT_LOCAL
        data = pd.read_csv(path, header=None, sep=r"\s+")
        source = {"route": "local_cache", "path": str(path), "sha256": sha256_file(path)}
    else:
        data = pd.read_csv(YACHT_URL, header=None, sep=r"\s+")
        source = {"route": "uci_url", "url": YACHT_URL}
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y, source


def make_outer_folds(n: int, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Match official regression_exp.py outer 90/10 split construction."""
    rng = np.random.RandomState(seed)
    folds = []
    for _ in range(n_splits):
        permutation = rng.choice(range(n), n, replace=False)
        end_train = round(n * 9.0 / 10)
        train_index = permutation[0:end_train]
        test_index = permutation[end_train:n]
        folds.append((train_index, test_index))
    return folds


def make_ngb(random_state: int) -> NGBRegressor:
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
    }


def evaluate_at_iter(ngb: NGBRegressor, X: np.ndarray, y: np.ndarray, max_iter: int) -> dict:
    forecast = ngb.pred_dist(X, max_iter=max_iter)
    rmse = float(np.sqrt(mean_squared_error(forecast.mean(), y)))
    nll = float(-forecast.logpdf(y.flatten()).mean())
    return {"rmse": rmse, "nll": nll}


def run_fold(
    fold_idx: int,
    X: np.ndarray,
    y: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
) -> dict:
    X_trainall, X_test = X[train_index], X[test_index]
    y_trainall, y_test = y[train_index], y[test_index]

    inner_seed = INNER_SEED_BASE + fold_idx
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainall,
        y_trainall,
        test_size=0.2,
        random_state=inner_seed,
    )

    # Recover inner index lists relative to outer train for hashing.
    # train_test_split shuffles; reconstruct by matching rows is fragile with dups.
    # Instead store absolute indices via a second call on index array.
    outer_train_pos = np.arange(len(train_index))
    pos_train, pos_val = train_test_split(
        outer_train_pos, test_size=0.2, random_state=inner_seed
    )
    inner_train_abs = train_index[pos_train]
    inner_val_abs = train_index[pos_val]

    stage_seed = STAGE_MODEL_SEED_BASE + fold_idx
    full_seed = FULL_MODEL_SEED_BASE + fold_idx

    t0 = time.perf_counter()
    ngb_stage = make_ngb(stage_seed)
    ngb_stage.fit(X_train, y_train)
    stage_fit_sec = time.perf_counter() - t0

    y_preds = list(ngb_stage.staged_predict(X_val))
    assert len(y_preds) == N_EST, f"expected {N_EST} stages, got {len(y_preds)}"

    val_rmse = np.asarray(
        [np.sqrt(mean_squared_error(y_pred, y_val)) for y_pred in y_preds],
        dtype=np.float64,
    )
    # Shared trajectory: first CAP_LOW stages are prefix of full 4000-stage path.
    shared_prefix_ok = bool(np.allclose(val_rmse[:CAP_LOW], val_rmse[:CAP_LOW]))

    best_iter_cap2000 = int(np.argmin(val_rmse[:CAP_LOW]) + 1)
    best_iter_cap4000 = int(np.argmin(val_rmse[:N_EST]) + 1)
    # If global argmin is within first 2000, both views must agree.
    if best_iter_cap4000 <= CAP_LOW:
        assert best_iter_cap2000 == best_iter_cap4000

    t1 = time.perf_counter()
    ngb_full = make_ngb(full_seed)
    ngb_full.fit(X_trainall, y_trainall)
    full_fit_sec = time.perf_counter() - t1

    # Same fitted full trajectory evaluated at both selected iteration counts.
    test_2000 = evaluate_at_iter(ngb_full, X_test, y_test, best_iter_cap2000)
    test_4000 = evaluate_at_iter(ngb_full, X_test, y_test, best_iter_cap4000)
    val_at_2000 = evaluate_at_iter(ngb_full, X_val, y_val, best_iter_cap2000)
    val_at_4000 = evaluate_at_iter(ngb_full, X_val, y_val, best_iter_cap4000)

    # Verify shared prediction path: max_iter truncation equals staged index.
    staged_test = list(ngb_full.staged_predict(X_test))
    assert len(staged_test) == N_EST
    mean_at_2000 = ngb_full.predict(X_test, max_iter=best_iter_cap2000)
    mean_at_4000 = ngb_full.predict(X_test, max_iter=best_iter_cap4000)
    shared_pred_ok = bool(
        np.allclose(mean_at_2000, staged_test[best_iter_cap2000 - 1])
        and np.allclose(mean_at_4000, staged_test[best_iter_cap4000 - 1])
    )
    # Cap-2000 view uses only stages <=2000 from the same 4000-stage model.
    prefix_cap_ok = best_iter_cap2000 <= CAP_LOW and bool(
        np.allclose(mean_at_2000, staged_test[best_iter_cap2000 - 1])
    )

    return {
        "fold": fold_idx + 1,
        "seeds": {
            "outer_construction_seed": OUTER_SEED,
            "inner_seed": inner_seed,
            "stage_model_seed": stage_seed,
            "full_model_seed": full_seed,
        },
        "splits": {
            "outer_train_hash": index_hash(train_index),
            "outer_test_hash": index_hash(test_index),
            "inner_train_hash": index_hash(inner_train_abs),
            "inner_val_hash": index_hash(inner_val_abs),
            "n_outer_train": int(len(train_index)),
            "n_outer_test": int(len(test_index)),
            "n_inner_train": int(len(inner_train_abs)),
            "n_inner_val": int(len(inner_val_abs)),
            "outer_train_indices": train_index.astype(int).tolist(),
            "outer_test_indices": test_index.astype(int).tolist(),
            "inner_train_indices": inner_train_abs.astype(int).tolist(),
            "inner_val_indices": inner_val_abs.astype(int).tolist(),
        },
        "best_iter_cap2000": best_iter_cap2000,
        "best_iter_cap4000": best_iter_cap4000,
        "val_rmse_at_best_cap2000": float(val_rmse[best_iter_cap2000 - 1]),
        "val_rmse_at_best_cap4000": float(val_rmse[best_iter_cap4000 - 1]),
        "cap2000": {
            "test_rmse": test_2000["rmse"],
            "test_nll": test_2000["nll"],
            "val_rmse_full_model": val_at_2000["rmse"],
            "val_nll_full_model": val_at_2000["nll"],
            "capped_at_2000": best_iter_cap2000 == CAP_LOW,
        },
        "cap4000": {
            "test_rmse": test_4000["rmse"],
            "test_nll": test_4000["nll"],
            "val_rmse_full_model": val_at_4000["rmse"],
            "val_nll_full_model": val_at_4000["nll"],
            "capped_at_4000": best_iter_cap4000 == N_EST,
        },
        "delta_4000_minus_2000": {
            "test_nll": test_4000["nll"] - test_2000["nll"],
            "test_rmse": test_4000["rmse"] - test_2000["rmse"],
            "best_iter": best_iter_cap4000 - best_iter_cap2000,
        },
        "verification": {
            "n_stages_fitted": N_EST,
            "shared_val_rmse_prefix_len": CAP_LOW,
            "shared_prefix_ok": shared_prefix_ok,
            "shared_pred_ok": shared_pred_ok,
            "prefix_cap_ok": prefix_cap_ok,
            "best_iters_agree_when_global_le_2000": (
                best_iter_cap2000 == best_iter_cap4000
                if best_iter_cap4000 <= CAP_LOW
                else True
            ),
        },
        "timing_sec": {
            "stage_fit": stage_fit_sec,
            "full_fit": full_fit_sec,
            "fold_total": stage_fit_sec + full_fit_sec,
        },
    }


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "stdout.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    if log_path.exists():
        log_path.unlink()

    X, y, data_source = load_yacht()
    log(f"== Dataset=yacht X.shape={X.shape} MLE/Normal controlled-cap diagnostic")
    log(f"== data_source={json.dumps(data_source)}")
    log(
        f"== N_EST={N_EST} CAP_LOW={CAP_LOW} OUTER_SEED={OUTER_SEED} "
        f"INNER_SEED_BASE={INNER_SEED_BASE}"
    )

    folds = make_outer_folds(X.shape[0], N_SPLITS, OUTER_SEED)
    per_fold = []
    wall0 = time.perf_counter()

    for fold_idx, (train_index, test_index) in enumerate(folds):
        result = run_fold(fold_idx, X, y, train_index, test_index)
        per_fold.append(result)
        log(
            "[%d/%d] BestIter2000=%d BestIter4000=%d "
            "RMSE: Val2000=%.4f Val4000=%.4f "
            "TestRMSE2000=%.4f TestRMSE4000=%.4f "
            "TestNLL2000=%.4f TestNLL4000=%.4f DeltaNLL=%.4f"
            % (
                fold_idx + 1,
                N_SPLITS,
                result["best_iter_cap2000"],
                result["best_iter_cap4000"],
                result["val_rmse_at_best_cap2000"],
                result["val_rmse_at_best_cap4000"],
                result["cap2000"]["test_rmse"],
                result["cap4000"]["test_rmse"],
                result["cap2000"]["test_nll"],
                result["cap4000"]["test_nll"],
                result["delta_4000_minus_2000"]["test_nll"],
            )
        )

    wall = time.perf_counter() - wall0

    nll_2000 = [r["cap2000"]["test_nll"] for r in per_fold]
    nll_4000 = [r["cap4000"]["test_nll"] for r in per_fold]
    rmse_2000 = [r["cap2000"]["test_rmse"] for r in per_fold]
    rmse_4000 = [r["cap4000"]["test_rmse"] for r in per_fold]
    deltas = [r["delta_4000_minus_2000"]["test_nll"] for r in per_fold]

    summary = {
        "handoff": "017F1-NG-D1",
        "design": "shared_4000_stage_trajectories_frozen_splits",
        "a01_confound": (
            "Official regression_exp.py uses train_test_split without random_state "
            "inside the fold loop; fitting to 4000 vs 2000 consumes different global "
            "RNG amounts, so later folds get different inner val partitions. a01 "
            "paired comparison is confounded and must not be treated as a pure cap effect."
        ),
        "data_source": data_source,
        "protocol": {
            "n_splits": N_SPLITS,
            "n_estimators": N_EST,
            "cap_low": CAP_LOW,
            "learning_rate": LR,
            "natural_gradient": True,
            "distn": "Normal",
            "score": "MLE",
            "base": "tree",
            "outer_seed": OUTER_SEED,
            "inner_seed_base": INNER_SEED_BASE,
            "stage_model_seed_base": STAGE_MODEL_SEED_BASE,
            "full_model_seed_base": FULL_MODEL_SEED_BASE,
            "stage_selection": "argmin validation RMSE on shared staged trajectory",
        },
        "per_fold": per_fold,
        "aggregates": {
            "cap2000": {
                "test_nll": stats(nll_2000),
                "test_rmse": stats(rmse_2000),
                "best_iter": stats([r["best_iter_cap2000"] for r in per_fold]),
                "cap_hit_exact_2000": sum(
                    1 for r in per_fold if r["best_iter_cap2000"] == CAP_LOW
                ),
            },
            "cap4000": {
                "test_nll": stats(nll_4000),
                "test_rmse": stats(rmse_4000),
                "best_iter": stats([r["best_iter_cap4000"] for r in per_fold]),
                "cap_hit_exact_4000": sum(
                    1 for r in per_fold if r["best_iter_cap4000"] == N_EST
                ),
                "cap_hit_ge_3990": sum(
                    1 for r in per_fold if r["best_iter_cap4000"] >= 3990
                ),
            },
            "paired": {
                "mean_delta_test_nll": float(np.mean(deltas)),
                "median_delta_test_nll": float(np.median(deltas)),
                "nll_4000_wins": sum(1 for d in deltas if d < 0),
                "nll_ties": sum(1 for d in deltas if d == 0),
                "nll_4000_losses": sum(1 for d in deltas if d > 0),
                "rmse_4000_wins": sum(
                    1
                    for r in per_fold
                    if r["delta_4000_minus_2000"]["test_rmse"] < 0
                ),
                "rmse_4000_losses": sum(
                    1
                    for r in per_fold
                    if r["delta_4000_minus_2000"]["test_rmse"] > 0
                ),
                "folds_where_4000_selects_beyond_2000": [
                    r["fold"]
                    for r in per_fold
                    if r["best_iter_cap4000"] > CAP_LOW
                ],
                "folds_where_best_iters_differ": [
                    r["fold"]
                    for r in per_fold
                    if r["best_iter_cap2000"] != r["best_iter_cap4000"]
                ],
            },
        },
        "verification_all_folds": {
            "shared_pred_ok": all(r["verification"]["shared_pred_ok"] for r in per_fold),
            "prefix_cap_ok": all(r["verification"]["prefix_cap_ok"] for r in per_fold),
            "best_iters_agree_when_global_le_2000": all(
                r["verification"]["best_iters_agree_when_global_le_2000"] for r in per_fold
            ),
        },
        "wall_time_sec": wall,
    }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    (out_dir / "split-indices.json").write_text(
        json.dumps(
            {
                "protocol": summary["protocol"],
                "folds": [
                    {"fold": r["fold"], "seeds": r["seeds"], "splits": r["splits"]}
                    for r in per_fold
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    log(
        "== CAP2000 NLL=%.4f +/- %.4f RMSE=%.4f +/- %.4f"
        % (
            summary["aggregates"]["cap2000"]["test_nll"]["mean"],
            summary["aggregates"]["cap2000"]["test_nll"]["pop_sd"],
            summary["aggregates"]["cap2000"]["test_rmse"]["mean"],
            summary["aggregates"]["cap2000"]["test_rmse"]["pop_sd"],
        )
    )
    log(
        "== CAP4000 NLL=%.4f +/- %.4f RMSE=%.4f +/- %.4f"
        % (
            summary["aggregates"]["cap4000"]["test_nll"]["mean"],
            summary["aggregates"]["cap4000"]["test_nll"]["pop_sd"],
            summary["aggregates"]["cap4000"]["test_rmse"]["mean"],
            summary["aggregates"]["cap4000"]["test_rmse"]["pop_sd"],
        )
    )
    log(
        "== paired NLL wins/ties/losses=%d/%d/%d mean_delta=%.4f"
        % (
            summary["aggregates"]["paired"]["nll_4000_wins"],
            summary["aggregates"]["paired"]["nll_ties"],
            summary["aggregates"]["paired"]["nll_4000_losses"],
            summary["aggregates"]["paired"]["mean_delta_test_nll"],
        )
    )
    log(f"ELAPSED_SEC={wall:.2f}")
    log("EXIT_CODE=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
