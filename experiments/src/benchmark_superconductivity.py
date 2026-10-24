#!/usr/bin/env python3
"""Time short, non-reportable NGBoost fits for the Superconductivity scope gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal, NormalFixedVar
from ngboost.learners import default_tree_learner
from ngboost.scores import MLE
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "superconductivity" / "train.csv"
EXPECTED_SHA256 = "4dfb6e3a1f6ffd969e5a5e42f093c4800d1e2a6c8b1e309f8fcd9f23d86952f3"
VARIANTS = ("natural", "ordinary", "homoscedastic")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_model(variant: str, n_estimators: int, seed: int) -> NGBRegressor:
    return NGBRegressor(
        Base=default_tree_learner,
        Dist=NormalFixedVar if variant == "homoscedastic" else Normal,
        Score=MLE,
        n_estimators=n_estimators,
        learning_rate=0.01,
        natural_gradient=variant != "ordinary",
        minibatch_frac=1.0,
        verbose=False,
        random_state=seed,
    )


def timed_fit(
    variant: str, X: np.ndarray, y: np.ndarray, n_estimators: int, seed: int
) -> float:
    start = time.perf_counter()
    make_model(variant, n_estimators, seed).fit(X, y)
    return time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_json")
    parser.add_argument("--n-estimators", type=int, default=25)
    args = parser.parse_args()
    if args.n_estimators < 5 or args.n_estimators > 100:
        raise ValueError("benchmark n-estimators must be between 5 and 100")

    raw_sha = sha256_file(RAW)
    if raw_sha != EXPECTED_SHA256:
        raise ValueError(f"raw hash mismatch: {raw_sha}")
    frame = pd.read_csv(RAW)
    if frame.shape != (21_263, 82) or frame.columns[-1] != "critical_temp":
        raise ValueError(f"unexpected table: {frame.shape}, {frame.columns[-1]}")
    X = np.ascontiguousarray(frame.iloc[:, :-1], dtype=np.float64)
    y = np.ascontiguousarray(frame["critical_temp"], dtype=np.float64)

    rng = np.random.RandomState(1)
    permutation = rng.choice(len(frame), len(frame), replace=False)
    boundary = round(len(frame) * 0.9)
    outer_train = permutation[:boundary]
    positions = np.arange(len(outer_train))
    inner_train_pos, _ = train_test_split(
        positions, test_size=0.2, random_state=10_000
    )
    inner_train = outer_train[inner_train_pos]

    results = []
    for index, variant in enumerate(VARIANTS):
        inner_seconds = timed_fit(
            variant, X[inner_train], y[inner_train], args.n_estimators, 20_000 + index
        )
        outer_seconds = timed_fit(
            variant, X[outer_train], y[outer_train], args.n_estimators, 30_000 + index
        )
        measured = inner_seconds + outer_seconds
        projected_fold = measured * (2_000 / args.n_estimators)
        results.append(
            {
                "variant": variant,
                "inner_rows": int(len(inner_train)),
                "outer_rows": int(len(outer_train)),
                "inner_fit_seconds": inner_seconds,
                "outer_refit_seconds": outer_seconds,
                "measured_seconds": measured,
                "projected_seconds_per_2000_stage_fold_linear": projected_fold,
                "projected_hours_for_5_folds_linear": projected_fold * 5 / 3600,
            }
        )
        print(
            f"{variant}: {measured:.1f}s measured; "
            f"{projected_fold / 60:.1f} min/fold linear projection",
            flush=True,
        )

    total_hours = sum(row["projected_hours_for_5_folds_linear"] for row in results)
    document = {
        "status": "non_reportable_timing_benchmark",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Choose full-five-fold or deterministic-5000-row Superconductivity scope before freezing splits",
        "raw_path": str(RAW.relative_to(ROOT)),
        "raw_sha256": raw_sha,
        "table_shape": list(frame.shape),
        "n_estimators_measured": args.n_estimators,
        "reportable_model_metrics_computed": False,
        "results": results,
        "projected_total_hours_full_data_5_folds_3_variants_linear": total_hours,
        "projection_caution": "Linear stage extrapolation from a short fit; use as a compute gate, not performance evidence.",
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite benchmark: {out_path}")
    out_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"projected full-data total: {total_hours:.2f} hours")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
