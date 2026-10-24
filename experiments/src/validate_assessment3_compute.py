#!/usr/bin/env python3
"""Small nonreportable fits and independent cross-checks before long jobs launch."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import run_external_ngboost as previous
from run_assessment3_compute import DEFAULT_BATCH, checksums, load_input, task, write_json


def main() -> None:
    batch = DEFAULT_BATCH.resolve()
    records = []
    for dataset in ("superconductivity-full", "airfoil-grouped", "synthetic-r01-A"):
        for variant in ("natural", "ordinary", "homoscedastic"):
            task(SimpleNamespace(batch=batch, dataset=dataset, variant=variant, smoke=True))
            records.append({"dataset": dataset, "variant": variant, "smoke_passed": True})
    # The new per-fold checkpoint runner must reproduce the previously used engine.
    frame, manifest, folds = load_input(batch, "airfoil-grouped")
    for variant in ("natural", "ordinary", "homoscedastic"):
        old_metrics, old_predictions = previous.run_fold(
            variant, folds[0], frame.row_id.to_numpy(), frame.condition_id.to_numpy(),
            frame[manifest["features"]].to_numpy(), frame[manifest["target"]].to_numpy(), 25,
        )
        pred = pd.read_csv(batch / "smoke/airfoil-grouped" / variant / "fold-00/predictions.csv")
        reference = pd.DataFrame(old_predictions)
        for column in ("pred_mean", "pred_scale", "y_true", "best_iter"):
            if not np.allclose(pred[column], reference[column], atol=1e-10, rtol=1e-12):
                raise AssertionError((variant, column))
        if pred.best_iter.iloc[0] != old_metrics["best_iter"]:
            raise AssertionError("selected stage mismatch")
        records.append({"variant": variant, "previous_engine_predictions_match": True})
    # Check complete three-regime pairing across every fresh replicate without fitting.
    for replicate in range(1, 11):
        frames = [load_input(batch, f"synthetic-r{replicate:02d}-{regime}")[0] for regime in "ABC"]
        columns = ["row_id", "split", "x1", "x2", "true_mu", "noise_z"]
        for frame in frames[1:]:
            pd.testing.assert_frame_equal(frames[0][columns], frame[columns])
        if not np.allclose(frames[2].true_sigma, 2 * frames[0].true_sigma):
            raise AssertionError("C is not twice A scale")
        if not np.allclose(frames[1].true_sigma, .6):
            raise AssertionError("B control scale changed")
    output = batch / "validation"
    write_json(output / "preflight.json", {"status": "passed", "nonreportable_smoke_cap": 25,
                                           "checks": records, "all_ten_replicates_paired": True})
    checksums(output)
    print("Preflight PASSED: nine smoke runs, three independent engine matches, ten paired replicate inputs")


if __name__ == "__main__":
    main()
