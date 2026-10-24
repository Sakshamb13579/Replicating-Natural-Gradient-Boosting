#!/usr/bin/env python3
"""Recheck historical constructed artifacts and add conditional coverage, without fitting."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from run_assessment3_compute import DEFAULT_BATCH, ROOT, checksums, measure_predictions, sha256_file, write_json


def main() -> None:
    source = ROOT / "data/processed/heteroscedastic-v1/heteroscedastic-v1.csv"
    manifest = json.loads((ROOT / "data/manifests/heteroscedastic-v1.json").read_text())
    if sha256_file(source) != manifest["checksums"][source.name]:
        raise ValueError("historical data hash mismatch")
    frame = pd.read_csv(source)
    if len(frame) != 12000:
        raise ValueError("unexpected historical row count")
    if not np.allclose(frame.y, frame.true_mu + frame.true_sigma * frame.noise_z, rtol=0, atol=1e-13):
        raise ValueError("generator formula mismatch")
    split = manifest["split"]
    expected_sets = {label: set(split[key]) for label, key in
                     (("train", "train_row_ids"), ("validation", "validation_row_ids"), ("test", "test_row_ids"))}
    if any(a & b for i, a in enumerate(expected_sets.values()) for b in list(expected_sets.values())[i + 1:]):
        raise ValueError("historical split leakage")
    bases = [frame[frame.regime == regime].reset_index(drop=True) for regime in "ABC"]
    for other in bases[1:]:
        pd.testing.assert_frame_equal(bases[0][["row_id", "x1", "x2", "true_mu", "noise_z", "split"]],
                                      other[["row_id", "x1", "x2", "true_mu", "noise_z", "split"]])
    paths = {
        "natural": "ngboost-natural/028-ng-p5-heteroscedastic-natural-s0-a01",
        "ordinary": "ngboost-ordinary/029-ng-p5-heteroscedastic-ordinary-s0-a01",
        "homoscedastic": "ngboost-homoscedastic/030-ng-p5-heteroscedastic-homoscedastic-s0-a01",
    }
    results = []
    hash_count = 0
    manifest_hash_exceptions = []
    for variant, relative in paths.items():
        directory = ROOT / "artifacts/heteroscedastic-v1" / relative
        for line in (directory / "checksums.txt").read_text().splitlines():
            if not re.match(r"^[a-f0-9]{64}  ", line):
                continue
            expected_hash, name = line.split("  ", 1)
            path = Path(name) if Path(name).is_absolute() else directory / name
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                if path.name != "manifest.json":
                    raise ValueError(f"historical artifact changed: {path}")
                manifest_hash_exceptions.append({"variant": variant, "path": str(path),
                                                 "recorded_sha256": expected_hash, "actual_sha256": actual_hash})
            else:
                hash_count += 1
        saved = json.loads((directory / "metrics.json").read_text())
        if saved["protocol"]["features"] != ["x1", "x2"]:
            raise ValueError("ground truth or unexpected inputs in model features")
        pred = pd.read_csv(directory / "predictions.csv")
        if len(pred) != 2400:
            raise ValueError("historical prediction count mismatch")
        for regime, group in pred.groupby("regime"):
            group = group.reset_index(drop=True)
            reference = frame[(frame.regime == regime) & (frame.split == "test")].reset_index(drop=True)
            if set(group.row_id) != expected_sets["test"] or len(group) != 800:
                raise ValueError("unexpected historical test rows")
            reference = reference.set_index("row_id").loc[group.row_id]
            for name, column in (("y_true", "y"), ("true_mu", "true_mu"), ("true_sigma", "true_sigma"), ("x1", "x1"), ("x2", "x2")):
                if not np.allclose(group[name], reference[column], rtol=0, atol=1e-13):
                    raise ValueError("historical prediction / frozen data mismatch")
            metric = measure_predictions(group)
            old = next(m for m in saved["per_regime"] if m["regime"] == regime)
            for key, old_key in (("test_nll", "test_nll"), ("test_rmse", "test_rmse"),
                                 ("oracle_nll", "oracle_nll"), ("scale_mae", "scale_mae_vs_true_sigma")):
                if not np.isclose(metric[key], old[old_key], rtol=1e-10, atol=1e-12):
                    raise ValueError((variant, regime, key))
            for key in ("coverage", "mean_width"):
                for level in ("50", "80", "95"):
                    if not np.isclose(metric[key][level], old[key][level], rtol=1e-10, atol=1e-12):
                        raise ValueError("historical interval metric mismatch")
            if group.best_iter.nunique() != 1 or int(group.best_iter.iloc[0]) != old["best_iter"]:
                raise ValueError("historical selected-stage metadata mismatch")
            if variant == "homoscedastic":
                if not np.allclose(group.pred_scale, old["fixed_sigma_from_stage_val"], rtol=1e-12):
                    raise ValueError("historical retained sigma mismatch")
                if old["sigma_reestimated_after_full_fit"]:
                    raise ValueError("historical refit scale rule violated")
            results.append({"variant": variant, "regime": regime, **metric})
    output = DEFAULT_BATCH / "historical-constructed-audit"
    write_json(output / "audit.json", {
        "status": "metrics_passed_with_manifest_checksum_exceptions" if manifest_hash_exceptions else "passed",
        "historical_checksum_entries_verified": hash_count,
        "historical_manifest_checksum_exceptions": manifest_hash_exceptions,
        "frozen_data_sha256": sha256_file(source), "predictions_verified": 7200,
        "paired_regimes_and_disjoint_splits": True, "features": ["x1", "x2"],
        "recomputed_metrics_match_saved": True, "per_variant_regime": results,
        "limits": ["No models rerun; full historical validation trajectories are not saved",
                   "Selected stage agrees with predictions/metrics metadata, not independently refitted argmin",
                   "Source/config feature roles verified; no claim of fresh training trace audit",
                   "New fixed-width |x1| bins supplement original quantile-bin diagnostics"],
    })
    status_text = "METRICS PASSED; historical manifest checksum exceptions retained" if manifest_hash_exceptions else "PASSED"
    lines = ["# Historical constructed-data audit for Assignment 3", "", f"Status: **{status_text}**.", "",
             f"Verified {hash_count} checksum entries and all 7,200 saved test predictions.",
             "Recomputed NLL, RMSE, oracle NLL, scale MAE, coverage and widths match stored metrics.", "",
             "## Conditional 95% coverage by fixed |x1| bins", "",
             "| Variant | Regime | [0,.5) | [.5,1) | [1,1.5) | [1.5,2) |", "|---|---|---:|---:|---:|---:|"]
    for item in results:
        values = [f"{b['coverage']['95']:.3f}" for b in item["abs_x1_bins"]]
        lines.append(f"| {item['variant']} | {item['regime']} | " + " | ".join(values) + " |")
    lines += ["", "## Scope", "", "This is artifact verification, not retraining or repeated-seed robustness.",
              "Historical v1 remains unchanged. Fresh replicates are reported separately."]
    if manifest_hash_exceptions:
        lines += ["", "## Historical checksum exceptions", "",
                  "The following manifest.json files do not match their recorded checksums.",
                  "Their manifests list a hash of manifest.json itself, which cannot describe the final self-containing file.",
                  "All data, source, prediction and metrics checksums checked above match. No historical files were rewritten."]
        lines += [f"- {item['variant']}: recorded `{item['recorded_sha256']}`, current `{item['actual_sha256']}`."
                  for item in manifest_hash_exceptions]
    (output / "audit.md").write_text("\n".join(lines) + "\n")
    checksums(output)
    print(f"Historical constructed audit: metrics PASSED; {hash_count} hashes verified; "
          f"{len(manifest_hash_exceptions)} manifest hash exceptions; 7200 predictions, nine metric sets")


if __name__ == "__main__":
    main()
