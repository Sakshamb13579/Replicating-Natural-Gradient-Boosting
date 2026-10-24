#!/usr/bin/env python3
"""Validate persisted predictions and aggregate the frozen Assignment 3 batch."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_assessment3_compute import DEFAULT_BATCH, checksums, load_input, measure_predictions, verify_batch, write_json


def summary(values) -> dict:
    a = np.asarray(values, dtype=float)
    return {"n": len(a), "mean": float(a.mean()),
            "sample_sd": float(a.std(ddof=1)) if len(a) > 1 else None,
            "se": float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else None,
            "min": float(a.min()), "max": float(a.max())}


def paired(values) -> dict:
    a = np.asarray(values, dtype=float)
    return {**summary(a), "first_wins": int((a < -1e-12).sum()),
            "ties": int((np.abs(a) <= 1e-12).sum()), "second_wins": int((a > 1e-12).sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    batch = args.batch.resolve()
    doc = verify_batch(batch)
    rows = []
    per_dataset = []
    missing = []
    inputs = {}
    for task in doc["tasks"]:
        name, variant = task["dataset"], task["variant"]
        directory = batch / "runs" / name / variant
        if not (directory / "complete.json").exists():
            missing.append(task)
            continue
        checksums(directory, verify=True)
        saved = json.loads((directory / "metrics.json").read_text())
        if not saved["reportable"]:
            raise ValueError("nonreportable task in main aggregate")
        if name not in inputs:
            inputs[name] = load_input(batch, name)
        frame, manifest, folds = inputs[name]
        if len(saved["per_fold"]) != len(folds):
            raise ValueError("incomplete task fold count")
        for metric, fold in zip(saved["per_fold"], folds):
            folder = directory / f"fold-{fold['fold_index']:02d}"
            checksums(folder, verify=True)
            pred = pd.read_csv(folder / "predictions.csv")
            expected = frame.iloc[fold["outer_test_indices"]]
            if pred.row_id.tolist() != expected.row_id.tolist():
                raise ValueError("test prediction row order mismatch")
            if not np.allclose(pred.y_true, expected[manifest["target"]], rtol=0, atol=1e-12):
                raise ValueError("test labels mismatch")
            recalculated = measure_predictions(pred)
            for key in ("test_nll", "test_rmse"):
                if not np.isclose(metric[key], recalculated[key], rtol=1e-12, atol=1e-12):
                    raise ValueError(f"metric recomputation failed: {key}")
            for key in ("coverage", "mean_width"):
                for level in ("50", "80", "95"):
                    if not np.isclose(metric[key][level], recalculated[key][level], rtol=1e-12, atol=1e-12):
                        raise ValueError("interval metric mismatch")
            trajectory = pd.read_csv(folder / "validation-trajectory.csv")
            if int(trajectory.stage.iloc[int(trajectory.validation_nll.argmin())]) != metric["best_iter"]:
                raise ValueError("selected stage not validation argmin")
            if variant == "homoscedastic":
                if not np.allclose(pred.pred_scale, metric["fixed_sigma_from_stage_val"], rtol=1e-12):
                    raise ValueError("homoscedastic scale not constant retained sigma")
            row = {"dataset": name, "variant": variant, "fold": fold["fold_index"],
                   "replicate": manifest.get("replicate"), "regime": manifest.get("regime"),
                   "nll": metric["test_nll"], "rmse": metric["test_rmse"],
                   "best_iter": metric["best_iter"], "cap_hit": metric["cap_hit"],
                   "wall_sec": metric["timing_sec"]["wall"]}
            for level in ("50", "80", "95"):
                row[f"coverage_{level}"] = metric["coverage"][level]
                row[f"width_{level}"] = metric["mean_width"][level]
            if "scale_mae" in metric:
                row.update({"scale_mae": metric["scale_mae"], "pearson": metric["scale_correlation"]["pearson"],
                            "spearman": metric["scale_correlation"]["spearman"], "oracle_nll": metric["oracle_nll"]})
            rows.append(row)
        per_dataset.append({"dataset": name, "variant": variant,
                            "nll": summary([m["test_nll"] for m in saved["per_fold"]]),
                            "rmse": summary([m["test_rmse"] for m in saved["per_fold"]]),
                            "cap_hits": sum(m["cap_hit"] for m in saved["per_fold"])})
    if missing and not args.allow_partial:
        raise RuntimeError(f"{len(missing)} tasks are not complete; use --allow-partial for progress only")
    table = pd.DataFrame(rows)
    output = batch / "aggregate"
    output.mkdir(exist_ok=True)
    table.to_csv(output / "fold-ledger.csv", index=False)
    differences = []
    if not table.empty:
        for dataset, subset in table.groupby("dataset"):
            pivot = subset.pivot(index="fold", columns="variant", values="nll")
            for first, second in itertools.combinations(("natural", "ordinary", "homoscedastic"), 2):
                if first in pivot and second in pivot and not pivot[[first, second]].isna().any().any():
                    differences.append({"dataset": dataset, "first": first, "second": second,
                                        "difference": "first_minus_second", **paired(pivot[first] - pivot[second])})
    synthetic = []
    synthetic_differences = []
    if not table.empty and "replicate" in table:
        synth = table[table.replicate.notna()]
        for (regime, variant), subset in synth.groupby(["regime", "variant"]):
            item = {"regime": regime, "variant": variant,
                    "replicates": sorted(subset.replicate.astype(int).tolist())}
            for metric in ("nll", "rmse", "scale_mae", "pearson", "coverage_50", "coverage_80", "coverage_95"):
                values = subset[metric].dropna()
                item[metric] = summary(values) if len(values) else None
            synthetic.append(item)
        for regime, subset in synth.groupby("regime"):
            pivot = subset.pivot(index="replicate", columns="variant", values="nll")
            for first, second in itertools.combinations(("natural", "ordinary", "homoscedastic"), 2):
                if first in pivot and second in pivot:
                    values = (pivot[first] - pivot[second]).dropna()
                    if len(values):
                        synthetic_differences.append({"regime": regime, "first": first, "second": second,
                                                      "difference": "first_minus_second", **paired(values)})
    result = {"status": "partial" if missing else "complete", "missing_tasks": missing,
              "validated_variant_fold_evaluations": len(rows), "dataset_variant_summaries": per_dataset,
              "paired_nll_differences": differences, "synthetic_seed_summaries": synthetic,
              "synthetic_paired_nll_differences": synthetic_differences,
              "interpretation": ["Fold summary SE is descriptive, not an iid inferential SE for overlapping holdouts",
                                 "Synthetic SE uses fresh independent data/split replicates, paired within replicate",
                                 "Do not pair row-split Airfoil with grouped Airfoil folds or pool subset/full Superconductivity",
                                 "v1 synthetic seed is historical reference and not included here"]}
    write_json(output / "summary.json", result)
    lines = ["# Assignment 3 compute results", "", f"Status: **{result['status']}**; validated evaluations: {len(rows)}.", "",
             "## External datasets", "", "| Dataset | Variant | Mean fold NLL | Mean fold RMSE | Cap hits |",
             "|---|---|---:|---:|---:|"]
    for item in per_dataset:
        if not item["dataset"].startswith("synthetic"):
            lines.append(f"| {item['dataset']} | {item['variant']} | {item['nll']['mean']:.6f} | {item['rmse']['mean']:.6f} | {item['cap_hits']} |")
    lines += ["", "## Fresh synthetic replicates", "", "| Regime | Variant | Replicates | Mean NLL | NLL SD | Mean scale MAE |",
              "|---|---|---:|---:|---:|---:|"]
    for item in synthetic:
        sd = item["nll"]["sample_sd"]
        sd_text = f"{sd:.6f}" if sd is not None else "NA"
        lines.append(f"| {item['regime']} | {item['variant']} | {item['nll']['n']} | {item['nll']['mean']:.6f} | {sd_text} | {item['scale_mae']['mean']:.6f} |")
    lines += ["", "## Interpretation", ""] + ["- " + text for text in result["interpretation"]]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    checksums(output)
    print(json.dumps({"status": result["status"], "validated_evaluations": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
