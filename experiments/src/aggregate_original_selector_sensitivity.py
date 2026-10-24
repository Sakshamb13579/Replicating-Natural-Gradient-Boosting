#!/usr/bin/env python3
"""Aggregate six controlled original-data selector-sensitivity run directories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("yacht", "kin8nm")
VARIANTS = ("natural", "ordinary", "homoscedastic")

PAPER = {
    ("yacht", "natural"): {"table2_nll_mean": 0.20, "table2_nll_se": 0.26, "table3_rmse_mean": 0.50, "table3_rmse_se": 0.20},
    ("yacht", "ordinary"): {"table2_nll_mean": 3.29, "table2_nll_se": 0.20, "table3_rmse_mean": None, "table3_rmse_se": None},
    ("yacht", "homoscedastic"): {"table2_nll_mean": 2.02, "table2_nll_se": 0.21, "table3_rmse_mean": None, "table3_rmse_se": None},
    ("kin8nm", "natural"): {"table2_nll_mean": -0.49, "table2_nll_se": 0.02, "table3_rmse_mean": 0.16, "table3_rmse_se": 0.00},
    ("kin8nm", "ordinary"): {"table2_nll_mean": -0.52, "table2_nll_se": 0.03, "table3_rmse_mean": None, "table3_rmse_se": None},
    ("kin8nm", "homoscedastic"): {"table2_nll_mean": -0.37, "table2_nll_se": 0.05, "table3_rmse_mean": None, "table3_rmse_se": None},
}

# Only values explicitly reported by the preserved authoritative reports are populated.
HISTORICAL = {
    ("yacht", "natural"): {"folds": 20, "cap_hits": 11, "nll_mean": 2.1060, "nll_median": 1.7924, "nll_pop_sd": 2.3466, "nll_sample_sd": 2.4075, "nll_se": 0.5247, "rmse_mean": 0.5576, "rmse_median": 0.5232},
    ("yacht", "ordinary"): {"folds": 20, "cap_hits": 19, "nll_mean": 11.5473, "nll_median": 6.7826, "nll_pop_sd": 12.4147, "nll_sample_sd": 12.7373, "nll_se": 2.7760, "rmse_mean": 13.8165, "rmse_median": 13.8304},
    ("yacht", "homoscedastic"): {"folds": 20, "cap_hits": 6, "nll_mean": 13.2821, "nll_median": 9.6021, "nll_pop_sd": 11.6455, "nll_sample_sd": 11.9481, "nll_se": 2.6040, "rmse_mean": 0.5508, "rmse_median": 0.4667},
    ("kin8nm", "natural"): {"folds": 5, "cap_hits": 3, "nll_mean": -0.4191, "nll_median": -0.4253, "nll_pop_sd": 0.0158, "nll_sample_sd": 0.0177, "nll_se": 0.0071, "rmse_mean": 0.1692, "rmse_median": None},
    ("kin8nm", "ordinary"): {"folds": 5, "cap_hits": 1, "nll_mean": -0.5078, "nll_median": -0.5167, "nll_pop_sd": 0.0224, "nll_sample_sd": 0.0251, "nll_se": 0.0100, "rmse_mean": 0.1566, "rmse_median": None},
    ("kin8nm", "homoscedastic"): {"folds": 5, "cap_hits": 2, "nll_mean": -0.3623, "nll_median": -0.3756, "nll_pop_sd": 0.0301, "nll_sample_sd": 0.0337, "nll_se": 0.0135, "rmse_mean": 0.1608, "rmse_median": None},
}

SOURCES = {
    "paper": "Duan et al., Tables 2 and 3; values transcribed in reports/original-selector-sensitivity-protocol-v1.md",
    "historical_yacht": "artifacts/released-code-run/implementation-report-ng1b.md",
    "historical_kin8nm": "artifacts/released-code-run/implementation-report-ng1c.md",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_stats(row: dict, prefix: str, values: dict) -> None:
    for key in ("mean", "median", "pop_sd", "sample_sd", "se", "n"):
        row[f"{prefix}_{key}"] = values[key]


def load_runs(run_dirs: list[Path], allow_non_reportable: bool) -> dict[tuple[str, str], dict]:
    runs = {}
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        manifest_path = run_dir / "manifest.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        key = (metrics["dataset"], metrics["variant"])
        if key not in PAPER:
            raise ValueError(f"unexpected dataset/variant in {run_dir}: {key}")
        if key in runs:
            raise ValueError(f"duplicate dataset/variant: {key}")
        if manifest.get("status") != "complete":
            raise ValueError(f"incomplete run: {run_dir}")
        if sha256_file(metrics_path) != manifest.get("metrics_sha256"):
            raise ValueError(f"metrics checksum mismatch: {run_dir}")
        if not metrics.get("reportable") and not allow_non_reportable:
            raise ValueError(f"non-reportable run rejected: {run_dir}; use --allow-non-reportable only to test aggregation")
        runs[key] = {"dir": str(run_dir), "metrics": metrics, "manifest_sha256": sha256_file(manifest_path), "metrics_sha256": sha256_file(metrics_path)}
    expected = set(PAPER)
    if set(runs) != expected:
        raise ValueError(f"expected all six combinations; missing={sorted(expected - set(runs))}, extra={sorted(set(runs) - expected)}")
    return runs


def make_row(dataset: str, variant: str, run: dict) -> dict:
    metrics = run["metrics"]
    aggregate = metrics["aggregate"]
    paper = PAPER[(dataset, variant)]
    historical = HISTORICAL[(dataset, variant)]
    row = {
        "dataset": dataset,
        "variant": variant,
        "controlled_reportable": metrics["reportable"],
        "controlled_folds": metrics["sample_counts"]["folds_run"],
        **paper,
        **{f"historical_{key}": value for key, value in historical.items()},
        "historical_rmse_pop_sd": None,
        "historical_rmse_sample_sd": None,
        "historical_rmse_se": None,
        "rmse_selector_cap_hits": aggregate["selectors"]["rmse"]["cap_hits"],
        "nll_selector_cap_hits": aggregate["selectors"]["nll"]["cap_hits"],
        "selected_stage_changed_folds": aggregate["selected_stage_changed_folds"],
        "run_dir": run["dir"],
    }
    for selector in ("rmse", "nll"):
        for metric in ("test_nll", "test_rmse", "selected_stage"):
            flatten_stats(row, f"controlled_{selector}_selector_{metric}", aggregate["selectors"][selector][metric])
    for metric in ("selected_stage", "test_nll", "test_rmse"):
        flatten_stats(row, f"paired_delta_{metric}_nll_minus_rmse", aggregate["paired_deltas"][metric])
        counts = aggregate["win_tie_counts"][metric]
        for key, value in counts.items():
            row[f"paired_{metric}_{key}"] = value
    row["rmse_selector_abs_nll_gap_from_table2"] = abs(row["controlled_rmse_selector_test_nll_mean"] - paper["table2_nll_mean"])
    row["nll_selector_abs_nll_gap_from_table2"] = abs(row["controlled_nll_selector_test_nll_mean"] - paper["table2_nll_mean"])
    row["abs_nll_gap_change_nll_minus_rmse_selector"] = row["nll_selector_abs_nll_gap_from_table2"] - row["rmse_selector_abs_nll_gap_from_table2"]
    return row


def markdown(rows: list[dict], reportable: bool) -> str:
    lines = [
        "# Original-Data Selector Sensitivity Comparison",
        "",
        f"**Reportable:** `{str(reportable).lower()}`",
        "",
        "| Dataset | Variant | Paper T2 NLL | Paper T3 RMSE | Historical NLL/RMSE | RMSE-selector NLL/RMSE | NLL-selector NLL/RMSE | Delta NLL/RMSE | NLL gap change | Stages RMSE/NLL | Cap hits RMSE/NLL |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {variant} | {paper:.4f} | {paper_rmse} | {historical:.4f}/{historical_rmse:.4f} | {rnll:.4f}/{rrmse:.4f} | {nnll:.4f}/{nrmse:.4f} | {dnll:+.4f}/{drmse:+.4f} | {gap:+.4f} | {rs:.1f}/{ns:.1f} | {rc}/{nc} |".format(
                dataset=row["dataset"], variant=row["variant"], paper=row["table2_nll_mean"],
                paper_rmse="" if row["table3_rmse_mean"] is None else f"{row['table3_rmse_mean']:.4f}",
                historical=row["historical_nll_mean"], historical_rmse=row["historical_rmse_mean"],
                rnll=row["controlled_rmse_selector_test_nll_mean"], nnll=row["controlled_nll_selector_test_nll_mean"], dnll=row["paired_delta_test_nll_nll_minus_rmse_mean"],
                rrmse=row["controlled_rmse_selector_test_rmse_mean"], nrmse=row["controlled_nll_selector_test_rmse_mean"], drmse=row["paired_delta_test_rmse_nll_minus_rmse_mean"],
                gap=row["abs_nll_gap_change_nll_minus_rmse_selector"],
                rs=row["controlled_rmse_selector_selected_stage_mean"], ns=row["controlled_nll_selector_selected_stage_mean"], rc=row["rmse_selector_cap_hits"], nc=row["nll_selector_cap_hits"],
            )
        )
    lines.extend([
        "",
        "## Comparability Notes",
        "",
        "- Paper Table 2 NLL values use the paper's best-validation-log-likelihood description; Table 3 RMSE is a direct benchmark only for natural NGBoost.",
        "- Historical local results used released-script validation-MSE selection, unpreserved RNG-coupled inner splits, and remain a separate baseline.",
        "- Controlled results use newly frozen inner splits and paired selectors on one inner trajectory and one outer refit trajectory per fold.",
        "- Kin8nm controlled and historical evidence uses five folds; the paper used twenty, so precision is not equivalent.",
        "- Ordinary and homoscedastic RMSE values are local diagnostics without matching published Table 3 ablation benchmarks.",
        "- Homoscedastic scaling follows the released script's post-refit validation-residual rule, not the paper's stated training-residual definition.",
        "- The paper's 2nd-Order arm is omitted because it is not implemented in the released library.",
        "- Blank historical RMSE dispersion fields are unavailable in the authoritative preserved reports and were not fabricated.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-non-reportable", action="store_true", help="aggregator smoke test only")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("run_dirs", nargs=6)
    args = parser.parse_args()
    out_dir = Path(args.out_dir).resolve()
    if not out_dir.is_relative_to(ROOT):
        parser.error(f"output directory must be under {ROOT}")
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {out_dir}")
    runs = load_runs([Path(path).resolve() for path in args.run_dirs], args.allow_non_reportable)
    rows = [make_row(dataset, variant, runs[(dataset, variant)]) for dataset in DATASETS for variant in VARIANTS]
    reportable = all(row["controlled_reportable"] for row in rows)
    if args.allow_non_reportable:
        reportable = False
    notes = [
        "Controlled sensitivity is not the paper's unpublished execution and not a bit-for-bit historical rerun.",
        "Paper Table 3 RMSE is comparable only to controlled natural NGBoost, with protocol differences disclosed.",
        "Historical null fields mean unavailable in preserved authoritative reports, not zero.",
        "Kin8nm five-fold uncertainty is not equivalent to the paper's twenty repetitions.",
        "Homoscedastic post-refit validation scaling differs from the paper's stated training-residual definition.",
        "The unimplemented paper 2nd-Order arm is omitted.",
    ]
    document = {
        "schema_version": 1,
        "reportable": reportable,
        "non_reportable_reason": "--allow-non-reportable aggregation smoke test" if args.allow_non_reportable else None,
        "sources": SOURCES,
        "comparability_notes": notes,
        "rows": rows,
        "run_checksums": {f"{dataset}/{variant}": {key: value for key, value in run.items() if key != "metrics"} for (dataset, variant), run in runs.items()},
    }
    out_dir.mkdir(parents=True)
    json_path = out_dir / "comparison.json"
    csv_path = out_dir / "comparison.csv"
    md_path = out_dir / "comparison.md"
    json_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(markdown(rows, reportable), encoding="utf-8")
    (out_dir / "checksums.txt").write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in (csv_path, json_path, md_path)), encoding="utf-8")
    print(json.dumps({"status": "complete", "reportable": reportable, "out_dir": str(out_dir), "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
