#!/usr/bin/env python3
"""Aggregate completed external NGBoost output directories into CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


FIELDS = [
    "dataset", "variant", "split", "selector", "n_folds", "dataset_rows",
    "test_predictions", "test_nll", "test_rmse", "coverage_50", "width_50",
    "coverage_80", "width_80", "coverage_95", "width_95", "cap_hits",
    "runtime_seconds", "reportable", "limitations", "output_dir",
]
PAIRED_FIELDS = [
    "dataset", "variant_a", "variant_b", "n_folds", "mean_fold_nll_delta",
    "nll_a_wins", "nll_ties", "nll_b_wins", "mean_fold_rmse_delta",
    "rmse_a_wins", "rmse_ties", "rmse_b_wins",
]
PAIRS = (("natural", "ordinary"), ("natural", "homoscedastic"), ("ordinary", "homoscedastic"))
TIE_ATOL = 1e-12


def discover_metrics(inputs: list[Path]) -> list[Path]:
    found = set()
    for path in inputs:
        if path.is_file() and path.name == "metrics.json":
            found.add(path.resolve())
        elif path.is_dir() and (path / "metrics.json").is_file():
            found.add((path / "metrics.json").resolve())
        elif path.is_dir():
            found.update(candidate.resolve() for candidate in path.rglob("metrics.json"))
    return sorted(found)


def mean_metric(aggregate: dict, key: str) -> float:
    value = aggregate[key]
    return float(value["mean"] if isinstance(value, dict) else value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_from_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"dataset", "variant", "selector", "per_fold", "aggregate", "sample_counts"}
    if not required.issubset(data):
        raise ValueError(f"not an external-run metrics schema: {path}")
    aggregate = data["aggregate"]
    row = {
        "dataset": data["dataset"],
        "variant": data["variant"],
        "split": data.get("split_policy", ""),
        "selector": data["selector"],
        "n_folds": len(data["per_fold"]),
        "dataset_rows": data["sample_counts"]["dataset_rows"],
        "test_predictions": data["sample_counts"]["test_predictions"],
        "test_nll": mean_metric(aggregate, "test_nll"),
        "test_rmse": mean_metric(aggregate, "test_rmse"),
        "cap_hits": aggregate["cap_hits"],
        "runtime_seconds": data["timing_sec"]["wall"],
        "reportable": data.get("reportable", False),
        "limitations": "; ".join(data.get("limitations", []) + data.get("non_reportable_reasons", [])),
        "output_dir": str(path.parent),
    }
    for level in ("50", "80", "95"):
        row[f"coverage_{level}"] = mean_metric(aggregate["coverage_per_fold"], level)
        row[f"width_{level}"] = mean_metric(aggregate["mean_width_per_fold"], level)
    return row


def markdown(rows: list[dict]) -> str:
    display = [
        "dataset", "variant", "selector", "n_folds", "dataset_rows", "test_predictions",
        "test_nll", "test_rmse", "coverage_50", "width_50", "coverage_80", "width_80",
        "coverage_95", "width_95", "cap_hits", "runtime_seconds", "reportable", "limitations",
    ]
    lines = [
        "# External NGBoost ledger",
        "",
        "NLL, RMSE, coverage, and width values are unweighted means of fold-level metrics.",
        "",
        "| " + " | ".join(display) + " |",
        "|" + "|".join(["---"] * len(display)) + "|",
    ]
    for row in rows:
        cells = []
        for field in display:
            value = row[field]
            if isinstance(value, float):
                value = f"{value:.6g}"
            cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "Split policies and complete artifact paths are retained in the CSV.", ""])
    return "\n".join(lines)


def paired_rows(metrics_paths: list[Path]) -> list[dict]:
    runs = {}
    for path in metrics_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        key = (data["dataset"], data["variant"])
        if key in runs:
            raise ValueError(f"duplicate dataset/variant run: {key}")
        if not data.get("reportable", False):
            raise ValueError(f"paired ledger requires reportable runs: {path}")
        runs[key] = data

    datasets = sorted({dataset for dataset, _ in runs})
    rows = []
    for dataset in datasets:
        variants = {variant for run_dataset, variant in runs if run_dataset == dataset}
        if variants != {"natural", "ordinary", "homoscedastic"}:
            raise ValueError(f"expected three variants for {dataset}, found {sorted(variants)}")
        for variant_a, variant_b in PAIRS:
            folds_a = {int(fold["fold_index"]): fold for fold in runs[(dataset, variant_a)]["per_fold"]}
            folds_b = {int(fold["fold_index"]): fold for fold in runs[(dataset, variant_b)]["per_fold"]}
            if folds_a.keys() != folds_b.keys():
                raise ValueError(f"fold mismatch for {dataset}: {variant_a} vs {variant_b}")
            nll_deltas = []
            rmse_deltas = []
            for fold_index in sorted(folds_a):
                fold_a, fold_b = folds_a[fold_index], folds_b[fold_index]
                if fold_a["split_hashes"] != fold_b["split_hashes"]:
                    raise ValueError(
                        f"split hash mismatch for {dataset} fold {fold_index}: {variant_a} vs {variant_b}"
                    )
                nll_deltas.append(float(fold_a["test_nll"] - fold_b["test_nll"]))
                rmse_deltas.append(float(fold_a["test_rmse"] - fold_b["test_rmse"]))

            def counts(deltas: list[float]) -> tuple[int, int, int]:
                return (
                    sum(delta < -TIE_ATOL for delta in deltas),
                    sum(abs(delta) <= TIE_ATOL for delta in deltas),
                    sum(delta > TIE_ATOL for delta in deltas),
                )

            nll_a, nll_ties, nll_b = counts(nll_deltas)
            rmse_a, rmse_ties, rmse_b = counts(rmse_deltas)
            rows.append({
                "dataset": dataset,
                "variant_a": variant_a,
                "variant_b": variant_b,
                "n_folds": len(nll_deltas),
                "mean_fold_nll_delta": sum(nll_deltas) / len(nll_deltas),
                "nll_a_wins": nll_a,
                "nll_ties": nll_ties,
                "nll_b_wins": nll_b,
                "mean_fold_rmse_delta": sum(rmse_deltas) / len(rmse_deltas),
                "rmse_a_wins": rmse_a,
                "rmse_ties": rmse_ties,
                "rmse_b_wins": rmse_b,
            })
    return rows


def paired_markdown(rows: list[dict]) -> str:
    lines = [
        "# External NGBoost paired fold ledger",
        "",
        "Deltas are variant A minus variant B; negative values favor A. Wins use lower metric values with `1e-12` tie tolerance.",
        "All pairs were validated to use identical split hashes at each fold.",
        "",
        "| Dataset | A | B | Folds | Mean fold NLL delta | NLL wins A/tie/B | Mean fold RMSE delta | RMSE wins A/tie/B |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {variant_a} | {variant_b} | {n_folds} | {mean_fold_nll_delta:.6f} | "
            "{nll_a_wins}/{nll_ties}/{nll_b_wins} | {mean_fold_rmse_delta:.6f} | "
            "{rmse_a_wins}/{rmse_ties}/{rmse_b_wins} |".format(**row)
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="run directories, metrics.json files, or roots")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    csv_path = Path(args.output_prefix + ".csv")
    markdown_path = Path(args.output_prefix + ".md")
    paired_csv_path = Path(args.output_prefix + ".paired.csv")
    paired_markdown_path = Path(args.output_prefix + ".paired.md")
    checksums_path = Path(args.output_prefix + ".checksums.txt")
    for output in (csv_path, markdown_path, paired_csv_path, paired_markdown_path, checksums_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite ledger: {output}")
    metrics_paths = discover_metrics([Path(value) for value in args.inputs])
    if not metrics_paths:
        raise FileNotFoundError("no metrics.json files found")
    rows = [row_from_metrics(path) for path in metrics_paths]
    rows.sort(key=lambda row: (row["dataset"], row["variant"], row["output_dir"]))
    comparisons = paired_rows(metrics_paths)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text(markdown(rows), encoding="utf-8")
    with paired_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIRED_FIELDS)
        writer.writeheader()
        writer.writerows(comparisons)
    paired_markdown_path.write_text(paired_markdown(comparisons), encoding="utf-8")
    outputs = (csv_path, markdown_path, paired_csv_path, paired_markdown_path)
    checksums_path.write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in outputs), encoding="utf-8"
    )
    print(json.dumps({
        "runs": len(rows),
        "paired_comparisons": len(comparisons),
        "csv": str(csv_path),
        "markdown": str(markdown_path),
        "paired_csv": str(paired_csv_path),
        "paired_markdown": str(paired_markdown_path),
        "checksums": str(checksums_path),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
