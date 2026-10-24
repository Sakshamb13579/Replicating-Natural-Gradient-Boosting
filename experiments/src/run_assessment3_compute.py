#!/usr/bin/env python3
"""Persistent, single-thread-per-worker Assignment 3 execution with fold checkpoints."""
from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import json
import os
import platform
import shlex
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, spearmanr

from run_external_ngboost import INTERVALS, index_hash, make_model, package_versions, select_stage, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = ROOT / "artifacts/assessment3-compute-v1"
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checksums(directory: Path, verify: bool = False) -> None:
    path = directory / "checksums.txt"
    if verify:
        for line in path.read_text().splitlines():
            expected, name = line.split("  ", 1)
            if sha256_file(directory / name) != expected:
                raise ValueError(f"artifact checksum mismatch: {directory / name}")
    else:
        paths = sorted(p for p in directory.iterdir() if p.is_file() and p.name != path.name)
        path.write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in paths))


def verify_batch(batch: Path) -> dict:
    if not batch.is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("batch must stay inside the active person's artifacts tree")
    doc = json.loads((batch / "batch.json").read_text())
    if sha256_file(batch / "protocol.json") != doc["protocol_sha256"]:
        raise ValueError("protocol hash mismatch")
    for name, expected in doc["source_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"frozen code changed: {name}")
    return doc


def load_input(batch: Path, dataset: str) -> tuple[pd.DataFrame, dict, list[dict]]:
    doc = verify_batch(batch)
    record = next(d for d in doc["datasets"] if d["dataset"] == dataset)
    directory = batch / "inputs" / dataset
    if sha256_file(directory / "manifest.json") != record["manifest_sha256"]:
        raise ValueError("input manifest hash mismatch")
    manifest = json.loads((directory / "manifest.json").read_text())
    for filename, key in (("data.csv", "csv_sha256"), ("splits.json", "splits_sha256")):
        if sha256_file(directory / filename) != manifest[key]:
            raise ValueError(f"input checksum mismatch: {filename}")
    frame = pd.read_csv(directory / "data.csv")
    folds = json.loads((directory / "splits.json").read_text())["folds"]
    if frame.columns.tolist() != manifest["columns"] or len(frame) != manifest["rows"]:
        raise ValueError("input schema mismatch")
    if frame.row_id.tolist() != list(range(len(frame))):
        raise ValueError("unexpected row IDs")
    features = manifest["features"]
    forbidden = {"row_id", manifest["target"], *manifest.get("evaluation_only", [])}
    if manifest.get("group_column"):
        forbidden.add(manifest["group_column"])
    if set(features) & forbidden or not np.isfinite(frame[features + [manifest["target"]]]).all().all():
        raise ValueError("invalid feature roles or nonfinite input")
    for fold in folds:
        sets = []
        for part in ("outer_train", "outer_test", "inner_train", "inner_val"):
            ids = fold[part + "_indices"]
            if index_hash(ids) != fold[part + "_hash"] or len(set(ids)) != len(ids):
                raise ValueError("split hash or unique-index mismatch")
            sets.append(set(ids))
        ot, oe, it, iv = sets
        if ot & oe or ot | oe != set(range(len(frame))) or it & iv or it | iv != ot:
            raise ValueError("invalid split partition")
        if manifest.get("group_column"):
            g = frame[manifest["group_column"]].to_numpy()
            if set(g[list(ot)]) & set(g[list(oe)]) or set(g[list(it)]) & set(g[list(iv)]):
                raise ValueError("group leakage")
    return frame, manifest, folds


def measure_predictions(pred: pd.DataFrame) -> dict:
    y = pred.y_true.to_numpy()
    mean = pred.pred_mean.to_numpy()
    scale = pred.pred_scale.to_numpy()
    if not np.isfinite(pred.select_dtypes(include=np.number)).all().all() or not (scale > 0).all():
        raise ValueError("invalid predictions")
    metrics = {
        "test_nll": float(-norm.logpdf(y, loc=mean, scale=scale).mean()),
        "test_rmse": float(np.sqrt(np.mean((y - mean) ** 2))),
        "coverage": {}, "mean_width": {},
        "pred_scale_min": float(scale.min()), "pred_scale_max": float(scale.max()),
        "pred_scale_mean": float(scale.mean()),
    }
    for level in INTERVALS:
        lo, hi = pred[f"interval_{level}_lo"].to_numpy(), pred[f"interval_{level}_hi"].to_numpy()
        if not (lo < hi).all():
            raise ValueError("invalid interval bounds")
        metrics["coverage"][level] = float(((y >= lo) & (y <= hi)).mean())
        metrics["mean_width"][level] = float((hi - lo).mean())
    if "true_sigma" in pred:
        truth = pred.true_sigma.to_numpy()
        metrics["oracle_nll"] = float(-norm.logpdf(y, loc=pred.true_mu, scale=truth).mean())
        metrics["scale_mae"] = float(np.abs(scale - truth).mean())
        metrics["scale_correlation"] = {"pearson": None, "spearman": None}
        if np.ptp(truth) > 1e-12 and np.ptp(scale) > 1e-12:
            metrics["scale_correlation"] = {"pearson": float(pearsonr(truth, scale)[0]),
                                            "spearman": float(spearmanr(truth, scale)[0])}
        metrics["abs_x1_bins"] = []
        # Fixed equal-width input bins, same interpretation across fresh replicates.
        x = np.abs(pred.x1.to_numpy())
        for i in range(4):
            mask = (x >= i * 0.5) & (x < (i + 1) * 0.5)
            if not mask.any():
                continue
            item = {"bin": i + 1, "lo": i * 0.5, "hi": (i + 1) * 0.5,
                    "n": int(mask.sum()), "mean_abs_x1": float(x[mask].mean()),
                    "mean_true_sigma": float(truth[mask].mean()),
                    "mean_pred_scale": float(scale[mask].mean()), "coverage": {}, "mean_width": {}}
            for level in INTERVALS:
                lo = pred[f"interval_{level}_lo"].to_numpy()[mask]
                hi = pred[f"interval_{level}_hi"].to_numpy()[mask]
                item["coverage"][level] = float(((y[mask] >= lo) & (y[mask] <= hi)).mean())
                item["mean_width"][level] = float((hi - lo).mean())
            metrics["abs_x1_bins"].append(item)
    return metrics


def run_fold(frame: pd.DataFrame, manifest: dict, fold: dict, variant: str,
             out: Path, n_estimators: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    complete = out / "complete.json"
    if complete.exists():
        checksums(out, verify=True)
        return json.loads((out / "metrics.json").read_text())
    index = fold["fold_index"]
    indices = {part: np.asarray(fold[part + "_indices"], dtype=np.int64) for part in
               ("outer_train", "outer_test", "inner_train", "inner_val")}
    X = np.ascontiguousarray(frame[manifest["features"]], dtype=np.float64)
    y = frame[manifest["target"]].to_numpy(dtype=np.float64)
    stage_seed = fold.get("stage_model_seed", 20000 + index)
    full_seed = fold.get("full_model_seed", 30000 + index)
    started = time.perf_counter()
    write_json(out / "progress.json", {"state": "stage_fit", "started_at": now(), "pid": os.getpid()})
    print(f"{now()} {manifest['dataset']} {variant} fold={index} stage_fit start", flush=True)
    model = make_model(variant, stage_seed, n_estimators)
    model.fit(X[indices["inner_train"]], y[indices["inner_train"]])
    stage_seconds = time.perf_counter() - started
    best, sigma, nlls = select_stage(variant, model, X[indices["inner_val"]], y[indices["inner_val"]])
    validation = pd.DataFrame({"stage": np.arange(1, len(nlls) + 1), "validation_nll": nlls})
    validation.to_csv(out / "validation-trajectory.csv", index=False)
    write_json(out / "selection.json", {"best_iter": best, "retained_sigma": sigma,
                                        "validation_nll": float(nlls[best - 1])})
    del model
    gc.collect()
    write_json(out / "progress.json", {"state": "full_refit", "updated_at": now(),
                                       "pid": os.getpid(), "best_iter": best})
    print(f"{now()} {manifest['dataset']} {variant} fold={index} refit start best={best}", flush=True)
    refit_start = time.perf_counter()
    model = make_model(variant, full_seed, n_estimators)
    model.fit(X[indices["outer_train"]], y[indices["outer_train"]])
    refit_seconds = time.perf_counter() - refit_start
    test = indices["outer_test"]
    distribution = model.pred_dist(X[test], max_iter=best)
    mean = np.asarray(distribution.loc).reshape(-1)
    scale = (np.full(len(test), sigma) if variant == "homoscedastic"
             else np.asarray(distribution.scale).reshape(-1))
    pred = pd.DataFrame({"fold_index": index, "row_id": frame.row_id.to_numpy()[test],
                         "y_true": y[test], "pred_mean": mean, "pred_scale": scale, "best_iter": best})
    for col in (manifest.get("group_column"), "x1", "x2", "true_mu", "true_sigma"):
        if col and col in frame:
            pred[col] = frame[col].to_numpy()[test]
    for level, (lo, hi) in INTERVALS.items():
        pred[f"interval_{level}_lo"] = norm.ppf(lo, loc=mean, scale=scale)
        pred[f"interval_{level}_hi"] = norm.ppf(hi, loc=mean, scale=scale)
    pred.to_csv(out / "predictions.csv", index=False)
    # Metrics are independently recomputed from the actual persisted prediction file.
    metrics = measure_predictions(pd.read_csv(out / "predictions.csv"))
    metrics.update({"dataset": manifest["dataset"], "variant": variant, "fold_index": index,
                    "seeds": {"stage_model": stage_seed, "full_model": full_seed},
                    "best_iter": best, "cap_hit": best == n_estimators,
                    "actual_validation_stages": len(nlls),
                    "actual_refit_stages": len(model.base_models),
                    "val_nll_at_best": float(nlls[best - 1]),
                    "fixed_sigma_from_stage_val": sigma,
                    "sigma_reestimated_after_refit": False if variant == "homoscedastic" else None,
                    "split_hashes": {k: fold[k] for k in fold if k.endswith("_hash")},
                    "sample_counts": {part: len(ids) for part, ids in indices.items()},
                    "timing_sec": {"stage_fit": stage_seconds, "refit": refit_seconds,
                                   "wall": time.perf_counter() - started},
                    "reportable": n_estimators == 2000,
                    "replicate": manifest.get("replicate"), "regime": manifest.get("regime")})
    if len(model.base_models) < best:
        raise ValueError("refit trajectory shorter than selected prefix")
    write_json(out / "metrics.json", metrics)
    write_json(out / "progress.json", {"state": "complete", "updated_at": now()})
    write_json(complete, {"status": "complete", "finished_at": now(),
                          "predictions_sha256": sha256_file(out / "predictions.csv")})
    checksums(out)
    print(f"{now()} {manifest['dataset']} {variant} fold={index} complete NLL={metrics['test_nll']:.6f}", flush=True)
    return metrics


def environment() -> dict:
    versions = package_versions()
    path = Path(versions["ngboost_file"]).resolve().parents[1]
    git = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    status = subprocess.run(["git", "-C", str(path), "status", "--short"], capture_output=True, text=True)
    return {"versions": versions, "platform": platform.platform(), "ngboost_git_commit": git.stdout.strip(),
            "ngboost_git_status": status.stdout.splitlines(),
            "thread_environment": {key: os.environ.get(key) for key in THREAD_KEYS}}


def task(args) -> None:
    frame, manifest, folds = load_input(args.batch, args.dataset)
    output = args.batch / ("smoke" if args.smoke else "runs") / args.dataset / args.variant
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "complete.json").exists():
            checksums(output, verify=True)
            print("Already complete, validated:", output, flush=True)
            return
        write_json(output / "config.json", {
            "dataset": args.dataset, "variant": args.variant, "features": manifest["features"],
            "target": manifest["target"], "n_estimators": 25 if args.smoke else 2000,
            "stage_selector": "validation_NLL", "learning_rate": 0.01,
            "reportable": not args.smoke, "manifest_sha256": sha256_file(args.batch / "inputs" / args.dataset / "manifest.json"),
        })
        write_json(output / "environment.json", environment())
        (output / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
        selected = folds[:1] if args.smoke else folds
        results = []
        for fold in selected:
            result = run_fold(frame, manifest, fold, args.variant,
                              output / f"fold-{fold['fold_index']:02d}", 25 if args.smoke else 2000)
            results.append(result)
        write_json(output / "metrics.json", {"dataset": args.dataset, "variant": args.variant,
                                             "reportable": not args.smoke, "per_fold": results})
        write_json(output / "complete.json", {"status": "complete", "finished_at": now()})
        checksums(output)


def supervise(args) -> None:
    doc = verify_batch(args.batch)
    with (args.batch / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tasks = list(doc["tasks"])
        active = []
        done = []
        failures = []
        while tasks or active:
            while tasks and len(active) < args.workers:
                item = tasks.pop(0)
                out = args.batch / "runs" / item["dataset"] / item["variant"]
                if (out / "complete.json").exists():
                    checksums(out, verify=True)
                    done.append(item)
                    continue
                logs = args.batch / "logs"
                logs.mkdir(exist_ok=True)
                handle = (logs / f"{item['dataset']}--{item['variant']}.log").open("a")
                command = [sys.executable, "-B", str(Path(__file__).resolve()), "task", "--batch", str(args.batch),
                           "--dataset", item["dataset"], "--variant", item["variant"]]
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
                active.append({"process": process, "log": handle, "task": item, "started_at": now()})
            for item in active[:]:
                code = item["process"].poll()
                if code is not None:
                    item["log"].close()
                    active.remove(item)
                    (done if code == 0 else failures).append({**item["task"], "exit_code": code})
            state = {"state": "running" if tasks or active else ("failed" if failures else "complete"),
                     "updated_at": now(), "supervisor_pid": os.getpid(), "workers": args.workers,
                     "pending_tasks": len(tasks), "completed_tasks": len(done), "failed_tasks": failures,
                     "active": [{**i["task"], "pid": i["process"].pid, "started_at": i["started_at"]} for i in active]}
            write_json(args.batch / "status.json", state)
            if tasks or active:
                time.sleep(5)
        aggregate = ROOT / "src/aggregate_assessment3_compute.py"
        result = subprocess.run([sys.executable, "-B", str(aggregate), "--batch", str(args.batch)])
        if failures or result.returncode:
            raise RuntimeError("batch has failed tasks or aggregate validation failed; inspect logs and resume")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["task", "supervise", "launch"])
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--dataset")
    parser.add_argument("--variant", choices=["natural", "ordinary", "homoscedastic"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.batch = args.batch.resolve()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be between 1 and 4")
    if any(os.environ.get(key) != "1" for key in THREAD_KEYS):
        parser.error("set all four numerical thread environment variables to 1 before starting Python")
    verify_batch(args.batch)
    if args.mode == "launch":
        # Probe the same lock as the supervisor so a second launch cannot duplicate running work.
        with (args.batch / "supervisor.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (args.batch / "supervisor.log").open("a") as handle:
            process = subprocess.Popen(
                [sys.executable, "-B", str(Path(__file__).resolve()), "supervise", "--batch", str(args.batch),
                 "--workers", str(args.workers)], stdin=subprocess.DEVNULL,
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
            )
        write_json(args.batch / "launch.json", {"pid": process.pid, "launched_at": now(),
                                               "workers": args.workers, "environment": environment()})
        print(json.dumps({"supervisor_pid": process.pid, "batch": str(args.batch)}, indent=2))
    elif args.mode == "supervise":
        supervise(args)
    else:
        if not args.dataset or not args.variant:
            parser.error("task requires --dataset and --variant")
        task(args)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
