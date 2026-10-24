#!/usr/bin/env python3
"""Describe full judge outputs without treating construction intent as human gold."""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_full_judge_v1 import BATCH

SCRIPT_SOURCE = Path(__file__).read_text()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_checksums(directory, name):
    for line in (directory / name).read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        if digest(directory / relative) != expected:
            raise RuntimeError(f"checksum failed: {relative}")


def human_agreement():
    """Read the completed-rating analysis when aggregate_judge_human_agreement.py has run."""
    path = BATCH / "analysis/human-agreement.json"
    if not path.exists():
        return None
    verify_checksums(BATCH / "analysis", "human-agreement-checksums.txt")
    return json.loads(path.read_text())


def reference_status(agreement):
    if agreement is None:
        return "pending"
    return f"complete ({agreement['n_raters']} rater(s))"


def inspect_run():
    """Check every planned payload and retain invalid judgements as measured failures."""
    verify_checksums(BATCH, "frozen-checksums.txt")
    directory = BATCH / "run"
    verify_checksums(directory, "checksums.txt")
    summary = json.loads((directory / "summary.json").read_text())
    source = json.loads((BATCH / "cases.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    spec = importlib.util.spec_from_file_location("frozen_judge_validation", BATCH / "code/llm_judge.py")
    judge_code = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge_code)
    if judge_code.fingerprint(source) != manifest["config"]["dataset_sha256"]:
        raise RuntimeError("run dataset fingerprint differs from the frozen cases")
    model = manifest["config"]["judge"]["model"]["name"]
    client = judge_code.OllamaJudge(model=model.removesuffix(":latest"))
    if client.options != manifest["config"]["judge"]["options"]:
        raise RuntimeError("model options differ from the frozen configuration")
    paths = sorted((directory / "requests").glob("*.json"))
    if summary["requests"] != 60 or len(paths) != 60:
        raise RuntimeError("all 60 request records must exist before aggregation")
    valid, invalid = [], []
    number = 0
    for case in source["cases"]:
        first, second = case["candidates"]
        for kind, a, b in (("pointwise", first, None), ("pointwise", second, None),
                           ("pairwise", first, second), ("pairwise", second, first)):
            number += 1
            record = json.loads((directory / "requests" / f"request-{number:03d}.json").read_text())
            expected = client.payload(kind, case["evidence"], a["text"], b["text"] if b else None)
            if record["request"] != expected or record["case_id"] != case["id"] or record["candidate_id"] != a["id"]:
                raise RuntimeError(f"request {number} does not match the frozen protocol")
            try:
                if "error" in record:
                    raise RuntimeError(record["error"])
                verdict = judge_code.extract_verdict(record["response"], kind, client.options)
                valid.append({"case_id": case["id"], "kind": kind, "first_id": a["id"],
                              "second_id": b["id"] if b else None, "verdict": verdict})
            except Exception as exc:
                invalid.append({"request": number, "case_id": case["id"], "kind": kind, "error": str(exc)})
    if len(invalid) != len(summary["errors"]):
        raise RuntimeError("independently validated failures differ from the runner summary")
    points = [r for r in valid if r["kind"] == "pointwise"]
    if len(points) != summary["valid_pointwise_explanations"] or summary["engineering_expectation_agreement"]:
        raise RuntimeError("invalid pointwise inventory or unexpected reference labels")
    checked = {"inference_status": "complete" if not invalid else "complete_with_invalid_outputs",
               "human_reference_status": reference_status(human_agreement()), "invalid_requests": invalid,
               "valid_pointwise_explanations": len(points), "timing": summary["timing"]}
    return summary, valid, checked


def main():
    summary, valid, checked = inspect_run()
    agreement = human_agreement()
    cases = json.loads((BATCH / "cases.json").read_text())["cases"]
    key = json.loads((BATCH / "construction-key.json").read_text())
    by_case = {item["case_id"]: item for item in key}
    intents = {(item["case_id"], c["candidate_id"]): c["construction_intent"] for item in key for c in item["candidates"]}
    expected = {(case["id"], c["id"]) for case in cases for c in case["candidates"]}
    actual = {(r["case_id"], r["candidate_id"]) for r in summary["pointwise"]}
    if not actual.issubset(expected) or len(summary["pointwise"]) != len(actual):
        raise RuntimeError("pointwise candidate inventory mismatch")
    criteria = ("clarity", "interval_consistency", "evidence_support")
    grouped = defaultdict(list)
    for row in summary["pointwise"]:
        grouped[intents[row["case_id"], row["candidate_id"]]].append(row["verdict"])
    planned = Counter(intents.values())
    pointwise = {intent: {"planned": planned[intent], "n": len(grouped[intent]),
                         **{criterion: sum(v[criterion] for v in grouped[intent]) for criterion in criteria}}
                 for intent in planned}
    labels = Counter()
    ordered = Counter()
    consistent = Counter()
    by_stratum = defaultdict(lambda: Counter(cases=0, consistent=0, order_sensitive=0, substantive_ties=0))
    # Count all valid order-specific verdicts, including a valid side of a pair
    # whose other order failed. Paired consistency itself requires both sides.
    for row in valid:
        if row["kind"] != "pairwise":
            continue
        letter = row["verdict"]["winner"]
        labels[letter] += 1
        winner = None if letter == "Tie" else row["first_id"] if letter == "A" else row["second_id"]
        ordered["Tie" if winner is None else intents[row["case_id"], winner]] += 1
    for row in summary["pairwise"]:
        case_id = row["case_id"]
        stratum = by_case[case_id]["stratum"]
        by_stratum[stratum]["cases"] += 1
        if row["consistent_under_swap"]:
            by_stratum[stratum]["consistent"] += 1
            winner = row["consensus_winner_id"]
            consistent["Tie" if winner is None else intents[case_id, winner]] += 1
            if winner is None:
                by_stratum[stratum]["substantive_ties"] += 1
        else:
            by_stratum[stratum]["order_sensitive"] += 1
    result = {"inference_status": checked["inference_status"], "human_reference_status": reference_status(agreement),
              "n_evidence_cases": 15, "n_explanations": 30, "n_requests": 60,
              "n_planned_pointwise_criterion_verdicts": 90,
              "n_valid_pointwise_criterion_verdicts": 3 * checked["valid_pointwise_explanations"],
              "invalid_requests": checked["invalid_requests"],
              "n_valid_pairwise_orders": sum(labels.values()),
              "n_valid_pairwise_cases": len(summary["pairwise"]),
              "n_incomplete_pairwise_cases": 15 - len(summary["pairwise"]),
              "pointwise_pass_counts_by_construction_intent": pointwise,
              "pairwise_label_counts": dict(labels), "ordered_winner_construction_intents": dict(ordered),
              "consistent_winner_construction_intents": dict(consistent),
              "pairwise_by_stratum": {k: dict(v) for k, v in by_stratum.items()},
              "order_consistent_cases": summary["order_consistent_cases"],
              "order_sensitive_cases": len(summary["pairwise"]) - summary["order_consistent_cases"], "timing": checked["timing"],
              "analysis_code_sha256": hashlib.sha256(SCRIPT_SOURCE.encode()).hexdigest(),
              "human_agreement": None if agreement is None else
                  {"n_raters": agreement["n_raters"], "n_items": agreement["n_items"],
                   "judge_versus_human": {rater: {criterion: table[criterion]["cohen_kappa"]
                                                  for criterion in (*criteria, "pooled")}
                                          for rater, table in agreement["judge_versus_human"].items()},
                   "detail": "analysis/human-agreement.json"},
              "interpretation": ["Construction intents are authored design categories, not independent reference ratings",
                                 "30 explanations share 15 evidence cases; the 30 pairwise decisions share 15 pairs",
                                 "No predictive calibration or model-ranking conclusion follows from these outputs"]
                                + (["No human agreement is available until a completed rating sheet is analysed"]
                                   if agreement is None else agreement["interpretation"])}
    output = BATCH / "analysis"
    output.mkdir(exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    rating_state = ("**human reference ratings remain pending**" if agreement is None else
                    f"**human reference ratings are complete for {agreement['n_raters']} rater(s)**")
    lines = ["# Full LLM-judge inference results v1", "", f"Inference attempts are complete; {rating_state}.", "",
             "## Scope", "", "15 held-out prediction cases and 30 authored explanations, with 90 pointwise criterion verdicts and 30 order-specific pairwise verdicts planned. Invalid outputs are counted separately below.",
             "The same smoke-tested MiniCPM4.1-8B Q8_0 prompt was frozen before evaluation; no prompt selection used these outputs.", "",
             "## Pointwise pass counts", "", "These are model pass counts grouped by construction intent, **not accuracies against human gold**.", "",
             "| Construction intent | Planned | Valid outputs | Clarity pass | Interval consistency pass | Evidence support pass |",
             "|---|---:|---:|---:|---:|---:|"]
    for intent in ("faithful_description", "interval_interpretation", "unsupported_cause", "obscuring_jargon"):
        r = pointwise[intent]
        lines.append(f"| {intent} | {r['planned']} | {r['n']} | {r['clarity']} | {r['interval_consistency']} | {r['evidence_support']} |")
    lines += ["", "## Pairwise order effects", "",
              f"- Same underlying verdict in both orders: **{result['order_consistent_cases']}/{result['n_valid_pairwise_cases']}** valid paired cases (15 planned).",
              f"- Order-sensitive: **{result['order_sensitive_cases']}** valid paired cases (15 planned).",
              f"- Invalid request records: **{len(result['invalid_requests'])}/60**; paired cases missing a valid order: {result['n_incomplete_pairwise_cases']}/15.",
              f"- Literal label counts across {result['n_valid_pairwise_orders']} valid order-specific comparisons (30 planned): {dict(labels)}.",
              f"- Ordered winner construction categories: {dict(ordered)}.",
              f"- Consistent winner construction categories: {dict(consistent)}.", "",
              "| Stratum | Cases | Consistent | Order-sensitive | Consistent explicit ties |", "|---|---:|---:|---:|---:|"]
    for stratum, r in by_stratum.items():
        lines.append(f"| {stratum} | {r['cases']} | {r['consistent']} | {r['order_sensitive']} | {r['substantive_ties']} |")
    lines += ["", "## Timing", "",
              f"Completed-request wall-time sum: {result['timing']['total_request_wall_seconds'] / 60:.1f} minutes.",
              f"Aggregate decode throughput: {result['timing']['decode_tokens_per_second']:.2f} tokens/second.", "",
              "## Human ratings", ""]
    if agreement is None:
        lines += ["Give each independent rater only the `human/` directory and a separate copy of `ratings-blank.csv`.",
                  "Keep individual ratings before adjudication; do not expose `construction-key.json` or judge verdicts to raters.",
                  "Compute judge-versus-each-rater and human-versus-human agreement only after the completed sheets are returned."]
    else:
        lines += [f"Completed sheets: {', '.join(r['sheet'] for r in agreement['raters'].values())}.",
                  "Judge-versus-human agreement, with the human ratings as the reference:", "",
                  "| Rater | Criterion | Agreement | Cohen's kappa |", "|---|---|---:|---:|"]
        for rater, table in agreement["judge_versus_human"].items():
            for criterion in (*criteria, "pooled"):
                row = table[criterion]
                kappa = "undefined" if row["cohen_kappa"] is None else f"{row['cohen_kappa']:.3f}"
                lines.append(f"| {rater} | {criterion} | {row['accuracy'] * 100:.1f}% | {kappa} |")
        lines += ["", "Per-item disagreements and the rater pass counts are in `analysis/human-agreement.md`."]
    lines += ["",
              "Raw outputs, settings and checksums remain in `run/`; source provenance and code snapshots were frozen before launch."]
    if result["invalid_requests"]:
        lines += ["", "## Recorded invalid outputs", "", "These requests remain failures; no verdicts were repaired or regenerated.", ""]
        lines += [f"- Request {r['request']} ({r['case_id']}, {r['kind']}): {r['error']}" for r in result["invalid_requests"]]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    (output / "analysis-code.py").write_text(SCRIPT_SOURCE)
    (output / "checksums.txt").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n"
        for p in (output / "summary.json", output / "summary.md", output / "analysis-code.py")))
    print(json.dumps({k: result[k] for k in ("inference_status", "human_reference_status", "n_requests", "order_consistent_cases", "order_sensitive_cases")}, indent=2))


def monitor_state(value):
    value["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = BATCH / "analysis-monitor.tmp"
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(BATCH / "analysis-monitor.json")


def wait_and_finish():
    with (BATCH / "analysis.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monitor_state({"state": "waiting", "pid": os.getpid()})
        try:
            while True:
                status = json.loads((BATCH / "status.json").read_text())
                if status["state"] in ("complete", "failed"):
                    break
                try:
                    os.kill(status["pid"], 0)
                except ProcessLookupError:
                    raise RuntimeError("inference worker exited before publishing its final state")
                time.sleep(30)
            main()
            monitor_state({"state": "complete", "pid": os.getpid(),
                           "human_reference_status": reference_status(human_agreement())})
        except BaseException as exc:
            monitor_state({"state": "failed", "pid": os.getpid(), "error": str(exc)})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.detach:
        if (BATCH / "analysis-monitor.json").exists():
            previous = json.loads((BATCH / "analysis-monitor.json").read_text())
            if previous["state"] == "waiting":
                try:
                    os.kill(previous["pid"], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError("analysis monitor already running")
        with (BATCH / "analysis-monitor.log").open("a") as log:
            process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "--wait"],
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Automatic validation/analysis monitor launched: PID {process.pid}")
    elif args.wait:
        wait_and_finish()
    else:
        main()
