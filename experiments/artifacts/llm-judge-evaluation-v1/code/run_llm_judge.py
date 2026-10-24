#!/usr/bin/env python3
"""Run a supplied explanation fixture through the local judge, with raw provenance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

from llm_judge import (CRITERIA, OllamaJudge, classification_summary, extract_verdict,
                       fingerprint, order_summary)


def save(path: Path, value) -> None:
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text)
    temp.replace(path)


def run(args) -> None:
    source = json.loads(args.cases.read_text())
    cases = source["cases"]
    if args.max_cases:
        cases = cases[:args.max_cases]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("expected nonempty cases with unique IDs")
    for case in cases:
        if len(case["candidates"]) != 2 or len({c["id"] for c in case["candidates"]}) != 2:
            raise ValueError("each case must have two uniquely named candidates")
    judge = OllamaJudge(args.model, args.base_url, args.timeout)
    meta = judge.metadata()
    config = {"dataset_sha256": fingerprint(source), "selected_cases": [c["id"] for c in cases],
              "judge": meta, "repeat_first": args.repeat_first,
              "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                              (Path(__file__), Path(__file__).with_name("llm_judge.py"))}}
    # Identical configuration may resume cached, checked requests. Other runs need a new directory.
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["config"] != config:
            raise ValueError("output directory belongs to another configuration")
    else:
        save(manifest_path, {"config": config, "purpose": source["purpose"],
             "reportable": source.get("reportable", False), "python": sys.version,
             "platform": platform.platform(), "created_at": datetime.now(timezone.utc).isoformat()})
    requests = args.output / "requests"
    requests.mkdir(exist_ok=True)
    point_rows, pair_rows, repeats, errors = [], [], [], []
    count = 0

    def evaluate(case, kind, first, second=None):
        nonlocal count
        count += 1
        payload = judge.payload(kind, case["evidence"], first["text"], second["text"] if second else None)
        path = requests / f"request-{count:03d}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["request"] != payload:
                raise ValueError("cached request payload mismatch")
        else:
            print(f"Request {count}: {case['id']} {kind} {first['id']}", flush=True)
            record = {"case_id": case["id"], "kind": kind, "candidate_id": first["id"], "request": payload}
            try:
                record.update(judge.call(payload))
            except Exception as exc:
                record["error"] = str(exc)
            save(path, record)
        try:
            if "error" in record:
                raise RuntimeError(record["error"])
            verdict = extract_verdict(record["response"], kind, judge.options)
        except Exception as exc:
            errors.append({"request": count, "case_id": case["id"], "error": str(exc)})
            return None
        return verdict

    for case in cases:
        first, second = case["candidates"]
        verdicts = {}
        for candidate in (first, second):
            verdict = evaluate(case, "pointwise", candidate)
            if verdict is not None:
                verdicts[candidate["id"]] = verdict
                point_rows.append({"case_id": case["id"], "candidate_id": candidate["id"], "verdict": verdict,
                                   "engineering_expectation": candidate.get("engineering_expectation")})
        forward = evaluate(case, "pairwise", first, second)
        reverse = evaluate(case, "pairwise", second, first)
        if forward and reverse:
            pair_rows.append({"case_id": case["id"], "forward": forward, "reverse": reverse,
                              **order_summary(forward["winner"], reverse["winner"], first["id"], second["id"])})
        if args.repeat_first and not repeats:
            repeat = evaluate(case, "pointwise", first)
            if repeat and first["id"] in verdicts:
                repeats.append({"case_id": case["id"], "same_scores": all(
                    repeat[k] == verdicts[first["id"]][k] for k in CRITERIA),
                    "same_complete_verdict": repeat == verdicts[first["id"]], "repeated_verdict": repeat})
    diagnostics = {}
    for criterion in CRITERIA:
        labelled = [r for r in point_rows if r["engineering_expectation"] is not None]
        if labelled:
            diagnostics[criterion] = classification_summary(
                [r["engineering_expectation"][criterion] for r in labelled],
                [r["verdict"][criterion] for r in labelled])
    recorded = [json.loads(p.read_text()) for p in sorted(requests.glob("*.json"))]
    completed = [r for r in recorded if "response" in r]
    decoded_tokens = sum(r["response"].get("eval_count", 0) for r in completed)
    decode_seconds = sum(r["response"].get("eval_duration", 0) for r in completed) / 1e9
    timing = {"completed_transport_requests": len(completed),
              "total_request_wall_seconds": sum(r["wall_seconds"] for r in completed),
              "prompt_tokens": sum(r["response"].get("prompt_eval_count", 0) for r in completed),
              "generated_tokens": decoded_tokens,
              "decode_tokens_per_second": decoded_tokens / decode_seconds if decode_seconds else None,
              "total_model_load_seconds": sum(r["response"].get("load_duration", 0) for r in completed) / 1e9}
    summary = {"purpose": source["purpose"], "reportable": source.get("reportable", False),
               "status": "complete" if not errors else "completed_with_errors", "requests": count,
               "valid_pointwise_explanations": len(point_rows), "valid_pairwise_cases": len(pair_rows),
               "order_consistent_cases": sum(p["consistent_under_swap"] for p in pair_rows),
               "pointwise": point_rows, "pairwise": pair_rows, "repeatability": repeats, "timing": timing,
               "engineering_expectation_agreement": diagnostics, "errors": errors,
               "interpretation": ["Smoke labels are authored engineering expectations, not human reference ratings",
                                  "No judge confidence probabilities or human agreement are inferred",
                                  "Order-sensitive pairs remain order-sensitive, not substantive ties"]}
    save(args.output / "summary.json", summary)
    files = [manifest_path, args.output / "summary.json", *sorted(requests.glob("*.json"))]
    (args.output / "checksums.txt").write_text("".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(args.output)}\n" for p in files))
    print(json.dumps({k: summary[k] for k in ("status", "requests", "valid_pointwise_explanations",
                                            "valid_pairwise_cases", "order_consistent_cases", "errors")}, indent=2))
    if errors:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="minicpm4.1-8b-q8_0-judge")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--repeat-first", action="store_true")
    args = parser.parse_args()
    allowed = Path(__file__).resolve().parents[1] / "artifacts"
    if not args.output.resolve().is_relative_to(allowed.resolve()):
        parser.error("output must be under this project's artifacts directory")
    run(args)


if __name__ == "__main__":
    main()
