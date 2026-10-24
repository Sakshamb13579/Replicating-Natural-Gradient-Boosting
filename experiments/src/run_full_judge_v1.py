#!/usr/bin/env python3
"""Freeze and supervise the authorised full explanation-judge inference study."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "artifacts/llm-judge-evaluation-v1"
COMPUTE = ROOT / "artifacts/assessment3-compute-v1"
SEED = 20261018


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def immutable(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != text:
        raise RuntimeError(f"refusing to overwrite differing frozen input: {path}")
    if not path.exists():
        path.write_text(text)


def frozen_json(path, value):
    immutable(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def state(value):
    value["updated_at"] = now()
    temp = BATCH / "status.tmp"
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(BATCH / "status.json")


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def verify_prediction_file(path):
    entries = {}
    for line in (path.parent / "checksums.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        entries[name.strip().lstrip("*")] = digest
    expected = entries.get(path.name)
    if expected != sha(path):
        raise RuntimeError(f"prediction checksum failed: {path}")


def rounded(value):
    return float(f"{float(value):.6g}")


def prepare():
    rng = random.Random(SEED)
    cases, key, sources, seen = [], [], {}, set()
    faults = ("interval_interpretation", "unsupported_cause", "obscuring_jargon")
    strata = ("synthetic-A", "synthetic-B", "synthetic-C", "airfoil", "real-estate")
    protocol = {
        "frozen_at": "before any full-evaluation judge output",
        "authorisation": "User requested full inference after reviewing the smoke findings",
        "selection_seed": SEED, "evidence_cases": 15, "explanations": 30,
        "pointwise_requests": 30, "pairwise_order_requests": 30, "total_requests": 60,
        "strata": {name: 3 for name in strata}, "prediction_variant": "natural only",
        "development": "three authored smoke cases, six explanations; not evaluation observations",
        "selection": "seeded uniform choice from sorted saved held-out rows; no observed-outcome or judge-score selection",
        "synthetic_replicates": [2, 5, 8], "public_fold_indices": [0, 1, 2],
        "matching": "paired explanations within evidence case; no duplicate latent synthetic row across regimes",
        "numeric_display": "six significant digits, shared exactly between packets and explanations",
        "explanation_construction": "versioned faithful descriptions plus five examples of each of three controlled edits",
        "prompt_policy": "unchanged smoke-tested rubric dev-v1; no retuning on evaluation output",
        "human_ratings": "pending; no human agreement, accuracy or kappa can be inferred from construction intent",
        "interpretation": "controlled benchmark-specific explanation audit, not predictive-model ranking or population calibration",
    }
    frozen_json(BATCH / "protocol.json", protocol)
    for stratum_index, stratum in enumerate(strata):
        for position in range(3):
            synthetic = stratum.startswith("synthetic")
            replicate = (2, 5, 8)[position] if synthetic else None
            fold_index = 0 if synthetic else position
            if synthetic:
                dataset = f"synthetic-r{replicate:02d}-{stratum[-1]}"
            else:
                dataset = "airfoil-grouped" if stratum == "airfoil" else "real-estate-valuation"
            if stratum == "real-estate":
                directory = ROOT / "artifacts/external-ngboost-v1/real-estate-valuation/natural/real-estate-natural-s1-a01"
                input_path = ROOT / "data/processed/real-estate-valuation-v1/real-estate-valuation-v1.csv"
                split_path = input_path.parent / "split-indices.json"
                split_doc = json.loads(split_path.read_text())
                input_rows = {int(r["row_id"]): r for r in rows(input_path)}
                split = split_doc["folds"][fold_index]
            else:
                directory = COMPUTE / f"runs/{dataset}/natural/fold-{fold_index:02d}"
                input_path = COMPUTE / f"inputs/{dataset}/data.csv"
                split_path = input_path.with_name("splits.json")
                manifest = json.loads(input_path.with_name("manifest.json").read_text())
                if sha(input_path) != manifest["csv_sha256"] or sha(split_path) != manifest["splits_sha256"]:
                    raise RuntimeError("frozen input/split checksum mismatch")
                input_rows = {int(r["row_id"]): r for r in rows(input_path)}
                split = json.loads(split_path.read_text())["folds"][fold_index]
            pred_path = directory / "predictions.csv"
            verify_prediction_file(pred_path)
            choices = [r for r in rows(pred_path) if int(r["fold_index"]) == fold_index]
            choices.sort(key=lambda r: int(r["row_id"]))
            choices = [r for r in choices if ((replicate, int(r["row_id"])) if synthetic else
                       (stratum, int(r["row_id"]))) not in seen]
            pred = choices[rng.randrange(len(choices))]
            row_id = int(pred["row_id"])
            seen.add((replicate, row_id) if synthetic else (stratum, row_id))
            if row_id not in split["outer_test_indices"] or row_id in split["outer_train_indices"]:
                raise RuntimeError("selected row was not held out for this prediction")
            source = input_rows[row_id]
            target = "y" if synthetic else "scaled_sound_pressure_db" if stratum == "airfoil" else "Y"
            if not math.isclose(float(pred["y_true"]), float(source[target]), rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError("prediction/target provenance mismatch")
            mu, sigma = float(pred["pred_mean"]), float(pred["pred_scale"])
            lo, hi = float(pred["interval_95_lo"]), float(pred["interval_95_hi"])
            if not sigma > 0 or not all(math.isfinite(v) for v in (mu, sigma, lo, hi)):
                raise RuntimeError("invalid prediction values")
            if not (math.isclose(lo, mu - 1.959963984540054 * sigma, abs_tol=1e-8) and
                    math.isclose(hi, mu + 1.959963984540054 * sigma, abs_tol=1e-8)):
                raise RuntimeError("interval does not match the saved Normal parameters")
            unit = "synthetic target units" if synthetic else "dB" if stratum == "airfoil" else "10000 New Taiwan dollars per ping"
            evidence = {"domain": stratum, "target_unit": unit,
                "predictive_distribution": "Normal", "mean": rounded(mu), "standard_deviation": rounded(sigma),
                "nominal_prediction_interval": {"level": .95, "lower": rounded(lo), "upper": rounded(hi)},
                "available_explanation_evidence": "No training-neighbour counts or local sample-variation statistics are supplied",
                "note": "Displayed numbers are rounded to six significant digits; no observed outcome or empirical coverage evidence is supplied"}
            if synthetic:
                evidence["generating_law"] = {
                    "mean_function": "sin(x1) + x2^2",
                    "scale_function": {"A": "0.1 + 0.5*abs(x1)", "B": "0.6", "C": "2*(0.1 + 0.5*abs(x1))"}[stratum[-1]],
                    "x1": rounded(source["x1"]), "x2": rounded(source["x2"]),
                    "true_conditional_mean": rounded(source["true_mu"]),
                    "true_conditional_scale": rounded(source["true_sigma"]),
                    "qualification": "Known simulated outcome noise; does not establish calibration of this fitted prediction"}
            prefix = f"The predicted mean is {evidence['mean']:g}. The nominal 95% prediction interval for one future outcome runs from {evidence['nominal_prediction_interval']['lower']:g} to {evidence['nominal_prediction_interval']['upper']:g} in {unit}."
            suffix = (f" The supplied simulation law gives a conditional noise scale of {rounded(source['true_sigma']):g}; it does not establish calibration of the fitted interval."
                      if synthetic else " The supplied packet does not establish a cause for this interval's width.")
            faithful = prefix + suffix
            fault = faults[position]
            if fault == "interval_interpretation":
                edited = prefix + " This is a 95% confidence interval for the true conditional mean, and it guarantees that the next individual outcome falls inside."
            elif fault == "unsupported_cause":
                edited = prefix + " Its width is caused by having only two comparable training examples, whose target values varied greatly."
            else:
                edited = prefix + " This mean-centred prognostic interval instantiates interquantilic spreadification via nominalisation of a singular futurate outcome."
            number = len(cases) + 1
            case_id = f"case-{number:03d}"
            descriptions = [(faithful, "faithful_description"), (edited, fault)]
            rng.shuffle(descriptions)
            candidates = []
            construction = []
            for index, (text, intent) in enumerate(descriptions):
                candidate_id = hashlib.sha256(f"{SEED}:{case_id}:{index}".encode()).hexdigest()[:12]
                candidates.append({"id": candidate_id, "text": text})
                construction.append({"candidate_id": candidate_id, "construction_intent": intent})
            cases.append({"id": case_id, "evidence": evidence, "candidates": candidates})
            key.append({"case_id": case_id, "stratum": stratum, "source_dataset": dataset,
                        "replicate": replicate, "fold_index": fold_index, "row_id": row_id,
                        "prediction_path": str(pred_path.relative_to(ROOT)), "candidates": construction})
            for path in (pred_path, input_path, split_path):
                sources[str(path.relative_to(ROOT))] = sha(path)
    document = {"purpose": "Authorised full controlled explanation-judge evaluation v1",
                "reportable": True, "human_reference_status": "pending", "cases": cases}
    frozen_json(BATCH / "cases.json", document)
    frozen_json(BATCH / "construction-key.json", key)
    frozen_json(BATCH / "source-hashes.json", sources)
    # Give raters only human/; no construction intents, pair identifiers or judge outputs.
    human = []
    for case in cases:
        for candidate in case["candidates"]:
            human.append({"item_id": candidate["id"], "evidence_json": json.dumps(case["evidence"], ensure_ascii=False),
                          "explanation": candidate["text"], "clarity": "", "interval_consistency": "",
                          "evidence_support": "", "rater_id": "", "notes": ""})
    random.Random(SEED + 1).shuffle(human)
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(human[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(human)
    immutable(BATCH / "human/ratings-blank.csv", buffer.getvalue())
    immutable(BATCH / "human/INSTRUCTIONS.md", """# Independent explanation ratings

Rate every item independently, without consulting model scores or another rater.
Use only its evidence packet. Enter 1 (pass) or 0 (fail) for each criterion:

- clarity: understandable and unambiguous wording.
- interval_consistency: all mean, scale, interval, unit and probability claims
  agree with the packet. A nominal prediction interval is not a confidence
  interval for a mean and does not guarantee an individual outcome is covered.
- evidence_support: all substantive reasons are supported by supplied evidence.
  Accurate description or an explicit statement that a cause is unknown can pass.
  Do not transfer a candidate's invented facts into the evidence packet.

Judge each criterion separately. Confusing language need not make otherwise
correct numeric statements false. Enter a rater ID and optional notes. Keep
individual ratings before any adjudication. Each rater should use a separate
copy of ratings-blank.csv. Do not view construction-key.json or judge outputs.
""")
    for name in ("llm_judge.py", "run_llm_judge.py", "run_full_judge_v1.py"):
        immutable(BATCH / "code" / name, (ROOT / "src" / name).read_text())
    stable = [BATCH / name for name in ("protocol.json", "cases.json", "construction-key.json", "source-hashes.json")]
    stable += sorted((BATCH / "human").glob("*")) + sorted((BATCH / "code").glob("*.py"))
    immutable(BATCH / "frozen-checksums.txt", "".join(f"{sha(p)}  {p.relative_to(BATCH)}\n" for p in stable))
    print(json.dumps({"batch": str(BATCH), "cases": len(cases), "explanations": len(human), "inference_requests": 60}, indent=2))


def validate():
    for line in (BATCH / "frozen-checksums.txt").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        if sha(BATCH / name) != expected:
            raise RuntimeError(f"frozen file changed: {name}")
    directory = BATCH / "run"
    for line in (directory / "checksums.txt").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        if sha(directory / name) != expected:
            raise RuntimeError(f"run checksum failed: {name}")
    summary = json.loads((directory / "summary.json").read_text())
    if summary["requests"] != 60 or len(list((directory / "requests").glob("*.json"))) != 60:
        raise RuntimeError("incomplete request inventory")
    if summary["errors"]:
        raise RuntimeError(f"judge returned invalid records: {summary['errors']}")
    if summary["valid_pointwise_explanations"] != 30 or summary["valid_pairwise_cases"] != 15:
        raise RuntimeError("incomplete valid verdict counts")
    if summary["engineering_expectation_agreement"]:
        raise RuntimeError("construction intentions must not be treated as reference labels")
    result = {"inference_status": "complete", "human_reference_status": "pending", "requests": 60,
              "pointwise_explanations": 30, "pairwise_cases": 15,
              "order_consistent_cases": summary["order_consistent_cases"],
              "timing": summary["timing"], "validated_at": now()}
    (BATCH / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def worker():
    with (BATCH / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state({"state": "running", "pid": os.getpid(), "started_at": now(), "expected_requests": 60})
        try:
            sys.path.insert(0, str(BATCH / "code"))
            from run_llm_judge import run
            from types import SimpleNamespace
            run(SimpleNamespace(cases=BATCH / "cases.json", output=BATCH / "run", model="minicpm4.1-8b-q8_0-judge",
                                base_url="http://127.0.0.1:11434", timeout=900, max_cases=None, repeat_first=False))
            result = validate()
            state({"state": "complete", "pid": os.getpid(), **result})
        except BaseException as exc:
            state({"state": "failed", "pid": os.getpid(), "error": str(exc)})
            raise
        finally:
            # Release model RAM; preserve the bare-metal service and registered weights.
            payload = json.dumps({"model": "minicpm4.1-8b-q8_0-judge", "keep_alive": 0}).encode()
            try:
                req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=payload,
                                             headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=30).close()
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "launch", "worker", "status", "validate"])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "worker":
        worker()
    elif args.action == "validate":
        print(json.dumps(validate(), indent=2))
    elif args.action == "launch":
        with (BATCH / "launch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (BATCH / "status.json").exists():
                previous = json.loads((BATCH / "status.json").read_text())
                if previous["state"] == "complete":
                    raise RuntimeError("full evaluation already complete")
                if previous["state"] in ("running", "starting"):
                    try:
                        os.kill(previous["pid"], 0)
                    except ProcessLookupError:
                        pass
                    else:
                        raise RuntimeError("full evaluation already running")
            with (BATCH / "worker.log").open("a") as log:
                proc = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "worker"],
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state({"state": "starting", "pid": proc.pid, "expected_requests": 60})
            print(f"Full inference launched: PID {proc.pid}")
    else:
        current = json.loads((BATCH / "status.json").read_text()) if (BATCH / "status.json").exists() else {"state": "prepared"}
        records = list((BATCH / "run/requests").glob("*.json"))
        current["persisted_requests"] = len(records)
        current["expected_requests"] = 60
        print(json.dumps(current, indent=2))


if __name__ == "__main__":
    main()
