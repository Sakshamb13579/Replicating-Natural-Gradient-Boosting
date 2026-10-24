#!/usr/bin/env python3
"""Measure judge agreement against completed human rating sheets."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

from run_full_judge_v1 import BATCH

CRITERIA = ("clarity", "interval_consistency", "evidence_support")
INTENTS = ("faithful_description", "interval_interpretation", "unsupported_cause", "obscuring_jargon")
# The distribution template keeps its historical name and is never a rating sheet.
TEMPLATE = "ratings-blank.csv"
SCRIPT_SOURCE = Path(__file__).read_text()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_checksums(directory, name):
    for line in (directory / name).read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        if digest(directory / relative) != expected:
            raise RuntimeError(f"checksum failed: {relative}")


def frozen_judge_code():
    spec = importlib.util.spec_from_file_location("frozen_judge_agreement", BATCH / "code/llm_judge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def judge_pointwise(cases, judge_code, options):
    """Re-derive each pointwise verdict from its raw record rather than the runner summary."""
    directory = BATCH / "run"
    verdicts, number = {}, 0
    for case in cases:
        first, second = case["candidates"]
        for kind, candidate in (("pointwise", first), ("pointwise", second),
                                ("pairwise", first), ("pairwise", second)):
            number += 1
            if kind != "pointwise":
                continue
            record = json.loads((directory / "requests" / f"request-{number:03d}.json").read_text())
            if record["case_id"] != case["id"] or record["candidate_id"] != candidate["id"]:
                raise RuntimeError(f"request {number} does not match the frozen protocol")
            verdict = judge_code.extract_verdict(record["response"], "pointwise", options)
            verdicts[candidate["id"]] = {c: bool(verdict[c]) for c in CRITERIA}
    return verdicts


def read_sheet(path, stimuli):
    """Accept a completed sheet only when its frozen stimuli and labels are intact."""
    rows = list(csv.DictReader(path.open()))
    if len(rows) != len(stimuli):
        raise RuntimeError(f"{path.name}: expected {len(stimuli)} rated items")
    raters, ratings, notes = set(), {}, {}
    for row in rows:
        item = row["item_id"]
        if item not in stimuli:
            raise RuntimeError(f"{path.name}: unknown item {item}")
        text, evidence = stimuli[item]
        if row["explanation"] != text or json.loads(row["evidence_json"]) != evidence:
            raise RuntimeError(f"{path.name}: item {item} no longer carries its frozen stimulus")
        if any(row[c] not in ("0", "1") for c in CRITERIA):
            raise RuntimeError(f"{path.name}: item {item} is unrated or not binary")
        raters.add(row["rater_id"])
        ratings[item] = {c: row[c] == "1" for c in CRITERIA}
        notes[item] = row["notes"]
    if len(raters) != 1 or not raters.issubset({r for r in raters if r.strip()}):
        raise RuntimeError(f"{path.name}: expected exactly one named rater, found {sorted(raters)}")
    return raters.pop(), ratings, notes


def compare(reference, predictions, items, judge_code):
    """Per-criterion and pooled agreement, with the reference supplied first."""
    result = {}
    for criterion in CRITERIA:
        result[criterion] = judge_code.classification_summary(
            [reference[i][criterion] for i in items], [predictions[i][criterion] for i in items])
    result["pooled"] = judge_code.classification_summary(
        [reference[i][c] for i in items for c in CRITERIA],
        [predictions[i][c] for i in items for c in CRITERIA])
    return result


def main():
    verify_checksums(BATCH, "frozen-checksums.txt")
    verify_checksums(BATCH / "run", "checksums.txt")
    source = json.loads((BATCH / "cases.json").read_text())
    cases = source["cases"]
    judge_code = frozen_judge_code()
    manifest = json.loads((BATCH / "run/manifest.json").read_text())
    if judge_code.fingerprint(source) != manifest["config"]["dataset_sha256"]:
        raise RuntimeError("run dataset fingerprint differs from the frozen cases")
    options = manifest["config"]["judge"]["options"]
    judge = judge_pointwise(cases, judge_code, options)
    stimuli = {c["id"]: (c["text"], case["evidence"]) for case in cases for c in case["candidates"]}
    intents = {c["candidate_id"]: c["construction_intent"]
               for item in json.loads((BATCH / "construction-key.json").read_text())
               for c in item["candidates"]}
    strata = {c["candidate_id"]: item["stratum"]
              for item in json.loads((BATCH / "construction-key.json").read_text())
              for c in item["candidates"]}

    sheets = sorted(p for p in (BATCH / "human").glob("ratings-*.csv") if p.name != TEMPLATE)
    if not sheets:
        raise RuntimeError("no completed rating sheet found beside the template")
    raters = {}
    for path in sheets:
        rater, ratings, notes = read_sheet(path, stimuli)
        if rater in raters:
            raise RuntimeError(f"rater {rater} appears in more than one sheet")
        raters[rater] = {"sheet": path.name, "sheet_sha256": digest(path), "ratings": ratings, "notes": notes}

    result = {"human_reference_status": "complete", "n_raters": len(raters),
              "n_items": len(stimuli), "n_criterion_verdicts_per_rater": 3 * len(stimuli),
              "raters": {}, "judge_versus_human": {}, "human_versus_human": {},
              "analysis_code_sha256": hashlib.sha256(SCRIPT_SOURCE.encode()).hexdigest()}
    for rater, record in raters.items():
        items = sorted(record["ratings"])
        passes = defaultdict(Counter)
        for item in items:
            for criterion in CRITERIA:
                if record["ratings"][item][criterion]:
                    passes[intents[item]][criterion] += 1
        planned = Counter(intents[i] for i in items)
        result["raters"][rater] = {"sheet": record["sheet"], "sheet_sha256": record["sheet_sha256"],
            "pass_counts_by_construction_intent": {intent: {"n": planned[intent],
                **{c: passes[intent][c] for c in CRITERIA}} for intent in INTENTS}}
        result["judge_versus_human"][rater] = compare(record["ratings"], judge, items, judge_code)
        result["judge_versus_human"][rater]["disagreements"] = [
            {"item_id": item, "construction_intent": intents[item], "stratum": strata[item],
             "criteria": [c for c in CRITERIA if record["ratings"][item][c] != judge[item][c]],
             "human": {c: record["ratings"][item][c] for c in CRITERIA},
             "judge": {c: judge[item][c] for c in CRITERIA}, "rater_note": record["notes"][item]}
            for item in items if any(record["ratings"][item][c] != judge[item][c] for c in CRITERIA)]
    names = sorted(raters)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            items = sorted(set(raters[first]["ratings"]) & set(raters[second]["ratings"]))
            result["human_versus_human"][f"{first}|{second}"] = compare(
                raters[first]["ratings"], raters[second]["ratings"], items, judge_code)
    result["interpretation"] = [
        "Human ratings are the reference; judge verdicts are the predictions compared against them",
        "Construction intent is an authored design category and is never used as a reference label",
        "The 30 items instantiate five explanation templates, so the items are not independent",
    ] + ([
        "A single rater cannot separate judge error from one rater's idiosyncrasy",
        "No human-versus-human agreement is available with one sheet",
    ] if len(raters) == 1 else [])

    output = BATCH / "analysis"
    output.mkdir(exist_ok=True)
    (output / "human-agreement.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "human-agreement.md").write_text(render(result) + "\n")
    (output / "human-agreement-code.py").write_text(SCRIPT_SOURCE)
    (output / "human-agreement-checksums.txt").write_text("".join(
        f"{digest(output / name)}  {name}\n" for name in
        ("human-agreement.json", "human-agreement.md", "human-agreement-code.py")))
    print(json.dumps({"raters": names, "judge_versus_human": {
        rater: {c: result["judge_versus_human"][rater][c]["cohen_kappa"] for c in CRITERIA}
        for rater in names}}, indent=2))


def kappa_text(value):
    return "undefined" if value is None else f"{value:.3f}"


def render(result) -> str:
    lines = ["# Judge agreement against human ratings", "",
             f"Human reference ratings are **complete** for {result['n_raters']} rater(s), "
             f"{result['n_items']} items and {result['n_criterion_verdicts_per_rater']} criterion verdicts each.",
             "Human ratings are the reference and the judge verdicts are compared against them.", ""]
    for rater, table in result["judge_versus_human"].items():
        sheet = result["raters"][rater]["sheet"]
        lines += [f"## Judge versus `{rater}`", "", f"Sheet: `{sheet}`.", "",
                  "| Criterion | n | Agreement | Cohen's kappa | Reference majority baseline |",
                  "|---|---:|---:|---:|---:|"]
        for criterion in CRITERIA + ("pooled",):
            row = table[criterion]
            lines.append(f"| {criterion} | {row['n']} | {row['accuracy'] * 100:.1f}% | "
                         f"{kappa_text(row['cohen_kappa'])} | {row['reference_majority_baseline_accuracy'] * 100:.1f}% |")
        lines += ["", f"Items differing on any criterion: {len(table['disagreements'])} of {result['n_items']}.", ""]
        if table["disagreements"]:
            lines += ["| Item | Construction intent | Stratum | Differing criteria | Human | Judge |", "|---|---|---|---|---|---|"]
            for row in table["disagreements"]:
                human = ", ".join(f"{c}={int(row['human'][c])}" for c in row["criteria"])
                judge = ", ".join(f"{c}={int(row['judge'][c])}" for c in row["criteria"])
                lines.append(f"| `{row['item_id']}` | {row['construction_intent']} | {row['stratum']} | "
                             f"{', '.join(row['criteria'])} | {human} | {judge} |")
            lines.append("")
        counts = result["raters"][rater]["pass_counts_by_construction_intent"]
        lines += ["Rater pass counts by construction intent, for description only:", "",
                  "| Construction intent | n | Clarity pass | Interval consistency pass | Evidence support pass |",
                  "|---|---:|---:|---:|---:|"]
        for intent in INTENTS:
            row = counts[intent]
            lines.append(f"| {intent} | {row['n']} | {row['clarity']} | "
                         f"{row['interval_consistency']} | {row['evidence_support']} |")
        lines.append("")
    if result["human_versus_human"]:
        lines += ["## Human versus human", "", "| Pair | Criterion | n | Agreement | Cohen's kappa |", "|---|---|---:|---:|---:|"]
        for pair, table in result["human_versus_human"].items():
            for criterion in CRITERIA + ("pooled",):
                row = table[criterion]
                lines.append(f"| {pair} | {criterion} | {row['n']} | {row['accuracy'] * 100:.1f}% | {kappa_text(row['cohen_kappa'])} |")
        lines.append("")
    lines += ["## Interpretation", ""] + [f"- {note}" for note in result["interpretation"]]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
