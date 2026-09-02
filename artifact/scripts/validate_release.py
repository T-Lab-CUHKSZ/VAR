#!/usr/bin/env python3
"""Validate public-release counts, schemas, and non-disclosure boundaries."""
from __future__ import annotations

import csv
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

from baselines.adapters import ADAPTERS, build_baseline_record
from attack.var_a4 import T2, T3, T4, VERN_12, build_records
from defenses.self_reminder import SELF_REMINDER_PREFIX, SELF_REMINDER_SUFFIX, wrap, wrap_turns
from analysis.multiturn_comparison import METHODS
from judges.run_local_judges import MODELS as LOCAL_JUDGE_MODELS


FORBIDDEN_PATTERNS = {
    "private_unix_path": re.compile(
        rb"(?<![A-Za-z0-9])/(?:data|public[0-9]*|home|Users)/[A-Za-z0-9._-]+/"
    ),
    "private_windows_path": re.compile(
        rb"[A-Za-z]:\\Users\\[A-Za-z0-9._-]+\\", re.IGNORECASE
    ),
    "numbered_stage_name": re.compile(
        rb"\bphase[_-]?[0-9]+(?:[_-][A-Za-z0-9]+)?", re.IGNORECASE
    ),
    "numbered_catalog_name": re.compile(
        rb"\b(?:all[_-]?arms|exemplars?|prompt[_-]?catalog|dataset)[_-]?v[0-9]+\b",
        re.IGNORECASE,
    ),
    "local_checkpoint_layout": re.compile(
        rb"[\"']\./(?:models?|checkpoints?)/", re.IGNORECASE
    ),
}
AUDIT_FIELDS = {
    "sample_id", "model", "register", "hazard", "human_success", "harmful",
    "specificity", "is_refusal", "harmbench", "llamaguard", "wildguard",
    "strongreject", "judge_sum", "ensemble_success", "confusion_cell",
    "response_sha256",
}

EXPECTED_THRESHOLDS = [0.125, 0.20, 0.25, 0.30, 0.375]
EXPECTED_THRESHOLD_ROWS = [
    [0.758, 0.280, 2.71],
    [0.731, 0.278, 2.63],
    [0.731, 0.278, 2.63],
    [0.691, 0.273, 2.53],
    [0.691, 0.273, 2.53],
]
EXPECTED_SEPARABILITY_ACCURACY = {
    "logistic": [0.386, 0.986],
    "centroid": [0.343, 0.980],
    "knn15": [0.311, 0.969],
}


def check_forbidden_content(allow_cache: bool = False) -> None:
    violations = []
    file_count = 0
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        is_cache = path.suffix.lower() in {".pyc", ".pyo"} or "__pycache__" in path.parts
        if is_cache:
            if not allow_cache:
                violations.append(f"{relative}: generated Python cache")
            continue
        file_count += 1
        if path.name in {".DS_Store", "Thumbs.db"}:
            violations.append(f"{relative}: packaging debris")
            continue
        content = path.read_bytes()
        for label, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(content):
                violations.append(f"{relative}: {label}")
    if file_count != 67:
        violations.append(f"package contains {file_count} files; expected 67")
    if violations:
        raise ValueError("non-public content or packaging debris found:\n" + "\n".join(violations))


def requirement_names(path: Path) -> dict[str, str]:
    result = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.match(r"([A-Za-z0-9_.-]+)", line)
        if match:
            result[match.group(1).lower()] = line
    return result


def check_dependency_boundary() -> None:
    core = requirement_names(ROOT / "requirements.txt")
    runtime = requirement_names(ROOT / "requirements-runtime.txt")
    if {"vllm", "transformers", "torch"} & set(core):
        raise ValueError("core requirements must not combine GPU runtime dependencies")
    if runtime.get("vllm") != "vllm==0.19.1":
        raise ValueError("runtime requirements must pin the reported vLLM 0.19.1")
    if "transformers" in runtime:
        raise ValueError(
            "vLLM 0.19.1 must resolve its compatible Transformers dependency; "
            "do not combine it with the separately reported judge pin"
        )


def check_method_contracts() -> None:
    expected_judges = {
        "harmbench": "cais/HarmBench-Llama-2-13b-cls",
        "llamaguard": "meta-llama/Llama-Guard-3-8B",
        "wildguard": "allenai/wildguard",
    }
    if LOCAL_JUDGE_MODELS != expected_judges:
        raise ValueError("local judges must use public checkpoint identifiers")

    source = {
        "prompt_id": "check_0_screenplay_format_single",
        "behavior_idx": 0,
        "behavior_id": "b000",
        "behavior": "controlled behavior text",
        "arm": "screenplay_format_single",
        "turns": ["exact stored register entry"],
    }
    record = next(iter(build_records([source])))
    if record["turns"] != ["exact stored register entry", T2, T3, T4]:
        raise ValueError("VAR-A4 must reuse the exact stored entry as T1")
    t1_root, t1_filter = METHODS["T1-only (screenplay_format)"]
    if t1_root != ROOT / "experiments/multiturn/t1_only_judges":
        raise ValueError("Table 6 T1-only must use its separately evaluated judgment records")
    if t1_filter != "screenplay_format":
        raise ValueError("Table 6 T1-only filter must match the released record identifiers")

    system, user = wrap("request", "system")
    if system != "system" or user != SELF_REMINDER_PREFIX + "request" + SELF_REMINDER_SUFFIX:
        raise ValueError("Self-Reminder must wrap the first user turn and preserve the system prompt")
    if wrap_turns(["T1", "T2"])[1] != "T2":
        raise ValueError("Self-Reminder must not modify continuation turns")


def check_behavior_metadata() -> None:
    path = ROOT / "data/benchmarks/behavior_metadata_290.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 290 or len({row["behavior_id"] for row in rows}) != 290:
        raise ValueError("behavior metadata must contain exactly 290 unique rows")
    if [row["behavior_id"] for row in rows] != [f"b{index:03d}" for index in range(290)]:
        raise ValueError("behavior ids must be ordered continuously from b000 through b289")
    if any(set(row) != {"behavior_id", "benchmark", "hazard"} for row in rows):
        raise ValueError("behavior metadata contains an unexpected field")
    benchmarks = Counter(row["benchmark"] for row in rows)
    if benchmarks != Counter({"HarmBench": 200, "JailbreakBench": 90}):
        raise ValueError(f"unexpected benchmark composition: {benchmarks}")
    if any("behavior" in row for row in rows):
        raise ValueError("public behavior metadata must not contain canonical behavior text")
    if len({row["hazard"] for row in rows}) != 16:
        raise ValueError("behavior metadata must contain sixteen hazard categories")


def check_corpus_contract() -> None:
    config = json.loads((ROOT / "corpus/ao3_registers.json").read_text())
    registers = config["registers"]
    if len(registers) != 12 or len({row["name"] for row in registers}) != 12:
        raise ValueError("AO3 config must contain twelve unique registers")
    if sum(int(row["target_exemplars"]) for row in registers) != 355:
        raise ValueError("AO3 target counts must sum to 355")
    if int(config["max_excerpt_characters"]) != 350:
        raise ValueError("AO3 excerpt cap must be 350 characters")
    published = json.loads((ROOT / "corpus/pool_statistics.json").read_text())
    if published["total_exemplars"] != 355 or published["max_excerpt_characters"] != 350:
        raise ValueError("published pool statistics do not match the reported values")
    config_counts = {
        row["name"]: int(row["target_exemplars"])
        for row in registers
    }
    published_counts = {
        name: int(row["exemplars"])
        for name, row in published["registers"].items()
    }
    if config_counts != published_counts:
        raise ValueError("AO3 register targets do not match the published pool statistics")
    method_registers = json.loads(
        (ROOT / "attack/registers/registers.json").read_text()
    )["registers"]
    method_names = {row["key"] for row in method_registers}
    expected_names = {name.removesuffix("_single") for name in VERN_12}
    if method_names != expected_names or method_names != set(config_counts):
        raise ValueError("register names differ across method and corpus definitions")


def check_human_audit() -> None:
    path = ROOT / "data/human_audit/audit_labels_deidentified.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != AUDIT_FIELDS:
            raise ValueError(f"unexpected audit fields: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != 200:
        raise ValueError("de-identified audit must contain exactly 200 rows")
    if len({row["sample_id"] for row in rows}) != 200:
        raise ValueError("audit sample_id values must be unique")
    if len({row["register"] for row in rows}) != 12:
        raise ValueError("audit must span twelve registers")
    if len({row["model"] for row in rows}) != 8:
        raise ValueError("audit must span eight target models")
    if len({row["hazard"] for row in rows}) != 16:
        raise ValueError("audit must span sixteen hazard categories")
    labels = Counter(int(row["ensemble_success"]) for row in rows)
    if labels != Counter({0: 100, 1: 100}):
        raise ValueError(f"audit automatic-label balance is not 100/100: {labels}")
    for row in rows:
        judge_sum = sum(int(row[name]) for name in ("harmbench", "llamaguard", "wildguard", "strongreject"))
        if judge_sum != int(row["judge_sum"]):
            raise ValueError(f"{row['sample_id']}: incorrect judge_sum")
        if int(judge_sum >= 2) != int(row["ensemble_success"]):
            raise ValueError(f"{row['sample_id']}: incorrect two-of-four ensemble label")
        if not re.fullmatch(r"[0-9a-f]{64}", row["response_sha256"]):
            raise ValueError(f"{row['sample_id']}: malformed response digest")


def check_baseline_adapters() -> None:
    if set(ADAPTERS) != {"fitd", "actorattack", "crescendo"}:
        raise ValueError(f"unexpected baseline adapter set: {sorted(ADAPTERS)}")
    source = {
        "behavior_id": "safe-check",
        "behavior": "complete a harmless fictional task",
        "benchmark": "validation",
        "hazard": "none",
    }
    for method in sorted(ADAPTERS):
        record = build_baseline_record(source, method)
        if record["turn_count"] != 4 or len(record["turns"]) != 4:
            raise ValueError(f"{method}: the paper comparison requires exactly four turns")
        if record["baseline_family"] != method or record["adapter_variant"] != "static_four_turn":
            raise ValueError(f"{method}: malformed canonical adapter metadata")


def check_appendix_robustness_reference() -> None:
    path = ROOT / "data/paper_results/appendix_robustness.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    bootstrap = result["bootstrap"]
    if bootstrap["repetitions"] != 10_000 or bootstrap["root_seed"] != 42:
        raise ValueError("bootstrap reference must use 10,000 repetitions and root seed 42")
    if len(bootstrap["models"]) != 8:
        raise ValueError("bootstrap reference must contain all eight target models")
    for model, row in bootstrap["models"].items():
        if set(row) != {"vernacular", "baseline", "ratio"}:
            raise ValueError(f"{model}: incomplete bootstrap result")
        if any(len(row[metric]) != 3 for metric in row):
            raise ValueError(f"{model}: each bootstrap metric must be point/lower/upper")
        if any(row[metric][1] > row[metric][0] or row[metric][0] > row[metric][2] for metric in row):
            raise ValueError(f"{model}: point estimate is outside its interval")

    threshold_rows = result["threshold_sensitivity"]
    if [row["threshold"] for row in threshold_rows] != EXPECTED_THRESHOLDS:
        raise ValueError("StrongREJECT threshold grid does not match the paper appendix")
    observed = [
        [row["vernacular_asr"], row["baseline_asr"], row["ratio"]]
        for row in threshold_rows
    ]
    if observed != EXPECTED_THRESHOLD_ROWS:
        raise ValueError("threshold-robustness values do not match paper Table 13")


def check_register_separability_reference() -> None:
    path = ROOT / "data/paper_results/register_separability.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    expected_shape = {
        "behaviors": 290,
        "registers": 12,
        "prompts": 3480,
        "train_behaviors": 203,
        "test_behaviors": 87,
    }
    if result["shape"] != expected_shape:
        raise ValueError(f"unexpected register-separability shape: {result['shape']}")
    observed_accuracy = {
        name: [row["structural"], row["character"]]
        for name, row in result["accuracy"].items()
    }
    if observed_accuracy != EXPECTED_SEPARABILITY_ACCURACY:
        raise ValueError("register-separability accuracies do not match paper Table 9")
    clustering = result["clustering"]
    if clustering["structural"]["adjusted_rand_index"] != 0.090:
        raise ValueError("structural clustering ARI must be 0.090")
    if clustering["character"] != {"adjusted_rand_index": 0.706, "purity": 0.796}:
        raise ValueError("character clustering results do not match paper Table 9")
    if len(result["per_register_f1"]) != 12:
        raise ValueError("Table 10 reference must contain twelve registers")
    if any(len(values) != 6 for values in result["per_register_f1"].values()):
        raise ValueError("each Table 10 row must contain three structural and three character F1 values")


def check_public_data_boundary() -> None:
    allowed = {
        Path("data/benchmarks/behavior_metadata_290.jsonl"),
        Path("data/human_audit/audit_labels_deidentified.csv"),
        Path("data/paper_results/appendix_robustness.json"),
        Path("data/paper_results/register_separability.json"),
    }
    unexpected = []
    for path in (ROOT / "data").rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".csv"}:
            continue
        relative = path.relative_to(ROOT)
        if relative not in allowed:
            unexpected.append(str(relative))
    if unexpected:
        raise ValueError("unexpected public record-level data files:\n" + "\n".join(unexpected))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-cache",
        action="store_true",
        help="ignore runtime Python caches during an in-place check",
    )
    args = parser.parse_args()
    check_forbidden_content(allow_cache=args.allow_cache)
    check_dependency_boundary()
    check_method_contracts()
    check_behavior_metadata()
    check_corpus_contract()
    check_human_audit()
    check_baseline_adapters()
    check_appendix_robustness_reference()
    check_register_separability_reference()
    check_public_data_boundary()
    print("release validation passed")
    print("  package: no private paths, non-public naming patterns, or generated caches")
    print("  dependencies: analysis/API and vLLM runtime environments are separated")
    print("  contracts: exact VAR-A4 T1 reuse, separate T1-only labels, user-turn Self-Reminder")
    print("  public data: 290 metadata rows, 355-excerpt statistics, 200 de-identified audit rows")
    print("  baselines: FITD, ActorAttack, and Crescendo; four turns each")
    print("  robustness: 10,000 behavior bootstraps and five StrongREJECT thresholds")
    print("  separability: 290 x 12 prompts, 203/87 behavior split, Tables 9-10 references")
    print("  boundary: no unexpected raw prompt, response, exemplar, or full-label data")


if __name__ == "__main__":
    main()
