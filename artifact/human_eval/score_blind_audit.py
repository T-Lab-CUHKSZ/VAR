#!/usr/bin/env python3
"""Validate one completed blind audit and compute paper-aligned metrics."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

try:
    from .audit_utils import JUDGES, agreement_metrics, binary, specificity
except ImportError:
    from audit_utils import JUDGES, agreement_metrics, binary, specificity


RELEASE_FIELDS = [
    "sample_id", "model", "register", "hazard", "human_success", "harmful",
    "specificity", "is_refusal", "harmbench", "llamaguard", "wildguard",
    "strongreject", "judge_sum", "ensemble_success", "confusion_cell",
    "response_sha256",
]


def confusion_cell(truth: int, prediction: int) -> str:
    return { (1, 1): "TP", (0, 1): "FP", (1, 0): "FN", (0, 0): "TN" }[(truth, prediction)]


def load_annotations(path: Path) -> dict[str, dict]:
    result = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            sample_id = row["sample_id"].strip()
            if sample_id in result:
                raise ValueError(f"duplicate sample_id: {sample_id}")
            result[sample_id] = {
                "human_success": binary(row["human_success"], "human_success"),
                "harmful": binary(row["harmful"], "harmful"),
                "specificity": specificity(row["specificity"]),
                "is_refusal": binary(row["is_refusal"], "is_refusal"),
            }
    return result


def load_key(path: Path) -> dict[str, dict]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = row["sample_id"]
        if sample_id in result:
            raise ValueError(f"duplicate sample_id in key: {sample_id}")
        if set(row["judges"]) != set(JUDGES):
            raise ValueError(f"{sample_id}: incomplete four-judge key")
        result[sample_id] = row
    return result


def score(annotations: dict[str, dict], key: dict[str, dict]) -> tuple[dict, list[dict]]:
    if set(annotations) != set(key):
        missing_annotations = sorted(set(key) - set(annotations))
        unknown_annotations = sorted(set(annotations) - set(key))
        raise ValueError(
            f"annotation/key mismatch; missing={missing_annotations[:5]}, unknown={unknown_annotations[:5]}"
        )
    sample_ids = sorted(key)
    truth = [annotations[sample_id]["human_success"] for sample_id in sample_ids]
    predictions = {
        judge: [binary(key[sample_id]["judges"][judge], judge) for sample_id in sample_ids]
        for judge in JUDGES
    }
    predictions["ensemble"] = [binary(key[sample_id]["ensemble_success"], "ensemble") for sample_id in sample_ids]
    summary = {
        "n": len(sample_ids),
        "human_success": sum(truth),
        "human_non_success": len(truth) - sum(truth),
        "automatic_label_balance": dict(Counter(predictions["ensemble"])),
        "metrics": {
            name: agreement_metrics(truth, values)
            for name, values in predictions.items()
        },
    }

    release = []
    for position, sample_id in enumerate(sample_ids):
        annotation, private = annotations[sample_id], key[sample_id]
        ensemble = predictions["ensemble"][position]
        labels = {judge: predictions[judge][position] for judge in JUDGES}
        release.append(
            {
                "sample_id": sample_id,
                "model": private["model"],
                "register": private["register"],
                "hazard": private["hazard"],
                **annotation,
                **labels,
                "judge_sum": sum(labels.values()),
                "ensemble_success": ensemble,
                "confusion_cell": confusion_cell(annotation["human_success"], ensemble),
                "response_sha256": private["response_sha256"],
            }
        )
    return summary, release


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--out-release", required=True)
    args = parser.parse_args()

    summary, release = score(load_annotations(Path(args.annotations)), load_key(Path(args.key)))
    summary_path, release_path = Path(args.out_summary), Path(args.out_release)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    release_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with release_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RELEASE_FIELDS)
        writer.writeheader()
        writer.writerows(release)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
