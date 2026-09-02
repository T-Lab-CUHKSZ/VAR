#!/usr/bin/env python3
"""Compute the blind-audit table from the de-identified CSV."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "human_eval"))
from audit_utils import agreement_metrics, binary


PREDICTORS = ("harmbench", "llamaguard", "wildguard", "strongreject", "ensemble_success")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        default=str(ROOT / "data/human_audit/audit_labels_deidentified.csv"),
    )
    parser.add_argument("--out")
    args = parser.parse_args()

    with Path(args.audit).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    truth = [binary(row["human_success"], "human_success") for row in rows]
    results = {
        predictor: agreement_metrics(
            truth, [binary(row[predictor], predictor) for row in rows]
        )
        for predictor in PREDICTORS
    }
    print("| Judge | TP | FP | FN | TN | Precision | Recall | Raw | Kappa | PABAK | AC1 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for predictor, metrics in results.items():
        print(
            f"| {predictor} | {metrics['tp']} | {metrics['fp']} | {metrics['fn']} | {metrics['tn']} | "
            f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['raw_agreement']:.3f} | "
            f"{metrics['cohen_kappa']:.3f} | {metrics['pabak']:.3f} | {metrics['gwet_ac1']:.3f} |"
        )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
