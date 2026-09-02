#!/usr/bin/env python3
"""Per-model four-judge agreement for paper Table 4 and Appendix C."""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    "qwen3.5-9b", "qwen3.5-4b", "llama-3.1-8b", "llama-3.2-3b",
    "gemma-4-E2B", "gemma-4-E4B", "gemini-2.5-pro", "deepseek-v4-pro",
]
JUDGES = ["harmbench", "llamaguard", "wildguard", "strongreject"]


def load_labels(root: Path, model: str) -> dict[str, dict[str, int]]:
    by_prompt = defaultdict(dict)
    for judge in JUDGES:
        path = root / model / f"{judge}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            record = json.loads(line)
            label = record.get("attack_success")
            if label is not None:
                by_prompt[record["prompt_id"]][judge] = int(bool(label))
    return {key: value for key, value in by_prompt.items() if len(value) == 4}


def pair_metrics(labels: dict[str, dict[str, int]], left: str, right: str) -> dict:
    a = np.array([record[left] for record in labels.values()], dtype=float)
    b = np.array([record[right] for record in labels.values()], dtype=float)
    if len(a) == 0:
        raise ValueError("empty label slice")
    observed = float(np.mean(a == b))
    pa, pb = float(a.mean()), float(b.mean())
    expected = pa * pb + (1 - pa) * (1 - pb)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else float("nan")
    pabak = 2 * observed - 1
    mean_positive = (pa + pb) / 2
    ac1_expected = 2 * mean_positive * (1 - mean_positive)
    ac1 = (observed - ac1_expected) / (1 - ac1_expected)
    return {
        "raw_agreement": observed,
        "cohen_kappa": kappa,
        "pabak": pabak,
        "gwet_ac1": ac1,
    }


def summarize_model(labels: dict[str, dict[str, int]]) -> dict:
    pairs = {
        f"{left}_vs_{right}": pair_metrics(labels, left, right)
        for left, right in itertools.combinations(JUDGES, 2)
    }
    return {
        "n": len(labels),
        "raw_agreement": float(np.mean([x["raw_agreement"] for x in pairs.values()])),
        "cohen_kappa": float(np.mean([x["cohen_kappa"] for x in pairs.values()])),
        "pabak": float(np.mean([x["pabak"] for x in pairs.values()])),
        "gwet_ac1": float(np.mean([x["gwet_ac1"] for x in pairs.values()])),
        "pairs": pairs,
    }


def vernacular_filter(prompt_id: str) -> bool:
    return prompt_id.endswith("_single")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vernacular_judges",
        default=str(ROOT / "experiments/single_turn/judges"),
    )
    parser.add_argument(
        "--factorial_judges",
        default=str(ROOT / "experiments/factorial_judges"),
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "experiments/factorial_judges/judge_agreement.json"),
    )
    args = parser.parse_args()

    results = {}
    for model in MODELS:
        vernacular = {
            prompt_id: labels
            for prompt_id, labels in load_labels(Path(args.vernacular_judges), model).items()
            if vernacular_filter(prompt_id)
        }
        factorial = load_labels(Path(args.factorial_judges), model)
        if not vernacular or not factorial:
            continue
        results[model] = {
            "V": summarize_model(vernacular),
            "F": summarize_model(factorial),
        }

    print("| Model | Raw V | Raw F | Kappa V | Kappa F | PABAK V | PABAK F | AC1 V | AC1 F |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model in MODELS:
        if model not in results:
            continue
        v, f = results[model]["V"], results[model]["F"]
        print(
            f"| {model} | {v['raw_agreement']:.3f} | {f['raw_agreement']:.3f} | "
            f"{v['cohen_kappa']:.3f} | {f['cohen_kappa']:.3f} | "
            f"{v['pabak']:.3f} | {f['pabak']:.3f} | "
            f"{v['gwet_ac1']:.3f} | {f['gwet_ac1']:.3f} |"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
