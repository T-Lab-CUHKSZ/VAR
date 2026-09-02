#!/usr/bin/env python3
"""Behavior-cluster bootstrap for paper Table 12."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .robustness_common import (
        DISPLAY_NAMES,
        MODELS,
        load_prompt_meta,
        load_votes,
        paired_behavior_means,
    )
except ImportError:
    from robustness_common import (
        DISPLAY_NAMES,
        MODELS,
        load_prompt_meta,
        load_votes,
        paired_behavior_means,
    )


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5], method="linear")
    return float(low), float(high)


def bootstrap_model(
    vernacular: np.ndarray,
    baseline: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> dict:
    if vernacular.shape != baseline.shape or vernacular.ndim != 1 or not len(vernacular):
        raise ValueError("vernacular and baseline must be paired non-empty vectors")
    indices = rng.integers(0, len(vernacular), size=(repetitions, len(vernacular)))
    vernacular_draws = vernacular[indices].mean(axis=1)
    baseline_draws = baseline[indices].mean(axis=1)
    if np.any(baseline_draws == 0):
        raise ZeroDivisionError("a bootstrap draw has zero baseline ASR")
    ratio_draws = vernacular_draws / baseline_draws
    v_point, b_point = float(vernacular.mean()), float(baseline.mean())
    return {
        "vernacular_asr": {"point": v_point, "ci": percentile_interval(vernacular_draws)},
        "baseline_asr": {"point": b_point, "ci": percentile_interval(baseline_draws)},
        "ratio": {"point": v_point / b_point, "ci": percentile_interval(ratio_draws)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True, help="controlled single-turn prompt JSONL")
    parser.add_argument("--judge-root", required=True, help="directory containing MODEL/JUDGE.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing-vote-policy", choices=("error", "invariant"), default="error")
    args = parser.parse_args()
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")

    prompt_meta = load_prompt_meta(Path(args.prompts))
    children = np.random.SeedSequence(args.seed).spawn(len(MODELS))
    result = {
        "config": {
            "repetitions": args.repetitions,
            "root_seed": args.seed,
            "cluster": "behavior",
            "per_model_streams": "SeedSequence.spawn(8)",
            "paired_resampling": True,
            "percentiles": [2.5, 97.5],
            "missing_vote_policy": args.missing_vote_policy,
        },
        "models": {},
    }
    for model, child in zip(MODELS, children):
        votes = load_votes(Path(args.judge_root), model, 0.25)
        vernacular, baseline, diagnostics = paired_behavior_means(
            prompt_meta, votes, args.missing_vote_policy
        )
        model_result = bootstrap_model(
            np.asarray(vernacular, dtype=float),
            np.asarray(baseline, dtype=float),
            args.repetitions,
            np.random.default_rng(child),
        )
        model_result["diagnostics"] = diagnostics
        model_result["rng_spawn_key"] = list(child.spawn_key)
        result["models"][model] = model_result

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("| Model | Vern ASR [95% CI] | Base ASR [95% CI] | Ratio [95% CI] |")
    print("|---|---:|---:|---:|")
    for model in MODELS:
        row = result["models"][model]
        v, b, ratio = row["vernacular_asr"], row["baseline_asr"], row["ratio"]
        print(
            f"| {DISPLAY_NAMES[model]} | {v['point']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}] | "
            f"{b['point']:.3f} [{b['ci'][0]:.3f}, {b['ci'][1]:.3f}] | "
            f"{ratio['point']:.2f} [{ratio['ci'][0]:.2f}, {ratio['ci'][1]:.2f}] |"
        )


if __name__ == "__main__":
    main()
