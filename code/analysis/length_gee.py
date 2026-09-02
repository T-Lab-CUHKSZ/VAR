#!/usr/bin/env python3
"""Per-model length-vs-register GEE analysis for paper Table 3."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    "qwen3.5-9b", "qwen3.5-4b", "llama-3.1-8b", "llama-3.2-3b",
    "gemma-4-E2B", "gemma-4-E4B", "gemini-2.5-pro", "deepseek-v4-pro",
]
JUDGES = ["harmbench", "llamaguard", "wildguard", "strongreject"]
VERNACULAR = {
    "frame_narrative_single", "second_person_pov_single", "epistolary_single",
    "hurt_comfort_nonabo_single", "slow_burn_single", "enemies_to_lovers_single",
    "soulmate_au_single", "ABO_omegaverse_single", "danmei_xianxia_single",
    "coffee_shop_au_single", "screenplay_format_single", "character_study_single",
}
BASELINES = {
    "deepinception", "adversarial_poetry", "dagger_happy_ending",
    "rolebreak_persona", "mimicry_narrative", "crescendo_condensed",
}


def behavior_key(record: dict) -> str | None:
    if record.get("behavior_id"):
        return str(record["behavior_id"])
    match = re.match(r"(HB|JBB)_(\d+)_", str(record.get("prompt_id", "")))
    return f"{match.group(1)}_{match.group(2)}" if match else None


def arm_class(arm: str) -> str | None:
    if arm in VERNACULAR:
        return "vernacular"
    if arm in BASELINES or arm.lower().endswith("_planx"):
        return "comparison"
    return None


def load_ensemble(judge_root: Path, model: str) -> dict[str, int]:
    by_prompt = defaultdict(dict)
    for judge in JUDGES:
        path = judge_root / model / f"{judge}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            record = json.loads(line)
            label = record.get("attack_success")
            if label is not None:
                by_prompt[record["prompt_id"]][judge] = int(bool(label))
    return {
        prompt_id: int(sum(labels.values()) >= 2)
        for prompt_id, labels in by_prompt.items()
        if len(labels) == 4
    }


def fit_gee(frame, groups: str, include_model_effects: bool = False):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    formula = "asr ~ vern + log_len_c"
    if include_model_effects:
        formula += " + C(model)"
    return smf.gee(
        formula,
        groups=groups,
        data=frame.sort_values(groups),
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()


def coefficient_row(model_name: str, fit) -> dict:
    beta_vern = float(fit.params["vern"])
    beta_length = float(fit.params["log_len_c"])
    return {
        "model": model_name,
        "beta_vern": beta_vern,
        "odds_ratio_vern": math.exp(beta_vern),
        "beta_log_length": beta_length,
        "ratio": None if beta_length <= 0 else beta_vern / beta_length,
        "se_vern": float(fit.bse["vern"]),
        "se_log_length": float(fit.bse["log_len_c"]),
        "p_vern": float(fit.pvalues["vern"]),
        "p_log_length": float(fit.pvalues["log_len_c"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument(
        "--judges", default=str(ROOT / "experiments/single_turn/judges")
    )
    parser.add_argument(
        "--out", default=str(ROOT / "experiments/single_turn/length_gee.json")
    )
    args = parser.parse_args()

    import pandas as pd

    prompt_meta = {}
    for line in Path(args.prompts).read_text().splitlines():
        record = json.loads(line)
        cls = arm_class(str(record.get("arm", "")))
        key = behavior_key(record)
        if cls is None or key is None:
            continue
        text = (record.get("turns") or [""])[0]
        prompt_meta[record["prompt_id"]] = {
            "behavior": key,
            "vern": int(cls == "vernacular"),
            "log_length": math.log(max(len(text), 1)),
        }

    rows = []
    for model in MODELS:
        for prompt_id, label in load_ensemble(Path(args.judges), model).items():
            meta = prompt_meta.get(prompt_id)
            if meta:
                rows.append({"model": model, "asr": label, **meta})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no complete four-judge records were found")
    frame["log_len_c"] = frame["log_length"] - frame["log_length"].mean()

    results = []
    for model in MODELS:
        model_frame = frame[frame["model"] == model].copy()
        if model_frame.empty:
            continue
        fit = fit_gee(model_frame, groups="behavior")
        results.append(coefficient_row(model, fit))

    frame["behavior_model"] = frame["behavior"] + "|" + frame["model"]
    pooled = fit_gee(frame, groups="behavior_model", include_model_effects=True)
    results.append(coefficient_row("Pooled model", pooled))

    print("| Model | beta_vern | OR_vern | beta_loglen | Ratio |")
    print("|---|---:|---:|---:|---:|")
    for row in results:
        ratio = "-" if row["ratio"] is None else f"{row['ratio']:.2f}"
        print(
            f"| {row['model']} | {row['beta_vern']:+.2f} | "
            f"{row['odds_ratio_vern']:.2f} | {row['beta_log_length']:+.2f} | {ratio} |"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "ensemble": "two_of_four",
                "cluster_unit": "behavior_by_model",
                "n": len(frame),
                "results": results,
            },
            indent=2,
        )
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
