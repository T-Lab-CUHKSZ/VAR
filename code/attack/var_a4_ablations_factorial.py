"""Build the eight VAR-A4 T2/T3/T4 structural-ablation cells (Table 7)."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADLINE_REGISTER = "screenplay_format_single"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--var_prompts",
        default=str(ROOT / "data/prompts/var_multiturn/var_a4_290.jsonl"),
        help="VAR-A4 records produced by attack/var_a4.py",
    )
    parser.add_argument(
        "--out_dir",
        default=str(ROOT / "data/prompts/var_multiturn/ablations"),
    )
    parser.add_argument("--register", default=HEADLINE_REGISTER)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in Path(args.var_prompts).read_text().splitlines()
        if line.strip()
    ]
    records = [record for record in records if record.get("register") == args.register]
    if len(records) != 290:
        raise RuntimeError(
            f"expected 290 record-specific T1 prompts for {args.register}; got {len(records)}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for t2_on, t3_on, t4_on in itertools.product((0, 1), repeat=3):
        bits = f"{t2_on}{t3_on}{t4_on}"
        out_path = out_dir / f"var_a4_cell_{bits}.jsonl"
        with out_path.open("w") as stream:
            for source in records:
                if len(source.get("turns") or []) != 4:
                    raise ValueError(f"invalid VAR-A4 source record: {source.get('prompt_id')}")
                selected = [source["turns"][0]]
                selected.extend(
                    turn
                    for enabled, turn in zip(
                        (t2_on, t3_on, t4_on), source["turns"][1:]
                    )
                    if enabled
                )
                record = dict(source)
                record.update(
                    {
                        "prompt_id": f"var_a4_cell_{bits}_b{source['behavior_idx']:03d}",
                        "source_prompt_id": source["prompt_id"],
                        "arm": f"var_a4_cell_{bits}",
                        "ablation_cell": bits,
                        "turn_count": len(selected),
                        "turns": selected,
                    }
                )
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"built {out_path}: 290 records")


if __name__ == "__main__":
    main()
