#!/usr/bin/env python3
"""Normalize authorized baseline records to the shared data contract."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", required=True, choices=("fitd", "actorattack", "crescendo"))
    parser.add_argument("--turns-field", default="turns")
    parser.add_argument("--behavior-field", default="behavior")
    parser.add_argument("--behavior-id-field", default="behavior_id")
    parser.add_argument("--hazard-field", default="hazard")
    parser.add_argument(
        "--allow-variable-turns",
        action="store_true",
        help="allow records outside the paper's four-turn comparison contract",
    )
    args = parser.parse_args()

    source, out = Path(args.input), Path(args.out)
    if source.resolve() == out.resolve():
        raise ValueError("--input and --out must be different files")
    rows = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        behavior = raw.get(args.behavior_field)
        turns = raw.get(args.turns_field)
        if not isinstance(behavior, str) or not behavior.strip():
            raise ValueError(f"line {line_number}: missing canonical behavior text")
        if not isinstance(turns, list) or not turns or any(not isinstance(turn, str) or not turn.strip() for turn in turns):
            raise ValueError(f"line {line_number}: turns must be a non-empty list of strings")
        if not args.allow_variable_turns and len(turns) != 4:
            raise ValueError(f"line {line_number}: expected four turns, found {len(turns)}")
        behavior_id = raw.get(args.behavior_id_field, line_number - 1)
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(behavior_id)).strip("-")
        record = {
            "prompt_id": f"{args.method}_external_{safe_id}",
            "behavior": behavior.strip(),
            "behavior_id": behavior_id,
            "hazard": raw.get(args.hazard_field, "unknown"),
            "arm": f"{args.method}_external",
            "baseline_family": args.method,
            "adapter_variant": "authorized_external",
            "turn_count": len(turns),
            "turns": [turn.strip() for turn in turns],
        }
        rows.append(record)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as stream:
        for record in rows:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} normalized records to {out}")


if __name__ == "__main__":
    main()
