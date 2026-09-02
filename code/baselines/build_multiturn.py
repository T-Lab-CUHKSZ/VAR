#!/usr/bin/env python3
"""Build static four-turn baseline records from controlled behaviors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .adapters import ADAPTERS, build_baseline_record
except ImportError:
    from adapters import ADAPTERS, build_baseline_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--behaviors", required=True, help="controlled behavior JSONL")
    parser.add_argument("--method", action="append", choices=sorted(ADAPTERS), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    methods = list(dict.fromkeys(args.method))
    behaviors = [
        json.loads(line)
        for line in Path(args.behaviors).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out = Path(args.out)
    if out.resolve() == Path(args.behaviors).resolve():
        raise ValueError("--behaviors and --out must be different files")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as stream:
        for behavior in behaviors:
            for method in methods:
                stream.write(json.dumps(build_baseline_record(behavior, method), ensure_ascii=False) + "\n")
    print(f"wrote {len(behaviors) * len(methods)} records ({len(behaviors)} behaviors x {len(methods)} methods) to {out}")


if __name__ == "__main__":
    main()
