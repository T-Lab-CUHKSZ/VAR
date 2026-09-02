#!/usr/bin/env python3
"""Build the controlled 355-excerpt pool after manual screening."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


TRUE = {"1", "true", "yes", "y"}
FALSE = {"0", "false", "no", "n"}


def flag(value: str, field: str) -> bool:
    normalized = str(value).strip().casefold()
    if normalized in TRUE:
        return True
    if normalized in FALSE:
        return False
    raise ValueError(f"{field} must be a binary 1/0 value, found {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", required=True, help="completed private review CSV")
    parser.add_argument("--config", default=str(Path(__file__).with_name("ao3_registers.json")))
    parser.add_argument("--out", required=True, help="controlled exemplar JSONL; never commit it")
    parser.add_argument("--stats-out")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    targets = {item["name"]: int(item["target_exemplars"]) for item in config["registers"]}
    cap = int(config["max_excerpt_characters"])
    accepted = []
    seen_within_register: defaultdict[str, set[str]] = defaultdict(set)
    with Path(args.review).open(encoding="utf-8", newline="") as stream:
        for line_number, row in enumerate(csv.DictReader(stream), 2):
            include = flag(row["include"], f"line {line_number} include")
            if not include:
                continue
            pii = flag(row["pii_found"], f"line {line_number} pii_found")
            identifying = flag(
                row["identifying_metadata_found"],
                f"line {line_number} identifying_metadata_found",
            )
            quality = flag(row["quality_ok"], f"line {line_number} quality_ok")
            if pii or identifying or not quality:
                raise ValueError(
                    f"line {line_number}: included rows must be PII-free, metadata-free, and quality-approved"
                )
            register = row["register"].strip()
            if register not in targets:
                raise ValueError(f"line {line_number}: unknown register {register!r}")
            excerpt = re.sub(r"\s+", " ", row["excerpt"]).strip()
            if not excerpt or len(excerpt) > cap:
                raise ValueError(f"line {line_number}: excerpt length must be 1..{cap}")
            source_id = row["source_id"].strip()
            if source_id in seen_within_register[register]:
                raise ValueError(f"line {line_number}: duplicate source in {register}: {source_id}")
            seen_within_register[register].add(source_id)
            accepted.append(
                {
                    "subgenre": register,
                    "text": excerpt,
                    "source_hash": hashlib.sha256(source_id.encode()).hexdigest(),
                    "char_len": len(excerpt),
                }
            )

    counts = Counter(row["subgenre"] for row in accepted)
    if counts != Counter(targets):
        raise ValueError(f"curated counts do not match the reported pool: got {dict(counts)}, expected {targets}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as stream:
        for row in sorted(accepted, key=lambda item: (item["subgenre"], item["source_hash"])):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    word_counts = defaultdict(list)
    for row in accepted:
        word_counts[row["subgenre"]].append(len(re.findall(r"\b\w+(?:[-’']\w+)*\b", row["text"])))
    statistics = {
        "total_exemplars": len(accepted),
        "max_excerpt_characters": cap,
        "registers": {
            register: {
                "exemplars": len(values),
                "mean_words": sum(values) / len(values),
                "min_words": min(values),
                "max_words": max(values),
            }
            for register, values in sorted(word_counts.items())
        },
    }
    if args.stats_out:
        stats_out = Path(args.stats_out)
        stats_out.parent.mkdir(parents=True, exist_ok=True)
        stats_out.write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
