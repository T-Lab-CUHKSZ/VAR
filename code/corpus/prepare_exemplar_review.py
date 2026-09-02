#!/usr/bin/env python3
"""Prepare a private manual-screening sheet from controlled AO3 candidates."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


FIELDS = [
    "source_id",
    "register",
    "source_url",
    "excerpt",
    "include",
    "pii_found",
    "identifying_metadata_found",
    "quality_ok",
    "notes",
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def representative_excerpt(text: str, limit: int) -> str:
    body = normalize(text)
    if len(body) <= limit:
        return body
    midpoint = len(body) // 2
    start = max(0, midpoint - limit // 2)
    window = body[start : start + limit + 120]
    if start:
        boundary = re.search(r"[.!?][\"'’”)]?\s+", window[:100])
        if boundary:
            window = window[boundary.end() :]
    excerpt = window[:limit]
    final_boundary = max(excerpt.rfind(". "), excerpt.rfind("? "), excerpt.rfind("! "))
    if final_boundary >= int(limit * 0.65):
        excerpt = excerpt[: final_boundary + 1]
    return excerpt.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-characters", type=int, default=350)
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.raw_dir).glob("*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "source_id": record["source_id"],
                "register": record["register"],
                "source_url": record["source_url"],
                "excerpt": representative_excerpt(record["text"], args.max_characters),
                "include": "",
                "pii_found": "",
                "identifying_metadata_found": "",
                "quality_ok": "",
                "notes": "",
            }
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} private review rows to {out}")


if __name__ == "__main__":
    main()
