#!/usr/bin/env python3
"""Provider-API inference for Gemini-2.5-Pro and DeepSeek-V4-Pro."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
from api_targets import FRONTIER_MODELS, chat_multiturn


def safe_write(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            done.add(json.loads(line)["prompt_id"])
        except (json.JSONDecodeError, KeyError) as error:
            raise ValueError(
                f"invalid response record at {path}:{line_number}"
            ) from error
    return done


def generate_one(prompt: dict, model: str):
    try:
        responses = chat_multiturn(model, prompt["turns"], retries=4, backoff=6)
        if not responses:
            return None, {"prompt_id": prompt["prompt_id"], "reason": "no_response"}
        record = dict(prompt)
        record.update(
            {
                "responses": responses,
                "combined_response": "\n\n".join(responses),
                "turn_count": prompt.get("turn_count", len(prompt["turns"])),
            }
        )
        return record, None
    except Exception as error:
        return None, {
            "prompt_id": prompt.get("prompt_id"),
            "reason": str(error)[:300],
            "timestamp": time.time(),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--model", required=True, choices=sorted(FRONTIER_MODELS))
    parser.add_argument("--out", required=True, help="response JSONL path")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument(
        "--behavior_filter",
        help="optional JSON list of behavior_idx values for a stratified subset",
    )
    args = parser.parse_args()

    prompts = [
        json.loads(line)
        for line in Path(args.prompts).read_text().splitlines()
        if line.strip()
    ]
    for prompt in prompts:
        if not prompt.get("prompt_id") or not prompt.get("turns"):
            raise ValueError("every prompt requires prompt_id and non-empty turns")
        if not isinstance(prompt.get("behavior"), str) or not prompt["behavior"].strip():
            raise ValueError(
                f"{prompt.get('prompt_id')}: canonical behavior text is required"
            )
    if args.behavior_filter:
        selected = {int(value) for value in json.loads(Path(args.behavior_filter).read_text())}
        prompts = [
            prompt for prompt in prompts
            if prompt.get("behavior_idx") is not None
            and int(prompt["behavior_idx"]) in selected
        ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fail_path = out_path.with_name(out_path.stem + "_failures.jsonl")
    done = load_done(out_path)
    pending = [prompt for prompt in prompts if prompt["prompt_id"] not in done]
    print(f"[{args.model}] total={len(prompts)} done={len(done)} pending={len(pending)}")

    with out_path.open("a") as responses_stream, fail_path.open("a") as failures_stream:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(generate_one, prompt, args.model): prompt
                for prompt in pending
            }
            for completed, future in enumerate(as_completed(futures), 1):
                record, failure = future.result()
                if record is not None:
                    safe_write(responses_stream, record)
                else:
                    safe_write(failures_stream, failure)
                if completed % 25 == 0:
                    print(f"[{args.model}] {completed}/{len(pending)}")


if __name__ == "__main__":
    main()
