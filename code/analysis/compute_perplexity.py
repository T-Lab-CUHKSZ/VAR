#!/usr/bin/env python3
"""Compute the per-prompt causal-LM perplexity feature for separability."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--model", required=True, help="checkpoint or local path; record this choice with results")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    model.to(device)
    model.eval()

    records = [
        json.loads(line)
        for line in Path(args.prompts).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as stream, torch.no_grad():
        for index, record in enumerate(records, 1):
            turns = record.get("turns") or []
            text = turns[0] if turns else record.get("prompt", "")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{record.get('prompt_id')}: non-empty single-turn text is required")
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            )
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            input_ids = encoded["input_ids"]
            if input_ids.shape[1] < 2:
                raise ValueError(f"{record.get('prompt_id')}: fewer than two tokens")
            loss = model(**encoded, labels=input_ids).loss
            perplexity = math.exp(float(loss))
            if not math.isfinite(perplexity):
                raise ValueError(f"{record.get('prompt_id')}: non-finite perplexity")
            stream.write(
                json.dumps(
                    {
                        "prompt_id": record["prompt_id"],
                        "perplexity": perplexity,
                        "model": args.model,
                        "tokens": int(input_ids.shape[1]),
                    }
                )
                + "\n"
            )
            if index % 100 == 0:
                print(f"{index}/{len(records)}")
    print(f"wrote {len(records)} perplexity records to {out}")


if __name__ == "__main__":
    main()
