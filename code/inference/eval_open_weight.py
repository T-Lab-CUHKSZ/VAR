#!/usr/bin/env python3
"""Open-weight target inference with the Section 4 generation settings."""
import argparse
import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


_MR = os.environ.get("MODEL_ROOT", "").rstrip("/")
def _resolve(local_name, hf_id):
    return f"{_MR}/{local_name}" if _MR else hf_id

MODEL_PATHS = {
    "qwen3.5-9b":    _resolve("Qwen3.5-9B",                "Qwen/Qwen3.5-9B"),
    "qwen3.5-4b":    _resolve("Qwen3.5-4B",                "Qwen/Qwen3.5-4B"),
    "llama-3.1-8b":  _resolve("Llama-3.1-8B-Instruct",     "meta-llama/Llama-3.1-8B-Instruct"),
    "llama-3.2-3b":  _resolve("Llama-3.2-3B-Instruct",     "meta-llama/Llama-3.2-3B-Instruct"),
    "gemma-4-E2B":   _resolve("gemma-4-E2B-it",            "google/gemma-4-E2B-it"),
    "gemma-4-E4B":   _resolve("gemma-4-E4B-it",            "google/gemma-4-E4B-it"),
}

async def chat_multiturn_async(engine, tokenizer, turns, sampling_params):
    """Run one serial multi-turn conversation."""
    messages = []
    responses = []
    for user_msg in turns:
        messages.append({"role": "user", "content": user_msg})
        try:
            prompt_input = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            prompt_input = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        req_id = str(uuid.uuid4())
        final_output = None
        async for output in engine.generate(prompt_input, sampling_params, req_id):
            final_output = output
        if final_output and final_output.outputs:
            resp = final_output.outputs[0].text.strip()
        else:
            resp = ""
        messages.append({"role": "assistant", "content": resp})
        responses.append(resp)
    return responses


def tuple_seed(base_seed, model, prompt_id):
    """Stable seed for one (model, behavior, arm) prompt record."""
    digest = hashlib.sha256(f"{base_seed}|{model}|{prompt_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def response_record(prompt, responses):
    """Preserve judge conditioning and the concatenated multi-turn output."""
    combined = "\n\n".join(responses)
    record = {
        "prompt_id": prompt["prompt_id"],
        "turn_count": prompt.get("turn_count", len(prompt["turns"])),
        "turns": prompt["turns"],
        "responses": responses,
        "combined_response": combined,
    }
    for key in (
        "behavior", "behavior_id", "behavior_idx", "benchmark", "hazard",
        "arm", "register", "structure", "source_prompt_id",
    ):
        if key in prompt:
            record[key] = prompt[key]
    return record


def load_completed_prompt_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            completed.add(json.loads(line)["prompt_id"])
        except (json.JSONDecodeError, KeyError) as error:
            raise ValueError(
                f"invalid response record at {path}:{line_number}"
            ) from error
    return completed


async def process_prompt(engine, tokenizer, sampling_cls, sampling_config,
                         prompt, model, base_seed, fp, lock, sem, progress):
    """Generate and save one response record."""
    async with sem:
        try:
            sampling_params = sampling_cls(
                **sampling_config,
                seed=tuple_seed(base_seed, model, prompt["prompt_id"]),
            )
            responses = await chat_multiturn_async(
                engine, tokenizer, prompt["turns"], sampling_params
            )
        except Exception as e:
            log.warning(f"{prompt['prompt_id']} gen fail: {e}")
            return
    record = response_record(prompt, responses)
    async with lock:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        fp.flush()
        progress["done"] += 1
        n = progress["done"]
        if n % 20 == 0:
            elapsed = time.time() - progress["t0"]
            rate = n / max(elapsed, 1)
            eta = (progress["total"] - n) / max(rate, 0.01)
            log.info(
                f"[{n}/{progress['total']}] elapsed={elapsed:.0f}s "
                f"rate={rate*60:.1f}/min eta={eta/60:.0f}min"
            )


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--model", required=True, choices=list(MODEL_PATHS.keys()))
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=32,
                    help="number of prompts to process concurrently")
    ap.add_argument("--gpu_mem_util", type=float, default=0.30)
    ap.add_argument("--enforce_eager", action="store_true")
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument(
        "--seed", type=int, default=42,
        help="base seed; a stable seed is derived per (model, behavior, arm) record",
    )
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    resp_fp = out_dir / f"{args.model}_responses.jsonl"

    prompts = [json.loads(l) for l in open(args.prompts)]
    for prompt in prompts:
        if not prompt.get("prompt_id") or not prompt.get("turns"):
            raise ValueError("every prompt requires prompt_id and non-empty turns")
        if not isinstance(prompt.get("behavior"), str) or not prompt["behavior"].strip():
            raise ValueError(
                f"{prompt.get('prompt_id')}: canonical behavior text is required"
            )
    if args.limit:
        prompts = prompts[:args.limit]
    log.info(f"loaded {len(prompts)} prompts for model={args.model}")

    done = load_completed_prompt_ids(resp_fp)
    if done:
        log.info(f"resume: {len(done)} done, {len(prompts) - len(done)} to go")

    pending = [p for p in prompts if p["prompt_id"] not in done]
    if not pending:
        log.info("nothing to do")
        return

    log.info(f"loading model from {MODEL_PATHS[args.model]}")
    engine_kwargs = dict(
        model=MODEL_PATHS[args.model],
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )
    engine_args = AsyncEngineArgs(**engine_kwargs)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS[args.model])

    sampling_config = dict(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
    )

    log.info(f"starting async eval concurrency={args.concurrency}")
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(args.concurrency)
    progress = {"done": 0, "total": len(pending), "t0": time.time()}

    with open(resp_fp, "a") as fp:
        await asyncio.gather(*[
            process_prompt(
                engine, tokenizer, SamplingParams, sampling_config, p,
                args.model, args.seed, fp, lock, sem, progress,
            )
            for p in pending
        ])

    total_elapsed = time.time() - progress["t0"]
    log.info(
        f"DONE {progress['done']} prompts in {total_elapsed:.0f}s "
        f"= {progress['done']/total_elapsed*60:.1f}/min"
    )


if __name__ == "__main__":
    asyncio.run(main())
