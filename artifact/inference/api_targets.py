"""Provider routing for closed-source chat APIs (Sections 3.5 and 4)."""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple


FRONTIER_MODELS = {"gemini-2.5-pro", "deepseek-v4-pro"}
GPT_MODELS = {"gpt-5.4-mini"}

ALL_API_TARGETS = FRONTIER_MODELS | GPT_MODELS

log = logging.getLogger(__name__)

_client_cache: Dict[Tuple[str, str], Any] = {}


def _provider_env(model_id: str) -> Tuple[str, str, Optional[str]]:
    if model_id.startswith("gemini"):
        return "GEMINI_API_KEY", "GEMINI_BASE_URL", None
    if model_id.startswith("deepseek"):
        return "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", None
    return "OPENAI_API_KEY", "OPENAI_BASE_URL", "https://api.openai.com/v1"


def _client(model_id: str) -> Any:
    """Return a cached provider-specific OpenAI-compatible client."""
    from openai import OpenAI

    key_var, base_var, default_base = _provider_env(model_id)
    key = os.environ.get(key_var)
    base = os.environ.get(base_var, default_base)
    if not key:
        raise EnvironmentError(f"{key_var} is not set for {model_id}")
    if not base:
        raise EnvironmentError(f"{base_var} is not set for {model_id}")
    cache_key = (base, key)
    if cache_key not in _client_cache:
        _client_cache[cache_key] = OpenAI(base_url=base, api_key=key, timeout=120)
    return _client_cache[cache_key]


def _call(client, model, hist, temperature, max_tokens, system, top_p):
    msgs = list(hist)
    if system and not any(m.get("role") == "system" for m in msgs):
        msgs = [{"role": "system", "content": system}] + msgs
    kwargs = {"model": model, "messages": msgs}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if top_p is not None:
        kwargs["top_p"] = top_p
    r = client.chat.completions.create(**kwargs)
    return r.choices[0].message.content or ""


def chat_multiturn(model_id, user_turns,
                   temperature=None, max_tokens=None, top_p=None,
                   system=None, retries=4, backoff=6):
    """Send user turns with retry backoff and return assistant replies."""
    if model_id not in ALL_API_TARGETS:
        raise ValueError(f"unknown model: {model_id}")
    cli = _client(model_id)

    responses, messages = [], []
    for ti, msg in enumerate(user_turns, 1):
        messages.append({"role": "user", "content": msg})
        resp, last_err = None, None
        for a in range(retries):
            try:
                resp = _call(cli, model_id, messages,
                             temperature, max_tokens, system, top_p)
                break
            except Exception as e:
                last_err = str(e)[:200]
                low = last_err.lower()
                if "429" in low or "502" in low or "503" in low or "rate" in low:
                    time.sleep(backoff * (2 ** a))
                else:
                    time.sleep(backoff)
                log.warning(f"{model_id} t{ti} a{a+1}: {last_err[:120]}")
        if resp is None:
            log.error(f"{model_id} t{ti} ALL retries failed: {last_err}")
            return None
        responses.append(resp)
        messages.append({"role": "assistant", "content": resp})
    return responses
