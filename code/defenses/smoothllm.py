"""SmoothLLM character-perturbation defense (Robey et al., 2025)."""
from __future__ import annotations

import random
import string
from typing import Callable, List


def perturb(text: str, p: float = 0.10, kind: str = "swap",
            rng: random.Random | None = None) -> str:
    """Character-level perturbation. kind in {'swap', 'insert', 'patch'}."""
    rng = rng or random.Random()
    chars = list(text)
    n = len(chars)
    if kind == "swap":
        for i in range(n):
            if rng.random() < p:
                chars[i] = rng.choice(string.printable[:-6])
    elif kind == "insert":
        out = []
        for c in chars:
            out.append(c)
            if rng.random() < p:
                out.append(rng.choice(string.printable[:-6]))
        chars = out
    elif kind == "patch":
        L = max(1, int(p * n))
        start = rng.randint(0, max(0, n - L))
        for i in range(start, start + L):
            if i < n:
                chars[i] = rng.choice(string.printable[:-6])
    else:
        raise ValueError(f"unknown perturbation kind: {kind}")
    return "".join(chars)


def majority_vote_labels(labels: List[int]) -> int:
    if not labels:
        raise ValueError("majority vote requires at least one label")
    # A tie is conservatively treated as a failed attack.
    return int(sum(int(label) for label in labels) > len(labels) / 2)


def smoothllm(prompt: str,
              target_fn: Callable[[str], str],
              judge_fn: Callable[[str, str], int],
              N: int = 10,
              p: float = 0.10,
              kind: str = "swap",
              rng: random.Random | None = None) -> tuple[str, int]:
    """Generate, judge, and majority-vote N perturbed prompts."""
    responses, labels = [], []
    rng = rng or random.Random()
    for _ in range(N):
        x = perturb(prompt, p=p, kind=kind, rng=rng)
        y = target_fn(x)
        responses.append(y)
        labels.append(int(judge_fn(prompt, y)))
    voted = majority_vote_labels(labels)
    rep = next((r for r, l in zip(responses, labels) if l == voted), responses[0])
    return rep, voted
