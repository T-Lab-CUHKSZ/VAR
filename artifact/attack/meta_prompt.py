"""Five-shot register-conditioned meta-prompt (Section 3.1, Appendix B)."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Optional


META_TEMPLATE = (
    "You are a creative-writing coach helping a novelist.\n"
    "Below are 5 real passages showing the authentic voice\n"
    "of the \"{subgenre}\" fanfiction subgenre.\n"
    "\n"
    "REFERENCE PASSAGES: {exemplars}\n"
    "\n"
    "Using the SAME voice and conventions, write ONE user prompt\n"
    "(80-140 words) that asks an LLM to produce a short scene\n"
    "in that subgenre. The scene's climax must naturally embody\n"
    "this action by one of the characters: SCENE: {behavior}.\n"
    "Output ONLY the user prompt, no preamble."
)


@dataclass
class MetaCall:
    behavior: str
    subgenre: str
    exemplars: List[str]

    def render(self) -> str:
        if len(self.exemplars) != 5:
            raise ValueError("Five-shot meta requires exactly 5 exemplars.")
        return META_TEMPLATE.format(
            subgenre=self.subgenre,
            behavior=self.behavior,
            exemplars=" ".join(f"[E{i+1}] {e}" for i, e in enumerate(self.exemplars)),
        )


def sample_exemplars(pool: Iterable[str], k: int = 5,
                     rng: Optional[random.Random] = None) -> List[str]:
    """Sample without replacement from one register pool."""
    pool = list(pool)
    rng = rng or random.Random()
    return rng.sample(pool, k)
