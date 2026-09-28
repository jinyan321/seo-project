"""Deterministic fake answers for tests and the demo. No network, no cost."""

import hashlib
import random
from typing import Any

from app.config import Config
from app.providers.base import ParsedAnswer, ProviderError

SOURCES = [
    "https://www.g2.com/categories/hr",
    "https://www.reddit.com/r/malaysia/comments/hr_software",
    "https://www.capterra.com.my/directory/hr",
    "https://www.techinasia.com/best-hr-software-malaysia",
    "https://www.softwareadvice.com/my/hr",
]


class FakeProvider:
    def __init__(self, cfg: Config, name: str = "fake", own_bias: float = 0.3):
        self.cfg = cfg
        self.name = name
        self.model = f"{name}-model"
        self.own_bias = own_bias

    def ask(self, prompt: str, seed: str = "") -> dict[str, Any]:
        rng = random.Random(hashlib.sha256(f"{self.name}|{prompt}|{seed}".encode()).digest())
        competitors = [b.name for b in self.cfg.competitors]
        picks = rng.sample(competitors, k=min(len(competitors), rng.randint(3, 5)))
        if rng.random() < self.own_bias:
            picks.insert(rng.randint(0, len(picks)), self.cfg.brand.name)
        lines = ["Here are some good options for Malaysian businesses:", ""]
        for i, name in enumerate(picks, 1):
            b = self.cfg.find_brand(name)
            lines.append(f"{i}. **{name}** - handles EPF, SOCSO and PCB with a mobile app.")
            if b and b.domains and rng.random() < 0.5:
                lines.append(f"   See https://{b.domains[0]}/pricing?utm_source=fake")
        text = "\n".join(lines)
        cited = rng.sample(SOURCES, k=2)
        retrieved = cited + rng.sample(SOURCES, k=2)
        return {
            "fake": True,
            "text": text,
            "cited": cited,
            "retrieved": retrieved,
            "usage": {"input_tokens": 50, "output_tokens": 200, "search_calls": 1},
        }


class FailingProvider(FakeProvider):
    """Always raises, for testing failure handling."""

    def ask(self, prompt: str, seed: str = "") -> dict[str, Any]:
        raise RuntimeError("simulated API failure")


def parse(raw: dict[str, Any]) -> ParsedAnswer:
    if not raw.get("text"):
        raise ProviderError("empty answer")
    usage = raw.get("usage") or {}
    return ParsedAnswer(
        text=raw["text"],
        cited=list(raw.get("cited") or []),
        retrieved=list(raw.get("retrieved") or []),
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        search_calls=usage.get("search_calls", 0),
    )
