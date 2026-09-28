"""Label how an answer describes each mentioned brand: positive / neutral / negative.

Runs in the worker only (never in the web app). One cheap-model call per answer.
"""

import re
from typing import Protocol

import anthropic

from app.config import Config

LABELS = ("positive", "neutral", "negative")
_LINE = re.compile(r"^\s*[-*]?\s*(.+?)\s*:\s*(positive|neutral|negative)\b", re.I | re.M)

INSTRUCTIONS = (
    "Below is an AI assistant's answer to a buyer question about HR software. "
    "For each brand listed, classify how the answer describes that brand: "
    "positive (recommended or praised), neutral (listed or described without judgement), "
    "or negative (criticised or advised against).\n"
    "Reply with exactly one line per brand, in the form `Brand: label`, and nothing else."
)


class Labeler(Protocol):
    model: str

    def label(self, answer: str, brands: list[str]) -> tuple[dict[str, str], dict[str, int]]: ...


def parse_labels(reply: str, brands: list[str]) -> dict[str, str]:
    wanted = {b.lower(): b for b in brands}
    out: dict[str, str] = {}
    for name, label in _LINE.findall(reply):
        brand = wanted.get(name.strip().strip("`*").lower())
        if brand:
            out[brand] = label.lower()
    return out


class ClaudeLabeler:
    def __init__(self, cfg: Config, api_key: str):
        self.model = cfg.sentiment.model
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=cfg.sdk_max_retries)

    def label(self, answer: str, brands: list[str]) -> tuple[dict[str, str], dict[str, int]]:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": f"{INSTRUCTIONS}\n\nBrands: {', '.join(brands)}\n\n"
                    f"<answer>\n{answer}\n</answer>",
                }
            ],
        )
        reply = "".join(b.text for b in resp.content if b.type == "text")
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        return parse_labels(reply, brands), usage


class FakeLabeler:
    model = "fake-labeler"

    def label(self, answer: str, brands: list[str]) -> tuple[dict[str, str], dict[str, int]]:
        return {b: "positive" if "recommend" in answer.lower() else "neutral" for b in brands}, {}
