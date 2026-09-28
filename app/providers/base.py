"""Provider interface. `ask()` makes the API call and returns raw JSON; `parse()` is pure."""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ParsedAnswer:
    text: str
    cited: list[str] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    search_calls: int = 0

    @property
    def usage(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "search_calls": self.search_calls,
        }


class ProviderError(Exception):
    """A call that finished but can't be used (refusal, empty answer). Never retried."""


class Provider(Protocol):
    name: str
    model: str

    def ask(self, prompt: str, seed: str) -> dict[str, Any]: ...


def dedupe(urls: list[str]) -> list[str]:
    return list(dict.fromkeys(u for u in urls if u))
