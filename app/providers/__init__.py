"""Provider registry. Parsers are keyed by `runs.provider`, so reextract needs no API keys."""

from collections.abc import Callable
from typing import Any

from app.config import Config
from app.providers import anthropic as anthropic_mod
from app.providers import fake as fake_mod
from app.providers import openai as openai_mod
from app.providers.base import ParsedAnswer, Provider, ProviderError
from app.settings import Settings

PARSERS: dict[str, Callable[[dict[str, Any]], ParsedAnswer]] = {
    "anthropic": anthropic_mod.parse,
    "openai": openai_mod.parse,
}


def parse_raw(provider: str, raw: dict[str, Any]) -> ParsedAnswer:
    if raw.get("fake"):
        return fake_mod.parse(raw)
    return PARSERS[provider](raw)


def build_providers(cfg: Config, settings: Settings) -> list[Provider]:
    if settings.provider_mode == "fake":
        return [
            fake_mod.FakeProvider(cfg, "anthropic", own_bias=0.35),
            fake_mod.FakeProvider(cfg, "openai", own_bias=0.2),
        ]
    missing = [
        k for k, v in (("ANTHROPIC_API_KEY", settings.anthropic_api_key),
                       ("OPENAI_API_KEY", settings.openai_api_key)) if not v
    ]
    if missing:
        raise RuntimeError(f"missing {', '.join(missing)} (or set PROVIDER_MODE=fake)")
    return [
        anthropic_mod.AnthropicProvider(cfg, settings.anthropic_api_key),  # type: ignore[arg-type]
        openai_mod.OpenAIProvider(cfg, settings.openai_api_key),  # type: ignore[arg-type]
    ]


__all__ = ["ParsedAnswer", "Provider", "ProviderError", "build_providers", "parse_raw"]
