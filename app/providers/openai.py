"""OpenAI via the Responses API with the web_search tool."""

from typing import Any

import openai

from app.config import Config
from app.providers.base import ParsedAnswer, ProviderError, dedupe


class OpenAIProvider:
    name = "openai"

    def __init__(self, cfg: Config, api_key: str):
        self.cfg = cfg
        self.model = cfg.providers.openai.model
        self.client = openai.OpenAI(
            api_key=api_key, max_retries=cfg.sdk_max_retries, timeout=cfg.request_timeout_s
        )

    def ask(self, prompt: str, seed: str = "") -> dict[str, Any]:
        loc = self.cfg.location
        resp = self.client.responses.create(
            model=self.model,
            input=prompt,
            tools=[
                {
                    "type": "web_search",
                    "user_location": {
                        "type": "approximate",
                        "country": loc.country,
                        "city": loc.city,
                        "region": loc.region,
                        "timezone": loc.timezone,
                    },
                }
            ],
            include=["web_search_call.action.sources"],
        )
        return resp.model_dump(mode="json")


def parse(raw: dict[str, Any]) -> ParsedAnswer:
    if raw.get("error"):
        raise ProviderError(f"api error: {raw['error']}")

    texts: list[str] = []
    cited: list[str] = []
    retrieved: list[str] = []
    searches = 0
    for item in raw.get("output") or []:
        itype = item.get("type")
        if itype == "web_search_call":
            searches += 1
            action = item.get("action") or {}
            retrieved.extend(s.get("url") for s in action.get("sources") or [])
        elif itype == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    texts.append(part.get("text") or "")
                    for ann in part.get("annotations") or []:
                        if ann.get("type") == "url_citation":
                            cited.append(ann.get("url"))
                elif part.get("type") == "refusal":
                    raise ProviderError(f"refusal: {part.get('refusal')}")

    text = "\n".join(texts).strip()
    if not text:
        raise ProviderError(f"empty answer (status={raw.get('status')})")
    usage = raw.get("usage") or {}
    return ParsedAnswer(
        text=text,
        cited=dedupe(cited),
        retrieved=dedupe(retrieved),
        input_tokens=usage.get("input_tokens") or 0,
        output_tokens=usage.get("output_tokens") or 0,
        search_calls=searches,
    )
