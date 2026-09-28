"""Claude via the Messages API with the server-side web search tool."""

from typing import Any

import anthropic

from app.config import Config
from app.providers.base import ParsedAnswer, ProviderError, dedupe

MAX_CONTINUATIONS = 5


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, cfg: Config, api_key: str):
        self.cfg = cfg
        self.model = cfg.providers.anthropic.model
        self.client = anthropic.Anthropic(
            api_key=api_key, max_retries=cfg.sdk_max_retries, timeout=cfg.request_timeout_s
        )

    def _tool(self) -> dict[str, Any]:
        a, loc = self.cfg.providers.anthropic, self.cfg.location
        return {
            "type": a.web_search_tool,
            "name": "web_search",
            "max_uses": a.max_searches,
            "user_location": {
                "type": "approximate",
                "city": loc.city,
                "region": loc.region,
                "country": loc.country,
                "timezone": loc.timezone,
            },
        }

    def ask(self, prompt: str, seed: str = "") -> dict[str, Any]:
        """Returns {"responses": [...]} - more than one when the server paused the turn."""
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        responses = []
        for _ in range(MAX_CONTINUATIONS + 1):
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.cfg.providers.anthropic.max_tokens,
                tools=[self._tool()],
                messages=messages,
            )
            responses.append(resp.model_dump(mode="json"))
            if resp.stop_reason != "pause_turn":
                break
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": resp.content},
            ]
        return {"responses": responses}


def parse(raw: dict[str, Any]) -> ParsedAnswer:
    responses = raw.get("responses") or []
    if not responses:
        raise ProviderError("no response")
    last = responses[-1]
    if last.get("stop_reason") == "refusal":
        raise ProviderError(f"refusal: {last.get('stop_details')}")

    texts: list[str] = []
    cited: list[str] = []
    retrieved: list[str] = []
    out = ParsedAnswer(text="")
    for resp in responses:
        usage = resp.get("usage") or {}
        out.input_tokens += usage.get("input_tokens") or 0
        out.output_tokens += usage.get("output_tokens") or 0
        server = usage.get("server_tool_use") or {}
        out.search_calls += server.get("web_search_requests") or 0
        for block in resp.get("content") or []:
            btype = block.get("type")
            if btype == "text":
                texts.append(block.get("text") or "")
                for c in block.get("citations") or []:
                    if c.get("type") == "web_search_result_location":
                        cited.append(c.get("url"))
            elif btype == "web_search_tool_result":
                content = block.get("content")
                if isinstance(content, list):  # an error result is a single object
                    retrieved.extend(r.get("url") for r in content if isinstance(r, dict))

    out.text = "".join(texts).strip()
    if not out.text:
        raise ProviderError(f"empty answer (stop_reason={last.get('stop_reason')})")
    out.cited = dedupe(cited)
    out.retrieved = dedupe(retrieved)
    return out
