import json

import pytest

from app.providers import ProviderError, parse_raw
from app.providers.anthropic import parse as parse_anthropic
from app.providers.openai import parse as parse_openai
from app.sentiment import parse_labels
from tests.conftest import FIXTURES


def load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_anthropic_parse():
    p = parse_anthropic(load("anthropic_web_search.json"))
    assert "Mochi HRMS" in p.text and "1. **Talenox**" in p.text
    assert p.cited == ["https://talenox.com/", "https://www.g2.com/categories/hr?utm_source=x"]
    assert len(p.retrieved) == 3
    assert (p.input_tokens, p.output_tokens, p.search_calls) == (5200, 410, 1)


def test_anthropic_sums_paused_turns_and_handles_search_errors():
    raw = {"responses": [
        {"stop_reason": "pause_turn", "usage": {"input_tokens": 10, "output_tokens": 1,
                                                "server_tool_use": {"web_search_requests": 2}},
         "content": [{"type": "web_search_tool_result",
                      "content": {"type": "web_search_tool_result_error",
                                  "error_code": "max_uses_exceeded"}}]},
        {"stop_reason": "end_turn", "usage": {"input_tokens": 20, "output_tokens": 5},
         "content": [{"type": "text", "text": "Answer"}]},
    ]}
    p = parse_anthropic(raw)
    assert (p.text, p.input_tokens, p.search_calls, p.retrieved) == ("Answer", 30, 2, [])


def test_anthropic_refusal_is_an_error():
    with pytest.raises(ProviderError):
        parse_anthropic({"responses": [{"stop_reason": "refusal", "content": []}]})


def test_openai_parse():
    p = parse_openai(load("openai_web_search.json"))
    assert p.text.startswith("Popular choices")
    assert p.cited == ["https://swingvy.com/my/?utm_source=openai",
                       "https://www.capterra.com.my/directory/hr?utm_source=openai"]
    assert p.retrieved == ["https://www.capterra.com.my/directory/hr", "https://swingvy.com/my/"]
    assert (p.search_calls, p.input_tokens, p.output_tokens) == (2, 3100, 620)


def test_openai_empty_is_an_error():
    with pytest.raises(ProviderError):
        parse_openai({"status": "incomplete", "output": []})


def test_parse_raw_routes_fake():
    assert parse_raw("anthropic", {"fake": True, "text": "x"}).text == "x"


def test_sentiment_label_parsing():
    reply = "Talenox: positive\n- **Mochi HRMS**: Negative\nUnknown: neutral"
    assert parse_labels(reply, ["Talenox", "Mochi HRMS"]) == {
        "Talenox": "positive", "Mochi HRMS": "negative"}
