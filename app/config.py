"""Load and validate config.yaml."""

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from app.settings import get_settings


class Brand(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    # True when the name is also an ordinary word (e.g. "kakitangan" = "staff" in Malay):
    # only the exact capitalisation counts as a mention.
    case_sensitive: bool = False
    description: str = ""  # one line for the strategy prompt, e.g. "Malaysian HR software"

    @property
    def names(self) -> list[str]:
        return [self.name, *self.aliases]


class AnthropicCfg(BaseModel):
    model: str = "claude-sonnet-5"
    web_search_tool: str = "web_search_20260209"
    max_tokens: int = 8000
    max_searches: int = 5


class OpenAICfg(BaseModel):
    model: str = "gpt-5"


class Providers(BaseModel):
    anthropic: AnthropicCfg = Field(default_factory=AnthropicCfg)
    openai: OpenAICfg = Field(default_factory=OpenAICfg)


class Location(BaseModel):
    country: str = "MY"
    city: str = "Kuala Lumpur"
    region: str = "Kuala Lumpur"
    timezone: str = "Asia/Kuala_Lumpur"


class Schedule(BaseModel):
    hour: int = 9
    minute: int = 7
    timezone: str = "Asia/Kuala_Lumpur"
    catch_up_on_start: bool = True


class Sentiment(BaseModel):
    enabled: bool = True
    model: str = "claude-haiku-4-5"


class StrategyCfg(BaseModel):
    enabled: bool = True
    model: str = "claude-opus-5"
    window_days: int = 7
    weekday: str = "mon"
    hour: int = 10
    max_tokens: int = 8000
    estimate_usd: float = 0.25  # cost guard estimate before the real cost is known


class Price(BaseModel):
    input: float
    output: float


class SeedPrompt(BaseModel):
    slug: str
    text: str
    intent: str | None = None


class Config(BaseModel):
    brand: Brand
    competitors: list[Brand] = Field(default_factory=list)
    providers: Providers = Field(default_factory=Providers)
    location: Location = Field(default_factory=Location)
    samples_per_prompt: int = 3
    concurrency: int = 4
    sdk_max_retries: int = 4
    request_timeout_s: float = 300
    schedule: Schedule = Field(default_factory=Schedule)
    sentiment: Sentiment = Field(default_factory=Sentiment)
    strategy: StrategyCfg = Field(default_factory=StrategyCfg)
    prices: dict[str, Price] = Field(default_factory=dict)
    search_fee_usd: dict[str, float] = Field(default_factory=dict)
    estimate_per_call_usd: float = 0.08
    seed_prompts: list[SeedPrompt] = Field(default_factory=list)

    @property
    def all_brands(self) -> list[Brand]:
        return [self.brand, *self.competitors]

    def find_brand(self, name: str) -> Brand | None:
        for b in self.all_brands:
            if b.name.lower() == name.lower():
                return b
        return None

    def call_cost(self, provider: str, model: str, usage: dict) -> float:
        price = self.prices.get(model)
        cost = usage.get("search_calls", 0) * self.search_fee_usd.get(provider, 0.0)
        if price:
            cost += usage.get("input_tokens", 0) * price.input / 1e6
            cost += usage.get("output_tokens", 0) * price.output / 1e6
        return round(cost, 6)


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        return Config.model_validate(yaml.safe_load(f))


@lru_cache
def get_config() -> Config:
    return load_config(get_settings().config_path)
