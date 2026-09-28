"""Environment settings. Secrets come only from the environment / .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./data/tracker.db"
    config_path: str = "config.yaml"

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    app_username: str | None = None  # bootstraps the first admin if the users table is empty
    app_password: str | None = None
    session_secret: str = "change-me"
    cookie_secure: bool = True

    alert_webhook_url: str | None = None  # Slack-compatible {"text": ...} webhook
    hc_ping_url: str | None = None  # healthchecks.io dead-man switch
    max_daily_usd: float = 5.0

    provider_mode: str = "live"  # "live" = real APIs, "fake" = FakeProvider (no cost)


@lru_cache
def get_settings() -> Settings:
    return Settings()
