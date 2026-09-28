"""Tests run on Postgres, the production database.

- Default: start a throwaway postgres:16 container in Docker, use it for the whole run, remove it.
- TEST_DATABASE_URL=postgresql+psycopg://...: use that (empty) database instead, e.g. in CI.
- TEST_DB=sqlite: quick local run on SQLite temp files (not what production uses).
Each test gets an empty schema.
"""

import os
import socket
import subprocess
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Must be set before any app module reads settings. These override .env, so tests never
# see real API keys and never make paid calls.
os.environ.update({
    "CONFIG_PATH": str(ROOT / "tests" / "config.test.yaml"),  # never the production config
    "SESSION_SECRET": "test-secret",
    "COOKIE_SECURE": "false",
    "PROVIDER_MODE": "fake",
    "MAX_DAILY_USD": "5",
    "APP_USERNAME": "",
    "APP_PASSWORD": "",
    "ALERT_WEBHOOK_URL": "",
    "HC_PING_URL": "",
    "ANTHROPIC_API_KEY": "",
    "OPENAI_API_KEY": "",
})
# Never the dev/prod database from .env: tests configure their own per-test database.
os.environ["DATABASE_URL"] = "sqlite:///" + str(ROOT / "data" / "unused-by-tests.db")

import pytest  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402

from app import db  # noqa: E402
from app.config import get_config  # noqa: E402
from app.models import Base  # noqa: E402
from app.prompts import seed_prompts  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
PG_IMAGE = "postgres:16-alpine"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for(url: str, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            eng = create_engine(url)
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))
            eng.dispose()
            return
        except Exception:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.5)


@pytest.fixture(scope="session")
def pg_url():
    """Postgres URL for the run, or None when TEST_DB=sqlite."""
    if os.environ.get("TEST_DB", "").lower() == "sqlite":
        yield None
        return
    if url := os.environ.get("TEST_DATABASE_URL"):
        yield url
        return
    name, port = f"tracker-test-{uuid.uuid4().hex[:8]}", _free_port()
    try:
        subprocess.run(
            ["docker", "run", "-d", "--rm", "--name", name, "-p", f"127.0.0.1:{port}:5432",
             "-e", "POSTGRES_PASSWORD=test", "-e", "POSTGRES_DB=tracker_test", PG_IMAGE],
            check=True, capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        detail = getattr(e, "stderr", "") or str(e)
        pytest.exit("Tests need Postgres: start Docker Desktop, or set TEST_DATABASE_URL, "
                    f"or run with TEST_DB=sqlite for a quick SQLite-only run.\n{detail}",
                    returncode=2)
    url = f"postgresql+psycopg://postgres:test@127.0.0.1:{port}/tracker_test"
    try:
        _wait_for(url)
        yield url
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.fixture
def engine(pg_url, tmp_path):
    if pg_url is None:
        eng = db.configure(f"sqlite:///{tmp_path / 'test.db'}")
    else:
        eng = db.configure(pg_url)
        _reset_schema(eng)
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


def _reset_schema(eng) -> None:
    """Empty database: drops everything, including alembic_version from migration tests."""
    with eng.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))


@pytest.fixture
def cfg():
    # tests/config.test.yaml: brand Mochi HRMS, 3 samples. Independent of config.yaml.
    return get_config()


@pytest.fixture
def seeded(engine, cfg):
    with db.session() as s:
        seed_prompts(s, cfg)
    return engine
