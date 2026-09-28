"""Postgres-specific behaviour: the advisory lock and migrations producing JSONB."""

import pytest
from sqlalchemy import create_engine, text

from app.runner import batch_lock
from tests.conftest import ROOT, _reset_schema

pytestmark = pytest.mark.pg


@pytest.fixture
def pg(pg_url):
    if pg_url is None:
        pytest.skip("TEST_DB=sqlite")
    eng = create_engine(pg_url)
    _reset_schema(eng)
    yield pg_url, eng
    eng.dispose()


def test_advisory_lock_is_exclusive_across_connections(pg):
    _url, eng = pg
    with batch_lock(eng) as first:
        assert first
        with batch_lock(eng) as second:
            assert not second
    with batch_lock(eng) as again:
        assert again


def test_migrations_match_models_on_postgres(pg):
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.config import Config
    from alembic.migration import MigrationContext

    from app.models import Base

    url, eng = pg
    acfg = Config(str(ROOT / "alembic.ini"))
    acfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(acfg, "head")
    with eng.connect() as c:
        kind = c.execute(text("SELECT data_type FROM information_schema.columns "
                              "WHERE table_name='runs' AND column_name='raw'")).scalar()
        diff = compare_metadata(MigrationContext.configure(c), Base.metadata)
    assert kind == "jsonb"
    assert diff == [], diff
    command.downgrade(acfg, "base")
