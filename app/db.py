"""Engine and session factory. `configure()` lets tests and the CLI point at another database."""

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.settings import get_settings

_engine: Engine | None = None
_Session: sessionmaker[Session] | None = None


def ensure_sqlite_dir(url: str) -> None:
    """SQLite can't create missing folders, so create the database file's folder first."""
    if url.startswith("sqlite"):
        db_path = url.split("///", 1)[-1]
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def configure(url: str) -> Engine:
    global _engine, _Session
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
        ensure_sqlite_dir(url)
    _engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
    if url.startswith("sqlite"):
        event.listen(_engine, "connect", _sqlite_pragmas)
    _Session = sessionmaker(_engine, expire_on_commit=False)
    return _engine


def _sqlite_pragmas(dbapi_conn, _record) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


def get_engine() -> Engine:
    if _engine is None:
        configure(get_settings().database_url)
    assert _engine is not None
    return _engine


def session() -> Session:
    get_engine()
    assert _Session is not None
    return _Session()


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    with session() as s:
        yield s
