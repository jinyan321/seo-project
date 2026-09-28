"""ORM tables. Every schema change needs an Alembic migration."""

from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# none_as_null: Python None is stored as SQL NULL, not the JSON value 'null', so
# `raw IS NULL` really finds runs that never got a response.
JSONType = JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Prompt(Base):
    """One version of a tracked question. Rows are never edited in place."""

    __tablename__ = "prompts"
    __table_args__ = (UniqueConstraint("slug", "version", name="uq_prompts_slug_version"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    text: Mapped[str] = mapped_column(Text)
    intent: Mapped[str | None] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default="active")  # active | stopped
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_by: Mapped[str | None] = mapped_column(String(64))


class Batch(Base):
    __tablename__ = "batches"
    id: Mapped[int] = mapped_column(primary_key=True)
    scheduled_for: Mapped[date] = mapped_column(Date, unique=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # running | complete | partial | failed | cost_capped
    status: Mapped[str] = mapped_column(String(16), default="running")
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    runs: Mapped[list["Run"]] = relationship(back_populates="batch")


class Run(Base):
    """One prompt asked to one provider once."""

    __tablename__ = "runs"
    __table_args__ = (
        UniqueConstraint(
            "batch_id", "prompt_id", "provider", "sample_idx", name="uq_runs_task"
        ),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    batch_id: Mapped[int] = mapped_column(ForeignKey("batches.id"), index=True)
    prompt_id: Mapped[int] = mapped_column(ForeignKey("prompts.id"), index=True)
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(64))
    sample_idx: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16))  # ok | error | skipped
    error: Mapped[str | None] = mapped_column(Text)
    answer_text: Mapped[str | None] = mapped_column(Text)
    raw: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    search_calls: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    extractor_version: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    batch: Mapped[Batch] = relationship(back_populates="runs")
    prompt: Mapped[Prompt] = relationship()
    mentions: Mapped[list["Mention"]] = relationship(cascade="all, delete-orphan")
    citations: Mapped[list["Citation"]] = relationship(cascade="all, delete-orphan")


class Mention(Base):
    __tablename__ = "mentions"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    brand: Mapped[str] = mapped_column(String(64), index=True)
    is_own: Mapped[bool] = mapped_column(Boolean, default=False)
    count: Mapped[int] = mapped_column(Integer)
    rank_first: Mapped[int] = mapped_column(Integer)
    rank_list: Mapped[int | None] = mapped_column(Integer)
    snippet: Mapped[str] = mapped_column(Text)
    sentiment: Mapped[str | None] = mapped_column(String(16))


class Strategy(Base):
    """LLM-written advice for one prompt, from stored answers. Written by the worker only;
    the web app just inserts status='pending' requests and reads results."""

    __tablename__ = "strategies"
    id: Mapped[int] = mapped_column(primary_key=True)
    prompt_id: Mapped[int] = mapped_column(ForeignKey("prompts.id"), index=True)
    status: Mapped[str] = mapped_column(String(16))  # pending | running | ok | error
    requested_by: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    window_start: Mapped[date | None] = mapped_column(Date)
    window_end: Mapped[date | None] = mapped_column(Date)
    runs_used: Mapped[int] = mapped_column(Integer, default=0)
    model: Mapped[str | None] = mapped_column(String(64))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    evidence: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    output: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    error: Mapped[str | None] = mapped_column(Text)


class Citation(Base):
    __tablename__ = "citations"
    __table_args__ = (UniqueConstraint("run_id", "url", "kind", name="uq_citations_run_url_kind"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    url: Mapped[str] = mapped_column(Text)
    domain: Mapped[str] = mapped_column(String(255), index=True)
    kind: Mapped[str] = mapped_column(String(16))  # cited | retrieved | inline
    position: Mapped[int] = mapped_column(Integer)
