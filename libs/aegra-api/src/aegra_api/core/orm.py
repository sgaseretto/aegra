"""SQLAlchemy ORM setup for persistent assistant/thread/run records.

This module creates:
• `Base` – the declarative base used by our models.
• `Assistant`, `Thread`, `Run` – ORM models mirroring the bootstrap tables
  already created in ``DatabaseManager._create_metadata_tables``.
• `async_session_maker` – a factory that hands out `AsyncSession` objects
  bound to the shared engine managed by `db_manager`.
• `get_session` – FastAPI dependency helper for routers.

Nothing is auto-imported by FastAPI yet; routers will `from ...core.db import get_session`.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, declarative_base, mapped_column

Base = declarative_base()

# ---------------------------------------------------------------------------
# Portable column helpers (work on both PostgreSQL and SQLite)
# ---------------------------------------------------------------------------

# Use JSON by default and JSONB on PostgreSQL for full index/query support.
PortableJSON = JSON().with_variant(JSONB(), "postgresql")

# SQLAlchemy DateTime (without timezone) stores UTC; TIMESTAMP(tz=True) on Postgres.
PortableDateTime = DateTime().with_variant(DateTime(timezone=True), "postgresql")


def _new_uuid() -> str:
    """Generate a new UUID4 string (Python-side default)."""
    return str(uuid.uuid4())


class Assistant(Base):
    __tablename__ = "assistant"

    assistant_id: Mapped[str] = mapped_column(Text, primary_key=True, default=_new_uuid)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    graph_id: Mapped[str] = mapped_column(Text, nullable=False)
    config: Mapped[dict] = mapped_column(PortableJSON, server_default=text("'{}'"))
    context: Mapped[dict] = mapped_column(PortableJSON, server_default=text("'{}'"))
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    metadata_dict: Mapped[dict] = mapped_column(PortableJSON, server_default=text("'{}'"), name="metadata")
    created_at: Mapped[datetime] = mapped_column(PortableDateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(PortableDateTime, server_default=func.now())

    __table_args__ = (
        Index("idx_assistant_user", "user_id"),
        Index("idx_assistant_user_assistant", "user_id", "assistant_id", unique=True),
        Index(
            "idx_assistant_user_graph_config",
            "user_id",
            "graph_id",
            "config",
            unique=True,
        ),
    )


class AssistantVersion(Base):
    __tablename__ = "assistant_versions"

    assistant_id: Mapped[str] = mapped_column(
        Text, ForeignKey("assistant.assistant_id", ondelete="CASCADE"), primary_key=True
    )
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    graph_id: Mapped[str] = mapped_column(Text, nullable=False)
    config: Mapped[dict | None] = mapped_column(PortableJSON)
    context: Mapped[dict | None] = mapped_column(PortableJSON)
    created_at: Mapped[datetime] = mapped_column(PortableDateTime, server_default=func.now())
    metadata_dict: Mapped[dict] = mapped_column(PortableJSON, server_default=text("'{}'"), name="metadata")
    name: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)


class Thread(Base):
    __tablename__ = "thread"

    thread_id: Mapped[str] = mapped_column(Text, primary_key=True)
    status: Mapped[str] = mapped_column(Text, server_default=text("'idle'"))
    metadata_json: Mapped[dict] = mapped_column("metadata_json", PortableJSON, server_default=text("'{}'"))
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(PortableDateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(PortableDateTime, server_default=func.now())

    __table_args__ = (Index("idx_thread_user", "user_id"),)


class Run(Base):
    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(Text, primary_key=True, default=_new_uuid)
    thread_id: Mapped[str] = mapped_column(Text, ForeignKey("thread.thread_id", ondelete="CASCADE"), nullable=False)
    assistant_id: Mapped[str | None] = mapped_column(Text, ForeignKey("assistant.assistant_id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(Text, server_default=text("'pending'"))
    input: Mapped[dict | None] = mapped_column(PortableJSON, server_default=text("'{}'"))
    config: Mapped[dict | None] = mapped_column(PortableJSON, nullable=True)
    context: Mapped[dict | None] = mapped_column(PortableJSON, nullable=True)
    output: Mapped[dict | None] = mapped_column(PortableJSON)
    error_message: Mapped[str | None] = mapped_column(Text)
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(PortableDateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(PortableDateTime, server_default=func.now())

    __table_args__ = (
        Index("idx_runs_thread_id", "thread_id"),
        Index("idx_runs_user", "user_id"),
        Index("idx_runs_status", "status"),
        Index("idx_runs_assistant_id", "assistant_id"),
        Index("idx_runs_created_at", "created_at"),
    )


class RunEvent(Base):
    __tablename__ = "run_events"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    run_id: Mapped[str] = mapped_column(Text, nullable=False)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    event: Mapped[str] = mapped_column(Text, nullable=False)
    data: Mapped[dict | None] = mapped_column(PortableJSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(PortableDateTime, server_default=func.now())

    __table_args__ = (
        Index("idx_run_events_run_id", "run_id"),
        Index("idx_run_events_seq", "run_id", "seq"),
    )


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

async_session_maker: async_sessionmaker[AsyncSession] | None = None


def _get_session_maker() -> async_sessionmaker[AsyncSession]:
    """Return a cached async_sessionmaker bound to db_manager.engine."""
    global async_session_maker
    if async_session_maker is None:
        from aegra_api.core.database import db_manager

        engine = db_manager.get_engine()
        async_session_maker = async_sessionmaker(engine, expire_on_commit=False)
    return async_session_maker


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency that yields an AsyncSession."""
    maker = _get_session_maker()
    async with maker() as session:
        yield session
