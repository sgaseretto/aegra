"""Database manager with dual-backend support (PostgreSQL + SQLite).

Initialises SQLAlchemy engine, LangGraph checkpointer, and LangGraph store.
The backend is selected automatically from ``settings.db.is_sqlite``.
"""

from __future__ import annotations

from pathlib import Path

import structlog
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from aegra_api.config import load_store_config, resolve_embed_config
from aegra_api.settings import settings

logger = structlog.get_logger(__name__)


class DatabaseManager:
    """Manages database connections and LangGraph persistence components.

    Supports both PostgreSQL and SQLite backends, selected via
    ``settings.db.is_sqlite``.
    """

    def __init__(self) -> None:
        self.engine: AsyncEngine | None = None

        # Shared psycopg pool – only used by Postgres backend
        self.lg_pool: object | None = None

        self._checkpointer: BaseCheckpointSaver | None = None
        self._store: BaseStore | None = None
        self._database_url: str = settings.db.database_url

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize database connections and LangGraph components."""
        if self.engine:
            return

        if settings.db.is_sqlite:
            await self._initialize_sqlite()
        else:
            await self._initialize_postgres()

    async def close(self) -> None:
        """Close all database connections."""
        # SQLAlchemy engine
        if self.engine:
            await self.engine.dispose()
            self.engine = None

        # SQLite-specific cleanup
        if settings.db.is_sqlite:
            if self._store is not None:
                from aegra_api.core.sqlite_store import AsyncSqliteStore

                if isinstance(self._store, AsyncSqliteStore):
                    await self._store.close()
            if self._checkpointer is not None:
                from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

                if isinstance(self._checkpointer, AsyncSqliteSaver):
                    # AsyncSqliteSaver supports async context manager protocol
                    await self._checkpointer.__aexit__(None, None, None)
        else:
            # Postgres: close shared LangGraph pool
            if self.lg_pool is not None:
                await self.lg_pool.close()  # type: ignore[union-attr]
                self.lg_pool = None

        self._checkpointer = None
        self._store = None
        logger.info("Database connections closed")

    def get_checkpointer(self) -> BaseCheckpointSaver:
        """Return the live checkpointer instance."""
        if self._checkpointer is None:
            raise RuntimeError("Database not initialized")
        return self._checkpointer

    def get_store(self) -> BaseStore:
        """Return the live store instance."""
        if self._store is None:
            raise RuntimeError("Database not initialized")
        return self._store

    def get_engine(self) -> AsyncEngine:
        """Get the SQLAlchemy engine for metadata tables."""
        if not self.engine:
            raise RuntimeError("Database not initialized")
        return self.engine

    # ------------------------------------------------------------------
    # SQLite backend
    # ------------------------------------------------------------------

    async def _initialize_sqlite(self) -> None:
        """Set up SQLAlchemy engine, checkpointer, and store for SQLite."""
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        from aegra_api.core.orm import Base
        from aegra_api.core.sqlite_store import AsyncSqliteStore

        db_path = settings.db.sqlite_db_path
        if not db_path:
            raise RuntimeError(
                f"SQLite db path could not be resolved from DATABASE_URL. DATABASE_URL={settings.db.DATABASE_URL}"
            )

        # Ensure parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # 1. SQLAlchemy engine (metadata tables)
        self.engine = create_async_engine(
            settings.db.database_url,
            echo=settings.db.DB_ECHO_LOG,
        )

        # SQLite PRAGMAs
        @event.listens_for(self.engine.sync_engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn: object, connection_record: object) -> None:
            cursor = dbapi_conn.cursor()  # type: ignore[union-attr]
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        # Create metadata tables via SQLAlchemy (skip Alembic for SQLite)
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("SQLite metadata tables created via create_all")

        # 2. LangGraph checkpointer
        self._checkpointer = AsyncSqliteSaver.from_conn_string(db_path)
        await self._checkpointer.setup()
        logger.info("AsyncSqliteSaver ready (db=%s)", db_path)

        # 3. LangGraph store (with optional semantic search)
        store_config = load_store_config()
        index_config = store_config.get("index") if store_config else None
        if index_config:
            index_config = resolve_embed_config(dict(index_config))

        self._store = AsyncSqliteStore(db_path=db_path, index=index_config)
        await self._store.setup()

        if index_config:
            embed_model = index_config.get("embed", "unknown")
            logger.info("Semantic store enabled with embeddings: %s", embed_model)

        logger.info("SQLite backend initialized (db=%s)", db_path)

    # ------------------------------------------------------------------
    # Postgres backend
    # ------------------------------------------------------------------

    async def _initialize_postgres(self) -> None:
        """Set up SQLAlchemy engine, LangGraph pool, checkpointer, and store for Postgres."""
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from langgraph.store.postgres.aio import AsyncPostgresStore
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool

        # 1. SQLAlchemy Engine (app metadata, uses asyncpg)
        self.engine = create_async_engine(
            self._database_url,
            pool_size=settings.pool.SQLALCHEMY_POOL_SIZE,
            max_overflow=settings.pool.SQLALCHEMY_MAX_OVERFLOW,
            pool_pre_ping=True,
            echo=settings.db.DB_ECHO_LOG,
        )

        lg_max = settings.pool.LANGGRAPH_MAX_POOL_SIZE
        lg_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        }

        pool = AsyncConnectionPool(
            conninfo=settings.db.database_url_sync,
            min_size=settings.pool.LANGGRAPH_MIN_POOL_SIZE,
            max_size=lg_max,
            open=False,
            kwargs=lg_kwargs,
            check=AsyncConnectionPool.check_connection,
        )
        await pool.open()
        self.lg_pool = pool

        logger.info("Initializing LangGraph components with shared pool (max %d conns)...", lg_max)

        # 2. LangGraph checkpointer
        self._checkpointer = AsyncPostgresSaver(conn=pool)
        await self._checkpointer.setup()

        # 3. LangGraph store
        store_config = load_store_config()
        index_config = store_config.get("index") if store_config else None

        self._store = AsyncPostgresStore(conn=pool, index=index_config)
        await self._store.setup()

        if index_config:
            embed_model = index_config.get("embed", "unknown")
            logger.info("Semantic store enabled with embeddings: %s", embed_model)

        logger.info("Postgres backend initialized")


# Global database manager instance
db_manager = DatabaseManager()
