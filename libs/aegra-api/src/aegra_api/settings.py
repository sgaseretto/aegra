from pathlib import Path
from typing import Annotated

from pydantic import BeforeValidator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


def parse_lower(v: str) -> str:
    """Converts to lowercase and strips whitespace."""
    return v.strip().lower() if isinstance(v, str) else v


def parse_upper(v: str) -> str:
    """Converts to uppercase and strips whitespace."""
    return v.strip().upper() if isinstance(v, str) else v


# Custom types for automatic formatting
LowerStr = Annotated[str, BeforeValidator(parse_lower)]
UpperStr = Annotated[str, BeforeValidator(parse_upper)]


class EnvBase(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class AppSettings(EnvBase):
    """General application settings."""

    PROJECT_NAME: str = "Aegra"
    VERSION: str = "0.1.0"

    # Server config
    HOST: str = "0.0.0.0"  # nosec B104
    PORT: int = 8000
    SERVER_URL: str = "http://localhost:8000"

    # App logic
    AEGRA_CONFIG: str = "aegra.json"  # Default config file path
    AUTH_TYPE: LowerStr = "noop"
    ENV_MODE: UpperStr = "LOCAL"
    DEBUG: bool = False

    # Logging
    LOG_LEVEL: UpperStr = "INFO"
    LOG_VERBOSITY: LowerStr = "verbose"


class DatabaseSettings(EnvBase):
    """Database connection settings.

    Supports both PostgreSQL and SQLite backends.
    Set ``DATABASE_URL`` to a ``sqlite:///path`` value to use SQLite mode.
    When ``DATABASE_URL`` is unset or starts with ``postgresql``, existing
    ``POSTGRES_*`` env vars are used (backward-compatible).
    """

    DATABASE_URL: str | None = None

    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: str = "5432"
    POSTGRES_DB: str = "aegra"
    DB_ECHO_LOG: bool = False

    @computed_field
    @property
    def is_sqlite(self) -> bool:
        """True when the configured backend is SQLite."""
        return self.DATABASE_URL is not None and self.DATABASE_URL.startswith("sqlite")

    @computed_field
    @property
    def database_url(self) -> str:
        """Async URL for SQLAlchemy.

        SQLite  : ``sqlite+aiosqlite:///path``
        Postgres: ``postgresql+asyncpg://…``
        """
        if self.DATABASE_URL is not None:
            raw = self.DATABASE_URL
            if raw.startswith("sqlite"):
                # Normalise to async driver: sqlite:///path → sqlite+aiosqlite:///path
                if "+aiosqlite" not in raw:
                    return raw.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
                return raw
            if raw.startswith("postgresql"):
                # Honour explicit DATABASE_URL for Postgres too
                if "+asyncpg" not in raw:
                    return raw.replace("postgresql://", "postgresql+asyncpg://", 1)
                return raw
        # Fallback: build from individual POSTGRES_* vars
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @computed_field
    @property
    def database_url_sync(self) -> str:
        """Sync URL.

        SQLite  : plain file path (for ``AsyncSqliteSaver.from_conn_string``)
        Postgres: ``postgresql://…`` (for LangGraph/psycopg)
        """
        if self.is_sqlite:
            return self.sqlite_db_path
        if self.DATABASE_URL is not None and self.DATABASE_URL.startswith("postgresql"):
            url = self.DATABASE_URL
            if "+asyncpg" in url:
                return url.replace("+asyncpg", "")
            return url
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @computed_field
    @property
    def sqlite_db_path(self) -> str:
        """Absolute path to the SQLite database file.

        Extracts the file path from ``DATABASE_URL`` (e.g. ``sqlite:///./foo.db`` → ``./foo.db``).
        Returns empty string for non-SQLite backends.
        """
        if not self.is_sqlite or self.DATABASE_URL is None:
            return ""
        url = self.DATABASE_URL
        # Strip scheme variants: sqlite+aiosqlite:/// or sqlite:///
        for prefix in ("sqlite+aiosqlite:///", "sqlite:///"):
            if url.startswith(prefix):
                raw_path = url[len(prefix) :]
                return str(Path(raw_path).resolve())
        return ""


class PoolSettings(EnvBase):
    """Connection pool settings for SQLAlchemy and LangGraph."""

    SQLALCHEMY_POOL_SIZE: int = 2
    SQLALCHEMY_MAX_OVERFLOW: int = 0

    LANGGRAPH_MIN_POOL_SIZE: int = 1
    LANGGRAPH_MAX_POOL_SIZE: int = 6


class ObservabilitySettings(EnvBase):
    """
    Unified settings for OpenTelemetry and Vendor targets.
    Supports Fan-out configuration via OTEL_TARGETS.
    """

    # General OTEL Config
    OTEL_SERVICE_NAME: str = "aegra-backend"
    OTEL_TARGETS: str = ""  # Comma-separated: "LANGFUSE,PHOENIX"
    OTEL_CONSOLE_EXPORT: bool = False  # For local debugging

    # --- Generic OTLP Target (Default/Custom) ---
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = None
    OTEL_EXPORTER_OTLP_HEADERS: str | None = None

    # --- Langfuse Specifics ---
    LANGFUSE_BASE_URL: str = "http://localhost:3000"
    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_SECRET_KEY: str | None = None

    # --- Phoenix Specifics ---
    PHOENIX_COLLECTOR_ENDPOINT: str = "http://127.0.0.1:6006/v1/traces"
    PHOENIX_API_KEY: str | None = None


class Settings:
    def __init__(self) -> None:
        self.app = AppSettings()
        self.db = DatabaseSettings()
        self.pool = PoolSettings()
        self.observability = ObservabilitySettings()


settings = Settings()
