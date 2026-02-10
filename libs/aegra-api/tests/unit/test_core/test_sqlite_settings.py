"""Tests for SQLite backend detection and URL construction in settings."""

from aegra_api.settings import DatabaseSettings


class TestSqliteDetection:
    """Test is_sqlite property and URL construction."""

    def test_is_sqlite_true_for_sqlite_url(self) -> None:
        """DATABASE_URL starting with 'sqlite' should enable SQLite mode."""
        s = DatabaseSettings(DATABASE_URL="sqlite:///./test.db")
        assert s.is_sqlite is True

    def test_is_sqlite_false_for_postgres_url(self) -> None:
        """DATABASE_URL starting with 'postgresql' should not enable SQLite mode."""
        s = DatabaseSettings(DATABASE_URL="postgresql://user:pass@localhost/db")
        assert s.is_sqlite is False

    def test_is_sqlite_false_when_no_url(self) -> None:
        """Without DATABASE_URL, default Postgres vars are used."""
        s = DatabaseSettings(DATABASE_URL=None)
        assert s.is_sqlite is False

    def test_database_url_async_sqlite(self) -> None:
        """Async URL should use aiosqlite driver for SQLite."""
        s = DatabaseSettings(DATABASE_URL="sqlite:///./test.db")
        assert s.database_url == "sqlite+aiosqlite:///./test.db"

    def test_database_url_preserves_aiosqlite_driver(self) -> None:
        """Already-async SQLite URL should not be modified."""
        s = DatabaseSettings(DATABASE_URL="sqlite+aiosqlite:///./test.db")
        assert s.database_url == "sqlite+aiosqlite:///./test.db"

    def test_database_url_async_postgres(self) -> None:
        """Async URL should use asyncpg driver for Postgres."""
        s = DatabaseSettings(DATABASE_URL="postgresql://user:pass@localhost/db")
        assert s.database_url == "postgresql+asyncpg://user:pass@localhost/db"

    def test_database_url_preserves_asyncpg_driver(self) -> None:
        """Already-async Postgres URL should not be modified."""
        s = DatabaseSettings(DATABASE_URL="postgresql+asyncpg://user:pass@localhost/db")
        assert s.database_url == "postgresql+asyncpg://user:pass@localhost/db"

    def test_database_url_fallback_to_postgres_vars(self) -> None:
        """Without DATABASE_URL, build from POSTGRES_* env vars."""
        s = DatabaseSettings(
            DATABASE_URL=None,
            POSTGRES_USER="u",
            POSTGRES_PASSWORD="p",
            POSTGRES_HOST="h",
            POSTGRES_PORT="5432",
            POSTGRES_DB="d",
        )
        assert s.database_url == "postgresql+asyncpg://u:p@h:5432/d"

    def test_database_url_sync_sqlite_returns_file_path(self) -> None:
        """Sync URL for SQLite should be the resolved file path."""
        s = DatabaseSettings(DATABASE_URL="sqlite:///./test.db")
        # Should be an absolute path, not a URL
        assert not s.database_url_sync.startswith("sqlite")
        assert s.database_url_sync.endswith("test.db")

    def test_database_url_sync_postgres(self) -> None:
        """Sync URL for Postgres should use plain postgresql:// scheme."""
        s = DatabaseSettings(DATABASE_URL="postgresql+asyncpg://u:p@h:5432/d")
        assert s.database_url_sync == "postgresql://u:p@h:5432/d"

    def test_sqlite_db_path_extracts_path(self) -> None:
        """sqlite_db_path should extract and resolve the file path."""
        s = DatabaseSettings(DATABASE_URL="sqlite:///./my_app.db")
        assert s.sqlite_db_path.endswith("my_app.db")
        assert "/" in s.sqlite_db_path  # should be absolute

    def test_sqlite_db_path_empty_for_postgres(self) -> None:
        """sqlite_db_path should be empty for Postgres backend."""
        s = DatabaseSettings(DATABASE_URL="postgresql://u:p@h:5432/d")
        assert s.sqlite_db_path == ""

    def test_sqlite_db_path_empty_when_no_url(self) -> None:
        """sqlite_db_path should be empty without DATABASE_URL."""
        s = DatabaseSettings(DATABASE_URL=None)
        assert s.sqlite_db_path == ""
