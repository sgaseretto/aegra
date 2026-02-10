"""Tests for SQLite migration skip behaviour."""

from unittest.mock import AsyncMock, patch

import pytest


class TestSqliteMigrationSkip:
    """Verify Alembic is skipped when running in SQLite mode."""

    @pytest.mark.asyncio
    async def test_run_migrations_async_skips_alembic_for_sqlite(self) -> None:
        """When is_sqlite is True, Alembic should NOT be invoked."""
        with (
            patch("aegra_api.settings.DatabaseSettings.is_sqlite", new_callable=lambda: property(lambda self: True)),
            patch("aegra_api.core.migrations.asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
        ):
            from aegra_api.core.migrations import run_migrations_async

            await run_migrations_async()

            mock_to_thread.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_migrations_async_runs_alembic_for_postgres(self) -> None:
        """When is_sqlite is False, Alembic should be invoked normally."""
        with patch("aegra_api.core.migrations.asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            from aegra_api.core.migrations import run_migrations_async

            await run_migrations_async()

            mock_to_thread.assert_called_once()
