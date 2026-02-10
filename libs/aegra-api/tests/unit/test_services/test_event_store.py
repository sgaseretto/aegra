"""Unit tests for EventStore service (SQLAlchemy ORM-based)."""

import asyncio
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from aegra_api.core.orm import RunEvent
from aegra_api.core.sse import SSEEvent
from aegra_api.services.event_store import EventStore, store_sse_event


def _make_mock_session() -> AsyncMock:
    """Create a mock async session with context manager support."""
    session = AsyncMock()
    session.add = MagicMock()
    return session


def _make_mock_session_maker(session: AsyncMock) -> MagicMock:
    """Create a mock session maker that yields the given session."""
    maker = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    maker.return_value = ctx
    return maker


class TestEventStore:
    """Unit tests for EventStore class (SQLAlchemy ORM-based)."""

    @pytest.fixture
    def event_store(self) -> EventStore:
        """Create EventStore instance."""
        return EventStore()

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        """Mock async session."""
        return _make_mock_session()

    @pytest.fixture
    def mock_session_maker(self, mock_session: AsyncMock) -> MagicMock:
        """Mock session maker returning mock_session."""
        return _make_mock_session_maker(mock_session)

    @pytest.mark.asyncio
    async def test_store_event_success(
        self, event_store: EventStore, mock_session: AsyncMock, mock_session_maker: MagicMock
    ) -> None:
        """Test successful event storage."""
        run_id = "test-run-123"
        event = SSEEvent(
            id=f"{run_id}_event_1",
            event="test_event",
            data={"key": "value"},
            timestamp=datetime.now(UTC),
        )

        with patch("aegra_api.services.event_store._get_session_maker", return_value=mock_session_maker):
            await event_store.store_event(run_id, event)

        # Verify session.add was called with a RunEvent
        mock_session.add.assert_called_once()
        row = mock_session.add.call_args[0][0]
        assert isinstance(row, RunEvent)
        assert row.id == event.id
        assert row.run_id == run_id
        assert row.seq == 1
        assert row.event == event.event
        assert row.data == event.data

        # Verify commit was called
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_store_event_sequence_extraction_edge_cases(
        self, event_store: EventStore, mock_session: AsyncMock, mock_session_maker: MagicMock
    ) -> None:
        """Test sequence extraction from various event ID formats."""
        test_cases = [
            ("run_123_event_42", 42),
            ("simple_event_0", 0),
            ("run_event_999", 999),
            ("broken_format", 0),
            ("run_event_", 0),
        ]

        with patch("aegra_api.services.event_store._get_session_maker", return_value=mock_session_maker):
            for event_id, expected_seq in test_cases:
                mock_session.add.reset_mock()
                event = SSEEvent(id=event_id, event="test", data={})
                await event_store.store_event("test-run", event)

                row = mock_session.add.call_args[0][0]
                assert row.seq == expected_seq, f"Failed for event_id: {event_id}"

    @pytest.mark.asyncio
    async def test_store_event_integrity_error(
        self, event_store: EventStore, mock_session: AsyncMock, mock_session_maker: MagicMock
    ) -> None:
        """Test handling of duplicate event (IntegrityError)."""
        from sqlalchemy.exc import IntegrityError

        event = SSEEvent(id="test_event_1", event="test", data={})
        mock_session.commit.side_effect = IntegrityError("", {}, Exception())

        with patch("aegra_api.services.event_store._get_session_maker", return_value=mock_session_maker):
            # Should not raise — IntegrityError is silently handled (ON CONFLICT DO NOTHING)
            await event_store.store_event("test-run", event)

        mock_session.rollback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_events_since_success(
        self, event_store: EventStore, mock_session: AsyncMock, mock_session_maker: MagicMock
    ) -> None:
        """Test successful event retrieval with last_event_id."""
        run_id = "test-run-123"
        last_event_id = f"{run_id}_event_5"

        # Mock RunEvent rows
        row1 = MagicMock(spec=RunEvent)
        row1.id = f"{run_id}_event_6"
        row1.event = "event6"
        row1.data = {"seq": 6}
        row1.created_at = datetime.now(UTC)

        row2 = MagicMock(spec=RunEvent)
        row2.id = f"{run_id}_event_7"
        row2.event = "event7"
        row2.data = {"seq": 7}
        row2.created_at = datetime.now(UTC)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [row1, row2]
        mock_session.execute.return_value = mock_result

        with patch("aegra_api.services.event_store._get_session_maker", return_value=mock_session_maker):
            events = await event_store.get_events_since(run_id, last_event_id)

        assert len(events) == 2
        assert events[0].id == f"{run_id}_event_6"
        assert events[1].id == f"{run_id}_event_7"

    @pytest.mark.asyncio
    async def test_get_events_since_no_events(
        self, event_store: EventStore, mock_session: AsyncMock, mock_session_maker: MagicMock
    ) -> None:
        """Test retrieval when no events exist after last_event_id."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        with patch("aegra_api.services.event_store._get_session_maker", return_value=mock_session_maker):
            events = await event_store.get_events_since("test-run", "test_event_1")

        assert events == []

    @pytest.mark.asyncio
    async def test_get_all_events_success(
        self, event_store: EventStore, mock_session: AsyncMock, mock_session_maker: MagicMock
    ) -> None:
        """Test successful retrieval of all events for a run."""
        run_id = "test-run-123"

        row = MagicMock(spec=RunEvent)
        row.id = f"{run_id}_event_1"
        row.event = "start"
        row.data = {"type": "start"}
        row.created_at = datetime.now(UTC)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [row]
        mock_session.execute.return_value = mock_result

        with patch("aegra_api.services.event_store._get_session_maker", return_value=mock_session_maker):
            events = await event_store.get_all_events(run_id)

        assert len(events) == 1
        assert events[0].event == "start"

    @pytest.mark.asyncio
    async def test_cleanup_events_success(
        self, event_store: EventStore, mock_session: AsyncMock, mock_session_maker: MagicMock
    ) -> None:
        """Test successful event cleanup for a specific run."""
        with patch("aegra_api.services.event_store._get_session_maker", return_value=mock_session_maker):
            await event_store.cleanup_events("test-run-123")

        mock_session.execute.assert_awaited_once()
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_run_info_success(
        self, event_store: EventStore, mock_session: AsyncMock, mock_session_maker: MagicMock
    ) -> None:
        """Test successful retrieval of run information."""
        run_id = "test-run-123"

        # Mock the sequence range query result
        range_row = MagicMock()
        range_row.first_seq = 1
        range_row.last_seq = 5

        # Mock the last event query result
        last_row = MagicMock()
        last_row.id = f"{run_id}_event_5"
        last_row.created_at = datetime.now(UTC)

        # Two execute calls: first for range, second for last event
        range_result = MagicMock()
        range_result.one_or_none.return_value = range_row

        last_result = MagicMock()
        last_result.one_or_none.return_value = last_row

        mock_session.execute.side_effect = [range_result, last_result]

        with patch("aegra_api.services.event_store._get_session_maker", return_value=mock_session_maker):
            info = await event_store.get_run_info(run_id)

        assert info is not None
        assert info["event_count"] == 5
        assert info["last_event_id"] == f"{run_id}_event_5"

    @pytest.mark.asyncio
    async def test_get_run_info_no_events(
        self, event_store: EventStore, mock_session: AsyncMock, mock_session_maker: MagicMock
    ) -> None:
        """Test run info when no events exist."""
        range_row = MagicMock()
        range_row.last_seq = None

        range_result = MagicMock()
        range_result.one_or_none.return_value = range_row

        mock_session.execute.return_value = range_result

        with patch("aegra_api.services.event_store._get_session_maker", return_value=mock_session_maker):
            info = await event_store.get_run_info("empty-run")

        assert info is None

    @pytest.mark.asyncio
    async def test_cleanup_task_management(self, event_store: EventStore) -> None:
        """Test cleanup task start and stop functionality."""
        assert event_store._cleanup_task is None
        await event_store.start_cleanup_task()
        assert event_store._cleanup_task is not None
        await event_store.stop_cleanup_task()
        assert event_store._cleanup_task.done()

    @pytest.mark.asyncio
    async def test_cleanup_loop_functionality(
        self, event_store: EventStore, mock_session: AsyncMock, mock_session_maker: MagicMock
    ) -> None:
        """Test the cleanup loop functionality."""
        with (
            patch.object(event_store, "CLEANUP_INTERVAL", 0.01),
            patch("aegra_api.services.event_store._get_session_maker", return_value=mock_session_maker),
        ):
            await event_store.start_cleanup_task()
            await asyncio.sleep(0.05)
            await event_store.stop_cleanup_task()

        assert mock_session.execute.called, "Cleanup loop did not execute SQL"


class TestStoreSSEEvent:
    """Unit tests for store_sse_event helper function."""

    @pytest.fixture
    def mock_event_store(self) -> Mock:
        """Mock EventStore instance."""
        return Mock()

    @pytest.mark.asyncio
    async def test_store_sse_event_success(self, mock_event_store: Mock) -> None:
        """Test successful SSE event storage."""
        mock_event_store.store_event = AsyncMock()

        with patch("aegra_api.services.event_store.event_store", mock_event_store):
            run_id = "test-run-123"
            event_id = f"{run_id}_event_1"
            event_type = "test_event"
            data = {"key": "value", "complex": datetime.now(UTC)}

            result = await store_sse_event(run_id, event_id, event_type, data)

            # Verify event_store.store_event was called
            mock_event_store.store_event.assert_called_once()
            call_args = mock_event_store.store_event.call_args
            stored_run_id, stored_event = call_args[0]

            assert stored_run_id == run_id
            assert isinstance(stored_event, SSEEvent)
            assert stored_event.id == event_id
            assert stored_event.event == event_type
            # Data should be JSON-serializable (datetime converted to string)
            json_str = json.dumps(stored_event.data)
            parsed_back = json.loads(json_str)
            assert parsed_back["key"] == "value"
            assert "complex" in parsed_back  # datetime should be serialized

            # Verify return value
            assert result == stored_event

    @pytest.mark.asyncio
    async def test_store_sse_event_json_serialization(self) -> None:
        """Test that complex objects are properly JSON serialized."""
        with patch("aegra_api.services.event_store.event_store") as mock_event_store:
            mock_event_store.store_event = AsyncMock()

            # Data with non-JSON serializable object
            data = {
                "datetime": datetime.now(UTC),
                "nested": {"complex": datetime(2023, 1, 1, tzinfo=UTC)},
                "normal": "string",
            }

            await store_sse_event("run-123", "event-1", "test", data)

            # Verify the event was stored with serialized data
            call_args = mock_event_store.store_event.call_args
            _, stored_event = call_args[0]

            # Data should be JSON serializable (datetime converted to string)
            json_str = json.dumps(stored_event.data)
            parsed_back = json.loads(json_str)
            assert "datetime" in parsed_back
            assert "nested" in parsed_back
            assert parsed_back["normal"] == "string"

    @pytest.mark.asyncio
    async def test_store_sse_event_serialization_fallback(self) -> None:
        """Test fallback behavior when JSON serialization fails."""
        with patch("aegra_api.services.event_store.event_store") as mock_event_store:
            mock_event_store.store_event = AsyncMock()

            # Create an object that can't be serialized even with custom serializer
            # by making the serializer itself fail
            class UnserializableClass:
                def __str__(self) -> str:
                    # Make str() fail to force the fallback
                    raise RuntimeError("Cannot stringify")

            data = {"unserializable": UnserializableClass()}

            await store_sse_event("run-123", "event-1", "test", data)

            # Should fallback to string representation
            call_args = mock_event_store.store_event.call_args
            _, stored_event = call_args[0]

            # The stored event should have fallback data format
            assert "raw" in stored_event.data
            assert isinstance(stored_event.data["raw"], str)
            # The raw string should contain some representation of the data
            assert len(stored_event.data["raw"]) > 0
