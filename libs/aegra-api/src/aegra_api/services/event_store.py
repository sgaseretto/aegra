"""Persistent event store for SSE replay functionality (backend-agnostic).

Uses SQLAlchemy ORM so it works with both PostgreSQL and SQLite.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError

from aegra_api.core.orm import RunEvent, _get_session_maker
from aegra_api.core.serializers import GeneralSerializer
from aegra_api.core.sse import SSEEvent

logger = structlog.get_logger(__name__)


class EventStore:
    """Backend-agnostic event store for SSE replay functionality."""

    CLEANUP_INTERVAL: int = 300  # seconds

    def __init__(self) -> None:
        self._cleanup_task: asyncio.Task[None] | None = None

    async def start_cleanup_task(self) -> None:
        """Start the periodic cleanup background task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_cleanup_task(self) -> None:
        """Cancel the cleanup background task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

    async def store_event(self, run_id: str, event: SSEEvent) -> None:
        """Persist an event with sequence extracted from id suffix.

        Expected event.id format: ``f"{run_id}_event_{seq}"``.
        """
        try:
            seq = int(str(event.id).split("_event_")[-1])
        except (ValueError, IndexError):
            seq = 0

        maker = _get_session_maker()
        async with maker() as session:
            row = RunEvent(
                id=str(event.id),
                run_id=run_id,
                seq=seq,
                event=event.event,
                data=event.data,
                created_at=datetime.now(UTC),
            )
            session.add(row)
            try:
                await session.commit()
            except IntegrityError:
                # ON CONFLICT (id) DO NOTHING equivalent
                await session.rollback()

    async def get_events_since(self, run_id: str, last_event_id: str) -> list[SSEEvent]:
        """Fetch all events for run after *last_event_id* sequence."""
        try:
            last_seq = int(str(last_event_id).split("_event_")[-1])
        except (ValueError, IndexError):
            last_seq = -1

        maker = _get_session_maker()
        async with maker() as session:
            result = await session.execute(
                select(RunEvent).where(RunEvent.run_id == run_id, RunEvent.seq > last_seq).order_by(RunEvent.seq.asc())
            )
            rows = result.scalars().all()

        return [SSEEvent(id=r.id, event=r.event, data=r.data, timestamp=r.created_at) for r in rows]

    async def get_all_events(self, run_id: str) -> list[SSEEvent]:
        """Fetch all events for a run, ordered by sequence."""
        maker = _get_session_maker()
        async with maker() as session:
            result = await session.execute(
                select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.seq.asc())
            )
            rows = result.scalars().all()

        return [SSEEvent(id=r.id, event=r.event, data=r.data, timestamp=r.created_at) for r in rows]

    async def cleanup_events(self, run_id: str) -> None:
        """Delete all events for a specific run."""
        maker = _get_session_maker()
        async with maker() as session:
            await session.execute(delete(RunEvent).where(RunEvent.run_id == run_id))
            await session.commit()

    async def get_run_info(self, run_id: str) -> dict | None:
        """Get summary info about a run's events."""
        maker = _get_session_maker()
        async with maker() as session:
            # Fetch sequence range
            result = await session.execute(
                select(
                    func.min(RunEvent.seq).label("first_seq"),
                    func.max(RunEvent.seq).label("last_seq"),
                ).where(RunEvent.run_id == run_id)
            )
            row = result.one_or_none()

            if not row or row.last_seq is None:
                return None

            # Fetch last event
            last_result = await session.execute(
                select(RunEvent.id, RunEvent.created_at)
                .where(RunEvent.run_id == run_id, RunEvent.seq == row.last_seq)
                .limit(1)
            )
            last = last_result.one_or_none()

        event_count = int(row.last_seq) - int(row.first_seq) + 1 if row.first_seq is not None else 0
        return {
            "run_id": run_id,
            "event_count": event_count,
            "first_event_time": None,
            "last_event_time": last.created_at if last else None,
            "last_event_id": last.id if last else None,
        }

    async def _cleanup_loop(self) -> None:
        """Periodically remove old events."""
        while True:
            try:
                await asyncio.sleep(self.CLEANUP_INTERVAL)
                await self._cleanup_old_runs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in event store cleanup: %s", e)

    async def _cleanup_old_runs(self) -> None:
        """Retain events for 1 hour by default."""
        cutoff = datetime.now(UTC) - timedelta(hours=1)
        try:
            maker = _get_session_maker()
            async with maker() as session:
                await session.execute(delete(RunEvent).where(RunEvent.created_at < cutoff))
                await session.commit()
        except Exception as e:
            logger.error("Failed to cleanup old runs: %s", e)


# Global event store instance
event_store = EventStore()


async def store_sse_event(run_id: str, event_id: str, event_type: str, data: dict) -> SSEEvent:
    """Store SSE event with proper serialization."""
    serializer = GeneralSerializer()

    try:
        safe_data = json.loads(json.dumps(data, default=serializer.serialize))
    except Exception:
        safe_data = {"raw": str(data)}

    event = SSEEvent(id=event_id, event=event_type, data=safe_data, timestamp=datetime.now(UTC))
    await event_store.store_event(run_id, event)
    return event
