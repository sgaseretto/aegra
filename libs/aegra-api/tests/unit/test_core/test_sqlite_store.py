"""Integration tests for AsyncSqliteStore using a real temp SQLite database.

These tests exercise the actual SQLite + sqlite-vec stack without mocking,
verifying CRUD, namespace handling, filtering, and list_namespaces.
Semantic search tests are excluded here (require embeddings model).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.store.base import GetOp, ListNamespacesOp, PutOp, SearchOp

from aegra_api.core.sqlite_store import AsyncSqliteStore


@pytest.fixture
async def store(tmp_path: Path) -> AsyncSqliteStore:
    """Create a real AsyncSqliteStore backed by a temp file (no embeddings)."""
    db_path = str(tmp_path / "test_store.db")
    s = AsyncSqliteStore(db_path=db_path, index=None)
    await s.setup()
    yield s  # type: ignore[misc]
    await s.close()


class TestAsyncSqliteStoreCRUD:
    """Basic put/get/delete operations."""

    @pytest.mark.asyncio
    async def test_put_and_get(self, store: AsyncSqliteStore) -> None:
        """Put an item then get it back."""
        ns = ("users",)
        key = "u1"
        value = {"name": "Alice", "age": 30}

        await store.abatch([PutOp(namespace=ns, key=key, value=value)])
        results = await store.abatch([GetOp(namespace=ns, key=key)])

        item = results[0]
        assert item is not None
        assert item.key == key
        assert item.namespace == ns
        assert item.value == value

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, store: AsyncSqliteStore) -> None:
        """Getting a key that doesn't exist should return None."""
        results = await store.abatch([GetOp(namespace=("x",), key="missing")])
        assert results[0] is None

    @pytest.mark.asyncio
    async def test_put_overwrite(self, store: AsyncSqliteStore) -> None:
        """Putting the same key twice should overwrite the value."""
        ns = ("items",)
        key = "k1"

        await store.abatch([PutOp(namespace=ns, key=key, value={"v": 1})])
        await store.abatch([PutOp(namespace=ns, key=key, value={"v": 2})])

        results = await store.abatch([GetOp(namespace=ns, key=key)])
        assert results[0].value == {"v": 2}

    @pytest.mark.asyncio
    async def test_delete_item(self, store: AsyncSqliteStore) -> None:
        """Putting value=None should delete the item."""
        ns = ("tmp",)
        key = "del_me"

        await store.abatch([PutOp(namespace=ns, key=key, value={"data": True})])
        # Delete
        await store.abatch([PutOp(namespace=ns, key=key, value=None)])

        results = await store.abatch([GetOp(namespace=ns, key=key)])
        assert results[0] is None

    @pytest.mark.asyncio
    async def test_batch_multiple_ops(self, store: AsyncSqliteStore) -> None:
        """Multiple operations in a single batch should all execute."""
        ns = ("batch",)
        await store.abatch(
            [
                PutOp(namespace=ns, key="a", value={"v": 1}),
                PutOp(namespace=ns, key="b", value={"v": 2}),
                PutOp(namespace=ns, key="c", value={"v": 3}),
            ]
        )

        results = await store.abatch(
            [
                GetOp(namespace=ns, key="a"),
                GetOp(namespace=ns, key="b"),
                GetOp(namespace=ns, key="c"),
            ]
        )

        assert results[0].value == {"v": 1}
        assert results[1].value == {"v": 2}
        assert results[2].value == {"v": 3}


class TestAsyncSqliteStoreSearch:
    """Non-semantic (filter-based) search operations."""

    @pytest.mark.asyncio
    async def test_search_by_namespace_prefix(self, store: AsyncSqliteStore) -> None:
        """Search should return items matching namespace prefix."""
        await store.abatch(
            [
                PutOp(namespace=("docs", "public"), key="d1", value={"title": "Doc 1"}),
                PutOp(namespace=("docs", "private"), key="d2", value={"title": "Doc 2"}),
                PutOp(namespace=("other",), key="o1", value={"title": "Other"}),
            ]
        )

        results = await store.abatch(
            [
                SearchOp(namespace_prefix=("docs",), filter=None, limit=10, offset=0),
            ]
        )

        items = results[0]
        assert len(items) == 2
        titles = {i.value["title"] for i in items}
        assert titles == {"Doc 1", "Doc 2"}

    @pytest.mark.asyncio
    async def test_search_with_filter(self, store: AsyncSqliteStore) -> None:
        """Search should apply filter conditions."""
        ns = ("items",)
        await store.abatch(
            [
                PutOp(namespace=ns, key="a", value={"status": "active", "score": 10}),
                PutOp(namespace=ns, key="b", value={"status": "inactive", "score": 20}),
                PutOp(namespace=ns, key="c", value={"status": "active", "score": 30}),
            ]
        )

        results = await store.abatch(
            [
                SearchOp(namespace_prefix=ns, filter={"status": "active"}, limit=10, offset=0),
            ]
        )

        items = results[0]
        assert len(items) == 2
        keys = {i.key for i in items}
        assert keys == {"a", "c"}

    @pytest.mark.asyncio
    async def test_search_with_offset_and_limit(self, store: AsyncSqliteStore) -> None:
        """Search should respect offset and limit."""
        ns = ("paginate",)
        for i in range(5):
            await store.abatch([PutOp(namespace=ns, key=f"k{i}", value={"idx": i})])

        results = await store.abatch(
            [
                SearchOp(namespace_prefix=ns, filter=None, limit=2, offset=1),
            ]
        )

        items = results[0]
        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_search_empty_prefix_returns_all(self, store: AsyncSqliteStore) -> None:
        """Empty namespace prefix should match all items."""
        await store.abatch(
            [
                PutOp(namespace=("a",), key="1", value={"v": 1}),
                PutOp(namespace=("b",), key="2", value={"v": 2}),
            ]
        )

        results = await store.abatch(
            [
                SearchOp(namespace_prefix=(), filter=None, limit=10, offset=0),
            ]
        )

        assert len(results[0]) == 2


class TestAsyncSqliteStoreListNamespaces:
    """ListNamespacesOp tests."""

    @pytest.mark.asyncio
    async def test_list_namespaces(self, store: AsyncSqliteStore) -> None:
        """Should return all distinct namespaces."""
        await store.abatch(
            [
                PutOp(namespace=("users", "admin"), key="u1", value={"n": 1}),
                PutOp(namespace=("users", "guest"), key="u2", value={"n": 2}),
                PutOp(namespace=("logs",), key="l1", value={"n": 3}),
            ]
        )

        results = await store.abatch(
            [
                ListNamespacesOp(match_conditions=None, max_depth=None, limit=100, offset=0),
            ]
        )

        namespaces = results[0]
        assert ("logs",) in namespaces
        assert ("users", "admin") in namespaces
        assert ("users", "guest") in namespaces

    @pytest.mark.asyncio
    async def test_list_namespaces_with_max_depth(self, store: AsyncSqliteStore) -> None:
        """max_depth should truncate namespaces to the given depth."""
        await store.abatch(
            [
                PutOp(namespace=("a", "b", "c"), key="k1", value={"v": 1}),
                PutOp(namespace=("a", "b", "d"), key="k2", value={"v": 2}),
                PutOp(namespace=("x", "y"), key="k3", value={"v": 3}),
            ]
        )

        results = await store.abatch(
            [
                ListNamespacesOp(match_conditions=None, max_depth=2, limit=100, offset=0),
            ]
        )

        namespaces = results[0]
        # ("a", "b", "c") and ("a", "b", "d") should both truncate to ("a", "b")
        assert ("a", "b") in namespaces
        assert ("x", "y") in namespaces
        # No full-depth namespaces
        assert ("a", "b", "c") not in namespaces

    @pytest.mark.asyncio
    async def test_list_namespaces_with_offset_limit(self, store: AsyncSqliteStore) -> None:
        """Offset and limit should paginate namespace results."""
        for i in range(5):
            await store.abatch(
                [
                    PutOp(namespace=(f"ns{i}",), key="k", value={"v": i}),
                ]
            )

        results = await store.abatch(
            [
                ListNamespacesOp(match_conditions=None, max_depth=None, limit=2, offset=1),
            ]
        )

        namespaces = results[0]
        assert len(namespaces) == 2


class TestAsyncSqliteStoreConvenience:
    """Test the convenience methods (aget, aput, asearch, etc.)."""

    @pytest.mark.asyncio
    async def test_aget_aput(self, store: AsyncSqliteStore) -> None:
        """Test aget and aput convenience methods."""
        await store.aput(("test",), "key1", {"hello": "world"})
        item = await store.aget(("test",), "key1")
        assert item is not None
        assert item.value == {"hello": "world"}

    @pytest.mark.asyncio
    async def test_adelete(self, store: AsyncSqliteStore) -> None:
        """Test adelete convenience method."""
        await store.aput(("test",), "key1", {"data": True})
        await store.adelete(("test",), "key1")
        item = await store.aget(("test",), "key1")
        assert item is None

    @pytest.mark.asyncio
    async def test_asearch(self, store: AsyncSqliteStore) -> None:
        """Test asearch convenience method."""
        await store.aput(("docs",), "d1", {"title": "Hello"})
        await store.aput(("docs",), "d2", {"title": "World"})

        items = await store.asearch(("docs",), limit=10)
        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_alist_namespaces(self, store: AsyncSqliteStore) -> None:
        """Test alist_namespaces convenience method."""
        await store.aput(("a", "b"), "k1", {"v": 1})
        await store.aput(("c",), "k2", {"v": 2})

        namespaces = await store.alist_namespaces()
        assert ("a", "b") in namespaces
        assert ("c",) in namespaces
