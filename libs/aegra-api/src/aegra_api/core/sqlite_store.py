"""SQLite-backed LangGraph Store with sqlite-vec vector search.

Implements ``BaseStore`` from LangGraph, providing persistent key-value
storage with optional semantic search via sqlite-vec + any LangChain-
compatible ``Embeddings`` provider (e.g. FastEmbed).

Usage::

    store = AsyncSqliteStore(db_path="/tmp/store.db", index={...})
    await store.setup()
    await store.aput(("users",), "u1", {"name": "Alice"})
    item = await store.aget(("users",), "u1")
    await store.close()
"""

from __future__ import annotations

import json
import logging
import struct
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import aiosqlite
from langchain_core.embeddings import Embeddings
from langgraph.store.base import (
    BaseStore,
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    ensure_embeddings,
    get_text_at_path,
    tokenize_path,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Namespace helpers
# ---------------------------------------------------------------------------


def _encode_ns(ns: tuple[str, ...]) -> str:
    """Encode a namespace tuple as a JSON array string."""
    return json.dumps(list(ns))


def _decode_ns(raw: str) -> tuple[str, ...]:
    """Decode a JSON array string back to a namespace tuple."""
    return tuple(json.loads(raw))


def _ns_prefix_like(prefix: tuple[str, ...]) -> str:
    """Build a SQL LIKE pattern matching all namespaces with the given prefix.

    Example: ``("users",)`` -> ``'["users"%'``
    """
    if not prefix:
        return "%"
    # Build the JSON prefix: '["a","b"' (without the closing bracket)
    parts = ",".join(json.dumps(p) for p in prefix)
    return f"[{parts}%"


# ---------------------------------------------------------------------------
# Filter helpers (reuse InMemoryStore semantics)
# ---------------------------------------------------------------------------


def _compare_values(item_value: Any, filter_value: Any) -> bool:
    """Compare values using JSONB-like semantics, supporting nested dicts & operators."""
    if isinstance(filter_value, dict):
        if any(k.startswith("$") for k in filter_value):
            return all(_apply_operator(item_value, op_key, op_value) for op_key, op_value in filter_value.items())
        if not isinstance(item_value, dict):
            return False
        return all(_compare_values(item_value.get(k), v) for k, v in filter_value.items())
    elif isinstance(filter_value, (list, tuple)):
        return (
            isinstance(item_value, (list, tuple))
            and len(item_value) == len(filter_value)
            and all(_compare_values(iv, fv) for iv, fv in zip(item_value, filter_value, strict=False))
        )
    return item_value == filter_value


def _apply_operator(value: Any, operator: str, op_value: Any) -> bool:
    """Apply a comparison operator ($eq, $ne, $gt, $gte, $lt, $lte)."""
    if operator == "$eq":
        return value == op_value
    elif operator == "$ne":
        return value != op_value
    elif operator == "$gt":
        return float(value) > float(op_value)
    elif operator == "$gte":
        return float(value) >= float(op_value)
    elif operator == "$lt":
        return float(value) < float(op_value)
    elif operator == "$lte":
        return float(value) <= float(op_value)
    raise ValueError(f"Unsupported operator: {operator}")


def _match_filter(value: dict[str, Any], filter_dict: dict[str, Any] | None) -> bool:
    """Return True if value matches all filter conditions."""
    if not filter_dict:
        return True
    return all(_compare_values(value.get(k), v) for k, v in filter_dict.items())


def _does_match(match_condition: MatchCondition, key: tuple[str, ...]) -> bool:
    """Whether a namespace key matches a match condition."""
    match_type = match_condition.match_type
    path = match_condition.path

    if len(key) < len(path):
        return False

    if match_type == "prefix":
        for k_elem, p_elem in zip(key, path, strict=False):
            if p_elem == "*":
                continue
            if k_elem != p_elem:
                return False
        return True
    elif match_type == "suffix":
        for k_elem, p_elem in zip(reversed(key), reversed(path), strict=False):
            if p_elem == "*":
                continue
            if k_elem != p_elem:
                return False
        return True
    raise ValueError(f"Unsupported match type: {match_type}")


def _serialize_f32(vec: list[float]) -> bytes:
    """Serialize a float32 vector for sqlite-vec queries."""
    return struct.pack(f"{len(vec)}f", *vec)


# ---------------------------------------------------------------------------
# AsyncSqliteStore
# ---------------------------------------------------------------------------


class AsyncSqliteStore(BaseStore):
    """SQLite-backed store with sqlite-vec semantic search.

    Args:
        db_path: Filesystem path to the SQLite database.
        index: Optional ``IndexConfig`` to enable vector search.
    """

    __slots__ = (
        "_db_path",
        "_conn",
        "index_config",
        "embeddings",
        "_dims",
    )

    def __init__(self, *, db_path: str, index: IndexConfig | None = None) -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None
        self._dims: int | None = None

        if index:
            self.index_config: IndexConfig | None = index.copy()
            self.embeddings: Embeddings | None = ensure_embeddings(
                self.index_config.get("embed"),
            )
            self.index_config["__tokenized_fields"] = [
                (p, tokenize_path(p)) if p != "$" else (p, p) for p in (self.index_config.get("fields") or ["$"])
            ]
            self._dims = self.index_config.get("dims")
        else:
            self.index_config = None
            self.embeddings = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Create tables, load sqlite-vec extension, auto-detect dims."""
        import sqlite_vec  # type: ignore[import-untyped]

        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")

        await self._conn.enable_load_extension(True)
        await self._conn.load_extension(sqlite_vec.loadable_path())
        await self._conn.enable_load_extension(False)

        # Core key-value table
        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS store_items (
                namespace TEXT NOT NULL,
                key       TEXT NOT NULL,
                value     TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (namespace, key)
            )
            """
        )

        # Vector mapping table
        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS store_vectors (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace TEXT NOT NULL,
                key       TEXT NOT NULL,
                path      TEXT NOT NULL,
                UNIQUE (namespace, key, path)
            )
            """
        )

        # Auto-detect dims if not provided
        if self.index_config and self.embeddings and not self._dims:
            sample = await self.embeddings.aembed_query("dimension probe")
            self._dims = len(sample)
            self.index_config["dims"] = self._dims
            logger.info("Auto-detected embedding dims=%d", self._dims)

        # Create sqlite-vec virtual table (requires known dims)
        if self._dims:
            await self._conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_store
                USING vec0(embedding float[{self._dims}])
                """
            )

        await self._conn.commit()
        logger.info("AsyncSqliteStore ready (db=%s, dims=%s)", self._db_path, self._dims)

    async def close(self) -> None:
        """Close the underlying aiosqlite connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    def _ensure_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Store not initialised – call setup() first")
        return self._conn

    # ------------------------------------------------------------------
    # BaseStore abstract methods
    # ------------------------------------------------------------------

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        raise NotImplementedError("AsyncSqliteStore is async-only. Use abatch() instead.")

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        conn = self._ensure_conn()
        results: list[Result] = []
        put_ops: dict[tuple[tuple[str, ...], str], PutOp] = {}
        search_ops: dict[int, SearchOp] = {}

        for idx, op in enumerate(ops):
            if isinstance(op, GetOp):
                item = await self._handle_get(conn, op)
                results.append(item)
            elif isinstance(op, SearchOp):
                search_ops[idx] = op
                results.append(None)  # placeholder
            elif isinstance(op, PutOp):
                put_ops[(op.namespace, op.key)] = op
                results.append(None)
            elif isinstance(op, ListNamespacesOp):
                ns_list = await self._handle_list_namespaces(conn, op)
                results.append(ns_list)
            else:
                raise ValueError(f"Unknown operation type: {type(op)}")

        # Execute search ops (may need embedding queries)
        if search_ops:
            query_embeddings = await self._embed_search_queries(search_ops)
            for idx, op in search_ops.items():
                results[idx] = await self._handle_search(conn, op, query_embeddings)

        # Execute put ops (batch embedding + write)
        if put_ops:
            await self._handle_puts(conn, put_ops)

        await conn.commit()
        return results

    # ------------------------------------------------------------------
    # GetOp handler
    # ------------------------------------------------------------------

    async def _handle_get(self, conn: aiosqlite.Connection, op: GetOp) -> Item | None:
        ns_enc = _encode_ns(op.namespace)
        cursor = await conn.execute(
            "SELECT value, created_at, updated_at FROM store_items WHERE namespace=? AND key=?",
            (ns_enc, op.key),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return Item(
            value=json.loads(row[0]),
            key=op.key,
            namespace=op.namespace,
            created_at=row[1],
            updated_at=row[2],
        )

    # ------------------------------------------------------------------
    # SearchOp handler
    # ------------------------------------------------------------------

    async def _embed_search_queries(self, search_ops: dict[int, SearchOp]) -> dict[str, list[float]]:
        """Embed unique query strings across all pending search ops."""
        if not self.index_config or not self.embeddings:
            return {}
        queries = {op.query for op in search_ops.values() if op.query}
        if not queries:
            return {}
        result: dict[str, list[float]] = {}
        for q in queries:
            result[q] = await self.embeddings.aembed_query(q)
        return result

    async def _handle_search(
        self,
        conn: aiosqlite.Connection,
        op: SearchOp,
        query_embeddings: dict[str, list[float]],
    ) -> list[SearchItem]:
        ns_like = _ns_prefix_like(op.namespace_prefix)

        if op.query and op.query in query_embeddings and self._dims:
            return await self._semantic_search(conn, op, query_embeddings[op.query], ns_like)
        return await self._filter_search(conn, op, ns_like)

    async def _semantic_search(
        self,
        conn: aiosqlite.Connection,
        op: SearchOp,
        query_vec: list[float],
        ns_like: str,
    ) -> list[SearchItem]:
        """Perform KNN search via sqlite-vec, then filter & paginate.

        sqlite-vec KNN uses ``WHERE embedding MATCH ? AND k = ?`` syntax.
        We first get candidate rowids, then join with store_vectors and
        store_items for full data, applying namespace + filter in Python.
        """
        # Fetch extra candidates to account for namespace/filter pruning
        fetch_k = (op.offset + op.limit) * 5

        # Step 1: KNN query against the virtual table
        cursor = await conn.execute(
            "SELECT rowid, distance FROM vec_store WHERE embedding MATCH ? AND k = ?",
            (_serialize_f32(query_vec), fetch_k),
        )
        knn_rows = await cursor.fetchall()

        if not knn_rows:
            return []

        # Step 2: Resolve rowids → (namespace, key) via store_vectors
        rowids = [row[0] for row in knn_rows]
        distance_map: dict[int, float] = {row[0]: float(row[1]) for row in knn_rows}

        placeholders = ",".join("?" * len(rowids))
        cursor = await conn.execute(
            f"SELECT id, namespace, key FROM store_vectors WHERE id IN ({placeholders})",
            rowids,
        )
        vector_rows = await cursor.fetchall()
        vec_lookup: dict[int, tuple[str, str]] = {row[0]: (row[1], row[2]) for row in vector_rows}

        # Step 3: Deduplicate by (namespace, key), keeping best (lowest distance) score
        best_scores: dict[tuple[str, str], float] = {}
        for rowid in rowids:
            pair = vec_lookup.get(rowid)
            if pair is None:
                continue
            distance = distance_map[rowid]
            score = 1.0 - distance
            if pair not in best_scores or score > best_scores[pair]:
                best_scores[pair] = score

        # Step 4: Fetch full items, apply namespace + filter
        results: list[tuple[float, Item]] = []
        for (ns_raw, key), score in best_scores.items():
            # Namespace prefix check
            if ns_like != "%" and not ns_raw.startswith(ns_like.rstrip("%")):
                continue

            item_cursor = await conn.execute(
                "SELECT value, created_at, updated_at FROM store_items WHERE namespace=? AND key=?",
                (ns_raw, key),
            )
            item_row = await item_cursor.fetchone()
            if item_row is None:
                continue
            value = json.loads(item_row[0])
            if not _match_filter(value, op.filter):
                continue
            results.append(
                (
                    score,
                    Item(
                        value=value,
                        key=key,
                        namespace=_decode_ns(ns_raw),
                        created_at=item_row[1],
                        updated_at=item_row[2],
                    ),
                )
            )

        # Sort by score descending, paginate
        results.sort(key=lambda x: x[0], reverse=True)
        page = results[op.offset : op.offset + op.limit]

        return [
            SearchItem(
                namespace=item.namespace,
                key=item.key,
                value=item.value,
                created_at=item.created_at,
                updated_at=item.updated_at,
                score=score,
            )
            for score, item in page
        ]

    async def _filter_search(
        self,
        conn: aiosqlite.Connection,
        op: SearchOp,
        ns_like: str,
    ) -> list[SearchItem]:
        """Non-semantic search: list items matching namespace + filter."""
        cursor = await conn.execute(
            "SELECT namespace, key, value, created_at, updated_at FROM store_items WHERE namespace LIKE ?",
            (ns_like,),
        )
        rows = await cursor.fetchall()

        items: list[SearchItem] = []
        for row in rows:
            value = json.loads(row[2])
            if not _match_filter(value, op.filter):
                continue
            items.append(
                SearchItem(
                    namespace=_decode_ns(row[0]),
                    key=row[1],
                    value=value,
                    created_at=row[3],
                    updated_at=row[4],
                )
            )

        return items[op.offset : op.offset + op.limit]

    # ------------------------------------------------------------------
    # ListNamespacesOp handler
    # ------------------------------------------------------------------

    async def _handle_list_namespaces(self, conn: aiosqlite.Connection, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        cursor = await conn.execute("SELECT DISTINCT namespace FROM store_items")
        rows = await cursor.fetchall()
        namespaces = [_decode_ns(row[0]) for row in rows]

        if op.match_conditions:
            namespaces = [ns for ns in namespaces if all(_does_match(cond, ns) for cond in op.match_conditions)]

        if op.max_depth is not None:
            namespaces = sorted({ns[: op.max_depth] for ns in namespaces})
        else:
            namespaces = sorted(namespaces)

        return namespaces[op.offset : op.offset + op.limit]

    # ------------------------------------------------------------------
    # PutOp handler (batch)
    # ------------------------------------------------------------------

    async def _handle_puts(
        self,
        conn: aiosqlite.Connection,
        put_ops: dict[tuple[tuple[str, ...], str], PutOp],
    ) -> None:
        # Separate deletes from upserts
        to_embed: dict[str, list[tuple[tuple[str, ...], str, str]]] = defaultdict(list)

        for (namespace, key), op in put_ops.items():
            ns_enc = _encode_ns(namespace)
            if op.value is None:
                # Delete
                await conn.execute(
                    "DELETE FROM store_items WHERE namespace=? AND key=?",
                    (ns_enc, key),
                )
                await self._delete_vectors(conn, ns_enc, key)
            else:
                # Upsert
                now = datetime.now(UTC).isoformat()
                await conn.execute(
                    """
                    INSERT INTO store_items (namespace, key, value, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(namespace, key)
                    DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                    """,
                    (ns_enc, key, json.dumps(op.value), now, now),
                )
                # Collect texts to embed
                if self.index_config and self.embeddings and op.index is not False:
                    # Remove old vectors first
                    await self._delete_vectors(conn, ns_enc, key)
                    if op.index is None:
                        paths = self.index_config["__tokenized_fields"]
                    else:
                        paths = [(ix, tokenize_path(ix)) for ix in op.index]
                    for path, field in paths:
                        texts = get_text_at_path(op.value, field)
                        if texts:
                            if len(texts) > 1:
                                for i, text in enumerate(texts):
                                    to_embed[text].append((namespace, key, f"{path}.{i}"))
                            else:
                                to_embed[texts[0]].append((namespace, key, path))

        # Batch embed and insert vectors
        if to_embed and self.embeddings:
            text_list = list(to_embed.keys())
            embeddings = await self.embeddings.aembed_documents(text_list)
            for text, embedding in zip(text_list, embeddings, strict=False):
                for ns, key, path in to_embed[text]:
                    ns_enc = _encode_ns(ns)
                    cursor = await conn.execute(
                        """
                        INSERT INTO store_vectors (namespace, key, path)
                        VALUES (?, ?, ?)
                        ON CONFLICT(namespace, key, path) DO UPDATE SET path=excluded.path
                        """,
                        (ns_enc, key, path),
                    )
                    rowid = cursor.lastrowid
                    await conn.execute(
                        "INSERT INTO vec_store (rowid, embedding) VALUES (?, ?)",
                        (rowid, _serialize_f32(embedding)),
                    )

    async def _delete_vectors(self, conn: aiosqlite.Connection, ns_enc: str, key: str) -> None:
        """Remove all vector entries for a given (namespace, key)."""
        cursor = await conn.execute(
            "SELECT id FROM store_vectors WHERE namespace=? AND key=?",
            (ns_enc, key),
        )
        ids = [row[0] for row in await cursor.fetchall()]
        if ids:
            placeholders = ",".join("?" * len(ids))
            await conn.execute(
                f"DELETE FROM vec_store WHERE rowid IN ({placeholders})",
                ids,
            )
            await conn.execute(
                "DELETE FROM store_vectors WHERE namespace=? AND key=?",
                (ns_enc, key),
            )
