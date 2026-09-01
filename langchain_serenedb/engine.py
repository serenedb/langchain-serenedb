"""Connection management for SereneDB, built on psycopg3.

SereneDB speaks the PostgreSQL wire protocol, so connections are made with psycopg3 and
pooled with ``psycopg_pool.AsyncConnectionPool``. A background event loop lets the
synchronous API delegate to the async implementation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from threading import Thread
from typing import Any, Awaitable, Optional, Sequence, TypedDict, TypeVar, Union, cast

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from .hybrid_search_config import HybridIndexConfig
from .indexes import (
    DEFAULT_INDEX_NAME_SUFFIX,
    BaseIndex,
    IVFIndex,
    MetadataIndexConfig,
    build_dictionary_ddl,
    build_hybrid_index_ddl,
    build_metadata_index_entries,
    build_vector_index_ddl,
)

T = TypeVar("T")


class ColumnDict(TypedDict):
    name: str
    data_type: str
    nullable: bool


@dataclass
class Column:
    name: str
    data_type: str
    nullable: bool = True

    def __post_init__(self) -> None:
        """Check if initialization parameters are valid.

        Raises:
            ValueError: If Column name is not string.
            ValueError: If data_type is not type string.
        """
        if not isinstance(self.name, str):
            raise ValueError("Column name must be type string")
        if not isinstance(self.data_type, str):
            raise ValueError("Column data_type must be type string")


class SereneDBEngine:
    """A class for managing connections to a SereneDB database."""

    _default_loop: Optional[asyncio.AbstractEventLoop] = None
    _default_thread: Optional[Thread] = None
    __create_key = object()

    def __init__(
        self,
        key: object,
        pool: AsyncConnectionPool,
        loop: Optional[asyncio.AbstractEventLoop],
        thread: Optional[Thread],
    ) -> None:
        """SereneDBEngine constructor.

        Args:
            key (object): Prevent direct constructor usage.
            pool (AsyncConnectionPool): Async psycopg connection pool.
            loop (Optional[asyncio.AbstractEventLoop]): Async event loop used to create the engine.
            thread (Optional[Thread]): Thread used to create the engine async.

        Raises:
            Exception: If the constructor is called directly by the user.
        """
        if key != SereneDBEngine.__create_key:
            raise Exception(
                "Only create class through 'from_connection_string' or 'from_pool' methods!"
            )
        self._pool = pool
        self._loop = loop
        self._thread = thread
        self._opened = False

    @classmethod
    def from_pool(
        cls: type[SereneDBEngine],
        pool: AsyncConnectionPool,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> SereneDBEngine:
        """Create a SereneDBEngine instance from an existing AsyncConnectionPool."""
        engine = cls(cls.__create_key, pool, loop, None)
        engine._opened = True
        return engine

    @classmethod
    def from_connection_string(
        cls,
        conninfo: str,
        **kwargs: Any,
    ) -> SereneDBEngine:
        """Create a SereneDBEngine from a psycopg connection string.

        Args:
            conninfo (str): libpq connection string, e.g.
                ``"host=127.0.0.1 port=7890 user=postgres dbname=postgres"``.
            **kwargs (Any): Extra keyword arguments forwarded to
                ``psycopg_pool.AsyncConnectionPool`` (e.g. ``min_size``, ``max_size``).

        Returns:
            SereneDBEngine
        """
        # Running a loop in a background thread allows us to support
        # async methods from non-async environments.
        if cls._default_loop is None:
            cls._default_loop = asyncio.new_event_loop()
            cls._default_thread = Thread(
                target=cls._default_loop.run_forever, daemon=True
            )
            cls._default_thread.start()

        connection_kwargs = kwargs.pop("kwargs", {})
        connection_kwargs.setdefault("row_factory", dict_row)
        # The pool is created but not opened here; it is opened lazily on the
        # background loop before its first use (AsyncConnectionPool must be opened
        # inside the running event loop).
        pool = AsyncConnectionPool(
            conninfo, open=False, kwargs=connection_kwargs, **kwargs
        )
        return cls(cls.__create_key, pool, cls._default_loop, cls._default_thread)

    async def _run_as_async(self, coro: Awaitable[T]) -> T:
        """Run an async coroutine asynchronously."""
        # If a loop has not been provided, attempt to run in current thread.
        if not self._loop:
            return await coro
        # Otherwise, run in the background thread.
        return await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(coro, self._loop)  # type: ignore[arg-type]
        )

    def _run_as_sync(self, coro: Awaitable[T]) -> T:
        """Run an async coroutine synchronously."""
        if not self._loop:
            raise Exception(
                "Engine was initialized without a background loop and cannot call sync methods."
            )
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()  # type: ignore[arg-type]

    async def _ensure_open(self) -> None:
        """Open the connection pool lazily, on the loop that will drive it."""
        if not self._opened:
            await self._pool.open()
            self._opened = True

    async def aclose(self) -> None:
        """Dispose of the connection pool."""
        await self._run_as_async(self._pool.close())

    def close(self) -> None:
        """Dispose of the connection pool (sync).

        Sync counterpart of :meth:`aclose`, so sync callers can just write
        ``engine.close()`` instead of ``engine._run_as_sync(engine.close())``.
        """
        self._run_as_sync(self._pool.close())

    # -- low-level execution helpers -------------------------------------------------

    async def _aexecute(
        self, query: str, params: Optional[Union[dict, Sequence]] = None
    ) -> None:
        """Execute a write/DDL statement and commit."""
        await self._ensure_open()
        async with self._pool.connection() as conn:
            await conn.execute(query, params)
            await conn.commit()

    async def _aexecute_many(
        self, query: str, params_seq: Sequence[Union[dict, Sequence]]
    ) -> None:
        """Execute one statement for many parameter sets on a single connection.

        Uses psycopg's ``executemany`` (pipelined) with a single commit, so a bulk
        write is one checkout + one transaction instead of one per row.
        """
        if not params_seq:
            return
        await self._ensure_open()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(query, params_seq)
            await conn.commit()

    async def _afetch(
        self, query: str, params: Optional[Union[dict, Sequence]] = None
    ) -> list[dict[str, Any]]:
        """Execute a query and return all rows as dicts."""
        await self._ensure_open()
        async with self._pool.connection() as conn:
            cur = await conn.execute(query, params)
            rows = await cur.fetchall()
            await conn.commit()
        # The pool sets row_factory=dict_row, so rows are dicts; psycopg's stubs cannot
        # see the dynamically-set factory and still type them as tuples.
        return cast(list[dict[str, Any]], rows)

    async def _aexecute_autocommit(self, query: str) -> None:
        """Execute a statement that cannot run inside a transaction (e.g. VACUUM)."""
        await self._ensure_open()
        async with self._pool.connection() as conn:
            await conn.set_autocommit(True)
            await conn.execute(query)

    @staticmethod
    def _escape_identifier(name: str) -> str:
        return name.replace('"', '""')

    def _validate_column_dict(self, col: ColumnDict) -> None:
        if not isinstance(col.get("name"), str):
            raise TypeError("The 'name' field must be a string.")
        if not isinstance(col.get("data_type"), str):
            raise TypeError("The 'data_type' field must be a string.")
        if not isinstance(col.get("nullable"), bool):
            raise TypeError("The 'nullable' field must be a boolean.")

    # -- table lifecycle -------------------------------------------------------------

    async def _ainit_vectorstore_table(
        self,
        table_name: str,
        vector_size: int,
        *,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: Optional[list[Union[Column, ColumnDict]]] = None,
        metadata_json_column: str = "langchain_metadata",
        id_column: Union[str, Column, ColumnDict] = "langchain_id",
        overwrite_existing: bool = False,
        if_not_exists: bool = False,
        store_metadata: bool = True,
        vector_index: Optional[BaseIndex] = None,
        hybrid_index_config: Optional[HybridIndexConfig] = None,
        metadata_index: Optional[MetadataIndexConfig] = None,
    ) -> None:
        """Create a table for storing vectors, for use with SereneDBVectorStore.

        The embedding column is a fixed-size ``FLOAT[vector_size]`` array. Metadata is
        stored in a ``JSON`` column. Pass ``hybrid_index_config`` to build the combined
        full-text + vector index (the ``content`` column is indexed directly, so no
        separate column is added here); see
        :class:`~langchain_serenedb.hybrid_search_config.HybridIndexConfig`.

        ``if_not_exists=True`` makes the call idempotent: the table (and its index) are
        created only when absent, so re-running does not error and keeps existing data.
        It does NOT reconcile an existing table's shape — if a table with this name
        already exists, its columns are left as-is (use :meth:`create` to validate them).
        Mutually exclusive with ``overwrite_existing`` (which drops and recreates).

        When an index is built, ``metadata_index`` controls which metadata columns / JSON
        sub-fields are added to it so metadata filters are served by the index scan. The
        default (``None``) indexes all declared ``metadata_columns`` verbatim and no JSON;
        pass a :class:`~langchain_serenedb.indexes.MetadataIndexConfig` to override.
        """
        if overwrite_existing and if_not_exists:
            raise ValueError(
                "Use only one of overwrite_existing (drop + recreate) or "
                "if_not_exists (create only when absent)."
            )
        # Keep raw names for the index-DDL builders (which escape internally).
        raw_schema_name = schema_name
        raw_table_name = table_name
        raw_content_column = content_column
        raw_embedding_column = embedding_column

        schema_name = self._escape_identifier(schema_name)
        table_name = self._escape_identifier(table_name)
        content_column = self._escape_identifier(content_column)
        embedding_column = self._escape_identifier(embedding_column)

        if metadata_columns is None:
            metadata_columns = []
        # Capture raw (un-escaped) metadata column names mapped to their declared types
        # for the index-DDL builder (which escapes internally, and uses the types to skip
        # unindexable columns in the default index-all case), then escape the names in
        # place for the CREATE TABLE.
        raw_metadata_columns: dict[str, str] = {}
        for col in metadata_columns:
            if isinstance(col, Column):
                raw_metadata_columns[col.name] = col.data_type
                col.name = self._escape_identifier(col.name)
            elif isinstance(col, dict):
                self._validate_column_dict(col)
                raw_metadata_columns[col["name"]] = col["data_type"]
                col["name"] = self._escape_identifier(col["name"])

        if isinstance(id_column, str):
            raw_id_column_name = id_column
            id_column = self._escape_identifier(id_column)
            id_data_type = "UUID"
            id_column_name = id_column
        elif isinstance(id_column, Column):
            raw_id_column_name = id_column.name
            id_column.name = self._escape_identifier(id_column.name)
            id_data_type = id_column.data_type
            id_column_name = id_column.name
        else:
            self._validate_column_dict(id_column)
            raw_id_column_name = id_column["name"]
            id_column["name"] = self._escape_identifier(id_column["name"])
            id_data_type = id_column["data_type"]
            id_column_name = id_column["name"]

        if overwrite_existing:
            await self._aexecute(f'DROP TABLE IF EXISTS "{schema_name}"."{table_name}"')

        ine = "IF NOT EXISTS " if if_not_exists else ""
        query = (
            f'CREATE TABLE {ine}"{schema_name}"."{table_name}"(\n'
            f'"{id_column_name}" {id_data_type} PRIMARY KEY,\n'
            f'"{content_column}" TEXT NOT NULL,\n'
            f'"{embedding_column}" FLOAT[{int(vector_size)}] NOT NULL'
        )
        for column in metadata_columns:
            if isinstance(column, Column):
                nullable = "" if column.nullable else "NOT NULL"
                query += f',\n"{column.name}" {column.data_type} {nullable}'
            elif isinstance(column, dict):
                nullable = "" if column["nullable"] else "NOT NULL"
                query += f',\n"{column["name"]}" {column["data_type"]} {nullable}'
        if store_metadata:
            query += f',\n"{metadata_json_column}" JSON'
        query += "\n);"

        await self._aexecute(query)

        # Optionally create the ANN index alongside the table so vector search is
        # accelerated from the start. (The docs recommend building the index *after* a
        # bulk load for a more compact graph; creating it up front is convenient for
        # incremental workloads.)
        want_index = hybrid_index_config is not None or vector_index is not None
        if want_index:
            index: BaseIndex = vector_index if vector_index is not None else IVFIndex()
            vector_opclass = index.index_options()
            metadata_entries = build_metadata_index_entries(
                metadata_index,
                metadata_json_column=metadata_json_column if store_metadata else None,
                metadata_columns=raw_metadata_columns,
            )
            if hybrid_index_config is not None:
                await self._aexecute(
                    build_dictionary_ddl(
                        raw_schema_name,
                        hybrid_index_config.dictionary_name,
                        hybrid_index_config.dictionary_options,
                    )
                )
                await self._aexecute(
                    build_hybrid_index_ddl(
                        schema_name=raw_schema_name,
                        table_name=raw_table_name,
                        content_column=raw_content_column,
                        embedding_column=raw_embedding_column,
                        id_column=raw_id_column_name,
                        index_name=raw_table_name + DEFAULT_INDEX_NAME_SUFFIX,
                        dictionary_name=hybrid_index_config.dictionary_name,
                        vector_opclass=vector_opclass,
                        metadata_entries=metadata_entries,
                        if_not_exists=if_not_exists,
                    )
                )
            else:
                await self._aexecute(
                    build_vector_index_ddl(
                        schema_name=raw_schema_name,
                        table_name=raw_table_name,
                        embedding_column=raw_embedding_column,
                        index_name=raw_table_name + DEFAULT_INDEX_NAME_SUFFIX,
                        vector_opclass=vector_opclass,
                        metadata_entries=metadata_entries,
                        if_not_exists=if_not_exists,
                    )
                )

    async def ainit_vectorstore_table(
        self,
        table_name: str,
        vector_size: int,
        *,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: Optional[list[Union[Column, ColumnDict]]] = None,
        metadata_json_column: str = "langchain_metadata",
        id_column: Union[str, Column, ColumnDict] = "langchain_id",
        overwrite_existing: bool = False,
        if_not_exists: bool = False,
        store_metadata: bool = True,
        vector_index: Optional[BaseIndex] = None,
        hybrid_index_config: Optional[HybridIndexConfig] = None,
        metadata_index: Optional[MetadataIndexConfig] = None,
    ) -> None:
        """Async: create a table for storing vectors.

        Pass ``vector_index`` (e.g. ``IVFIndex(distance_strategy=...)``) to also build
        the ANN index on the embedding column, or ``hybrid_index_config`` to build the
        combined full-text + vector index — both in the same call as table creation.
        ``if_not_exists=True`` makes the call idempotent (create only when absent; keeps
        existing data); it is mutually exclusive with ``overwrite_existing``.
        ``metadata_index`` (a :class:`~langchain_serenedb.indexes.MetadataIndexConfig`)
        selects which metadata columns / JSON sub-fields join the index for filter
        pushdown; the default indexes all declared ``metadata_columns`` verbatim.
        """
        await self._run_as_async(
            self._ainit_vectorstore_table(
                table_name,
                vector_size,
                schema_name=schema_name,
                content_column=content_column,
                embedding_column=embedding_column,
                metadata_columns=metadata_columns,
                metadata_json_column=metadata_json_column,
                id_column=id_column,
                overwrite_existing=overwrite_existing,
                if_not_exists=if_not_exists,
                store_metadata=store_metadata,
                vector_index=vector_index,
                hybrid_index_config=hybrid_index_config,
                metadata_index=metadata_index,
            )
        )

    def init_vectorstore_table(
        self,
        table_name: str,
        vector_size: int,
        *,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: Optional[list[Union[Column, ColumnDict]]] = None,
        metadata_json_column: str = "langchain_metadata",
        id_column: Union[str, Column, ColumnDict] = "langchain_id",
        overwrite_existing: bool = False,
        if_not_exists: bool = False,
        store_metadata: bool = True,
        vector_index: Optional[BaseIndex] = None,
        hybrid_index_config: Optional[HybridIndexConfig] = None,
        metadata_index: Optional[MetadataIndexConfig] = None,
    ) -> None:
        """Sync: create a table for storing vectors.

        Pass ``vector_index`` (e.g. ``IVFIndex(distance_strategy=...)``) to also build
        the ANN index on the embedding column, or ``hybrid_index_config`` to build the
        combined full-text + vector index — both in the same call as table creation.
        ``if_not_exists=True`` makes the call idempotent (create only when absent; keeps
        existing data); it is mutually exclusive with ``overwrite_existing``.
        ``metadata_index`` (a :class:`~langchain_serenedb.indexes.MetadataIndexConfig`)
        selects which metadata columns / JSON sub-fields join the index for filter
        pushdown; the default indexes all declared ``metadata_columns`` verbatim.
        """
        self._run_as_sync(
            self._ainit_vectorstore_table(
                table_name,
                vector_size,
                schema_name=schema_name,
                content_column=content_column,
                embedding_column=embedding_column,
                metadata_columns=metadata_columns,
                metadata_json_column=metadata_json_column,
                id_column=id_column,
                overwrite_existing=overwrite_existing,
                if_not_exists=if_not_exists,
                store_metadata=store_metadata,
                vector_index=vector_index,
                hybrid_index_config=hybrid_index_config,
                metadata_index=metadata_index,
            )
        )

    async def _adrop_table(
        self, table_name: str, *, schema_name: str = "public"
    ) -> None:
        schema_name = self._escape_identifier(schema_name)
        table_name = self._escape_identifier(table_name)
        await self._aexecute(f'DROP TABLE IF EXISTS "{schema_name}"."{table_name}";')

    async def adrop_table(
        self, table_name: str, *, schema_name: str = "public"
    ) -> None:
        await self._run_as_async(
            self._adrop_table(table_name=table_name, schema_name=schema_name)
        )

    def drop_table(self, table_name: str, *, schema_name: str = "public") -> None:
        self._run_as_sync(
            self._adrop_table(table_name=table_name, schema_name=schema_name)
        )

    # -- inverted-index refresh (eventual consistency) -------------------------------

    async def _arefresh_table(
        self, table_name: str, *, schema_name: str = "public"
    ) -> None:
        """Publish buffered writes to the inverted index so new rows are searchable.

        SereneDB's inverted index is eventually consistent: rows written since the last
        refresh are invisible to full-text and vector-index-routed queries until
        refreshed. Call
        this after ``add_texts``/``delete`` when the table has an inverted index.
        """
        schema_name = self._escape_identifier(schema_name)
        table_name = self._escape_identifier(table_name)
        await self._aexecute_autocommit(
            f'VACUUM (REFRESH_TABLE) "{schema_name}"."{table_name}";'
        )

    async def arefresh_table(
        self, table_name: str, *, schema_name: str = "public"
    ) -> None:
        await self._run_as_async(
            self._arefresh_table(table_name=table_name, schema_name=schema_name)
        )

    def refresh_table(self, table_name: str, *, schema_name: str = "public") -> None:
        self._run_as_sync(
            self._arefresh_table(table_name=table_name, schema_name=schema_name)
        )
