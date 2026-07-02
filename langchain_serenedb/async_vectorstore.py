"""Async vector store backed by SereneDB.

``AsyncSereneDBVectorStore`` is the async core; ``SereneDBVectorStore`` wraps it for
synchronous callers.

SereneDB specifics honored here:
- Embedding literal: bind a Python list and cast ``%(param)s::FLOAT[N]`` (``N`` is the
  query/embedding dimension).
- ``$in`` / ``$nin``: expand to ``IN (...)`` / ``NOT IN (...)`` — ``= ANY(:arr)`` /
  ``<> ALL(:arr)`` are unsupported ("UNNEST not supported here").
- JSON filters: extractions are parenthesized — ``(md ->> 'k')``, ``(md ->> 'k')::INT``
  — because ``->`` is a low-precedence operator in SereneDB.
- Eventual consistency: after writes, ``VACUUM (REFRESH_TABLE)`` publishes rows to the
  inverted index; ``aadd_embeddings`` / ``adelete`` refresh automatically.
- Vector index: ``CREATE INDEX ... USING inverted (emb hnsw (metric='cosine', ...))``.
"""

from __future__ import annotations

import datetime
import json
import uuid
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores import utils as lc_utils

from .engine import SereneDBEngine
from .hybrid_search_config import HybridSearchConfig
from .indexes import (
    DEFAULT_DISTANCE_STRATEGY,
    DEFAULT_INDEX_NAME_SUFFIX,
    BaseIndex,
    DistanceStrategy,
    ExactNearestNeighbor,
    QueryOptions,
    build_dictionary_ddl,
    build_hybrid_index_ddl,
    build_vector_index_ddl,
)

COMPARISONS_TO_NATIVE = {
    "$eq": "=",
    "$ne": "!=",
    "$lt": "<",
    "$lte": "<=",
    "$gt": ">",
    "$gte": ">=",
}
SPECIAL_CASED_OPERATORS = {"$in", "$nin", "$between", "$exists"}
TEXT_OPERATORS = {"$like", "$ilike"}
LOGICAL_OPERATORS = {"$and", "$or", "$not"}
SUPPORTED_OPERATORS = (
    set(COMPARISONS_TO_NATIVE)
    | TEXT_OPERATORS
    | SPECIAL_CASED_OPERATORS
    | set(LOGICAL_OPERATORS)
)

# SereneDB scalar types used when casting a JSON-extracted value for comparison.
PYTHON_TO_SDB_TYPE_MAP = {
    int: "INTEGER",
    float: "FLOAT",
    str: "TEXT",
    bool: "BOOLEAN",
    datetime.date: "DATE",
    datetime.datetime: "TIMESTAMP",
    datetime.time: "TIME",
}

class AsyncSereneDBVectorStore(VectorStore):
    """Async vector store for SereneDB. Construct via :meth:`create`."""

    __create_key = object()

    def __init__(
        self,
        key: object,
        engine: SereneDBEngine,
        embedding_service: Embeddings,
        table_name: str,
        *,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: Optional[list[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: Optional[str] = "langchain_metadata",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        index_query_options: Optional[QueryOptions] = None,
        hybrid_search_config: Optional[HybridSearchConfig] = None,
    ) -> None:
        if key != AsyncSereneDBVectorStore.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )
        self.engine = engine
        self.embedding_service = embedding_service
        self.table_name = table_name
        self.schema_name = schema_name
        self.content_column = content_column
        self.embedding_column = embedding_column
        self.metadata_columns = metadata_columns if metadata_columns is not None else []
        self.id_column = id_column
        self.metadata_json_column = metadata_json_column
        self.distance_strategy = distance_strategy
        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.index_query_options = index_query_options
        self.hybrid_search_config = hybrid_search_config

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_service

    @classmethod
    async def create(
        cls,
        engine: SereneDBEngine,
        embedding_service: Embeddings,
        table_name: str,
        *,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: Optional[list[str]] = None,
        ignore_metadata_columns: Optional[list[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: Optional[str] = "langchain_metadata",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        index_query_options: Optional[QueryOptions] = None,
        hybrid_search_config: Optional[HybridSearchConfig] = None,
    ) -> AsyncSereneDBVectorStore:
        """Create an ``AsyncSereneDBVectorStore`` bound to an existing table.

        Validates the requested columns against ``information_schema.columns``. Note
        that SereneDB reports a ``FLOAT[N]`` embedding column with ``data_type =
        'ARRAY'``.
        """
        if metadata_columns is None:
            metadata_columns = []
        if metadata_columns and ignore_metadata_columns:
            raise ValueError(
                "Can not use both metadata_columns and ignore_metadata_columns."
            )

        stmt = (
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = %(table_name)s AND table_schema = %(schema_name)s"
        )
        results = await engine._afetch(
            stmt, {"table_name": table_name, "schema_name": schema_name}
        )
        columns = {row["column_name"]: row["data_type"] for row in results}

        if not columns:
            raise ValueError(
                f'Table "{schema_name}"."{table_name}" does not exist or has no columns.'
            )
        if id_column not in columns:
            raise ValueError(f"Id column, {id_column}, does not exist.")
        if content_column not in columns:
            raise ValueError(f"Content column, {content_column}, does not exist.")
        content_type = columns[content_column]
        if content_type != "text" and "char" not in content_type:
            raise ValueError(
                f"Content column, {content_column}, is type, {content_type}. "
                "It must be a type of character string."
            )
        if embedding_column not in columns:
            raise ValueError(f"Embedding column, {embedding_column}, does not exist.")
        if columns[embedding_column] not in ("ARRAY", "USER-DEFINED", "vector"):
            raise ValueError(
                f"Embedding column, {embedding_column}, is type "
                f"{columns[embedding_column]}. It must be a FLOAT[N] array."
            )

        metadata_json_column = (
            None if metadata_json_column not in columns else metadata_json_column
        )

        for column in metadata_columns:
            if column not in columns:
                raise ValueError(f"Metadata column, {column}, does not exist.")

        all_columns = columns
        if ignore_metadata_columns:
            for column in ignore_metadata_columns:
                del all_columns[column]
            del all_columns[id_column]
            del all_columns[content_column]
            del all_columns[embedding_column]
            metadata_columns = list(all_columns.keys())

        return cls(
            cls.__create_key,
            engine,
            embedding_service,
            table_name,
            schema_name=schema_name,
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_columns=metadata_columns,
            id_column=id_column,
            metadata_json_column=metadata_json_column,
            distance_strategy=distance_strategy,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            index_query_options=index_query_options,
            hybrid_search_config=hybrid_search_config,
        )

    # -- low-level query helper ------------------------------------------------------

    async def _aquery(
        self,
        query: str,
        params: Optional[dict] = None,
        prelude: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Run ``prelude`` statements (e.g. ``SET LOCAL``) then ``query`` on one conn."""
        await self.engine._ensure_open()
        async with self.engine._pool.connection() as conn:
            if prelude:
                for stmt in prelude:
                    await conn.execute(stmt)
            cur = await conn.execute(query, params)
            rows = await cur.fetchall()
            await conn.commit()
        return rows

    # -- writes ----------------------------------------------------------------------

    async def aadd_embeddings(
        self,
        texts: Iterable[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add data along with embeddings to the table (upsert), then refresh."""
        texts = list(texts)
        if not ids:
            ids = [str(uuid.uuid4()) for _ in texts]
        else:
            ids = [id if id is not None else str(uuid.uuid4()) for id in ids]
        if not metadatas:
            metadatas = [{} for _ in texts]

        for id, content, embedding, metadata in zip(ids, texts, embeddings, metadatas):
            dim = len(embedding)
            metadata_col_names = (
                ", " + ", ".join(f'"{col}"' for col in self.metadata_columns)
                if self.metadata_columns
                else ""
            )
            insert_stmt = (
                f'INSERT INTO "{self.schema_name}"."{self.table_name}"('
                f'"{self.id_column}", "{self.content_column}", "{self.embedding_column}"'
                f"{metadata_col_names}"
            )
            values: dict[str, Any] = {
                "langchain_id": id,
                "content": content,
                "embedding": [float(d) for d in embedding],
            }
            values_stmt = (
                f"VALUES (%(langchain_id)s, %(content)s, %(embedding)s::FLOAT[{dim}]"
            )

            extra = dict(metadata)
            for metadata_column in self.metadata_columns:
                if metadata_column in metadata:
                    values_stmt += f", %({metadata_column})s"
                    values[metadata_column] = (
                        json.dumps(metadata[metadata_column])
                        if isinstance(metadata[metadata_column], dict)
                        else metadata[metadata_column]
                    )
                    del extra[metadata_column]
                else:
                    values_stmt += ", null"

            if self.metadata_json_column:
                insert_stmt += f', "{self.metadata_json_column}")'
                values_stmt += ", %(extra)s)"
                values["extra"] = json.dumps(extra)
            else:
                insert_stmt += ")"
                values_stmt += ")"

            upsert_stmt = (
                f' ON CONFLICT ("{self.id_column}") DO UPDATE SET '
                f'"{self.content_column}" = EXCLUDED."{self.content_column}", '
                f'"{self.embedding_column}" = EXCLUDED."{self.embedding_column}"'
            )
            if self.metadata_json_column:
                upsert_stmt += (
                    f', "{self.metadata_json_column}" = '
                    f'EXCLUDED."{self.metadata_json_column}"'
                )
            for column in self.metadata_columns:
                upsert_stmt += f', "{column}" = EXCLUDED."{column}"'
            upsert_stmt += ";"

            await self.engine._aexecute(insert_stmt + values_stmt + upsert_stmt, values)

        # Publish the writes to the inverted index (eventual consistency).
        await self.engine._arefresh_table(
            self.table_name, schema_name=self.schema_name
        )
        return ids

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed texts and add them to the table."""
        texts = list(texts)
        embeddings = await self.embedding_service.aembed_documents(texts)
        return await self.aadd_embeddings(
            texts, embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

    async def aadd_documents(
        self, documents: list[Document], ids: Optional[list] = None, **kwargs: Any
    ) -> list[str]:
        """Embed documents and add them to the table."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        if not ids:
            ids = [doc.id for doc in documents]
        return await self.aadd_texts(texts, metadatas=metadatas, ids=ids, **kwargs)

    async def adelete(
        self,
        ids: Optional[list] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete records by ids and/or metadata filter, then refresh."""
        if not ids and not filter:
            return False

        where_clauses = []
        param_dict: dict[str, Any] = {}
        if ids:
            placeholders = ", ".join(f"%(id_{i})s" for i in range(len(ids)))
            param_dict.update({f"id_{i}": id for i, id in enumerate(ids)})
            where_clauses.append(f'"{self.id_column}" IN ({placeholders})')
        if filter:
            filter_clause, filter_params = self._create_filter_clause(filter)
            param_dict.update(filter_params)
            where_clauses.append(filter_clause)

        where_clause = " AND ".join(where_clauses)
        query = (
            f'DELETE FROM "{self.schema_name}"."{self.table_name}" WHERE {where_clause}'
        )
        await self.engine._aexecute(query, param_dict)
        await self.engine._arefresh_table(
            self.table_name, schema_name=self.schema_name
        )
        return True

    # -- search ----------------------------------------------------------------------

    def _select_columns(self, *, include_embedding: bool = True) -> str:
        columns = [self.id_column, self.content_column]
        if include_embedding:
            columns.append(self.embedding_column)
        columns += self.metadata_columns
        if self.metadata_json_column:
            columns.append(self.metadata_json_column)
        return ", ".join(f'"{col}"' for col in columns)

    def _rows_to_documents(
        self, rows: list[dict[str, Any]], *, with_score: bool = False
    ) -> list:
        out = []
        for row in rows:
            raw_meta = (
                row.get(self.metadata_json_column) if self.metadata_json_column else None
            )
            if isinstance(raw_meta, str):
                metadata = json.loads(raw_meta) if raw_meta else {}
            elif isinstance(raw_meta, dict):
                metadata = dict(raw_meta)
            else:
                metadata = {}
            for col in self.metadata_columns:
                metadata[col] = row[col]
            doc = Document(
                page_content=row[self.content_column],
                metadata=metadata,
                id=str(row[self.id_column]),
            )
            out.append((doc, row["distance"]) if with_score else doc)
        return out

    async def _adense_query(
        self,
        embedding: list[float],
        *,
        limit: int,
        filter: Optional[dict] = None,
    ) -> list[dict[str, Any]]:
        """Run a dense (ANN) similarity query and return raw rows with a distance."""
        dim = len(embedding)
        operator = self.distance_strategy.operator
        search_function = self.distance_strategy.search_function
        column_names = self._select_columns()

        safe_filter, filter_dict = (None, None)
        if filter and isinstance(filter, dict):
            safe_filter, filter_dict = self._create_filter_clause(filter)
        where_filters = f"WHERE {safe_filter}" if safe_filter else ""

        query = (
            f"SELECT {column_names}, "
            f'{search_function}("{self.embedding_column}", %(query_embedding)s::FLOAT[{dim}]) AS distance '
            f'FROM "{self.schema_name}"."{self.table_name}" {where_filters} '
            f'ORDER BY "{self.embedding_column}" {operator} %(query_embedding)s::FLOAT[{dim}] '
            f"LIMIT %(dense_limit)s;"
        )
        params: dict[str, Any] = {
            "query_embedding": [float(d) for d in embedding],
            "dense_limit": limit,
        }
        if filter_dict:
            params.update(filter_dict)

        prelude = None
        if self.index_query_options:
            prelude = [f"SET LOCAL {opt};" for opt in self.index_query_options.to_parameter()]
        return await self._aquery(query, params, prelude=prelude)

    async def _asparse_query(
        self,
        hybrid_search_config: HybridSearchConfig,
        fts_query: str,
        *,
        limit: int,
        filter: Optional[dict] = None,
    ) -> list[dict[str, Any]]:
        """Run the lexical (BM25) branch against the inverted index by name."""
        column_names = self._select_columns(include_embedding=False)
        index_name = self.engine._escape_identifier(hybrid_search_config.index_name)
        scorer = hybrid_search_config.scorer
        tsquery_fn = hybrid_search_config.tsquery_function

        safe_filter, filter_dict = (None, None)
        if filter and isinstance(filter, dict):
            safe_filter, filter_dict = self._create_filter_clause(filter)
        and_filters = f"AND ({safe_filter})" if safe_filter else ""

        query = (
            f'SELECT {column_names}, {scorer}("{index_name}".tableoid) AS distance '
            f'FROM "{index_name}" '
            f'WHERE "{self.content_column}" @@ {tsquery_fn}(%(fts_query)s) {and_filters} '
            f'ORDER BY distance DESC, "{self.id_column}" '
            f"LIMIT %(sparse_limit)s;"
        )
        params: dict[str, Any] = {"fts_query": fts_query, "sparse_limit": limit}
        if filter_dict:
            params.update(filter_dict)
        return await self._aquery(query, params)

    async def _aquery_collection(
        self,
        embedding: list[float],
        *,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Dense search, or hybrid (dense + BM25 fused) when configured."""
        hybrid_search_config = kwargs.get(
            "hybrid_search_config", self.hybrid_search_config
        )
        final_k = k if k is not None else self.k
        dense_limit = (
            hybrid_search_config.primary_top_k if hybrid_search_config else final_k
        )
        dense_rows = await self._adense_query(embedding, limit=dense_limit, filter=filter)

        fts_query = ""
        if hybrid_search_config:
            fts_query = hybrid_search_config.fts_query or kwargs.get("fts_query", "")
        if hybrid_search_config and fts_query:
            sparse_rows = await self._asparse_query(
                hybrid_search_config,
                fts_query,
                limit=hybrid_search_config.secondary_top_k,
                filter=filter,
            )
            fusion_params = dict(hybrid_search_config.fusion_function_parameters)
            fusion_params["fetch_top_k"] = final_k
            combined = hybrid_search_config.fusion_function(
                dense_rows,
                sparse_rows,
                **fusion_params,
                distance_strategy=self.distance_strategy,
            )
            return list(combined)
        return dense_rows

    async def asimilarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        embedding = await self.embedding_service.aembed_query(text=query)
        kwargs.setdefault("fts_query", query)
        return await self.asimilarity_search_by_vector(
            embedding=embedding, k=k, filter=filter, **kwargs
        )

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        embedding = await self.embedding_service.aembed_query(text=query)
        kwargs.setdefault("fts_query", query)
        return await self.asimilarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter, **kwargs
        )

    async def asimilarity_search_by_vector(
        self,
        embedding: list[float],
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        rows = await self._aquery_collection(embedding, k=k, filter=filter, **kwargs)
        return [doc for doc, _ in self._rows_to_documents(rows, with_score=True)]

    async def asimilarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        rows = await self._aquery_collection(embedding, k=k, filter=filter, **kwargs)
        return self._rows_to_documents(rows, with_score=True)

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        embedding = await self.embedding_service.aembed_query(text=query)
        return await self.amax_marginal_relevance_search_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        docs_and_scores = await self.amax_marginal_relevance_search_with_score_by_vector(
            embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    async def amax_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        fetch_k = fetch_k if fetch_k is not None else self.fetch_k
        rows = await self._adense_query(embedding, limit=fetch_k, filter=filter)

        k = k if k is not None else self.k
        lambda_mult = lambda_mult if lambda_mult is not None else self.lambda_mult
        embedding_list = [row[self.embedding_column] for row in rows]
        if not embedding_list:
            return []
        mmr_selected = lc_utils.maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            embedding_list,
            k=k,
            lambda_mult=lambda_mult,
        )
        docs_and_scores = self._rows_to_documents(rows, with_score=True)
        return [r for i, r in enumerate(docs_and_scores) if i in mmr_selected]

    async def aget_by_ids(self, ids: Sequence[str]) -> list[Document]:
        """Fetch documents by id (order is not guaranteed)."""
        if not ids:
            return []
        column_names = self._select_columns()
        placeholders = ", ".join(f"%(id_{i})s" for i in range(len(ids)))
        params = {f"id_{i}": id for i, id in enumerate(ids)}
        query = (
            f"SELECT {column_names} "
            f'FROM "{self.schema_name}"."{self.table_name}" '
            f'WHERE "{self.id_column}" IN ({placeholders});'
        )
        rows = await self._aquery(query, params)
        # No distance column here; add a placeholder so the shared builder works.
        for row in rows:
            row.setdefault("distance", 0.0)
        docs = self._rows_to_documents(rows)
        # Return in the same order as the requested ids (VectorStore contract).
        by_id = {doc.id: doc for doc in docs}
        return [by_id[str(i)] for i in ids if str(i) in by_id]

    # -- index management ------------------------------------------------------------

    async def _acreate_text_search_dictionary(
        self, config: HybridSearchConfig
    ) -> None:
        """Create the text search dictionary used to analyze the content column."""
        await self.engine._aexecute(
            build_dictionary_ddl(config.dictionary_name, config.dictionary_options)
        )

    async def aapply_vector_index(
        self,
        index: BaseIndex,
        name: Optional[str] = None,
        *,
        concurrently: bool = False,
    ) -> None:
        """Create the HNSW inverted index on the embedding column.

        When a ``hybrid_search_config`` is set, a single combined inverted index is
        created over both the content column (analyzed with a BM25-capable dictionary)
        and the embedding column, with the id stored via ``INCLUDE`` so the lexical
        branch can return it.
        """
        if isinstance(index, ExactNearestNeighbor):
            await self.adrop_vector_index()
            return
        if index.distance_strategy != self.distance_strategy:
            # Keep the index metric aligned with the store's query operator.
            index.distance_strategy = self.distance_strategy
        hnsw_options = index.index_options()

        if self.hybrid_search_config:
            await self._acreate_text_search_dictionary(self.hybrid_search_config)
            index_name = name or self.hybrid_search_config.index_name
            stmt = build_hybrid_index_ddl(
                schema_name=self.schema_name,
                table_name=self.table_name,
                content_column=self.content_column,
                embedding_column=self.embedding_column,
                id_column=self.id_column,
                index_name=index_name,
                dictionary_name=self.hybrid_search_config.dictionary_name,
                hnsw_options=hnsw_options,
            )
        else:
            index_name = name or index.name or (self.table_name + DEFAULT_INDEX_NAME_SUFFIX)
            index.name = index_name
            stmt = build_vector_index_ddl(
                schema_name=self.schema_name,
                table_name=self.table_name,
                embedding_column=self.embedding_column,
                index_name=index_name,
                hnsw_options=hnsw_options,
            )
        await self.engine._aexecute(stmt)
        # Publish existing rows so the new index is immediately searchable.
        await self.engine._arefresh_table(
            self.table_name, schema_name=self.schema_name
        )

    async def aapply_hybrid_search_index(self, concurrently: bool = False) -> None:
        """Create the combined inverted index for hybrid search."""
        if not self.hybrid_search_config:
            raise ValueError(
                "hybrid_search_config is required to create a hybrid search index."
            )
        from .indexes import HNSWIndex

        await self.aapply_vector_index(
            HNSWIndex(distance_strategy=self.distance_strategy)
        )

    async def areindex(self, index_name: Optional[str] = None) -> None:
        """Recompute inverted-index statistics (SereneDB has no ``REINDEX``)."""
        await self.engine._aexecute_autocommit(
            f'VACUUM (RECOMPUTE_STATS_TABLE) "{self.schema_name}"."{self.table_name}";'
        )

    async def adrop_vector_index(self, index_name: Optional[str] = None) -> None:
        index_name = index_name or (self.table_name + DEFAULT_INDEX_NAME_SUFFIX)
        await self.engine._aexecute(
            f'DROP INDEX IF EXISTS "{self.schema_name}"."{index_name}";'
        )

    async def is_valid_index(self, index_name: Optional[str] = None) -> bool:
        index_name = index_name or (self.table_name + DEFAULT_INDEX_NAME_SUFFIX)
        query = (
            "SELECT tablename, indexname FROM pg_indexes "
            "WHERE tablename = %(table_name)s AND schemaname = %(schema_name)s "
            "AND indexname = %(index_name)s;"
        )
        rows = await self.engine._afetch(
            query,
            {
                "table_name": self.table_name,
                "schema_name": self.schema_name,
                "index_name": index_name,
            },
        )
        return len(rows) == 1

    # -- filter translation ----------------------------------------------------------

    def _handle_field_filter(self, *, field: str, value: Any) -> tuple[str, dict]:
        """Translate a single field filter to a SereneDB WHERE clause fragment."""
        if not isinstance(field, str):
            raise ValueError(
                f"field should be a string but got: {type(field)} with value: {field}"
            )
        if field.startswith("$"):
            raise ValueError(
                f"Invalid filter condition. Expected a field but got an operator: {field}"
            )
        if not (
            field.isidentifier()
            or all(part.isidentifier() for part in field.split("."))
        ):
            raise ValueError(f"Invalid field name: {field}. Expected a valid identifier.")

        if isinstance(value, dict):
            if len(value) != 1:
                raise ValueError(
                    "Invalid filter condition. Expected a value which is a dictionary "
                    "with a single key that corresponds to an operator but got a "
                    f"dictionary with {len(value)} keys. The first few keys are: "
                    f"{list(value.keys())[:3]}"
                )
            operator, filter_value = list(value.items())[0]
            if operator not in SUPPORTED_OPERATORS:
                raise ValueError(
                    f"Invalid operator: {operator}. Expected one of {SUPPORTED_OPERATORS}"
                )
        else:
            operator = "$eq"
            filter_value = value

        field_selector = field
        field_column = field.split(".")[0]
        field_param_prefix = field.replace(".", "_")

        is_json_field = (
            self.metadata_json_column is not None
            and field_column not in self.metadata_columns
            and field_column
            not in (self.id_column, self.content_column, self.embedding_column)
        )
        if is_json_field:
            field_selector = f"{self.metadata_json_column}.{field_selector}"

        if "." in field_selector:
            n_dots = field_selector.count(".")
            field_selector = "->".join(
                part
                if ind == 0
                else f"{'>' if ind == n_dots else ''}'{part}'"
                for ind, part in enumerate(field_selector.split("."))
            )
            value_type = (
                type(filter_value[0])
                if isinstance(filter_value, (list, tuple)) and filter_value
                else type(filter_value)
            )
            sdb_type = PYTHON_TO_SDB_TYPE_MAP.get(value_type)
            if sdb_type is None:
                raise ValueError(f"Unsupported type: {value_type}")
            # SereneDB: '->' is low-precedence, so the extraction MUST be parenthesized
            # before any comparison, whether or not a cast follows.
            if sdb_type != "TEXT" and operator != "$exists":
                field_selector = f"({field_selector})::{sdb_type}"
            else:
                field_selector = f"({field_selector})"
        else:
            field_selector = f'"{field_selector}"'

        suffix_id = str(uuid.uuid4()).split("-")[0]

        if operator in COMPARISONS_TO_NATIVE:
            native = COMPARISONS_TO_NATIVE[operator]
            param_name = f"{field_param_prefix}_{suffix_id}"
            return f"{field_selector} {native} %({param_name})s", {param_name: filter_value}

        if operator == "$between":
            low, high = filter_value
            low_p = f"{field_param_prefix}_low_{suffix_id}"
            high_p = f"{field_param_prefix}_high_{suffix_id}"
            return (
                f"({field_selector} BETWEEN %({low_p})s AND %({high_p})s)",
                {low_p: low, high_p: high},
            )

        if operator in {"$in", "$nin"}:
            for val in filter_value:
                if not isinstance(val, (str, int, float)) or isinstance(val, bool):
                    raise NotImplementedError(
                        f"Unsupported type: {type(val)} for value: {val}"
                    )
            keyword = "IN" if operator == "$in" else "NOT IN"
            if not filter_value:
                # Empty set: $in matches nothing, $nin matches everything.
                return ("(1 = 0)" if operator == "$in" else "(1 = 1)", {})
            placeholders = []
            params: dict[str, Any] = {}
            for i, val in enumerate(filter_value):
                pname = f"{field_param_prefix}_{operator[1:]}_{i}_{suffix_id}"
                placeholders.append(f"%({pname})s")
                params[pname] = val
            return f"{field_selector} {keyword} ({', '.join(placeholders)})", params

        if operator in {"$like", "$ilike"}:
            param_name = f"{field_param_prefix}_{operator[1:]}_{suffix_id}"
            keyword = "LIKE" if operator == "$like" else "ILIKE"
            return f"({field_selector} {keyword} %({param_name})s)", {param_name: filter_value}

        if operator == "$exists":
            if not isinstance(filter_value, bool):
                raise ValueError(
                    f"Expected a boolean value for $exists operator, but got: {filter_value}"
                )
            null_test = "IS NOT NULL" if filter_value else "IS NULL"
            return f"({field_selector} {null_test})", {}

        raise NotImplementedError()

    def _create_filter_clause(self, filters: Any) -> tuple[str, dict]:
        """Translate a LangChain filter dict into a SereneDB WHERE clause."""
        if not isinstance(filters, dict):
            raise ValueError(
                f"Invalid type: Expected a dictionary but got type: {type(filters)}"
            )
        if len(filters) == 1:
            key, value = list(filters.items())[0]
            if key.startswith("$"):
                if key.lower() not in ("$and", "$or", "$not"):
                    raise ValueError(
                        f"Invalid filter condition. Expected $and, $or or $not but got: {key}"
                    )
            else:
                return self._handle_field_filter(field=key, value=filters[key])

            if key.lower() in ("$and", "$or"):
                if not isinstance(value, list):
                    raise ValueError(
                        f"Expected a list, but got {type(value)} for value: {value}"
                    )
                op = key[1:].upper()
                clauses = [self._create_filter_clause(el) for el in value]
                if len(clauses) > 1:
                    all_clauses = [c[0] for c in clauses]
                    params: dict[str, Any] = {}
                    for c in clauses:
                        params.update(c[1])
                    return f"({f' {op} '.join(all_clauses)})", params
                elif len(clauses) == 1:
                    return clauses[0]
                raise ValueError(
                    "Invalid filter condition. Expected a dictionary but got an empty dictionary"
                )
            else:  # $not
                if isinstance(value, list):
                    not_conditions = [self._create_filter_clause(item) for item in value]
                    all_clauses = [c[0] for c in not_conditions]
                    params = {}
                    for c in not_conditions:
                        params.update(c[1])
                    not_stmts = [f"NOT {c}" for c in all_clauses]
                    return f"({' AND '.join(not_stmts)})", params
                elif isinstance(value, dict):
                    clause, params = self._create_filter_clause(value)
                    return f"(NOT {clause})", params
                raise ValueError(
                    f"Invalid filter condition. Expected a dictionary or a list but got: {type(value)}"
                )
        elif len(filters) > 1:
            for key in filters.keys():
                if key.startswith("$"):
                    raise ValueError(
                        f"Invalid filter condition. Expected a field but got: {key}"
                    )
            clauses = [
                self._handle_field_filter(field=k, value=v) for k, v in filters.items()
            ]
            all_clauses = [c[0] for c in clauses]
            params = {}
            for c in clauses:
                params.update(c[1])
            return f"({' AND '.join(all_clauses)})", params
        else:
            return "", {}

    # -- classmethod constructors ----------------------------------------------------

    @classmethod
    async def afrom_texts(  # type: ignore[override]
        cls,
        texts: list[str],
        embedding: Embeddings,
        engine: SereneDBEngine,
        table_name: str,
        *,
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> AsyncSereneDBVectorStore:
        vs = await cls.create(engine, embedding, table_name, **kwargs)
        await vs.aadd_texts(texts, metadatas=metadatas, ids=ids)
        return vs

    @classmethod
    async def afrom_documents(  # type: ignore[override]
        cls,
        documents: list[Document],
        embedding: Embeddings,
        engine: SereneDBEngine,
        table_name: str,
        *,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> AsyncSereneDBVectorStore:
        vs = await cls.create(engine, embedding, table_name, **kwargs)
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        await vs.aadd_texts(texts, metadatas=metadatas, ids=ids)
        return vs

    # -- sync surface required by VectorStore (use SereneDBVectorStore instead) -------

    def add_texts(self, *args: Any, **kwargs: Any) -> list[str]:
        raise NotImplementedError(
            "Use SereneDBVectorStore for sync access, or await aadd_texts."
        )

    def similarity_search(self, *args: Any, **kwargs: Any) -> list[Document]:
        raise NotImplementedError(
            "Use SereneDBVectorStore for sync access, or await asimilarity_search."
        )

    @classmethod
    def from_texts(cls, *args: Any, **kwargs: Any) -> AsyncSereneDBVectorStore:
        raise NotImplementedError("Use afrom_texts.")
