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
  inverted index; ``aadd_embeddings`` / ``adelete`` refresh automatically unless the
  store was created with ``sync_load=False`` (then the caller owns refreshing).
- Vector index: ``CREATE INDEX ... USING inverted (emb ivf (metric='cosine', ...))``.
"""

from __future__ import annotations

import datetime
import json
import uuid
from typing import Any, Callable, Iterable, Optional, Sequence, cast

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores import utils as lc_utils

from .engine import SereneDBEngine
from .hybrid_search_config import FusionStrategy, HybridSearchConfig
from .indexes import (
    DEFAULT_DISTANCE_STRATEGY,
    DEFAULT_INDEX_NAME_SUFFIX,
    BaseIndex,
    DistanceStrategy,
    ExactNearestNeighbor,
    JsonFieldIndex,
    MetadataIndexConfig,
    QueryOptions,
    build_dictionary_ddl,
    build_hybrid_index_ddl,
    build_json_field_selector,
    build_metadata_index_entries,
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

# Full-text-search operators. Each translates to ``column @@ ts_*(...)`` — the inverted
# index match. SereneDB evaluates ``@@`` only against an inverted-indexed column, so the
# filter translator gates these to indexed metadata columns (raising a clear error
# otherwise) rather than letting the DB reject them cryptically. Unlike the plain
# operators above, they have no residual/exact-scan fallback.
FTS_UNARY_FUNCTIONS = {
    "$startswith": "ts_starts_with",  # prefix match
    "$regex": "ts_regexp",  # regular-expression match against indexed terms
    "$phrase": "ts_phrase",  # ordered-adjacent phrase match
    "$fuzzy": "ts_levenshtein",  # typo-tolerant match (auto edit distance)
}
FTS_OPERATORS = set(FTS_UNARY_FUNCTIONS) | {"$match"}  # $match -> ts_any (OR / N-of-M)

SUPPORTED_OPERATORS = (
    set(COMPARISONS_TO_NATIVE)
    | TEXT_OPERATORS
    | SPECIAL_CASED_OPERATORS
    | set(LOGICAL_OPERATORS)
    | FTS_OPERATORS
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
        metadata_columns: Optional[dict[str, Optional[str]]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: Optional[str] = "langchain_metadata",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        index_query_options: Optional[QueryOptions] = None,
        hybrid_search_config: Optional[HybridSearchConfig] = None,
        metadata_index: Optional[MetadataIndexConfig] = None,
        sync_load: bool = True,
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
        # Ordered mapping of metadata column name -> declared SQL type (``None`` if
        # unknown). Iterating / membership-testing yields the names, as before; the type
        # lets the default index-all case skip columns the inverted index can't accept.
        self.metadata_columns = metadata_columns if metadata_columns is not None else {}
        self.id_column = id_column
        self.metadata_json_column = metadata_json_column
        self.distance_strategy = distance_strategy
        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.index_query_options = index_query_options
        self.hybrid_search_config = hybrid_search_config
        # Which metadata columns / JSON sub-fields are (or will be) added to the
        # inverted index; consulted by the filter translator so a declared JSON field's
        # query expression matches its indexed expression (arrow + cast).
        self.metadata_index = metadata_index
        # When False, writes do NOT auto-refresh the inverted index; the caller is
        # responsible for calling refresh_table() to publish them (faster bulk loads).
        self.sync_load = sync_load
        # Cached result of is_valid_index(): None = not checked yet. Kept in sync by
        # aapply_vector_index / adrop_vector_index so the dense query need not probe
        # pg_indexes on every search.
        self._index_exists: Optional[bool] = None
        # Output-column alias for the computed distance/score. "distance" unless a real
        # column (id/content/embedding/json/metadata) already uses that name, in which
        # case a non-colliding name is chosen so the SELECT has no duplicate output
        # column (which would otherwise shadow the real column in the row dict).
        self._distance_alias = self._unique_output_alias("distance")

    def _unique_output_alias(self, base: str) -> str:
        """Pick a result-column alias that doesn't collide with any selected column."""
        reserved = {self.id_column, self.content_column, self.embedding_column}
        reserved.update(self.metadata_columns)
        if self.metadata_json_column:
            reserved.add(self.metadata_json_column)
        if base not in reserved:
            return base
        i = 1
        while f"{base}_{i}" in reserved:
            i += 1
        return f"{base}_{i}"

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_service

    async def _avector_index_available(self) -> bool:
        """Whether this collection's vector (IVF) index exists (cached).

        The dense query only uses the index when selecting from it by name, so it needs
        to know whether the index is there. Determined once via ``is_valid_index`` and
        then cached; index create/drop update the cache.
        """
        if self._index_exists is None:
            self._index_exists = await self.is_valid_index()
        return self._index_exists

    @property
    def _vector_index_name(self) -> str:
        """The single inverted index this collection uses.

        One index per collection (vector-only or the combined hybrid index); its name is
        derived from the table so the store always knows it without tracking user input.
        """
        return self.table_name + DEFAULT_INDEX_NAME_SUFFIX

    @staticmethod
    def _l1_relevance_score_fn(distance: float) -> float:
        """Relevance score for L1 (Manhattan) distance, mirroring the base Euclidean fn.

        For unit-normalized embeddings the L1 distance is 0 for identical vectors and 2
        for orthogonal ones (the analogue of Euclidean's ``sqrt(2)``), so map onto a
        [0, 1] similarity with ``1 - d/2``.
        """
        return 1.0 - distance / 2.0

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """Map the distance strategy to a [0, 1] relevance-score function.

        Required by ``similarity_search_with_relevance_scores`` and the
        ``similarity_score_threshold`` retriever (the base class raises otherwise). The
        distance we report matches each strategy's named function, so the base helpers
        apply directly: cosine and L2 are true distances, and inner product is reported
        as the *negative* inner product (see :class:`DistanceStrategy`), which is exactly
        what the base max-inner-product helper expects. L1 uses our own helper above.
        """
        if self.distance_strategy == DistanceStrategy.COSINE_DISTANCE:
            return self._cosine_relevance_score_fn
        if self.distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self._euclidean_relevance_score_fn
        if self.distance_strategy == DistanceStrategy.INNER_PRODUCT:
            return self._max_inner_product_relevance_score_fn
        if self.distance_strategy == DistanceStrategy.MANHATTAN:
            return self._l1_relevance_score_fn
        raise NotImplementedError(
            f"No relevance-score function for distance strategy "
            f"{self.distance_strategy.name}."
        )

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
        metadata_index: Optional[MetadataIndexConfig] = None,
        sync_load: bool = True,
    ) -> AsyncSereneDBVectorStore:
        """Create an ``AsyncSereneDBVectorStore`` bound to an existing table.

        ``sync_load=False`` skips the automatic inverted-index refresh after each
        write (add/delete), so a bulk load runs faster; the caller then owns calling
        ``refresh_table()`` before those rows are visible to full-text / vector-index-
        routed queries.

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

        # Map each metadata column to its reported SQL type (from information_schema), so
        # the default index-all case can skip columns whose type the inverted index can't
        # accept.
        metadata_columns_with_types = {c: columns.get(c) for c in metadata_columns}

        return cls(
            cls.__create_key,
            engine,
            embedding_service,
            table_name,
            schema_name=schema_name,
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_columns=metadata_columns_with_types,
            id_column=id_column,
            metadata_json_column=metadata_json_column,
            distance_strategy=distance_strategy,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            index_query_options=index_query_options,
            hybrid_search_config=hybrid_search_config,
            metadata_index=metadata_index,
            sync_load=sync_load,
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
        # The pool sets row_factory=dict_row, so rows are dicts; psycopg's stubs cannot
        # see the dynamically-set factory and still type them as tuples.
        return cast(list[dict[str, Any]], rows)

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

        if not texts:
            return ids

        # Build ONE uniform INSERT ... ON CONFLICT statement for every row (the columns
        # are fixed; absent metadata columns bind NULL), then run it as a single batched
        # statement -- one connection, one commit -- instead of one round-trip per row.
        dim = len(embeddings[0])
        metadata_col_names = "".join(f', "{col}"' for col in self.metadata_columns)
        insert_stmt = (
            f'INSERT INTO "{self.schema_name}"."{self.table_name}"('
            f'"{self.id_column}", "{self.content_column}", "{self.embedding_column}"'
            f"{metadata_col_names}"
        )
        values_stmt = (
            f"VALUES (%(langchain_id)s, %(content)s, %(embedding)s::FLOAT[{dim}]"
        )
        for col in self.metadata_columns:
            values_stmt += f", %({col})s"
        if self.metadata_json_column:
            insert_stmt += f', "{self.metadata_json_column}")'
            values_stmt += ", %(extra)s)"
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
        statement = insert_stmt + values_stmt + upsert_stmt

        params_seq: list[dict[str, Any]] = []
        for id, content, embedding, metadata in zip(ids, texts, embeddings, metadatas):
            values: dict[str, Any] = {
                "langchain_id": id,
                "content": content,
                "embedding": [float(d) for d in embedding],
            }
            extra = dict(metadata)
            for metadata_column in self.metadata_columns:
                if metadata_column in metadata:
                    value = metadata[metadata_column]
                    values[metadata_column] = (
                        json.dumps(value) if isinstance(value, dict) else value
                    )
                    del extra[metadata_column]
                else:
                    values[metadata_column] = None
            if self.metadata_json_column:
                values["extra"] = json.dumps(extra)
            params_seq.append(values)

        await self.engine._aexecute_many(statement, params_seq)
        # Publish the writes to the inverted index (eventual consistency) unless the
        # caller opted out via sync_load=False to speed up a bulk load.
        if self.sync_load:
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
        if self.sync_load:
            await self.engine._arefresh_table(
                self.table_name, schema_name=self.schema_name
            )
        return True

    # -- search ----------------------------------------------------------------------

    def _select_columns(self, *, include_embedding: bool = True) -> str:
        columns = [self.id_column, self.content_column]
        if include_embedding:
            columns.append(self.embedding_column)
        columns += list(self.metadata_columns)
        if self.metadata_json_column:
            columns.append(self.metadata_json_column)
        return ", ".join(f'"{col}"' for col in columns)

    def _rows_to_documents(
        self, rows: list[dict[str, Any]], *, with_score: bool = False
    ) -> list:
        out = []
        for row in rows:
            raw_meta = (
                row.get(self.metadata_json_column)
                if self.metadata_json_column
                else None
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
            out.append((doc, row[self._distance_alias]) if with_score else doc)
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
        search_function = self.distance_strategy.search_function
        column_names = self._select_columns()

        safe_filter, filter_dict = (None, None)
        if filter and isinstance(filter, dict):
            safe_filter, filter_dict = self._create_filter_clause(filter)
        where_filters = f"WHERE {safe_filter}" if safe_filter else ""

        # SereneDB routes an ``ORDER BY emb <op> q LIMIT k`` through the vector index only
        # when selecting *from the index by name*; querying the base table scans exactly.
        # So read from the index when it exists, and fall back to an exact scan of the
        # table otherwise.
        if await self._avector_index_available():
            source = f'"{self.schema_name}"."{self._vector_index_name}"'
        else:
            source = f'"{self.schema_name}"."{self.table_name}"'

        alias = self._distance_alias
        query = (
            f"SELECT {column_names}, "
            f'{search_function}("{self.embedding_column}", %(query_embedding)s::FLOAT[{dim}]) AS "{alias}" '
            f"FROM {source} {where_filters} "
            f'ORDER BY "{alias}" '
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
            prelude = [
                f"SET LOCAL {opt};" for opt in self.index_query_options.to_parameter()
            ]
        return await self._aquery(query, params, prelude=prelude)

    async def _ahybrid_query(
        self,
        embedding: list[float],
        cfg: HybridSearchConfig,
        fts_query: str,
        *,
        k: int,
        filter: Optional[dict] = None,
    ) -> list[dict[str, Any]]:
        """Fuse the lexical (BM25) and vector rankings in ONE SereneDB query.

        Both branches select from the combined inverted index by name (BM25 needs the
        index ``tableoid``; the vector ANN routes through the same index), each capped at
        its own per-branch window and ranked/scored in a ``WITH`` CTE. The outer query
        combines them per :attr:`HybridSearchConfig.fusion` and joins the base table back
        to project the content/metadata columns. The fused score (higher = better) is
        emitted under :attr:`_distance_alias`.
        """
        dim = len(embedding)
        esc = self.engine._escape_identifier
        schema = esc(self.schema_name)
        index = esc(self._vector_index_name)
        table = esc(self.table_name)
        operator = self.distance_strategy.operator
        alias = self._distance_alias

        safe_filter, filter_dict = (None, None)
        if filter and isinstance(filter, dict):
            safe_filter, filter_dict = self._create_filter_clause(filter)
        lex_and = f" AND ({safe_filter})" if safe_filter else ""
        vec_where = f"WHERE ({safe_filter})" if safe_filter else ""

        # Inner branch subqueries, each producing (fid, sc): lexical BM25 (higher better)
        # and vector distance (smaller better). Both read from the index by name.
        lex_inner = (
            f'SELECT "{self.id_column}" AS fid, '
            f'{cfg.scorer}("{index}".tableoid) AS sc '
            f'FROM "{schema}"."{index}" '
            f'WHERE "{self.content_column}" @@ {cfg.tsquery_function}(%(fts_query)s)'
            f"{lex_and} "
            f"ORDER BY sc DESC LIMIT %(lex_window)s"
        )
        vec_inner = (
            f'SELECT "{self.id_column}" AS fid, '
            f'"{self.embedding_column}" {operator} %(query_embedding)s::FLOAT[{dim}] AS sc '
            f'FROM "{schema}"."{index}" '
            f"{vec_where} "
            f"ORDER BY sc LIMIT %(vec_window)s"
        )

        params: dict[str, Any] = {
            "fts_query": fts_query,
            "query_embedding": [float(d) for d in embedding],
            "lex_window": cfg.secondary_top_k,
            "vec_window": cfg.primary_top_k,
            "final_k": k,
        }
        if filter_dict:
            params.update(filter_dict)

        # Per-strategy CTE producing (fid, _score); _score is higher-is-better.
        if cfg.fusion == FusionStrategy.RRF:
            with_prefix = (
                f"WITH fused AS ("
                f"SELECT fid, RANK() OVER (ORDER BY sc DESC) AS rnk FROM ({lex_inner}) lex "
                f"UNION ALL "
                f"SELECT fid, RANK() OVER (ORDER BY sc) AS rnk FROM ({vec_inner}) vec) "
            )
            scored = (
                "SELECT fid, SUM(1.0 / (%(rrf_k)s + rnk)) AS _score "
                "FROM fused GROUP BY fid"
            )
            params["rrf_k"] = cfg.rrf_k
        elif cfg.fusion == FusionStrategy.NORMALIZED:
            # branch 1 = lexical (higher better), branch 2 = vector (distance -> invert).
            with_prefix = (
                f"WITH hits AS ("
                f"SELECT 1 AS branch, fid, sc FROM ({lex_inner}) lex "
                f"UNION ALL "
                f"SELECT 2 AS branch, fid, sc FROM ({vec_inner}) vec), "
                f"normed AS (SELECT fid, ("
                f"CASE WHEN MAX(sc) OVER w = MIN(sc) OVER w THEN 1.0 "
                f"WHEN branch = 2 THEN (MAX(sc) OVER w - sc) / "
                f"(MAX(sc) OVER w - MIN(sc) OVER w) "
                f"ELSE (sc - MIN(sc) OVER w) / (MAX(sc) OVER w - MIN(sc) OVER w) END"
                f") * (CASE WHEN branch = 2 THEN %(w_primary)s ELSE %(w_secondary)s END) "
                f"AS ns FROM hits WINDOW w AS (PARTITION BY branch)) "
            )
            scored = "SELECT fid, SUM(ns) AS _score FROM normed GROUP BY fid"
            params["w_primary"] = cfg.primary_results_weight
            params["w_secondary"] = cfg.secondary_results_weight
        elif cfg.fusion == FusionStrategy.WEIGHTED_SUM:
            # Raw weighted sum; negate the vector distance so higher = better.
            with_prefix = (
                f"WITH hits AS ("
                f"SELECT fid, (%(w_secondary)s * sc) AS contrib FROM ({lex_inner}) lex "
                f"UNION ALL "
                f"SELECT fid, (%(w_primary)s * (-sc)) AS contrib FROM ({vec_inner}) vec) "
            )
            scored = "SELECT fid, SUM(contrib) AS _score FROM hits GROUP BY fid"
            params["w_primary"] = cfg.primary_results_weight
            params["w_secondary"] = cfg.secondary_results_weight
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unknown fusion strategy: {cfg.fusion}")

        proj = [self.id_column, self.content_column, *self.metadata_columns]
        if self.metadata_json_column:
            proj.append(self.metadata_json_column)
        m_cols = ", ".join(f'm."{c}"' for c in proj)

        query = (
            f"{with_prefix}"
            f'SELECT {m_cols}, f._score AS "{alias}" '
            f"FROM ({scored}) f "
            f'JOIN "{schema}"."{table}" m ON m."{self.id_column}" = f.fid '
            f'ORDER BY f._score DESC, m."{self.id_column}" '
            f"LIMIT %(final_k)s;"
        )

        prelude = None
        if self.index_query_options:
            prelude = [
                f"SET LOCAL {opt};" for opt in self.index_query_options.to_parameter()
            ]
        return await self._aquery(query, params, prelude=prelude)

    async def _aquery_collection(
        self,
        embedding: list[float],
        *,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Dense search, or a single-query hybrid (BM25 + vector fused) when configured."""
        hybrid_search_config = kwargs.get(
            "hybrid_search_config", self.hybrid_search_config
        )
        final_k = k if k is not None else self.k

        fts_query = ""
        if hybrid_search_config:
            fts_query = hybrid_search_config.fts_query or kwargs.get("fts_query", "")
        if hybrid_search_config and fts_query:
            return await self._ahybrid_query(
                embedding, hybrid_search_config, fts_query, k=final_k, filter=filter
            )
        return await self._adense_query(embedding, limit=final_k, filter=filter)

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
        docs_and_scores = (
            await self.amax_marginal_relevance_search_with_score_by_vector(
                embedding,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter,
                **kwargs,
            )
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
            row.setdefault(self._distance_alias, 0.0)
        docs = self._rows_to_documents(rows)
        # Return in the same order as the requested ids (VectorStore contract).
        by_id = {doc.id: doc for doc in docs}
        return [by_id[str(i)] for i in ids if str(i) in by_id]

    # -- index management ------------------------------------------------------------

    async def _acreate_text_search_dictionary(self, config: HybridSearchConfig) -> None:
        """Create the text search dictionary used to analyze the content column."""
        await self.engine._aexecute(
            build_dictionary_ddl(
                self.schema_name, config.dictionary_name, config.dictionary_options
            )
        )

    async def aapply_vector_index(
        self,
        index: BaseIndex,
        *,
        concurrently: bool = False,
    ) -> None:
        """Create the collection's single vector (IVF) inverted index on the embedding column.

        The index name is derived from the table (:attr:`_vector_index_name`) — one
        index per collection, fully controlled by the store. When a
        ``hybrid_search_config`` is set, a single combined inverted index is created over
        both the content column (analyzed with a BM25-capable dictionary) and the
        embedding column, with the id stored via ``INCLUDE`` so the lexical branch can
        return it.
        """
        if isinstance(index, ExactNearestNeighbor):
            await self.adrop_vector_index()
            return
        if index.distance_strategy != self.distance_strategy:
            # Keep the index metric aligned with the store's query operator.
            index.distance_strategy = self.distance_strategy
        vector_opclass = index.index_options()
        index_name = self._vector_index_name
        metadata_entries = build_metadata_index_entries(
            self.metadata_index,
            metadata_json_column=self.metadata_json_column,
            metadata_columns=self.metadata_columns,
        )

        if self.hybrid_search_config:
            await self._acreate_text_search_dictionary(self.hybrid_search_config)
            stmt = build_hybrid_index_ddl(
                schema_name=self.schema_name,
                table_name=self.table_name,
                content_column=self.content_column,
                embedding_column=self.embedding_column,
                id_column=self.id_column,
                index_name=index_name,
                dictionary_name=self.hybrid_search_config.dictionary_name,
                vector_opclass=vector_opclass,
                metadata_entries=metadata_entries,
            )
        else:
            stmt = build_vector_index_ddl(
                schema_name=self.schema_name,
                table_name=self.table_name,
                embedding_column=self.embedding_column,
                index_name=index_name,
                vector_opclass=vector_opclass,
                metadata_entries=metadata_entries,
            )
        await self.engine._aexecute(stmt)
        self._index_exists = True

    async def aapply_hybrid_search_index(
        self, index: Optional[BaseIndex] = None, concurrently: bool = False
    ) -> None:
        """Create the combined inverted index for hybrid search.

        ``index`` selects the vector-index configuration (defaults to a plain
        :class:`~langchain_serenedb.indexes.IVFIndex`); pass e.g.
        ``IVFIndex(quant="sq8", ...)`` to build a quantized combined index.
        """
        if not self.hybrid_search_config:
            raise ValueError(
                "hybrid_search_config is required to create a hybrid search index."
            )
        if index is None:
            from .indexes import IVFIndex

            index = IVFIndex(distance_strategy=self.distance_strategy)
        await self.aapply_vector_index(index, concurrently=concurrently)

    async def areindex(self) -> None:
        """Recompute inverted-index statistics (SereneDB has no ``REINDEX``)."""
        await self.engine._aexecute_autocommit(
            f'VACUUM (RECOMPUTE_STATS_TABLE) "{self.schema_name}"."{self.table_name}";'
        )

    async def adrop_vector_index(self) -> None:
        index_name = self._vector_index_name
        await self.engine._aexecute(
            f'DROP INDEX IF EXISTS "{self.schema_name}"."{index_name}";'
        )
        self._index_exists = False

    async def is_valid_index(self) -> bool:
        index_name = self._vector_index_name
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

    def _declared_json_field(self, field: str) -> Optional[JsonFieldIndex]:
        """Return the ``JsonFieldIndex`` declared for ``field``, or ``None``.

        A JSON sub-field added to the inverted index (via ``metadata_index.json_fields``)
        must be filtered with the *same* arrow/cast the index used, or the predicate
        won't push into the index scan. This tells the filter translator which fields
        are declared (and with what SQL type).
        """
        if self.metadata_index is None:
            return None
        for json_field in self.metadata_index.json_fields:
            if json_field.field == field:
                return json_field
        return None

    def _is_fts_filterable(self, field: str, field_column: str) -> bool:
        """Whether ``field`` is something this store's inverted index covers as a text term.

        Full-text operators (``$startswith``/``$regex``/``$fuzzy``/``$match``/``$phrase``)
        compile to ``<expr> @@ ts_*(...)``, which SereneDB evaluates only against an
        inverted-indexed text term. That is either:

        - an indexed metadata column — the default (``metadata_index`` None or ``columns``
          None) indexes every metadata column; an explicit ``columns`` list narrows it; or
        - a declared JSON sub-field of ``TEXT`` type (indexed as a bare ``(md ->> 'k')``
          term). Non-``TEXT`` declared fields carry a ``::TYPE`` cast, so they are not a
          text term and cannot be FTS-matched.

        Un-indexed columns and undeclared JSON fields are not FTS-filterable.
        """
        declared = self._declared_json_field(field)
        if declared is not None:
            return declared.data_type.strip().upper() == "TEXT"
        if field_column not in self.metadata_columns:
            return False
        mi = self.metadata_index
        if mi is None or mi.columns is None:
            return True
        return any(col.name == field_column for col in mi.columns)

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
            raise ValueError(
                f"Invalid field name: {field}. Expected a valid identifier."
            )

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

        declared_json = self._declared_json_field(field) if is_json_field else None
        if declared_json is not None:
            # Declared JSON field: emit the exact expression the index was built with
            # ('->>' extraction, plus the declared cast for non-TEXT types) so the
            # predicate pushes into the index scan instead of running as a post-filter.
            assert self.metadata_json_column is not None
            field_selector = build_json_field_selector(
                self.metadata_json_column,
                field,
                declared_json.data_type,
            )
        else:
            if is_json_field:
                field_selector = f"{self.metadata_json_column}.{field_selector}"

            if "." in field_selector:
                n_dots = field_selector.count(".")
                field_selector = "->".join(
                    part if ind == 0 else f"{'>' if ind == n_dots else ''}'{part}'"
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
                # SereneDB: '->' is low-precedence, so the extraction MUST be
                # parenthesized before any comparison, whether or not a cast follows.
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
            return f"{field_selector} {native} %({param_name})s", {
                param_name: filter_value
            }

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
            return f"({field_selector} {keyword} %({param_name})s)", {
                param_name: filter_value
            }

        if operator == "$exists":
            if not isinstance(filter_value, bool):
                raise ValueError(
                    f"Expected a boolean value for $exists operator, but got: {filter_value}"
                )
            null_test = "IS NOT NULL" if filter_value else "IS NULL"
            return f"({field_selector} {null_test})", {}

        if operator in FTS_OPERATORS:
            if not self._is_fts_filterable(field, field_column):
                raise ValueError(
                    f"Operator {operator} requires '{field}' to be an inverted-indexed "
                    "metadata column or a TEXT JSON field declared in "
                    "metadata_index.json_fields: full-text operators use the `@@` match, "
                    "which SereneDB evaluates only against an indexed text term. Add it to "
                    "the store's metadata index (or drop the operator)."
                )
            if operator == "$match":
                tokens = filter_value
                min_match = None
                if isinstance(filter_value, dict):
                    tokens = filter_value.get("tokens")
                    min_match = filter_value.get("min_match")
                if not isinstance(tokens, (list, tuple)) or not tokens:
                    raise ValueError(
                        "$match expects a non-empty list of tokens, or "
                        "{'tokens': [...], 'min_match': N}."
                    )
                params = {}
                placeholders = []
                for i, tok in enumerate(tokens):
                    pname = f"{field_param_prefix}_match_{i}_{suffix_id}"
                    placeholders.append(f"%({pname})s")
                    params[pname] = tok
                array = f"ARRAY[{', '.join(placeholders)}]"
                if min_match is not None:
                    # ts_any requires min_match as an INTEGER literal (not a bind param);
                    # validate it is an int so inlining it is injection-safe.
                    if isinstance(min_match, bool) or not isinstance(min_match, int):
                        raise ValueError("$match 'min_match' must be an integer.")
                    return (
                        f"({field_selector} @@ ts_any({array}, {min_match}))",
                        params,
                    )
                return f"({field_selector} @@ ts_any({array}))", params

            fn = FTS_UNARY_FUNCTIONS[operator]
            param_name = f"{field_param_prefix}_{operator[1:]}_{suffix_id}"
            return f"({field_selector} @@ {fn}(%({param_name})s))", {
                param_name: filter_value
            }

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
                    not_conditions = [
                        self._create_filter_clause(item) for item in value
                    ]
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
