"""Synchronous vector store backed by SereneDB (scaffold).

``SereneDBVectorStore`` is a thin synchronous wrapper around
:class:`~langchain_serenedb.async_vectorstore.AsyncSereneDBVectorStore`. It delegates to
the async implementation through the engine's background event loop, so once the async
methods are implemented the sync surface works automatically.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from .async_vectorstore import AsyncSereneDBVectorStore
from .engine import SereneDBEngine
from .hybrid_search_config import HybridSearchConfig
from .indexes import (
    DEFAULT_DISTANCE_STRATEGY,
    BaseIndex,
    DistanceStrategy,
    QueryOptions,
)


class SereneDBVectorStore(VectorStore):
    """Synchronous vector store for SereneDB. Construct via :meth:`create_sync`."""

    __create_key = object()

    def __init__(
        self, key: object, engine: SereneDBEngine, vs: AsyncSereneDBVectorStore
    ) -> None:
        if key != SereneDBVectorStore.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )
        self._engine = engine
        self.__vs = vs

    @property
    def embeddings(self) -> Embeddings:
        return self.__vs.embedding_service

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
        id_column: str = "langchain_id",
        metadata_json_column: Optional[str] = "langchain_metadata",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        index_query_options: Optional[QueryOptions] = None,
        hybrid_search_config: Optional[HybridSearchConfig] = None,
    ) -> SereneDBVectorStore:
        """Async factory that builds the sync store (validating the table)."""
        coro = AsyncSereneDBVectorStore.create(
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
        vs = await engine._run_as_async(coro)
        return cls(cls.__create_key, engine, vs)

    @classmethod
    def create_sync(
        cls,
        engine: SereneDBEngine,
        embedding_service: Embeddings,
        table_name: str,
        **kwargs: Any,
    ) -> SereneDBVectorStore:
        """Sync factory that builds the sync store (validating the table)."""
        coro = AsyncSereneDBVectorStore.create(
            engine, embedding_service, table_name, **kwargs
        )
        vs = engine._run_as_sync(coro)
        return cls(cls.__create_key, engine, vs)

    # -- writes ----------------------------------------------------------------------

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> list[str]:
        return self._engine._run_as_sync(
            self.__vs.aadd_texts(texts, metadatas=metadatas, ids=ids, **kwargs)
        )

    def add_documents(
        self, documents: list[Document], ids: Optional[list] = None, **kwargs: Any
    ) -> list[str]:
        return self._engine._run_as_sync(
            self.__vs.aadd_documents(documents, ids=ids, **kwargs)
        )

    def delete(self, ids: Optional[list] = None, **kwargs: Any) -> Optional[bool]:
        return self._engine._run_as_sync(self.__vs.adelete(ids=ids, **kwargs))

    # -- search ----------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        return self._engine._run_as_sync(
            self.__vs.asimilarity_search(query, k=k, filter=filter, **kwargs)
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        return self._engine._run_as_sync(
            self.__vs.asimilarity_search_with_score(query, k=k, filter=filter, **kwargs)
        )

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        return self._engine._run_as_sync(
            self.__vs.asimilarity_search_by_vector(
                embedding, k=k, filter=filter, **kwargs
            )
        )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        return self._engine._run_as_sync(
            self.__vs.amax_marginal_relevance_search(
                query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter,
                **kwargs,
            )
        )

    def get_by_ids(self, ids: Sequence[str]) -> list[Document]:
        return self._engine._run_as_sync(self.__vs.aget_by_ids(ids))

    # -- index management ------------------------------------------------------------

    def apply_vector_index(
        self,
        index: BaseIndex,
        name: Optional[str] = None,
        *,
        concurrently: bool = False,
    ) -> None:
        return self._engine._run_as_sync(
            self.__vs.aapply_vector_index(index, name=name, concurrently=concurrently)
        )

    def apply_hybrid_search_index(self, *, concurrently: bool = False) -> None:
        return self._engine._run_as_sync(
            self.__vs.aapply_hybrid_search_index(concurrently=concurrently)
        )

    def reindex(self, index_name: Optional[str] = None) -> None:
        return self._engine._run_as_sync(self.__vs.areindex(index_name))

    def drop_vector_index(self, index_name: Optional[str] = None) -> None:
        return self._engine._run_as_sync(self.__vs.adrop_vector_index(index_name))

    def is_valid_index(self, index_name: Optional[str] = None) -> bool:
        return self._engine._run_as_sync(self.__vs.is_valid_index(index_name))

    @classmethod
    def from_texts(cls, *args: Any, **kwargs: Any) -> SereneDBVectorStore:
        raise NotImplementedError(
            "Initialize the table with SereneDBEngine.init_vectorstore_table, then use "
            "create_sync()."
        )
