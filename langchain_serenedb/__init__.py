"""LangChain integration for SereneDB.

SereneDB speaks the PostgreSQL wire protocol; this package connects with psycopg3 and
provides a LangChain vector store over SereneDB's native ``FLOAT[N]`` vectors, HNSW
inverted index, and BM25 full-text search.
"""

from importlib import metadata

from langchain_serenedb.async_vectorstore import AsyncSereneDBVectorStore
from langchain_serenedb.engine import Column, ColumnDict, SereneDBEngine
from langchain_serenedb.hybrid_search_config import (
    HybridSearchConfig,
    reciprocal_rank_fusion,
    weighted_sum_ranking,
)
from langchain_serenedb.indexes import (
    DistanceStrategy,
    HNSWIndex,
    HNSWQueryOptions,
)
from langchain_serenedb.vectorstores import SereneDBVectorStore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""

__all__ = [
    "__version__",
    "AsyncSereneDBVectorStore",
    "Column",
    "ColumnDict",
    "DistanceStrategy",
    "HNSWIndex",
    "HNSWQueryOptions",
    "HybridSearchConfig",
    "SereneDBEngine",
    "SereneDBVectorStore",
    "reciprocal_rank_fusion",
    "weighted_sum_ranking",
]
