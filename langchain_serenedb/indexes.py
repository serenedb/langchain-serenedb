"""Index classes for adding vector indexes on the SereneDBVectorStore.

SereneDB provides approximate-nearest-neighbor search through an **inverted index**
with an ``hnsw`` operator class on a fixed-size ``FLOAT[N]`` column::

    CREATE INDEX idx ON tbl USING inverted (embedding hnsw (metric = 'cosine', m = 16, ef_construction = 64))

HNSW is the only ANN index type. The distance *metric* is required and must match the
operator used at query time for the index to accelerate the search. See
https://<serenedb-docs>/sql/indexes/inverted/vector-search.
"""

import enum
import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StrategyMixin:
    """Bundles the SQL forms tied to a distance strategy.

    Attributes:
        operator: The infix distance operator used in ``ORDER BY`` (e.g. ``<=>``).
        search_function: The named scalar distance function used to compute the
            score column (e.g. ``cosine_distance``).
        index_metric: The ``metric`` value the HNSW index must be built with for the
            operator to accelerate (one of ``l2``, ``cosine``, ``ip``, ``l1``).
    """

    operator: str
    search_function: str
    index_metric: str


class DistanceStrategy(StrategyMixin, enum.Enum):
    """Enumerator of the distance strategies supported by SereneDB.

    Each strategy bundles the query operator, the named distance function and the HNSW
    index ``metric`` for one distance measure: Euclidean (L2), cosine, inner product,
    and Manhattan (L1).
    """

    EUCLIDEAN = "<->", "l2_distance", "l2"
    COSINE_DISTANCE = "<=>", "cosine_distance", "cosine"
    INNER_PRODUCT = "<#>", "inner_product", "ip"
    MANHATTAN = "<+>", "l1_distance", "l1"


DEFAULT_DISTANCE_STRATEGY: DistanceStrategy = DistanceStrategy.COSINE_DISTANCE
DEFAULT_INDEX_NAME_SUFFIX: str = "langchainvectorindex"


def validate_identifier(identifier: str) -> None:
    if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier) is None:
        raise ValueError(
            f"Invalid identifier: {identifier}. Identifiers must start with a letter "
            "or underscore, and subsequent characters can be letters, digits, or "
            "underscores."
        )


@dataclass
class BaseIndex(ABC):
    """Abstract base class for defining vector indexes.

    Attributes:
        name (Optional[str]): A human-readable name for the index. Defaults to None.
        index_type (str): A string identifying the type of index. Defaults to "base".
        distance_strategy (DistanceStrategy): The strategy used to calculate distances
            between vectors in the index. Defaults to DistanceStrategy.COSINE_DISTANCE.
        partial_indexes (Optional[list[str]]): Not supported; a non-empty value raises
            at build time.
    """

    name: Optional[str] = None
    index_type: str = "base"
    distance_strategy: DistanceStrategy = field(
        default_factory=lambda: DistanceStrategy.COSINE_DISTANCE
    )
    partial_indexes: Optional[list[str]] = None

    @abstractmethod
    def index_options(self) -> str:
        """Set index query options for vector store initialization."""
        raise NotImplementedError(
            "index_options method must be implemented by subclass"
        )

    def get_index_metric(self) -> str:
        return self.distance_strategy.index_metric

    def __post_init__(self) -> None:
        """Validate initialization parameters."""
        if self.index_type:
            validate_identifier(self.index_type)
        if self.partial_indexes:
            raise ValueError(
                "SereneDB inverted indexes do not support partial indexes "
                "(CREATE INDEX ... WHERE ...)."
            )


@dataclass
class ExactNearestNeighbor(BaseIndex):
    """Sentinel index type meaning 'no ANN index' — searches scan exactly."""

    index_type: str = "exactnearestneighbor"

    def index_options(self) -> str:
        return ""


@dataclass
class QueryOptions(ABC):
    @abstractmethod
    def to_parameter(self) -> list[str]:
        """Convert index attributes to list of configurations."""
        raise NotImplementedError("to_parameter method must be implemented by subclass")

    @abstractmethod
    def to_string(self) -> str:
        """Convert index attributes to string."""
        raise NotImplementedError("to_string method must be implemented by subclass")


@dataclass
class HNSWIndex(BaseIndex):
    """HNSW inverted-index configuration.

    ``m`` and ``ef_construction`` tune the graph; the ``metric`` is derived from
    :attr:`distance_strategy`. Recommended ranges: ``m`` 2-128, ``ef_construction``
    10-500.
    """

    index_type: str = "hnsw"
    m: int = 16
    ef_construction: int = 64

    def index_options(self) -> str:
        """Return the ``hnsw`` operator-class spec for the embedding column.

        Example: ``hnsw (metric = 'cosine', m = 16, ef_construction = 64)``.
        """
        return (
            f"hnsw (metric = '{self.get_index_metric()}', "
            f"m = {self.m}, ef_construction = {self.ef_construction})"
        )


@dataclass
class HNSWQueryOptions(QueryOptions):
    """Per-session HNSW search tuning.

    ``ef_search`` is applied as the ``sdb_ef_search`` session setting; ``0`` uses the
    query's ``LIMIT`` as the search beam.
    """

    ef_search: int = 0

    def to_parameter(self) -> list[str]:
        """Convert index attributes to list of configurations."""
        return [f"sdb_ef_search = {self.ef_search}"]

    def to_string(self) -> str:
        """Convert index attributes to string."""
        warnings.warn(
            "to_string is deprecated, use to_parameter instead.",
            DeprecationWarning,
        )
        return f"sdb_ef_search = {self.ef_search}"


def _quote_ident(name: str) -> str:
    """Escape and double-quote a SQL identifier."""
    return '"' + name.replace('"', '""') + '"'


def build_dictionary_ddl(
    dictionary_name: str, options: str, *, if_not_exists: bool = True
) -> str:
    """CREATE TEXT SEARCH DICTIONARY statement (for the full-text/hybrid analyzer)."""
    ine = "IF NOT EXISTS " if if_not_exists else ""
    return f"CREATE TEXT SEARCH DICTIONARY {ine}{_quote_ident(dictionary_name)} ({options});"


def build_vector_index_ddl(
    *,
    schema_name: str,
    table_name: str,
    embedding_column: str,
    index_name: str,
    hnsw_options: str,
    if_not_exists: bool = False,
) -> str:
    """CREATE INDEX for a vector-only HNSW inverted index on the embedding column.

    ``hnsw_options`` is the operator-class spec produced by :meth:`HNSWIndex.index_options`,
    e.g. ``hnsw (metric = 'cosine', m = 16, ef_construction = 64)``.
    """
    ine = "IF NOT EXISTS " if if_not_exists else ""
    return (
        f"CREATE INDEX {ine}{_quote_ident(index_name)} "
        f"ON {_quote_ident(schema_name)}.{_quote_ident(table_name)} "
        f"USING inverted ({_quote_ident(embedding_column)} {hnsw_options});"
    )


def build_hybrid_index_ddl(
    *,
    schema_name: str,
    table_name: str,
    content_column: str,
    embedding_column: str,
    id_column: str,
    index_name: str,
    dictionary_name: str,
    hnsw_options: str,
    if_not_exists: bool = True,
) -> str:
    """CREATE INDEX for one combined inverted index covering the content column
    (analyzed for BM25 full-text) and the embedding column (HNSW), storing the id via
    ``INCLUDE`` so the lexical branch can return it.
    """
    ine = "IF NOT EXISTS " if if_not_exists else ""
    return (
        f"CREATE INDEX {ine}{_quote_ident(index_name)} "
        f"ON {_quote_ident(schema_name)}.{_quote_ident(table_name)} "
        f"USING inverted ({_quote_ident(content_column)} {_quote_ident(dictionary_name)}, "
        f"{_quote_ident(embedding_column)} {hnsw_options}) "
        f"INCLUDE ({_quote_ident(id_column)});"
    )
