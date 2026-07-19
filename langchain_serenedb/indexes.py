"""Index classes for adding vector indexes on the SereneDBVectorStore.

SereneDB provides approximate-nearest-neighbor search through an **inverted index**
with an ``ivf`` operator class on a fixed-size ``FLOAT[N]`` column::

    CREATE INDEX idx ON tbl USING inverted (embedding ivf (metric = 'cosine', nlist = 100))

IVF is the ANN index type. The distance *metric* is required and must match the operator
used at query time for the index to accelerate the search; optional quantization
(``quant``) trades recall for memory/speed. See
https://<serenedb-docs>/sql/indexes/inverted/vector-search.
"""

import enum
import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StrategyMixin:
    """Bundles the SQL forms tied to a distance strategy.

    Attributes:
        operator: The infix distance operator used in ``ORDER BY`` (e.g. ``<=>``).
        search_function: The named scalar distance function used to compute the
            score column (e.g. ``cosine_distance``).
        index_metric: The ``metric`` value the IVF index must be built with for the
            operator to accelerate (one of ``l2``, ``cosine``, ``ip``, ``l1``).
    """

    operator: str
    search_function: str
    index_metric: str


class DistanceStrategy(StrategyMixin, enum.Enum):
    """Enumerator of the distance strategies supported by SereneDB.

    Each strategy bundles the query operator, the named distance function and the IVF
    index ``metric`` for one distance measure: Euclidean (L2), cosine, inner product,
    and Manhattan (L1).
    """

    EUCLIDEAN = "<->", "l2_distance", "l2"
    COSINE_DISTANCE = "<=>", "cosine_distance", "cosine"
    # ``negative_inner_product`` (not raw ``inner_product``) is the function form of the
    # ``<#>`` operator: it returns -IP, so smaller = more similar (a true distance,
    # consistent with ORDER BY ``<#>`` and with the max-inner-product relevance score).
    INNER_PRODUCT = "<#>", "negative_inner_product", "ip"
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

    The index name is not configurable here — the store derives one name per collection
    from the table (see ``SereneDBVectorStore``), so a collection has exactly one index.

    Attributes:
        index_type (str): A string identifying the type of index. Defaults to "base".
        distance_strategy (DistanceStrategy): The strategy used to calculate distances
            between vectors in the index. Defaults to DistanceStrategy.COSINE_DISTANCE.
        partial_indexes (Optional[list[str]]): Not supported; a non-empty value raises
            at build time.
    """

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
class IVFIndex(BaseIndex):
    """IVF inverted-index configuration.

    The ``metric`` is derived from :attr:`distance_strategy`. Every other option is
    optional and, when left ``None``, is omitted from the DDL so SereneDB chooses its
    default (e.g. ``nlist`` auto-scales with the row count).

    Attributes:
        nlist: Number of IVF cluster lists (``>= 1``). Mutually exclusive with
            ``nlist_factor``.
        nlist_factor: Scales the auto-chosen list count
            (``nlist = round(nlist_factor * sqrt(rows))``). Mutually exclusive with
            ``nlist``.
        quant: Vector quantization — one of ``"sq8"``, ``"sq4"``, ``"pq"``,
            ``"rabitq"``, ``"none"``. Quantized modes require metric ``l2`` or ``ip``.
        pq_m: Number of PQ sub-quantizers (must divide the vector dimension); valid only
            with ``quant="pq"``.
        rabitq_bits: RaBitQ bit count (1-9); valid only with ``quant="rabitq"``.

    Option combinations are validated by the database at CREATE INDEX time, not here.
    """

    index_type: str = "ivf"
    nlist: Optional[int] = None
    nlist_factor: Optional[float] = None
    quant: Optional[str] = None
    pq_m: Optional[int] = None
    rabitq_bits: Optional[int] = None

    def index_options(self) -> str:
        """Return the ``ivf`` operator-class spec for the embedding column.

        Only the options that are set are emitted, e.g. ``ivf (metric = 'cosine')`` or
        ``ivf (metric = 'l2', quant = 'sq8', nlist = 100)``.
        """
        opts = [f"metric = '{self.get_index_metric()}'"]
        if self.quant is not None:
            opts.append(f"quant = '{self.quant}'")
        if self.pq_m is not None:
            opts.append(f"pq_m = {self.pq_m}")
        if self.rabitq_bits is not None:
            opts.append(f"rabitq_bits = {self.rabitq_bits}")
        if self.nlist is not None:
            opts.append(f"nlist = {self.nlist}")
        if self.nlist_factor is not None:
            opts.append(f"nlist_factor = {self.nlist_factor}")
        return f"ivf ({', '.join(opts)})"


@dataclass
class IVFQueryOptions(QueryOptions):
    """Per-session IVF search tuning.

    ``nprobe`` is the number of IVF cluster lists scanned per query (higher = better
    recall, slower), applied as the ``sdb_nprobe`` session setting. ``rerank_factor``
    sizes the exact-distance rerank pool for a *quantized* index (``pool =
    rerank_factor * k``; ``0`` disables reranking), applied as ``sdb_rerank_factor``.
    A field left ``None`` is omitted so SereneDB's own default applies.
    """

    nprobe: Optional[int] = None
    rerank_factor: Optional[int] = None

    def to_parameter(self) -> list[str]:
        """Return ``SET LOCAL`` bodies for the options that were set (empty if none)."""
        params: list[str] = []
        if self.nprobe is not None:
            params.append(f"sdb_nprobe = {self.nprobe}")
        if self.rerank_factor is not None:
            params.append(f"sdb_rerank_factor = {self.rerank_factor}")
        return params

    def to_string(self) -> str:
        """Convert index attributes to string."""
        warnings.warn(
            "to_string is deprecated, use to_parameter instead.",
            DeprecationWarning,
        )
        return "; ".join(self.to_parameter())


@dataclass
class MetadataColumnIndex:
    """An explicit (typed) metadata column to add to the inverted index.

    ``dictionary=None`` indexes the column verbatim (one token per value), which is what
    lets plain ``=`` / ``IN`` / range filters be served by the index scan. Attaching a
    dictionary analyzes the column for full-text instead and defeats plain-equality
    pushdown (such queries then require the ``@@`` operator).
    """

    name: str
    dictionary: Optional[str] = None


@dataclass
class JsonFieldIndex:
    """A JSON metadata sub-field to add to the inverted index.

    ``field`` is the key, or dotted path (e.g. ``"attrs.brand"``), inside the JSON
    metadata column. ``data_type`` is the SQL type the sub-field is indexed/queried as
    (``"TEXT"``, ``"INTEGER"``, ``"DOUBLE"``, ``"BOOLEAN"``, ``"DATE"``, ...); it is NOT
    validated here — an unsupported type surfaces the database's error at CREATE INDEX
    time. The value is extracted with ``->>``; for non-``TEXT`` types a ``::data_type``
    cast is applied to both the index entry and the query so the typed comparison pushes
    into the index scan correctly (see :func:`build_json_field_selector`).
    """

    field: str
    data_type: str
    dictionary: Optional[str] = None


@dataclass
class MetadataIndexConfig:
    """Which metadata to add to the inverted index so filters are index-covered.

    ``columns=None`` (the default) indexes *all* declared metadata columns verbatim;
    pass an explicit list to index a subset (each optionally with a dictionary), or an
    empty list to index none. ``json_fields`` adds JSON metadata sub-fields (none by
    default).
    """

    columns: Optional[list[MetadataColumnIndex]] = None
    json_fields: list[JsonFieldIndex] = field(default_factory=list)


def _quote_ident(name: str) -> str:
    """Escape and double-quote a SQL identifier."""
    return '"' + name.replace('"', '""') + '"'


# Base SQL types SereneDB's inverted index accepts as a verbatim scalar member (verified
# on serenedb 26.06.3). Used ONLY as a safety net for the default "index every metadata
# column" case: an auto-selected column whose type is not in this set is silently skipped
# so the auto-built index can't fail at CREATE INDEX. Explicitly configured columns are
# NOT filtered — there the database decides. Notably unindexable: NUMERIC/DECIMAL/HUGEINT,
# UUID, INTERVAL, VARIANT.
INDEXABLE_METADATA_TYPES = frozenset(
    {
        "TEXT", "VARCHAR", "CHAR", "CHARACTER", "BPCHAR", "STRING",
        "INT", "INTEGER", "INT2", "INT4", "INT8", "BIGINT", "SMALLINT", "TINYINT",
        "BOOL", "BOOLEAN",
        "REAL", "FLOAT", "FLOAT4", "FLOAT8", "DOUBLE",
        "DATE", "TIME", "TIMETZ", "TIMESTAMP", "TIMESTAMPTZ",
        "BYTEA", "BLOB",
        "JSON",
    }
)  # fmt: skip


def _normalize_sql_type(sql_type: str) -> str:
    """Reduce a declared/reported SQL type to its leading base-type keyword.

    Drops length/precision (``VARCHAR(50)`` → ``VARCHAR``) and trailing modifiers
    (``DOUBLE PRECISION`` → ``DOUBLE``, ``timestamp without time zone`` → ``TIMESTAMP``).
    """
    token = sql_type.strip().upper().split("(", 1)[0].split()
    return token[0] if token else ""


def _is_indexable_metadata_type(sql_type: str) -> bool:
    """Whether a metadata column of this SQL type can join the inverted index verbatim."""
    return _normalize_sql_type(sql_type) in INDEXABLE_METADATA_TYPES


def build_json_field_selector(json_column: str, field: str, data_type: str) -> str:
    """Build the JSON sub-field extraction expression for indexing or querying.

    The index expression and the query expression MUST be byte-identical — same arrow
    *and* same cast — for SereneDB to serve the predicate from the index scan, so both
    the CREATE INDEX entry and the WHERE clause go through this one function.

    The value is extracted with the ``->>`` (text) leaf arrow. For non-``TEXT`` types a
    ``::data_type`` cast is appended on *both* sides: the index then stores typed
    tokens, so a pushed-down range/equality on the cast value is correct. (Indexing the
    bare extraction and casting only in the query pushes down but returns wrong results
    on SereneDB, because the index compares the raw string tokens against the cast bound.)
    """
    parts = field.split(".")
    is_text = data_type.strip().upper() == "TEXT"
    expr = _quote_ident(json_column)
    for i, part in enumerate(parts):
        leaf = i == len(parts) - 1
        arrow = "->>" if leaf else "->"
        expr += f"{arrow}'{part}'"
    expr = f"({expr})"
    if not is_text:
        expr = f"{expr}::{data_type}"
    return expr


def build_dictionary_ddl(
    schema_name: str, dictionary_name: str, options: str, *, if_not_exists: bool = True
) -> str:
    """CREATE TEXT SEARCH DICTIONARY statement (for the full-text/hybrid analyzer).

    Created in the table's schema so the hybrid index (which resolves the dictionary
    opclass in its own schema) can find it — a bare/other-schema dictionary is invisible
    to an index in a non-public schema.
    """
    ine = "IF NOT EXISTS " if if_not_exists else ""
    qualified = f"{_quote_ident(schema_name)}.{_quote_ident(dictionary_name)}"
    return f"CREATE TEXT SEARCH DICTIONARY {ine}{qualified} ({options});"


def build_metadata_index_entries(
    metadata_index: Optional[MetadataIndexConfig],
    *,
    metadata_json_column: Optional[str],
    metadata_columns: Mapping[str, Optional[str]],
) -> list[str]:
    """Render the extra ``USING inverted (...)`` entries for metadata columns + JSON.

    ``metadata_columns`` maps each metadata column name to its SQL type (``None`` if
    unknown). ``metadata_index=None`` indexes all of them verbatim (no JSON). Column
    entries with no dictionary are verbatim (exact/range match); JSON entries use
    :func:`build_json_field_selector` so they byte-match the query side.

    When the column set is auto-derived (``metadata_index`` is ``None`` or its ``columns``
    is ``None``), a column whose (known) type is not inverted-indexable is silently
    skipped so the auto-built index can't fail at CREATE INDEX. Explicitly configured
    columns are never filtered — there the database decides (an unindexable entry
    surfaces its DDL error).
    """

    def _auto_cols() -> list[MetadataColumnIndex]:
        return [
            MetadataColumnIndex(name=name)
            for name, sql_type in metadata_columns.items()
            if sql_type is None or _is_indexable_metadata_type(sql_type)
        ]

    if metadata_index is None:
        col_specs: list[MetadataColumnIndex] = _auto_cols()
        json_specs: list[JsonFieldIndex] = []
    else:
        col_specs = (
            _auto_cols() if metadata_index.columns is None else metadata_index.columns
        )
        json_specs = metadata_index.json_fields

    entries: list[str] = []
    for col in col_specs:
        entry = _quote_ident(col.name)
        if col.dictionary:
            entry += f" {_quote_ident(col.dictionary)}"
        entries.append(entry)

    if json_specs and not metadata_json_column:
        raise ValueError(
            "Cannot index JSON metadata fields: the store has no JSON metadata column "
            "(store_metadata=False)."
        )
    for jf in json_specs:
        assert metadata_json_column is not None  # guaranteed by the check above
        entry = build_json_field_selector(metadata_json_column, jf.field, jf.data_type)
        if jf.dictionary:
            entry += f" {_quote_ident(jf.dictionary)}"
        entries.append(entry)
    return entries


def build_vector_index_ddl(
    *,
    schema_name: str,
    table_name: str,
    embedding_column: str,
    index_name: str,
    vector_opclass: str,
    metadata_entries: Optional[list[str]] = None,
    if_not_exists: bool = False,
) -> str:
    """CREATE INDEX for a vector inverted index on the embedding column.

    ``vector_opclass`` is the operator-class spec produced by :meth:`IVFIndex.index_options`,
    e.g. ``ivf (metric = 'cosine', nlist = 100)``. ``metadata_entries`` (from
    :func:`build_metadata_index_entries`) are appended so metadata filters can be served
    by the index scan.
    """
    ine = "IF NOT EXISTS " if if_not_exists else ""
    columns = f"{_quote_ident(embedding_column)} {vector_opclass}"
    if metadata_entries:
        columns += ", " + ", ".join(metadata_entries)
    return (
        f"CREATE INDEX {ine}{_quote_ident(index_name)} "
        f"ON {_quote_ident(schema_name)}.{_quote_ident(table_name)} "
        f"USING inverted ({columns});"
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
    vector_opclass: str,
    metadata_entries: Optional[list[str]] = None,
    if_not_exists: bool = True,
) -> str:
    """CREATE INDEX for one combined inverted index covering the content column
    (analyzed for BM25 full-text) and the embedding column (vector ANN), storing the id
    via ``INCLUDE`` so the lexical branch can return it. ``metadata_entries`` (from
    :func:`build_metadata_index_entries`) are appended so metadata filters can be served
    by the index scan.
    """
    ine = "IF NOT EXISTS " if if_not_exists else ""
    columns = (
        f"{_quote_ident(content_column)} {_quote_ident(dictionary_name)}, "
        f"{_quote_ident(embedding_column)} {vector_opclass}"
    )
    if metadata_entries:
        columns += ", " + ", ".join(metadata_entries)
    return (
        f"CREATE INDEX {ine}{_quote_ident(index_name)} "
        f"ON {_quote_ident(schema_name)}.{_quote_ident(table_name)} "
        f"USING inverted ({columns}) "
        f"INCLUDE ({_quote_ident(id_column)});"
    )
