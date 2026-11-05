"""Index classes for adding vector indexes on the SereneDBVectorStore.

SereneDB provides approximate-nearest-neighbor search through an **inverted index**
with an ``ivf`` or ``hnsw`` operator class on a fixed-size ``FLOAT[N]`` column::

    CREATE INDEX idx ON tbl USING inverted (embedding ivf (metric = 'cosine', quant = 'sq8'))
    CREATE INDEX idx ON tbl USING inverted (embedding hnsw (metric = 'cosine', m = 32))

:class:`IVFIndex` and :class:`HNSWIndex` configure the two index types. The distance
*metric* is required and must match the operator used at query time for the index to
accelerate the search; optional quantization (``quant``) trades recall for memory/speed.
A collection has one vector index at a time, of either type. See
https://serenedb.com/docs/sql/indexes/inverted/vector-search.
"""

import enum
import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar, Optional


@dataclass
class StrategyMixin:
    """Bundles the SQL forms tied to a distance strategy.

    Attributes:
        operator: The infix distance operator used in ``ORDER BY`` (e.g. ``<=>``).
        search_function: The named scalar distance function used to compute the
            score column (e.g. ``cosine_distance``).
        index_metric: The ``metric`` value the vector index must be built with for the
            operator to accelerate (one of ``l2``, ``cosine``, ``ip``, ``l1``).
    """

    operator: str
    search_function: str
    index_metric: str


class DistanceStrategy(StrategyMixin, enum.Enum):
    """Enumerator of the distance strategies supported by SereneDB.

    Each strategy bundles the query operator, the named distance function and the vector
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
class QueryOptions(ABC):
    """Per-query search settings, applied as ``SET LOCAL`` before each search.

    ``index_type`` names the vector index type the settings tune (matching
    :attr:`BaseIndex.index_type`). The store refuses to search a collection whose index
    is of a different type with them, since SereneDB would silently ignore the
    settings. ``None`` means the settings apply to any index type.
    """

    index_type: ClassVar[Optional[str]] = None

    @abstractmethod
    def to_parameter(self) -> list[str]:
        """Convert index attributes to list of configurations."""
        raise NotImplementedError("to_parameter method must be implemented by subclass")

    @abstractmethod
    def to_string(self) -> str:
        """Convert index attributes to string."""
        raise NotImplementedError("to_string method must be implemented by subclass")


def _quantization_options(
    quant: Optional[str],
    pq_m: Optional[int],
    nb_bits: Optional[int],
    query_bits: Optional[int],
    compression: Optional[bool],
) -> list[str]:
    """Render the quantization options shared by the ``ivf`` and ``hnsw`` opclasses."""
    opts: list[str] = []
    if quant is not None:
        opts.append(f"quant = '{quant}'")
    if pq_m is not None:
        opts.append(f"pq_m = {pq_m}")
    if nb_bits is not None:
        opts.append(f"nb_bits = {nb_bits}")
    if query_bits is not None:
        opts.append(f"query_bits = {query_bits}")
    if compression is not None:
        opts.append(f"compression = {str(compression).lower()}")
    return opts


@dataclass
class IVFIndex(BaseIndex):
    """IVF inverted-index configuration.

    The ``metric`` is derived from :attr:`distance_strategy`. Every other option is
    optional and, when left ``None``, is omitted from the DDL so SereneDB chooses its
    default (the number of cluster lists is auto-scaled with the row count, and vectors
    are quantized with ``sq8`` for the ``l2``, ``ip`` and ``cosine`` metrics).

    Attributes:
        quant: Vector quantization, e.g. ``"sq8"``, ``"sq4"``, ``"usq8"``, ``"usq4"``,
            ``"fp16"``, ``"bf16"``, ``"pq"``, ``"rabitq"``, ``"tq"``, ``"bq"``,
            ``"osq"`` or ``"none"`` (full precision). See the SereneDB vector search
            documentation for the full list and which metrics each supports.
        pq_m: Number of PQ sub-quantizers (must divide the vector dimension); valid only
            with ``quant="pq"``.
        rabitq_bits: Deprecated; use ``nb_bits``. Sent to the server as ``nb_bits``.
        nb_bits: Bits per stored code for the quantizers that take a bit count
            (``rabitq``, ``tq``, ``bq``, ``osq``).
        query_bits: Bits used to quantize the query vector; valid only with
            ``quant="bq"`` or ``quant="osq"``.
        compression: When ``False``, store the index vectors uncompressed (default is
            ``True`` in the database).

    Option combinations are validated by the database at CREATE INDEX time, not here.
    """

    index_type: str = "ivf"
    quant: Optional[str] = None
    pq_m: Optional[int] = None
    rabitq_bits: Optional[int] = None
    nb_bits: Optional[int] = None
    query_bits: Optional[int] = None
    compression: Optional[bool] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.rabitq_bits is not None:
            warnings.warn(
                "IVFIndex(rabitq_bits=...) is deprecated and will be removed in a "
                "future release; use nb_bits instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            if self.nb_bits is not None and self.nb_bits != self.rabitq_bits:
                raise ValueError(
                    f"IVFIndex got rabitq_bits={self.rabitq_bits} and "
                    f"nb_bits={self.nb_bits}; they are the same setting. Set only "
                    "nb_bits."
                )

    def index_options(self) -> str:
        """Return the ``ivf`` operator-class spec for the embedding column.

        Only the options that are set are emitted, e.g. ``ivf (metric = 'cosine')`` or
        ``ivf (metric = 'l2', quant = 'sq8', compression = false)``.
        """
        opts = [f"metric = '{self.get_index_metric()}'"]
        nb_bits = self.nb_bits if self.nb_bits is not None else self.rabitq_bits
        opts += _quantization_options(
            self.quant, self.pq_m, nb_bits, self.query_bits, self.compression
        )
        return f"ivf ({', '.join(opts)})"


@dataclass
class HNSWIndex(BaseIndex):
    """HNSW inverted-index configuration.

    HNSW searches a navigable graph of the vectors. Metadata filters are applied during
    the graph walk, so filtered searches keep returning ``k`` results. The ``metric`` is
    derived from :attr:`distance_strategy`; every other option is optional and, when
    left ``None``, is omitted from the DDL so SereneDB chooses its default.

    Attributes:
        m: Maximum number of graph neighbours per vector (SereneDB default 32, at least
            2). Larger values improve recall and use more memory.
        ef_construction: Candidate-list size while building the graph (SereneDB default
            200, at least ``m``). Larger values build a better graph, more slowly.
        quant: Vector quantization, as for :class:`IVFIndex` (SereneDB default
            ``"sq8"``; the ``l1`` metric supports only ``"none"``).
        pq_m: Number of PQ sub-quantizers; valid only with ``quant="pq"``.
        nb_bits: Bits per stored code for the quantizers that take a bit count.
        query_bits: Bits used to quantize the query vector; valid only with
            ``quant="bq"`` or ``quant="osq"``.
        compression: When ``False``, store the index vectors uncompressed.

    Option combinations are validated by the database at CREATE INDEX time, not here.
    """

    index_type: str = "hnsw"
    m: Optional[int] = None
    ef_construction: Optional[int] = None
    quant: Optional[str] = None
    pq_m: Optional[int] = None
    nb_bits: Optional[int] = None
    query_bits: Optional[int] = None
    compression: Optional[bool] = None

    def index_options(self) -> str:
        """Return the ``hnsw`` operator-class spec for the embedding column.

        Only the options that are set are emitted, e.g. ``hnsw (metric = 'cosine')`` or
        ``hnsw (metric = 'l2', m = 16, ef_construction = 64)``.
        """
        opts = [f"metric = '{self.get_index_metric()}'"]
        if self.m is not None:
            opts.append(f"m = {self.m}")
        if self.ef_construction is not None:
            opts.append(f"ef_construction = {self.ef_construction}")
        opts += _quantization_options(
            self.quant, self.pq_m, self.nb_bits, self.query_bits, self.compression
        )
        return f"hnsw ({', '.join(opts)})"


@dataclass
class IVFQueryOptions(QueryOptions):
    """Per-session IVF search tuning.

    ``nprobe`` is the number of IVF cluster lists scanned per query (higher = better
    recall, slower), applied as the ``sdb_ivf_search_nprobe`` session setting.
    ``rescore_width`` applies to a *quantized* index only, as ``sdb_ann_rescore_width``:
    the search runs on the quantized codes, and a candidate is re-read at full precision
    while its code score plus ``rescore_width`` times the code's estimated error could
    still beat the k-th best exact score. Larger values rescore more candidates and miss
    fewer neighbours; ``0`` answers from the codes alone; ``-1`` lets SereneDB pick a
    width for the index's quantizer. A field left ``None`` is omitted so SereneDB's own
    default applies.
    """

    index_type: ClassVar[Optional[str]] = "ivf"

    nprobe: Optional[int] = None
    rescore_width: Optional[float] = None

    def to_parameter(self) -> list[str]:
        """Return ``SET LOCAL`` bodies for the options that were set (empty if none)."""
        params: list[str] = []
        if self.nprobe is not None:
            params.append(f"sdb_ivf_search_nprobe = {self.nprobe}")
        if self.rescore_width is not None:
            params.append(f"sdb_ann_rescore_width = {self.rescore_width}")
        return params

    def to_string(self) -> str:
        """Convert index attributes to string."""
        warnings.warn(
            "to_string is deprecated, use to_parameter instead.",
            DeprecationWarning,
        )
        return "; ".join(self.to_parameter())


@dataclass
class HNSWQueryOptions(QueryOptions):
    """Per-session HNSW search tuning.

    ``ef_search`` is the candidate-list size of the graph search (higher = better
    recall, slower), applied as the ``sdb_hnsw_ef_search`` session setting; ``-1`` lets
    SereneDB size it from ``k``. ``rescore_width`` applies to a *quantized* index only,
    as ``sdb_ann_rescore_width``, with the same meaning as in :class:`IVFQueryOptions`.
    A field left ``None`` is omitted so SereneDB's own default applies.
    """

    index_type: ClassVar[Optional[str]] = "hnsw"

    ef_search: Optional[int] = None
    rescore_width: Optional[float] = None

    def to_parameter(self) -> list[str]:
        """Return ``SET LOCAL`` bodies for the options that were set (empty if none)."""
        params: list[str] = []
        if self.ef_search is not None:
            params.append(f"sdb_hnsw_ef_search = {self.ef_search}")
        if self.rescore_width is not None:
            params.append(f"sdb_ann_rescore_width = {self.rescore_width}")
        return params

    def to_string(self) -> str:
        """Convert index attributes to string."""
        warnings.warn(
            "to_string is deprecated, use to_parameter instead.",
            DeprecationWarning,
        )
        return "; ".join(self.to_parameter())


def vector_index_type_from_indexdef(
    indexdef: str, embedding_column: str
) -> Optional[str]:
    """Return the vector opclass (``"ivf"``/``"hnsw"``) applied to ``embedding_column``.

    ``indexdef`` is the ``pg_indexes.indexdef`` text, in which SereneDB prints each
    indexed column followed by its opclass, e.g. ``(embedding hnsw (m='16', metric=l2),
    id)``. Returns ``None`` when the embedding column carries neither opclass.
    """
    column = re.escape(embedding_column)
    match = re.search(
        rf'(?:"{column}"|(?<![\w"]){column})\s+(ivf|hnsw)\s*\(',
        indexdef,
        flags=re.IGNORECASE,
    )
    return match.group(1).lower() if match else None


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
    schema_name: str,
    dictionary_name: str,
    expression: str,
    *,
    if_not_exists: bool = True,
) -> str:
    """CREATE TEXT SEARCH DICTIONARY statement (for the full-text/hybrid analyzer).

    Created in the table's schema so the hybrid index (which resolves the dictionary
    opclass in its own schema) can find it — a bare/other-schema dictionary is invisible
    to an index in a non-public schema. ``expression`` is the analyzer expression
    (``AS <expression>``).
    """
    ine = "IF NOT EXISTS " if if_not_exists else ""
    qualified = f"{_quote_ident(schema_name)}.{_quote_ident(dictionary_name)}"
    return f"CREATE TEXT SEARCH DICTIONARY {ine}{qualified} AS {expression};"


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

    ``vector_opclass`` is the operator-class spec produced by
    :meth:`BaseIndex.index_options`, e.g. ``ivf (metric = 'cosine', quant = 'sq8')`` or
    ``hnsw (metric = 'cosine', m = 32)``. ``metadata_entries`` (from
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
