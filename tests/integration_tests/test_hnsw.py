"""HNSW vector index support, and the check that query options match the index type.

Set ``SERENEDB_CONNINFO`` to point at a running instance (default targets port 7890).
Live tests are skipped if the database is unreachable.
"""

import hashlib
import math
import os
import uuid

import psycopg
import pytest
from langchain_core.embeddings import Embeddings

from langchain_serenedb import (
    Column,
    DistanceStrategy,
    HNSWIndex,
    HNSWQueryOptions,
    HybridSearchConfig,
    IVFIndex,
    IVFQueryOptions,
    SereneDBEngine,
    SereneDBVectorStore,
)
from langchain_serenedb.indexes import vector_index_type_from_indexdef

CONNINFO = os.environ.get(
    "SERENEDB_CONNINFO", "host=127.0.0.1 port=7890 user=postgres dbname=postgres"
)
DIM = 8
L2 = DistanceStrategy.EUCLIDEAN
TEXTS = ["red", "green", "blue", "cyan", "magenta", "yellow"]


def _db_available() -> bool:
    try:
        with psycopg.connect(CONNINFO, connect_timeout=3):
            return True
    except Exception:
        return False


requires_db = pytest.mark.skipif(
    not _db_available(), reason=f"SereneDB not reachable at {CONNINFO!r}"
)


class DetEmb(Embeddings):
    """Deterministic per-text embedding so an exact-text query has distance ~0."""

    def _vec(self, text: str) -> list[float]:
        out = [
            hashlib.sha256(f"{i}:{text}".encode()).digest()[0] / 255.0
            for i in range(DIM)
        ]
        n = math.sqrt(sum(x * x for x in out)) or 1.0
        return [x / n for x in out]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


# -- no database needed ----------------------------------------------------------------


def test_hnsw_index_options_string():
    """HNSWIndex emits only the options that are set (unset ones defer to the DB)."""
    assert HNSWIndex().index_options() == "hnsw (metric = 'cosine')"
    assert (
        HNSWIndex(
            distance_strategy=L2,
            m=16,
            ef_construction=64,
            quant="sq8",
            compression=False,
        ).index_options()
        == "hnsw (metric = 'l2', m = 16, ef_construction = 64, quant = 'sq8', "
        "compression = false)"
    )


def test_ivf_index_bit_options_string():
    assert (
        IVFIndex(
            distance_strategy=L2, quant="bq", nb_bits=1, query_bits=4
        ).index_options()
        == "ivf (metric = 'l2', quant = 'bq', nb_bits = 1, query_bits = 4)"
    )


def test_fractional_nb_bits_string():
    assert "nb_bits = 1.5" in HNSWIndex(quant="tq", nb_bits=1.5).index_options()
    assert (
        "nb_bits = 4,"
        in IVFIndex(quant="tq", nb_bits=4.0, compression=True).index_options()
    )


def test_ivf_rabitq_bits_is_deprecated_alias_of_nb_bits():
    with pytest.warns(DeprecationWarning, match="use nb_bits instead"):
        index = IVFIndex(distance_strategy=L2, quant="rabitq", rabitq_bits=4)
    assert index.index_options() == "ivf (metric = 'l2', quant = 'rabitq', nb_bits = 4)"

    with pytest.warns(DeprecationWarning):
        same = IVFIndex(quant="rabitq", rabitq_bits=4, nb_bits=4)
    assert "nb_bits = 4" in same.index_options()

    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="nb_bits"):
        IVFIndex(quant="rabitq", rabitq_bits=4, nb_bits=2)


def test_hnsw_query_options_to_parameter():
    assert HNSWQueryOptions().to_parameter() == []
    assert HNSWQueryOptions(ef_search=128, rescore_width=3).to_parameter() == [
        "sdb_hnsw_ef_search = 128",
        "sdb_ann_rescore_width = 3",
    ]


def test_query_options_declare_index_type():
    assert IVFQueryOptions.index_type == "ivf"
    assert HNSWQueryOptions.index_type == "hnsw"


@pytest.mark.parametrize(
    "indexdef,expected",
    [
        (
            "CREATE INDEX i ON public.t USING inverted (embedding hnsw (m='16', "
            "metric=l2), id) WITH (row_group_size='122880')",
            "hnsw",
        ),
        ('CREATE INDEX i ON t USING inverted ("embedding" ivf (metric=cosine))', "ivf"),
        # Another column's opclass is not the embedding's.
        (
            "CREATE INDEX i ON t USING inverted (my_embedding hnsw (m=1), embedding ivf ())",
            "ivf",
        ),
        ("CREATE INDEX i ON t USING inverted (content dict, id)", None),
    ],
)
def test_vector_index_type_from_indexdef(indexdef, expected):
    assert vector_index_type_from_indexdef(indexdef, "embedding") == expected


# -- live ------------------------------------------------------------------------------


@pytest.fixture
def engine():
    eng = SereneDBEngine.from_connection_string(CONNINFO)
    yield eng
    eng.close()


@pytest.fixture
def table(engine):
    name = f"lc_hnsw_{uuid.uuid4().hex[:8]}"
    engine.init_vectorstore_table(
        name, DIM, overwrite_existing=True, metadata_columns=[Column("tag", "TEXT")]
    )
    yield name
    engine.drop_table(name)


def _store(engine, table, **kwargs) -> SereneDBVectorStore:
    return SereneDBVectorStore.create_sync(
        engine,
        DetEmb(),
        table,
        distance_strategy=L2,
        metadata_columns=["tag"],
        **kwargs,
    )


def _fill(store: SereneDBVectorStore) -> None:
    store.add_texts(
        TEXTS, metadatas=[{"tag": "even" if i % 2 == 0 else "odd"} for i in range(6)]
    )


def _index_def(engine, table) -> str:
    rows = engine._run_as_sync(
        engine._afetch(
            "SELECT indexdef FROM pg_indexes "
            "WHERE tablename = %(t)s AND indexname = %(i)s",
            {"t": table, "i": table + "langchainvectorindex"},
        )
    )
    assert len(rows) == 1
    return rows[0]["indexdef"]


@requires_db
def test_hnsw_index_search_and_filter(engine, table):
    vs = _store(engine, table)
    _fill(vs)
    vs.apply_vector_index(HNSWIndex(distance_strategy=L2, m=16, ef_construction=64))
    assert vs.is_valid_index() is True
    assert " hnsw (" in _index_def(engine, table)

    assert vs.similarity_search("green", k=1)[0].page_content == "green"
    res = vs.similarity_search("green", k=3, filter={"tag": "even"})
    assert len(res) == 3
    assert all(d.metadata["tag"] == "even" for d in res)


@requires_db
def test_init_vectorstore_table_with_hnsw(engine):
    name = f"lc_hnsw_{uuid.uuid4().hex[:8]}"
    engine.init_vectorstore_table(
        name, DIM, overwrite_existing=True, vector_index=HNSWIndex(distance_strategy=L2)
    )
    try:
        assert " hnsw (" in _index_def(engine, name)
        vs = SereneDBVectorStore.create_sync(
            engine,
            DetEmb(),
            name,
            distance_strategy=L2,
            index_query_options=HNSWQueryOptions(ef_search=32),
        )
        vs.add_texts(TEXTS)
        assert vs.similarity_search("blue", k=1)[0].page_content == "blue"
    finally:
        engine.drop_table(name)


@requires_db
def test_hybrid_index_with_hnsw(engine, table):
    vs = _store(
        engine,
        table,
        hybrid_search_config=HybridSearchConfig(primary_top_k=10, secondary_top_k=10),
    )
    vs.add_texts(["the quick brown fox", "a lazy brown dog", "quantum physics"])
    vs.apply_hybrid_search_index(index=HNSWIndex(distance_strategy=L2))
    assert " hnsw (" in _index_def(engine, table)

    res = vs.similarity_search("brown", k=3)
    assert len(res) == 3
    assert sum("brown" in d.page_content for d in res) >= 2


@requires_db
def test_hnsw_query_options_reach_db(engine, table):
    """Every HNSWQueryOptions setting is accepted by the server and applied per query.

    A search with all of them set succeeds only if the server recognizes every setting
    name. An out-of-range value for each then makes the search raise "invalid value
    for parameter", which can only happen if that setting's prelude reached the server.
    """
    base = _store(engine, table)
    _fill(base)
    # rescore_width only matters on a quantized index.
    base.apply_vector_index(HNSWIndex(distance_strategy=L2, quant="sq8"))

    for options in (
        HNSWQueryOptions(ef_search=64, rescore_width=3.0),
        HNSWQueryOptions(ef_search=-1, rescore_width=0),
    ):
        vs = _store(engine, table, index_query_options=options)
        assert vs.similarity_search("green", k=1)[0].page_content == "green", options

    for options, setting in (
        (HNSWQueryOptions(ef_search=0), "sdb_hnsw_ef_search"),
        (HNSWQueryOptions(rescore_width=-2), "sdb_ann_rescore_width"),
    ):
        vs = _store(engine, table, index_query_options=options)
        with pytest.raises(psycopg.Error) as exc:
            vs.similarity_search("green", k=1)
        assert f'invalid value for parameter "{setting}"' in str(exc.value)


@requires_db
@pytest.mark.parametrize(
    "index,options",
    [
        (HNSWIndex(distance_strategy=L2), IVFQueryOptions(nprobe=4)),
        (IVFIndex(distance_strategy=L2), HNSWQueryOptions(ef_search=32)),
    ],
)
def test_query_options_for_other_index_type_raise(engine, table, index, options):
    """Options for one index type are refused on the other (the server ignores them)."""
    base = _store(engine, table)
    _fill(base)
    base.apply_vector_index(index)

    vs = _store(engine, table, index_query_options=options)
    with pytest.raises(ValueError, match=r"tunes .* indexes, but the index"):
        vs.similarity_search("green", k=1)


@requires_db
def test_query_options_without_index_are_not_checked(engine, table):
    """With no index the search is an exact scan, so any index type's options are fine."""
    vs = _store(engine, table, index_query_options=IVFQueryOptions(nprobe=4))
    _fill(vs)
    assert vs.similarity_search("green", k=1)[0].page_content == "green"


@requires_db
def test_switch_ivf_to_hnsw_updates_check(engine, table):
    """Dropping the IVF index and applying HNSW on the same store is seen at once."""
    vs = _store(engine, table, index_query_options=HNSWQueryOptions(ef_search=32))
    _fill(vs)
    vs.apply_vector_index(IVFIndex(distance_strategy=L2))
    with pytest.raises(ValueError, match="is IVF"):
        vs.similarity_search("green", k=1)

    vs.drop_vector_index()
    vs.apply_vector_index(HNSWIndex(distance_strategy=L2))
    assert vs.similarity_search("green", k=1)[0].page_content == "green"


@requires_db
def test_hybrid_apply_over_existing_index_keeps_its_type(engine, table):
    """The hybrid DDL is IF NOT EXISTS: the existing index stays, and the store says so."""
    vs = _store(engine, table, index_query_options=HNSWQueryOptions(ef_search=32))
    _fill(vs)
    vs.apply_vector_index(IVFIndex(distance_strategy=L2))
    vs.apply_hybrid_search_index(index=HNSWIndex(distance_strategy=L2))
    assert " ivf (" in _index_def(engine, table)
    with pytest.raises(ValueError, match="is IVF"):
        vs.similarity_search("green", k=1)
