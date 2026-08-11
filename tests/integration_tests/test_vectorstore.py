"""Round-trip tests for SereneDBVectorStore against a live SereneDB.

Set ``SERENEDB_CONNINFO`` to point at a running instance (default targets port 7890).
Tests are skipped if the database is unreachable.
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
    FusionStrategy,
    HybridSearchConfig,
    IVFIndex,
    IVFQueryOptions,
    JsonFieldIndex,
    MetadataColumnIndex,
    MetadataIndexConfig,
    SereneDBEngine,
    SereneDBVectorStore,
)

CONNINFO = os.environ.get(
    "SERENEDB_CONNINFO", "host=127.0.0.1 port=7890 user=postgres dbname=postgres"
)
DIM = 8


def _db_available() -> bool:
    try:
        with psycopg.connect(CONNINFO, connect_timeout=3):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _db_available(), reason=f"SereneDB not reachable at {CONNINFO!r}"
)


class DetEmb(Embeddings):
    """Deterministic per-text embedding so an exact-text query has distance ~0."""

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        out = [
            hashlib.sha256(f"{i}:{text}".encode()).digest()[0] / 255.0
            for i in range(self.dim)
        ]
        n = math.sqrt(sum(x * x for x in out)) or 1.0
        return [x / n for x in out]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    async def aembed_documents(self, texts):
        return self.embed_documents(texts)

    async def aembed_query(self, text):
        return self.embed_query(text)


@pytest.fixture
def store():
    table = f"lc_test_{uuid.uuid4().hex[:8]}"
    engine = SereneDBEngine.from_connection_string(CONNINFO)
    engine.init_vectorstore_table(
        table,
        DIM,
        overwrite_existing=True,
        metadata_columns=[Column("category", "TEXT")],
    )
    vs = SereneDBVectorStore.create_sync(
        engine,
        DetEmb(),
        table,
        metadata_columns=["category"],
        distance_strategy=DistanceStrategy.COSINE_DISTANCE,
    )
    texts = ["the quick brown fox", "a lazy dog", "quantum physics"]
    metas = [
        {"category": "animal", "n": 1},
        {"category": "animal", "n": 2},
        {"category": "science", "n": 3},
    ]
    ids = vs.add_texts(texts, metadatas=metas)
    yield vs, ids
    engine.drop_table(table)
    engine.close()


def test_dense_search_exact_match(store):
    vs, _ = store
    res = vs.similarity_search("the quick brown fox", k=2)
    assert res[0].page_content == "the quick brown fox"
    assert res[0].metadata == {"category": "animal", "n": 1}


def test_search_with_score(store):
    vs, _ = store
    res = vs.similarity_search_with_score("quantum physics", k=1)
    doc, distance = res[0]
    assert doc.page_content == "quantum physics"
    assert distance < 1e-6


def test_metadata_column_filter(store):
    vs, _ = store
    res = vs.similarity_search(
        "the quick brown fox", k=5, filter={"category": "science"}
    )
    assert [d.page_content for d in res] == ["quantum physics"]


def test_json_numeric_filter(store):
    vs, _ = store
    res = vs.similarity_search("the quick brown fox", k=5, filter={"n": {"$gte": 2}})
    assert sorted(d.page_content for d in res) == ["a lazy dog", "quantum physics"]


def test_in_filter(store):
    vs, _ = store
    res = vs.similarity_search("x", k=5, filter={"category": {"$in": ["animal"]}})
    assert sorted(d.page_content for d in res) == ["a lazy dog", "the quick brown fox"]


def test_and_with_json_filter(store):
    vs, _ = store
    res = vs.similarity_search(
        "x", k=5, filter={"$and": [{"category": "animal"}, {"n": {"$lt": 2}}]}
    )
    assert [d.page_content for d in res] == ["the quick brown fox"]


def test_add_rows_with_missing_metadata_column(store):
    vs, _ = store
    # The batched insert uses one uniform statement, so a row that omits a metadata
    # column must bind NULL for it (not error / not shift columns).
    vs.add_texts(
        ["a red apple", "a blue sky"],
        metadatas=[{"category": "fruit"}, {}],
    )
    only_fruit = [
        d.page_content
        for d in vs.similarity_search("a red apple", k=10, filter={"category": "fruit"})
    ]
    assert "a red apple" in only_fruit
    assert "a blue sky" not in only_fruit  # NULL category excluded by the filter
    # The row with no category is still retrievable without a filter.
    unfiltered = vs.similarity_search("a blue sky", k=10)
    assert any(d.page_content == "a blue sky" for d in unfiltered)


def test_upsert_many_updates_existing_rows(store):
    """Batched add where every id conflicts: each row is updated in place."""
    vs, _ = store
    ids = [str(uuid.uuid4()) for _ in range(3)]
    vs.add_texts(
        ["old a", "old b", "old c"],
        metadatas=[
            {"category": "x", "n": 1},
            {"category": "x", "n": 2},
            {"category": "y", "n": 3},
        ],
        ids=ids,
    )
    # Re-add the SAME ids with new content + metadata (typed column and JSON both change).
    returned = vs.add_texts(
        ["new a", "new b", "new c"],
        metadatas=[
            {"category": "z", "n": 10},
            {"category": "z", "n": 20},
            {"category": "z", "n": 30},
        ],
        ids=ids,
    )
    assert returned == ids
    got = {d.id: d for d in vs.get_by_ids(ids)}
    assert len(got) == 3  # upsert, not duplicate insert
    assert got[ids[0]].page_content == "new a"
    # Both the typed column (category) and the JSON blob (n) are updated.
    assert got[ids[0]].metadata == {"category": "z", "n": 10}
    assert {d.page_content for d in got.values()} == {"new a", "new b", "new c"}


def test_upsert_many_mixed_insert_and_update(store):
    """One batched add mixing existing ids (update) and new ids (insert)."""
    vs, _ = store
    existing = [str(uuid.uuid4()) for _ in range(2)]
    vs.add_texts(
        ["keep 1", "keep 2"],
        metadatas=[{"category": "a"}, {"category": "a"}],
        ids=existing,
    )
    new = [str(uuid.uuid4()) for _ in range(2)]
    vs.add_texts(
        ["upd 1", "upd 2", "ins 1", "ins 2"],
        metadatas=[
            {"category": "b"},
            {"category": "b"},
            {"category": "c"},
            {"category": "c"},
        ],
        ids=existing + new,
    )
    got = {d.id: d for d in vs.get_by_ids(existing + new)}
    assert len(got) == 4
    assert got[existing[0]].page_content == "upd 1"  # existing row updated
    assert got[existing[0]].metadata["category"] == "b"
    assert got[new[0]].page_content == "ins 1"  # new row inserted
    assert got[new[0]].metadata["category"] == "c"


def test_upsert_duplicate_id_within_one_batch(store):
    """The same id twice in a single batch: the last write wins."""
    vs, _ = store
    dup = str(uuid.uuid4())
    other = str(uuid.uuid4())
    vs.add_texts(
        ["first write", "unrelated", "second write"],
        metadatas=[{"category": "a"}, {"category": "b"}, {"category": "c"}],
        ids=[dup, other, dup],
    )
    got = {d.id: d for d in vs.get_by_ids([dup, other])}
    assert len(got) == 2  # dup collapsed to one row
    assert got[dup].page_content == "second write"
    assert got[dup].metadata["category"] == "c"
    assert got[other].page_content == "unrelated"


def test_upsert_interleaved_repeated_ids_within_one_batch(store):
    """Several ids repeated and interleaved in one batch: each ends at its last write."""
    vs, _ = store
    a, b, c = (str(uuid.uuid4()) for _ in range(3))
    # a appears 3x, b 2x, c once, interleaved so ordering (not grouping) decides winners.
    ids = [a, b, a, c, b, a]
    texts = ["a1", "b1", "a2", "c1", "b2", "a3"]
    metas = [
        {"category": "a", "n": 1},
        {"category": "b", "n": 1},
        {"category": "a", "n": 2},
        {"category": "c", "n": 1},
        {"category": "b", "n": 2},
        {"category": "a", "n": 3},
    ]
    vs.add_texts(texts, metadatas=metas, ids=ids)
    got = {d.id: d for d in vs.get_by_ids([a, b, c])}
    assert len(got) == 3  # 6 input rows collapse to 3 distinct ids
    # Each id reflects its LAST occurrence, content + typed column + JSON alike.
    assert got[a].page_content == "a3"
    assert got[a].metadata == {"category": "a", "n": 3}
    assert got[b].page_content == "b2"
    assert got[b].metadata == {"category": "b", "n": 2}
    assert got[c].page_content == "c1"
    assert got[c].metadata == {"category": "c", "n": 1}


def test_get_by_ids(store):
    vs, ids = store
    got = vs.get_by_ids(ids[:2])
    assert {d.id for d in got} == set(ids[:2])


def test_apply_index_and_search(store):
    vs, _ = store
    vs.apply_vector_index(IVFIndex(distance_strategy=DistanceStrategy.COSINE_DISTANCE))
    assert vs.is_valid_index() is True
    res = vs.similarity_search("quantum physics", k=1)
    assert res[0].page_content == "quantum physics"


def test_ivf_index_options_string():
    """IVFIndex emits only the options that are set (unset ones defer to DB auto)."""
    assert IVFIndex().index_options() == "ivf (metric = 'cosine')"
    assert (
        IVFIndex(
            distance_strategy=DistanceStrategy.EUCLIDEAN, quant="sq8", compression=False
        ).index_options()
        == "ivf (metric = 'l2', quant = 'sq8', compression = false)"
    )


def test_ivf_quantized_index_and_search():
    """A quantized IVF index (sq8, l2) builds and serves an exact-match query."""
    table = f"lc_ivfq_{uuid.uuid4().hex[:8]}"
    engine = SereneDBEngine.from_connection_string(CONNINFO)
    engine.init_vectorstore_table(table, DIM, overwrite_existing=True)
    vs = SereneDBVectorStore.create_sync(
        engine, DetEmb(), table, distance_strategy=DistanceStrategy.EUCLIDEAN
    )
    try:
        vs.add_texts(
            ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
        )
        # quant='sq8' requires metric l2/ip; the cluster count is auto-scaled.
        vs.apply_vector_index(
            IVFIndex(distance_strategy=DistanceStrategy.EUCLIDEAN, quant="sq8")
        )
        assert vs.is_valid_index() is True
        res = vs.similarity_search("gamma", k=1)
        assert res[0].page_content == "gamma"
    finally:
        engine.drop_table(table)
        engine.close()


def test_ivf_query_options_to_parameter():
    """IVFQueryOptions emits SET LOCAL bodies only for the fields that are set."""
    assert IVFQueryOptions().to_parameter() == []
    assert IVFQueryOptions(nprobe=10).to_parameter() == ["sdb_nprobe = 10"]
    assert IVFQueryOptions(nprobe=10, rerank_factor=4).to_parameter() == [
        "sdb_nprobe = 10",
        "sdb_rerank_factor = 4",
    ]


def test_ivf_query_options_nprobe_reaches_db():
    """IVFQueryOptions is actually applied to the query as a SET LOCAL prelude.

    Proven behaviorally: a valid nprobe searches fine, while nprobe=0 (which SereneDB
    rejects -- sdb_nprobe must be >= 1) makes the search raise. That error can only occur
    if the ``SET LOCAL sdb_nprobe`` prelude genuinely reached the database.
    """
    table = f"lc_ivfnp_{uuid.uuid4().hex[:8]}"
    engine = SereneDBEngine.from_connection_string(CONNINFO)
    engine.init_vectorstore_table(
        table, DIM, overwrite_existing=True, vector_index=IVFIndex()
    )
    try:
        ok = SereneDBVectorStore.create_sync(
            engine, DetEmb(), table, index_query_options=IVFQueryOptions(nprobe=10)
        )
        ok.add_texts(["red", "green", "blue"])
        assert ok.similarity_search("green", k=1)[0].page_content == "green"

        # A rejected nprobe value must surface -- only possible if the prelude is issued.
        bad = SereneDBVectorStore.create_sync(
            engine, DetEmb(), table, index_query_options=IVFQueryOptions(nprobe=0)
        )
        with pytest.raises(Exception) as exc:
            bad.similarity_search("green", k=1)
        assert "sdb_nprobe" in str(exc.value)
    finally:
        engine.drop_table(table)
        engine.close()


def test_delete(store):
    vs, ids = store
    assert vs.delete(ids=[ids[0]]) is True
    res = vs.similarity_search("the quick brown fox", k=5)
    assert all(d.page_content != "the quick brown fox" for d in res)


def test_mmr(store):
    vs, _ = store
    res = vs.max_marginal_relevance_search("the quick brown fox", k=2, fetch_k=3)
    assert len(res) == 2
    assert any(d.page_content == "the quick brown fox" for d in res)


@pytest.fixture
def hybrid_store():
    table = f"lc_htest_{uuid.uuid4().hex[:8]}"
    engine = SereneDBEngine.from_connection_string(CONNINFO)
    engine.init_vectorstore_table(
        table,
        DIM,
        overwrite_existing=True,
        metadata_columns=[Column("category", "TEXT")],
    )
    cfg = HybridSearchConfig(primary_top_k=10, secondary_top_k=10)
    vs = SereneDBVectorStore.create_sync(
        engine, DetEmb(), table, metadata_columns=["category"], hybrid_search_config=cfg
    )
    vs.add_texts(
        ["the quick brown fox", "a lazy brown dog", "quantum physics"],
        metadatas=[
            {"category": "animal"},
            {"category": "animal"},
            {"category": "science"},
        ],
    )
    vs.apply_hybrid_search_index()
    yield vs
    engine.drop_table(table)
    engine.close()


def test_hybrid_search(hybrid_store):
    res = hybrid_store.similarity_search("brown", k=3)
    assert len(res) == 3
    assert sum("brown" in d.page_content for d in res) >= 2


def test_hybrid_search_with_score(hybrid_store):
    res = hybrid_store.similarity_search_with_score("brown", k=2)
    assert len(res) == 2
    assert all(isinstance(score, float) for _, score in res)


def test_hybrid_filter(hybrid_store):
    res = hybrid_store.similarity_search("brown", k=5, filter={"category": "animal"})
    assert res
    assert all(d.metadata["category"] == "animal" for d in res)


def test_init_creates_vector_index():
    table = f"lc_vidx_{uuid.uuid4().hex[:8]}"
    engine = SereneDBEngine.from_connection_string(CONNINFO)
    engine.init_vectorstore_table(
        table,
        DIM,
        overwrite_existing=True,
        vector_index=IVFIndex(distance_strategy=DistanceStrategy.COSINE_DISTANCE),
    )
    vs = SereneDBVectorStore.create_sync(
        engine, DetEmb(), table, distance_strategy=DistanceStrategy.COSINE_DISTANCE
    )
    try:
        # The ANN index exists immediately, before any rows are added.
        assert vs.is_valid_index() is True
        vs.add_texts(["alpha", "beta", "gamma"])
        res = vs.similarity_search("alpha", k=1)
        assert res[0].page_content == "alpha"
    finally:
        engine.drop_table(table)
        engine.close()


def test_init_creates_hybrid_index():
    table = f"lc_hidx_{uuid.uuid4().hex[:8]}"
    engine = SereneDBEngine.from_connection_string(CONNINFO)
    # Build the combined full-text + vector index together with the table.
    engine.init_vectorstore_table(
        table, DIM, overwrite_existing=True, hybrid_search_config=HybridSearchConfig()
    )
    vs = SereneDBVectorStore.create_sync(
        engine,
        DetEmb(),
        table,
        hybrid_search_config=HybridSearchConfig(primary_top_k=10, secondary_top_k=10),
    )
    try:
        # No apply_hybrid_search_index() call: the index came from init.
        vs.add_texts(["the quick brown fox", "a lazy brown dog", "quantum physics"])
        res = vs.similarity_search("brown", k=3)
        assert len(res) == 3
        assert sum("brown" in d.page_content for d in res) >= 2
    finally:
        engine.drop_table(table)
        engine.close()


def _run_fusion_strategy(fusion: FusionStrategy):
    """Build a hybrid store using the given single-query fusion strategy and search it."""
    table = f"lc_fuse_{uuid.uuid4().hex[:8]}"
    engine = SereneDBEngine.from_connection_string(CONNINFO)
    engine.init_vectorstore_table(table, DIM, overwrite_existing=True)
    cfg = HybridSearchConfig(primary_top_k=10, secondary_top_k=10, fusion=fusion)
    vs = SereneDBVectorStore.create_sync(
        engine, DetEmb(), table, hybrid_search_config=cfg
    )
    vs.add_texts(["the quick brown fox", "a lazy brown dog", "quantum physics"])
    vs.apply_hybrid_search_index()
    try:
        res = vs.similarity_search("brown dog", k=3)
        assert len(res) >= 1
        with_score = vs.similarity_search_with_score("brown dog", k=2)
        assert with_score and all(isinstance(s, float) for _, s in with_score)
    finally:
        engine.drop_table(table)
        engine.close()


def test_rrf_fusion():
    _run_fusion_strategy(FusionStrategy.RRF)


def test_normalized_fusion():
    _run_fusion_strategy(FusionStrategy.NORMALIZED)


def test_weighted_sum_fusion():
    _run_fusion_strategy(FusionStrategy.WEIGHTED_SUM)


def test_vector_search_in_non_public_schema():
    """Dense search must route through the schema-qualified vector index."""
    schema = f"lc_sch_{uuid.uuid4().hex[:8]}"
    table = "vitems"
    engine = SereneDBEngine.from_connection_string(CONNINFO)
    engine._run_as_sync(engine._aexecute(f'CREATE SCHEMA IF NOT EXISTS "{schema}";'))
    engine.init_vectorstore_table(
        table,
        DIM,
        overwrite_existing=True,
        schema_name=schema,
        metadata_columns=[Column("category", "TEXT")],
        vector_index=IVFIndex(distance_strategy=DistanceStrategy.COSINE_DISTANCE),
    )
    vs = SereneDBVectorStore.create_sync(
        engine, DetEmb(), table, schema_name=schema, metadata_columns=["category"]
    )
    try:
        # The vector index lives in the custom schema and must resolve there.
        assert vs.is_valid_index() is True
        vs.add_texts(
            ["the quick brown fox", "a lazy dog", "quantum physics"],
            metadatas=[
                {"category": "animal"},
                {"category": "animal"},
                {"category": "science"},
            ],
        )
        res = vs.similarity_search("the quick brown fox", k=1)
        assert res[0].page_content == "the quick brown fox"
        # Filtered dense search, also from the schema-qualified index relation.
        res = vs.similarity_search("x", k=5, filter={"category": "science"})
        assert [d.page_content for d in res] == ["quantum physics"]
    finally:
        engine.drop_table(table, schema_name=schema)
        engine._run_as_sync(
            engine._aexecute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE;')
        )
        engine.close()


def test_hybrid_search_in_non_public_schema():
    """Hybrid needs the text search dictionary and index both in the custom schema."""
    schema = f"lc_sch_{uuid.uuid4().hex[:8]}"
    table = "hitems"
    engine = SereneDBEngine.from_connection_string(CONNINFO)
    engine._run_as_sync(engine._aexecute(f'CREATE SCHEMA IF NOT EXISTS "{schema}";'))
    engine.init_vectorstore_table(
        table,
        DIM,
        overwrite_existing=True,
        schema_name=schema,
        metadata_columns=[Column("category", "TEXT")],
        hybrid_search_config=HybridSearchConfig(),
    )
    vs = SereneDBVectorStore.create_sync(
        engine,
        DetEmb(),
        table,
        schema_name=schema,
        metadata_columns=["category"],
        hybrid_search_config=HybridSearchConfig(primary_top_k=10, secondary_top_k=10),
    )
    try:
        vs.add_texts(
            ["the quick brown fox", "a lazy brown dog", "quantum physics"],
            metadatas=[
                {"category": "animal"},
                {"category": "animal"},
                {"category": "science"},
            ],
        )
        res = vs.similarity_search("brown", k=3)
        assert len(res) == 3
        assert sum("brown" in d.page_content for d in res) >= 2
        # Filter through the fused (vector + BM25) path.
        res = vs.similarity_search("brown", k=5, filter={"category": "animal"})
        assert res
        assert all(d.metadata["category"] == "animal" for d in res)
    finally:
        engine.drop_table(table, schema_name=schema)
        engine._run_as_sync(
            engine._aexecute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE;')
        )
        engine.close()


def test_init_if_not_exists_is_idempotent():
    """A second init with if_not_exists is a no-op: no error, data + index kept."""
    table = f"lc_ine_{uuid.uuid4().hex[:8]}"
    engine = SereneDBEngine.from_connection_string(CONNINFO)
    engine.init_vectorstore_table(
        table, DIM, if_not_exists=True, vector_index=IVFIndex()
    )
    vs = SereneDBVectorStore.create_sync(engine, DetEmb(), table)
    try:
        vs.add_texts(["alpha", "beta"])
        # Re-init: must not error, must not wipe the rows, must keep the index.
        engine.init_vectorstore_table(
            table, DIM, if_not_exists=True, vector_index=IVFIndex()
        )
        assert vs.is_valid_index() is True
        res = vs.similarity_search("alpha", k=1)
        assert res[0].page_content == "alpha"
        # if_not_exists and overwrite_existing are mutually exclusive.
        with pytest.raises(ValueError):
            engine.init_vectorstore_table(
                table, DIM, overwrite_existing=True, if_not_exists=True
            )
    finally:
        engine.drop_table(table)
        engine.close()


# -- JSON metadata-column filter tests -------------------------------------------------


def _json_matches(vs, flt):
    """page_content set of docs passing ``flt`` (k large enough to return all matches)."""
    return {d.page_content for d in vs.similarity_search("alpha", k=10, filter=flt)}


@pytest.fixture
def json_store():
    """Store with NO typed metadata columns, so every metadata key lives in the JSON blob.

    Row / author / year / score / published / tag:
      alpha   alice 2019 0.9 True  ml
      bravo   bob   2020 0.5 False db
      charlie carol 2021 0.1 True  ml-ops
      delta   alice 2022 0.7 True  (no tag)
      echo    dave  2018 0.3 False search
    """
    table = f"lc_json_{uuid.uuid4().hex[:8]}"
    engine = SereneDBEngine.from_connection_string(CONNINFO)
    engine.init_vectorstore_table(table, DIM, overwrite_existing=True)
    vs = SereneDBVectorStore.create_sync(engine, DetEmb(), table)
    vs.add_texts(
        ["alpha", "bravo", "charlie", "delta", "echo"],
        metadatas=[
            {
                "author": "alice",
                "year": 2019,
                "score": 0.9,
                "published": True,
                "tag": "ml",
            },
            {
                "author": "bob",
                "year": 2020,
                "score": 0.5,
                "published": False,
                "tag": "db",
            },
            {
                "author": "carol",
                "year": 2021,
                "score": 0.1,
                "published": True,
                "tag": "ml-ops",
            },
            {"author": "alice", "year": 2022, "score": 0.7, "published": True},
            {
                "author": "dave",
                "year": 2018,
                "score": 0.3,
                "published": False,
                "tag": "search",
            },
        ],
    )
    yield vs
    engine.drop_table(table)
    engine.close()


def test_json_eq_string(json_store):
    assert _json_matches(json_store, {"author": "alice"}) == {"alpha", "delta"}


def test_json_ne_string(json_store):
    assert _json_matches(json_store, {"author": {"$ne": "alice"}}) == {
        "bravo",
        "charlie",
        "echo",
    }


def test_json_int_comparison_casts(json_store):
    assert _json_matches(json_store, {"year": {"$gt": 2020}}) == {"charlie", "delta"}


def test_json_float_between(json_store):
    assert _json_matches(json_store, {"score": {"$between": [0.4, 0.8]}}) == {
        "bravo",
        "delta",
    }


def test_json_in(json_store):
    assert _json_matches(json_store, {"author": {"$in": ["alice", "bob"]}}) == {
        "alpha",
        "bravo",
        "delta",
    }


def test_json_nin(json_store):
    assert _json_matches(json_store, {"author": {"$nin": ["alice", "bob"]}}) == {
        "charlie",
        "echo",
    }


def test_json_like(json_store):
    assert _json_matches(json_store, {"tag": {"$like": "ml%"}}) == {"alpha", "charlie"}


def test_json_ilike(json_store):
    assert _json_matches(json_store, {"author": {"$ilike": "A%"}}) == {"alpha", "delta"}


def test_json_exists(json_store):
    assert _json_matches(json_store, {"tag": {"$exists": False}}) == {"delta"}
    assert _json_matches(json_store, {"tag": {"$exists": True}}) == {
        "alpha",
        "bravo",
        "charlie",
        "echo",
    }


def test_json_bool(json_store):
    assert _json_matches(json_store, {"published": True}) == {
        "alpha",
        "charlie",
        "delta",
    }


def test_json_implicit_and_multikey(json_store):
    # A multi-key dict is an implicit AND across JSON keys.
    assert _json_matches(json_store, {"author": "alice", "published": True}) == {
        "alpha",
        "delta",
    }


def test_json_logical_operators(json_store):
    assert _json_matches(
        json_store, {"$and": [{"author": "alice"}, {"year": {"$gte": 2020}}]}
    ) == {"delta"}
    assert _json_matches(
        json_store, {"$or": [{"author": "bob"}, {"year": {"$lt": 2019}}]}
    ) == {"bravo", "echo"}
    assert _json_matches(json_store, {"$not": [{"published": True}]}) == {
        "bravo",
        "echo",
    }


# -- combined typed-column + JSON filter tests -----------------------------------------


@pytest.fixture
def mixed_store():
    """Store with a typed ``category`` column PLUS JSON keys (``year``, ``author``).

    Row / category (typed) / year (json) / author (json):
      alpha   animal  2019 alice
      bravo   animal  2021 bob
      charlie science 2020 alice
      delta   science 2022 carol
      echo    animal  2018 dave
    """
    table = f"lc_mixed_{uuid.uuid4().hex[:8]}"
    engine = SereneDBEngine.from_connection_string(CONNINFO)
    engine.init_vectorstore_table(
        table,
        DIM,
        overwrite_existing=True,
        metadata_columns=[Column("category", "TEXT")],
    )
    vs = SereneDBVectorStore.create_sync(
        engine, DetEmb(), table, metadata_columns=["category"]
    )
    vs.add_texts(
        ["alpha", "bravo", "charlie", "delta", "echo"],
        metadatas=[
            {"category": "animal", "year": 2019, "author": "alice"},
            {"category": "animal", "year": 2021, "author": "bob"},
            {"category": "science", "year": 2020, "author": "alice"},
            {"category": "science", "year": 2022, "author": "carol"},
            {"category": "animal", "year": 2018, "author": "dave"},
        ],
    )
    yield vs
    engine.drop_table(table)
    engine.close()


def test_mixed_implicit_and(mixed_store):
    # Implicit AND: typed column (category) + JSON key (year) in one dict.
    assert _json_matches(
        mixed_store, {"category": "animal", "year": {"$gt": 2019}}
    ) == {"bravo"}


def test_mixed_and_typed_and_json(mixed_store):
    assert _json_matches(
        mixed_store, {"$and": [{"category": "science"}, {"author": "alice"}]}
    ) == {"charlie"}


def test_mixed_or_typed_and_json(mixed_store):
    assert _json_matches(
        mixed_store, {"$or": [{"category": "science"}, {"author": "bob"}]}
    ) == {"bravo", "charlie", "delta"}


def test_mixed_typed_and_multiple_json_ops(mixed_store):
    # A typed-column predicate combined with several JSON operators.
    flt = {
        "$and": [
            {"category": "animal"},
            {"year": {"$between": [2019, 2021]}},
            {"author": {"$in": ["alice", "bob"]}},
        ]
    }
    assert _json_matches(mixed_store, flt) == {"alpha", "bravo"}


# -- relevance scores / similarity_score_threshold retriever ---------------------------


def test_relevance_score_fn_selection():
    """Each distance strategy maps to a [0, 1] relevance fn (needed for thresholding)."""
    table = f"lc_relfn_{uuid.uuid4().hex[:8]}"
    engine = SereneDBEngine.from_connection_string(CONNINFO)
    engine.init_vectorstore_table(table, DIM, overwrite_existing=True)

    def fn_for(ds):
        vs = SereneDBVectorStore.create_sync(
            engine, DetEmb(), table, distance_strategy=ds
        )
        return vs._select_relevance_score_fn()

    try:
        # For true distances (cosine/L2/L1), distance 0 == identical -> relevance 1.0.
        assert fn_for(DistanceStrategy.COSINE_DISTANCE)(0.0) == pytest.approx(1.0)
        assert fn_for(DistanceStrategy.EUCLIDEAN)(0.0) == pytest.approx(1.0)
        assert fn_for(DistanceStrategy.MANHATTAN)(0.0) == pytest.approx(1.0)
        # Inner product distance is -IP: a perfect unit match is -1 -> relevance 1.0.
        assert fn_for(DistanceStrategy.INNER_PRODUCT)(-1.0) == pytest.approx(1.0)
    finally:
        engine.drop_table(table)
        engine.close()


def test_similarity_search_with_relevance_scores(store):
    vs, _ = store
    res = vs.similarity_search_with_relevance_scores("the quick brown fox", k=1)
    doc, score = res[0]
    assert doc.page_content == "the quick brown fox"
    assert score == pytest.approx(1.0, abs=1e-3)  # exact match -> ~1.0 relevance


def test_score_threshold_retriever(store):
    vs, _ = store
    retriever = vs.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.9, "k": 3},
    )
    docs = retriever.invoke("the quick brown fox")
    assert any(d.page_content == "the quick brown fox" for d in docs)


# -- sync_load (deferred inverted-index refresh) ---------------------------------------


def test_sync_load_false_then_manual_refresh():
    """sync_load=False skips the auto-refresh; a manual refresh publishes the rows.

    We do NOT assert the rows are hidden before the refresh: SereneDB has a background
    sync that may publish them on its own, so that would be flaky. What is reliable is
    that an explicit refresh_table() succeeds and the rows are then visible.
    """
    table = f"lc_syncload_{uuid.uuid4().hex[:8]}"
    engine = SereneDBEngine.from_connection_string(CONNINFO)
    engine.init_vectorstore_table(
        table, DIM, overwrite_existing=True, vector_index=IVFIndex()
    )
    vs = SereneDBVectorStore.create_sync(engine, DetEmb(), table, sync_load=False)
    try:
        vs.add_texts(["alpha", "beta", "gamma"])
        engine.refresh_table(table)  # caller publishes explicitly; must not raise
        after = [d.page_content for d in vs.similarity_search("alpha", k=5)]
        assert "alpha" in after
    finally:
        engine.drop_table(table)
        engine.close()


# -- metadata-column / JSON-field indexing --------------------------------------------

# One data set reused by both the indexed and control stores. ``category`` is a typed
# column; ``n`` (int) and ``tag`` (text) live in the JSON metadata column.
_MI_TEXTS = ["the quick brown fox", "a lazy dog", "quantum physics", "linear algebra"]
_MI_METAS = [
    {"category": "animal", "n": 1, "tag": "t0"},
    {"category": "animal", "n": 2, "tag": "t1"},
    {"category": "science", "n": 3, "tag": "t2"},
    {"category": "science", "n": 4, "tag": "t0"},
]


def _make_indexed_store(metadata_index):
    """Create a table + store with the given ``metadata_index``, load data, index it."""
    table = f"lc_mi_{uuid.uuid4().hex[:8]}"
    engine = SereneDBEngine.from_connection_string(CONNINFO)
    engine.init_vectorstore_table(
        table,
        DIM,
        overwrite_existing=True,
        metadata_columns=[Column("category", "TEXT")],
    )
    vs = SereneDBVectorStore.create_sync(
        engine,
        DetEmb(),
        table,
        metadata_columns=["category"],
        distance_strategy=DistanceStrategy.COSINE_DISTANCE,
        metadata_index=metadata_index,
    )
    vs.add_texts(_MI_TEXTS, metadatas=_MI_METAS)
    vs.apply_vector_index(IVFIndex(distance_strategy=DistanceStrategy.COSINE_DISTANCE))
    return vs, engine, table


@pytest.fixture
def indexed_store():
    """Store whose index covers the ``category`` column (verbatim) and JSON ``n``/``tag``."""
    vs, engine, table = _make_indexed_store(
        MetadataIndexConfig(
            json_fields=[
                JsonFieldIndex("n", "INTEGER"),
                JsonFieldIndex("tag", "TEXT"),
            ]
        )
    )
    try:
        yield vs, table
    finally:
        engine.drop_table(table)
        engine.close()


@pytest.fixture
def unindexed_store():
    """Control store: ``columns=[]`` and no JSON fields, so nothing extra is indexed."""
    vs, engine, table = _make_indexed_store(MetadataIndexConfig(columns=[]))
    try:
        yield vs, table
    finally:
        engine.drop_table(table)
        engine.close()


_PARITY_FILTERS = [
    {"category": "animal"},
    {"category": {"$in": ["animal", "science"]}},
    {"category": {"$ne": "animal"}},
    {"n": {"$gte": 3}},
    {"n": {"$between": [2, 3]}},
    {"tag": "t0"},
    {"tag": {"$in": ["t1", "t2"]}},
    {"$and": [{"category": "science"}, {"n": {"$gte": 4}}]},
    {"$or": [{"tag": "t2"}, {"n": {"$lte": 1}}]},
]


@pytest.mark.parametrize("filt", _PARITY_FILTERS)
def test_indexed_and_unindexed_return_identical_rows(
    indexed_store, unindexed_store, filt
):
    """Indexing metadata is a pure optimization: results must be identical either way."""
    indexed_vs, _ = indexed_store
    control_vs, _ = unindexed_store
    got_indexed = sorted(
        d.page_content for d in indexed_vs.similarity_search("x", k=10, filter=filt)
    )
    got_control = sorted(
        d.page_content for d in control_vs.similarity_search("x", k=10, filter=filt)
    )
    assert got_indexed == got_control


def test_indexed_json_filter_returns_expected_rows(indexed_store):
    """Sanity: the declared-JSON filter path (arrow + cast) returns the right rows."""
    vs, _ = indexed_store
    res = vs.similarity_search("x", k=10, filter={"n": {"$gte": 3}})
    assert sorted(d.page_content for d in res) == ["linear algebra", "quantum physics"]
    res = vs.similarity_search("x", k=10, filter={"tag": "t0"})
    assert sorted(d.page_content for d in res) == [
        "linear algebra",
        "the quick brown fox",
    ]


def test_metadata_index_columns_subset():
    """An explicit column subset (MetadataColumnIndex) builds and filters correctly."""
    vs, engine, table = _make_indexed_store(
        MetadataIndexConfig(columns=[MetadataColumnIndex("category")])
    )
    try:
        res = vs.similarity_search("x", k=10, filter={"category": "animal"})
        assert sorted(d.page_content for d in res) == [
            "a lazy dog",
            "the quick brown fox",
        ]
    finally:
        engine.drop_table(table)
        engine.close()


# -- safety net: default index-all silently skips unindexable column types -------------


def test_default_index_skips_unindexable_column_at_table_creation():
    """Building the index with the table must not fail on an unindexable column type.

    ``price`` is DECIMAL (not inverted-indexable); the default index-all case skips it
    silently, while ``category`` (TEXT) is still indexed and pushes down.
    """
    table = f"lc_mi_skip_{uuid.uuid4().hex[:8]}"
    engine = SereneDBEngine.from_connection_string(CONNINFO)
    engine.init_vectorstore_table(
        table,
        DIM,
        overwrite_existing=True,
        metadata_columns=[Column("category", "TEXT"), Column("price", "DECIMAL(10,2)")],
        vector_index=IVFIndex(distance_strategy=DistanceStrategy.COSINE_DISTANCE),
    )  # must NOT raise despite the DECIMAL column
    vs = SereneDBVectorStore.create_sync(
        engine,
        DetEmb(),
        table,
        metadata_columns=["category", "price"],
        distance_strategy=DistanceStrategy.COSINE_DISTANCE,
    )
    try:
        assert vs.is_valid_index() is True
        vs.add_texts(
            ["a", "b"],
            metadatas=[
                {"category": "x", "price": 1.5},
                {"category": "y", "price": 2.5},
            ],
        )
        # category is indexed, price was skipped; filters on both still return the
        # right rows (price runs as a residual filter).
        assert [
            d.page_content
            for d in vs.similarity_search("a", k=5, filter={"category": "x"})
        ] == ["a"]
        assert [
            d.page_content
            for d in vs.similarity_search("a", k=5, filter={"price": 1.5})
        ] == ["a"]
    finally:
        engine.drop_table(table)
        engine.close()


def test_default_index_skips_unindexable_column_at_apply():
    """The same safety net applies when the index is built later via apply_vector_index.

    Here the column type comes from information_schema (the store's own lookup), not from
    a user-declared Column, exercising the second code path.
    """
    table = f"lc_mi_skip2_{uuid.uuid4().hex[:8]}"
    engine = SereneDBEngine.from_connection_string(CONNINFO)
    engine.init_vectorstore_table(
        table,
        DIM,
        overwrite_existing=True,
        metadata_columns=[Column("category", "TEXT"), Column("price", "DECIMAL(10,2)")],
    )
    vs = SereneDBVectorStore.create_sync(
        engine,
        DetEmb(),
        table,
        metadata_columns=["category", "price"],
        distance_strategy=DistanceStrategy.COSINE_DISTANCE,
    )
    try:
        vs.add_texts(["a", "b"], metadatas=[{"category": "x"}, {"category": "y"}])
        # Default (metadata_index=None): must not raise; skips the DECIMAL column.
        vs.apply_vector_index(
            IVFIndex(distance_strategy=DistanceStrategy.COSINE_DISTANCE)
        )
        assert vs.is_valid_index() is True
        assert [
            d.page_content
            for d in vs.similarity_search("a", k=5, filter={"category": "x"})
        ] == ["a"]
    finally:
        engine.drop_table(table)
        engine.close()


def test_explicit_unindexable_column_surfaces_db_error():
    """An explicitly configured column is trusted: the DB's CREATE INDEX error surfaces."""
    table = f"lc_mi_err_{uuid.uuid4().hex[:8]}"
    engine = SereneDBEngine.from_connection_string(CONNINFO)
    engine.init_vectorstore_table(
        table,
        DIM,
        overwrite_existing=True,
        metadata_columns=[Column("price", "DECIMAL(10,2)")],
    )
    vs = SereneDBVectorStore.create_sync(
        engine,
        DetEmb(),
        table,
        metadata_columns=["price"],
        distance_strategy=DistanceStrategy.COSINE_DISTANCE,
        metadata_index=MetadataIndexConfig(columns=[MetadataColumnIndex("price")]),
    )
    try:
        with pytest.raises(Exception):
            vs.apply_vector_index(
                IVFIndex(distance_strategy=DistanceStrategy.COSINE_DISTANCE)
            )
    finally:
        engine.drop_table(table)
        engine.close()


# -- collision: a metadata column literally named "distance" ---------------------------


def test_distance_named_metadata_column_dense():
    """A metadata column named 'distance' must not be shadowed by the score alias."""
    table = f"lc_dist_{uuid.uuid4().hex[:8]}"
    engine = SereneDBEngine.from_connection_string(CONNINFO)
    engine.init_vectorstore_table(
        table,
        DIM,
        overwrite_existing=True,
        metadata_columns=[Column("distance", "TEXT")],
        vector_index=IVFIndex(distance_strategy=DistanceStrategy.COSINE_DISTANCE),
    )
    vs = SereneDBVectorStore.create_sync(
        engine,
        DetEmb(),
        table,
        metadata_columns=["distance"],
        distance_strategy=DistanceStrategy.COSINE_DISTANCE,
    )
    try:
        vs.add_texts(
            ["red", "blue"], metadatas=[{"distance": "far"}, {"distance": "near"}]
        )
        by_text = {
            d.page_content: (d, s)
            for d, s in vs.similarity_search_with_score("red", k=2)
        }
        # The 'distance' metadata value is preserved (not overwritten by the score)...
        assert by_text["red"][0].metadata["distance"] == "far"
        assert by_text["blue"][0].metadata["distance"] == "near"
        # ...and the score is a real distance (~0 for the exact-text match).
        assert by_text["red"][1] < 1e-6
    finally:
        engine.drop_table(table)
        engine.close()


def test_distance_named_metadata_column_hybrid():
    """The same collision must be handled through the hybrid (RRF) fusion path."""
    table = f"lc_disth_{uuid.uuid4().hex[:8]}"
    engine = SereneDBEngine.from_connection_string(CONNINFO)
    engine.init_vectorstore_table(
        table,
        DIM,
        overwrite_existing=True,
        metadata_columns=[Column("distance", "TEXT")],
    )
    cfg = HybridSearchConfig(
        primary_top_k=10, secondary_top_k=10, fusion=FusionStrategy.RRF
    )
    vs = SereneDBVectorStore.create_sync(
        engine, DetEmb(), table, metadata_columns=["distance"], hybrid_search_config=cfg
    )
    try:
        vs.add_texts(
            ["the quick brown fox", "a lazy brown dog"],
            metadatas=[{"distance": "far"}, {"distance": "near"}],
        )
        vs.apply_hybrid_search_index()
        res = vs.similarity_search("brown", k=2)
        assert res
        # Metadata survives fusion; the fused score doesn't collide with the column.
        assert all("distance" in d.metadata for d in res)
        assert {d.metadata["distance"] for d in res} <= {"far", "near"}
    finally:
        engine.drop_table(table)
        engine.close()


# -- full-text-search filter operators (@@ / ts_*) on indexed metadata columns ---------


@pytest.fixture
def fts_store():
    """Store with an inverted-indexed ``category`` column for the @@/ts_* operators."""
    table = f"lc_fts_{uuid.uuid4().hex[:8]}"
    engine = SereneDBEngine.from_connection_string(CONNINFO)
    engine.init_vectorstore_table(
        table,
        DIM,
        overwrite_existing=True,
        metadata_columns=[Column("category", "TEXT")],
        vector_index=IVFIndex(distance_strategy=DistanceStrategy.COSINE_DISTANCE),
    )
    vs = SereneDBVectorStore.create_sync(
        engine,
        DetEmb(),
        table,
        metadata_columns=["category"],
        distance_strategy=DistanceStrategy.COSINE_DISTANCE,
    )
    vs.add_texts(
        ["a", "b", "c", "d", "e"],
        metadatas=[
            {"category": "sci-fi"},
            {"category": "science"},
            {"category": "drama"},
            {"category": "comedy"},
            {"category": "documentary"},
        ],
    )
    try:
        yield vs
    finally:
        engine.drop_table(table)
        engine.close()


def _fts_cats(vs, flt):
    return sorted(
        d.metadata["category"] for d in vs.similarity_search("x", k=10, filter=flt)
    )


def test_fts_startswith(fts_store):
    assert _fts_cats(fts_store, {"category": {"$startswith": "sci"}}) == [
        "sci-fi",
        "science",
    ]


def test_fts_regex(fts_store):
    assert _fts_cats(fts_store, {"category": {"$regex": "dram."}}) == ["drama"]


def test_fts_fuzzy(fts_store):
    assert _fts_cats(fts_store, {"category": {"$fuzzy": "scifi"}}) == ["sci-fi"]


def test_fts_match_list(fts_store):
    assert _fts_cats(fts_store, {"category": {"$match": ["drama", "comedy"]}}) == [
        "comedy",
        "drama",
    ]


def test_fts_match_min_match(fts_store):
    # min_match = 1 is a plain OR over the tokens.
    got = _fts_cats(
        fts_store,
        {"category": {"$match": {"tokens": ["drama", "comedy"], "min_match": 1}}},
    )
    assert got == ["comedy", "drama"]


def test_fts_phrase(fts_store):
    assert _fts_cats(fts_store, {"category": {"$phrase": "drama"}}) == ["drama"]


def test_fts_operator_composes_with_logical(fts_store):
    got = _fts_cats(
        fts_store,
        {
            "$or": [
                {"category": {"$startswith": "com"}},
                {"category": {"$regex": "dram."}},
            ]
        },
    )
    assert got == ["comedy", "drama"]


def test_fts_operator_rejects_non_indexed_field(fts_store):
    # `note` is not an indexed metadata column (it lands in the JSON blob), so a
    # full-text operator on it must raise rather than emit an invalid `@@`.
    with pytest.raises(ValueError, match="inverted-indexed"):
        fts_store.similarity_search("x", k=5, filter={"note": {"$startswith": "foo"}})


def test_fts_on_declared_json_text_field():
    """FTS operators also work on a declared TEXT JSON sub-field (indexed as (md->>'k'))."""
    table = f"lc_ftsj_{uuid.uuid4().hex[:8]}"
    engine = SereneDBEngine.from_connection_string(CONNINFO)
    mi = MetadataIndexConfig(json_fields=[JsonFieldIndex("tag", "TEXT")])
    engine.init_vectorstore_table(
        table,
        DIM,
        overwrite_existing=True,
        vector_index=IVFIndex(distance_strategy=DistanceStrategy.COSINE_DISTANCE),
        metadata_index=mi,
    )
    vs = SereneDBVectorStore.create_sync(
        engine,
        DetEmb(),
        table,
        distance_strategy=DistanceStrategy.COSINE_DISTANCE,
        metadata_index=mi,
    )
    vs.add_texts(
        ["a", "b", "c"],
        metadatas=[{"tag": "science"}, {"tag": "scifi"}, {"tag": "drama"}],
    )
    try:
        got = sorted(
            d.metadata["tag"]
            for d in vs.similarity_search(
                "x", k=10, filter={"tag": {"$startswith": "sci"}}
            )
        )
        assert got == ["science", "scifi"]
        # An undeclared JSON field (not indexed) is still rejected.
        with pytest.raises(ValueError, match="inverted-indexed"):
            vs.similarity_search("x", k=5, filter={"other": {"$regex": "x.*"}})
    finally:
        engine.drop_table(table)
        engine.close()


def test_fts_ngram_on_dictionary_indexed_column():
    """$ngram works on a metadata column indexed with an n-gram dictionary.

    ts_ngram needs the column tokenized by an n-gram dictionary (with frequency AND
    position); we can't enforce that from the client, so the user creates the dictionary
    and indexes the column with it. Here we set that up explicitly.
    """
    table = f"lc_ngram_{uuid.uuid4().hex[:8]}"
    dict_name = f"ng_dict_{uuid.uuid4().hex[:8]}"
    engine = SereneDBEngine.from_connection_string(CONNINFO)
    engine.init_vectorstore_table(
        table, DIM, overwrite_existing=True, metadata_columns=[Column("title", "TEXT")]
    )
    engine._run_as_sync(
        engine._aexecute(
            f'CREATE TEXT SEARCH DICTIONARY "public"."{dict_name}" '
            "(template = 'ngram', mingram = 2, maxgram = 3, "
            "frequency = true, position = true)"
        )
    )
    mi = MetadataIndexConfig(
        columns=[MetadataColumnIndex("title", dictionary=dict_name)]
    )
    vs = SereneDBVectorStore.create_sync(
        engine,
        DetEmb(),
        table,
        metadata_columns=["title"],
        distance_strategy=DistanceStrategy.COSINE_DISTANCE,
        metadata_index=mi,
    )
    vs.add_texts(
        ["a", "b", "c"],
        metadatas=[{"title": "hello"}, {"title": "help"}, {"title": "world"}],
    )
    vs.apply_vector_index(IVFIndex(distance_strategy=DistanceStrategy.COSINE_DISTANCE))
    try:
        got = sorted(
            d.metadata["title"]
            for d in vs.similarity_search(
                "x",
                k=10,
                filter={"title": {"$ngram": {"text": "hello", "threshold": 0.3}}},
            )
        )
        assert "hello" in got
    finally:
        engine.drop_table(table)
        engine._run_as_sync(
            engine._aexecute(
                f'DROP TEXT SEARCH DICTIONARY IF EXISTS "public"."{dict_name}"'
            )
        )
        engine.close()
