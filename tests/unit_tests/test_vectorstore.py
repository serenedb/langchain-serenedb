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
    HNSWIndex,
    HybridSearchConfig,
    SereneDBEngine,
    SereneDBVectorStore,
    reciprocal_rank_fusion,
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
    engine._run_as_sync(engine.close())


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


def test_get_by_ids(store):
    vs, ids = store
    got = vs.get_by_ids(ids[:2])
    assert {d.id for d in got} == set(ids[:2])


def test_apply_index_and_search(store):
    vs, _ = store
    vs.apply_vector_index(HNSWIndex(distance_strategy=DistanceStrategy.COSINE_DISTANCE))
    assert vs.is_valid_index() is True
    res = vs.similarity_search("quantum physics", k=1)
    assert res[0].page_content == "quantum physics"


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
    engine._run_as_sync(engine.close())


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
        vector_index=HNSWIndex(distance_strategy=DistanceStrategy.COSINE_DISTANCE),
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
        engine._run_as_sync(engine.close())


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
        engine._run_as_sync(engine.close())


def test_rrf_fusion():
    table = f"lc_rrf_{uuid.uuid4().hex[:8]}"
    engine = SereneDBEngine.from_connection_string(CONNINFO)
    engine.init_vectorstore_table(table, DIM, overwrite_existing=True)
    cfg = HybridSearchConfig(
        primary_top_k=10, secondary_top_k=10, fusion_function=reciprocal_rank_fusion
    )
    vs = SereneDBVectorStore.create_sync(
        engine, DetEmb(), table, hybrid_search_config=cfg
    )
    vs.add_texts(["the quick brown fox", "a lazy brown dog", "quantum physics"])
    vs.apply_hybrid_search_index()
    try:
        res = vs.similarity_search("brown dog", k=3)
        assert len(res) >= 1
    finally:
        engine.drop_table(table)
        engine._run_as_sync(engine.close())
