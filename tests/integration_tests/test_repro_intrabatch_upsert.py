"""TEMPORARY repro of the intra-batch upsert duplicate-key failure.

Mimics the real workload that fails on the serened binary, to help debug/fix the binary:
  * TEXT id column, ids of the real form "usc:tN:sNg" (e.g. "usc:t10:s130g").
  * TOTAL_DOCS documents loaded in batches of BATCH_SIZE (like the real app). Every
    batch places a constant CONFLICT_ID at its first and last position, so batch 0 seeds
    it and every later batch conflicts BOTH with an already-committed row AND with itself
    (the two conflicts the real failing batch had -- it was not the first load).
  * 1536-dim embeddings, each row a UNIQUE random vector (shared vectors may be
    internally deduplicated and hide the bug -- data size in memory seems to matter).
  * one metadata column "namespace" = constant "usc".

To reproduce, the batch-dedup fix in aadd_embeddings must be DISABLED (so duplicate ids
reach the single executemany transaction). On a buggy engine `add_documents` raises
``UniqueViolation: duplicate key``; on a fixed engine it upserts (last write wins).

Run against a custom binary, e.g.:
    tests/run_tests_local.sh /path/to/serened \\
        tests/integration_tests/test_repro_intrabatch_upsert.py

DELETE this file (and re-enable the dedup) once the binary is fixed.
"""

import os
import random

import psycopg
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_serenedb import Column, HNSWIndex, SereneDBEngine, SereneDBVectorStore

CONNINFO = os.environ.get(
    "SERENEDB_CONNINFO", "host=127.0.0.1 port=8549 user=postgres dbname=postgres"
)
DIM = 1536
TOTAL_DOCS = 15000  # total documents loaded (sweep this to hunt the trigger)
BATCH_SIZE = 1024  # documents per add_documents call, like the real app
CONFLICT_ID = "usc:t0:s0g"  # placed at index 0 and the last index of every batch


def _db_available() -> bool:
    try:
        with psycopg.connect(CONNINFO, connect_timeout=3):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _db_available(), reason=f"SereneDB not reachable at {CONNINFO!r}"
)


class RandomEmbedding(Embeddings):
    """A fresh, unique random DIM-vector per text.

    Uniqueness is deliberate: if every row shared one vector the engine might dedup the
    payload internally and mask the id-conflict bug.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[random.random() for _ in range(DIM)] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [random.random() for _ in range(DIM)]


@pytest.mark.timeout(0)  # disable the global 60s cap (0 = no timeout); a slow debug
# binary can take minutes. Replace 0 with a number of seconds if you want a cap.
def test_repro_intrabatch_duplicate_upsert():
    engine = SereneDBEngine.from_connection_string(CONNINFO)
    table = "repro_intrabatch_upsert"
    engine.init_vectorstore_table(
        table,
        DIM,
        overwrite_existing=True,
        id_column=Column("langchain_id", "TEXT"),
        metadata_columns=[Column("namespace", "TEXT", nullable=True)],
        vector_index=HNSWIndex(),
    )
    store = SereneDBVectorStore.create_sync(
        engine, RandomEmbedding(), table, metadata_columns=["namespace"]
    )
    try:
        for batch_no, start in enumerate(range(0, TOTAL_DOCS, BATCH_SIZE)):
            n = min(start + BATCH_SIZE, TOTAL_DOCS) - start
            # Unique ids for the batch (+1 offset so none equals CONFLICT_ID by accident),
            # then force the first and last positions to the recurring CONFLICT_ID.
            batch_ids = [f"usc:t{start + j + 1}:s{start + j + 1}g" for j in range(n)]
            batch_ids[0] = CONFLICT_ID
            if n > 1:
                batch_ids[-1] = CONFLICT_ID  # intra-batch conflict (index 0 and 1023)
            if n > 512:
                batch_ids[-512] = CONFLICT_ID  # intra-batch conflict (index 0 and 1023)
            
            docs = [
                Document(
                    page_content=f"b{batch_no}:row{j}",
                    metadata={"namespace": "usc"},
                    id=batch_ids[j],
                )
                for j in range(n)
            ]
            # Batch 0 seeds CONFLICT_ID; every later batch conflicts with the committed
            # row AND has the same id twice within itself. On a buggy engine one of these
            # add_documents raises UniqueViolation; on a fixed engine each upserts.
            store.add_documents(docs)

        # Reached only on a fixed/working engine: the recurring id is a single row.
        got = store.get_by_ids([CONFLICT_ID])
        assert len(got) == 1
    finally:
        engine.drop_table(table)
        engine.close()
