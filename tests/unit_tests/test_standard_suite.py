"""LangChain standard ``VectorStore`` compliance suite for SereneDB.

Runs ``langchain_tests.integration_tests.VectorStoreIntegrationTests`` against a live
SereneDB (set ``SERENEDB_CONNINFO``; default targets port 7890). The suite uses
``DeterministicFakeEmbedding`` and non-UUID string ids, so the table is created with a
``VARCHAR`` id column. Tests are skipped if the database is unreachable.
"""

import os
import uuid
from typing import AsyncGenerator, Generator

import psycopg
import pytest
import pytest_asyncio
from langchain_tests.integration_tests import VectorStoreIntegrationTests
from langchain_tests.integration_tests.vectorstores import EMBEDDING_SIZE

from langchain_serenedb import Column, SereneDBEngine, SereneDBVectorStore

CONNINFO = os.environ.get(
    "SERENEDB_CONNINFO", "host=127.0.0.1 port=7890 user=postgres dbname=postgres"
)

_ID_COLUMN = Column(name="langchain_id", data_type="VARCHAR", nullable=False)


def _db_available() -> bool:
    try:
        with psycopg.connect(CONNINFO, connect_timeout=3):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _db_available(), reason=f"SereneDB not reachable at {CONNINFO!r}"
)


class TestSereneDBStandardSuiteSync(VectorStoreIntegrationTests):
    @pytest.fixture
    def vectorstore(self) -> Generator[SereneDBVectorStore, None, None]:
        table = "std_sync_" + uuid.uuid4().hex[:12]
        engine = SereneDBEngine.from_connection_string(CONNINFO)
        engine.init_vectorstore_table(
            table, EMBEDDING_SIZE, overwrite_existing=True, id_column=_ID_COLUMN
        )
        vs = SereneDBVectorStore.create_sync(engine, self.get_embeddings(), table)
        try:
            yield vs
        finally:
            engine.drop_table(table)
            engine._run_as_sync(engine.close())


class TestSereneDBStandardSuiteAsync(VectorStoreIntegrationTests):
    @pytest_asyncio.fixture
    async def vectorstore(self) -> AsyncGenerator[SereneDBVectorStore, None]:
        table = "std_async_" + uuid.uuid4().hex[:12]
        engine = SereneDBEngine.from_connection_string(CONNINFO)
        await engine.ainit_vectorstore_table(
            table, EMBEDDING_SIZE, overwrite_existing=True, id_column=_ID_COLUMN
        )
        vs = await SereneDBVectorStore.create(engine, self.get_embeddings(), table)
        try:
            yield vs
        finally:
            await engine.adrop_table(table)
            await engine.close()
