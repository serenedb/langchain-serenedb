"""Sync methods called from the engine's own event loop raise instead of hanging.

Set ``SERENEDB_CONNINFO`` to point at a running instance (default targets port 7890).
Tests are skipped if the database is unreachable.
"""

import asyncio
import contextlib
import gc
import os
import uuid
import warnings

import psycopg
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from langchain_serenedb import SereneDBEngine, SereneDBVectorStore

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


# A regression would deadlock rather than fail, so bound every test.
pytestmark = [
    pytest.mark.skipif(
        not _db_available(), reason=f"SereneDB not reachable at {CONNINFO!r}"
    ),
    pytest.mark.timeout(60),
]

ON_LOOP_MESSAGE = "called from the event loop the engine runs on"


@contextlib.contextmanager
def _no_unawaited_coroutine_warning():
    """Fail if a coroutine created inside the block is dropped without being closed.

    Python reports an un-awaited coroutine when it is garbage-collected, so collect
    before checking the recorded warnings.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
        gc.collect()
    leaked = [w for w in caught if "was never awaited" in str(w.message)]
    assert not leaked, [str(w.message) for w in leaked]


async def _open_pool() -> AsyncConnectionPool:
    pool = AsyncConnectionPool(CONNINFO, open=False, kwargs={"row_factory": dict_row})
    await pool.open()
    return pool


def test_sync_engine_method_on_own_loop_raises():
    table = f"lc_loop_{uuid.uuid4().hex[:8]}"

    async def scenario() -> None:
        pool = await _open_pool()
        try:
            engine = SereneDBEngine.from_pool(pool, loop=asyncio.get_running_loop())
            with _no_unawaited_coroutine_warning():
                with pytest.raises(RuntimeError, match=ON_LOOP_MESSAGE):
                    engine.init_vectorstore_table(table, DIM)
            # The loop is still alive and the async form works.
            await engine.ainit_vectorstore_table(table, DIM)
            await engine.adrop_table(table)
        finally:
            await pool.close()

    asyncio.run(scenario())


def test_sync_store_method_on_own_loop_raises():
    table = f"lc_loop_{uuid.uuid4().hex[:8]}"

    async def scenario() -> None:
        pool = await _open_pool()
        try:
            engine = SereneDBEngine.from_pool(pool, loop=asyncio.get_running_loop())
            await engine.ainit_vectorstore_table(table, DIM)
            try:
                with pytest.raises(RuntimeError, match=ON_LOOP_MESSAGE):
                    SereneDBVectorStore.create_sync(
                        engine, DeterministicFakeEmbedding(size=DIM), table
                    )
            finally:
                await engine.adrop_table(table)
        finally:
            await pool.close()

    asyncio.run(scenario())


def test_sync_method_from_another_thread_works():
    """The supported way to call a sync method from async code: off the loop thread."""
    table = f"lc_loop_{uuid.uuid4().hex[:8]}"

    async def scenario() -> None:
        pool = await _open_pool()
        try:
            engine = SereneDBEngine.from_pool(pool, loop=asyncio.get_running_loop())
            await asyncio.to_thread(engine.init_vectorstore_table, table, DIM)
            await asyncio.to_thread(engine.drop_table, table)
        finally:
            await pool.close()

    asyncio.run(scenario())


def test_sync_method_without_loop_raises_without_warning():
    async def scenario() -> None:
        pool = await _open_pool()
        try:
            engine = SereneDBEngine.from_pool(pool)
            with _no_unawaited_coroutine_warning():
                with pytest.raises(Exception, match="without a background loop"):
                    engine.drop_table("never_created")
        finally:
            await pool.close()

    asyncio.run(scenario())
