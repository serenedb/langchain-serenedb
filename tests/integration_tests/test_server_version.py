"""Server version check performed when a SereneDBEngine is created.

Set ``SERENEDB_CONNINFO`` to point at a running instance (default targets port 7890).
Live tests are skipped if the database is unreachable.
"""

import asyncio
import os

import psycopg
import pytest
from psycopg_pool import AsyncConnectionPool

from langchain_serenedb import SereneDBEngine, UnsupportedServerVersionError
from langchain_serenedb import engine as engine_module
from langchain_serenedb.engine import (
    MIN_SUPPORTED_SERVER_VERSION,
    _parse_server_version,
)

CONNINFO = os.environ.get(
    "SERENEDB_CONNINFO", "host=127.0.0.1 port=7890 user=postgres dbname=postgres"
)


def _db_available() -> bool:
    try:
        with psycopg.connect(CONNINFO, connect_timeout=3):
            return True
    except Exception:
        return False


requires_db = pytest.mark.skipif(
    not _db_available(), reason=f"SereneDB not reachable at {CONNINFO!r}"
)

UNSUPPORTED_MESSAGE = (
    r"version \d+\.\d+\.\d+.* is not supported\. Minimum supported version is "
    r"99\.01\.0\. To disable version check use ignore_db_version=True"
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("PostgreSQL 18.3 (SereneDB 26.11.2)", (26, 11, 2)),
        ("PostgreSQL 18.3 (SereneDB 26.09.1.1-lts)", (26, 9, 1, 1)),
        ("PostgreSQL 18.3 (SereneDB 26.10.0)", (26, 10, 0)),
        ("PostgreSQL 16.2 on x86_64-pc-linux-gnu", None),
    ],
)
def test_parse_server_version(text, expected):
    assert _parse_server_version(text) == expected


@requires_db
def test_supported_server_records_version():
    engine = SereneDBEngine.from_connection_string(CONNINFO)
    try:
        assert engine.server_version is not None
        assert engine.server_version >= MIN_SUPPORTED_SERVER_VERSION
    finally:
        engine.close()


@requires_db
def test_unsupported_server_raises(monkeypatch):
    monkeypatch.setattr(engine_module, "MIN_SUPPORTED_SERVER_VERSION", (99, 1, 0))
    with pytest.raises(UnsupportedServerVersionError, match=UNSUPPORTED_MESSAGE):
        SereneDBEngine.from_connection_string(CONNINFO)


@requires_db
def test_ignore_db_version_skips_check(monkeypatch):
    monkeypatch.setattr(engine_module, "MIN_SUPPORTED_SERVER_VERSION", (99, 1, 0))
    engine = SereneDBEngine.from_connection_string(CONNINFO, ignore_db_version=True)
    try:
        assert engine.server_version is None
        rows = engine._run_as_sync(engine._afetch("SELECT 1 AS x"))
        assert rows == [{"x": 1}]
    finally:
        engine.close()


@requires_db
def test_from_pool_without_loop_checks_on_first_query(monkeypatch):
    monkeypatch.setattr(engine_module, "MIN_SUPPORTED_SERVER_VERSION", (99, 1, 0))

    async def scenario() -> None:
        pool = AsyncConnectionPool(CONNINFO, open=False)
        await pool.open()
        try:
            engine = SereneDBEngine.from_pool(pool)
            with pytest.raises(
                UnsupportedServerVersionError, match=UNSUPPORTED_MESSAGE
            ):
                await engine._afetch("SELECT 1")
        finally:
            await pool.close()

    asyncio.run(scenario())
