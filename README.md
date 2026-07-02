# langchain-serenedb

A [LangChain](https://python.langchain.com/) vector store integration for
[SereneDB](https://serenedb.com)

SereneDB speaks the PostgreSQL wire protocol, so this package connects with **psycopg3**.
It maps the integration onto SereneDB's native capabilities:

| Vector Store Search Feature | SereneDB Feature used |
|---|---|
| Vector column |  `FLOAT[N]`  |
| Distance ops | `<->`, `<=>`, `<#>`, `<+>` |
| ANN index |  inverted index on the vector column e.g. `USING inverted (emb hnsw (metric='cosine', ...))` |
| Full-text |  inverted index on the text column + `BM25(idx.tableoid)` |
| Metadata | `JSON` column, explicit columns |

## Quickstart (engine + table)

```python
from langchain_serenedb import SereneDBEngine, HNSWIndex

engine = SereneDBEngine.from_connection_string(
    "host=127.0.0.1 port=7890 user=postgres dbname=postgres"
)

# Table only (vector search falls back to an exact scan until an index is built):
engine.init_vectorstore_table(table_name="my_docs", vector_size=768)

# Or create the table and its HNSW ANN index in one call, so vector search is
# accelerated from the start:
engine.init_vectorstore_table("my_docs", 768, vector_index=HNSWIndex())
#   ...or the combined full-text + vector index for hybrid search:
#   engine.init_vectorstore_table("my_docs", 768, hybrid_search_config=HybridSearchConfig())

# ... after writing rows, publish them to the inverted index:
engine.refresh_table("my_docs")
```

> **Tip:** for a large bulk load, SereneDB builds a more compact graph if you create the
> index *after* loading (`store.apply_vector_index(HNSWIndex())`); creating it up front
> with the table is the convenient choice for incremental workloads.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"   # runtime + test dependencies, in one command
pytest
```

For a **reproducible** install (pinned versions from `uv.lock`), use [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra test       # creates .venv and installs exactly what uv.lock pins
```

Regenerate the lock after changing dependencies in `pyproject.toml` with `uv lock`.

### Linting, formatting, and type-checking

The `[test]` extra installs Ruff (linter + formatter) and mypy; both read their config
from `pyproject.toml`. None of these need a running database:

```bash
ruff check .             # lint (E/F/I/T201); add --fix to auto-fix
ruff format .            # format (or --check to only report)
mypy langchain_serenedb  # type-check
```

Without an activated venv, prefix each with `uv run --extra test` (e.g.
`uv run --extra test ruff check .`).

Ruff and repo-hygiene checks are also bundled in [pre-commit](https://pre-commit.com)
(`.pre-commit-config.yaml`, with Ruff pinned to the `uv.lock` version). It is run
**explicitly, not as a git hook** — so the formatter never rewrites your commits and you
stay in control of what/when to format. Run a full sweep (applies fixes) with:

```bash
pre-commit run --all-files
```

CI validates pull requests without modifying anything, using Ruff in check mode
(`ruff check . && ruff format --check .`) — to be added to the workflow later.

mypy is intentionally not a pre-commit hook — it needs the project's dependencies to
resolve imports — so run it via the commands above or in CI.

Tests expect a running SereneDB reachable via the `SERENEDB_CONNINFO` environment
variable (default `host=127.0.0.1 port=7890 user=postgres dbname=postgres`).

### Running the suite

Two scripts, split by responsibility. Both exit with pytest's exit code, so either
drops straight into CI.

`tests/run.sh` is the **executor**: it is instance-agnostic and runs the suite against
an *already-running* SereneDB. It waits for the instance to accept connections, then
runs pytest against it. Point it at a local process or another docker-compose service —
it does not care how the instance was started:

```bash
tests/run.sh                                     # localhost:7890, whole suite
tests/run.sh --host serened --port 7890          # a docker-compose service
SERENEDB_HOST=serened tests/run.sh -- -k hybrid  # env + forwarded pytest args
```

`tests/run_tests_local.sh` is the **local provisioner**: it starts a fresh SereneDB from
a `serened` binary, hands it off to `run.sh`, then tears it down:

```bash
tests/run_tests_local.sh /path/to/serened                 # free ephemeral port + temp data dir
tests/run_tests_local.sh /path/to/serened --port 9999      # fixed port
tests/run_tests_local.sh /path/to/serened -- -k hybrid -q  # forward args to pytest
```

If `--port` is omitted a free port is requested from the kernel; if `--data-dir` is
omitted a temp directory is created and removed on exit (a directory you pass in is
left untouched). Set `PYTHON` to choose the interpreter (defaults to `.venv/bin/python`).

### Running the suite in Docker Compose (CI)

`tests/docker-compose.yml` brings up SereneDB in one container and runs the suite
against it from another. Nothing is built: both services use prebuilt images that CI
agents already keep cached. The test service uses the SereneDB build image (which
carries almost all test deps), installs anything missing from `requirements.txt` at
startup, and bind-mounts the checked-out repo (in CI, the workspace). The test runner
uses `run.sh`, so it just waits for the database to accept connections, then runs pytest.

For a local run, `tests/run_tests_docker.sh` wraps all of this in one command — it
brings the stack up, streams the logs, exits with pytest's code, and tears everything
down afterward:

```bash
tests/run_tests_docker.sh                 # whole suite
tests/run_tests_docker.sh -- -k hybrid -q # forward pytest args
tests/run_tests_docker.sh --keep          # leave containers up for inspection
tests/run_tests_docker.sh --unique        # parallel-safe project name (shared CI agent)
```

Under the hood (and in CI) that is just:

```bash
docker compose -f tests/docker-compose.yml up \
  --abort-on-container-exit --exit-code-from tests
```

`--exit-code-from tests` makes the whole command exit with pytest's exit code. The
images default to `serenedb/serenedb:latest` and `serenedb/serenedb-build-ubuntu:latest`;
override them with the `SERENEDB_IMAGE` / `BUILD_IMAGE` environment variables (or a
`.env` file):

```bash
SERENEDB_IMAGE=serenedb/serenedb:1.2.3 docker compose -f tests/docker-compose.yml up \
  --abort-on-container-exit --exit-code-from tests
```

To forward pytest args, set `PYTEST_ADDOPTS` (pytest reads it natively):

```bash
PYTEST_ADDOPTS="-k hybrid -q" docker compose -f tests/docker-compose.yml run --rm tests
```

## License

MIT
