# Contributing to langchain-serenedb

Everything you need to build, test, lint, and format the project. For what the
integration does and how to use it, see the [README](README.md).

## Development environment

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

## Linting, formatting, and type-checking

All checks run through [pre-commit](https://pre-commit.com): Ruff (lint + format),
repo-hygiene hooks, and mypy (type-check). Config is in `.pre-commit-config.yaml`, with
Ruff pinned to the `uv.lock` version. Run a full sweep (Ruff applies fixes; mypy only
reports):

```bash
pre-commit run --all-files
```

It is run **explicitly, not as a git hook** — so the formatter never rewrites your
commits and you stay in control of what/when to format. CI runs the same hooks as a gate
with `pre-commit run --all-files --show-diff-on-failure`: any fix a hook would apply
fails the job (printing the diff) but is never committed, since the CI checkout is
throwaway. (Added to the workflow later.)

mypy is a `local` hook that runs the project's own environment (`uv run --extra test
mypy`, so it needs `uv` on PATH) rather than an isolated one that could not resolve the
project's imports. Run it standalone with:

```bash
mypy langchain_serenedb   # or: uv run --extra test mypy langchain_serenedb
```

## Running the tests

Tests expect a running SereneDB reachable via the `SERENEDB_CONNINFO` environment
variable (default `host=127.0.0.1 port=7890 user=postgres dbname=postgres`).

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

## Running the tests in Docker Compose (CI)

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
