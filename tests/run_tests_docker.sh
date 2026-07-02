#!/usr/bin/env bash
#
# Emulate the CI run locally: use Docker Compose to bring up a SereneDB instance and a
# test-runner container, run the suite against it, stream the logs, and exit with
# pytest's exit code. A convenience wrapper around tests/docker-compose.yml for local
# debugging -- the same thing CI does, in one command.
#
# Usage:
#   tests/run_tests_docker.sh [options] [-- <pytest args>...]
#
# Options:
#   --pull             Pull the images first (otherwise Compose pulls only if an image
#                      is missing locally).
#   --keep             Leave the containers up after the run instead of tearing them
#                      down, so you can inspect them (logs / exec). By default
#                      everything is removed on exit.
#   -p, --project-name NAME
#                      Compose project name (default: langchain-serenedb-tests, or
#                      $COMPOSE_PROJECT_NAME if set). Determines the container/network
#                      names.
#   --unique           Append a unique suffix to the project name so parallel runs on
#                      one host do not collide. Implies its own teardown target.
#   -h, --help         Show this help.
#
# Anything after `--` (or any extra positional args) is forwarded to pytest via
# PYTEST_ADDOPTS. With no args the whole suite runs.
#
# Environment:
#   SERENEDB_IMAGE       SereneDB image (default serenedb/serenedb:latest).
#   BUILD_IMAGE          Test-runner image (default serenedb/serenedb-build-ubuntu:latest).
#   PYTEST_ADDOPTS       Extra pytest args; anything passed on the command line is appended.
#   COMPOSE_PROJECT_NAME Base project name (overridden by --project-name / --unique).
#
# Examples:
#   tests/run_tests_docker.sh
#   tests/run_tests_docker.sh -- -k hybrid -q
#   SERENEDB_IMAGE=serenedb/serenedb:1.2.3 tests/run_tests_docker.sh --pull
#   tests/run_tests_docker.sh --unique          # parallel-safe (CI on a shared agent)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

usage() { sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

PULL=0
KEEP=0
UNIQUE=0
PROJECT="${COMPOSE_PROJECT_NAME:-langchain-serenedb-tests}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pull) PULL=1; shift ;;
    --keep) KEEP=1; shift ;;
    -p|--project-name) PROJECT="${2:-}"; shift 2 ;;
    -p=*|--project-name=*) PROJECT="${1#*=}"; shift ;;
    --unique) UNIQUE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# A unique suffix keeps parallel runs on one host from colliding on container/network
# names. $$ (this script's PID) is unique among live processes; $RANDOM guards reuse.
if [[ "$UNIQUE" == "1" ]]; then
  PROJECT="${PROJECT}-$$-${RANDOM}"
fi
if [[ -z "$PROJECT" ]]; then
  echo "Error: project name must not be empty." >&2
  exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker not found on PATH." >&2
  exit 2
fi
if ! docker compose version >/dev/null 2>&1; then
  echo "Error: 'docker compose' (v2) is required." >&2
  exit 2
fi

# Append any command-line pytest args to PYTEST_ADDOPTS; docker-compose.yml forwards it
# into the tests container, where pytest reads it natively.
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  if [[ -n "${PYTEST_ADDOPTS:-}" ]]; then
    export PYTEST_ADDOPTS="$PYTEST_ADDOPTS ${EXTRA_ARGS[*]}"
  else
    export PYTEST_ADDOPTS="${EXTRA_ARGS[*]}"
  fi
fi

compose() { docker compose -p "$PROJECT" -f "$COMPOSE_FILE" "$@"; }

cleanup() {
  local rc=$?
  if [[ "$KEEP" == "1" ]]; then
    echo
    echo ">> --keep: leaving containers up. Tear down with:"
    echo "   docker compose -p $PROJECT -f $COMPOSE_FILE down --remove-orphans"
  else
    compose down --remove-orphans >/dev/null 2>&1 || true
  fi
  return "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo ">> compose file : $COMPOSE_FILE"
echo ">> project      : $PROJECT"
echo ">> serenedb     : ${SERENEDB_IMAGE:-serenedb/serenedb:latest}"
echo ">> build image  : ${BUILD_IMAGE:-serenedb/serenedb-build-ubuntu:latest}"
[[ -n "${PYTEST_ADDOPTS:-}" ]] && echo ">> pytest args  : $PYTEST_ADDOPTS"
echo

if [[ "$PULL" == "1" ]]; then
  echo ">> pulling images"
  compose pull
fi

# --exit-code-from implies --abort-on-container-exit: when the tests container exits,
# Compose stops the run and returns pytest's exit code. Logs stream live to the console.
compose up --exit-code-from tests
rc=$?

echo
echo ">> exit code: $rc"
exit "$rc"
