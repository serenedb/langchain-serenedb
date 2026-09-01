#!/usr/bin/env bash
#
# Run the test suite against an already-running SereneDB instance. This script is
# instance-agnostic: it does not start, stop, or provision anything. It only needs to
# know where the instance lives, so it works the same whether the instance is a local
# process (see run_tests_local.sh) or another container in a docker-compose network.
#
# It waits for the instance to accept connections, then runs pytest with
# SERENEDB_CONNINFO pointed at it. The script's exit code is the pytest exit code, so
# it can be used directly in CI.
#
# Usage:
#   tests/run.sh [--host H] [--port N] [--conninfo STR] [--timeout SECS] [--no-wait] [-- <pytest args>...]
#
# Options:
#   --host H           Host of the SereneDB instance. Default 127.0.0.1.
#   --port N           Port of the pg-wire listener. Default 7890.
#   --conninfo STR     Full libpq conninfo string. Overrides --host/--port and builds
#                      nothing; passed to pytest verbatim.
#   --timeout SECS     How long to wait for the instance to accept connections before
#                      giving up. Default 30.
#   --no-wait          Skip the readiness wait and run pytest immediately.
#   -h, --help         Show this help.
#
# Anything after `--` (or any extra positional args) is forwarded to pytest. With no
# pytest args, the whole tests/ directory is run.
#
# Environment (flags win over these):
#   SERENEDB_CONNINFO  Full conninfo, equivalent to --conninfo.
#   SERENEDB_HOST      Host, equivalent to --host.
#   SERENEDB_PORT      Port, equivalent to --port.
#   SERENEDB_USER      User used when building a conninfo. Default postgres.
#   SERENEDB_DBNAME    Database used when building a conninfo. Default postgres.
#   SERENEDB_PASSWORD  Password used when building a conninfo. Default: none (unset).
#                      The containerized image requires password auth over the network
#                      (loopback still trusts a passwordless superuser), so set this to
#                      the superuser password there. Ignored when --conninfo/
#                      SERENEDB_CONNINFO is given (pass the password in that string).
#   SERENEDB_WAIT_TIMEOUT  Readiness timeout in seconds, equivalent to --timeout.
#   PYTHON             Interpreter used for pytest and readiness checks. Defaults to
#                      the repo's .venv/bin/python if present, else python3. It must
#                      have langchain-serenedb and the test deps installed.
#
# Examples:
#   tests/run.sh                                    # localhost:7890, whole suite
#   tests/run.sh --host serened --port 7890         # a docker-compose service
#   SERENEDB_HOST=serened tests/run.sh -- -k hybrid # env + forwarded pytest args

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() { sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

HOST=""
PORT=""
CONNINFO=""
TIMEOUT=""
WAIT=1
PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="${2:-}"; shift 2 ;;
    --host=*) HOST="${1#*=}"; shift ;;
    --port) PORT="${2:-}"; shift 2 ;;
    --port=*) PORT="${1#*=}"; shift ;;
    --conninfo) CONNINFO="${2:-}"; shift 2 ;;
    --conninfo=*) CONNINFO="${1#*=}"; shift ;;
    --timeout) TIMEOUT="${2:-}"; shift 2 ;;
    --timeout=*) TIMEOUT="${1#*=}"; shift ;;
    --no-wait) WAIT=0; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; PYTEST_ARGS+=("$@"); break ;;
    -*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) PYTEST_ARGS+=("$1"); shift ;;
  esac
done

# Resolve the Python interpreter.
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

# Resolve where the instance lives. A conninfo (flag or env) is used verbatim;
# otherwise one is built from host/port/user/dbname.
CONNINFO="${CONNINFO:-${SERENEDB_CONNINFO:-}}"
if [[ -z "$CONNINFO" ]]; then
  HOST="${HOST:-${SERENEDB_HOST:-127.0.0.1}}"
  PORT="${PORT:-${SERENEDB_PORT:-7890}}"
  if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "Error: invalid port: $PORT" >&2
    exit 2
  fi
  CONNINFO="host=$HOST port=$PORT user=${SERENEDB_USER:-postgres} dbname=${SERENEDB_DBNAME:-postgres}"
  if [[ -n "${SERENEDB_PASSWORD:-}" ]]; then
    CONNINFO="$CONNINFO password=${SERENEDB_PASSWORD}"
  fi
fi

TIMEOUT="${TIMEOUT:-${SERENEDB_WAIT_TIMEOUT:-30}}"
if ! [[ "$TIMEOUT" =~ ^[0-9]+$ ]]; then
  echo "Error: invalid timeout: $TIMEOUT" >&2
  exit 2
fi

db_ready() {
  "$PYTHON" - "$CONNINFO" <<'PYEOF'
import sys
import psycopg
try:
    with psycopg.connect(sys.argv[1], connect_timeout=2):
        pass
except Exception:
    sys.exit(1)
PYEOF
}

# Redact any password before logging so it does not leak into CI output.
CONNINFO_REDACTED="$(printf '%s' "$CONNINFO" | sed -E 's/password=[^ ]*/password=***/')"
echo ">> target   : $CONNINFO_REDACTED"
echo ">> python   : $PYTHON"

# Wait until the instance accepts connections. Instance-agnostic: we only poll the
# connection, we do not assume anything about how (or where) it was started.
if [[ "$WAIT" == "1" ]]; then
  ready=0
  attempts=$(( TIMEOUT * 4 ))
  [[ "$attempts" -lt 1 ]] && attempts=1
  for _ in $(seq 1 "$attempts"); do
    if db_ready; then ready=1; break; fi
    sleep 0.25
  done
  if [[ "$ready" != "1" ]]; then
    echo "Error: SereneDB did not accept connections within ${TIMEOUT}s at: $CONNINFO" >&2
    exit 1
  fi
  echo ">> instance is ready; running tests"
else
  echo ">> skipping readiness wait (--no-wait); running tests"
fi
echo

# Run the suite. Default to the whole tests/ directory.
if [[ ${#PYTEST_ARGS[@]} -eq 0 ]]; then
  PYTEST_ARGS=("$SCRIPT_DIR")
fi

cd "$REPO_ROOT"
SERENEDB_CONNINFO="$CONNINFO" "$PYTHON" -m pytest "${PYTEST_ARGS[@]}"
rc=$?

echo
echo ">> pytest exit code: $rc"
exit "$rc"
