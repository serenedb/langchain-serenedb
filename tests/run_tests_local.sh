#!/usr/bin/env bash
#
# Start a throwaway SereneDB instance on the local machine from a `serened` binary,
# hand it off to run.sh to run the suite against it, then tear the instance down.
# The script's exit code is the pytest exit code, so it can be used directly in CI.
#
# This script owns only local provisioning (resolving the binary, picking a port,
# creating a data dir, starting and stopping the server). The actual readiness wait
# and pytest run are delegated to run.sh, which is instance-agnostic -- so the CI
# path (serened in another docker-compose container) can call run.sh directly and
# skip this script entirely.
#
# Usage:
#   tests/run_tests_local.sh <serened-binary> [--port N] [--data-dir DIR] [-- <pytest args>...]
#
# Arguments:
#   <serened-binary>   Path to (or name on PATH of) the `serened` executable. Required.
#
# Options:
#   --port N           Port for the pg-wire listener. If omitted, a free ephemeral
#                      port is requested from the kernel.
#   --data-dir DIR     SereneDB data directory. If omitted, a random directory is
#                      created under $TMPDIR (or /tmp) and removed on exit. A
#                      directory you pass in is used as-is and never deleted.
#   -h, --help         Show this help.
#
# Anything after `--` (or any extra positional args) is forwarded to pytest via run.sh.
# With no pytest args, the whole tests/ directory is run.
#
# Environment:
#   PYTHON             Interpreter used for pytest and readiness checks. Defaults to
#                      the repo's .venv/bin/python if present, else python3. It must
#                      have langchain-serenedb and the test deps installed.
#
# Examples:
#   tests/run_tests_local.sh /path/to/serened
#   tests/run_tests_local.sh serened --port 9999 -- -k hybrid -q
#   PYTHON=.venv/bin/python tests/run_tests_local.sh ./build/bin/serened tests/integration_tests/test_standard_suite.py

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() { sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

BINARY=""
PORT=""
DATA_DIR=""
PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="${2:-}"; shift 2 ;;
    --port=*) PORT="${1#*=}"; shift ;;
    --data-dir) DATA_DIR="${2:-}"; shift 2 ;;
    --data-dir=*) DATA_DIR="${1#*=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; PYTEST_ARGS+=("$@"); break ;;
    -*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)
      if [[ -z "$BINARY" ]]; then BINARY="$1"; else PYTEST_ARGS+=("$1"); fi
      shift ;;
  esac
done

if [[ -z "$BINARY" ]]; then
  echo "Error: path to the serened binary is required." >&2
  usage >&2
  exit 2
fi

# Resolve the Python interpreter (also used to pick a free port). Exported so run.sh
# uses the same interpreter.
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi
export PYTHON

# Resolve the binary to an absolute path (or a PATH lookup).
if [[ "$BINARY" == */* ]]; then
  bin_dir="$(cd "$(dirname "$BINARY")" 2>/dev/null && pwd || true)"
  if [[ -z "$bin_dir" || ! -x "$bin_dir/$(basename "$BINARY")" ]]; then
    echo "Error: serened binary not found or not executable: $BINARY" >&2
    exit 2
  fi
  BINARY="$bin_dir/$(basename "$BINARY")"
elif ! command -v "$BINARY" >/dev/null 2>&1; then
  echo "Error: serened binary not found on PATH: $BINARY" >&2
  exit 2
fi

# Pick a free ephemeral port from the kernel if one was not supplied.
if [[ -z "$PORT" ]]; then
  PORT="$("$PYTHON" - <<'PYEOF'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PYEOF
)"
fi
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
  echo "Error: invalid port: $PORT" >&2
  exit 2
fi

# Create a data directory if one was not supplied.
CREATED_DATA_DIR=0
if [[ -z "$DATA_DIR" ]]; then
  DATA_DIR="$(mktemp -d "${TMPDIR:-/tmp}/serenedb-test.XXXXXX")"
  CREATED_DATA_DIR=1
else
  mkdir -p "$DATA_DIR"
  DATA_DIR="$(cd "$DATA_DIR" && pwd)"
fi

LOG_FILE="$DATA_DIR/serened.log"
SERENED_PID=""

cleanup() {
  local rc=$?
  if [[ -n "$SERENED_PID" ]] && kill -0 "$SERENED_PID" 2>/dev/null; then
    kill "$SERENED_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$SERENED_PID" 2>/dev/null || break
      sleep 0.2
    done
    kill -9 "$SERENED_PID" 2>/dev/null || true
    wait "$SERENED_PID" 2>/dev/null || true
  fi
  if [[ "$CREATED_DATA_DIR" == "1" && -n "$DATA_DIR" ]]; then
    rm -rf "$DATA_DIR"
  fi
  return "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo ">> serened binary : $BINARY"
echo ">> listen         : postgres://127.0.0.1:$PORT"
echo ">> data dir       : $DATA_DIR$([[ "$CREATED_DATA_DIR" == 1 ]] && echo ' (temporary)')"
echo ">> python         : $PYTHON"

# Start the server.
"$BINARY" "$DATA_DIR" --listen="postgres://127.0.0.1:$PORT" >"$LOG_FILE" 2>&1 &
SERENED_PID=$!
echo ">> serened pid    : $SERENED_PID"

# Catch an immediate crash (bad binary/args) fast, rather than waiting out run.sh's
# full readiness timeout. Deeper readiness is run.sh's job.
sleep 0.3
if ! kill -0 "$SERENED_PID" 2>/dev/null; then
  echo "Error: serened exited on startup. Last log lines:" >&2
  tail -n 30 "$LOG_FILE" >&2
  exit 1
fi
echo

# Hand off to the instance-agnostic executor.
if [[ ${#PYTEST_ARGS[@]} -gt 0 ]]; then
  "$SCRIPT_DIR/run.sh" --host 127.0.0.1 --port "$PORT" -- "${PYTEST_ARGS[@]}"
else
  "$SCRIPT_DIR/run.sh" --host 127.0.0.1 --port "$PORT"
fi
rc=$?

# If the run failed because the server fell over, surface its log.
if [[ "$rc" -ne 0 ]] && ! kill -0 "$SERENED_PID" 2>/dev/null; then
  echo "Error: serened is no longer running. Last log lines:" >&2
  tail -n 30 "$LOG_FILE" >&2
fi

exit "$rc"
