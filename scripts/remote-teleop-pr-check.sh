#!/usr/bin/env bash

set -euo pipefail

REMOTE_TELEOP_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
REMOTE_TELEOP_INSTALL=1
REMOTE_TELEOP_FULL=0

usage() {
    printf '%s\n' \
        'Usage: scripts/remote-teleop-pr-check.sh [--verify-only] [--full]' \
        '' \
        '  --verify-only  Reuse an existing supported venv and frontend dependencies.' \
        '  --full         Run the complete backend and frontend suites after the smoke gate.'
}

while (($#)); do
    case "$1" in
        --verify-only)
            REMOTE_TELEOP_INSTALL=0
            ;;
        --full)
            REMOTE_TELEOP_FULL=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
    shift
done

cd "$REMOTE_TELEOP_REPO_ROOT"

command -v uv >/dev/null || {
    printf '%s\n' 'uv is required: https://docs.astral.sh/uv/' >&2
    exit 1
}
command -v npm >/dev/null || {
    printf '%s\n' 'npm is required to build the local UI.' >&2
    exit 1
}

if ((REMOTE_TELEOP_INSTALL)); then
    uv python install 3.12
    if [[ ! -x .venv/bin/python ]]; then
        uv venv --python 3.12 --managed-python
    fi
else
    [[ -x .venv/bin/python ]] || {
        printf '%s\n' '--verify-only requires an existing .venv.' >&2
        exit 1
    }
    [[ -d frontend/node_modules ]] || {
        printf '%s\n' '--verify-only requires existing frontend/node_modules.' >&2
        exit 1
    }
fi

.venv/bin/python - <<'PY'
import platform
import sys

if not (sys.version_info[:2] >= (3, 12) and sys.version_info[:2] < (3, 14)):
    raise SystemExit(
        f"Expected Python 3.12 or 3.13, got {platform.python_version()}. "
        "Move the incompatible .venv aside and rerun without --verify-only."
    )
if platform.system() == "Darwin" and platform.machine() != "arm64":
    raise SystemExit("The macOS trial requires native Apple Silicon Python, not Rosetta/x86_64.")
print(f"host={platform.system()} architecture={platform.machine()} python={platform.python_version()}")
PY

if ((REMOTE_TELEOP_INSTALL)); then
    uv pip install -e ".[dev]"
    npm --prefix frontend ci
    npm --prefix frontend audit --audit-level=high
fi

npm --prefix frontend run build
.venv/bin/python -m pytest -q \
    tests/test_remote_teleop_two_process.py \
    tests/test_remote_teleop_runtime_process.py

if ((REMOTE_TELEOP_FULL)); then
    .venv/bin/python -m pytest -q
    npm --prefix frontend test -- --run
    npm --prefix frontend exec tsc -- -p frontend/tsconfig.app.json --noEmit
    npm --prefix frontend exec tsc -- -p frontend/tsconfig.node.json --noEmit
fi

printf '%s\n' \
    'Remote teleoperation software gate passed.' \
    'Only ephemeral loopback test sockets were opened.' \
    'No MakerMods application listener or arm device was opened by this script.' \
    'Next: docs/remote-teleop/two-laptop-quickstart.md'
