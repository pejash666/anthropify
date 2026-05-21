#!/usr/bin/env bash
# Loads .env.e2e into the environment and runs the E2E suite. We rely on
# `set -a` rather than a Go dotenv library to keep the project dep-free.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env.e2e ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env.e2e
    set +a
else
    echo "warn: .env.e2e not found; tests will skip every provider" >&2
fi

# Forward extra args (-run, -count, etc.) verbatim.
exec go test -tags=e2e ./e2e/... -count=1 -v "$@"
