#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export PYTHONDONTWRITEBYTECODE=1
exec "$repo/.venv-linux/bin/python" -B "$repo/tools/linux_runtime/launch.py" native "$@"
