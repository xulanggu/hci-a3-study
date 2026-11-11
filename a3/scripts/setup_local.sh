#!/usr/bin/env bash
# Run from any directory. Network access is required on the first run.
set -euo pipefail
A3_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
A3_PYTHON="${A3_PYTHON:-python3.11}"
"$A3_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,11), "Python 3.11 is required; set A3_PYTHON to its executable."'
"$A3_PYTHON" -m venv "$A3_ROOT/.venv"
"$A3_ROOT/.venv/bin/python" -m pip install -r "$A3_ROOT/requirements-dev.txt"
"$A3_ROOT/.venv/bin/python" -m pip check
"$A3_ROOT/.venv/bin/python" "$A3_ROOT/scripts/verify_model.py"
"$A3_ROOT/.venv/bin/python" -m pytest "$A3_ROOT/tests" -q
if [[ "${1:-}" == "--serve" ]]; then
  cd "$A3_ROOT"
  export A3_STORAGE_BACKEND=sqlite
  exec "$A3_ROOT/.venv/bin/python" -m streamlit run app.py
fi
