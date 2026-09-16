#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "================================================================="
echo "  Starting Parallel Constrained Decision Engine (Local Apple Silicon)"
echo "  URL: http://localhost:8000"
echo "================================================================="

export PYTHONPATH="$DIR:$PYTHONPATH"
python3 -m uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
