#!/usr/bin/env bash
# ATOBS - macOS/Linux launcher
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  . .venv/bin/activate
  pip install -r requirements.txt
else
  . .venv/bin/activate
fi
if [ ! -f data/atobs.db ]; then
  echo "Database missing. Run: python scripts/import_dataset.py --source path/to/datasetfinal.zip"
  exit 1
fi
echo "Open http://localhost:8000"
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
