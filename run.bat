@echo off
REM ATOBS - Windows launcher
cd /d %~dp0
if not exist .venv (
  python -m venv .venv
  call .venv\Scripts\activate
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate
)
if not exist data\atobs.db (
  echo Database missing. Run: python scripts\import_dataset.py --source path\to\datasetfinal.zip
  pause
  exit /b 1
)
echo Open http://localhost:8000
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
