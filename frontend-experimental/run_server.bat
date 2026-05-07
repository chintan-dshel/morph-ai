@echo off
cd /d "%~dp0"
echo Starting MorphAI...
echo Open http://localhost:8000 in your browser
echo.
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
pause
