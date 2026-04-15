@echo off
:: Use the Python Launcher to pin Python 3.12 (the env that has Streamlit installed).
:: 'python' alone may resolve to a different version if multiple Pythons are on PATH.
py -3.12 -c "import streamlit" 2>nul
if errorlevel 1 (
    echo ERROR: Streamlit not found in Python 3.12.
    echo Run:  py -3.12 -m pip install -r requirements.txt
    pause
    exit /b 1
)
start "" "http://localhost:8501"
py -3.12 -m streamlit run app.py --server.headless true
pause