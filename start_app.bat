@echo off
setlocal enabledelayedexpansion

title PHENIX - Interactive Research Dashboard
echo =====================================================================
echo           PHENIX: Interactive Research Dashboard Launcher
echo =====================================================================
echo.

cd /d "%~dp0"

:: 1. Check for Virtual Environment (.venv, venv, or env)
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment ^(.venv^)...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment ^(venv^)...
    call venv\Scripts\activate.bat
) else if exist "env\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment ^(env^)...
    call env\Scripts\activate.bat
) else (
    echo [INFO] Using system Python environment...
)

:: 2. Verify Python availability
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found on your system PATH!
    echo Please install Python 3.10+ from https://www.python.org/ and check "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

:: 3. Verify Streamlit availability
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Streamlit is not installed in the active environment.
    echo [INFO] Installing required dependencies: streamlit, pandas, numpy, matplotlib...
    python -m pip install streamlit pandas numpy matplotlib
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
)

:: 4. Start the Streamlit Dashboard
echo.
echo [INFO] Starting PHENIX Streamlit Application...
echo [INFO] Dashboard URL: http://localhost:8501
echo [INFO] Close this window or press Ctrl+C to stop the server.
echo.

python -m streamlit run app.py --server.port=8501 --browser.gatherUsageStats=false

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application exited with error code: %errorlevel%
    pause
)

endlocal
