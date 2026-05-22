@echo off
setlocal
chcp 65001 >nul

cd /d "%~dp0"

echo [INFO] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not added to PATH.
    pause
    exit /b 1
)

set "PYTHON_EXE=python"

if exist ".venv\Scripts\python.exe" if not exist ".venv\Scripts\pip.exe" (
    echo [INFO] Existing virtual environment is incomplete. Recreating it...
    rmdir /s /q ".venv"
)

if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Creating local virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [WARN] Failed to create virtual environment. Falling back to system Python.
    )
)

if exist ".venv\Scripts\python.exe" if exist ".venv\Scripts\pip.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
)

echo [INFO] Installing dependencies with %PYTHON_EXE%...
"%PYTHON_EXE%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    echo Please check network access, or run manually:
    echo %PYTHON_EXE% -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo [INFO] Done. Next run 03_build_topics_and_start_teaching_app.bat
pause
