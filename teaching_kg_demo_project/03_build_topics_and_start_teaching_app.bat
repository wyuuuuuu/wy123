@echo off
setlocal
chcp 65001 >nul

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" if exist ".venv\Scripts\pip.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo [INFO] Checking Python dependencies...
%PYTHON_EXE% -c "import pandas, networkx, streamlit" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Missing dependencies detected. Installing requirements...
    %PYTHON_EXE% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        echo Please run 01_install_dependencies.bat or install packages manually:
        echo %PYTHON_EXE% -m pip install -r requirements.txt
        pause
        exit /b 1
    )
)

echo [INFO] Building teaching topic clusters...
%PYTHON_EXE% "kg\build_topic_clusters.py"
if errorlevel 1 (
    echo [ERROR] Failed to build topic clusters.
    pause
    exit /b 1
)

echo [INFO] Starting Streamlit teaching KG prototype...
%PYTHON_EXE% -m streamlit run "kg\teaching_kg_app.py"

if errorlevel 1 (
    echo [ERROR] Failed to start Streamlit app.
    echo Please run 01_install_dependencies.bat first.
    pause
    exit /b 1
)
