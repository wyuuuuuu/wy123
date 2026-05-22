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
%PYTHON_EXE% -c "import networkx" >nul 2>&1
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

echo [INFO] Rebuilding KG tables from final reviewed triples...
%PYTHON_EXE% "kg\build_knowledge_graph.py" ^
  --triples "inputs\triples_all_reviewed.json" ^
  --dictionary "data\dictionaries\dictionary.csv" ^
  --output-dir "outputs\kg"

if errorlevel 1 (
    echo [ERROR] KG rebuild failed.
    pause
    exit /b 1
)

echo [INFO] KG rebuild completed. Outputs are in outputs\kg
pause
