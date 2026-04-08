@echo off
setlocal enabledelayedexpansion

echo.
echo ========================================
echo Ferromode Training - Setup
echo ========================================
echo.

REM Check Python version
echo Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found
    echo Install from: https://www.python.org/downloads/
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python: %PYTHON_VERSION%

REM Check Python 3.12+
python -c "import sys; exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1
if errorlevel 1 (
    echo Error: Python 3.12+ required
    echo Current version: %PYTHON_VERSION%
    echo Install from: https://www.python.org/downloads/
    exit /b 1
)

echo.✓ Python version OK
echo.

REM Create virtual environment
echo Creating virtual environment...
if exist ".venv" (
    echo Virtual environment already exists. Skipping...
) else (
    python -m venv .venv
    echo.✓ Virtual environment created
)

echo.

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.✓ Virtual environment activated

echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel >nul 2>&1
echo.✓ pip upgraded

echo.

REM Install dependencies
echo Installing dependencies from requirements.txt...
echo This may take 5-10 minutes...
pip install -r requirements.txt

echo.
echo ========================================
echo.✓ Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Activate environment: .venv\Scripts\activate.bat
echo   2. Run training: scripts\train.bat
echo.
