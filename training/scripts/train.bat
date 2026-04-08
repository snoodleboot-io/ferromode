@echo off
REM Train LSTM model (Windows)

setlocal enabledelayedexpansion
cd /d "%~dp0\.."

REM Check if venv is activated
if "!VIRTUAL_ENV!"=="" (
    if exist ".venv\Scripts\activate.bat" (
        call .venv\Scripts\activate.bat
    ) else (
        echo Error: Virtual environment not found. Run setup.bat first.
        exit /b 1
    )
)

REM Run training
echo Starting LSTM training...
python -m training.train %*
