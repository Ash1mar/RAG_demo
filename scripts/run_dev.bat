@echo off
setlocal enabledelayedexpansion

REM Start FastAPI for LAN access (bind 0.0.0.0) with centralized config file.

cd /d %~dp0\..

set "VENV_DIR=.venv"
if not "%VENV_DIR_OVERRIDE%"=="" set "VENV_DIR=%VENV_DIR_OVERRIDE%"

if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo [run_dev] ERROR: venv not found: %cd%\%VENV_DIR%
  echo [run_dev] Run scripts\install_offline.bat first.
  exit /b 1
)

set "PY=%VENV_DIR%\Scripts\python.exe"

REM Centralized env file loader uses APP_CONFIG (see app\config_loader.py)
set "APP_CONFIG=config\app.env"

set "HOST=0.0.0.0"
if "%PORT%"=="" (set "PORT=8000")

echo [run_dev] APP_CONFIG=%APP_CONFIG%
echo [run_dev] Listening on http://%HOST%:%PORT%  (LAN-accessible)

"%PY%" -m uvicorn app.demo_app:app --host "%HOST%" --port "%PORT%" --reload
endlocal
