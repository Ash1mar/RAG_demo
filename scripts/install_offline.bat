@echo off
setlocal enabledelayedexpansion

REM Offline install for Windows intranet environments.
REM Assumptions:
REM - wheelhouse\ contains all required wheels for requirements.txt
REM - Python is already installed on this machine

cd /d %~dp0\..

set "VENV_DIR=.venv"
if not "%VENV_DIR_OVERRIDE%"=="" set "VENV_DIR=%VENV_DIR_OVERRIDE%"

if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo [install_offline] Creating venv: %VENV_DIR%
  python -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo [install_offline] ERROR: failed to create venv. Ensure Python is installed and on PATH.
    exit /b 1
  )
)

set "PY=%VENV_DIR%\Scripts\python.exe"

if not exist "wheelhouse" (
  echo [install_offline] ERROR: wheelhouse\ not found at: %cd%\wheelhouse
  echo [install_offline] Copy wheelhouse\ (pip download output) into project root first.
  exit /b 1
)

if not exist "requirements.txt" (
  echo [install_offline] ERROR: requirements.txt not found at project root.
  exit /b 1
)

echo [install_offline] Installing dependencies from wheelhouse...
"%PY%" -m pip install --no-index --find-links "wheelhouse" -r "requirements.txt"
if errorlevel 1 (
  echo [install_offline] ERROR: pip install failed. Common causes:
  echo   - wheelhouse missing packages for this Python version/Windows arch
  echo   - a package only has sdist and needs compilation
  exit /b 1
)

echo [install_offline] OK
endlocal
