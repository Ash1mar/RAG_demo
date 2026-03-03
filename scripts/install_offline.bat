@echo off
setlocal

REM Go to project root (script_dir\..)
pushd "%~dp0\.."

set "VENV_DIR=.venv"
if defined VENV_DIR_OVERRIDE set "VENV_DIR=%VENV_DIR_OVERRIDE%"

set "PY=%VENV_DIR%\Scripts\python.exe"

if exist "%PY%" goto :venv_ok
echo [install_offline] Creating venv: %VENV_DIR%
python -m venv "%VENV_DIR%"
if errorlevel 1 goto :venv_fail

:venv_ok
if not exist "wheelhouse" goto :no_wheelhouse
if not exist "requirements.txt" goto :no_requirements

echo [install_offline] Installing dependencies from wheelhouse...
"%PY%" -m pip install --no-index --find-links "wheelhouse" -r "requirements.txt"
if errorlevel 1 goto :pip_fail

echo [install_offline] OK
goto :end

:venv_fail
echo [install_offline] ERROR: failed to create venv. Ensure Python is installed and on PATH.
exit /b 1

:no_wheelhouse
echo [install_offline] ERROR: wheelhouse\ not found at: %cd%\wheelhouse
echo [install_offline] Copy wheelhouse\ (pip download output) into project root first.
exit /b 1

:no_requirements
echo [install_offline] ERROR: requirements.txt not found at project root.
exit /b 1

:pip_fail
echo [install_offline] ERROR: pip install failed. Common causes:
echo   - wheelhouse missing packages for this Python version/Windows arch
echo   - a package only has sdist and needs compilation
exit /b 1

:end
popd
endlocal