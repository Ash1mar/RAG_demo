@echo off
setlocal

cd /d %~dp0\..

if "%PORT%"=="" (set "PORT=8000")

echo [check_net] Hostname:
hostname

echo.
echo [check_net] IPv4 addresses:
ipconfig | findstr /R /C:"IPv4.*:"

echo.
echo [check_net] Listening check (port %PORT%):
netstat -ano | findstr /R /C:":%PORT% "

echo.
echo [check_net] Local HTTP check (if the API is running):
curl -sS "http://127.0.0.1:%PORT%/docs" >nul 2>nul
if errorlevel 1 (
  echo   - curl failed (API may not be running yet, or curl not available). Try in PowerShell:
  echo     powershell -c "irm http://127.0.0.1:%PORT%/docs"
) else (
  echo   - OK: /docs reachable locally
)

echo.
echo [check_net] From the caller/server machine, run:
echo   - Test-NetConnection ^<API_PC_IP^> -Port %PORT%
echo   - curl http://^<API_PC_IP^>:%PORT%/docs

endlocal
