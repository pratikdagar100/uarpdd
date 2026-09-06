@echo off
setlocal
REM Heavy Rainfall Early Warning System -- Windows launcher.
REM Double-click this file to start the app. It installs everything it
REM needs on first run, then opens the dashboard in your browser.

cd /d "%~dp0"
title Heavy Rainfall Early Warning

REM The pipeline's progress messages contain non-ASCII characters, which
REM the console's default codepage cannot encode -- printing one raises
REM UnicodeEncodeError and kills the run. UTF-8 everywhere avoids it.
chcp 65001 >nul 2>&1
set "PYTHONUTF8=1"

if "%PORT%"=="" set "PORT=8501"
set "URL=http://localhost:%PORT%"

REM ---------------------------------------------------------- find Python
set "PY="
py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY python --version >nul 2>&1 && set "PY=python"
if not defined PY (
  echo Python was not found on this computer.
  echo Install it from https://www.python.org/downloads/ ^(tick "Add
  echo python.exe to PATH" in the installer^), then run this file again.
  pause
  exit /b 1
)

REM ------------------------------------------------- private virtual env
if not exist ".venv\Scripts\python.exe" (
  echo First run: setting up a private Python environment...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo Could not create the virtual environment.
    pause
    exit /b 1
  )
)
set "VPY=.venv\Scripts\python.exe"

REM ------------------------- install requirements (only when they change)
fc /b requirements.txt ".venv\installed-requirements.txt" >nul 2>&1
if errorlevel 1 (
  echo Installing required packages ^(this can take a few minutes^)...
  "%VPY%" -m pip install --upgrade pip --quiet
  "%VPY%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo Package installation failed. Check your internet connection
    echo and run this file again.
    pause
    exit /b 1
  )
  copy /y requirements.txt ".venv\installed-requirements.txt" >nul
)

REM ------------------------------------------------------------- run app
echo.
echo Starting the Heavy Rainfall Early Warning dashboard...
echo The browser will open at %URL%
echo Keep this window open. Press Ctrl+C here to stop the app.
echo.

REM .streamlit\config.toml runs the server headless, so it will not open a
REM browser itself. Poll in the background and open one once the server is
REM actually accepting connections.
REM Kept to one line and free of pipes: cmd parses `|` and `^` before it
REM hands the string to PowerShell, so both would need escaping here.
start "" /b powershell -NoProfile -ExecutionPolicy Bypass -Command "for($i=0; $i -lt 90; $i++){ try{ $null = Invoke-WebRequest -UseBasicParsing -Uri '%URL%' -TimeoutSec 2; Start-Process '%URL%'; break } catch { Start-Sleep -Seconds 1 } }"

"%VPY%" -m streamlit run app.py --server.port %PORT%

if errorlevel 1 (
  echo.
  echo The app exited with an error ^(details above^).
  pause
)
endlocal
