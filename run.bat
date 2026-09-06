@echo off
setlocal
REM Uncertainty-Aware Rainfall Prediction & Decision Dashboard
REM Double-click this file to start the app. It installs everything it
REM needs on first run, then opens the dashboard in your browser.

cd /d "%~dp0"
title Rainfall Uncertainty Lab

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
echo Starting the Rainfall Uncertainty Lab dashboard...
echo The browser will open at http://localhost:8501
echo Keep this window open. Press Ctrl+C here to stop the app.
echo.

"%VPY%" -m streamlit run app.py --server.port 8501

if errorlevel 1 (
  echo.
  echo The app exited with an error ^(details above^).
  pause
)
endlocal
