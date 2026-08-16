@echo off
REM Uncertainty-Aware Rainfall Prediction & Decision Dashboard
REM Double-click this file (or run it from a terminal) to start the app.

cd /d "%~dp0"

echo Starting the Rainfall Uncertainty Lab dashboard...
echo The browser should open at http://localhost:8501
echo Press Ctrl+C in this window to stop the app.
echo.

python -m streamlit run app.py

REM Keep the window open if Streamlit exits with an error
if errorlevel 1 (
  echo.
  echo The app exited with an error. If a package is missing, run:
  echo     pip install -r requirements.txt
  pause
)
