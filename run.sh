#!/usr/bin/env bash
# Heavy Rainfall Early Warning System -- macOS / Linux launcher.
# The counterpart to run.bat: sets up a private environment on first run,
# then starts the dashboard. Run it with:  ./run.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PORT="${PORT:-8501}"
URL="http://localhost:${PORT}"

# --------------------------------------------------------------- find Python
PY=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PY="$candidate"
    break
  fi
done

if [ -z "$PY" ]; then
  echo "Python 3 was not found on this computer."
  if [ "$(uname -s)" = "Darwin" ]; then
    echo "Install it with:  brew install python"
    echo "or from https://www.python.org/downloads/"
  else
    echo "Install it with your package manager, e.g.  sudo apt install python3-venv"
  fi
  exit 1
fi

# ---------------------------------------------------- private virtual env
VPY=".venv/bin/python"
if [ ! -x "$VPY" ]; then
  echo "First run: setting up a private Python environment..."
  "$PY" -m venv .venv
fi

# --------------------------- install requirements (only when they change)
STAMP=".venv/installed-requirements.txt"
if ! cmp -s requirements.txt "$STAMP"; then
  echo "Installing required packages (this can take a few minutes)..."
  "$VPY" -m pip install --upgrade pip --quiet
  "$VPY" -m pip install -r requirements.txt
  cp requirements.txt "$STAMP"
fi

# ------------------------------------------------------------------ run app
echo
echo "Starting the Heavy Rainfall Early Warning dashboard..."
echo "The browser will open at ${URL}"
echo "Keep this window open. Press Ctrl+C here to stop the app."
echo

# .streamlit/config.toml runs the server headless, so open the browser
# ourselves once it is actually accepting connections.
if [ "$(uname -s)" = "Darwin" ]; then
  (
    for _ in $(seq 1 60); do
      if curl -sf -o /dev/null "$URL"; then
        open "$URL"
        break
      fi
      sleep 1
    done
  ) &
fi

exec "$VPY" -m streamlit run app.py --server.port "$PORT"
