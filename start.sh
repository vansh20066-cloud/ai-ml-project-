#!/bin/bash

# AI Mental Health Platform - Startup Script
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="/opt/homebrew/bin:/Users/vanshsinghal/.gemini/antigravity/scratch/node/bin:$PATH"

echo "================================================="
echo " Starting AI Mental Health Platform"
echo "================================================="

# 0. Ensure Ollama is running
echo "Checking Ollama status..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama..."
    if command -v ollama > /dev/null 2>&1; then
        ollama serve > /dev/null 2>&1 &
        sleep 2
    elif [ -d "/Applications/Ollama.app" ]; then
        open -a Ollama
        sleep 3
    else
        echo "Warning: Ollama not found. Empathetic fallbacks will be used."
    fi
else
    echo "✓ Ollama is running."
fi

# 1. Start Backend (FastAPI on port 8000)
echo "Starting FastAPI Backend on http://127.0.0.1:8000..."
cd "$ROOT_DIR"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
UVICORN_BIN="$ROOT_DIR/.venv/bin/uvicorn"

if [ -f "$UVICORN_BIN" ]; then
    "$UVICORN_BIN" app.main:app --host 127.0.0.1 --port 8000 &
    BACKEND_PID=$!
else
    python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000 &
    BACKEND_PID=$!
fi

# 2. Start Frontend (Node.js Express on port 3000)
echo "Starting Node.js Frontend on http://localhost:3000..."
cd "$ROOT_DIR/app"
node server.js &
FRONTEND_PID=$!

echo ""
echo "Platform is running!"
echo "  - Frontend UI: http://localhost:3000"
echo "  - Backend API: http://127.0.0.1:8000"
echo "  - API Docs:    http://127.0.0.1:8000/docs"
echo ""
echo "Press CTRL+C to stop all servers."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
wait
