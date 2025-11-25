#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "../venv" ]; then
    source ../venv/bin/activate
fi

# Default host and port (can be overridden with command line args)
HOST="${1:-0.0.0.0}"
PORT="${2:-5006}"
AUDIO_PORT=5002

# URL for the web interface
URL="http://${HOST}:${PORT}"

# Check for and kill any existing instance
echo "Checking for existing server instances..."

# Find processes running app.py
EXISTING_PIDS=$(pgrep -f "app.py" 2>/dev/null)

if [ ! -z "$EXISTING_PIDS" ]; then
    echo "Found existing server process(es): $EXISTING_PIDS"
    echo "Killing existing process(es)..."
    kill $EXISTING_PIDS 2>/dev/null
    sleep 1
    kill -9 $EXISTING_PIDS 2>/dev/null
    echo "Existing process(es) terminated"
fi

# Also check if ports are in use and kill the processes using them
if command -v lsof >/dev/null 2>&1; then
    for CHECK_PORT in $PORT $AUDIO_PORT; do
        PORT_PID=$(lsof -ti:$CHECK_PORT 2>/dev/null)
        if [ ! -z "$PORT_PID" ]; then
            echo "Found process using port $CHECK_PORT: $PORT_PID"
            echo "Killing process on port $CHECK_PORT..."
            kill $PORT_PID 2>/dev/null
            sleep 1
            kill -9 $PORT_PID 2>/dev/null
            echo "Process on port $CHECK_PORT terminated"
        fi
    done
fi

echo "Starting Intermediated Dialog (IDi) server..."
echo "URL: $URL"

# Determine Python command for main app
if [ -d ".venv" ]; then
    PYTHON_CMD=".venv/bin/python3"
elif [ -d "venv" ]; then
    PYTHON_CMD="venv/bin/python3"
elif [ -d "../venv" ]; then
    PYTHON_CMD="../venv/bin/python3"
else
    PYTHON_CMD="python3"
fi

# Run the main Python script in the background
$PYTHON_CMD app.py --host "$HOST" --port "$PORT" &
SERVER_PID=$!

# Start the audio player server (uses its own venv)
AUDIO_PLAYER_DIR="$SCRIPT_DIR/output/audio/audio_player_py"
AUDIO_PYTHON_CMD="$AUDIO_PLAYER_DIR/venv/bin/python3"

if [ -d "$AUDIO_PLAYER_DIR" ] && [ -f "$AUDIO_PYTHON_CMD" ]; then
    echo "Starting Audio Player server on port $AUDIO_PORT..."
    cd "$AUDIO_PLAYER_DIR"
    $AUDIO_PYTHON_CMD app.py &
    AUDIO_PID=$!
    cd "$SCRIPT_DIR"
    echo "Audio Player running (PID: $AUDIO_PID) - http://localhost:$AUDIO_PORT"
else
    echo "Warning: Audio player venv not found at $AUDIO_PLAYER_DIR"
    AUDIO_PID=""
fi

# Wait a moment for the servers to start
sleep 2

# Check if the main server is still running (it might have failed to start)
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Error: Main server failed to start"
    [ ! -z "$AUDIO_PID" ] && kill $AUDIO_PID 2>/dev/null
    exit 1
fi

# Open the web page in the default browser
echo "Opening web page in browser..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if [[ "$HOST" == "0.0.0.0" ]]; then
        BROWSER_URL="http://localhost:${PORT}"
    else
        BROWSER_URL="$URL"
    fi
    open "$BROWSER_URL"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    xdg-open "$URL" 2>/dev/null || echo "Please open $URL in your browser"
else
    echo "Please open $URL in your browser"
fi

echo ""
echo "============================================"
echo "Servers running:"
echo "  Main IDi:      http://localhost:$PORT (PID: $SERVER_PID)"
[ ! -z "$AUDIO_PID" ] && echo "  Audio Player:  http://localhost:$AUDIO_PORT (PID: $AUDIO_PID)"
echo "============================================"
echo "Press Ctrl+C to stop all servers"
echo ""

# Cleanup function to kill both servers
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $SERVER_PID 2>/dev/null
    [ ! -z "$AUDIO_PID" ] && kill $AUDIO_PID 2>/dev/null
    wait
    echo "All servers stopped."
    exit 0
}

# Trap Ctrl+C and call cleanup
trap cleanup SIGINT SIGTERM

# Wait for the main server process
wait $SERVER_PID
