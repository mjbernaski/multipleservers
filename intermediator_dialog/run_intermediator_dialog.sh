#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d "../venv" ]; then
    source ../venv/bin/activate
fi

# Default host and port (can be overridden with command line args)
HOST="${1:-0.0.0.0}"
PORT="${2:-5006}"

# URL for the web interface
URL="http://${HOST}:${PORT}"

# Check for and kill any existing instance
echo "Checking for existing server instances..."

# Find processes running intermediator_dialog.py
EXISTING_PIDS=$(pgrep -f "intermediator_dialog.py" 2>/dev/null)

if [ ! -z "$EXISTING_PIDS" ]; then
    echo "Found existing server process(es): $EXISTING_PIDS"
    echo "Killing existing process(es)..."
    kill $EXISTING_PIDS 2>/dev/null
    sleep 1
    kill -9 $EXISTING_PIDS 2>/dev/null
    echo "Existing process(es) terminated"
fi

# Also check if port is in use and kill the process using it
if command -v lsof >/dev/null 2>&1; then
    PORT_PID=$(lsof -ti:$PORT 2>/dev/null)
    if [ ! -z "$PORT_PID" ]; then
        echo "Found process using port $PORT: $PORT_PID"
        echo "Killing process on port $PORT..."
        kill $PORT_PID 2>/dev/null
        sleep 1
        kill -9 $PORT_PID 2>/dev/null
        echo "Process on port $PORT terminated"
    fi
fi

echo "Starting Intermediated Dialog (IDi) server..."
echo "URL: $URL"

# Run the Python script in the background
python intermediator_dialog.py --host "$HOST" --port "$PORT" &
SERVER_PID=$!

# Wait a moment for the server to start
sleep 2

# Check if the server is still running (it might have failed to start)
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Error: Server failed to start"
    exit 1
fi

# Open the web page in the default browser
echo "Opening web page in browser..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open "$URL"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    xdg-open "$URL" 2>/dev/null || echo "Please open $URL in your browser"
else
    echo "Please open $URL in your browser"
fi

echo ""
echo "Server is running (PID: $SERVER_PID)"
echo "Press Ctrl+C to stop the server"
echo ""

# Wait for the server process
wait $SERVER_PID
