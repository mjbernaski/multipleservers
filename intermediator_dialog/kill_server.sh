#!/bin/bash

echo "Checking for running app.py instances..."

# Find processes running app.py
EXISTING_PIDS=$(pgrep -f "app.py" 2>/dev/null)

if [ -z "$EXISTING_PIDS" ]; then
    echo "No running instances found."
else
    echo "Found running process(es): $EXISTING_PIDS"
    echo "Killing processes..."
    kill $EXISTING_PIDS 2>/dev/null
    sleep 1
    # Force kill if still running
    kill -9 $EXISTING_PIDS 2>/dev/null
    echo "All instances terminated."
fi

# Also check port 5005 (default port)
if command -v lsof >/dev/null 2>&1; then
    PORT_PID=$(lsof -ti:5005 2>/dev/null)
    if [ ! -z "$PORT_PID" ]; then
        echo "Found process using port 5005: $PORT_PID"
        echo "Killing process on port 5005..."
        kill $PORT_PID 2>/dev/null
        sleep 1
        kill -9 $PORT_PID 2>/dev/null
        echo "Process on port 5005 terminated."
    fi
fi

echo "Done."
