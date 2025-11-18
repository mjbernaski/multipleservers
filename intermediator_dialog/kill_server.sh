#!/bin/bash

echo "Checking for running intermediator_dialog.py instances..."

# Find processes running intermediator_dialog.py
EXISTING_PIDS=$(pgrep -f "intermediator_dialog.py" 2>/dev/null)

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

# Also check port 5006 (default port)
if command -v lsof >/dev/null 2>&1; then
    PORT_PID=$(lsof -ti:5006 2>/dev/null)
    if [ ! -z "$PORT_PID" ]; then
        echo "Found process using port 5006: $PORT_PID"
        echo "Killing process on port 5006..."
        kill $PORT_PID 2>/dev/null
        sleep 1
        kill -9 $PORT_PID 2>/dev/null
        echo "Process on port 5006 terminated."
    fi
fi

echo "Done."
