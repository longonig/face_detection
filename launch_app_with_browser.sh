#!/bin/bash

# Face Detection Web App Launcher with Auto Browser
# This script starts the app and opens the browser automatically

echo "=============================================="
echo "    Face Detection Web Application"
echo "=============================================="
echo ""
echo "Starting the application and opening browser..."
echo ""

# Change to the application directory
cd "/home/jetson/face_detection"

# Start the application in the background
echo "Launching Face Detection Web App..."
python3 run.py &
APP_PID=$!

# Wait for the server to be ready
echo "Waiting for server to start..."
SERVER_READY=false
MAX_ATTEMPTS=30  # Maximum 30 seconds wait time
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ] && [ "$SERVER_READY" = false ]; do
    sleep 1
    ATTEMPT=$((ATTEMPT + 1))
    
    # Check if server is responding
    if curl -s http://localhost:5000/health > /dev/null 2>&1; then
        SERVER_READY=true
        echo "Server is ready! (took ${ATTEMPT} seconds)"
    elif [ $((ATTEMPT % 5)) -eq 0 ]; then
        echo "Still waiting... (${ATTEMPT}/${MAX_ATTEMPTS} seconds)"
    fi
done

if [ "$SERVER_READY" = false ]; then
    echo "Warning: Server may not be ready yet, but opening browser anyway..."
fi

# Try to open the browser
echo "Opening browser..."
if command -v firefox &> /dev/null; then
    firefox http://localhost:5000 &
elif command -v google-chrome &> /dev/null; then
    google-chrome http://localhost:5000 &
elif command -v chromium-browser &> /dev/null; then
    chromium-browser http://localhost:5000 &
elif command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:5000 &
else
    echo "No browser found. Please manually open: http://localhost:5000"
fi

echo ""
echo "Application is running!"
echo "Browser should open automatically."
echo "If not, manually open: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Wait for the application to finish
wait $APP_PID

echo ""
echo "Application stopped."
read -p "Press Enter to close this window..."
