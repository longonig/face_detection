#!/bin/bash

# Face Detection Web App Launcher with Auto Browser (Simple Version)
# This script starts the app and opens the browser with a simple delay

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

# Wait for the server to start with progress indicator
echo "Waiting for server to start..."
for i in {1..8}; do
    echo "Starting server... ($i/8 seconds)"
    sleep 1
done

echo "Server should be ready now!"

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
echo "Browser should open automatically to: http://localhost:5000"
echo ""
echo "If you see 'Connection Refused', wait a moment and refresh the page."
echo "Press Ctrl+C to stop the application"
echo ""

# Wait for the application to finish
wait $APP_PID

echo ""
echo "Application stopped."
read -p "Press Enter to close this window..."
