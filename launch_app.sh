#!/bin/bash

# Face Detection Web App Launcher
# This script starts the Face Detection Web Application

echo "=============================================="
echo "    Face Detection Web Application"
echo "=============================================="
echo ""
echo "Starting the application..."
echo "Once started, open your web browser and go to:"
echo "  http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Change to the application directory
cd "/home/jetson/face_detection"

# Check if requirements are met
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found!"
    echo "Please make sure you're in the correct directory."
    read -p "Press Enter to exit..."
    exit 1
fi

if [ ! -f "run.py" ]; then
    echo "Error: run.py not found!"
    echo "Please make sure you're in the correct directory."
    read -p "Press Enter to exit..."
    exit 1
fi

# Start the application
echo "Launching Face Detection Web App..."
echo ""

python3 run.py

echo ""
echo "Application stopped."
read -p "Press Enter to close this window..."
