#!/usr/bin/env python3
"""
Run script for Face Detection Web Application

This script starts the Flask web server for the face detection application.
It includes proper error handling and graceful shutdown.
"""

import os
import sys
import signal
import atexit
from app import app, face_detector
from config import Config

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    print('\nShutting down gracefully...')
    face_detector.stop_camera()
    sys.exit(0)

def cleanup():
    """Cleanup function called on exit"""
    face_detector.stop_camera()

if __name__ == '__main__':
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup)
    
    print("=" * 50)
    print("🔍 AI Face Detection Web Application")
    print("=" * 50)
    print(f"Starting server on http://{Config.HOST}:{Config.PORT}")
    print(f"Device: {face_detector.device}")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        app.run(
            host=Config.HOST,
            port=Config.PORT,
            debug=Config.DEBUG,
            threaded=True,
            use_reloader=False  # Disable reloader to avoid camera conflicts
        )
    except Exception as e:
        print(f"Error starting server: {e}")
        face_detector.stop_camera()
        sys.exit(1)
