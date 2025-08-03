import os

class Config:
    """Application configuration"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    DEBUG = True
    
    # Camera settings - optimized for streaming performance
    DEFAULT_CAMERA_INDEX = 0
    CAMERA_WIDTH = 960  # Reduced from 1280 for faster processing
    CAMERA_HEIGHT = 540  # Reduced from 720 for faster processing  
    CAMERA_FPS = 30
    
    # Detection settings
    FACE_CONFIDENCE_THRESHOLD = 0.9
    FRAME_SKIP_RATE = 3  # Process every 3rd frame for better GPU utilization
    JPEG_QUALITY = 70  # Reduced for faster encoding and streaming
    SHOW_FACIAL_LANDMARKS = True  # Enable facial landmark tracking visualization
    LANDMARK_TRIANGULATION = True  # Show triangular mesh like in movies
    
    # Performance settings
    MAX_FACES_TO_DISPLAY = 10
    FACE_UPDATE_INTERVAL = 0.1  # seconds - faster updates for smoother streaming
    
    # Streaming performance optimizations
    STREAM_FPS = 45  # Target streaming frame rate
    STREAM_BUFFER_SIZE = 2  # Smaller buffer for lower latency
    
    # Device settings - optimized for Jetson Orin
    FORCE_CPU = False  # Set to True to force CPU usage
    GPU_MEMORY_FRACTION = 0.7  # Use 70% of GPU memory (conservative for stability)
    ENABLE_MIXED_PRECISION = False  # Disable mixed precision for stability on Jetson
    
    # Performance optimizations for Jetson
    JETSON_OPTIMIZATION = True  # Enable Jetson-specific optimizations
    USE_TENSORRT = False  # Enable TensorRT acceleration if available
    
    # Web interface settings
    HOST = '0.0.0.0'
    PORT = 5000
