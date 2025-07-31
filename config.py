import os

class Config:
    """Application configuration"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    DEBUG = True
    
    # Camera settings
    DEFAULT_CAMERA_INDEX = 0
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 30
    
    # Detection settings
    FACE_CONFIDENCE_THRESHOLD = 0.9
    FRAME_SKIP_RATE = 3  # Process every 3rd frame for better GPU utilization
    JPEG_QUALITY = 85
    SHOW_FACIAL_LANDMARKS = True  # Enable facial landmark tracking visualization
    LANDMARK_TRIANGULATION = True  # Show triangular mesh like in movies
    
    # Performance settings
    MAX_FACES_TO_DISPLAY = 10
    FACE_UPDATE_INTERVAL = 0.3  # seconds - faster updates for GPU
    
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
