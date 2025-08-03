# AI Face Detection Web Application

A real-time face detection web application built with Flask, OpenCV, and PyTorch. This application transforms your existing face detection script into a modern web interface that can be accessed from any browser.

## Features

- 🎥 **Real-time Video Feed**: Live camera streaming with face detection overlay
- 👥 **Face Detection**: Advanced MTCNN-based face detection with confidence scoring
- 🎯 **Interactive Interface**: Modern, responsive web interface with Bootstrap styling
- 📊 **Live Statistics**: Real-time face count monitoring
- 🖼️ **Face Gallery**: Display detected faces with confidence scores and positions
- ⚡ **Performance Optimized**: Configurable frame skipping and quality settings
- 🔧 **Easy Configuration**: Centralized configuration for all settings

## Requirements

- Python 3.7+
- Webcam or camera device
- Modern web browser

## Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd face-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify your camera is working**:
   - Ensure your camera is connected and not being used by other applications
   - The application will try to use camera index 0 by default (usually the built-in camera)

## Running the Application

### Method 1: Using the run script (Recommended)
```bash
python run.py
```

### Method 2: Direct Flask execution
```bash
python app.py
```

### Method 3: Using Flask command line
```bash
flask --app app run --host=0.0.0.0 --port=5000
```

## Usage

1. **Start the application** using one of the methods above
2. **Open your web browser** and navigate to:
   ```
   http://localhost:5000
   ```
3. **Click "Start Detection"** to begin real-time face detection
4. **View the results**:
   - Live video feed with face detection boxes
   - Real-time statistics (face count)
   - Gallery of detected faces with confidence scores
5. **Click "Stop Detection"** when finished

## Configuration

Edit `config.py` to customize the application:

```python
# Camera settings
DEFAULT_CAMERA_INDEX = 0        # Change camera source
CAMERA_WIDTH = 640              # Camera resolution width
CAMERA_HEIGHT = 480             # Camera resolution height

# Detection settings
FACE_CONFIDENCE_THRESHOLD = 0.9 # Minimum confidence for face detection
FRAME_SKIP_RATE = 5            # Process every Nth frame (higher = faster)

# Performance settings
JPEG_QUALITY = 85              # Video stream quality (1-100)
MAX_FACES_TO_DISPLAY = 10      # Maximum faces to show in gallery

# Server settings
HOST = '0.0.0.0'               # Server host (0.0.0.0 for all interfaces)
PORT = 5000                    # Server port
```

## Troubleshooting

### Camera Issues
- **Camera not found**: Try changing `DEFAULT_CAMERA_INDEX` in `config.py` (try 0, 1, 2...)
- **Permission denied**: Ensure camera permissions are granted to your browser/system
- **Camera in use**: Close other applications that might be using the camera

### Performance Issues
- **Slow detection**: Increase `FRAME_SKIP_RATE` in `config.py`
- **High CPU usage**: Set `FORCE_CPU = True` in `config.py` if using GPU causes issues
- **Poor performance**: Reduce `CAMERA_WIDTH` and `CAMERA_HEIGHT`, increase `FRAME_SKIP_RATE`

### Web Interface Issues
- **Page not loading**: Check if the server started successfully and no firewall is blocking port 5000
- **Video not showing**: Ensure camera permissions and try refreshing the page
- **Alerts not working**: Check browser console for JavaScript errors

## API Endpoints

The application provides several REST API endpoints:

- `GET /` - Main web interface
- `POST /start_detection` - Start face detection
- `POST /stop_detection` - Stop face detection
- `GET /get_faces` - Get current detected faces data
- `GET /video_feed` - Video stream endpoint
- `GET /health` - Health check endpoint

## Technical Details

### Architecture
- **Backend**: Flask web server with threading support
- **Frontend**: Bootstrap 5 + vanilla JavaScript
- **AI/ML**: PyTorch + MTCNN for face detection
- **Video**: OpenCV for camera capture and processing

### Browser Compatibility
- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge

### Security Notes
- The application runs on all network interfaces (`0.0.0.0`) by default
- In production, consider using HTTPS and proper authentication
- Camera access requires appropriate permissions

## Original Application

This web interface is based on the original `AI_video_targeting.py` script. The core face detection logic remains the same, but now with:
- Web-based interface instead of desktop window
- REST API for remote control
- Better error handling and user feedback
- Configurable settings
- Multiple device support through the browser

## License

This project maintains the same license as the original face detection script.

---

**Happy Face Detecting! 🎯👨‍💻**
