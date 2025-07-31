#!/usr/bin/env python3
import cv2

def test_camera_with_settings():
    """Test camera with specific settings for EMEET SmartCam C960 4K"""
    device_path = "/dev/video0"
    
    print("Testing camera with MJPEG format and higher resolution...")
    
    cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print("Failed to open camera")
        return False
        
    # Set MJPEG format for better performance with USB cameras
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    
    # Try 1280x720 at 60fps (supported by the camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    # Get actual values
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cap.get(cv2.CAP_PROP_FOURCC)
    
    print(f"Actual settings:")
    print(f"  Resolution: {int(width)}x{int(height)}")
    print(f"  FPS: {fps}")
    print(f"  FOURCC: {int(fourcc)} ({chr(int(fourcc) & 0xFF)}{chr((int(fourcc) >> 8) & 0xFF)}{chr((int(fourcc) >> 16) & 0xFF)}{chr((int(fourcc) >> 24) & 0xFF)})")
    
    # Test frame capture
    ret, frame = cap.read()
    if ret:
        print(f"Frame captured: {frame.shape}")
        print(f"Frame mean value: {frame.mean()}")
        
        # Try a few more frames to make sure it's consistent
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                print(f"Frame {i+2} mean: {frame.mean()}")
            else:
                print(f"Failed to capture frame {i+2}")
    else:
        print("Failed to capture frame")
        
    cap.release()
    return ret

if __name__ == "__main__":
    test_camera_with_settings()
