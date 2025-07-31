#!/usr/bin/env python3
"""
Test script to verify the camera fix works with the application's camera class
"""
import sys
import os
sys.path.append('/home/jetson/face_detection')

import cv2
from config import Config

def test_app_camera_fix():
    """Test the fixed camera initialization"""
    print("Testing fixed camera initialization...")
    
    # Simulate the fixed start_camera method
    camera_index = Config.DEFAULT_CAMERA_INDEX
    
    # Use device path for better compatibility (the fix)
    if isinstance(camera_index, int):
        device_path = f"/dev/video{camera_index}"
        cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(camera_index)
        
    if not cap.isOpened():
        print("Failed to open camera")
        return False
    
    print("Camera opened successfully!")
    
    # Set MJPEG format for better performance with USB cameras
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    
    # Set camera properties from config
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
    
    # Add buffer size setting to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Get actual values
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera settings applied:")
    print(f"  Resolution: {int(width)}x{int(height)}")
    print(f"  FPS: {fps}")
    
    # Test frame capture
    ret, frame = cap.read()
    if ret:
        print(f"Frame captured successfully: {frame.shape}")
        print(f"Frame mean value: {frame.mean()}")
        
        if frame.mean() > 10:
            print("SUCCESS: Frame has valid image data!")
        else:
            print("WARNING: Frame appears dark")
    else:
        print("Failed to capture frame")
        
    cap.release()
    return ret

if __name__ == "__main__":
    success = test_app_camera_fix()
    if success:
        print("\n✅ Camera fix test PASSED! Your application should now work.")
    else:
        print("\n❌ Camera fix test FAILED.")
