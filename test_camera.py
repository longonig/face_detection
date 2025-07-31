#!/usr/bin/env python3
import cv2
import sys

def test_camera(device_index):
    """Test camera functionality"""
    device_path = f"/dev/video{device_index}"
    print(f"Testing camera at {device_path}")
    
    # Try different backends and methods
    test_methods = [
        (device_path, cv2.CAP_V4L2, "V4L2 with device path"),
        (device_index, cv2.CAP_V4L2, "V4L2 with index"),
        (device_path, cv2.CAP_GSTREAMER, "GStreamer with device path"),
        (device_index, cv2.CAP_GSTREAMER, "GStreamer with index"),
        (device_path, cv2.CAP_ANY, "Default with device path"),
        (device_index, cv2.CAP_ANY, "Default with index")
    ]
    
    for device, backend, backend_name in test_methods:
        print(f"\nTrying {backend_name}...")
        
        cap = cv2.VideoCapture(device, backend)
        
        if not cap.isOpened():
            print(f"  Failed to open camera with {backend_name}")
            continue
            
        # Get camera properties
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        
        print(f"  {backend_name} - Camera opened successfully!")
        print(f"  Resolution: {int(width)}x{int(height)}")
        print(f"  FPS: {fps}")
        print(f"  FOURCC: {int(fourcc)}")
        
        # Try to read a frame
        ret, frame = cap.read()
        if ret:
            print(f"  Frame captured successfully! Shape: {frame.shape}")
            print(f"  Frame data type: {frame.dtype}")
            print(f"  Frame min/max values: {frame.min()}/{frame.max()}")
            
            # Check if frame is black
            mean_value = frame.mean()
            print(f"  Frame mean value: {mean_value}")
            if mean_value < 10:
                print("  WARNING: Frame appears to be black or very dark!")
            else:
                print("  Frame appears to have valid image data")
        else:
            print(f"  Failed to capture frame with {backend_name}")
            
        cap.release()
        
        if ret:
            return True, backend_name
            
    return False, None

if __name__ == "__main__":
    device = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    success, backend = test_camera(device)
    
    if success:
        print(f"\nCamera test PASSED with {backend} backend")
    else:
        print("\nCamera test FAILED with all backends")
