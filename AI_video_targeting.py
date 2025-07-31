import cv2
from facenet_pytorch import MTCNN
import torch
from PIL import Image, ImageDraw
from IPython import display
import numpy as np
import os
import requests
from io import BytesIO
import base64
import json
import uuid
from datetime import datetime

# Device selection with MacOS MPS support
# if torch.backends.mps.is_available():
#     # Enable CPU fallback for operations not yet implemented in MPS
#     os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
#     device = torch.device('mps')
#     print('Running on MPS device with CPU fallback enabled')
    
#     # Additional MPS fallback setup for adaptive pooling issues
#     import torch.backends.mps as mps
#     if hasattr(mps, 'enable_mps_fallback'):
#         mps.enable_mps_fallback(True)
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('Running on CUDA device')
else:
    device = torch.device('cpu')
    print('Running on CPU device')

print('Running on device: {}'.format(device))

# For MTCNN, use CPU to avoid MPS adaptive pooling issues
# MTCNN has known compatibility issues with MPS backend
mtcnn_device = torch.device(device)
# print(f'Using CPU for MTCNN to avoid MPS compatibility issues')
mtcnn = MTCNN(keep_all=True, device=mtcnn_device)

# Initialize video capture - use device path for better compatibility on Jetson
# For EMEET SmartCam C960 4K on Jetson Orin
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

# Set MJPEG format for better performance with USB cameras
if cap.isOpened():
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


# def add_unknown_to_database(image_path, confidence=0.0, source="auto_detection", 
#                           location=None, priority="normal", use_json=True):
#     """
#     Add an unknown face to the FAISS database
    
#     Args:
#         image_path: Path to the face image
#         confidence: Detection confidence score
#         source: Source identifier (camera, video file, etc.)
#         location: Optional location information
#         priority: "normal" or "high"
#         use_json: Use JSON format (True) or form-data (False)
#     """
    
#     # Generate unique name for unknown person
#     timestamp = int(datetime.now().timestamp())
#     unique_id = str(uuid.uuid4())[:8]
#     unknown_name = f"Unknown_{timestamp}_{unique_id}"
    
#     metadata = {
#         "source": source,
#         "priority": priority,
#         "category": "suspect",
#         "detection_confidence": confidence,
#         "auto_generated": True,
#         "detection_timestamp": datetime.now().isoformat(),
#         "original_source": source,
#         "alert_triggered": False
#     }
    
#     if location:
#         metadata["location"] = location
    
#     if use_json:
#         # JSON format
#         with open(image_path, 'rb') as img_file:
#             img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
#         payload = {
#             "image": img_base64,
#             "name": unknown_name,
#             "description": "automatically detected unknown individual",
#             "metadata": metadata
#         }
        
#         response = requests.post(
#             'http://localhost:5000/add', 
#             json=payload,
#             headers={'Content-Type': 'application/json'}
#         )
#     else:
#         # Form-data format
#         files = {'image': open(image_path, 'rb')}
#         data = {
#             'name': unknown_name,
#             'description': 'automatically detected unknown individual',
#             'metadata': json.dumps(metadata)
#         }
        
#         response = requests.post('http://localhost:5000/add', files=files, data=data)
#         files['image'].close()
    
#     return response.json(), unknown_name


# Define the number of frames to skip
frame_skip = 10
n_frame = 0

while True:
    
    # print('\rTracking frame: {}'.format(n_frame + 1), end='')
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames
    if n_frame % frame_skip != 0:
        n_frame += 1
        continue
    
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Detect faces
    boxes, _ = mtcnn.detect(frame)

    # Draw faces
    frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)
    
    # Only process boxes if they exist and are not empty
    if boxes is not None:
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=4)
    
    # Convert frame_draw back to OpenCV format for display
    frame_draw_cv = cv2.cvtColor(np.array(frame_draw), cv2.COLOR_RGB2BGR)
    
    # Resize the frame for a bigger display window
    display_width = 1280  # Increase width
    display_height = 720  # Increase height
    frame_draw_cv = cv2.resize(frame_draw_cv, (display_width, display_height))
    
    cv2.imshow("Face Tracking", frame_draw_cv)

    
    # Extract and display faces
    if boxes is not None:
        for idx, box in enumerate(boxes):
            # Convert box coordinates to integers
            box = [int(coord) for coord in box]
            x1, y1, x2, y2 = box
            print(f"Face {idx + 1}: ({x1}, {y1}, {x2}, {y2})")

            # Crop the face from the frame
            face = frame.crop((x1, y1, x2, y2))

            # Resize the cropped face to a smaller size
            face_resized = face.resize((100, 100), Image.BILINEAR)

            # Convert the resized face to OpenCV format
            face_resized_cv = cv2.cvtColor(np.array(face_resized), cv2.COLOR_RGB2BGR)

            # Superimpose the resized face on the main frame
            x_offset, y_offset = 10 + idx * 110, 10  # Adjust position for each face
            
            # Ensure the overlay respects the dimensions of the main frame
            y_end = min(y_offset + 100, frame_draw_cv.shape[0])
            x_end = min(x_offset + 100, frame_draw_cv.shape[1])

            # Adjust the resized face dimensions to fit within the main frame
            face_resized_cv = face_resized_cv[:y_end - y_offset, :x_end - x_offset]

            # Overlay the resized face on the main frame
            frame_draw_cv[y_offset:y_end, x_offset:x_end] = face_resized_cv

    
    cv2.imshow("Face Tracking", frame_draw_cv)
    

cap.release()
cv2.destroyAllWindows()