from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import base64
import json
import threading
import time
from datetime import datetime
from config import Config
import faiss
import os
import gc
from scipy.spatial import Delaunay

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)  # Enable CORS for all routes

class FaceDetectionApp:
    def __init__(self):
        # Enhanced device selection with GPU optimization
        if Config.FORCE_CPU:
            self.device = torch.device('cpu')
            print('Forced to use CPU device')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            print(f'Running on CUDA device: {torch.cuda.get_device_name(0)}')
            print(f'CUDA version: {torch.version.cuda}')
            print(f'Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
            
            # Set GPU memory management
            if hasattr(Config, 'GPU_MEMORY_FRACTION'):
                torch.cuda.set_per_process_memory_fraction(Config.GPU_MEMORY_FRACTION)
            else:
                torch.cuda.set_per_process_memory_fraction(0.8)
            
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
        else:
            self.device = torch.device('cpu')
            print('CUDA not available, running on CPU device')
        
        print('Running on device: {}'.format(self.device))
        
        # Initialize MTCNN with GPU optimizations
        self.mtcnn = MTCNN(
            keep_all=True, 
            device=self.device,
            post_process=True,  # Enable post-processing for better accuracy
            image_size=160,  # Standard size for face recognition
            margin=40,  # Margin around detected face
            selection_method='probability'  # Select faces based on probability
        )
        
        # Initialize video capture
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.detected_faces = []
        self.frame_lock = threading.Lock()
        
        # Face tracking for persistent IDs
        self.face_tracker = {}  # persistent face data
        self.next_face_id = 0
        self.face_similarity_threshold = 100  # pixel distance threshold for face tracking

        # FAISS index and metadata with GPU support
        self.faiss_index_path = 'faiss_faces.index'
        self.faiss_metadata_path = 'faiss_faces_metadata.json'
        self.embedding_dim = 512  # For facenet-pytorch InceptionResnetV1
        
        # Initialize InceptionResnetV1 with GPU optimizations
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Enable mixed precision if available (for RTX cards)
        self.use_amp = (torch.cuda.is_available() and 
                       hasattr(torch.cuda, 'amp') and 
                       getattr(Config, 'ENABLE_MIXED_PRECISION', True))
        if self.use_amp:
            print("Using Automatic Mixed Precision (AMP) for faster inference")
        
        # Initialize FAISS with GPU support if available
        self._load_faiss()

    def _load_faiss(self):
        """Load FAISS index with GPU support if available"""
        if os.path.exists(self.faiss_index_path):
            self.faiss_index = faiss.read_index(self.faiss_index_path)
        else:
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Try to move FAISS index to GPU if available
        if torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources'):
            try:
                self.gpu_res = faiss.StandardGpuResources()
                self.faiss_index = faiss.index_cpu_to_gpu(self.gpu_res, 0, self.faiss_index)
                print("FAISS index moved to GPU for faster search")
            except Exception as e:
                print(f"Could not move FAISS to GPU: {e}, using CPU")
        
        if os.path.exists(self.faiss_metadata_path):
            with open(self.faiss_metadata_path, 'r') as f:
                self.faiss_metadata = json.load(f)
        else:
            self.faiss_metadata = []

    def _save_faiss(self):
        """Save FAISS index (move to CPU first if on GPU)"""
        index_to_save = self.faiss_index
        
        # If index is on GPU, move to CPU for saving
        if hasattr(self, 'gpu_res') and hasattr(faiss, 'index_gpu_to_cpu'):
            try:
                index_to_save = faiss.index_gpu_to_cpu(self.faiss_index)
            except Exception as e:
                print(f"Error moving index to CPU for saving: {e}")
        
        faiss.write_index(index_to_save, self.faiss_index_path)
        with open(self.faiss_metadata_path, 'w') as f:
            json.dump(self.faiss_metadata, f)

    def get_face_embedding(self, face_img):
        """Generate face embedding with GPU optimization and mixed precision"""
        # Preprocess image
        preprocess = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        
        img_tensor = preprocess(face_img).unsqueeze(0).to(self.device, non_blocking=True)
        
        # Use mixed precision if available
        if self.use_amp:
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    emb = self.resnet(img_tensor)
        else:
            with torch.no_grad():
                emb = self.resnet(img_tensor)
        
        # Move to CPU and convert to numpy
        emb_cpu = emb.cpu().numpy()
        
        # Normalize embedding for better FAISS performance
        emb_normalized = F.normalize(torch.from_numpy(emb_cpu), p=2, dim=1).numpy()
        
        return emb_normalized.astype('float32')

    def get_face_embeddings_batch(self, face_imgs):
        """Generate face embeddings for multiple faces in batch for better GPU utilization"""
        if not face_imgs:
            return np.array([])
        
        # Preprocess all images
        preprocess = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        
        batch_tensors = []
        for face_img in face_imgs:
            img_tensor = preprocess(face_img)
            batch_tensors.append(img_tensor)
        
        # Stack into batch
        batch = torch.stack(batch_tensors).to(self.device, non_blocking=True)
        
        # Generate embeddings with mixed precision
        if self.use_amp:
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    embeddings = self.resnet(batch)
        else:
            with torch.no_grad():
                embeddings = self.resnet(batch)
        
        # Normalize and convert to numpy
        embeddings_normalized = F.normalize(embeddings, p=2, dim=1).cpu().numpy()
        
        return embeddings_normalized.astype('float32')

    def search_faiss(self, embedding, threshold=0.6):
        """Search FAISS index with optimized similarity threshold"""
        if self.faiss_index.ntotal == 0:
            return None, None
        
        # Ensure embedding is the right shape
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        D, I = self.faiss_index.search(embedding, 1)
        idx = int(I[0][0])
        dist = float(D[0][0])
        
        # Use cosine similarity threshold (lower is better for normalized embeddings)
        if dist < threshold:
            return self.faiss_metadata[idx], dist
        return None, None

    def add_to_faiss(self, embedding, name, status, notes, image_b64):
        """Add face embedding to FAISS index"""
        # Ensure embedding is the right shape
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        self.faiss_index.add(embedding)
        self.faiss_metadata.append({
            'name': name,
            'status': status,
            'notes': notes,
            'image': image_b64
        })
        self._save_faiss()

    def cleanup_gpu_memory(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def get_face_distance(self, box1, box2):
        """Calculate distance between two face bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate center points
        center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
        center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
        
        # Euclidean distance
        distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
        return distance

    def assign_face_id(self, box, prob):
        """Assign a persistent ID to a detected face"""
        # Try to match with existing tracked faces
        best_match_id = None
        best_distance = float('inf')
        
        for face_id, face_data in self.face_tracker.items():
            distance = self.get_face_distance(box, face_data['box'])
            if distance < self.face_similarity_threshold and distance < best_distance:
                best_distance = distance
                best_match_id = face_id
        
        if best_match_id is not None:
            # Update existing face
            self.face_tracker[best_match_id].update({
                'box': box,
                'confidence': prob,
                'last_seen': time.time()
            })
            return best_match_id
        else:
            # Create new face
            face_id = self.next_face_id
            self.next_face_id += 1
            self.face_tracker[face_id] = {
                'box': box,
                'confidence': prob,
                'last_seen': time.time(),
                'created': time.time()
            }
            return face_id

    def cleanup_old_faces(self, max_age=5.0):
        """Remove faces that haven't been seen for a while"""
        current_time = time.time()
        old_faces = [face_id for face_id, face_data in self.face_tracker.items() 
                     if current_time - face_data['last_seen'] > max_age]
        
        for face_id in old_faces:
            del self.face_tracker[face_id]
    
    def draw_facial_landmarks(self, draw, landmarks, color=(0, 255, 255)):
        """Draw facial landmarks with triangular mesh like in movies"""
        if landmarks is None or len(landmarks) != 5:
            return
        
        # Convert landmarks to numpy array for easier manipulation
        points = np.array(landmarks, dtype=np.float32)
        
        if Config.SHOW_FACIAL_LANDMARKS:
            # Draw landmark points
            for point in points:
                x, y = int(point[0]), int(point[1])
                draw.ellipse([x-2, y-2, x+2, y+2], fill=color, outline=color)
        
        if Config.LANDMARK_TRIANGULATION:
            # Create triangulation for movie-like effect
            # Define connections between facial landmarks (MTCNN gives 5 points)
            # Points: left_eye, right_eye, nose, left_mouth, right_mouth
            connections = [
                (0, 1),  # eyes
                (0, 2),  # left eye to nose
                (1, 2),  # right eye to nose
                (2, 3),  # nose to left mouth
                (2, 4),  # nose to right mouth
                (3, 4),  # mouth corners
                (0, 3),  # left eye to left mouth
                (1, 4),  # right eye to right mouth
            ]
            
            # Draw triangular connections
            for start_idx, end_idx in connections:
                start_point = points[start_idx]
                end_point = points[end_idx]
                
                start_x, start_y = int(start_point[0]), int(start_point[1])
                end_x, end_y = int(end_point[0]), int(end_point[1])
                
                # Draw line with transparency effect
                draw.line([start_x, start_y, end_x, end_y], fill=color, width=1)
            
            # Add additional triangular mesh for more movie-like effect
            if len(points) >= 5:
                # Create additional triangulation points around the face
                try:
                    # Use Delaunay triangulation for more complex mesh
                    tri = Delaunay(points)
                    for simplex in tri.simplices:
                        triangle_points = []
                        for vertex in simplex:
                            x, y = int(points[vertex][0]), int(points[vertex][1])
                            triangle_points.extend([x, y])
                        
                        # Draw triangle outline
                        if len(triangle_points) >= 6:
                            draw.polygon(triangle_points, outline=color, fill=None, width=1)
                except Exception as e:
                    # Fallback to simple connections if triangulation fails
                    pass
        
    def start_camera(self, camera_index=None):
        """Start camera capture"""
        if camera_index is None:
            camera_index = Config.DEFAULT_CAMERA_INDEX
            
        if self.cap is not None:
            self.cap.release()
        
        # For Jetson with USB cameras, use device path for better compatibility
        if isinstance(camera_index, int):
            device_path = f"/dev/video{camera_index}"
            self.cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(camera_index)
            
        if not self.cap.isOpened():
            return False
        
        # Set MJPEG format for better performance with USB cameras
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
        
        # Add buffer size setting to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.is_running = True
        return True
    
    def stop_camera(self):
        """Stop camera capture and cleanup GPU memory"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Clear face tracker
        self.face_tracker.clear()
        self.next_face_id = 0
        
        # Clean up GPU memory
        self.cleanup_gpu_memory()
    
    def process_frame(self):
        """Process a single frame for face detection with GPU optimization"""
        if self.cap is None or not self.is_running:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Convert BGR to RGB for MTCNN
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        # Detect faces with GPU acceleration
        try:
            boxes, probs, landmarks = self.mtcnn.detect(frame_pil, landmarks=True)
        except Exception as e:
            print(f"Error in face detection: {e}")
            return self.current_frame
        
        # Clean up old faces from tracker
        self.cleanup_old_faces()
        
        # Draw bounding boxes
        frame_draw = frame_pil.copy()
        draw = ImageDraw.Draw(frame_draw)
        
        faces_info = []
        face_crops = []  # Collect face crops for batch processing
        
        if boxes is not None:
            valid_faces = []
            for idx, (box, prob) in enumerate(zip(boxes, probs)):
                if prob > Config.FACE_CONFIDENCE_THRESHOLD:
                    # Assign persistent ID to this face
                    face_id = self.assign_face_id([int(coord) for coord in box], prob)
                    # Get corresponding landmarks for this face
                    face_landmarks = landmarks[idx] if landmarks is not None and idx < len(landmarks) else None
                    valid_faces.append((face_id, box, prob, face_landmarks))
            
            # Process faces in batch for better GPU utilization
            for face_id, box, prob, face_landmarks in valid_faces[:Config.MAX_FACES_TO_DISPLAY]:
                # For locked faces, use the stored box coordinates to prevent visual updates
                display_box = box
                face_status = 'unknown'  # Default status
                face_name = ''  # Default name
                
                # Find existing face data to get stored box coordinates, status, and name
                with self.frame_lock:
                    for existing in self.detected_faces:
                        if existing.get('id') == face_id:
                            if face_id in locked_faces:
                                display_box = existing.get('box', box)
                            if existing.get('faiss_found', False):
                                face_status = existing.get('status', 'unknown')
                                face_name = existing.get('name', '')
                            break
                
                # Determine box color based on face status
                if face_status == 'friendly':
                    color = (0, 255, 0)  # Green
                elif face_status == 'target':
                    color = (255, 0, 0)  # Red
                elif face_status == 'suspect':
                    color = (255, 255, 0)  # Yellow
                else:  # unknown
                    color = (128, 0, 128)  # Purple
                
                # Draw the face box
                if face_id in locked_faces:
                    draw.rectangle(display_box, outline=color, width=3)
                    box_to_use = display_box
                else:
                    draw.rectangle(box.tolist(), outline=color, width=3)
                    box_to_use = box.tolist()
                
                # Draw name below the face box if available
                if face_name:
                    # Calculate text position (below the box)
                    text_x = int(box_to_use[0])
                    text_y = int(box_to_use[3]) + 5  # 5 pixels below the bottom of the box
                    
                    # Try to use a larger font, fallback to default if not available
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
                    except (OSError, IOError):
                        try:
                            font = ImageFont.truetype("arial.ttf", 24)
                        except (OSError, IOError):
                            font = ImageFont.load_default()
                    
                    # Draw text background for better readability
                    text_bbox = draw.textbbox((text_x, text_y), face_name, font=font)
                    draw.rectangle(text_bbox, fill=(0, 0, 0, 128))  # Semi-transparent black background
                    
                    # Draw the name text with larger font
                    draw.text((text_x, text_y), face_name, fill=color, font=font, stroke_width=2, stroke_fill=(0, 0, 0))
                
                # Draw facial landmarks with triangular mesh
                if face_landmarks is not None:
                    # Use cyan color for landmarks to make them visible against face box color
                    landmark_color = (0, 255, 255)  # Cyan
                    self.draw_facial_landmarks(draw, face_landmarks, landmark_color)
                
                # Extract face coordinates
                x1, y1, x2, y2 = [int(coord) for coord in box]
                
                # Crop face
                face_crop = frame_pil.crop((x1, y1, x2, y2))
                face_crops.append((face_id, face_crop, [x1, y1, x2, y2], prob))
            
            # Generate embeddings in batch if we have faces (only for unlocked faces)
            if face_crops:
                # Separate locked and unlocked faces
                unlocked_face_crops = [(i, face_data) for i, face_data in enumerate(face_crops) 
                                     if face_data[0] not in locked_faces]
                
                if unlocked_face_crops:
                    face_images = [crop_data[1][1] for crop_data in unlocked_face_crops]  # Extract face images
                    try:
                        if len(face_images) > 1:
                            # Use batch processing for multiple faces
                            embeddings = self.get_face_embeddings_batch(face_images)
                        else:
                            # Single face processing
                            embeddings = [self.get_face_embedding(face_images[0])]
                            
                    except Exception as e:
                        print(f"Error generating embeddings: {e}")
                        embeddings = []
                else:
                    embeddings = []
                
                # Process each face with its embedding
                embedding_idx = 0  # Index for unlocked faces' embeddings
                for i, (face_id, face_crop, box, prob) in enumerate(face_crops):
                    # Check if this face is locked - if so, find existing data and preserve it COMPLETELY
                    if face_id in locked_faces:
                        # Find existing face data to preserve
                        existing_face = None
                        with self.frame_lock:
                            for existing in self.detected_faces:
                                if existing.get('id') == face_id:
                                    existing_face = existing.copy()
                                    break
                        
                        if existing_face:
                            # For locked faces, preserve EVERYTHING - no updates at all to prevent UI interference
                            # This ensures the user can type without interruption
                            faces_info.append(existing_face)
                            continue
                    
                    # For unlocked faces, process normally
                    # Resize face for display - use Image.LANCZOS for compatibility with older Pillow versions
                    try:
                        # Try new Pillow 10+ syntax first
                        face_resized = face_crop.resize((100, 100), Image.Resampling.LANCZOS)
                    except AttributeError:
                        # Fallback to older Pillow syntax
                        face_resized = face_crop.resize((100, 100), Image.LANCZOS)
                    
                    # Convert face to base64 for web display
                    import io
                    face_buffer = io.BytesIO()
                    face_resized.save(face_buffer, format='JPEG')
                    face_base64 = base64.b64encode(face_buffer.getvalue()).decode()

                    # Search FAISS database if embedding was generated for this unlocked face
                    if embedding_idx < len(embeddings):
                        faiss_result, faiss_dist = self.search_faiss(embeddings[embedding_idx])
                        if faiss_result:
                            name = faiss_result.get('name', 'Unknown')
                            status = faiss_result.get('status', 'Unknown')
                            notes = faiss_result.get('notes', '')
                            faiss_found = True
                        else:
                            name = ''
                            status = ''
                            notes = ''
                            faiss_found = False
                        embedding_idx += 1  # Move to next embedding
                    else:
                        # Fallback if embedding generation failed
                        name = ''
                        status = ''
                        notes = ''
                        faiss_found = False
                    
                    faces_info.append({
                        'id': face_id,  # Use persistent face ID
                        'box': box,
                        'confidence': float(prob),
                        'image': face_base64,
                        'faiss_found': faiss_found,
                        'name': name,
                        'status': status,
                        'notes': notes
                    })
        
        # Convert back to BGR for OpenCV
        frame_bgr = cv2.cvtColor(np.array(frame_draw), cv2.COLOR_RGB2BGR)
        
        with self.frame_lock:
            self.current_frame = frame_bgr
            self.detected_faces = faces_info
        
        # Periodic GPU memory cleanup
        if hasattr(self, '_frame_count'):
            self._frame_count += 1
            if self._frame_count % 100 == 0:  # Clean up every 100 frames
                self.cleanup_gpu_memory()
        else:
            self._frame_count = 1
        
        return frame_bgr
    
    def get_frame_data(self):
        """Get current frame and detected faces data"""
        with self.frame_lock:
            return self.current_frame, self.detected_faces.copy()

# Global face detection instance
face_detector = FaceDetectionApp()

# Store locked faces on the backend
locked_faces = set()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/get_locked_faces')
def get_locked_faces():
    """Get list of locked face IDs"""
    return jsonify({'locked_faces': list(locked_faces)})

@app.route('/lock_face', methods=['POST'])
def lock_face():
    """Lock a face for editing"""
    data = request.get_json()
    face_id = data.get('face_id')
    if face_id is not None:
        locked_faces.add(int(face_id))
        return jsonify({'status': 'success', 'message': f'Face {face_id} locked'})
    return jsonify({'status': 'error', 'message': 'Invalid face ID'}), 400

@app.route('/unlock_face', methods=['POST'])
def unlock_face():
    """Unlock a face"""
    data = request.get_json()
    face_id = data.get('face_id')
    if face_id is not None:
        locked_faces.discard(int(face_id))
        return jsonify({'status': 'success', 'message': f'Face {face_id} unlocked'})
    return jsonify({'status': 'error', 'message': 'Invalid face ID'}), 400

@app.route('/toggle_landmarks', methods=['POST'])
def toggle_landmarks():
    """Toggle facial landmarks display"""
    data = request.get_json()
    show_landmarks = data.get('show_landmarks', True)
    show_triangulation = data.get('show_triangulation', True)
    
    # Update the config dynamically
    Config.SHOW_FACIAL_LANDMARKS = show_landmarks
    Config.LANDMARK_TRIANGULATION = show_triangulation
    
    return jsonify({
        'status': 'success', 
        'show_landmarks': Config.SHOW_FACIAL_LANDMARKS,
        'show_triangulation': Config.LANDMARK_TRIANGULATION
    })

@app.route('/get_landmark_settings')
def get_landmark_settings():
    """Get current landmark settings"""
    return jsonify({
        'show_landmarks': getattr(Config, 'SHOW_FACIAL_LANDMARKS', True),
        'show_triangulation': getattr(Config, 'LANDMARK_TRIANGULATION', True)
    })

@app.route('/start_detection', methods=['POST'])
def start_detection():
    """Start face detection"""
    data = request.get_json()
    camera_index = data.get('camera_index', 0)
    
    success = face_detector.start_camera(camera_index)
    if success:
        return jsonify({'status': 'success', 'message': 'Face detection started'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to start camera'})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    """Stop face detection"""
    face_detector.stop_camera()
    # Clear locked faces when stopping detection
    locked_faces.clear()
    return jsonify({'status': 'success', 'message': 'Face detection stopped'})

@app.route('/get_faces')
def get_faces():
    """Get detected faces data"""
    _, faces = face_detector.get_frame_data()
    # Add locked status to each face
    for face in faces:
        face['locked'] = face.get('id') in locked_faces
    return jsonify({'faces': faces, 'locked_faces': list(locked_faces)})

# API to add a new face to FAISS DB
@app.route('/add_face', methods=['POST'])
def add_face():
    data = request.get_json()
    image_b64 = data.get('image')
    name = data.get('name', '')
    status = data.get('status', '')
    notes = data.get('notes', '')
    if not image_b64 or not name or not status:
        return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400
    # Decode image
    import io
    face_bytes = base64.b64decode(image_b64)
    face_img = Image.open(io.BytesIO(face_bytes)).convert('RGB')
    embedding = face_detector.get_face_embedding(face_img)
    face_detector.add_to_faiss(embedding, name, status, notes, image_b64)
    return jsonify({'status': 'success', 'message': 'Face added to database'})

# API to update face information for locked faces
@app.route('/update_face', methods=['POST'])
def update_face():
    """Update face information for a locked face"""
    data = request.get_json()
    face_id = data.get('face_id')
    name = data.get('name', '')
    status = data.get('status', '')
    notes = data.get('notes', '')
    
    if face_id is None:
        return jsonify({'status': 'error', 'message': 'Face ID is required'}), 400
    
    face_id = int(face_id)
    
    # Find and update the face in detected_faces
    with face_detector.frame_lock:
        for i, face in enumerate(face_detector.detected_faces):
            if face.get('id') == face_id:
                face_detector.detected_faces[i]['name'] = name
                face_detector.detected_faces[i]['status'] = status
                face_detector.detected_faces[i]['notes'] = notes
                return jsonify({'status': 'success', 'message': f'Face {face_id} updated'})
    
    return jsonify({'status': 'error', 'message': 'Face not found'}), 404

def generate_frames():
    """Generate video frames for streaming with optimized performance"""
    frame_skip = Config.FRAME_SKIP_RATE  # Use config frame skip rate
    frame_count = 0
    target_fps = getattr(Config, 'STREAM_FPS', 45)  # Default to 45 FPS if not set
    sleep_time = 1.0 / target_fps  # Calculate sleep time based on target FPS
    
    while True:
        if not face_detector.is_running:
            time.sleep(0.1)
            continue
        
        frame_count += 1
        if frame_count % frame_skip == 0:
            frame = face_detector.process_frame()
        else:
            frame, _ = face_detector.get_frame_data()
        
        if frame is not None:
            # Encode frame as JPEG with optimized quality for speed
            encode_params = [
                cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1,  # Enable JPEG optimization
                cv2.IMWRITE_JPEG_PROGRESSIVE, 1  # Enable progressive JPEG for faster loading
            ]
            ret, buffer = cv2.imencode('.jpg', frame, encode_params)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(sleep_time)  # Dynamic FPS based on config

@app.route('/video_feed')
def video_feed():
    """Video streaming route with optimized headers for performance"""
    response = Response(generate_frames(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    # Add headers for better streaming performance
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['Connection'] = 'keep-alive'
    return response

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'camera_active': face_detector.is_running,
        'device': str(face_detector.device)
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('error.html', error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('error.html', error_message="Internal server error"), 500

if __name__ == '__main__':
    print("Starting Face Detection Web Application...")
    print("Access the application at: http://localhost:5000")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    finally:
        print("Shutting down application...")
        face_detector.stop_camera()
        face_detector.cleanup_gpu_memory()
        print("Cleanup completed.")
