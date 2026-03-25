import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# --- Model Architecture ---
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        attn_weights = self.attention(x)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * x, dim=1)
        return context

# --- Advanced Analytics ---
def calculate_distance_3d(lm1, lm2):
    """Calculate Euclidean distance between two 3D landmarks."""
    return np.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2 + (lm1.z - lm2.z)**2)

def calculate_angle_3d(a, b, c):
    """
    Calculate angle between 3 points in 3D space.
    Returns angle in degrees.
    """
    a_vec = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    c_vec = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    
    cosine_angle = np.dot(a_vec, c_vec) / (np.linalg.norm(a_vec) * np.linalg.norm(c_vec))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

class BoxingModel(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=8):
        super(BoxingModel, self).__init__()
        self.input_bn = nn.BatchNorm1d(input_size)
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        if x.dim() == 3:
            x = x.transpose(1, 2)
            x = self.input_bn(x)
            x = x.transpose(1, 2)
        else:
            x = self.input_bn(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.attention(x)
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def get_boxing_class_name(class_id):
    # This is a placeholder list; adjust based on your training labels!
    labels = ["Jab", "Cross", "Hook", "Uppercut", "Slip", "Duck", "Block", "Idle"]
    if 0 <= class_id < len(labels):
        return labels[class_id]
    return f"Class {class_id}"

# --- Global Model Loading (for backend efficiency) ---
MODEL_PATH = os.getenv("BOXING_MODEL_PATH", "models/best_boxing_model.pth")
_boxing_model = None

def get_model():
    """Singleton pattern to load the model only once into memory."""
    global _boxing_model
    if _boxing_model is None:
        print("Loading PyTorch model into memory...")
        _boxing_model = BoxingModel()
        try:
            state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
            _boxing_model.load_state_dict(state_dict)
            _boxing_model.eval()
        except Exception as e:
            print(f"Failed to load model: {e}")
            _boxing_model = None
    return _boxing_model

# --- Color Re-Identification (HSV Histograms) ---

def extract_color_histogram(frame, box):
    """Extracts a normalized HSV color histogram from the top half and bottom half of a bounding box."""
    x1, y1, x2, y2 = map(int, box)
    
    # Ensure coordinates are within frame bounds to prevent OpenCV crashes
    h_frame, w_frame = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_frame, x2), min(h_frame, y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
        
    crop = frame[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    
    # Split into top (shirt/headgear) and bottom (trunks)
    top_half = crop[0:h//2, 0:w]
    bottom_half = crop[h//2:h, 0:w]
    
    # Convert to HSV 
    hsv_top = cv2.cvtColor(top_half, cv2.COLOR_BGR2HSV)
    hsv_bottom = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2HSV)
    
    # Calculate 2D histograms (Hue and Saturation)
    # Using 16 bins for Hue (colors) and 8 bins for Saturation
    hist_top = cv2.calcHist([hsv_top], [0, 1], None, [16, 8], [0, 180, 0, 256])
    hist_bottom = cv2.calcHist([hsv_bottom], [0, 1], None, [16, 8], [0, 180, 0, 256])
    
    # Normalize to account for different box sizes
    cv2.normalize(hist_top, hist_top, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_bottom, hist_bottom, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    return {'top': hist_top, 'bottom': hist_bottom}

def compare_histograms(hist1, hist2):
    """Compares two color profiles using Bhattacharyya distance. Lower score = better match."""
    if hist1 is None or hist2 is None:
        return float('inf')
        
    score_top = cv2.compareHist(hist1['top'], hist2['top'], cv2.HISTCMP_BHATTACHARYYA)
    score_bottom = cv2.compareHist(hist1['bottom'], hist2['bottom'], cv2.HISTCMP_BHATTACHARYYA)
    
    # Average the similarities
    return (score_top + score_bottom) / 2.0

def hsv_to_color_name(h, s, v):
    """
    Maps an HSV value to a human-readable color name.
    H: 0-180 (OpenCV's range), S: 0-255, V: 0-255.
    """
    if v < 50:
        return "Black"
    if s < 40 and v > 200:
        return "White"
    if s < 40:
        return "Gray"
    # Chromatic colors (by hue)
    if h < 8 or h >= 165:
        return "Red"
    if h < 22:
        return "Orange"
    if h < 35:
        return "Yellow"
    if h < 78:
        return "Green"
    if h < 131:
        return "Blue"
    if h < 165:
        return "Purple"
    return "Unknown"

def extract_dominant_color(frame, cx, cy, crop_x1, crop_y1, crop_w, crop_h, patch_radius=15):
    """
    Extracts the dominant color name from a small patch around a landmark point.
    cx, cy are normalized [0,1] MediaPipe coordinates within the crop.
    Returns a human-readable color string like 'Red', 'Blue', 'Black', etc.
    """
    # Convert normalized coords to pixel coords within the crop
    px = int(cx * crop_w)
    py = int(cy * crop_h)
    
    # Define patch boundaries (clamp to crop size)
    x1 = max(0, px - patch_radius)
    y1 = max(0, py - patch_radius)
    x2 = min(crop_w, px + patch_radius)
    y2 = min(crop_h, py + patch_radius)
    
    # Get the patch from the full frame (offset by crop position)
    patch = frame[crop_y1 + y1:crop_y1 + y2, crop_x1 + x1:crop_x1 + x2]
    if patch.size == 0:
        return "Unknown"
    
    # Convert to HSV and compute the median color
    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    median_h = int(np.median(hsv_patch[:, :, 0]))
    median_s = int(np.median(hsv_patch[:, :, 1]))
    median_v = int(np.median(hsv_patch[:, :, 2]))
    
    return hsv_to_color_name(median_h, median_s, median_v)

def extract_fighter_appearance(frame, landmarks, mp_pose_lm, crop_x1, crop_y1, crop_w, crop_h):
    """
    Extracts visual appearance description of a fighter using MediaPipe landmarks.
    Returns a dict with headgear, gloves, and trunks colors.
    """
    nose = landmarks[mp_pose_lm.NOSE.value]
    left_wrist = landmarks[mp_pose_lm.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose_lm.RIGHT_WRIST.value]
    left_hip = landmarks[mp_pose_lm.LEFT_HIP.value]
    right_hip = landmarks[mp_pose_lm.RIGHT_HIP.value]
    left_knee = landmarks[mp_pose_lm.LEFT_KNEE.value]
    right_knee = landmarks[mp_pose_lm.RIGHT_KNEE.value]
    
    # Headgear: sample around the top of the head (above nose)
    head_y = max(0.0, nose.y - 0.06)  # slightly above nose
    headgear_color = extract_dominant_color(frame, nose.x, head_y, crop_x1, crop_y1, crop_w, crop_h, patch_radius=20)
    
    # Gloves: average color from both wrists
    left_glove = extract_dominant_color(frame, left_wrist.x, left_wrist.y, crop_x1, crop_y1, crop_w, crop_h)
    right_glove = extract_dominant_color(frame, right_wrist.x, right_wrist.y, crop_x1, crop_y1, crop_w, crop_h)
    # Use the most common of the two (they should match)
    glove_color = left_glove if left_glove == right_glove else f"{left_glove}/{right_glove}"
    
    # Trunks: sample between hips and knees
    trunk_cx = (left_hip.x + right_hip.x) / 2
    trunk_cy = (left_hip.y + right_hip.y + left_knee.y + right_knee.y) / 4  # midpoint between hips and knees
    trunks_color = extract_dominant_color(frame, trunk_cx, trunk_cy, crop_x1, crop_y1, crop_w, crop_h, patch_radius=20)
    
    return {
        "headgear": headgear_color,
        "gloves": glove_color,
        "trunks": trunks_color
    }

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

# --- Main Refactored Function ---
def identify_punches_in_video(video_path: str, confidence_threshold: float = 0.8, user_bbox: list = None) -> dict:
    """
    Processes a video file and returns identified punches and analytics.
    
    Args:
        video_path: The absolute or relative path to the video file.
        confidence_threshold: Minimum confidence [0.0 - 1.0] to register a punch.
        
    Returns:
        A list of dictionaries containing:
          - punch_type: Name of the identified action (Jab, Cross, etc.)
          - start_time_sec: Approximate start timestamp in seconds
          - confidence: The model's confident score [0.0 - 1.0]
    """
    model = get_model()
    if model is None:
        raise ValueError("Model failed to initialize.")

    # Load YOLOv8 for Multi-Person BBox Tracking using the compiled OpenVINO format for speed
    print("Loading YOLOv8 object detector...")
    try:
        yolo_model = YOLO('models/yolov8n_openvino_model') 
    except Exception as e:
        print(f"Warning: Failed to load OpenVINO model ({e}). Falling back to standard PyTorch model.")
        yolo_model = YOLO('models/yolov8n.pt')
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0 # Fallback 

    # We need a dictionary to store sequences and states for MULTIPLE fighters
    # Key: YOLO Track ID, Value: tracking data dict
    fighters_data = {}
    
    mp_pose = mp.solutions.pose
    
    # We create a small pool of reusable MediaPipe trackers to prevent C++ thread exhaustion
    # instead of creating a new one every time YOLO creates a new track_id
    MAX_TRACKERS = 5
    # Optimization 3: Lowering min_detection_confidence to 0.4 so it relies on the faster internal tracker more
    pose_trackers = [mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.4, min_tracking_confidence=0.5) for _ in range(MAX_TRACKERS)]
    tracker_index = 0
    
    # Optimization 2: Frame Skipping
    # Assuming video is 30 or 60 fps, we don't need every frame to track a punch
    frame_skip = 2 # Process every 2nd frame
    
    sequence_length = 30 # Must match model's expected sequence length
    cooldown_frames = int(fps * 0.5)

    identified_actions = [] # We'll flatten this later
    frame_analytics = []

    frame_count = 0
    
    # Static tracking assignments for User vs Opponent
    user_tracked_ids = set() # Store all historical IDs
    opponent_tracked_ids = set()
    current_user_id = None # Store the currently active YOLO ID
    current_opponent_id = None
    last_known_user_center = None
    last_known_opponent_center = None
    
    # Store HSV visual signatures for Re-Identification during clinches
    user_color_profile = None
    opponent_color_profile = None
    
    # Store historic coordinates to track who initiated distance closures
    previous_global_feet = {}
    previous_global_distance = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Better Time Calculation: Use OpenCV's actual video position instead of frame math
        # to prevent drift when skipping frames or conditionally dropping analysis
        frame_count += 1
        current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if frame_count % frame_skip != 0:
            continue

        # 1. Run YOLO to get tracked bounding boxes (classes=[0] means only 'person')
        # Optimization 1: Use OpenVINO Compiled format for Apple Silicon hardware acceleration 
        # Stabilization: Use ByteTrack and enforce 0.6 confidence to prevent generic blobs from getting track IDs
        results = yolo_model.track(frame, persist=True, classes=[0], tracker="config/bytetrack.yaml", conf=0.6, verbose=False)
        
        # We need to calculate distance between fighters if exactly 2 are detected
        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            xyxys = boxes.xyxy.cpu().numpy()
            
            # --- Enforce 2 Fighters Maximum ---
            # Sort detected boxes by area to ensure we only track the two largest people (the fighters)
            box_areas = [(xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]) for xyxy in xyxys]
            
            # Create a list of tuples (area, index) and sort descending
            sorted_indices = [i for _, i in sorted(zip(box_areas, range(len(box_areas))), reverse=True)]
            
            # Take only the top 2 largest boxes
            top_2_indices = sorted_indices[:2]
            
            # --- Personalization: You vs Opponent (or Generic F1 vs F2) ---
            # Extract coordinates for the top two tracked fighters
            f1_idx = top_2_indices[0] if len(top_2_indices) > 0 else None
            f2_idx = top_2_indices[1] if len(top_2_indices) > 1 else None
            
            # Helper to get center of a bounding box
            def get_bbox_center(bbox):
                return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            
            # Scenario A: First frame initialization if user_bbox is provided
            if user_bbox and current_user_id is None and f1_idx is not None and f2_idx is not None:
                box1 = xyxys[f1_idx]
                box2 = xyxys[f2_idx]
                iou1 = calculate_iou(box1, user_bbox)
                iou2 = calculate_iou(box2, user_bbox)
                
                if iou1 > iou2:
                    current_user_id = track_ids[f1_idx]
                    current_opponent_id = track_ids[f2_idx]
                    user_color_profile = extract_color_histogram(frame, box1)
                    opponent_color_profile = extract_color_histogram(frame, box2)
                else:
                    current_user_id = track_ids[f2_idx]
                    current_opponent_id = track_ids[f1_idx]
                    user_color_profile = extract_color_histogram(frame, box2)
                    opponent_color_profile = extract_color_histogram(frame, box1)
                    
                user_tracked_ids.discard(current_opponent_id)
                opponent_tracked_ids.discard(current_user_id)
                user_tracked_ids.add(current_user_id)
                opponent_tracked_ids.add(current_opponent_id)
                
            # Scenario A.5: First frame initialization without user_bbox (General mode)
            elif not user_bbox and current_user_id is None and f1_idx is not None and f2_idx is not None:
                b1_center_x = get_bbox_center(xyxys[f1_idx])[0]
                b2_center_x = get_bbox_center(xyxys[f2_idx])[0]
                
                # We assign "User" to whoever is on the Left, "Opponent" to Right
                if b1_center_x < b2_center_x:
                    current_user_id = track_ids[f1_idx]
                    current_opponent_id = track_ids[f2_idx]
                    user_color_profile = extract_color_histogram(frame, xyxys[f1_idx])
                    opponent_color_profile = extract_color_histogram(frame, xyxys[f2_idx])
                else:
                    current_user_id = track_ids[f2_idx]
                    current_opponent_id = track_ids[f1_idx]
                    user_color_profile = extract_color_histogram(frame, xyxys[f2_idx])
                    opponent_color_profile = extract_color_histogram(frame, xyxys[f1_idx])
                    
                user_tracked_ids.discard(current_opponent_id)
                opponent_tracked_ids.discard(current_user_id)
                user_tracked_ids.add(current_user_id)
                opponent_tracked_ids.add(current_opponent_id)
                    
            # Scenario B: Tracking identities when YOLO loses track and creates new IDs
            elif current_user_id is not None and current_opponent_id is not None and f1_idx is not None and f2_idx is not None:
                current_top_2_ids = [track_ids[f1_idx], track_ids[f2_idx]]
                
                # If YOLO swapped IDs (e.g., after a clinch)
                if current_user_id not in current_top_2_ids or current_opponent_id not in current_top_2_ids:
                    
                    # 1. Primary Re-ID Strategy: HSV Color Profiles
                    box1_color = extract_color_histogram(frame, xyxys[f1_idx])
                    box2_color = extract_color_histogram(frame, xyxys[f2_idx])
                    
                    if user_color_profile is not None and box1_color is not None and box2_color is not None:
                        # Compare Box 1 and Box 2 to the User's historic color profile
                        score_b1_user = compare_histograms(box1_color, user_color_profile)
                        score_b2_user = compare_histograms(box2_color, user_color_profile)
                        
                        # Lower Bhattacharyya distance = more visually similar
                        if score_b1_user < score_b2_user:
                            current_user_id = track_ids[f1_idx]
                            current_opponent_id = track_ids[f2_idx]
                        else:
                            current_user_id = track_ids[f2_idx]
                            current_opponent_id = track_ids[f1_idx]
                    
                    # 2. Fallback Re-ID Strategy: Spatial Memory
                    elif last_known_user_center is not None and last_known_opponent_center is not None:
                        b1_center = get_bbox_center(xyxys[f1_idx])
                        b2_center = get_bbox_center(xyxys[f2_idx])
                        dist_b1_user = np.sqrt((b1_center[0] - last_known_user_center[0])**2 + (b1_center[1] - last_known_user_center[1])**2)
                        dist_b2_user = np.sqrt((b2_center[0] - last_known_user_center[0])**2 + (b2_center[1] - last_known_user_center[1])**2)
                        
                        if dist_b1_user < dist_b2_user:
                            current_user_id = track_ids[f1_idx]
                            current_opponent_id = track_ids[f2_idx]
                        else:
                            current_user_id = track_ids[f2_idx]
                            current_opponent_id = track_ids[f1_idx]
                            
                    user_tracked_ids.discard(current_opponent_id)
                    opponent_tracked_ids.discard(current_user_id)
                    user_tracked_ids.add(current_user_id)
                    opponent_tracked_ids.add(current_opponent_id)
                            
            # Update last known spatial positions for Re-ID recovery
            if current_user_id is not None and current_opponent_id is not None:
                for idx in top_2_indices:
                    t_id = track_ids[idx]
                    if t_id == current_user_id:
                        last_known_user_center = get_bbox_center(xyxys[idx])
                    elif t_id == current_opponent_id:
                        last_known_opponent_center = get_bbox_center(xyxys[idx])
            
            # --- Store this frame's 3D feet coordinates for the Distance analytic ---
            current_frame_feet = {}
            
            for i in top_2_indices:
                track_id = track_ids[i]
                # Ensure this fighter exists in our state dictionary
                if track_id not in fighters_data:
                    # Assign a tracker from the pool
                    assigned_tracker = pose_trackers[tracker_index % MAX_TRACKERS]
                    tracker_index += 1
                    
                    fighters_data[track_id] = {
                        'pose_tracker': assigned_tracker,
                        'sequence': [],
                        'stance_history': [],
                        'current_state': "Idle",
                        'frames_since_last_log': 0,
                        'punches': [],
                        'prev_left_wrist': None,  # For velocity tracking
                        'prev_right_wrist': None,
                        'appearance': None  # Will be extracted on first landmark detection
                    }
                    
                fighter = fighters_data[track_id]
                fighter['frames_since_last_log'] += 1
                
                # Get exact bounding box for this fighter
                x1, y1, x2, y2 = map(int, xyxys[i])
                
                # Expand box slightly to ensure limbs aren't cut off
                h, w = frame.shape[:2]
                pad = 30
                x1, y1 = max(0, x1-pad), max(0, y1-pad)
                x2, y2 = min(w, x2+pad), min(h, y2+pad)
                
                fighter_crop = frame[y1:y2, x1:x2]
                if fighter_crop.size == 0: continue
                
                # 2. Run MediaPipe ON THE CROPPED IMAGE
                image_rgb = cv2.cvtColor(fighter_crop, cv2.COLOR_BGR2RGB)
                pose_results = fighter['pose_tracker'].process(image_rgb)

                if pose_results.pose_landmarks:
                    landmarks = pose_results.pose_landmarks.landmark
                    frame_features = []
                    
                    for lm in landmarks[:21]:
                        frame_features.extend([lm.x, lm.y, lm.z])
                        
                    fighter['sequence'].append(frame_features)
                    
                    # --- Advanced Analytics ---
                    mp_pose_lm = mp_pose.PoseLandmark
                    
                    # --- Extract Fighter Appearance (once per fighter) ---
                    if fighter.get('appearance') is None:
                        crop_w = x2 - x1
                        crop_h = y2 - y1
                        fighter['appearance'] = extract_fighter_appearance(
                            frame, landmarks, mp_pose_lm, x1, y1, crop_w, crop_h
                        )
                        print(f"Fighter {track_id} appearance: {fighter['appearance']}")
                    
                    left_ankle = landmarks[mp_pose_lm.LEFT_ANKLE.value]
                    right_ankle = landmarks[mp_pose_lm.RIGHT_ANKLE.value]
                    stance_width = calculate_distance_3d(left_ankle, right_ankle)
                    
                    # Store global ankle coords for fighter distance
                    current_frame_feet[track_id] = {
                        # Map cropped relative coords back to global pixel space
                        # MediaPipe x,y are normalized [0.0, 1.0], multiply by crop size and add offset
                        'global_x': (left_ankle.x * (x2 - x1)) + x1,
                        'global_y': (left_ankle.y * (y2 - y1)) + y1,
                        'z': left_ankle.z # depth is relative, keep as is
                    }
                    
                    left_wrist = landmarks[mp_pose_lm.LEFT_WRIST.value]
                    right_wrist = landmarks[mp_pose_lm.RIGHT_WRIST.value]
                    nose = landmarks[mp_pose_lm.NOSE.value]
                    left_guard_dist = calculate_distance_3d(left_wrist, nose)
                    right_guard_dist = calculate_distance_3d(right_wrist, nose)
                    avg_guard_distance = (left_guard_dist + right_guard_dist) / 2.0
                    
                    left_shoulder = landmarks[mp_pose_lm.LEFT_SHOULDER.value]
                    left_hip = landmarks[mp_pose_lm.LEFT_HIP.value]
                    left_knee = landmarks[mp_pose_lm.LEFT_KNEE.value]
                    torso_angle = calculate_angle_3d(left_shoulder, left_hip, left_knee)
                    
                    left_elbow = landmarks[mp_pose_lm.LEFT_ELBOW.value]
                    left_wrist = landmarks[mp_pose_lm.LEFT_WRIST.value]
                    left_elbow_angle = calculate_angle_3d(left_shoulder, left_elbow, left_wrist)
                    
                    right_shoulder = landmarks[mp_pose_lm.RIGHT_SHOULDER.value]
                    right_elbow = landmarks[mp_pose_lm.RIGHT_ELBOW.value]
                    right_wrist = landmarks[mp_pose_lm.RIGHT_WRIST.value]
                    right_elbow_angle = calculate_angle_3d(right_shoulder, right_elbow, right_wrist)
                    
                    # 1. Determine Left/Right Static ID based on Re-ID tracking graph
                    # Instead of calculating X position dynamically every frame (which breaks when crossing over),
                    # we rely on the tracked `current_user_id` vs `current_opponent_id` which uses persistent box math across clinches!
                    if track_id == current_user_id or track_id in user_tracked_ids:
                        fighter_static_id = 1
                    else:
                        fighter_static_id = 2
                    
                    # Store center X of both explicit F1 and F2 for distance math internally
                    other_center_x = 0
                    if fighter_static_id == 1 and last_known_opponent_center is not None:
                        other_center_x = last_known_opponent_center[0]
                    elif fighter_static_id == 2 and last_known_user_center is not None:
                        other_center_x = last_known_user_center[0]
                    else:
                        # Fallback
                        current_center_x = (x1 + x2) / 2
                        if f1_idx is not None and f2_idx is not None:
                            other_idx = f2_idx if track_id == track_ids[f1_idx] else f1_idx
                            ox1, _, ox2, _ = map(int, xyxys[other_idx])
                            other_center_x = (ox1 + ox2) / 2
                        else:
                            other_center_x = current_center_x # Failsafe
                    
                    shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
                    head_offset_x = nose.x - shoulder_center_x 
                    
                    # Stance Classification
                    # "If the left foot is in front it is always orthodox, if the right foot is in front the stance is southpaw"
                    # What defines "in front"? The foot closest to the opponent on the X-axis.
                    # We have the OTHER fighter's center_x from the ID assignment block (other_center_x).
                    # We just see which ankle's X coordinate is closer to other_center_x.
                    
                    raw_stance = "Orthodox" # Fallback
                    if f1_idx is not None and f2_idx is not None:
                        # Convert normalized ankles to absolute X pixels in the crop, then add crop offset (x1)
                        global_left_ankle_x = (left_ankle.x * (x2 - x1)) + x1
                        global_right_ankle_x = (right_ankle.x * (x2 - x1)) + x1
                        
                        dist_left_to_opp = abs(global_left_ankle_x - other_center_x)
                        dist_right_to_opp = abs(global_right_ankle_x - other_center_x)
                        
                        # The foot with the SMALLER distance to the opponent is "in front"
                        if dist_left_to_opp < dist_right_to_opp:
                            raw_stance = "Orthodox"
                        else:
                            raw_stance = "Southpaw"
                            
                    # Smooth Stance over 60 frames to prevent jitter when squared up
                    # (Increased from 30 because side-angle cameras cause ankle X-positions to appear very close)
                    fighter['stance_history'].append(raw_stance)
                    if len(fighter['stance_history']) > 60:
                        fighter['stance_history'].pop(0)
                        
                    stance_classification = max(set(fighter['stance_history']), key=fighter['stance_history'].count)
                    
                    # --- AI Prediction Loop ---
                    if len(fighter['sequence']) == sequence_length:
                        input_tensor = torch.tensor([fighter['sequence']], dtype=torch.float32)
                        
                        with torch.no_grad():
                            output = model(input_tensor)
                            probabilities = F.softmax(output, dim=1)
                            predicted_class = torch.argmax(probabilities, dim=1).item()
                            confidence = probabilities[0][predicted_class].item()
                            
                        punch_name = get_boxing_class_name(predicted_class)
                        
                        if confidence >= confidence_threshold and punch_name != "Idle":
                            
                            # --- Jab/Cross/Hook Mathematical Override ---
                            # The base PyTorch model fails to distinguish between Left/Right arms.
                            # We will mathematically determine which arm is punching based on reach,
                            # and strictly classify Jab vs Cross based on Stance.
                            
                            elbow_bend = 0.0
                            if punch_name in ["Jab", "Cross", "Hook", "Uppercut", "Straight"]:
                                sc_x = (left_shoulder.x + right_shoulder.x) / 2
                                sc_y = (left_shoulder.y + right_shoulder.y) / 2
                                
                                # Use absolute global X distances to see which hand is physically extending closer to opponent
                                if f1_idx is not None and f2_idx is not None:
                                    global_left_wrist_x = (left_wrist.x * (x2 - x1)) + x1
                                    global_right_wrist_x = (right_wrist.x * (x2 - x1)) + x1
                                    
                                    dist_left_wrist_to_opp = abs(global_left_wrist_x - other_center_x)
                                    dist_right_wrist_to_opp = abs(global_right_wrist_x - other_center_x)
                                    
                                    # Arm extending closer to opponent's center X is the "active punch"
                                    if dist_left_wrist_to_opp < dist_right_wrist_to_opp:
                                        active_arm = "Left"
                                        active_elbow_angle = left_elbow_angle
                                    else:
                                        active_arm = "Right"
                                        active_elbow_angle = right_elbow_angle
                                else:
                                    # Fallback to local coordinate stretch if the opponent tracker is temporarily lost
                                    dist_l_stretch = np.sqrt((left_wrist.x - sc_x)**2 + (left_wrist.y - sc_y)**2)
                                    dist_r_stretch = np.sqrt((right_wrist.x - sc_x)**2 + (right_wrist.y - sc_y)**2)
                                    if dist_l_stretch > dist_r_stretch:
                                        active_arm = "Left"
                                        active_elbow_angle = left_elbow_angle
                                    else:
                                        active_arm = "Right"
                                        active_elbow_angle = right_elbow_angle
                                
                                # Calculate elbow bend once for all classification checks
                                elbow_bend = 180.0 - active_elbow_angle
                                
                                # Track active wrist and shoulder for Uppercut/Body detection
                                if active_arm == "Left":
                                    active_wrist = left_wrist
                                    active_shoulder = left_shoulder
                                else:
                                    active_wrist = right_wrist
                                    active_shoulder = right_shoulder
                                
                                # --- Uppercut Detection (HIGHEST PRIORITY) ---
                                # An uppercut has the wrist rising vertically (Y decreases in image coords)
                                # while the elbow stays heavily bent (loaded arm, not straight)
                                # Check: wrist Y is ABOVE shoulder Y AND elbow bend is 60-120°
                                hip_center_y = (landmarks[mp_pose_lm.LEFT_HIP.value].y + landmarks[mp_pose_lm.RIGHT_HIP.value].y) / 2
                                
                                if active_wrist.y < active_shoulder.y and 60.0 < elbow_bend < 120.0:
                                    punch_name = "Uppercut"
                                
                                # --- Hook Detection (elbow bend > 90° AND wrist is moving fast) ---
                                # A guard position has bent arms but ZERO velocity.
                                # A real hook has the wrist traveling horizontally at speed.
                                # We gate hooks behind a minimum wrist displacement threshold.
                                elif elbow_bend > 90.0:
                                    # Calculate wrist velocity (displacement from previous frame)
                                    wrist_velocity = 0.0
                                    if active_arm == "Left" and fighter.get('prev_left_wrist') is not None:
                                        prev = fighter['prev_left_wrist']
                                        wrist_velocity = np.sqrt((active_wrist.x - prev[0])**2 + (active_wrist.y - prev[1])**2)
                                    elif active_arm == "Right" and fighter.get('prev_right_wrist') is not None:
                                        prev = fighter['prev_right_wrist']
                                        wrist_velocity = np.sqrt((active_wrist.x - prev[0])**2 + (active_wrist.y - prev[1])**2)
                                    
                                    # Min velocity threshold: 0.05 normalized units/frame
                                    # (A real hook displaces the wrist significantly per frame)
                                    if wrist_velocity > 0.05:
                                        punch_name = "Hook"
                                    else:
                                        # Arm is bent but stationary = guard position, classify as straight punch
                                        if stance_classification == "Orthodox":
                                            punch_name = "Jab" if active_arm == "Left" else "Cross"
                                        else:
                                            punch_name = "Jab" if active_arm == "Right" else "Cross"
                                
                                # --- Jab / Cross (straight punches) ---
                                else:
                                    # Jab = Front Hand / Cross = Back Hand
                                    if stance_classification == "Orthodox":
                                        if active_arm == "Left":
                                            punch_name = "Jab"
                                        else:
                                            punch_name = "Cross"
                                    elif stance_classification == "Southpaw":
                                        if active_arm == "Right":
                                            punch_name = "Jab"
                                        else:
                                            punch_name = "Cross"
                                
                                # --- Body Shot Classification ---
                                # If the active wrist Y is below the fighter's own hip center,
                                # it's targeting the body rather than the head
                                if active_wrist.y > hip_center_y:
                                    punch_name = punch_name + " (Body)"
                                
                                # Update previous wrist positions for velocity tracking next frame
                                fighter['prev_left_wrist'] = (left_wrist.x, left_wrist.y)
                                fighter['prev_right_wrist'] = (right_wrist.x, right_wrist.y)
                                        
                            if punch_name != fighter['current_state']:
                                fighter['punches'].append({
                                    "fighter_id": fighter_static_id,
                                    "raw_tracker_id": track_id,
                                    "punch_type": punch_name,
                                    "start_time_sec": round(current_time_sec, 2),
                                    "confidence": round(confidence, 2),
                                    "analytics": {
                                        "stance": stance_classification,
                                        "stance_width": round(float(stance_width), 3),
                                        "avg_guard_distance": round(float(avg_guard_distance), 3),
                                        "torso_angle_degrees": round(float(torso_angle), 1),
                                        "head_offset_x": round(float(head_offset_x), 3),
                                        "active_elbow_bend": round(float(elbow_bend), 1)
                                    }
                                })
                                fighter['current_state'] = punch_name
                                fighter['frames_since_last_log'] = 0
                        
                        if punch_name == "Idle" and confidence >= 0.5:
                             fighter['current_state'] = "Idle"
                             
                        fighter['sequence'].pop(0)
                        
            # --- Global Ring Analytics (Distance Between Fighters) ---
            # If exactly 2 fighters are tracked this frame, calculate distance between their lead feet
            if len(current_frame_feet.keys()) == 2:
                ids = list(current_frame_feet.keys())
                f1_foot = current_frame_feet[ids[0]]
                f2_foot = current_frame_feet[ids[1]]
                
                # 2D Euclidean distance of the global pixel coordinates
                dist_pixels = np.sqrt((f1_foot['global_x'] - f2_foot['global_x'])**2 + (f1_foot['global_y'] - f2_foot['global_y'])**2)
                
                initiator = None
                if previous_global_distance is not None and previous_global_feet:
                    # Did the distance between them shrink?
                    if dist_pixels < previous_global_distance - 2.0: # 2 pixels threshold to ignore jitter
                        m1 = 0
                        m2 = 0
                        
                        # Calculate how much each fighter moved since the last tracked frame
                        if ids[0] in previous_global_feet:
                            prev_f1 = previous_global_feet[ids[0]]
                            m1 = np.sqrt((f1_foot['global_x'] - prev_f1['global_x'])**2 + (f1_foot['global_y'] - prev_f1['global_y'])**2)
                            
                        if ids[1] in previous_global_feet:
                            prev_f2 = previous_global_feet[ids[1]]
                            m2 = np.sqrt((f2_foot['global_x'] - prev_f2['global_x'])**2 + (f2_foot['global_y'] - prev_f2['global_y'])**2)
                        
                        # The one who moved more is the initiator of the interaction
                        if m1 > m2 + 1.0: # +1px threshold to ensure decisive movement
                            if current_user_id and current_opponent_id:
                                initiator = "You" if ids[0] in user_tracked_ids else "Opponent"
                            else:
                                initiator = f"Tracker {ids[0]}"
                        elif m2 > m1 + 1.0:
                            if current_user_id and current_opponent_id:
                                initiator = "You" if ids[1] in user_tracked_ids else "Opponent"
                            else:
                                initiator = f"Tracker {ids[1]}"
                
                filtered_analytic = {
                    "start_time_sec": round(current_time_sec, 2),
                    "distance_between_fighters_px": round(float(dist_pixels), 2)
                }
                
                if initiator:
                     filtered_analytic["distance_closed_by"] = initiator
                     
                frame_analytics.append(filtered_analytic)
                
                previous_global_distance = dist_pixels
                previous_global_feet = current_frame_feet

    cap.release()
    for tracker in pose_trackers:
        tracker.close()

    # Flatten all punches
    consolidated_punches = []
    for f in fighters_data.values():
        consolidated_punches.extend(f['punches'])
        
    consolidated_punches.sort(key=lambda x: x['start_time_sec'])

    # Format output based on whether a User BBox was provided
    if user_bbox:
        user_punches = []
        user_defenses = []
        opponent_punches = []
        opponent_defenses = []
        
        for p in consolidated_punches:
            is_defensive = p['punch_type'] in ["Slip", "Duck", "Block"]
            # Aggregate all punches based on the collection of historical IDs
            if p.get('raw_tracker_id') in user_tracked_ids:
                if is_defensive:
                    user_defenses.append(p)
                else:
                    user_punches.append(p)
            elif p.get('raw_tracker_id') in opponent_tracked_ids:
                if is_defensive:
                    opponent_defenses.append(p)
                else:
                    opponent_punches.append(p)
                
        # Collect appearance data per role
        user_appearance = None
        opponent_appearance = None
        for tid, fdata in fighters_data.items():
            if tid in user_tracked_ids and fdata.get('appearance'):
                user_appearance = fdata['appearance']
            elif tid in opponent_tracked_ids and fdata.get('appearance'):
                opponent_appearance = fdata['appearance']
        
        return {
            "global_fight_analytics": frame_analytics,
            "you": {
                "appearance": user_appearance or {"headgear": "Unknown", "gloves": "Unknown", "trunks": "Unknown"},
                "punches_identified": user_punches,
                "defensive_moves": user_defenses
            },
            "opponent": {
                "appearance": opponent_appearance or {"headgear": "Unknown", "gloves": "Unknown", "trunks": "Unknown"},
                "punches_identified": opponent_punches,
                "defensive_moves": opponent_defenses
            }
        }
    else:
        # Default behavior (Left/Right) — collect appearances for all fighters
        fighter_appearances = {}
        for tid, fdata in fighters_data.items():
            if fdata.get('appearance'):
                # Map tracker ID to static fighter ID (1=Left, 2=Right)
                if tid in user_tracked_ids:
                    fighter_appearances["Fighter 1 (Left)"] = fdata['appearance']
                elif tid in opponent_tracked_ids:
                    fighter_appearances["Fighter 2 (Right)"] = fdata['appearance']
                else:
                    fighter_appearances[f"Tracker {tid}"] = fdata['appearance']
        
        return {
            "global_fight_analytics": frame_analytics,
            "fighter_appearances": fighter_appearances,
            "punches_identified": consolidated_punches
        }



# --- Example Usage (If running script directly) ---
if __name__ == "__main__":
    import argparse
    import json
    import os
    
    parser = argparse.ArgumentParser(description="Sparring AI Video Analyzer")
    parser.add_argument("video", help="Path to the video file to analyze")
    parser.add_argument("--user-bbox", type=str, help="Optional initial bounding box for the user (format: x1,y1,x2,y2)")
    parser.add_argument("--draw-bbox", action="store_true", help="Interactively draw the initial bounding box for the user on the first frame")
    
    args = parser.parse_args()
    test_video = args.video
    
    # Parse the user bbox if provided
    user_bbox = None
    if args.user_bbox:
        try:
            user_bbox = [float(x) for x in args.user_bbox.split(",")]
            if len(user_bbox) != 4:
                raise ValueError
        except ValueError:
            print("Error: --user-bbox must be in the format x1,y1,x2,y2 (e.g., 100,200,300,400)")
            sys.exit(1)
            
    # Interactively draw bbox if requested
    if args.draw_bbox and not user_bbox:
        import cv2
        print(f"Opening first frame of {test_video}...")
        cap = cv2.VideoCapture(test_video)
        ret, frame = cap.read()
        if ret:
            print("Please draw a bounding box around 'You'.")
            print(" - Click and drag to draw the box.")
            print(" - Press SPACE or ENTER to confirm.")
            print(" - Press C to cancel.")
            
            roi = cv2.selectROI("Select User Bounding Box", frame, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("Select User Bounding Box")
            cv2.waitKey(1)  # Required to fully close the window on macOS
            
            # roi is (x, y, w, h). If not selected, it returns (0, 0, 0, 0)
            if roi != (0, 0, 0, 0):
                x, y, w, h = roi
                user_bbox = [float(x), float(y), float(x + w), float(y + h)]
            else:
                print("No bounding box selected. Proceeding with default Left/Right tracking.")
        cap.release()
            
    print(f"Analyzing {test_video} for punches and analytics...")
    if user_bbox:
        print(f"User Identification Bounding Box provided: {user_bbox}")
        
    punches = identify_punches_in_video(test_video, confidence_threshold=0.7, user_bbox=user_bbox)
    
    # Save the results to a JSON file
    video_basename = os.path.basename(test_video)
    video_name, _ = os.path.splitext(video_basename)
    json_path = f"results/{video_name}_results.json"
    
    with open(json_path, "w") as f:
        json.dump(punches, f, indent=4)
        
    print("\n--- Analysis Results ---")
    if user_bbox:
        print(f"Results Split into 'you' and 'opponent' and saved to '{json_path}'")
    else:
        if not punches['punches_identified']:
            print("No punches detected matching the confidence threshold.")
        else:
            for idx, p in enumerate(punches['punches_identified']):
                fighter_name = "Fighter 1 (Left)" if p['fighter_id'] == 1 else "Fighter 2 (Right)"
                print(f"{idx+1}. {p['punch_type']} detected at {p['start_time_sec']}s (Confidence: {p['confidence']} - {fighter_name})")
        print(f"\nResults have been saved to '{json_path}'")