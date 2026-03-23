import cv2
import sys
from ultralytics import YOLO

def check_id_mapping(video_path):
    print(f"Loading YOLO...")
    model = YOLO("models/yolov8n.pt")
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_start_frame = int(30 * fps)
    target_end_frame = int(40 * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_start_frame)
    
    print(f"Checking frames {target_start_frame} to {target_end_frame}")
    print("Format: Frame | Track ID -> Fighter Static ID (1=Left, 2=Right)")
    print("-" * 60)
    
    frame_count = target_start_frame
    while cap.isOpened() and frame_count < target_end_frame:
        ret, frame = cap.read()
        if not ret: break
        
        results = model.track(frame, persist=True, classes=[0], verbose=False)
        
        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes
            track_ids = boxes.id.int().cpu().tolist()
            xyxys = boxes.xyxy.cpu().numpy()
            
            # Keep top 2 largest
            box_areas = [(xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]) for xyxy in xyxys]
            sorted_indices = [i for _, i in sorted(zip(box_areas, range(len(box_areas))), reverse=True)]
            top_2_indices = sorted_indices[:2]
            
            f1_idx = top_2_indices[0] if len(top_2_indices) > 0 else None
            f2_idx = top_2_indices[1] if len(top_2_indices) > 1 else None
            
            mappings = []
            
            for i in top_2_indices:
                track_id = track_ids[i]
                x1, _, x2, _ = xyxys[i]
                current_center_x = (x1 + x2) / 2
                fighter_static_id = 1
                
                if f1_idx is not None and f2_idx is not None:
                    other_idx = f2_idx if track_id == track_ids[f1_idx] else f1_idx
                    ox1, _, ox2, _ = xyxys[other_idx]
                    other_center_x = (ox1 + ox2) / 2
                    
                    if current_center_x > other_center_x:
                        fighter_static_id = 2
                        
                mappings.append(f"T{track_id}->F{fighter_static_id}")
                
            print(f"{frame_count:04d} | {', '.join(mappings)}")
            
        frame_count += 1
        
    cap.release()

if __name__ == "__main__":
    check_id_mapping("example_videos/example_2.mp4")
