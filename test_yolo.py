import cv2
import sys
from ultralytics import YOLO

def check_yolo(video_path):
    print(f"Loading YOLO...")
    model = YOLO("yolov8n.pt") # Fast model for debug
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_start_frame = int(30 * fps)
    target_end_frame = int(40 * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_start_frame)
    
    print(f"Checking frames {target_start_frame} to {target_end_frame}")
    print("Format: Frame | Active Tracking IDs (Top 2 size)")
    print("-" * 50)
    
    frame_count = target_start_frame
    while cap.isOpened() and frame_count < target_end_frame:
        ret, frame = cap.read()
        if not ret: break
        
        results = model.track(frame, persist=True, classes=[0], verbose=False)
        
        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes
            
            # Sort by area
            areas = [(boxes.xyxy[i][2] - boxes.xyxy[i][0]) * (boxes.xyxy[i][3] - boxes.xyxy[i][1]) for i in range(len(boxes))]
            id_area_pairs = list(zip(boxes.id.int().tolist(), areas))
            id_area_pairs.sort(key=lambda x: x[1], reverse=True)
            
            top_ids = [str(x[0]) for x in id_area_pairs[:2]]
            print(f"{frame_count:04d} | IDs: {', '.join(top_ids)}")
        else:
            print(f"{frame_count:04d} | IDs: NONE")
            
        frame_count += 1
        
    cap.release()

if __name__ == "__main__":
    check_yolo("example_videos/example_2.mp4")
