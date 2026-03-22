import cv2
import mediapipe as mp

def check_stance(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    print(f"Checking Stance in {video_path}")
    print("Format: Frame | Left Ankle X | Right Ankle X | Stance Prediction")
    print("-" * 70)
    
    while cap.isOpened() and frame_count < 60:
        ret, frame = cap.read()
        if not ret: break
        
        # Test script is limited because without YOLO, MediaPipe only finds 1 person.
        # But we can verify if the new logic makes sense based on raw X values.
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            l_ankle_x = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x
            r_ankle_x = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x
            
            # In example_2.mp4, the main visible fighter early on is facing right (towards positive X).
            # Therefore, the foot with the larger X coordinate is physically "in front".
            if l_ankle_x > r_ankle_x:
                stance = "Orthodox"
            else:
                stance = "Southpaw"
                
            print(f"{frame_count:03d} | L:{l_ankle_x:.3f} | R:{r_ankle_x:.3f} | {stance}")
            
        frame_count += 1
        
    cap.release()

if __name__ == "__main__":
    check_stance("example_videos/example_2.mp4")
