import cv2
import mediapipe as mp

def test_camera_and_mediapipe():
    print("Testing camera and MediaPipe...")
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return False
    
    print("✓ Camera opened successfully")
    
    # Test for a few seconds
    frame_count = 0
    hand_detected = False
    
    while frame_count < 60:  # Test for ~2 seconds at 30fps
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot read from camera")
            return False
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_detected = True
            print("✓ Hand detected!")
            break
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Testing... ({frame_count//30}s)")
    
    cap.release()
    hands.close()
    
    if hand_detected:
        print("✓ All tests passed! Camera and MediaPipe working correctly.")
        return True
    else:
        print("⚠ No hand detected in 2 seconds. This might be normal if no hand was in view.")
        print("✓ Camera and MediaPipe are initialized correctly.")
        return True

if __name__ == "__main__":
    test_camera_and_mediapipe()
    input("Press Enter to exit...")