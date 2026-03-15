"""
GestureMouse MVP - Bare Python Implementation
Controls mouse cursor via hand gestures using webcam.
No backend, no frontend - just pure Python.

Dependencies:
    pip install opencv-python mediapipe pyautogui

Controls:
    - Move index finger to control cursor
    - Pinch index+thumb for left click
    - Pinch middle+thumb for right click
    - Press 'q' to quit
"""

import cv2
import pyautogui
import math

# MediaPipe 0.10+ API
from mediapipe import tasks
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.core import BaseOptions
from mediapipe.framework.formats import landmark_pb2


class GestureMouse:
    def __init__(self):
        # Setup HandLandmarker for MediaPipe 0.10+
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=""),
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Create detector - use default model
        self.detector = HandLandmarker.create_from_options(options)
        
        # Get screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Webcam dimensions
        self.cap = cv2.VideoCapture(0)
        self.cam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Sensitivity and smoothing
        self.sensitivity = 1.5
        self.smoothing = 0.5
        self.prev_x, self.prev_y = self.screen_width // 2, self.screen_height // 2
        
        # Click state to prevent multiple clicks
        self.left_click_pressed = False
        self.right_click_pressed = False
        
        # Click cooldown
        self.click_cooldown = 0
        
        # Safety
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0
    
    def get_finger_tip(self, landmarks, finger_idx):
        """Get coordinates of a finger tip."""
        return landmarks[finger_idx]
    
    def distance(self, p1, p2):
        """Calculate Euclidean distance between two landmark points."""
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)
    
    def map_to_screen(self, x, y):
        """Map normalized camera coordinates to screen coordinates."""
        # Flip x for mirror effect
        x = 1 - x
        
        # Apply sensitivity
        x = (x - 0.5) * self.sensitivity + 0.5
        y = (y - 0.5) * self.sensitivity + 0.5
        
        # Clamp to [0, 1]
        x = max(0, min(1, x))
        y = max(0, min(1, y))
        
        # Map to screen
        screen_x = int(x * self.screen_width)
        screen_y = int(y * self.screen_height)
        
        return screen_x, screen_y
    
    def smooth_move(self, target_x, target_y):
        """Smooth cursor movement."""
        smooth_x = int(self.prev_x * (1 - self.smoothing) + target_x * self.smoothing)
        smooth_y = int(self.prev_y * (1 - self.smoothing) + target_y * self.smoothing)
        self.prev_x, self.prev_y = smooth_x, smooth_y
        return smooth_x, smooth_y
    
    def detect_gestures(self, landmarks):
        """Detect hand gestures and execute mouse actions."""
        # Finger indices
        THUMB_TIP = 4
        INDEX_TIP = 8
        MIDDLE_TIP = 12
        
        # Get finger tips
        thumb = landmarks[THUMB_TIP]
        index = landmarks[INDEX_TIP]
        middle = landmarks[MIDDLE_TIP]
        
        # Cursor control with index finger
        cursor_x, cursor_y = self.map_to_screen(index.x, index.y)
        smooth_x, smooth_y = self.smooth_move(cursor_x, cursor_y)
        
        # Move cursor
        pyautogui.moveTo(smooth_x, smooth_y)
        
        # Detect pinches for clicks
        index_thumb_dist = self.distance(index, thumb)
        middle_thumb_dist = self.distance(middle, thumb)
        
        # Threshold for pinch detection
        pinch_threshold = 0.15  # Adjusted for 3D distance
        
        # Left click: index + thumb pinch
        if index_thumb_dist < pinch_threshold:
            if not self.left_click_pressed and self.click_cooldown == 0:
                pyautogui.mouseDown()
                self.left_click_pressed = True
                print("Left click down")
        else:
            if self.left_click_pressed:
                pyautogui.mouseUp()
                self.left_click_pressed = False
                self.click_cooldown = 5
                print("Left click up")
        
        # Right click: middle + thumb pinch
        if not self.left_click_pressed and middle_thumb_dist < pinch_threshold:
            if not self.right_click_pressed and self.click_cooldown == 0:
                pyautogui.rightClick()
                self.right_click_pressed = True
                self.click_cooldown = 10
                print("Right click")
        else:
            self.right_click_pressed = False
        
        # Decrement cooldown
        if self.click_cooldown > 0:
            self.click_cooldown -= 1
        
        return smooth_x, smooth_y, index_thumb_dist < pinch_threshold, middle_thumb_dist < pinch_threshold
    
    def draw_landmarks(self, frame, landmarks):
        """Draw simple hand landmarks on frame."""
        h, w = frame.shape[:2]
        
        # Define connections (simplified hand skeleton)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]
        
        # Draw connections
        for start, end in connections:
            if start < len(landmarks) and end < len(landmarks):
                x1, y1 = int(landmarks[start].x * w), int(landmarks[start].y * h)
                x2, y2 = int(landmarks[end].x * w), int(landmarks[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw joints
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
    
    def run(self):
        """Main loop."""
        print("=" * 50)
        print("GestureMouse MVP Started")
        print("=" * 50)
        print("Controls:")
        print("  - Move index finger to control cursor")
        print("  - Pinch index + thumb for LEFT CLICK")
        print("  - Pinch middle + thumb for RIGHT CLICK")
        print("  - Press 'q' to quit")
        print("=" * 50)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = tasks.vision.Image(image_format=tasks.vision.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect hands
            result = self.detector.detect(mp_image)
            
            # Create debug overlay
            debug_frame = frame.copy()
            
            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    # Draw landmarks
                    self.draw_landmarks(debug_frame, hand_landmarks)
                    
                    # Detect gestures and control mouse
                    cursor_x, cursor_y, left_pinch, right_pinch = self.detect_gestures(hand_landmarks)
                    
                    # Visual feedback
                    cx = int(hand_landmarks[8].x * self.cam_width)
                    cy = int(hand_landmarks[8].y * self.cam_height)
                    
                    # Draw cursor indicator
                    color = (0, 255, 0) if not (left_pinch or right_pinch) else (0, 0, 255)
                    cv2.circle(debug_frame, (cx, cy), 15, color, 2)
                    
                    # Draw pinch indicators
                    if left_pinch:
                        cv2.putText(debug_frame, "LEFT CLICK", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if right_pinch:
                        cv2.putText(debug_frame, "RIGHT CLICK", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                # No hand detected
                cv2.putText(debug_frame, "No hand detected", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display status
            cv2.putText(debug_frame, "GestureMouse MVP - Press 'q' to quit", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow("GestureMouse", debug_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nGestureMouse stopped.")


if __name__ == "__main__":
    try:
        mouse = GestureMouse()
        mouse.run()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
