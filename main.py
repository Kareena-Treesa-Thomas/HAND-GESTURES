import cv2
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe import Image, ImageFormat
import numpy as np
import pyautogui
import keyboard
import time
from collections import deque
import math

class HandGestureController:
    def __init__(self):
        # Initialize MediaPipe
        base_options = BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Gesture tracking variables
        self.hand_history = deque(maxlen=15)  # For swipe detection with 4/5 fingers
        self.gesture_start_time = None
        self.last_gesture = None
        self.gesture_cooldown = 0.5  # Reduced cooldown for better responsiveness
        
        # Finger count tracking for both hands
        self.left_hand_fingers = 0
        self.right_hand_fingers = 0
        self.left_finger_history = deque(maxlen=5)
        self.right_finger_history = deque(maxlen=5)
        self.min_stable_frames = 5  # Frames needed for stable finger count
        
        # Gesture state management
        self.gesture_in_progress = False
        self.gesture_completed = False
        self.last_gesture_end_time = 0
        self.gesture_reset_delay = 0.3  # Time to wait before detecting new gestures
        
        # 4/5 finger swipe detection for tab switching
        self.swipe_start_x = None
        self.swipe_threshold = 0.15  # Minimum horizontal movement for swipe
        
        # Scroll control
        self.last_scroll_time = 0
        self.scroll_delay = 0.1  # Delay between scroll actions
        
        # Two-hand detection for close
        self.both_hands_detected = False
        self.close_gesture_start_time = None
        
        print("Hand Gesture Controller initialized")
        print("Gestures:")
        print("- 1 Finger: Scroll Down")
        print("- 2 Fingers: Scroll Up")
        print("- 4 or 5 Fingers + Swipe Right: Switch to next tab")
        print("- Both Hands with 5 Fingers Each: Minimize application")
        print("- Press 'q' to quit")

    def detect_hand_landmarks(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect(mp_image)
        return result

    def get_hand_center(self, landmarks):
        if not landmarks:
            return None
        
        x_coords = [landmark.x for landmark in landmarks]
        y_coords = [landmark.y for landmark in landmarks]
        
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        
        return (center_x, center_y)

    def count_fingers(self, landmarks):
        """Count number of extended fingers with much more accurate detection"""
        if not landmarks or len(landmarks) < 21:
            return 0
        
        finger_count = 0
        
        # Thumb detection - more robust
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        
        # Check thumb extension based on horizontal position
        thumb_extended = False
        if abs(thumb_tip.x - thumb_ip.x) > 0.05:  # Significant horizontal separation
            thumb_extended = True
        
        # Other fingers with very strict criteria
        finger_states = []
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
        finger_pips = [6, 10, 14, 18]  # Index, Middle, Ring, Pinky PIP joints
        finger_mcps = [5, 9, 13, 17]  # Index, Middle, Ring, Pinky MCP joints
        
        for i, (tip, pip, mcp) in enumerate(zip(finger_tips, finger_pips, finger_mcps)):
            tip_y = landmarks[tip].y
            pip_y = landmarks[pip].y
            mcp_y = landmarks[mcp].y
            
            # Very strict finger extension detection
            # Finger must be significantly above both PIP and MCP
            extension_threshold = 0.03  # Stricter threshold
            
            if tip_y < pip_y and tip_y < mcp_y and (pip_y - tip_y) > extension_threshold:
                finger_states.append(True)
                finger_count += 1
            else:
                finger_states.append(False)
        
        # Add thumb if extended
        if thumb_extended:
            finger_count += 1
        
        return finger_count
    
    def detect_hand_type(self, landmarks):
        """Detect if hand is left or right based on wrist and finger positions"""
        if not landmarks:
            return "unknown"
        
        # Use wrist position relative to finger tips to determine handedness
        wrist = landmarks[0]
        
        # Get average x position of fingertips
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        avg_finger_x = sum(landmarks[tip].x for tip in finger_tips) / len(finger_tips)
        
        # If fingertips are to the right of wrist, it's a right hand (mirror effect)
        if avg_finger_x > wrist.x:
            return "right"
        else:
            return "left"
    
    def detect_multi_finger_swipe(self, landmarks, hand_center, finger_count):
        """Detect right swipe gesture when 4 or 5 fingers are shown"""
        if not landmarks or not hand_center:
            return None, 0
        
        # Only detect swipe if we have 4 or 5 fingers
        if finger_count not in [4, 5]:
            self.swipe_start_x = None
            return None, 0
        
        current_time = time.time()
        
        # Initialize swipe start position
        if self.swipe_start_x is None:
            self.swipe_start_x = hand_center[0]
            return None, 0
        
        # Calculate horizontal movement (only right swipe)
        movement = hand_center[0] - self.swipe_start_x
        
        # Check if right movement is significant enough
        if movement > self.swipe_threshold:
            confidence = min(movement / 0.3, 1.0)  # Normalize confidence
            return "swipe_right", confidence
        
        # Reset if movement is left or too small
        if movement < -0.05:  # Left movement resets the swipe
            self.swipe_start_x = hand_center[0]
        
        return None, 0

    def execute_gesture(self, gesture, confidence=0):
        current_time = time.time()
        
        # Apply cooldown to all gestures except scroll
        if gesture in ["swipe_right", "close_app"]:
            if self.last_gesture and (current_time - self.gesture_start_time) < self.gesture_cooldown:
                return
        
        # Apply scroll delay
        if gesture in ["scroll_up", "scroll_down"]:
            if current_time - self.last_scroll_time < self.scroll_delay:
                return
            self.last_scroll_time = current_time
        
        if gesture == "swipe_right":
            pyautogui.hotkey('ctrl', 'tab')
            print("Gesture: 4/5 Fingers Swipe Right - Next Tab")
            self.last_gesture = gesture
            self.gesture_start_time = current_time
            self.gesture_completed = True
            self.last_gesture_end_time = current_time
            self.swipe_start_x = None  # Reset for next swipe
            
        elif gesture == "scroll_up":
            pyautogui.scroll(3)
            print("Gesture: 2 Fingers - Scroll Up")
            
        elif gesture == "scroll_down":
            pyautogui.scroll(-3)
            print("Gesture: 1 Finger - Scroll Down")
            
        elif gesture == "close_app":
            # Try multiple minimize methods
            try:
                pyautogui.hotkey('win', 'm')  # Windows minimize
            except:
                try:
                    pyautogui.hotkey('cmd', 'm')  # Mac minimize
                except:
                    pyautogui.hotkey('alt', 'space', 'n')  # Alternative minimize
            print("Gesture: Both Hands 5 Fingers - Minimize Application")
            self.last_gesture = gesture
            self.gesture_start_time = current_time
            self.gesture_completed = True
            self.last_gesture_end_time = current_time

    def draw_info(self, frame, gesture_text=""):
        h, w = frame.shape[:2]
        
        # Draw gesture status
        cv2.putText(frame, f"Gesture: {gesture_text}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw instructions
        instructions = [
            "1 Finger: Scroll Down",
            "2 Fingers: Scroll Up",
            "4/5 Fingers + Swipe Right: Next tab",
            "Both Hands 5 Fingers: Minimize app",
            "Press 'q' to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, h - 120 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect hand landmarks
                results = self.detect_hand_landmarks(frame)
                
                gesture_text = ""
                
                if results.hand_landmarks:
                    # Initialize counts for this frame
                    frame_left_fingers = 0
                    frame_right_fingers = 0
                    hands_detected = []
                    
                    for hand_index, hand_landmarks in enumerate(results.hand_landmarks):
                        # Get hand center position
                        hand_center = self.get_hand_center(hand_landmarks)
                        
                        if hand_center:
                            # Count fingers
                            finger_count = self.count_fingers(hand_landmarks)
                            handedness = results.handedness[hand_index][0].category_name.lower()
                            
                            # Debug output
                            print(f"Debug: {handedness} hand detected {finger_count} fingers")
                            
                            hands_detected.append((handedness, finger_count))
                            
                            # Update finger count for appropriate hand
                            if handedness == "left":
                                frame_left_fingers = finger_count
                            elif handedness == "right":
                                frame_right_fingers = finger_count
                            
                            # Add to history for swipe detection
                            self.hand_history.append(hand_center)
                    
                    # Update stable finger counts with smoothing
                    if frame_left_fingers > 0:
                        self.left_finger_history.append(frame_left_fingers)
                        if len(self.left_finger_history) >= self.min_stable_frames:
                            from collections import Counter
                            most_common = Counter(self.left_finger_history).most_common(1)[0][0]
                            self.left_hand_fingers = most_common
                    
                    if frame_right_fingers > 0:
                        self.right_finger_history.append(frame_right_fingers)
                        if len(self.right_finger_history) >= self.min_stable_frames:
                            from collections import Counter
                            most_common = Counter(self.right_finger_history).most_common(1)[0][0]
                            self.right_hand_fingers = most_common
                    
                    # Check for two-hand close gesture (both hands with 5 fingers)
                    if self.left_hand_fingers == 5 and self.right_hand_fingers == 5:
                        current_time = time.time()
                        if not self.both_hands_detected:
                            self.both_hands_detected = True
                            self.close_gesture_start_time = current_time
                            print("Both hands with 5 fingers detected...")
                        elif current_time - self.close_gesture_start_time > 1.5:  # Hold for 1.5 seconds
                            self.execute_gesture("close_app", 0.9)
                            gesture_text = "Both Hands 5 Fingers - Minimize App"
                    else:
                        if self.both_hands_detected:
                            print("Lost both hands gesture")
                        self.both_hands_detected = False
                        self.close_gesture_start_time = None
                    
                    # Single-hand gestures for scrolling and tab switching
                    if not self.both_hands_detected:
                        # Use hand with more fingers for gesture detection
                        active_fingers = max(self.left_hand_fingers, self.right_hand_fingers)
                        gesture_text = f"{active_fingers} Fingers"
                        
                        # Execute gestures based on finger count
                        if active_fingers == 1:
                            # 1 finger - scroll down
                            self.execute_gesture("scroll_down", 0.8)
                            gesture_text = "1 Finger - Scroll Down"
                            
                        elif active_fingers == 2:
                            # 2 fingers - scroll up
                            self.execute_gesture("scroll_up", 0.8)
                            gesture_text = "2 Fingers - Scroll Up"
                            
                        elif active_fingers in [4, 5]:
                            # 4 or 5 fingers - detect swipe for tab switching
                            # Use the hand that actually has 4/5 fingers
                            for hand_type, finger_count in hands_detected:
                                if finger_count in [4, 5] and hand_center:
                                    swipe, swipe_confidence = self.detect_multi_finger_swipe(hand_landmarks, hand_center, finger_count)
                                    if swipe and swipe_confidence > 0.6:
                                        self.execute_gesture(swipe, swipe_confidence)
                                        gesture_text = f"{finger_count} Fingers - {swipe.replace('_', ' ').title()}"
                                    break
                    
                    # Display finger counts and hand detection info
                    cv2.putText(frame, f"L: {self.left_hand_fingers} R: {self.right_hand_fingers}", (50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    if len(hands_detected) >= 2:
                        cv2.putText(frame, f"Hands: {len(hands_detected)}", (50, 140), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    # Reset when no hand detected
                    self.hand_history.clear()
                    self.left_finger_history.clear()
                    self.right_finger_history.clear()
                    self.left_hand_fingers = 0
                    self.right_hand_fingers = 0
                    self.swipe_start_x = None
                    self.both_hands_detected = False
                    self.close_gesture_start_time = None
                    self.gesture_in_progress = False
                    self.gesture_completed = False
                
                # Draw info and gestures
                self.draw_info(frame, gesture_text)
                
                # Display frame
                cv2.imshow('Hand Gesture Controller', frame)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nStopping gesture controller...")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.landmarker.close()

if __name__ == "__main__":
    controller = HandGestureController()
    controller.run()
