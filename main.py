import cv2
import numpy as np
import pyautogui
import time
from collections import deque
import math

# MediaPipe new API imports
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core import base_options
from mediapipe import Image, ImageFormat

class HandGestureController:
    def __init__(self):
        # Initialize MediaPipe HandLandmarker
        base_options_obj = base_options.BaseOptions(model_asset_path="hand_landmarker.task")
        options = HandLandmarkerOptions(
            base_options=base_options_obj,
            num_hands=2,  # detect both hands
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=RunningMode.VIDEO
        )
        self.detector = HandLandmarker.create_from_options(options)

        # Initialize camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Gesture tracking variables
        self.hand_history = deque(maxlen=15)
        self.gesture_start_time = None
        self.last_gesture = None
        self.gesture_cooldown = 0.5

        # Finger count tracking
        self.left_hand_fingers = 0
        self.right_hand_fingers = 0
        self.left_finger_history = deque(maxlen=5)
        self.right_finger_history = deque(maxlen=5)
        self.min_stable_frames = 5

        # Gesture state
        self.gesture_in_progress = False
        self.gesture_completed = False
        self.last_gesture_end_time = 0
        self.gesture_reset_delay = 0.3

        # Swipe detection
        self.swipe_start_x = None
        self.swipe_threshold = 0.15

        # Scroll control
        self.last_scroll_time = 0
        self.scroll_delay = 0.1

        # Two-hand detection
        self.both_hands_detected = False
        self.close_gesture_start_time = None

        print("Hand Gesture Controller initialized")
        print("Gestures:")
        print("- 1 Finger: Scroll Down")
        print("- 2 Fingers: Scroll Up")
        print("- 4 or 5 Fingers + Swipe Right: Switch to next tab")
        print("- Both Hands with 5 Fingers Each: Minimize application")
        print("- Press 'q' to quit")

    # ------------------ Hand Detection Functions ------------------ #
    def detect_hand_landmarks(self, frame):
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame)
        timestamp_ms = int(time.time() * 1000)
        results = self.detector.detect_for_video(mp_image, timestamp_ms)
        return results

    def get_hand_center(self, landmarks):
        if not landmarks:
            return None
        x_coords = [l.x for l in landmarks]
        y_coords = [l.y for l in landmarks]
        return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

    def count_fingers(self, landmarks):
        if not landmarks or len(landmarks) < 21:
            return 0
        finger_count = 0
        # Thumb
        thumb_extended = abs(landmarks[4].x - landmarks[3].x) > 0.05
        if thumb_extended:
            finger_count += 1
        # Other fingers
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        mcps = [5, 9, 13, 17]
        for tip, pip, mcp in zip(tips, pips, mcps):
            if landmarks[tip].y < landmarks[pip].y and landmarks[tip].y < landmarks[mcp].y:
                finger_count += 1
        return finger_count

    def detect_hand_type(self, landmarks):
        if not landmarks:
            return "unknown"
        wrist = landmarks[0]
        fingertips = [4, 8, 12, 16, 20]
        avg_x = sum(landmarks[i].x for i in fingertips) / len(fingertips)
        return "right" if avg_x > wrist.x else "left"

    def detect_multi_finger_swipe(self, landmarks, hand_center, finger_count):
        if finger_count not in [4, 5] or not hand_center:
            self.swipe_start_x = None
            return None, 0
        if self.swipe_start_x is None:
            self.swipe_start_x = hand_center[0]
            return None, 0
        movement = hand_center[0] - self.swipe_start_x
        if movement > self.swipe_threshold:
            confidence = min(movement / 0.3, 1.0)
            return "swipe_right", confidence
        if movement < -0.05:
            self.swipe_start_x = hand_center[0]
        return None, 0

    # ------------------ Gesture Execution ------------------ #
    def execute_gesture(self, gesture, confidence=0):
        current_time = time.time()
        if gesture in ["swipe_right", "close_app"]:
            if self.last_gesture and (current_time - self.gesture_start_time) < self.gesture_cooldown:
                return
        if gesture in ["scroll_up", "scroll_down"]:
            if current_time - self.last_scroll_time < self.scroll_delay:
                return
            self.last_scroll_time = current_time

        if gesture == "swipe_right":
            pyautogui.hotkey('ctrl', 'tab')
            print("Gesture: 4/5 Fingers Swipe Right - Next Tab")
        elif gesture == "scroll_up":
            pyautogui.scroll(3)
            print("Gesture: 2 Fingers - Scroll Up")
        elif gesture == "scroll_down":
            pyautogui.scroll(-3)
            print("Gesture: 1 Finger - Scroll Down")
        elif gesture == "close_app":
            pyautogui.hotkey('win', 'm')
            print("Gesture: Both Hands 5 Fingers - Minimize App")

        self.last_gesture = gesture
        self.gesture_start_time = current_time
        self.gesture_completed = True
        self.last_gesture_end_time = current_time

    # ------------------ UI / Info ------------------ #
    def draw_info(self, frame, gesture_text=""):
        h, w = frame.shape[:2]
        cv2.putText(frame, f"Gesture: {gesture_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
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

    # ------------------ Main Loop ------------------ #
    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                results = self.detect_hand_landmarks(frame)
                gesture_text = ""

                if results.hand_landmarks:
                    hands_detected = []
                    frame_left_fingers = frame_right_fingers = 0

                    for hand_landmarks in results.hand_landmarks:
                        hand_center = self.get_hand_center(hand_landmarks)
                        finger_count = self.count_fingers(hand_landmarks)
                        hand_type = self.detect_hand_type(hand_landmarks)
                        hands_detected.append((hand_type, finger_count))
                        if hand_type == "left":
                            frame_left_fingers = finger_count
                        else:
                            frame_right_fingers = finger_count

                        self.hand_history.append(hand_center)

                    # Update smoothed finger counts
                    from collections import Counter
                    if frame_left_fingers > 0:
                        self.left_finger_history.append(frame_left_fingers)
                        if len(self.left_finger_history) >= self.min_stable_frames:
                            self.left_hand_fingers = Counter(self.left_finger_history).most_common(1)[0][0]
                    if frame_right_fingers > 0:
                        self.right_finger_history.append(frame_right_fingers)
                        if len(self.right_finger_history) >= self.min_stable_frames:
                            self.right_hand_fingers = Counter(self.right_finger_history).most_common(1)[0][0]

                    # Both hands close gesture
                    if self.left_hand_fingers == 5 and self.right_hand_fingers == 5:
                        current_time = time.time()
                        if not self.both_hands_detected:
                            self.both_hands_detected = True
                            self.close_gesture_start_time = current_time
                        elif current_time - self.close_gesture_start_time > 1.5:
                            self.execute_gesture("close_app")
                            gesture_text = "Both Hands 5 Fingers - Minimize App"
                    else:
                        self.both_hands_detected = False
                        self.close_gesture_start_time = None

                    # Single hand gestures
                    if not self.both_hands_detected:
                        active_fingers = max(self.left_hand_fingers, self.right_hand_fingers)
                        gesture_text = f"{active_fingers} Fingers"

                        if active_fingers == 1:
                            self.execute_gesture("scroll_down")
                            gesture_text = "1 Finger - Scroll Down"
                        elif active_fingers == 2:
                            self.execute_gesture("scroll_up")
                            gesture_text = "2 Fingers - Scroll Up"
                        elif active_fingers in [4, 5]:
                            for hand_type, finger_count in hands_detected:
                                hand_center = self.get_hand_center(hand_landmarks)
                                swipe, confidence = self.detect_multi_finger_swipe(hand_landmarks, hand_center, finger_count)
                                if swipe and confidence > 0.6:
                                    self.execute_gesture(swipe, confidence)
                                    gesture_text = f"{finger_count} Fingers - {swipe.replace('_', ' ').title()}"
                                    break

                    cv2.putText(frame, f"L: {self.left_hand_fingers} R: {self.right_hand_fingers}", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    if len(hands_detected) >= 2:
                        cv2.putText(frame, f"Hands: {len(hands_detected)}", (50, 140),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                else:
                    self.hand_history.clear()
                    self.left_finger_history.clear()
                    self.right_finger_history.clear()
                    self.left_hand_fingers = 0
                    self.right_hand_fingers = 0
                    self.swipe_start_x = None
                    self.both_hands_detected = False
                    self.close_gesture_start_time = None

                self.draw_info(frame, gesture_text)
                cv2.imshow('Hand Gesture Controller', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.detector.close()


if __name__ == "__main__":
    controller = HandGestureController()
    controller.run()