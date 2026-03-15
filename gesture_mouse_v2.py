"""
GestureMouse MVP - Bare Python Implementation
Uses MediaPipe 0.10+ new Task API with auto-downloaded model.
"""

import cv2
import pyautogui
import math
import numpy as np
import urllib.request
import os

# Download model if not present
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded!")

download_model()

# MediaPipe 0.10+ imports
from mediapipe import tasks
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.core import BaseOptions


class GestureMouse:
    def __init__(self):
        # Setup HandLandmarker with downloaded model
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = HandLandmarker.create_from_options(options)
        
        # Screen and camera setup
        self.screen_width, self.screen_height = pyautogui.size()
        self.cap = cv2.VideoCapture(0)
        self.cam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Settings
        self.sensitivity = 1.5
        self.smoothing = 0.5
        self.prev_x = self.screen_width // 2
        self.prev_y = self.screen_height // 2
        self.left_click_pressed = False
        self.right_click_pressed = False
        self.click_cooldown = 0
        
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0
    
    def distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def map_to_screen(self, x, y):
        x = 1 - x  # Mirror
        x = (x - 0.5) * self.sensitivity + 0.5
        y = (y - 0.5) * self.sensitivity + 0.5
        x = max(0, min(1, x))
        y = max(0, min(1, y))
        return int(x * self.screen_width), int(y * self.screen_height)
    
    def smooth_move(self, tx, ty):
        sx = int(self.prev_x * (1 - self.smoothing) + tx * self.smoothing)
        sy = int(self.prev_y * (1 - self.smoothing) + ty * self.smoothing)
        self.prev_x, self.prev_y = sx, sy
        return sx, sy
    
    def process_gestures(self, landmarks):
        THUMB, INDEX, MIDDLE = 4, 8, 12
        
        thumb, index, middle = landmarks[THUMB], landmarks[INDEX], landmarks[MIDDLE]
        
        # Move cursor
        cx, cy = self.map_to_screen(index.x, index.y)
        sx, sy = self.smooth_move(cx, cy)
        pyautogui.moveTo(sx, sy)
        
        # Pinch detection
        d_index = self.distance(index, thumb)
        d_middle = self.distance(middle, thumb)
        pinch = 0.08
        
        # Left click
        left_pinch = d_index < pinch
        if left_pinch and not self.left_click_pressed and self.click_cooldown == 0:
            pyautogui.mouseDown()
            self.left_click_pressed = True
        elif not left_pinch and self.left_click_pressed:
            pyautogui.mouseUp()
            self.left_click_pressed = False
            self.click_cooldown = 5
        
        # Right click
        right_pinch = d_middle < pinch and not self.left_click_pressed
        if right_pinch and not self.right_click_pressed and self.click_cooldown == 0:
            pyautogui.rightClick()
            self.right_click_pressed = True
            self.click_cooldown = 10
        elif not right_pinch:
            self.right_click_pressed = False
        
        if self.click_cooldown > 0:
            self.click_cooldown -= 1
        
        return sx, sy, left_pinch, right_pinch
    
    def draw_hand(self, frame, landmarks):
        h, w = frame.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        
        # Connections
        pairs = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
                 (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
        for a, b in pairs:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], (0,255,0), 2)
        
        # Points
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (0,0,255), -1)
        
        return pts[8] if pts else (0, 0)  # Return index finger tip
    
    def run(self):
        print("GestureMouse MVP - Press 'q' to quit")
        print("Index finger = move, Index+Thumb = left click, Middle+Thumb = right click")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = tasks.vision.Image(image_format=tasks.vision.ImageFormat.SRGB, data=rgb)
            result = self.detector.detect(mp_img)
            
            debug = frame.copy()
            
            if result.hand_landmarks:
                for hand in result.hand_landmarks:
                    self.draw_hand(debug, hand)
                    x, y, left, right = self.process_gestures(hand)
                    
                    ix, iy = int(hand[8].x * self.cam_width), int(hand[8].y * self.cam_height)
                    color = (0,0,255) if (left or right) else (0,255,0)
                    cv2.circle(debug, (ix, iy), 15, color, 2)
                    
                    if left:
                        cv2.putText(debug, "LEFT CLICK", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    if right:
                        cv2.putText(debug, "RIGHT CLICK", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            else:
                cv2.putText(debug, "No hand", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            
            cv2.putText(debug, "GestureMouse - Press 'q' to quit", (10,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.imshow("GestureMouse", debug)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        GestureMouse().run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
