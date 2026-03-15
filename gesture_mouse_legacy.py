"""
GestureMouse MVP - For MediaPipe 0.9.x (legacy solutions API)
First downgrade MediaPipe: pip install mediapipe==0.9.3.0

Controls:
    - Move index finger to control cursor
    - Pinch index+thumb for left click
    - Pinch middle+thumb for right click
    - Press 'q' to quit
"""

import cv2
import mediapipe as mp
import pyautogui
import math


class GestureMouse:
    def __init__(self):
        # Legacy MediaPipe API (0.9.x)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Screen and camera
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
        
        thumb = landmarks.landmark[THUMB]
        index = landmarks.landmark[INDEX]
        middle = landmarks.landmark[MIDDLE]
        
        # Move cursor
        cx, cy = self.map_to_screen(index.x, index.y)
        sx, sy = self.smooth_move(cx, cy)
        pyautogui.moveTo(sx, sy)
        
        # Pinch detection
        d_index = self.distance(index, thumb)
        d_middle = self.distance(middle, thumb)
        pinch = 0.05
        
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
    
    def run(self):
        print("GestureMouse MVP - Press 'q' to quit")
        print("Index finger = move, Index+Thumb = left click, Middle+Thumb = right click")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb)
            
            debug = frame.copy()
            
            if result.multi_hand_landmarks:
                for hand in result.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(debug, hand, self.mp_hands.HAND_CONNECTIONS)
                    x, y, left, right = self.process_gestures(hand)
                    
                    ix = int(hand.landmark[8].x * self.cam_width)
                    iy = int(hand.landmark[8].y * self.cam_height)
                    color = (0, 0, 255) if (left or right) else (0, 255, 0)
                    cv2.circle(debug, (ix, iy), 15, color, 2)
                    
                    if left:
                        cv2.putText(debug, "LEFT CLICK", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if right:
                        cv2.putText(debug, "RIGHT CLICK", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                cv2.putText(debug, "No hand", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(debug, "GestureMouse - Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
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
