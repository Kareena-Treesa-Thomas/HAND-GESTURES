import cv2
import mediapipe as mp


class GestureDetector:

    def __init__(self):

        self.mpHands = mp.solutions.hands

        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.mpDraw = mp.solutions.drawing_utils

    def detect(self, img):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        gesture = None

        if results.multi_hand_landmarks:

            for handLms in results.multi_hand_landmarks:

                # Draw landmarks
                self.mpDraw.draw_landmarks(
                    img,
                    handLms,
                    self.mpHands.HAND_CONNECTIONS
                )

                gesture = "Hand Detected"

        return img, gesture