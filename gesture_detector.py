import os
import cv2
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe import Image, ImageFormat


class GestureDetector:

    def __init__(self):

        model_path = 'hand_landmarker.task'
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Required model file not found: {model_path}.\n"
                "Download it with: curl -o hand_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )

        base_options = BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, img):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=imgRGB)
        result = self.landmarker.detect(mp_image)

        gesture = None

        if result.hand_landmarks:
            gesture = "Hand Detected"

        return img, gesture