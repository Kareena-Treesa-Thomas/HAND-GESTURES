# Configuration settings for Hand Gesture Controller

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# MediaPipe settings
HAND_DETECTION_CONFIDENCE = 0.7
HAND_TRACKING_CONFIDENCE = 0.5
MAX_NUM_HANDS = 1

# Gesture detection thresholds
SWIPE_THRESHOLD = 0.15  # Minimum horizontal movement for swipe
SWIPE_COOLDOWN = 1.0    # Seconds between swipe gestures
SCROLL_THRESHOLD = 30    # Minimum vertical movement for scroll
CROSS_DISTANCE_THRESHOLD = 0.15  # Distance between thumb and other fingers for cross
CROSS_TIME_WINDOW = 2.0  # Seconds to complete cross + swipe down

# Visual feedback settings
SHOW_LANDMARKS = True
SHOW_CONNECTIONS = True
MIRROR_FRAME = True

# System control settings
SCROLL_AMOUNT = 3  # Number of scroll units per gesture

# Debug settings
DEBUG_MODE = False
SAVE_DEBUG_FRAMES = False