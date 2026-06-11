# 🖐 Hand Gesture Recognition

[![Python](https://img.shields.io/badge/Python-3.7+-3b82f6?style=flat-square)](https://python.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Landmarks-22c55e?style=flat-square)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-e11d48?style=flat-square)](https://opencv.org)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-94a3b8?style=flat-square)](#requirements)
[![License: MIT](https://img.shields.io/badge/License-MIT-f59e0b?style=flat-square)](LICENSE)

> Control your computer with hand gestures — no mouse, no keyboard. Just your hands and a webcam.

---

## 🎬 Demo

![Hand Gesture Demo](img-hand-gesture.jpeg)

---

## 🤌 Gestures

| Gesture | Action |
|---|---|
| ☝️ 1 finger | Scroll down |
| ✌️ 2 fingers | Scroll up |
| 🖐 4 or 5 fingers + swipe right | Switch to next tab |
| 🙌 5 fingers on both hands | Minimize current app |

---

## ⚡ Quick Start

**1. Clone the repo**
```bash
git clone https://github.com/Kareena-Treesa-Thomas/hand-gesture-recognition.git
cd hand-gesture-recognition
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Windows (cmd.exe)
venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the MediaPipe model** (~15MB, not tracked by Git)
```bash
curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

**5. Run**
```bash
python main.py
```

Press **`q`** to quit.

---

## 🛠 Tech Stack

```
Language        Python 3.7+
Vision          OpenCV — webcam capture and frame processing
Detection       MediaPipe Tasks API — hand landmark model
Automation      PyAutoGUI — mouse and keyboard control
Input events    Keyboard — hotkey handling
Math            NumPy — gesture calculations
```

---

## 📁 Project Structure

```
hand-gesture-recognition/
├── main.py                 # Main loop + gesture processing
├── gesture_detector.py     # MediaPipe hand landmark wrapper
├── config.py               # Camera index, display settings
├── requirements.txt        # Python dependencies
├── hand_landmarker.task    # MediaPipe model (download separately)
└── README.md
```

---

## 🔧 Requirements

- Python 3.7+
- Webcam
- Windows / macOS / Linux

---

## 🐞 Troubleshooting

**`ModuleNotFoundError`** — make sure your virtual environment is active and you ran `pip install -r requirements.txt`.

**Gestures not detected:**
- Confirm `hand_landmarker.task` is in the project root
- Use good lighting and keep hands fully in frame
- Try adjusting your distance from the camera

---

## 📝 Notes

- `hand_landmarker.task` is ~15MB and excluded from Git via `.gitignore`. Download it once using the `curl` command above.
- Using a virtual environment is strongly recommended to avoid package conflicts.

---

## 👩‍💻 Author

**Kareena Treesa Thomas** · [github.com/Kareena-Treesa-Thomas](https://github.com/Kareena-Treesa-Thomas)
