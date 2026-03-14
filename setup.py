import subprocess
import sys
import os

def install_requirements():
    """Install the required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing packages: {e}")
        return False

def check_camera():
    """Check if camera is available"""
    print("Checking camera availability...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print("✓ Camera is available and working!")
                return True
            else:
                print("✗ Camera found but couldn't capture image")
                return False
        else:
            print("✗ No camera found")
            return False
    except Exception as e:
        print(f"✗ Error checking camera: {e}")
        return False

def main():
    print("=== Hand Gesture Controller Setup ===")
    print()
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("✗ Python 3.7+ is required")
        return False
    
    print(f"✓ Python {sys.version.split()[0]} detected")
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Check camera
    if not check_camera():
        print("\n⚠ Warning: Camera not detected. Please ensure your webcam is connected.")
        input("Press Enter to continue anyway...")
    
    print("\n=== Setup Complete! ===")
    print("Run 'python main.py' to start the gesture controller")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        input("\nPress Enter to exit...")
    else:
        input("\nSetup failed. Press Enter to exit...")