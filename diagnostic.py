#!/usr/bin/env python3
"""
Diagnostic script for Face Authentication App
This script helps identify potential issues with the face recognition system.
"""

import sys
import os
import json
import cv2
import numpy as np
from datetime import datetime

def check_dependencies():
    """Check if all required dependencies are installed"""
    print(" Checking dependencies...")
    
    required_packages = [
        'streamlit',
        'opencv-python',
        'numpy',
        'bcrypt',
        'scikit-learn',
        'mediapipe',
        'insightface',
        'onnxruntime'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
                print(f" {package}: {cv2.__version__}")
            elif package == 'scikit-learn':
                import sklearn
                print(f" {package}: {sklearn.__version__}")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                print(f" {package}: {version}")
        except ImportError:
            print(f" {package}: NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n  Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    print(" All dependencies are installed!")
    return True

def check_camera():
    """Test camera functionality"""
    print("\n Testing camera access...")
    
    try:
        # Try to open the default camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print(" Cannot access camera (device 0)")
            return False
        
        # Try to read a frame
        ret, frame = cap.read()
        
        if not ret or frame is None:
            print(" Cannot read from camera")
            cap.release()
            return False
        
        print(f" Camera working! Frame shape: {frame.shape}")
        cap.release()
        return True
        
    except Exception as e:
        print(f" Camera error: {e}")
        return False

def check_insightface():
    """Test InsightFace model loading"""
    print("\n Testing InsightFace model...")
    
    try:
        from insightface.app import FaceAnalysis
        
        print("Loading InsightFace model...")
        app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(240, 240))
        
        print(" InsightFace model loaded successfully!")
        
        # Test with a dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        faces = app.get(dummy_image)
        
        print(f" Model inference test completed (detected {len(faces)} faces in random image)")
        return True
        
    except Exception as e:
        print(f" InsightFace error: {e}")
        return False

def check_data_files():
    """Check data file integrity"""
    print("\n Checking data files...")
    
    files_to_check = ['users.json', 'people.json']
    
    for filename in files_to_check:
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                print(f"{filename}: Valid JSON with {len(data)} entries")
            except json.JSONDecodeError as e:
                print(f"{filename}: Invalid JSON - {e}")
            except Exception as e:
                print(f"{filename}: Error reading file - {e}")
        else:
            print(f"{filename}: File doesn't exist (will be created when needed)")

def check_system_info():
    """Display system information"""
    print("\n💻 System Information:")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Available memory: {memory.available / (1024**3):.1f} GB / {memory.total / (1024**3):.1f} GB")
    except ImportError:
        print("Memory info not available (psutil not installed)")

def run_diagnostics():
    """Run all diagnostic checks"""
    print("Face Authentication App Diagnostics")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Camera", check_camera),
        ("InsightFace", check_insightface),
        ("Data Files", check_data_files),
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"{check_name} check failed with exception: {e}")
            results[check_name] = False
    
    check_system_info()
    
    print("\n" + "=" * 50)
    print("DIAGNOSTIC SUMMARY:")
    
    all_passed = True
    for check_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{check_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\nAll checks passed! Your system should work correctly.")
    else:
        print("\n  Some checks failed. Please address the issues above.")
        print("\n Common solutions:")
        print("- Install missing dependencies: pip install -r requirements.txt")
        print("- Check camera permissions in your browser/system")
        print("- Restart the application")
        print("- Try a different browser if camera issues persist")

if __name__ == "__main__":
    run_diagnostics()