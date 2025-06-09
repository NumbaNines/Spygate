#!/usr/bin/env python3

"""
Test Script for SpygateAI Desktop Application
============================================

Simple test to verify the desktop application can launch and core components work.
"""

import sys
import os
from pathlib import Path

# Add project paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "spygate"))

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        # Test PyQt6
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import QObject, pyqtSignal
        from PyQt6.QtGui import QFont, QPalette, QColor
        print("✅ PyQt6 imported successfully")
        
        # Test OpenCV
        import cv2
        import numpy as np
        print("✅ OpenCV and NumPy imported successfully")
        
        # Test core modules
        from spygate.core.hardware import HardwareDetector
        print("✅ HardwareDetector imported successfully")
        
        from spygate.core.optimizer import TierOptimizer
        print("✅ Optimizer modules imported successfully")
        
        from spygate.core.game_detector import GameDetector
        print("✅ GameDetector imported successfully")
        
        print("✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_hardware_detection():
    """Test hardware detection functionality."""
    print("\n🔧 Testing hardware detection...")
    
    try:
        from spygate.core.hardware import HardwareDetector
        
        detector = HardwareDetector()
        
        # Test hardware tier detection
        tier = detector.tier
        print(f"✅ Hardware tier detected: {tier.name}")
        
        # Test memory detection
        memory_info = detector.get_system_memory()
        total_gb = memory_info.get('total', 0) / (1024**3)
        print(f"✅ Memory detected: {total_gb:.1f} GB")
        
        # Test GPU detection
        print(f"✅ CUDA available: {detector.has_cuda}")
        if detector.has_cuda:
            print(f"✅ GPU detected: {detector.gpu_count} device(s)")
            print(f"✅ GPU name: {detector.gpu_name}")
        else:
            print("⚠️ No CUDA GPU detected")
            
        return True
        
    except Exception as e:
        print(f"❌ Hardware detection error: {e}")
        return False


def test_desktop_app_launch():
    """Test that the desktop application can be created (without showing)."""
    print("\n🖥️ Testing desktop application creation...")
    
    try:
        from PyQt6.QtWidgets import QApplication
        
        # Create QApplication (required for any Qt widgets)
        app = QApplication([])
        
        # Import and create the main window (without showing)
        import spygate_desktop_app
        window = spygate_desktop_app.SpygateDesktopApp()
        
        print("✅ Desktop application created successfully")
        print(f"✅ Window title: {window.windowTitle()}")
        print(f"✅ Window size: {window.size().width()}x{window.size().height()}")
        
        # Test that key components exist
        assert hasattr(window, 'sidebar'), "Sidebar component missing"
        assert hasattr(window, 'content_stack'), "Content stack missing"
        assert hasattr(window, 'auto_detect_widget'), "Auto-detect widget missing"
        
        print("✅ All key components present")
        
        # Clean up
        window.close()
        app.quit()
        
        return True
        
    except Exception as e:
        print(f"❌ Desktop application test error: {e}")
        return False


def test_video_processing():
    """Test basic video processing capabilities."""
    print("\n🎬 Testing video processing...")
    
    try:
        import cv2
        
        # Test OpenCV video capabilities
        print(f"✅ OpenCV version: {cv2.__version__}")
        
        # Test video codecs
        fourcc_test = cv2.VideoWriter_fourcc(*'mp4v')
        print("✅ Video codec support available")
        
        # Test CUDA if available
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print(f"✅ CUDA support: {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
        else:
            print("⚠️ No CUDA support detected")
            
        return True
        
    except Exception as e:
        print(f"❌ Video processing test error: {e}")
        return False


def main():
    """Run all tests."""
    print("🏈 SpygateAI Desktop Application Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Hardware Detection", test_hardware_detection),
        ("Desktop App Launch", test_desktop_app_launch),
        ("Video Processing", test_video_processing)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Desktop application is ready.")
        print("\nTo run the desktop application:")
        print("python spygate_desktop_app.py")
    else:
        print(f"\n⚠️ {len(results) - passed} test(s) failed. Please check dependencies.")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main()) 