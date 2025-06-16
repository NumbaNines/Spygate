#!/usr/bin/env python3
"""
Simple test script to identify the "list has no attribute" error.
"""

import sys
import traceback

def test_imports():
    """Test all imports step by step."""
    print("Testing imports...")
    
    try:
        print("1. Testing UI_CLASSES import...")
        from src.spygate.ml.yolov8_model import UI_CLASSES
        print(f"   ✓ UI_CLASSES type: {type(UI_CLASSES)}")
        print(f"   ✓ UI_CLASSES content: {UI_CLASSES}")
        print(f"   ✓ UI_CLASSES length: {len(UI_CLASSES)}")
        
        # Test accessing elements
        print(f"   ✓ First element: {UI_CLASSES[0]}")
        print(f"   ✓ 'down_distance_area' in list: {'down_distance_area' in UI_CLASSES}")
        
    except Exception as e:
        print(f"   ✗ UI_CLASSES import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        print("2. Testing DownTemplateDetector import...")
        from src.spygate.ml.down_template_detector import DownTemplateDetector, DownDetectionContext
        print("   ✓ DownTemplateDetector imported successfully")
        
        # Test instantiation
        detector = DownTemplateDetector()
        print("   ✓ DownTemplateDetector instantiated successfully")
        
        context = DownDetectionContext(quality_mode="medium", confidence_threshold=0.25)
        print("   ✓ DownDetectionContext created successfully")
        
    except Exception as e:
        print(f"   ✗ DownTemplateDetector import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        print("3. Testing EnhancedYOLOv8 import...")
        from src.spygate.ml.yolov8_model import EnhancedYOLOv8
        print("   ✓ EnhancedYOLOv8 imported successfully")
        
    except Exception as e:
        print(f"   ✗ EnhancedYOLOv8 import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        print("4. Testing PyQt6 imports...")
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import QThread, pyqtSignal
        print("   ✓ PyQt6 imports successful")
        
    except Exception as e:
        print(f"   ✗ PyQt6 import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        print("5. Testing screen capture imports...")
        try:
            import mss
            print("   ✓ mss available")
        except ImportError:
            print("   - mss not available")
        
        try:
            import pyautogui
            print("   ✓ pyautogui available")
        except ImportError:
            print("   - pyautogui not available")
            
    except Exception as e:
        print(f"   ✗ Screen capture import test failed: {e}")
        traceback.print_exc()
    
    print("\nAll basic imports successful!")
    return True

def test_class_instantiation():
    """Test creating instances of key classes."""
    print("\nTesting class instantiation...")
    
    try:
        print("1. Testing DownTemplateDetector instantiation...")
        from src.spygate.ml.down_template_detector import DownTemplateDetector, DownDetectionContext
        
        detector = DownTemplateDetector()
        print("   ✓ DownTemplateDetector created")
        
        # Test methods exist
        if hasattr(detector, 'detect_down_distance'):
            print("   ✓ detect_down_distance method exists")
        else:
            print("   ✗ detect_down_distance method missing")
            
        context = DownDetectionContext(quality_mode="medium")
        print("   ✓ DownDetectionContext created")
        
    except Exception as e:
        print(f"   ✗ DownTemplateDetector instantiation failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        print("2. Testing list operations on UI_CLASSES...")
        from src.spygate.ml.yolov8_model import UI_CLASSES
        
        # Test list comprehension (common source of "list has no attribute" errors)
        down_classes = [cls for cls in UI_CLASSES if 'down' in cls]
        print(f"   ✓ List comprehension works: {down_classes}")
        
        # Test filtering
        filtered = [d for d in [{'class_name': 'down_distance_area'}] if d.get('class_name') == 'down_distance_area']
        print(f"   ✓ Dictionary filtering works: {filtered}")
        
    except Exception as e:
        print(f"   ✗ List operations failed: {e}")
        traceback.print_exc()
        return False
    
    print("All instantiation tests successful!")
    return True

def test_detection_simulation():
    """Test a simulated detection workflow."""
    print("\nTesting detection simulation...")
    
    try:
        import numpy as np
        from src.spygate.ml.down_template_detector import DownTemplateDetector, DownDetectionContext
        
        # Create a dummy image
        dummy_frame = np.zeros((100, 200, 3), dtype=np.uint8)
        print("   ✓ Dummy frame created")
        
        # Create detector
        detector = DownTemplateDetector()
        context = DownDetectionContext(quality_mode="medium")
        print("   ✓ Detector and context created")
        
        # Simulate detection (this might fail due to no templates, but shouldn't crash)
        try:
            result = detector.detect_down_distance(dummy_frame, context)
            print(f"   ✓ Detection completed: {type(result)}")
        except Exception as detection_error:
            print(f"   - Detection failed (expected): {detection_error}")
        
    except Exception as e:
        print(f"   ✗ Detection simulation failed: {e}")
        traceback.print_exc()
        return False
    
    print("Detection simulation completed!")
    return True

def main():
    """Main test function."""
    print("=" * 60)
    print("SpygateAI Live Template Detection - Import Test")
    print("=" * 60)
    
    success = True
    
    success &= test_imports()
    success &= test_class_instantiation()
    success &= test_detection_simulation()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ ALL TESTS PASSED - No 'list has no attribute' errors found")
        print("The issue might be in the GUI event handling or threading.")
    else:
        print("✗ SOME TESTS FAILED - Check the errors above")
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 