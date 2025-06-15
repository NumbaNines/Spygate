#!/usr/bin/env python3
"""Test script to validate imports from the new src structure."""

import sys
import traceback
from pathlib import Path


def test_import(module_name, description):
    """Test importing a specific module."""
    try:
        if module_name == "hardware":
            from src.spygate.core.hardware import HardwareDetector, HardwareTier

            detector = HardwareDetector()
            tier = detector.detect_tier()
            print(f"✓ {description}: SUCCESS - Detected {tier.name}")
            return True
        elif module_name == "yolo":
            from src.spygate.ml.yolov8_model import UI_CLASSES, EnhancedYOLOv8, OptimizationConfig

            print(f"✓ {description}: SUCCESS - Classes: {len(UI_CLASSES)}")
            return True
        elif module_name == "analyzer":
            from src.spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer

            print(f"✓ {description}: SUCCESS")
            return True
        elif module_name == "ocr":
            from src.spygate.ml.enhanced_ocr import EnhancedOCR

            print(f"✓ {description}: SUCCESS")
            return True
        else:
            print(f"✗ {description}: Unknown module")
            return False
    except Exception as e:
        print(f"✗ {description}: FAILED - {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False


def main():
    """Run all import tests."""
    print("Testing imports from src.spygate structure...")
    print("=" * 50)

    tests = [
        ("hardware", "Hardware Detection"),
        ("yolo", "YOLOv8 Model"),
        ("analyzer", "Game Analyzer"),
        ("ocr", "Enhanced OCR"),
    ]

    passed = 0
    total = len(tests)

    for module_name, description in tests:
        if test_import(module_name, description):
            passed += 1

    print("=" * 50)
    print(f"Import test results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All imports working correctly!")
        return True
    else:
        print("⚠️ Some imports are failing")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
